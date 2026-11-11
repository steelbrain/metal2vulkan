//! The descriptor a dma-buf import hands its driver (`memory::import_dma_buf`): a clone of the
//! dma-buf's, at a number of a range kept for imports in flight, which no other import holds.
//!
//! Vulkan gives the driver the descriptor only on a successful import (`VkImportMemoryFdInfoKHR`).
//! NVIDIA's closes it on a failed one too, and at times closes its number twice within the one
//! failed call (620.70.06 on an RTX 2080, measured with `strace` under the eight-thread dma-buf
//! probe, `tests::dma_buf_leak_probe`: every chunk's first import, in a type the driver lists for
//! the dma-buf and then refuses as out of device memory, closed the number once or, in 3 of 10,
//! twice, with two more opens of `/dev/nvidiactl` between). The kernel gives a new descriptor the
//! lowest free number, so between the two closes another thread's new descriptor took the number,
//! and the second close closed it: that thread then imported a third thread's dma-buf under the
//! number, and the process ended on Rust's I/O safety check (`EBADF` in an `OwnedFd`'s close).
//!
//! A number of the range is above the process's other descriptors: a file opened or a descriptor
//! cloned elsewhere in the process takes the lowest free number, which is never one of the range
//! while a lower number is free, and no two imports in flight hold the same one. So a number its
//! driver closed stays free until the import returns, and a second close there closes nothing.
//! Once a failed import returns, its number is still the import's alone, so whether the driver
//! left it open is read without a race and, where it did, it is closed ([`Handed::failed`]).
//!
//! The range is the [`NUMBERS`] numbers below the process's descriptor limit (`RLIMIT_NOFILE`'s
//! soft limit), or below [`CEILING`] where the limit is higher, which bounds the table the kernel
//! grows to hold them. Where the process holds the range's numbers itself, or more than
//! [`NUMBERS`] imports are in flight at once, an import's clone takes the lowest free number as any
//! clone does, and a failed import lets it go rather than close a number that may be another's by
//! then.

use crate::device::lock;
use std::ops::Range;
use std::os::fd::{AsRawFd, BorrowedFd, IntoRawFd, RawFd};
use std::sync::Mutex;

/// How many numbers the range holds: the imports in flight at once that take one.
const NUMBERS: RawFd = 64;

/// The range ends below this number, whatever the process's descriptor limit.
const CEILING: libc::rlim_t = 1 << 16;

/// The range's numbers that imports in flight hold.
static IN_FLIGHT: Mutex<Vec<RawFd>> = Mutex::new(Vec::new());

/// The range of numbers kept for imports, as the process's descriptor limit stands now
/// ([`range_below`]).
fn range() -> Option<Range<RawFd>> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes the structure, which outlives the call.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return None;
    }
    range_below(limit.rlim_cur)
}

/// The range of numbers kept for imports under a descriptor limit of `limit`: the [`NUMBERS`]
/// below it, or below [`CEILING`] where it is higher. `None` where the limit is under four times
/// [`NUMBERS`], too low to keep them from the process's other descriptors.
pub(super) fn range_below(limit: libc::rlim_t) -> Option<Range<RawFd>> {
    let top = RawFd::try_from(limit.min(CEILING)).ok()?;
    (top >= 4 * NUMBERS).then(|| top - NUMBERS..top)
}

/// Whether descriptor `number` is open.
fn open(number: RawFd) -> bool {
    // SAFETY: `F_GETFD` reads the descriptor's flags and changes nothing.
    unsafe { libc::fcntl(number, libc::F_GETFD) >= 0 }
}

/// Close descriptor `number`, ignoring whether it was open: a number of this module's own, or one
/// no descriptor holds.
fn close(number: RawFd) {
    // SAFETY: the number is this module's own descriptor, or no descriptor at all (`EBADF`).
    unsafe { libc::close(number) };
}

/// A clone of a dma-buf's descriptor, handed to one import (`vkAllocateMemory` with
/// `VkImportMemoryFdInfoKHR`), which is the import's to give back as [`Handed::imported`] or
/// [`Handed::failed`].
pub(super) struct Handed {
    number: RawFd,
    /// Whether the number is one of the range's, held in [`IN_FLIGHT`] until the import returns.
    kept: bool,
}

impl Handed {
    /// A clone of `fd` at a number of the range no other import holds, where one is free, else at
    /// the lowest free number. Refused where no clone can be made at all.
    pub(super) fn of(fd: BorrowedFd<'_>) -> std::io::Result<Handed> {
        Handed::within(fd, range())
    }

    /// As [`Handed::of`], the numbers kept for imports `range`, where there are any. A number is
    /// taken only where the clone landed on it (`F_DUPFD_CLOEXEC` from it up), never by `dup2`,
    /// which would close what the process holds there.
    pub(super) fn within(
        fd: BorrowedFd<'_>,
        range: Option<Range<RawFd>>,
    ) -> std::io::Result<Handed> {
        if let Some(range) = range {
            let mut in_flight = lock(&IN_FLIGHT);
            for number in range {
                if in_flight.contains(&number) {
                    continue;
                }
                // SAFETY: a clone of `fd` at the lowest free number from `number` up, which is
                // this function's own descriptor.
                let clone = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, number) };
                if clone == number {
                    in_flight.push(number);
                    return Ok(Handed { number, kept: true });
                }
                if clone < 0 {
                    break;
                }
                // `number` is the process's own: the clone took a higher one, which is closed
                // at once, and the next number tried.
                close(clone);
            }
        }
        let clone = fd.try_clone_to_owned()?;
        Ok(Handed {
            number: clone.into_raw_fd(),
            kept: false,
        })
    }

    /// The number to hand the driver.
    pub(super) fn number(&self) -> RawFd {
        self.number
    }

    /// Whether the number is one of the range's, for the crate's tests.
    #[cfg(test)]
    pub(super) fn kept(&self) -> bool {
        self.kept
    }

    /// The import succeeded: the descriptor is the driver's, which may have closed it already.
    pub(super) fn imported(self) {}

    /// The import failed, or was never made: the descriptor is closed where the driver left it
    /// open. A clone at the lowest free number is let go of instead, as its number may be
    /// another's by now.
    pub(super) fn failed(self) {
        if self.kept {
            let _in_flight = lock(&IN_FLIGHT);
            if open(self.number) {
                close(self.number);
            }
        }
    }
}

impl Drop for Handed {
    /// The number leaves [`IN_FLIGHT`], free for the next import that finds it closed.
    fn drop(&mut self) {
        if self.kept {
            lock(&IN_FLIGHT).retain(|&number| number != self.number);
        }
    }
}
