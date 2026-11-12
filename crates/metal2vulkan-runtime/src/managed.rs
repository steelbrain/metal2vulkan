//! Managed storage (`MTLStorageModeManaged`): `Buffer::did_modify_range`.
//! `BlitCommandEncoder::synchronize_resource` is in `blit.rs`.
//!
//! A managed buffer follows Metal's contract for the device's memory model
//! ([`DeviceProperties::has_unified_memory`](super::DeviceProperties::has_unified_memory)):
//!
//! - **With unified memory** (an integrated GPU, a CPU implementation), it lives in host-visible
//!   coherent memory, cached where the device has such
//!   ([`MemoryClass::Managed`](super::memory::MemoryClass)), which the GPU uses itself, as Apple
//!   silicon keeps managed resources in memory the CPU and GPU share. Coherent memory needs no
//!   flush and no invalidate: the CPU's writes reach the GPU at the next submission, and every
//!   command buffer ends with a barrier to the host. So these calls check a managed buffer's
//!   range and record nothing more. On a resource of another storage mode they do nothing, as
//!   plain Metal does (its validation stops them).
//! - **Without unified memory** (a discrete GPU), it has two copies, as Metal's managed resources
//!   have on such a device: its own bytes in device-local memory, which the GPU uses, and a host
//!   copy in such cached memory, which `contents` maps (`buffer.rs`'s `Mirror`). `did_modify_range`
//!   notes the range ([`Modified`]); the device's next commit, of a command buffer of any of its
//!   queues, takes the range's bytes as the host copy holds them then, before that call returns,
//!   and copies them to the device's own by the next submission, before any of its commands
//!   (`synchronize_resource` copies the buffer back where the command buffer runs it). A program
//!   that skips them sees stale bytes there, as on a Mac with a discrete GPU. A GPU writing
//!   host-cached memory across the bus ran a kernel's writes at 0.03 to 0.1 GB/s on an RTX 3090
//!   and an RTX 5080, where a device-local write and a copy back ran at several GB/s. A `Managed`
//!   no-copy buffer is two copies the same way, its host copy the caller's pages
//!   (`managed_copy.rs`).
//! - **The bytes a command buffer reads** are the host copy's at the commit after the
//!   `did_modify_range`, Metal's answer on a Mac with a discrete GPU: on an AMD Radeon Pro 5300M
//!   (macOS 26.7.1), a kernel copying a 4 KiB `Managed` buffer behind a spin kernel read the CPU's
//!   write made before the commit (after the call, and with the command buffer encoded before the
//!   write), and not one made 30 ms after the commit while the GPU still spun. On an Intel UHD
//!   Graphics 630 (unified memory, one copy) it read the later write, as the GPU reads the memory
//!   the CPU wrote last. Read as the copy ran, a CPU write after the commit reached a command
//!   buffer the GPU had not reached yet. Taking the bytes costs a CPU copy of each range noted
//!   since the last commit into staging buffers of the runtime's own at that commit, freed once
//!   the submission that copies them in completed (`managed_copy.rs`, at most `STAGED_BYTES`
//!   staged ahead). A range the staging buffers find no memory for is copied in from the host
//!   copy as that submission runs, as before.

use super::buffer::{Buffer, WeakBuffer};
use super::device::lock;
use super::error::{invalid, Error, Problem};
use super::values::StorageMode;
use std::ops::Range;
use std::sync::Mutex;

impl Buffer {
    /// `didModifyRange:`: the CPU wrote `range` of a managed buffer; command buffers committed
    /// after this call see it. Where the buffer has a host copy (a device without unified memory),
    /// the range's bytes are taken from it at the device's next commit and copied to its own bytes
    /// by the submission after it, before any of its commands (the module docs); the call does not
    /// wait for the GPU. On a buffer of another storage mode it does nothing, whatever the range,
    /// as plain Metal does (RT2 `managed.didmodify.shared`: `Shared` and `Private` buffers, in
    /// range and past it; Metal's validation stops it). Refused for a managed buffer's range past
    /// the buffer.
    pub fn did_modify_range(&self, range: Range<u64>) -> Result<(), Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        const CALL: &str = "did_modify_range";
        if self.storage_mode() != StorageMode::Managed {
            return Ok(());
        }
        let length = self.length();
        if range.start > range.end || range.end > length {
            return Err(invalid(
                CALL,
                Problem::RangeOutOfBounds {
                    offset: range.start,
                    size: range.end.saturating_sub(range.start),
                    length,
                },
            ));
        }
        // A two-copy buffer over the caller's pages (`managed_copy.rs`): the range the program
        // asked, which the next commit copies in.
        if self.caller_copy().is_some() && !range.is_empty() {
            let counts = &self.device().shared().counts;
            counts.managed_asked(range.end - range.start, true);
        }
        self.modified(range);
        Ok(())
    }

    /// The CPU wrote `range` of the buffer's host copy, where it has one: the device's next
    /// commit takes the bytes and copies them in ([`Modified`]).
    pub(super) fn modified(&self, range: Range<u64>) {
        if self.mirror().is_none() || range.is_empty() {
            return;
        }
        self.device().shared().modified.note(self, range);
    }
}

/// The ranges of buffers with a host copy the CPU wrote and named (`did_modify_range`, a CPU
/// write through a texture over the buffer) since the device's last commit: the next commit
/// copies each in, as the host copy holds it then (the module docs). A buffer gone meanwhile
/// takes its ranges with it.
#[derive(Default)]
pub(super) struct Modified(Mutex<Vec<(WeakBuffer, Range<u64>)>>);

impl Modified {
    /// Note `range` of `buffer`: once, where a range noted of it already holds it.
    fn note(&self, buffer: &Buffer, range: Range<u64>) {
        let mut noted = lock(&self.0);
        let held = noted.iter().any(|(of, at)| {
            at.start <= range.start
                && range.end <= at.end
                && of.upgrade().is_some_and(|of| of == *buffer)
        });
        if !held {
            noted.push((buffer.downgrade(), range));
        }
    }

    /// A commit: take every range noted so far and copy its bytes in as its host copy holds them
    /// now (staged, `Buffer::upload_now`), before the commit's command buffer is submitted, so the
    /// submission that carries its work, or one before it, copies them in.
    pub(crate) fn take_at_commit(&self) {
        let noted: Vec<(WeakBuffer, Range<u64>)> = lock(&self.0).drain(..).collect();
        for (buffer, range) in noted {
            if let Some(buffer) = buffer.upgrade() {
                buffer.upload_now(range);
            }
        }
    }
}
