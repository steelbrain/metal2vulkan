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
//!   notes the range ([`Modified`]); the next commit of a command buffer whose commands may use
//!   the buffer (its encoders' uses name it, `use_resource` made it resident, or a command may
//!   reach memory no use names), of any of the device's queues, takes the range's bytes as the
//!   host copy holds them then, before that call returns, and copies them to the device's own by
//!   the next submission whose commands may use the buffer of a command buffer committed at or
//!   after that commit, before any of them, after the command buffers before it on its
//!   `VkQueue` (`texture/first_move.rs`'s `register_upload`): Metal's shape, the upload made
//!   before the GPU uses the buffer, on the user's own queue. A command buffer committed before,
//!   whose submission the scheduler makes after the later commit, never takes it, so it reads
//!   the bytes of its own commit and none a later commit took. A
//!   commit that uses none of it leaves the range noted, and its submission leaves the copy
//!   waiting, so the copy never lands under a command buffer of another queue still to read the
//!   bytes it replaces (`synchronize_resource` copies the buffer back where the command buffer
//!   runs it). A program
//!   that skips them sees stale bytes there, as on a Mac with a discrete GPU. A GPU writing
//!   host-cached memory across the bus ran a kernel's writes at 0.03 to 0.1 GB/s on an RTX 3090
//!   and an RTX 5080, where a device-local write and a copy back ran at several GB/s. A `Managed`
//!   no-copy buffer is two copies the same way, its host copy the caller's pages
//!   (`managed_copy.rs`).
//! - **The bytes a command buffer reads** are the host copy's at the commit, after the
//!   `did_modify_range`, of the first command buffer that may use the buffer, Metal's answer on a
//!   Mac with a discrete GPU: on an AMD Radeon Pro 5300M
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
    /// the range's bytes are taken from it at the next commit of a command buffer that may use the
    /// buffer and copied to its own bytes by that command buffer's submission, or one before it
    /// that may use the buffer, before any of its commands (the module docs); the call does not
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
        // asked, which the next commit that may use the buffer copies in.
        if self.caller_copy().is_some() && !range.is_empty() {
            let counts = &self.generation().shared().counts;
            counts.managed_asked(range.end - range.start, true);
        }
        self.modified(range);
        Ok(())
    }

    /// The CPU wrote `range` of the buffer's host copy, where it has one: the next commit of a
    /// command buffer that may use the buffer takes the bytes and copies them in ([`Modified`]).
    pub(super) fn modified(&self, range: Range<u64>) {
        if self.mirror().is_none() || range.is_empty() {
            return;
        }
        self.generation().shared().modified.note(self, range);
    }
}

/// The ranges of buffers with a host copy the CPU wrote and named (`did_modify_range`, a CPU
/// write through a texture over the buffer) that no commit of a command buffer that may use
/// their buffer took yet: the next such commit copies each in, as the host copy holds it then
/// (the module docs), and the commits' stamps. A buffer gone meanwhile takes its ranges with it.
#[derive(Default)]
pub(super) struct Modified(Mutex<Ranges>);

#[derive(Default)]
struct Ranges {
    noted: Noted<WeakBuffer>,
    /// The commits so far: the stamp of the last ([`Modified::take_at_commit`]).
    commits: u64,
}

impl Modified {
    /// Note `range` of `buffer`: once, where a range noted of it already holds it.
    fn note(&self, buffer: &Buffer, range: Range<u64>) {
        let same = |of: &WeakBuffer| of.upgrade().is_some_and(|of| of == *buffer);
        lock(&self.0).noted.note(buffer.downgrade(), range, same);
    }

    /// A commit of a command buffer whose commands may use a buffer as `uses` answers: take the
    /// ranges noted of those it may use, and copy their bytes in as their host copies hold them
    /// now (staged, `Buffer::upload_now`), before the command buffer is submitted. The commit's
    /// stamp, which the command buffer keeps: one past every commit's taken before it, under the
    /// same lock. Each upload carries it, and only a batch of a command buffer whose stamp is at
    /// least as large takes it (`texture/first_move.rs`'s `register_upload`): its own submission
    /// that may use the buffer copies them in, or a later commit's, never one committed before,
    /// which reads the bytes its own commit took. The ranges of the rest stay noted, in their
    /// order.
    pub(crate) fn take_at_commit(&self, uses: impl Fn(&Buffer) -> bool) -> u64 {
        // Each buffer read here, let go of once the lock is free.
        let mut read = Vec::new();
        let (commit, taken) = {
            let mut ranges = lock(&self.0);
            ranges.commits += 1;
            let taken = ranges.noted.take(|of| {
                let buffer = of.upgrade()?;
                let used = uses(&buffer);
                read.push(buffer);
                Some(used)
            });
            (ranges.commits, taken)
        };
        for (buffer, range) in taken {
            if let Some(buffer) = buffer.upgrade() {
                buffer.upload_now(range, commit);
            }
        }
        drop(read);
        commit
    }
}

/// Ranges of buffers named by `B`, in the order they were noted ([`Modified`]).
pub(super) struct Noted<B>(Vec<(B, Range<u64>)>);

impl<B> Default for Noted<B> {
    fn default() -> Self {
        Noted(Vec::new())
    }
}

impl<B> Noted<B> {
    /// Note `range` of `of`: once, where a range noted of a buffer `same` says is `of` already
    /// holds it.
    pub(super) fn note(&mut self, of: B, range: Range<u64>, same: impl Fn(&B) -> bool) {
        let held = (self.0.iter())
            .any(|(noted, at)| at.start <= range.start && range.end <= at.end && same(noted));
        if !held {
            self.0.push((of, range));
        }
    }

    /// The ranges whose buffer `taken` answers true for, in their order; those it answers
    /// `None` for (their buffer gone) go, and the rest stay, in their order.
    pub(super) fn take(
        &mut self,
        mut taken: impl FnMut(&B) -> Option<bool>,
    ) -> Vec<(B, Range<u64>)> {
        let mut out = Vec::new();
        let mut kept = Vec::new();
        for (of, range) in self.0.drain(..) {
            match taken(&of) {
                Some(true) => out.push((of, range)),
                Some(false) => kept.push((of, range)),
                None => {}
            }
        }
        self.0 = kept;
        out
    }
}
