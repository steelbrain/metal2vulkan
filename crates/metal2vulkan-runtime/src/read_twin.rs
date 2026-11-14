//! Read twins: a no-copy buffer whose GPU writes Metal's private option bit 24 drops
//! ([`ResourceOptions::PRIVATE_BIT`]), where the system maps no read-only view of its pages
//! (`read_only_view.rs`), read in place by the commands that cannot write it.
//!
//! - **Why.** Off macOS such a buffer is a copy of its pages (`copied.rs`), and a copy is
//!   compared with its pages at each commit that uses it: a guest's window server reads such
//!   buffers of 30 MB every frame, and the comparisons cost it whole seconds of a walk while the
//!   GPU never wrote them. No import keeps the GPU's writes from the pages there (`host_memory.rs`),
//!   and Vulkan has no read-only memory, so the pages can be imported in place only where no
//!   command that may store into them can reach the import.
//! - **The twin.** Beside its copy, such a buffer takes the runtime's own buffer over the same
//!   pages ([`Device::make_read_twin`]), made as a writable no-copy buffer's import would be (in a
//!   guest RAM's chunk, inside a live import, by its pointer or as dma-bufs; never a second import
//!   of a page, `host_aliases.rs`), and never a copy: where the import would be one, or fails,
//!   the buffer takes no twin (`WorkCounts::read_twins_unimported`). Nor does it where the
//!   device's policy contains no access (`ContainmentPolicy::UnsafeOff`,
//!   `WorkCounts::read_twins_uncontained`), where a store may reach past what it names. The twin
//!   takes no address of the Metal space (it lies outside it, `address.rs`), no resource ID, and
//!   no place among the live buffers (`live_buffers.rs`); no call returns it. Its key is a read
//!   twin's ([`ResourceKey::read_twin`](super::resource::ResourceKey::read_twin)), owned by its
//!   buffer, so a command that reads through it uses the buffer (`CommandBuffer::used_resources`).
//!   It holds the pages its buffer lies in, so the caller's deallocator runs once both let go of
//!   them, and it goes as the buffer lies elsewhere (a rebind, a detach) or goes.
//! - **Where it is read, and why nothing writes it.** The program names the buffer alone, and
//!   every call that names it (a binding, a blit, a pass's visibility results, an attachment, an
//!   indirect command, `useResource`) reaches its copy, whose writes are dropped at completion as
//!   before. Only a site whose command cannot store into the bytes it reads takes the twin in its
//!   place:
//!   - a blit's copy out of the buffer, into a buffer or a texture: a copy reads its source alone
//!     (`vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`). Counted: `WorkCounts::read_twin_blit_reads`.
//!
//!   Each such site takes the twin only while no earlier command of its command buffer may have
//!   written the buffer's copy (a write of its key, or a function that may store through an
//!   address, `State::writes_unnamed`): after one, the command buffer reads the copy, so a later
//!   command reads the dropped write back as Metal's GPU caches let it (`read_only_view.rs`),
//!   and the next command buffer reads the pages (`WorkCounts::read_twin_reads_after_writes`).
//!   And every encoder refuses a command whose uses write a twin's key: it records nothing, and
//!   its command buffer fails (`WorkCounts::read_twin_writes_refused`), which only a defect of
//!   the runtime's meets.
//! - **What it saves.** A command buffer that reads such a buffer only through its twin uses
//!   none of its copy, so its commit compares none of it (`copied.rs`,
//!   `WorkCounts::copied_copy_ins_compared`, `copied_copy_in_bytes_compared`): the twin is placed
//!   over the bytes before its buffer, so the bytes' key is the twin's (`host_aliases.rs`'s
//!   `Placements`), which a use of the copy takes and a use of the twin takes nothing beside.

use super::address::{AddressPlacement, Place};
use super::buffer::{Buffer, Naming};
use super::command_buffer::CommandBuffer;
use super::device::Device;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::host_memory::HostParts;
use super::values::ResourceOptions;
use metal2vulkan::device::ContainmentPolicy;
use std::sync::Arc;

/// The refusal a read twin answers where its pages' import would be a copy of them
/// (`Device::copied_buffer`): it takes none.
pub(super) fn uncopied() -> Error {
    internal("a read twin is never a copy of its pages")
}

#[cfg(test)]
thread_local! {
    /// The tests' negative control: an encoder on this thread records a command that writes a
    /// read twin, which it refuses otherwise ([`refuses_writes`]).
    pub(crate) static UNGUARDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether an encoder refuses a command whose uses write a read twin (`encoder.rs`): always, but
/// on a test's thread that turned the guard off ([`UNGUARDED`]), to show what it keeps from the
/// pages.
#[cfg(test)]
pub(super) fn refuses_writes() -> bool {
    !UNGUARDED.with(std::cell::Cell::get)
}

/// Whether an encoder refuses a command whose uses write a read twin (`encoder.rs`): always.
#[cfg(not(test))]
pub(super) fn refuses_writes() -> bool {
    true
}

impl Device {
    /// Give `of`, a new no-copy buffer whose GPU writes are dropped lying in its copy, made for
    /// `call` of `length` bytes and `options` over the caller's `parts`, its read twin (the module
    /// docs), where the device's policy contains every access and the pages are imported in
    /// place; else nothing, counted.
    ///
    /// # Safety
    ///
    /// `parts` are the memory `of` was made over, which meets [`HostMemory::new`]'s contract for
    /// `length` bytes while `of` holds its pages.
    ///
    /// [`HostMemory::new`]: crate::HostMemory::new
    pub(super) unsafe fn make_read_twin(
        &self,
        of: &Buffer,
        (call, length, options): (&'static str, u64, ResourceOptions),
        parts: HostParts,
    ) {
        let shared = self.shared();
        if self.compile_device().policy().containment != ContainmentPolicy::Contained {
            shared.counts.read_twin_not_made(true);
            return;
        }
        let Some(cell) = of.host_cell() else {
            return;
        };
        let Some(pages) = cell.current() else {
            return;
        };
        let made = (call, length, options, Place::At(AddressPlacement::Outside));
        // SAFETY: the caller's contract; the twin releases nothing to the caller, and holds the
        // pages `of` lies in, whose release keeps them mapped (below).
        let twin =
            unsafe { self.one_copy_buffer(made, parts, None, false, false, Naming::ReadTwin) };
        let twin =
            (twin.ok()).filter(|twin| twin.host_pages().is_some_and(|in_pages| !in_pages.copied()));
        let Some(twin) = twin else {
            shared.counts.read_twin_not_made(false);
            return;
        };
        // The pages `of` lies in, and so their release to the caller, go only once the twin's
        // pages go too.
        if let Some(in_pages) = twin.host_pages() {
            in_pages.park(Arc::clone(&pages));
        }
        (shared.owners).read_twin(&twin, of, shared.counts.submissions_made());
        cell.keep_read_twin(twin);
        shared.counts.read_twin_made();
    }
}

impl Buffer {
    /// Its read twin, while it has one (the module docs): a no-copy buffer whose GPU writes are
    /// dropped, lying in its copy, over pages imported in place for it.
    pub(super) fn read_twin(&self) -> Option<Buffer> {
        self.host_cell()?.read_twin()
    }
}

impl CommandBuffer {
    /// Whether a command encoded into it so far may have written the bytes of `key`: a use that
    /// wrote it, a function that may store through an address (`State::writes_unnamed`), or
    /// writes it did not keep apart.
    pub(super) fn may_have_written(&self, key: super::resource::ResourceKey) -> bool {
        if self.uses_unnoted() || self.state().writes_unnamed {
            return true;
        }
        self.dirty().may_have_written(key)
    }
}

impl EncoderCore {
    /// The buffer a blit's copy out of `buffer` reads its bytes through: its read twin, in place,
    /// where it has one and no earlier command of the command buffer may have written its copy
    /// (the module docs); else `None`, the buffer itself.
    pub(super) fn read_source(&self, buffer: &Buffer) -> Option<Buffer> {
        let twin = buffer.read_twin()?;
        let counts = &self.device().shared().counts;
        if self
            .command_buffer()
            .may_have_written(buffer.identity().key)
        {
            counts.read_twin_read_after_write();
            return None;
        }
        counts.read_twin_blit_read();
        Some(twin)
    }
}
