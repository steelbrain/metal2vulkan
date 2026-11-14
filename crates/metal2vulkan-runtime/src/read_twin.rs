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
//!   - a texture slot of an encoder's own draw or dispatch, bound to the program's texture over
//!     the buffer's rows, where the command only reads the slot (`binding::TwinSlots`): every use
//!     of it the plan lists only reads, as a sampled or storage image (proven from lowering, an
//!     unknown access being a write), and every descriptor of it is that image, none an element
//!     of the heap, a texel buffer or a plane. The slot then binds the texture's read twin: the
//!     runtime's texture of the same descriptor over the twin's bytes, made with the texture and
//!     kept only where it lies in place over them, the device's check of textures in place
//!     passed (`Buffer::make_twin_texture`; else `WorkCounts::read_twin_textures_not_in_place`),
//!     and only where it serves the slot's bindings with no shadow of its usage. A slot read
//!     through a snapshot, narrowed to levels, or presenting an attachment of the draw's instance
//!     binds the texture itself. Counted: `WorkCounts::read_twin_slot_reads`.
//!   - a buffer slot of an encoder's own draw or dispatch bound to the buffer, where the command
//!     reaches no memory through a root table (so no address of the buffer reaches its
//!     functions, and no loaded address reaches the twin) and only reads the slot: every use of
//!     it the plan lists only reads, as a storage or uniform buffer (proven from lowering, writes
//!     through the slot's escaped address among them), and every descriptor of it is the
//!     buffer's, none a vertex buffer. The slot then binds the twin at the slot's offset, where
//!     the twin's base is the buffer's (`Buffer::slot_twin`). Counted:
//!     `WorkCounts::read_twin_slot_reads`.
//!
//!   Each such site takes the twin only while no earlier command of its command buffer may have
//!   written the buffer's copy (a write of its key, or a function that may store through an
//!   address, `State::writes_unnamed`): after one, the command buffer reads the copy, so a later
//!   command reads the dropped write back as Metal's GPU caches let it (`read_only_view.rs`),
//!   and the next command buffer reads the pages (`WorkCounts::read_twin_reads_after_writes`).
//!   And every encoder refuses a command whose uses write a twin's key, a texture twin's uses
//!   being its buffer twin's bytes: it records nothing, and its command buffer fails
//!   (`WorkCounts::read_twin_writes_refused`), which only a defect of the runtime's meets.
//! - **What it saves.** A command buffer that reads such a buffer only through its twin uses
//!   none of its copy, so its commit compares none of it (`copied.rs`,
//!   `WorkCounts::copied_copy_ins_compared`, `copied_copy_in_bytes_compared`): the twin is placed
//!   over the bytes before its buffer, so the bytes' key is the twin's (`host_aliases.rs`'s
//!   `Placements`), which a use of the copy takes and a use of the twin takes nothing beside.

use super::address::{AddressPlacement, Place};
use super::binding::TwinKinds;
use super::buffer::{Buffer, Naming, TextureTwin};
use super::command_buffer::CommandBuffer;
use super::device::{Device, DeviceShared};
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::host_memory::HostParts;
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::Access;
use super::values::ResourceOptions;
use metal2vulkan::device::ContainmentPolicy;
use std::any::Any;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

/// The refusal a read twin answers where its pages' import would be a copy of them
/// (`Device::copied_buffer`): it takes none.
pub(super) fn uncopied() -> Error {
    internal("a read twin is never a copy of its pages")
}

/// Whether a command binds a read twin at a slot it may write (`binding::TwinSlots`): never,
/// but on a test's device that asked it to (`DeviceShared::read_twin_unproven`), to show that the
/// twin is the pages in place and what the slots' proof keeps from them.
#[cfg(test)]
pub(super) fn binds_unproven(device: &DeviceShared) -> bool {
    device.read_twin_unproven.load(Ordering::Relaxed)
}

/// Whether a command binds a read twin at a slot it may write (`binding::TwinSlots`): never.
#[cfg(not(test))]
pub(super) fn binds_unproven(_device: &DeviceShared) -> bool {
    false
}

/// Whether an encoder refuses a command whose uses write a read twin (`EncoderCore`'s
/// `refuses_twin_write`): always, but on a test's device that turned the guard off
/// (`DeviceShared::read_twin_unguarded`), to show what it keeps from the pages.
#[cfg(test)]
pub(super) fn refuses_writes(device: &DeviceShared) -> bool {
    !device.read_twin_unguarded.load(Ordering::Relaxed)
}

/// Whether an encoder refuses a command whose uses write a read twin (`EncoderCore`'s
/// `refuses_twin_write`): always.
#[cfg(not(test))]
pub(super) fn refuses_writes(_device: &DeviceShared) -> bool {
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
        // Its copy is read in place through the twin, which a reach of memory no use names does
        // not take (`WorkCounts::copied_unnamed_reach_twin_copy_ins`).
        pages.set_copy_twinned(true);
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

impl Buffer {
    /// Give `texture`, the program's texture of `descriptor` just made over the buffer's bytes
    /// at `offset` with rows `bytes_per_row` apart, its read twin, where the buffer has one (the
    /// module docs): the runtime's texture of the same descriptor over the read twin's bytes,
    /// kept where it lies in place over them; else nothing, counted.
    pub(super) fn make_twin_texture(
        &self,
        texture: &Texture,
        descriptor: &TextureDescriptor,
        (offset, bytes_per_row): (u64, u64),
    ) {
        let (Some(twin), Some(cell)) = (self.read_twin(), self.host_cell()) else {
            return;
        };
        let counts = &self.generation().shared().counts;
        let made = (twin
            .new_runtime_texture(descriptor, offset, bytes_per_row)
            .ok())
        .filter(Texture::aliases_buffer);
        let Some(made) = made else {
            counts.read_twin_texture(false);
            return;
        };
        let weak: Weak<dyn Any + Send + Sync> = texture.downgrade();
        cell.keep_twin_texture(TextureTwin {
            of: texture.object_id(),
            texture: weak,
            twin: Arc::new(made),
        });
        counts.read_twin_texture(true);
    }
}

/// Whether a command of `command_buffer` may read a slot through a read twin: a buffer of its
/// device took one.
pub(super) fn twinned(command_buffer: &CommandBuffer) -> bool {
    command_buffer.generation().counts.read_twins_any()
}

impl Buffer {
    /// The buffer's read twin, for a command of `command_buffer` that binds it at a buffer slot it
    /// only reads (`binding::TwinSlots`): where it has one at its own base (so the ranges the
    /// command planned for the buffer are the twin's), and no earlier command of the command
    /// buffer may have written its copy (after one, the command reads the copy, so it reads that
    /// write back). Else `None`, the buffer itself.
    pub(super) fn slot_twin(&self, command_buffer: &CommandBuffer) -> Option<Buffer> {
        let twin = self.read_twin()?;
        if twin.base() != self.base() {
            return None;
        }
        let counts = &command_buffer.generation().counts;
        if command_buffer.may_have_written(self.identity().key) {
            counts.read_twin_read_after_write();
            return None;
        }
        counts.read_twin_slot_read();
        Some(twin)
    }
}

impl Texture {
    /// The texture's read twin, for a command of `command_buffer` that binds it at a slot of
    /// `kinds` and only reads it there (`binding::TwinSlots`): the runtime's texture over its
    /// buffer's read twin's bytes, in place, where it has one that serves each kind of binding
    /// with no shadow of its usage, and no earlier command of the command buffer may have written
    /// the buffer's copy or the texture (after one, the command reads the copy, so it reads that
    /// write back as Metal's GPU caches let it). Else `None`, the texture itself.
    pub(super) fn read_twin(
        &self,
        command_buffer: &CommandBuffer,
        kinds: TwinKinds,
    ) -> Option<Texture> {
        let buffer = &self.buffer_texels()?.buffer;
        let kept = buffer.host_cell()?.twin_texture(self.object_id())?;
        let twin = kept.downcast_ref::<Texture>()?.clone();
        let shadowed = (kinds.storage && twin.needs_usage_shadow(true))
            || (kinds.sampled && twin.needs_usage_shadow(false));
        if shadowed || !twin.aliases_buffer() {
            return None;
        }
        let counts = &command_buffer.generation().counts;
        let own = self.bound_use(Access::NONE, Access::NONE).key;
        let key = buffer.identity().key;
        if command_buffer.may_have_written(key) || command_buffer.may_have_written(own) {
            counts.read_twin_read_after_write();
            return None;
        }
        counts.read_twin_slot_read();
        Some(twin)
    }
}
