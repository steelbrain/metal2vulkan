//! Textures over host pages, in place: a 2D texture (or array) whose declared usage samples, reads,
//! renders into or writes it, over a no-copy buffer's host pages or over a buffer the runtime
//! placed in fresh host pages, is a linear image aliasing those pages, where the device's own check
//! shows that the operations that may touch bytes beside the texels leave every byte alone. Over a
//! copied host buffer (`copied.rs`) the image aliases the copy the GPU uses in the pages' stead,
//! where the check passed for the pages, and for fresh pages where the copy lies in them, for
//! texels that lie in lines as they lie in each (`texture/route.rs`); a `DRM_FORMAT_MOD_LINEAR`
//! image over a copy in memory the device allocates, for the runtime's pooled memory too.
//!
//! - **What Vulkan says.** An image bound to imported memory starts `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`: "If the pNext chain includes a
//!   VkExternalMemoryImageCreateInfo ... structure whose handleTypes member is not 0, initialLayout
//!   must be VK_IMAGE_LAYOUT_UNDEFINED"), and every image leaves the layout it is made in before
//!   any use: its first move, to `GENERAL` (`texture/first_move.rs`), an acquire from
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT` for one over dma-buf pages (`dma_buf.rs`). Of that move the
//!   Synchronization chapter says "Image layout transitions may perform read and write accesses on
//!   all memory bound to the image subresource range", and "If the old layout is
//!   VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded". Metal keeps every
//!   CPU write to bytes no command writes, at every moment. A shader's read touches the texels
//!   alone (the Memory Model: "A memory operation must not affect memory at a memory location not
//!   within its set of memory locations"), and every command uses the image in `GENERAL` with
//!   memory barriers, which move no layout (`tracker.rs`), but for two operations of an image that
//!   renders ([`Access::Renders`]): a render pass's load and store operations, of which the Render
//!   Pass chapter says "any writes performed by a load operation (as defined by its access masks)
//!   to a given attachment may read and write back any memory locations within the image
//!   subresource bound for that attachment", and a store alike; and, for an image with feedback
//!   loop usage, its moves into `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` and back around a pass
//!   whose draws read it through a snapshot (`render_target::moved_barriers`), layout transitions
//!   as the first move is. A shader's storage write of a texel is a memory operation on that
//!   texel's locations, but of an image whose tiling is linear a driver may write more than the
//!   texel (a block of them, past a row's last texel), which Vulkan's text does not rule out for
//!   the bytes beside it: an image a shader writes ([`Access::Writes`]) is checked for its storage
//!   writes.
//! - **Relying on the driver, checked.** The owner's ruling of 2026-10-26 ("In place, trusting the
//!   driver") takes the image where the device's own check shows the move leaving every byte
//!   alone: a deliberate, scoped exception to never relying on undocumented Vulkan behaviour,
//!   recorded in `docs/FAST-PATHS.md`. The owner's rulings of 2026-10-27 and 2026-11-07 widen it to
//!   textures that render, where the check shows a pass's load and store and the feedback loop
//!   moves leaving every byte alone too, whatever a linear image costs the GPU: a texture over the
//!   caller's pages is the GPU's to work on there, as a Mac's GPU works on them, and so are
//!   textures a shader writes, where the check shows a kernel's storage writes of the texels
//!   leaving every other byte alone, on a discrete GPU as on any other (a discrete Mac's GPU
//!   writes a `Shared` texture across its bus). The check
//!   (`in_place_check.rs`) runs once per driver and its version, per memory class
//!   ([`PagesClass`]), access ([`Access`]) and how the texels lie in the device's lines of memory
//!   ([`RowEnds`]: whether a line holds both a texel and a byte beside the texels, which a device
//!   that writes back a texel's line whole puts back over the CPU's write), and its verdicts are
//!   kept in the shader cache's content records ([`record_name`]); a class, access and row ends it
//!   did not pass keep the shadow.
//! - **Which textures** ([`in_place`]): under `HostPageTexturePolicy::InPlace`, those over a class
//!   whose check passed, for texels that lie in lines as the texture's do ([`RowEnds::of`], from
//!   the texels' address, offset and pitches), for each access the texture's declared usage (as
//!   Metal reports it, an `Unknown` one every usage) takes ([`Access::takes`]): it renders (`RenderTarget`), a shader
//!   writes it (`ShaderWrite` or `ShaderAtomic`, of texels of 1, 2, 4, 8 or 16 bytes, the widths
//!   the check's kernel stores: [`STORED_TEXEL_BYTES`]; those of 1 or 2 bytes by a check of their
//!   own, [`Access::NarrowWrites`], so a driver's writes of such texels decide none of the wider
//!   ones'), or it is sampled or read alone. Of a texture
//!   a shader writes over a buffer's rows whose padding is free ([`Padding`]: real Macs' GPUs
//!   disagree on those bytes under a kernel's writes, and the owner's ruling of 2026-11-07 takes
//!   the x86 Macs' envelope), the check of writes is the one that lets a byte of the rows'
//!   padding change ([`Access::BufferWrites`], [`Access::NarrowBufferWrites`]); renders keep
//!   every byte, but over a buffer that declares its rows' padding unobserved
//!   (`BufferDescriptor::padding_unobserved`), whose free padding the check of renders that lets
//!   it change takes ([`Padding::Unobserved`], [`Access::UnobservedRenders`]: never a byte past
//!   the last row's pitch); an IOSurface plane's rows keep every byte. Plain Metal also writes through a texture beyond its declared usage, and renders
//!   into one made without `RenderTarget`: such a write binds the texture's usage shadow
//!   (`texture/usage_shadow.rs`), and such a pass renders through an attachment stand-in
//!   (`attachment_stand_in.rs`), since the image has neither storage nor attachment usage; the
//!   copies of both touch texels alone.
//!   Elsewhere the texture is a shadow (`shadow.rs`), its notice naming why ([`ShadowCause`]) and
//!   counted by cause (`WorkCounts::host_texture_shadows_*`, and those that render apart).
//! - **The runtime's pooled memory.** A texture over a buffer of the runtime's own in its pooled
//!   memory whose rows the device's own linear image lays out otherwise (a pitch the driver does
//!   not take: ANV lays linear rows tight, RADV at multiples of 256 bytes) takes a
//!   `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch bound in the buffer's memory, no external
//!   memory, where the device makes one ([`PagesClass::Pooled`], `texture.rs`'s
//!   `pooled_pitched_image`): Vulkan lets such an image be bound to memory of no external handle
//!   type (`VUID-VkImageCreateInfo-tiling-02261`, `VUID-vkBindImageMemory-image-01445`), but it
//!   starts `UNDEFINED` (`VUID-VkImageCreateInfo-initialLayout-12478`), so its first move may
//!   discard the bytes as an image over host pages' may, and the same rule takes it: where the
//!   device's policy and its check of that class pass for the texture's accesses (of one that
//!   renders over padding its buffer declares unobserved, [`Access::UnobservedRenders`]). A
//!   texture made there before the class's verdicts are known is a shadow that moves to that image
//!   once they pass, as one over host pages does (below;
//!   `WorkCounts::pooled_textures_moved_in_place`).
//! - **The runtime's exported memory.** A texture over a buffer whose memory the runtime exports
//!   (`exported_memory.rs`: a block that exportable buffers of one handle type share) takes the
//!   device's linear image, else the `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch, each made for
//!   the handle type the memory's own record names (`memory.rs`'s `Memory::bound_memory`), as an
//!   image bound in exported memory must name it (`VUID-vkBindImageMemory-memory-02728`): so it
//!   starts `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`) whichever it is, and the same rule
//!   takes it, under the check of that handle type's exported memory
//!   ([`PagesClass::ExportedFd`], [`PagesClass::ExportedDmaBuf`]), moving in place once the
//!   verdicts pass as over the pooled memory. Where no such image binds in the memory itself (the
//!   device's image format query, `VkPhysicalDeviceExternalImageFormatInfo`, takes the handle
//!   type for none of the texture's format, usage and tiling but with a dedicated allocation, or
//!   the image made asks one, or the memory is the buffer's dedicated allocation,
//!   `VUID-vkBindImageMemory-memory-10926`), a texture over memory exported as a dma-buf takes a
//!   `DRM_FORMAT_MOD_LINEAR` image at the texels' offset with Metal's pitches, bound to a
//!   dedicated import of that memory's own dma-buf (`texture.rs`'s `exported_dma_buf_alias`, as
//!   an image over dma-buf pages is, its first move acquiring it from
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT`): the buffer's bytes in place through a second
//!   `VkDeviceMemory` of one payload, under the same check, whose own textures take that route
//!   wherever the image in the memory is refused, so its verdicts are of the image a texture
//!   takes. Elsewhere (an opaque file descriptor, or no such image either) the texture is a
//!   shadow, its notice naming why. Where an image in the memory lies is decided as for any
//!   texture over a buffer (`linear_plan::aliases`), of the image made for the handle type, for
//!   each texture (each plane of a picture) apart: the texels' offset in the memory (the buffer's
//!   first byte lies at
//!   a multiple of 256 bytes there, or its `aligned_byte` does, `BufferDescriptor::aligned_byte`;
//!   plus the texture's offset in the buffer) is a multiple of the
//!   `alignment` that `vkGetImageMemoryRequirements` answers for that image of the texture's
//!   format, extent and usage; its first subresource (`vkGetImageSubresourceLayout`) starts at
//!   offset 0 with Metal's `rowPitch`; and its `size` stays inside the buffer. The device's
//!   linear image is tried first, then the `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch, whose
//!   own alignment decides.
//! - **The runtime's buffers' exportable blocks.** Where the device allocates the blocks its
//!   buffers lie in exportable as opaque file descriptors and never exports them (`memory.rs`'s
//!   "Buffers in exportable blocks"), a texture over such a buffer takes the device's linear image
//!   made for that handle type, so starting `UNDEFINED`, under the check of such blocks
//!   ([`PagesClass::FdBlocks`]), and no `DRM_FORMAT_MOD_LINEAR` one. The class is made only
//!   where every image a texture over a plain block's buffer takes has that image over such a
//!   block (`texture.rs`'s `Device::fd_blocks_cover`), so elsewhere its verdicts are unchecked. Its
//!   verdicts decide, before any buffer lies there, whether the device's buffers do
//!   ([`fd_blocks_take`]): where every texture over such a buffer takes in place what it takes
//!   over a plain block's, whose linear image starts `PREINITIALIZED` and needs no verdict.
//! - **Before the verdict.** A texture made before its class's verdict for its access is known
//!   waits for nothing: it is a shadow ([`ShadowCause::CheckPending`], counted apart), and the
//!   first such texture starts the check on a thread of its own. Once its class's verdicts are
//!   known, it moves in place where they pass, at the commit of the device's only uncommitted
//!   command buffer (`texture/storage_form.rs`, `WorkCounts::host_textures_moved_in_place`), and
//!   stays a shadow where they do not, as a texture made then would be. Its making waits for no
//!   check: the check's rounds take seconds on some devices and builds (1.8 s for the first
//!   texture, waited for, on a Radeon 780M in a debug build), past what a program's first texture
//!   may take, a guest's message that makes one among them. A device whose record holds the
//!   verdicts knows them as it is made, and starts none: those its driver's earlier check
//!   measured (passed or failed), and those it could not check for a cause that lasts for the
//!   driver's identity (unchecked: the device lacks an extension, a feature, an import, a format
//!   or an image of the shape). One it could not check for a cause that may pass in another run
//!   (pages or memory not made, an import refused, the writer thread not started, a first move,
//!   pass or kernel's stores that failed) is no measurement, so the record keeps it pending and
//!   the next device's check runs that class and access again (`encode`), its note saying which
//!   the cause was. A program that will make such textures
//!   starts the check earlier, where it likes, through `Device::settle_in_place_checks` (with no
//!   wait: a zero timeout). A buffer's making starts none: until the verdicts of exportable
//!   blocks are known the runtime's own buffers lie in plain blocks ([`Device::fd_blocks_checked`]).
//! - **Visibility.** A command reads the image's texels in the memory a buffer over the same pages
//!   reads: the pages' host-visible, host-coherent import (a dma-buf's own import is of the same
//!   memory type). So a CPU write reaches a command through the image as it reaches the buffer's
//!   commands: one the program made before the scheduler submits the command buffer's batch is in
//!   that submission's first access scope ("The first access scope includes all host writes to
//!   mappable device memory that are available to the host memory domain", Host Write Ordering
//!   Guarantees), and the commit submits nothing itself (`scheduler/mod.rs`), so a write made after
//!   the commit and before that submission is too.

use crate::device::lock;
use crate::shader_cache::store::{hex, Lookup};
use crate::Device;
use ash::vk;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::TextureUsage;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// The memory a texture over host pages lies in, each checked apart: the import and the image a
/// texture there takes differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PagesClass {
    /// A no-copy buffer's pages (or a no-copy heap's buffer's), imported by their pointer
    /// (`VK_EXT_external_memory_host`): a linear image made for host allocations, bound in the
    /// pages' import.
    Pointer,
    /// A no-copy buffer's pages imported as dma-bufs (`dma_buf.rs`): a `DRM_FORMAT_MOD_LINEAR`
    /// image bound to a dedicated import of a dma-buf of its own pages, its first move an acquire
    /// from `VK_QUEUE_FAMILY_FOREIGN_EXT`.
    DmaBuf,
    /// A buffer of the runtime's own in fresh anonymous pages it imported by their pointer
    /// (`fresh_pages.rs`): as [`Pointer`](Self::Pointer), in the memory type of the buffer's class.
    Fresh,
    /// A buffer of the runtime's own in a block of its pooled memory (`memory.rs`), no external
    /// memory, where the device's linear image of the texture lays its rows out otherwise than
    /// Metal (`texture.rs`'s `alias_image`): a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch
    /// (`VkImageDrmFormatModifierExplicitCreateInfoEXT`), bound in the buffer's memory, whose
    /// first move leaves `UNDEFINED`, as an image of a DRM format modifier's tiling is made in
    /// (`VUID-VkImageCreateInfo-initialLayout-12478`: `PREINITIALIZED`, whose move keeps the
    /// bytes, is for linear tiling alone).
    Pooled,
    /// A buffer of the runtime's own whose memory it exports as an opaque file descriptor
    /// (`exported_memory.rs`, a block that such buffers share): the device's linear image made
    /// for that handle type, else a `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch made for it,
    /// bound in the buffer's memory (`texture.rs`'s `alias_image`). Either names the handle type
    /// (`VUID-vkBindImageMemory-memory-02728`), so it is made `UNDEFINED`
    /// (`VUID-VkImageCreateInfo-pNext-01443`), and its first move leaves that layout.
    ExportedFd,
    /// As [`ExportedFd`](Self::ExportedFd), of memory the runtime exports as a dma-buf.
    ExportedDmaBuf,
    /// A buffer of the runtime's own in a block of the kind its buffers take where the device
    /// allocates their blocks exportable and never exports them (`memory.rs`'s
    /// `Holds::FdBuffers`, "Buffers in exportable blocks"): the device's linear image made for
    /// the block's handle type, an opaque file descriptor, bound in the buffer's memory
    /// (`texture.rs`'s `alias_image`). It names that handle type
    /// (`VUID-vkBindImageMemory-memory-02728`), so it is made `UNDEFINED`
    /// (`VUID-VkImageCreateInfo-pNext-01443`), and its first move leaves that layout. Its verdicts
    /// decide whether the device's buffers lie in such blocks at all
    /// ([`Device::fd_blocks_checked`]).
    FdBlocks,
}

impl PagesClass {
    /// Every class, in the order the check runs them and the record holds their verdicts.
    pub(crate) const ALL: [PagesClass; 7] = [
        PagesClass::Pointer,
        PagesClass::Fresh,
        PagesClass::DmaBuf,
        PagesClass::Pooled,
        PagesClass::ExportedFd,
        PagesClass::ExportedDmaBuf,
        PagesClass::FdBlocks,
    ];

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            PagesClass::Pointer => 0,
            PagesClass::Fresh => 1,
            PagesClass::DmaBuf => 2,
            PagesClass::Pooled => 3,
            PagesClass::ExportedFd => 4,
            PagesClass::ExportedDmaBuf => 5,
            PagesClass::FdBlocks => 6,
        }
    }

    /// The class in words, for a notice.
    pub(crate) fn words(self) -> &'static str {
        match self {
            PagesClass::Pointer => "host pages imported by their pointer",
            PagesClass::DmaBuf => "host pages imported as dma-bufs",
            PagesClass::Fresh => "fresh host pages the runtime imported for the buffer",
            PagesClass::Pooled => "the runtime's pooled memory at Metal's pitch",
            PagesClass::ExportedFd => "the runtime's memory exported as opaque file descriptors",
            PagesClass::ExportedDmaBuf => "the runtime's memory exported as dma-bufs",
            PagesClass::FdBlocks => "the runtime's buffers' blocks allocated exportable",
        }
    }

    /// The class of the runtime's memory exported as `handle` (`memory.rs`'s
    /// `Memory::bound_memory`): `None` for a handle type the runtime exports no memory as.
    pub(crate) fn exported(handle: vk::ExternalMemoryHandleTypeFlags) -> Option<PagesClass> {
        match handle {
            vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD => Some(PagesClass::ExportedFd),
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT => Some(PagesClass::ExportedDmaBuf),
            _ => None,
        }
    }

    /// Whether its textures' images lie in a buffer's memory of the runtime's own
    /// (`texture.rs`'s `alias_image`), which a shadow made before its verdicts moves to as such a
    /// texture's making makes it (`Device::pooled_move_image`), not over host pages.
    pub(crate) fn over_own_memory(self) -> bool {
        match self {
            PagesClass::Pooled
            | PagesClass::ExportedFd
            | PagesClass::ExportedDmaBuf
            | PagesClass::FdBlocks => true,
            PagesClass::Pointer | PagesClass::Fresh | PagesClass::DmaBuf => false,
        }
    }
}

/// What a texture over host pages does with its image in place, each checked apart: the
/// operations that may touch bytes bound to the image beside its texels differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Access {
    /// Its usage samples or reads it alone: its image takes its first move, and no other
    /// operation that may touch those bytes.
    Reads,
    /// Its usage renders into it (`RenderTarget`, whatever else it holds): its image takes its
    /// first move, render passes' load and store operations, and, made with feedback loop usage,
    /// the moves into `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` and back.
    Renders,
    /// As [`Renders`](Self::Renders), of a texture whose rows' padding is free and its buffer
    /// declares unobserved ([`Padding::Unobserved`]): its rounds are those of
    /// [`Renders`](Self::Renders), but a byte of its rows' padding (each row's bytes after its
    /// last texel to its pitch, the last row's included, and no byte past the last row's pitch)
    /// may change. A usage never takes it alone: [`in_place`] asks it in the stead of
    /// [`Renders`](Self::Renders) for such a texture ([`Access::rendered`]).
    UnobservedRenders,
    /// Its usage writes it through a shader (`ShaderWrite` or `ShaderAtomic`, whatever else it
    /// holds): its image takes its first move and kernels' storage writes of its texels.
    Writes,
    /// As [`Writes`](Self::Writes), of a texture whose rows' padding is free ([`Padding::Free`]):
    /// its rounds are those of [`Writes`](Self::Writes), but a byte of its rows' padding (each
    /// row's bytes after its last texel to its pitch, the last row's included) may change. A
    /// usage never takes it alone: [`in_place`] asks it in the stead of
    /// [`Writes`](Self::Writes) for such a texture.
    BufferWrites,
    /// As [`Writes`](Self::Writes), of a texture of 1- or 2-byte texels
    /// ([`NARROW_TEXEL_BYTES`]), checked apart in rounds of their own that store such texels: a
    /// driver may write a narrow texel through a wider access than a 4-byte one's. A usage never
    /// takes it alone: [`in_place`] asks it in the stead of [`Writes`](Self::Writes) for such a
    /// texture ([`Access::written`]).
    NarrowWrites,
    /// As [`BufferWrites`](Self::BufferWrites), of a texture of 1- or 2-byte texels: its rounds
    /// are those of [`NarrowWrites`](Self::NarrowWrites), but a byte of its rows' padding may
    /// change.
    NarrowBufferWrites,
}

/// The bytes of the texels a texture a shader writes may have to be in place: those the check's
/// kernel stores ([`Access::Writes`] and [`Access::NarrowWrites`]; the crate-owned copy-in
/// kernel's unsigned-integer views).
pub(crate) const STORED_TEXEL_BYTES: [u64; 5] = [1, 2, 4, 8, 16];

/// The bytes of the texels the check's narrow rounds store ([`Access::NarrowWrites`]), apart
/// from the others' ([`Access::Writes`]).
pub(crate) const NARROW_TEXEL_BYTES: [u64; 2] = [1, 2];

impl Access {
    /// Every access, in the order the check runs them and the record holds their verdicts.
    pub(crate) const ALL: [Access; 7] = [
        Access::Reads,
        Access::Renders,
        Access::UnobservedRenders,
        Access::Writes,
        Access::BufferWrites,
        Access::NarrowWrites,
        Access::NarrowBufferWrites,
    ];

    /// Whether a texture of Metal `usage` (as Metal reports it) takes this access: one that
    /// renders, one a shader writes (both, where it does both), or one that only samples or
    /// reads. Each of its accesses' checks covers its first move. None takes
    /// [`UnobservedRenders`](Self::UnobservedRenders), [`BufferWrites`](Self::BufferWrites) or
    /// the narrow accesses alone ([`in_place`] asks the one of its padding in the stead of
    /// [`Renders`](Self::Renders), [`Access::rendered`], and the one of its texels and padding in
    /// the stead of [`Writes`](Self::Writes), [`Access::written`]).
    pub(crate) fn takes(self, usage: TextureUsage) -> bool {
        let renders = usage.contains(TextureUsage::RENDER_TARGET);
        let writes = usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
        match self {
            Access::Reads => !renders && !writes,
            Access::Renders => renders,
            Access::Writes => writes,
            Access::UnobservedRenders
            | Access::BufferWrites
            | Access::NarrowWrites
            | Access::NarrowBufferWrites => false,
        }
    }

    /// The check of a texture that renders, whose rows' padding is as `padding` says:
    /// [`UnobservedRenders`](Self::UnobservedRenders) where its buffer declares that padding
    /// unobserved and it is free ([`Padding::Unobserved`]), else [`Renders`](Self::Renders).
    pub(crate) fn rendered(padding: Padding) -> Access {
        match padding {
            Padding::Unobserved => Access::UnobservedRenders,
            Padding::Kept | Padding::Free => Access::Renders,
        }
    }

    /// The check of a shader's writes of texels `texel` bytes wide, of a texture whose rows'
    /// padding is as `padding` says: [`Writes`](Self::Writes) of texels of 4, 8 or 16 bytes,
    /// [`NarrowWrites`](Self::NarrowWrites) of those of 1 or 2 ([`NARROW_TEXEL_BYTES`]), each's
    /// form that lets the padding change where it is free (declared unobserved or not); `None`
    /// for a width the check's kernel stores none of ([`STORED_TEXEL_BYTES`]).
    pub(crate) fn written(texel: u64, padding: Padding) -> Option<Access> {
        if !STORED_TEXEL_BYTES.contains(&texel) {
            return None;
        }
        let free = padding != Padding::Kept;
        Some(match (NARROW_TEXEL_BYTES.contains(&texel), free) {
            (false, false) => Access::Writes,
            (false, true) => Access::BufferWrites,
            (true, false) => Access::NarrowWrites,
            (true, true) => Access::NarrowBufferWrites,
        })
    }

    /// The accesses the check's rounds of this one give verdicts of, in [`ALL`](Self::ALL)'s
    /// order: its own, and of [`Renders`](Self::Renders), [`Writes`](Self::Writes) and
    /// [`NarrowWrites`](Self::NarrowWrites) those of
    /// [`UnobservedRenders`](Self::UnobservedRenders), [`BufferWrites`](Self::BufferWrites) and
    /// [`NarrowBufferWrites`](Self::NarrowBufferWrites) too, which have no rounds of their own.
    pub(crate) fn answered(self) -> &'static [Access] {
        match self {
            Access::Reads => &[Access::Reads],
            Access::Renders => &[Access::Renders, Access::UnobservedRenders],
            Access::Writes => &[Access::Writes, Access::BufferWrites],
            Access::NarrowWrites => &[Access::NarrowWrites, Access::NarrowBufferWrites],
            Access::UnobservedRenders | Access::BufferWrites | Access::NarrowBufferWrites => &[],
        }
    }

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            Access::Reads => 0,
            Access::Renders => 1,
            Access::UnobservedRenders => 2,
            Access::Writes => 3,
            Access::BufferWrites => 4,
            Access::NarrowWrites => 5,
            Access::NarrowBufferWrites => 6,
        }
    }

    /// The operations its check covers, in words, for a notice.
    pub(crate) fn words(self) -> &'static str {
        match self {
            Access::Reads => "the first move of an image",
            Access::Renders => {
                "the first move, a render pass's load and store and the feedback loop moves of an \
                 image that renders"
            }
            Access::UnobservedRenders => {
                "the first move, a render pass's load and store and the feedback loop moves of an \
                 image that renders over a buffer's rows, beside their padding it declares \
                 unobserved"
            }
            Access::Writes => {
                "the first move and a kernel's storage writes of the texels of an image a shader \
                 writes"
            }
            Access::BufferWrites => {
                "the first move and a kernel's storage writes of the texels of an image a shader \
                 writes over a buffer's rows, beside their padding"
            }
            Access::NarrowWrites => {
                "the first move and a kernel's storage writes of the 1- or 2-byte texels of an \
                 image a shader writes"
            }
            Access::NarrowBufferWrites => {
                "the first move and a kernel's storage writes of the 1- or 2-byte texels of an \
                 image a shader writes over a buffer's rows, beside their padding"
            }
        }
    }
}

/// How a texture's texels lie in the device's lines of memory, each checked apart: a line is
/// `nonCoherentAtomSize` bytes at a multiple of it, "the size and alignment in bytes that bounds
/// concurrent access to host-mapped device memory" (vkspec, Limits). A device may write back a
/// line that holds a texel whole, as the Iris Xe's clear and store, and its draws, write back the
/// line that holds a row's last texels: a line that also holds bytes beside the texels then puts
/// back an older value over a CPU write there, where a line of texels alone holds nothing the CPU
/// writes beside them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RowEnds {
    /// A line holds a texel and a byte beside the texels: between two rows or two slices, before
    /// the first texel or after the last.
    InLines,
    /// Every line holds texels alone or bytes beside them alone: the texels start and end on a
    /// line, and so does each row (or slice) where bytes lie between rows (or slices).
    OnLines,
}

impl RowEnds {
    /// Both, in the order the record holds their verdicts.
    pub(crate) const ALL: [RowEnds; 2] = [RowEnds::InLines, RowEnds::OnLines];

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            RowEnds::InLines => 0,
            RowEnds::OnLines => 1,
        }
    }

    /// How texels laid out as `layout` in a buffer whose first byte lies `start` bytes past a
    /// multiple of `line` lie in the lines of `line` bytes.
    pub(crate) fn of(start: u64, layout: &crate::linear_plan::BufferLayout, line: u64) -> RowEnds {
        let on = |at: u64| at.is_multiple_of(line.max(1));
        let first = start + layout.offset;
        let slice = (layout.rows - 1) * layout.row_pitch + layout.row_bytes;
        let rows_apart = layout.rows > 1 && layout.row_pitch != layout.row_bytes;
        let slices_apart = layout.slices > 1 && layout.slice_pitch != slice;
        let owned = on(first)
            && on(start + layout.end())
            && (!rows_apart || (on(layout.row_pitch) && on(first + layout.row_bytes)))
            && (!slices_apart || (on(layout.slice_pitch) && on(first + slice)));
        match owned {
            true => RowEnds::OnLines,
            false => RowEnds::InLines,
        }
    }

    /// How the texels lie, in words, for a notice.
    pub(crate) fn words(self) -> &'static str {
        match self {
            RowEnds::InLines => {
                "an image whose texels share a line of memory (the device's nonCoherentAtomSize) \
                 with bytes beside them"
            }
            RowEnds::OnLines => {
                "an image whose texels share no line of memory (the device's \
                 nonCoherentAtomSize) with bytes beside them"
            }
        }
    }
}

/// Whether a shader's writes of a texture's texels may change its rows' padding: each row's bytes
/// after its last texel to its pitch, the last row's included (the bytes between slices are no
/// row's). Metal leaves them to its GPU under a kernel's writes of a texture over a buffer: an
/// Intel UHD Graphics 630 (macOS 26.7.1) wrote back older values over the CPU's writes to them,
/// the last row's included, under such writes, where an M3 Max (macOS 27.0.1) and a Radeon Pro
/// 5300M kept them (`crates/metal2vulkan-conformance/probes/row_padding_writes.m`); every GPU
/// kept them under its draws, clears and stores, and an IOSurface plane's under kernels too. The
/// rule follows the x86 Macs' envelope: what one of them does to those bytes, a texture over a
/// buffer may do ([`Access::BufferWrites`]; docs/API.md, section 10, **Kernels**). A render
/// pass's load and store keep them, but over a buffer that declares them unobserved
/// (`BufferDescriptor::padding_unobserved`), which lets them change ([`Padding::Unobserved`],
/// [`Access::UnobservedRenders`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Padding {
    /// Its rows' padding is kept, as every byte beside its texels is: an IOSurface plane's rows,
    /// a texture whose buffer ends before its last row's pitch, or one whose texels and padding
    /// share a line of the device's memory with bytes past them ([`Padding::of`]).
    Kept,
    /// A texture over a buffer's rows, its buffer holding its last row to its pitch, whose
    /// texels and padding share no line with bytes past them: a shader's writes may change its
    /// padding.
    Free,
    /// As [`Free`](Self::Free), of a buffer that declares the bytes between a texture's rows
    /// unobserved (`BufferDescriptor::padding_unobserved`): a render pass's load and store may
    /// change its padding too. No byte past the last row's pitch: it shares no line with the
    /// padding.
    Unobserved,
}

impl Padding {
    /// The padding of texels laid out as `layout` in a buffer whose first byte lies `start`
    /// bytes past a multiple of `line` ([`RowEnds::of`]), where `owned` (they lie over a
    /// buffer's rows, no IOSurface plane's, and the buffer holds their last row to its pitch):
    /// free where the rows, each to its pitch, share no line with bytes past them.
    pub(crate) fn of(
        start: u64,
        layout: &crate::linear_plan::BufferLayout,
        line: u64,
        owned: bool,
    ) -> Padding {
        let padded = crate::linear_plan::BufferLayout {
            row_bytes: layout.row_pitch,
            ..*layout
        };
        match owned && RowEnds::of(start, &padded, line) == RowEnds::OnLines {
            true => Padding::Free,
            false => Padding::Kept,
        }
    }

    /// This padding, of a buffer that declares the bytes between a texture's rows unobserved
    /// where `unobserved`: [`Unobserved`](Self::Unobserved) where it is free; else as it is.
    pub(crate) fn declared(self, unobserved: bool) -> Padding {
        match (self, unobserved) {
            (Padding::Free, true) => Padding::Unobserved,
            (padding, _) => padding,
        }
    }
}

/// The verdicts a device's check gives: one for each class, access and row ends, those of
/// [`RowEnds::InLines`] first, then those of [`RowEnds::OnLines`], each class by class
/// ([`PagesClass::ALL`]), each class's accesses in [`Access::ALL`]'s order.
pub(crate) const VERDICTS: usize = RowEnds::ALL.len() * PagesClass::ALL.len() * Access::ALL.len();

/// The place of `class`'s verdict for `access` of texels that lie as `rows` says among the
/// [`VERDICTS`].
pub(crate) fn verdict_index(class: PagesClass, access: Access, rows: RowEnds) -> usize {
    (rows.index() * PagesClass::ALL.len() + class.index()) * Access::ALL.len() + access.index()
}

/// What the device's check answered for a class and access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Not known yet: no record held it, and the check has not answered.
    Pending,
    /// The access's operations on every image the check made left every byte alone (but the
    /// texels they wrote).
    Passed,
    /// The access's operations on an image the check made changed a byte.
    Failed,
    /// The check made no image of the class for the access: the device makes no such import,
    /// image or operation.
    Unchecked,
}

impl Verdict {
    fn byte(self) -> u8 {
        match self {
            Verdict::Pending => 0,
            Verdict::Passed => 1,
            Verdict::Failed => 2,
            Verdict::Unchecked => 3,
        }
    }

    fn of_byte(byte: u8) -> Option<Verdict> {
        match byte {
            0 => Some(Verdict::Pending),
            1 => Some(Verdict::Passed),
            2 => Some(Verdict::Failed),
            3 => Some(Verdict::Unchecked),
            _ => None,
        }
    }
}

/// What a shadow over host pages made before its class's verdict for its access
/// ([`ShadowCause::CheckPending`]) needs to move in place once the verdicts pass
/// (`texture/storage_form.rs`): its class and usage, its declared plan and its descriptor, which
/// the image in place is made of as the texture's would have been.
#[derive(Clone)]
pub(crate) struct PendingMove {
    pub(crate) class: PagesClass,
    pub(crate) rows: RowEnds,
    pub(crate) padding: Padding,
    pub(crate) usage: TextureUsage,
    pub(crate) plan: crate::image_plan::ImagePlan,
    pub(crate) descriptor: crate::texture_descriptor::TextureDescriptor,
}

/// Why a texture over host pages is a shadow and not an image in place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShadowCause {
    /// The device's policy asks the shadow (`HostPageTexturePolicy::Shadow`).
    Exact,
    /// Its usage writes it through a shader, and its texels are of a width (the bytes given) the
    /// check's kernel stores none of ([`STORED_TEXEL_BYTES`]): no check covers its writes. Every
    /// width a buffer backs (1, 2, 4, 8 and 16 bytes) is stored; this names one outside them.
    StoreWidth(u64),
    /// Its class's check for its access, of texels that lie in lines as its own do, has not
    /// answered yet.
    CheckPending(PagesClass, Access, RowEnds),
    /// Its class's check for its access, of texels that lie in lines as its own do, found a byte
    /// changed.
    CheckFailed(PagesClass, Access, RowEnds),
    /// Its class's check for its access, of texels that lie in lines as its own do, made no image.
    Unchecked(PagesClass, Access, RowEnds),
    /// The device makes no image over its texels as Metal lays them out: why, and the words.
    Refused(Refusal, String),
    /// Over the runtime's pooled memory ([`PagesClass::Pooled`]) that only the runtime's
    /// commands write (a tracked texture's over a `Private` buffer), its rows to their pitch take
    /// this many bytes, past [`POOLED_PITCHED_BYTES`]: a `DRM_FORMAT_MOD_LINEAR` image of them
    /// renders and samples slower than the shadow's own image with its copies.
    LinearCost(u64),
}

/// The most bytes (a texture's rows to their pitch) a tracked texture over a `Private` buffer of
/// the runtime's pooled memory takes a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch for
/// ([`PagesClass::Pooled`]); past it the texture is a shadow ([`ShadowCause::LinearCost`]). Only
/// the runtime's commands write such bytes (`Texels::only_commands_write`): no CPU reads them and
/// the device's check judges the image's first move, so both forms give the same answer, and the
/// bound chooses between two speeds: a shadow's copy per command buffer that
/// writes the texture costs a fixed few microseconds, which a small texture's linear image saves,
/// and every draw into a linear image costs more than into the shadow's optimal one, about 270
/// picoseconds a pixel against 74 on an Iris Xe whatever the texture's size, so the crossover
/// falls as a frame draws more into the texture. Measured on an Iris Xe (ANV, Mesa 26.2.6;
/// `runtime_lane::pooled_draw_costs`), each frame a pass that clears and stores each of four
/// `BGRA8Unorm` textures over `Private` buffers, drawing 0, 1, 4 or 16 blended rectangles over the
/// whole texture, then a draw that reads each whole, a frame's GPU time in microseconds as shadows
/// against in place: 104x52 at 448-byte rows (23 KB) 98.6/83.2, 110.4/99.2, 112.4/114.9,
/// 124.1/177.2; 136x68 at 576 (39 KB) 107.9/92.7, 119.8/112.4, 123.2/139.3, 143.2/246.1; 172x67 at
/// 704 (47 KB) 110.3/97.1, 123.3/119.0, 127.7/152.1, 152.9/285.5; 250x125 at 1024 (128 KB)
/// 138.1/132.0, 153.7/172.4, 169.4/260.1, 239.0/608.7; 200x200 at 832 (166 KB) 160.8/150.0,
/// 177.7/197.6, 198.3/306.7, 290.5/743.3; 250x250 at 1024 (256 KB) 187.8/190.1, 208.2/258.9,
/// 242.0/430.8, 388.3/1114.9. A texture of up to 47 KB drawn at most once a frame runs faster in
/// place; past that one draw a frame runs faster as a shadow, and four or more draws a frame do at
/// every size measured. Over bytes the CPU may write (a `Shared` or `Managed` buffer's) and an
/// untracked texture's, a texture of any access takes the image at any size wherever the device's
/// check passed, as over host pages: the GPU works on CPU-visible memory in place however much
/// slower, a shadow there only where in place is impossible (the owner's ruling, 2026-11-07).
/// There a texture that only samples gains as well, as its shadow is copied in whole at every
/// command buffer that samples it: each frame a command buffer that samples each of four
/// `BGRA8Unorm` textures whole after the CPU wrote a texel of each
/// (`runtime_lane::pooled_draw_costs::pooled_sample_costs`), a frame's GPU time in microseconds as
/// shadows at a padded pitch against linear images at the tight one, over `Shared` buffers:
/// 1308x512 (2.7 MB) 327.8/215.9, 1324x940 (5.0 MB) 802.2/380.1, 3648x2038 (30 MB)
/// 5789.3/2601.6; over `Private` ones, whose shadows stand copied in once (`shadow_standing.rs`),
/// 237.7/216.2, 404.4/409.4 and 2028.1/2433.2. A Radeon 780M (RADV) makes no such image at a pitch
/// off a multiple of 256 bytes, at which its own linear image lays rows, so no texture there takes
/// one. The same bound on every device.
pub(crate) const POOLED_PITCHED_BYTES: u64 = 48 << 10;

/// Why the device makes no image in place over a texture's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Its rows share bytes (an IOSurface plane's GPU rows): no image holds a texel twice.
    Rows,
    /// Its bytes were detached from their pages (`detach.rs`): no image is bound to them.
    Detached,
    /// The device makes no such image of the plan: its format, features, usage or limits, or an
    /// import only a dedicated allocation takes.
    Image,
    /// No image the device makes lays the rows (or slices) at the buffer's pitch.
    Pitch,
    /// The image's alignment, size or memory type misses the bytes' place.
    Placement,
    /// Over pages imported as dma-bufs: the caller's source made no one dma-buf of the texture's
    /// own pages (it failed, as a source of `udmabuf`s does where the pages are more runs of its
    /// file than one `udmabuf` takes, or answered with several chunks).
    Source,
    /// Over pages imported as dma-bufs: no `DRM_FORMAT_MOD_LINEAR` image over one dma-buf of the
    /// texture's own pages was made for another reason (the modifier's features or limits, the
    /// image's memory past the pages, or the driver's import).
    DmaBuf,
}

impl Refusal {
    /// Its place among `WorkCounts::host_texture_shadows_refused_*`.
    pub(crate) fn index(self) -> usize {
        match self {
            Refusal::Rows => 0,
            Refusal::Detached => 1,
            Refusal::Image => 2,
            Refusal::Pitch => 3,
            Refusal::Placement => 4,
            Refusal::Source => 5,
            Refusal::DmaBuf => 6,
        }
    }
}

thread_local! {
    /// This thread is the device's check's (`in_place_check.rs`): the textures it makes over host
    /// pages take the image in place whatever the verdict, and are neither counted nor announced.
    static CHECKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is the device's check's ([`CHECKING`]).
pub(crate) fn checking() -> bool {
    CHECKING.with(std::cell::Cell::get)
}

/// Mark this thread as the device's check's, for good: it is a thread of its own.
pub(crate) fn mark_checking() {
    CHECKING.with(|checking| checking.set(true));
}

impl ShadowCause {
    /// The cause in words, for the `shadowed-texture` notice.
    pub(crate) fn words(&self) -> String {
        match self {
            ShadowCause::Exact => "the device's policy asks a shadow of every texture over host \
                                   pages (HostPageTexturePolicy::Shadow)"
                .to_owned(),
            ShadowCause::StoreWidth(bytes) => format!(
                "its usage writes it through a shader (ShaderWrite or ShaderAtomic), and the \
                 device's check of storage writes stores texels of 1, 2, 4, 8 and 16 bytes, not \
                 of {bytes}: no check covers a driver's writes of such texels beside them"
            ),
            ShadowCause::CheckPending(class, access, rows) => format!(
                "the device's check of {} over {}, for {}, has not answered yet",
                access.words(),
                class.words(),
                rows.words()
            ),
            ShadowCause::CheckFailed(class, access, rows) => format!(
                "the device's check found {} over {} changing a byte the CPU wrote beside it, \
                 for {}",
                access.words(),
                class.words(),
                rows.words()
            ),
            ShadowCause::Unchecked(class, access, rows) => format!(
                "the device's check made no image over {} for {}, for {}",
                class.words(),
                access.words(),
                rows.words()
            ),
            ShadowCause::Refused(_, why) => why.clone(),
            ShadowCause::LinearCost(bytes) => format!(
                "its rows take {bytes} bytes to their pitch, past the {POOLED_PITCHED_BYTES} past \
                 which a DRM_FORMAT_MOD_LINEAR image at Metal's pitch renders and samples slower \
                 than the shadow's own image with its copies"
            ),
        }
    }
}

/// Whether a texture over host pages of `class`, its texels lying in the device's lines as `rows`
/// says, its rows' padding as `padding` says, of Metal `usage` (as Metal reports it) and texels
/// of `texel` bytes, on a device of `policy`, is tried as an image in place; else why it is a
/// shadow (the module docs). `verdict` answers the device's check for the class, each access the
/// texture takes ([`Access::takes`]), in [`Access::ALL`]'s order, and the row ends, asked only of
/// a texture the policy and texel width let through (a pending one starts the check), until one
/// did not pass: for a shader's writes, the verdict of the check of its texels' width and
/// padding ([`Access::written`]): of 1- or 2-byte texels [`Access::NarrowWrites`]', and of a
/// texture whose padding is free [`Access::BufferWrites`]' (or [`Access::NarrowBufferWrites`]'),
/// which passes wherever the one that keeps the padding does (the same rounds give both, and it
/// lets more change); for a texture that renders whose padding is free and declared unobserved,
/// [`Access::UnobservedRenders`]' ([`Access::rendered`]), likewise.
pub(crate) fn in_place(
    policy: HostPageTexturePolicy,
    usage: TextureUsage,
    (class, rows, texel, padding): (PagesClass, RowEnds, u64, Padding),
    mut verdict: impl FnMut(PagesClass, Access, RowEnds) -> Verdict,
) -> Result<(), ShadowCause> {
    if policy == HostPageTexturePolicy::Shadow {
        return Err(ShadowCause::Exact);
    }
    let written = Access::written(texel, padding);
    if Access::Writes.takes(usage) && written.is_none() {
        return Err(ShadowCause::StoreWidth(texel));
    }
    for access in Access::ALL.into_iter().filter(|access| access.takes(usage)) {
        let access = match (access, written) {
            (Access::Renders, _) => Access::rendered(padding),
            (Access::Writes, Some(written)) => written,
            (access, _) => access,
        };
        match verdict(class, access, rows) {
            Verdict::Passed => {}
            Verdict::Pending => return Err(ShadowCause::CheckPending(class, access, rows)),
            Verdict::Failed => return Err(ShadowCause::CheckFailed(class, access, rows)),
            Verdict::Unchecked => return Err(ShadowCause::Unchecked(class, access, rows)),
        }
    }
    Ok(())
}

/// Whether an image in place of a texture of Metal `usage` that renders, over host pages of
/// `class`, its texels of `texel` bytes lying in the device's lines as `rows` says, takes the
/// storage usage too: where no shader writes it by its usage (whose own check covers such
/// writes), and `verdict` passed the check of a shader's writes of texels of its width that keeps
/// every byte beside them ([`Access::written`] of [`Padding::Kept`]: a render target's rows keep
/// their padding). A draw that reads such a texture through a snapshot fetches it in the pixel
/// interlock through that usage (`render/snapshot.rs`), and with it a shader's writes reach the
/// texels in place, as the check's writes do.
pub(crate) fn storable(
    usage: TextureUsage,
    (class, rows, texel): (PagesClass, RowEnds, u64),
    mut verdict: impl FnMut(PagesClass, Access, RowEnds) -> Verdict,
) -> bool {
    Access::Renders.takes(usage)
        && !Access::Writes.takes(usage)
        && Access::written(texel, Padding::Kept)
            .is_some_and(|written| verdict(class, written, rows) == Verdict::Passed)
}

/// A device's check verdicts (`DeviceShared::in_place`): one per class, access and row ends
/// ([`VERDICTS`]), and whether its check started and settled.
pub(crate) struct InPlaceChecks {
    verdicts: [AtomicU8; VERDICTS],
    /// The device's line of memory, `nonCoherentAtomSize` bytes ([`RowEnds`]).
    line: u64,
    /// The check was started on its thread (at most once per device).
    started: AtomicBool,
    /// Every class's verdict is known: from the record, or the check ended.
    settled: Mutex<bool>,
    wake: Condvar,
    /// The device fails its check whatever the moves did (`DeviceOptions::fail_in_place_check`),
    /// and keeps no record of it.
    fail: bool,
    /// The check's run, from its start to its last verdict, in nanoseconds (0 before it ends).
    took: AtomicU64,
    /// What the check found changed, where a class failed: a line for each.
    notes: Mutex<Vec<String>>,
    /// The calls waiting for the verdicts ([`InPlaceChecks::wait`]), for which no round steps
    /// aside (`step_aside.rs`).
    waiters: AtomicU32,
}

impl InPlaceChecks {
    /// No verdict yet, on a device whose line is `line` bytes ([`RowEnds`]); `fail` as
    /// [`InPlaceChecks::fail`] says.
    pub(crate) fn new(fail: bool, line: u64) -> InPlaceChecks {
        InPlaceChecks {
            verdicts: [0; VERDICTS].map(AtomicU8::new),
            line: line.max(1),
            started: AtomicBool::new(false),
            settled: Mutex::new(false),
            wake: Condvar::new(),
            fail,
            took: AtomicU64::new(0),
            notes: Mutex::new(Vec::new()),
            waiters: AtomicU32::new(0),
        }
    }

    /// Whether a call waits for the verdicts ([`InPlaceChecks::wait`]).
    pub(crate) fn awaited(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }

    /// Note what the check found changed.
    pub(crate) fn note(&self, line: String) {
        lock(&self.notes).push(line);
    }

    /// The device's line of memory, in bytes ([`RowEnds`]).
    pub(crate) fn line(&self) -> u64 {
        self.line
    }

    /// `class`'s verdict for `access` of texels that lie in lines as `rows` says, now.
    pub(crate) fn verdict(&self, class: PagesClass, access: Access, rows: RowEnds) -> Verdict {
        let byte = self.verdicts[verdict_index(class, access, rows)].load(Ordering::Acquire);
        Verdict::of_byte(byte).unwrap_or(Verdict::Pending)
    }

    /// Every verdict now, in [`VERDICTS`]' order.
    pub(crate) fn verdicts(&self) -> [Verdict; VERDICTS] {
        std::array::from_fn(|at| {
            let byte = self.verdicts[at].load(Ordering::Acquire);
            Verdict::of_byte(byte).unwrap_or(Verdict::Pending)
        })
    }

    /// Give `class` its verdict for `access` of texels that lie in lines as `rows` says.
    pub(crate) fn set(&self, class: PagesClass, access: Access, rows: RowEnds, verdict: Verdict) {
        let at = verdict_index(class, access, rows);
        self.verdicts[at].store(verdict.byte(), Ordering::Release);
    }

    /// Whether the check is to fail whatever the moves did.
    pub(crate) fn forced_failure(&self) -> bool {
        self.fail
    }

    /// Note the check started; whether it had not been before.
    pub(crate) fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// Every verdict is known now; the check took `took` where it ran.
    pub(crate) fn settle(&self, took: Option<Duration>) {
        if let Some(took) = took {
            let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
            self.took.store(nanos.max(1), Ordering::Release);
        }
        *lock(&self.settled) = true;
        self.wake.notify_all();
    }

    /// Wait until every verdict is known, or `timeout` passed: whether they are. Meanwhile the
    /// check's rounds step aside for nothing (`step_aside.rs`).
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiting = Waiting(&self.waiters);
        let mut settled = lock(&self.settled);
        while !*settled {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            settled = match self.wake.wait_timeout(settled, left) {
                Ok((settled, _)) => settled,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        true
    }

    /// How long the check ran in this process, where it ran and ended.
    pub(crate) fn took(&self) -> Option<Duration> {
        match self.took.load(Ordering::Acquire) {
            0 => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }
}

/// A call waiting for the verdicts ([`InPlaceChecks::wait`]), counted until it returns.
struct Waiting<'a>(&'a AtomicU32);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What the device's verdicts say of the runtime's own buffers' blocks allocated exportable
/// ([`fd_blocks_take`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FdBlocksTake {
    /// Its buffers lie there: a texture over one takes in place whatever it takes over a plain
    /// block's buffer.
    Take,
    /// A verdict that decides it is pending: its buffers lie in plain blocks meanwhile.
    Pending,
    /// A texture over such a buffer, of this access and row ends, would be a shadow where over a
    /// plain block's buffer it is in place: its buffers lie in plain blocks.
    Refused(Access, RowEnds),
}

/// What `verdict` (each class's, access's and row ends') says of the runtime's own buffers lying
/// in blocks allocated exportable, never exported ([`PagesClass::FdBlocks`]; `memory.rs`'s
/// "Buffers in exportable blocks"): they lie there where every texture over such a buffer takes
/// in place what it takes over a buffer of a plain block, whose blocks hold the same memory
/// exported as no handle type. Over a plain block's buffer, the device's own linear image is
/// made `PREINITIALIZED`, whose first move keeps its bytes, and needs no verdict; over an
/// exportable block's, it names the block's handle type and so is made `UNDEFINED`
/// (`VUID-VkImageCreateInfo-pNext-01443`), its other operations the same: so the class's verdict
/// of that move ([`Access::Reads`]) must pass for texels of both row ends; and every access a
/// texture over a plain block's buffer takes in place by the verdicts of plain pooled memory
/// ([`PagesClass::Pooled`]) it must take over such a block's by the class's: so every verdict of
/// the pooled class that passed must pass for the class too. Pending where one of those is
/// pending; else refused for the first access and row ends that did not pass.
pub(crate) fn fd_blocks_take(
    verdict: impl Fn(PagesClass, Access, RowEnds) -> Verdict,
) -> FdBlocksTake {
    let mut pending = false;
    for rows in RowEnds::ALL {
        match verdict(PagesClass::FdBlocks, Access::Reads, rows) {
            Verdict::Passed => {}
            Verdict::Pending => pending = true,
            Verdict::Failed | Verdict::Unchecked => {
                return FdBlocksTake::Refused(Access::Reads, rows)
            }
        }
    }
    for access in Access::ALL {
        for rows in RowEnds::ALL {
            let plain = verdict(PagesClass::Pooled, access, rows);
            let exportable = verdict(PagesClass::FdBlocks, access, rows);
            match (plain, exportable) {
                (Verdict::Pending, _) | (Verdict::Passed, Verdict::Pending) => pending = true,
                (Verdict::Passed, Verdict::Failed | Verdict::Unchecked) => {
                    return FdBlocksTake::Refused(access, rows)
                }
                (Verdict::Passed, Verdict::Passed) | (Verdict::Failed | Verdict::Unchecked, _) => {}
            }
        }
    }
    match pending {
        true => FdBlocksTake::Pending,
        false => FdBlocksTake::Take,
    }
}

/// The version of what the check does, of the images over host pages it checks as the product
/// makes them (their usage, layout and moves), and of how a record of it is named and framed: a
/// change to any of them bumps it, and a record of another version is never read (its name
/// differs, and so does the version it holds). The runtime's own version is not in the record: a
/// release that changes none of these keeps the verdicts the driver gave, so a device made after
/// it reads them again rather than checking anew.
pub(crate) const CHECK_VERSION: u32 = 16;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VINPL\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan in-place check record\0";

/// What the check checks, in a record's name: the first move of an image over host pages, of one
/// that renders its render passes' operations and feedback loop moves, and of one a shader writes
/// its kernels' storage writes. A check of another operation keeps its own record beside this
/// one's, which it leaves valid.
const CHECK_KIND: &[u8] = b"first move\0";

/// The driver identity a verdict holds for: `VkPhysicalDeviceProperties`' vendor, device and
/// driver version, `VkPhysicalDeviceDriverProperties`' driver ID, name and info, and
/// `VkPhysicalDeviceIDProperties`' device UUID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DriverIdentity {
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) driver_version: u32,
    pub(crate) driver_id: i32,
    pub(crate) driver_name: String,
    pub(crate) driver_info: String,
    pub(crate) device_uuid: [u8; 16],
}

impl DriverIdentity {
    /// The SHA-256 of [`NAME_TAG`], [`CHECK_KIND`], [`CHECK_VERSION`] and every field: what names
    /// a record and what the record holds again. What decides the check's answer alone: the
    /// check, and the driver and device it ran on; not the runtime's version.
    pub(crate) fn digest(&self) -> [u8; 32] {
        self.hash(NAME_TAG, CHECK_KIND, CHECK_VERSION, None)
    }

    /// The SHA-256 of `tag`, `kind`, `version`, the runtime's `RUNTIME_VERSION` and every field:
    /// what names the record of another device check's `kind` at `version` (one whose record is
    /// the runtime release's own), and what that record holds again.
    pub(crate) fn digest_for(&self, tag: &[u8], kind: &[u8], version: u32) -> [u8; 32] {
        let runtime = metal2vulkan::cache::RUNTIME_VERSION;
        self.hash(tag, kind, version, Some(runtime))
    }

    /// The SHA-256 of `tag`, `kind`, `version`, `runtime` where given and every field.
    fn hash(&self, tag: &[u8], kind: &[u8], version: u32, runtime: Option<u32>) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(tag);
        hash.update(kind);
        hash.update(version.to_le_bytes());
        if let Some(runtime) = runtime {
            hash.update(runtime.to_le_bytes());
        }
        hash.update(self.vendor_id.to_le_bytes());
        hash.update(self.device_id.to_le_bytes());
        hash.update(self.driver_version.to_le_bytes());
        hash.update(self.driver_id.to_le_bytes());
        for text in [&self.driver_name, &self.driver_info] {
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.update(self.device_uuid);
        hash.finalize().into()
    }
}

/// The name of the record of `identity`'s verdicts in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&identity.digest())
}

/// The bytes of a record of `verdicts` for `identity`:
///
/// ```text
/// magic          8 bytes   "M2VINPL\0"
/// version        u32       CHECK_VERSION
/// identity       32 bytes  DriverIdentity::digest
/// verdicts       56 bytes  one per class, access and row ends, VERDICTS' order (verdict_index)
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian. A record holds what the check measured, a verdict passed or
/// failed, and what is as lasting for the driver's identity: a verdict unchecked for a cause that
/// lasts (the device lacks an extension, a feature, an import, a format or an image of the
/// shape: `in_place_check.rs`'s `Unmade`). The check gives one unchecked for a cause that may
/// pass in another run (pages or memory not made, an import refused, the writer thread not
/// started, a first move, pass or kernel's stores that failed) as unmeasured, a pending one's
/// byte, so the device made after it checks that class, access and row ends again
/// ([`Device::recall_in_place_checks`]).
pub(crate) fn encode(identity: &DriverIdentity, verdicts: [Verdict; VERDICTS]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&identity.digest());
    bytes.extend(verdicts.map(Verdict::byte));
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// A record's length.
const RECORD_BYTES: usize = 8 + 4 + 32 + VERDICTS + 32;

/// The verdicts `bytes` record for `identity`: `None` for anything but a whole record of this
/// version and runtime, for this identity, whose checksum holds and whose every verdict is one
/// ([`encode`]): passed, failed, unchecked for a cause that lasts, or unmeasured (pending).
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<[Verdict; VERDICTS]> {
    if bytes.len() != RECORD_BYTES {
        return None;
    }
    let (body, checksum) = bytes.split_at(RECORD_BYTES - 32);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC || word(version) != CHECK_VERSION || digest != identity.digest() {
        return None;
    }
    let mut verdicts = [Verdict::Pending; VERDICTS];
    for (verdict, &byte) in verdicts.iter_mut().zip(rest) {
        *verdict = Verdict::of_byte(byte)?;
    }
    Some(verdicts)
}

impl Device {
    /// The identity of the device's driver (`DriverIdentity`), read from its physical device.
    pub(crate) fn driver_identity(&self) -> DriverIdentity {
        let shared = self.shared();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut driver)
            .push_next(&mut ids);
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later, where both
        // structures are core; the chain lives for the call.
        unsafe {
            shared
                .instance
                .raw
                .get_physical_device_properties2(shared.physical, &mut properties)
        };
        let base = properties.properties;
        let text = |chars: &[std::ffi::c_char]| {
            let bytes: Vec<u8> = chars
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        DriverIdentity {
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            driver_version: base.driver_version,
            driver_id: driver.driver_id.as_raw(),
            driver_name: text(&driver.driver_name),
            driver_info: text(&driver.driver_info),
            device_uuid: ids.device_uuid,
        }
    }

    /// As the device is made: where its policy takes textures over host pages in place, its
    /// verdicts from the record of its driver's identity where one holds them (each one it holds
    /// counted as recalled), settling them where none is unmeasured; else they wait for the check,
    /// which the first texture that asks starts. The check then runs only the classes and accesses
    /// whose verdicts are pending: of a record, those it could not check for a cause that may pass
    /// ([`encode`]). A record that does not decode is no record: the check runs, and its verdicts
    /// replace it.
    pub(crate) fn recall_in_place_checks(&self) {
        let shared = self.shared();
        if self.compile_device().policy().host_page_textures != HostPageTexturePolicy::InPlace {
            return;
        }
        if shared.in_place.forced_failure() {
            return;
        }
        let identity = self.driver_identity();
        let name = record_name(&identity);
        let looked = shared
            .shader_cache
            .content_store()
            .load_content(&name, |bytes| decode(&identity, &bytes));
        if let Lookup::Hit(verdicts) = looked {
            for class in PagesClass::ALL {
                for access in Access::ALL {
                    for rows in RowEnds::ALL {
                        let verdict = verdicts[verdict_index(class, access, rows)];
                        if verdict == Verdict::Pending {
                            continue;
                        }
                        shared.in_place.set(class, access, rows, verdict);
                        shared.counts.in_place_check_recalled();
                    }
                }
            }
            if !verdicts.contains(&Verdict::Pending) {
                shared.in_place.settle(None);
            }
        }
    }

    /// `class`'s verdict for `access` of texels that lie in lines as `rows` says, for a texture
    /// asking it as it is made: a pending one starts the check, unless it started already. Never
    /// waits (the module docs).
    pub(crate) fn in_place_verdict(
        &self,
        class: PagesClass,
        access: Access,
        rows: RowEnds,
    ) -> Verdict {
        let verdict = self.shared().in_place.verdict(class, access, rows);
        if verdict == Verdict::Pending {
            self.start_in_place_checks();
        }
        verdict
    }

    /// Whether the runtime's own buffers lie in blocks allocated exportable (`memory.rs`'s
    /// "Buffers in exportable blocks"), by the device's verdicts ([`fd_blocks_take`]), asked as
    /// such a buffer is made: where a verdict is pending the buffer lies in a plain block, as it
    /// does where the verdicts refuse such blocks; each counted (`WorkCounts::fd_blocks_awaited`,
    /// `WorkCounts::fd_blocks_refused`). Never waits, and starts no check: the verdicts come from
    /// the check's record as the device is made, or from a check a texture over host pages or
    /// `Device::settle_in_place_checks` started. A check a buffer's making started would run
    /// beside every program from its first buffer, its own buffers, imports and command buffers
    /// among the program's, where a program that makes no texture over host pages has no use
    /// for it.
    pub(crate) fn fd_blocks_checked(&self) -> bool {
        match self.fd_blocks_now() {
            FdBlocksTake::Take => true,
            FdBlocksTake::Pending => {
                self.shared().counts.fd_blocks_awaited();
                false
            }
            FdBlocksTake::Refused(..) => {
                self.shared().counts.fd_blocks_refused();
                false
            }
        }
    }

    /// Whether the runtime's own buffers lie in blocks allocated exportable now, as
    /// [`Device::fd_blocks_checked`] answers it, counting nothing: for the pools that keep room
    /// (`memory.rs`'s `keep_room`).
    pub(crate) fn fd_blocks_taken(&self) -> bool {
        self.fd_blocks_now() == FdBlocksTake::Take
    }

    /// [`fd_blocks_take`] of the device's verdicts now, starting no check.
    fn fd_blocks_now(&self) -> FdBlocksTake {
        fd_blocks_take(|class, access, rows| self.shared().in_place.verdict(class, access, rows))
    }

    /// Start the device's check of textures over host pages in place where its verdicts are
    /// pending and no check started, and wait until every verdict is known or `timeout` passed:
    /// whether they are. A runtime extension: Metal has no such check, as a Mac's GPU works on a
    /// texture over the caller's pages in place, where the runtime takes an image in place only
    /// where the device's own check passed (docs/API.md, textures over host pages).
    ///
    /// A texture over host pages made while its verdict is pending is a shadow that moves in
    /// place at a later commit where the verdict passes
    /// ([`TexturePlacement::Pending`](crate::TexturePlacement::Pending)); once this answers
    /// `true`, no verdict is pending, and
    /// [`Buffer::texture_placement`](crate::Buffer::texture_placement) answers in place or a
    /// shadow for good, for as long as the device serves. A zero `timeout` waits for nothing: it
    /// starts the check and answers whether the verdicts are known, so a program may start the
    /// check before it makes such textures and ask again later. Under
    /// `HostPageTexturePolicy::Shadow` no check runs and no verdict is known: `false` at once.
    ///
    /// The verdicts go from pending to known once for each `VkDevice` behind the device. A device
    /// of a driver whose check ran before reads them from the record the check kept in the
    /// shader cache's directory as it is made, known at once, where the check ended and wrote it:
    /// the verdicts it measured (passed or failed) and those it could not check for a cause that
    /// lasts for the driver (the device lacks an extension, a feature, an import, a format or an
    /// image of the shape), not those it could not check for a cause that may pass (pages or
    /// memory not made, an import refused, its writer thread not started, a move or the stores
    /// that failed), which stay pending until the new device's check answers them again. A GPU
    /// recovery's new one (`Degradation::DeviceRecovered`) reads the record so too;
    /// where the lost one's check had not ended, or the device keeps no record (no shader cache
    /// directory, or `DeviceOptions::fail_in_place_check`), they are pending again, and this
    /// answers `false` until the new one's check ends. No-copy buffers made before a
    /// recovery are refused on the new device (`RuntimeGap::MadeBeforeRecovery`), so a program
    /// makes its textures over host pages again after one, and asks again.
    pub fn settle_in_place_checks(&self, timeout: Duration) -> bool {
        self.current().settle_in_place_checks_now(timeout)
    }

    /// [`Device::settle_in_place_checks`] on the generation that serves (`device/line.rs`).
    pub(crate) fn settle_in_place_checks_now(&self, timeout: Duration) -> bool {
        if self.compile_device().policy().host_page_textures != HostPageTexturePolicy::InPlace {
            return false;
        }
        let checks = &self.shared().in_place;
        if checks.verdicts().contains(&Verdict::Pending) {
            self.start_in_place_checks();
        }
        checks.wait(timeout)
    }

    /// How long the device's check ran in this process, where it ran and ended: `None` where its
    /// verdicts came from a record, or the check has not ended. A runtime extension for the lanes
    /// that measure it.
    #[doc(hidden)]
    pub fn in_place_check_took(&self) -> Option<Duration> {
        self.current().in_place_check_took_now()
    }

    /// [`Device::in_place_check_took`] on the generation that serves (`device/line.rs`).
    pub(crate) fn in_place_check_took_now(&self) -> Option<Duration> {
        self.shared().in_place.took()
    }

    /// The bytes of the device's line of memory the check's verdicts tell apart
    /// (`nonCoherentAtomSize`, `RowEnds`): those of texels that share a line with bytes beside
    /// them, and those of texels that share none. A runtime extension for the lanes that read
    /// the verdict a texture of theirs takes.
    #[doc(hidden)]
    pub fn in_place_check_line(&self) -> u64 {
        self.current().shared().in_place.line()
    }

    /// The device's check's verdicts now, one line per class, access and row end (`RowEnds`):
    /// first those for texels that share a line with bytes beside them, then those for texels that
    /// share none (`, rows on lines`); within each, the accesses in turn (`Access::ALL`), and for
    /// each access one line per class, the classes always in one order (`PagesClass::ALL`). A
    /// line names its class, then its access but for a texture that reads, then its row end where
    /// the texels share no line: `Pointer: Passed`, `Pointer renders: Passed`, `Pointer writes,
    /// rows on lines: Passed`. Then the check's notes: what it found changed where a verdict failed
    /// (or where only a row's padding changed, `in_place::Padding`), and why one is unchecked. A
    /// runtime extension for the lanes that report it.
    #[doc(hidden)]
    pub fn in_place_check_report(&self) -> Vec<String> {
        self.current().in_place_check_report_now()
    }

    /// [`Device::in_place_check_report`] on the generation that serves (`device/line.rs`).
    pub(crate) fn in_place_check_report_now(&self) -> Vec<String> {
        let checks = &self.shared().in_place;
        let line = |class: PagesClass, access: Access, rows: RowEnds| {
            let verdict = checks.verdict(class, access, rows);
            let lines = match rows {
                RowEnds::InLines => "",
                RowEnds::OnLines => ", rows on lines",
            };
            match access {
                Access::Reads => format!("{class:?}{lines}: {verdict:?}"),
                Access::Renders => format!("{class:?} renders{lines}: {verdict:?}"),
                Access::UnobservedRenders => {
                    format!("{class:?} unobserved renders{lines}: {verdict:?}")
                }
                Access::Writes => format!("{class:?} writes{lines}: {verdict:?}"),
                Access::BufferWrites => format!("{class:?} buffer writes{lines}: {verdict:?}"),
                Access::NarrowWrites => format!("{class:?} narrow writes{lines}: {verdict:?}"),
                Access::NarrowBufferWrites => {
                    format!("{class:?} narrow buffer writes{lines}: {verdict:?}")
                }
            }
        };
        let line = &line;
        let verdicts = RowEnds::ALL.into_iter().flat_map(move |rows| {
            Access::ALL
                .into_iter()
                .flat_map(move |access| PagesClass::ALL.map(|class| line(class, access, rows)))
        });
        verdicts
            .chain(lock(&checks.notes).iter().cloned())
            .collect()
    }
}
