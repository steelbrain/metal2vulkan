//! Textures over host pages, in place: a 2D texture (or array) whose declared usage samples, reads,
//! renders into or writes it, over a no-copy buffer's host pages or over a buffer the runtime
//! placed in fresh host pages, is a linear image aliasing those pages, where the device's own check
//! shows that the operations that may touch bytes beside the texels leave every byte alone.
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
//!   writes it (`ShaderWrite` or `ShaderAtomic`, of texels of 4, 8 or 16 bytes, the widths the
//!   check's kernel stores: [`STORED_TEXEL_BYTES`]), or it is sampled or read alone. Of a texture
//!   a shader writes over a buffer's rows whose padding is free ([`Padding`]: real Macs' GPUs
//!   disagree on those bytes under a kernel's writes, and the owner's ruling of 2026-11-07 takes
//!   the x86 Macs' envelope), the check of writes is the one that lets a byte of the rows'
//!   padding change ([`Access::BufferWrites`]); renders, and an IOSurface plane's rows, keep
//!   every byte. Plain Metal also writes through a texture beyond its declared usage, and renders
//!   into one made without `RenderTarget`: such a write binds the texture's usage shadow
//!   (`texture/usage_shadow.rs`), and such a pass renders through an attachment stand-in
//!   (`attachment_stand_in.rs`), since the image has neither storage nor attachment usage; the
//!   copies of both touch texels alone.
//!   Elsewhere the texture is a shadow (`shadow.rs`), its notice naming why ([`ShadowCause`]) and
//!   counted by cause (`WorkCounts::host_texture_shadows_*`, and those that render apart).
//! - **Before the verdict.** A texture made before its class's verdict for its access is known
//!   waits for nothing: it is a shadow ([`ShadowCause::CheckPending`], counted apart), and the
//!   first such texture starts the check on a thread of its own. Once its class's verdicts are
//!   known, it moves in place where they pass, at the commit of the device's only uncommitted
//!   command buffer (`texture/storage_form.rs`, `WorkCounts::host_textures_moved_in_place`), and
//!   stays a shadow where they do not, as a texture made then would be. Its making waits for no
//!   check: the check's rounds take seconds on some devices and builds (1.8 s for the first
//!   texture, waited for, on a Radeon 780M in a debug build), past what a program's first texture
//!   may take, a guest's message that makes one among them. A device whose record holds the
//!   verdicts knows them as it is made, and starts none; a program that will make such textures
//!   starts the check earlier, where it likes, through `Device::settle_in_place_checks` (with no
//!   wait: a zero timeout).
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
}

impl PagesClass {
    /// Every class, in the order the check runs them and the record holds their verdicts.
    pub(crate) const ALL: [PagesClass; 3] =
        [PagesClass::Pointer, PagesClass::Fresh, PagesClass::DmaBuf];

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            PagesClass::Pointer => 0,
            PagesClass::Fresh => 1,
            PagesClass::DmaBuf => 2,
        }
    }

    /// The class in words, for a notice.
    pub(crate) fn words(self) -> &'static str {
        match self {
            PagesClass::Pointer => "host pages imported by their pointer",
            PagesClass::DmaBuf => "host pages imported as dma-bufs",
            PagesClass::Fresh => "fresh host pages the runtime imported for the buffer",
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
    /// Its usage writes it through a shader (`ShaderWrite` or `ShaderAtomic`, whatever else it
    /// holds): its image takes its first move and kernels' storage writes of its texels.
    Writes,
    /// As [`Writes`](Self::Writes), of a texture whose rows' padding is free ([`Padding::Free`]):
    /// its rounds are those of [`Writes`](Self::Writes), but a byte of its rows' padding (each
    /// row's bytes after its last texel to its pitch, the last row's included) may change. A
    /// usage never takes it alone: [`in_place`] asks it in the stead of
    /// [`Writes`](Self::Writes) for such a texture.
    BufferWrites,
}

/// The bytes of the texels a texture a shader writes may have to be in place: those the check's
/// kernel stores ([`Access::Writes`]; the crate-owned copy-in kernel's unsigned-integer views).
pub(crate) const STORED_TEXEL_BYTES: [u64; 3] = [4, 8, 16];

impl Access {
    /// Every access, in the order the check runs them and the record holds their verdicts.
    pub(crate) const ALL: [Access; 4] = [
        Access::Reads,
        Access::Renders,
        Access::Writes,
        Access::BufferWrites,
    ];

    /// Whether a texture of Metal `usage` (as Metal reports it) takes this access: one that
    /// renders, one a shader writes (both, where it does both), or one that only samples or
    /// reads. Each of its accesses' checks covers its first move. None takes
    /// [`BufferWrites`](Self::BufferWrites) alone ([`in_place`]).
    pub(crate) fn takes(self, usage: TextureUsage) -> bool {
        let renders = usage.contains(TextureUsage::RENDER_TARGET);
        let writes = usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC);
        match self {
            Access::Reads => !renders && !writes,
            Access::Renders => renders,
            Access::Writes => writes,
            Access::BufferWrites => false,
        }
    }

    /// The accesses the check's rounds of this one give verdicts of, in [`ALL`](Self::ALL)'s
    /// order: its own, and of [`Writes`](Self::Writes) those of
    /// [`BufferWrites`](Self::BufferWrites) too, which has no rounds of its own.
    pub(crate) fn answered(self) -> &'static [Access] {
        match self {
            Access::Reads => &[Access::Reads],
            Access::Renders => &[Access::Renders],
            Access::Writes => &[Access::Writes, Access::BufferWrites],
            Access::BufferWrites => &[],
        }
    }

    /// Its place in [`ALL`](Self::ALL).
    pub(crate) fn index(self) -> usize {
        match self {
            Access::Reads => 0,
            Access::Renders => 1,
            Access::Writes => 2,
            Access::BufferWrites => 3,
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
            Access::Writes => {
                "the first move and a kernel's storage writes of the texels of an image a shader \
                 writes"
            }
            Access::BufferWrites => {
                "the first move and a kernel's storage writes of the texels of an image a shader \
                 writes over a buffer's rows, beside their padding"
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
/// buffer may do ([`Access::BufferWrites`]; docs/API.md, section 10, **Kernels**).
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
    /// check's kernel stores none of ([`STORED_TEXEL_BYTES`]): no check covers its writes.
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
}

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
                 device's check of storage writes stores texels of 4, 8 and 16 bytes, not of \
                 {bytes}: no check covers a driver's writes of such texels beside them"
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
        }
    }
}

/// Whether a texture over host pages of `class`, its texels lying in the device's lines as `rows`
/// says, its rows' padding as `padding` says, of Metal `usage` (as Metal reports it) and texels
/// of `texel` bytes, on a device of `policy`, is tried as an image in place; else why it is a
/// shadow (the module docs). `verdict` answers the device's check for the class, each access the
/// texture takes ([`Access::takes`]), in [`Access::ALL`]'s order, and the row ends, asked only of
/// a texture the policy and texel width let through (a pending one starts the check), until one
/// did not pass: for a shader's writes of a texture whose padding is free, the verdict of
/// [`Access::BufferWrites`], which passes wherever that of [`Access::Writes`] does (the same
/// rounds give both, and it lets more change).
pub(crate) fn in_place(
    policy: HostPageTexturePolicy,
    usage: TextureUsage,
    (class, rows, texel, padding): (PagesClass, RowEnds, u64, Padding),
    mut verdict: impl FnMut(PagesClass, Access, RowEnds) -> Verdict,
) -> Result<(), ShadowCause> {
    if policy == HostPageTexturePolicy::Shadow {
        return Err(ShadowCause::Exact);
    }
    if Access::Writes.takes(usage) && !STORED_TEXEL_BYTES.contains(&texel) {
        return Err(ShadowCause::StoreWidth(texel));
    }
    for access in Access::ALL.into_iter().filter(|access| access.takes(usage)) {
        let access = match (access, padding) {
            (Access::Writes, Padding::Free) => Access::BufferWrites,
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

/// The version of what the check does, of the images over host pages it checks as the product
/// makes them (their usage, layout and moves), and of how a record of it is named and framed: a
/// change to any of them bumps it, and a record of another version is never read (its name
/// differs, and so does the version it holds). The runtime's own version is not in the record: a
/// release that changes none of these keeps the verdicts the driver gave, so a device made after
/// it reads them again rather than checking anew.
pub(crate) const CHECK_VERSION: u32 = 8;

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
/// verdicts       24 bytes  one per class, access and row ends, VERDICTS' order (verdict_index)
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
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
/// version and runtime, for this identity, whose checksum holds and whose every verdict is an
/// answer (a pending one is never written).
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
        *verdict = Verdict::of_byte(byte).filter(|&known| known != Verdict::Pending)?;
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
    /// verdicts from the record of its driver's identity where one holds them (counted as
    /// recalled), settling them; else they wait for the check, which the first texture that asks
    /// starts. A record that does not decode is no record: the check runs, and its verdicts
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
                        shared.in_place.set(class, access, rows, verdict);
                        shared.counts.in_place_check_recalled();
                    }
                }
            }
            shared.in_place.settle(None);
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
    /// The verdicts go from pending to known once for each `VkDevice` behind the device. A GPU
    /// recovery's new one (`Degradation::DeviceRecovered`) reads them again from the record the
    /// check kept in the shader cache's directory as it is made, known at once, where the check
    /// ended and wrote it; where the lost one's check had not ended, or the device keeps no record
    /// (no shader cache directory, or `DeviceOptions::fail_in_place_check`), they are pending
    /// again, and this answers `false` until the new one's check ends. Buffers made before a
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

    /// The device's check's verdicts now, one line per class for a texture that reads
    /// (`Pointer: Passed`), then one per class for one that renders (`Pointer renders: Passed`),
    /// then one per class for one a shader writes (`Pointer writes: Passed`), then one per class
    /// for one a shader writes over a buffer's rows whose padding may change (`Pointer buffer
    /// writes: Passed`, `in_place::Padding`), each for texels that share a line with bytes beside
    /// them (`RowEnds::InLines`); then the same twelve for texels that share none (`Pointer, rows
    /// on lines: Passed`, `Pointer renders, rows on lines: Passed`, `Pointer writes, rows on
    /// lines: Passed`, `Pointer buffer writes, rows on lines: Passed`); and what it found changed
    /// where one failed (or where only a row's padding changed). A runtime extension for the
    /// lanes that report it.
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
                Access::Writes => format!("{class:?} writes{lines}: {verdict:?}"),
                Access::BufferWrites => format!("{class:?} buffer writes{lines}: {verdict:?}"),
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
