//! A new resource's first GPU work: an image's move from the layout it was made in to `GENERAL`,
//! where every image of the runtime rests, and the zeros a new resource reads, recorded by the
//! device's next submission rather than by one of its own that the making waits for.
//!
//! - **Why.** Making a texture or a buffer on Metal does no GPU work and waits for none. A waited
//!   internal submission on the device's first `VkQueue` (`Device::submit_internal`) waits for
//!   everything submitted there before it, the program's command buffers among them: a texture
//!   made while a 2 s kernel ran there came back after 2.07 s, where Metal's came back in 102 µs
//!   and 18 µs with its command buffer still scheduled (the consumer's PS ask, an M3 Max). An
//!   encoder that made a scratch buffer for its own commands waited so too.
//! - **What waits.** Every image the runtime makes in memory it binds (its own, a buffer's, a
//!   heap's, a no-copy heap's host pages'), and a stand-in's images
//!   (`stand_in.rs`), starts in `UNDEFINED` (`PREINITIALIZED` for a linear image whose bytes it
//!   keeps) and notes its move ([`FirstMove`]): for one that reads zero (a new `Shared` or
//!   `Managed` texture, a stand-in) with the copies of the device's zeros ([`Zeros`]) that write
//!   every texel. A new texture of every storage mode but `Memoryless` reads zero, a `Private` one
//!   too; its image of a format a clear writes is cleared ([`FirstMove::cleared`]) rather than
//!   copied over, single- or multisampled. A new `Memoryless` one in memory that is not lazily
//!   allocated is cleared to zero in render passes of each level alone
//!   ([`FirstMove::zero_in_passes`]), as no transfer writes a transient attachment. An image that takes a rebind's new `VkImage` notes that
//!   one's (`rebind.rs`). The image (or stand-in) is registered weakly once it is shared
//!   ([`FirstMoves::register`]); an image that copies on the device's staging queue queues its
//!   move and zeros there instead, which every submission of the device's makes there before its
//!   own (`texture/staging_copy.rs`, [`FirstMoves::queue_staging`]). An image over a dma-buf of
//!   its own pages (`dma_buf.rs`) starts `UNDEFINED` too, and its move is an acquire from
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT` for the device's queue family (`VK_QUEUE_FAMILY_IGNORED` where
//!   its images are concurrent among several), from `GENERAL` (which keeps its bytes, as Vulkan's
//!   foreign images keep theirs only through such an acquire) to `GENERAL`
//!   ([`FirstMove::acquired`]); it rests there. A parked one a submission moved, taken by a new
//!   texture over the same pages, notes no move (`dma_buf/parked.rs`).
//!   A new buffer whose bytes the host cannot write (`Private`, and a
//!   `Managed` buffer's device copy) notes its fill with zeros ([`FirstFill`]) the same way
//!   (`Buffer::clear_first`), and a new `Private` buffer made with bytes then the copy of a staging
//!   buffer's (`Buffer::fill_first_from`). So do a sparse texture's image (its move; an emulated
//!   one's zeros too) and its tile map's bytes, and the device's neutrals: a neutral image's move
//!   and its clear ([`FirstMove::cleared`]: it may be multisampled, which no copy writes), and a
//!   neutral texel buffer's fill (`neutral.rs`). An exportable buffer whose memory an external
//!   agent wrote notes its acquire from that agent's reserved queue family
//!   (`Buffer::external_write_done`, [`FirstFill::acquired`]), recorded in the barrier that leads
//!   the fills.
//! - **Zeros a first use writes over.** A new zero-filled `Private` texture's zeros
//!   ([`FirstMove::mark_overwritable`]) stand for what no write left, which a first use that
//!   writes every texel first never shows. A batch whose first use of such an image among its
//!   commands writes it whole before any command reads it, by that encoder's own description
//!   (`encoder.rs`'s `Overwrite`; `command_buffer/submit.rs`'s `first_overwrites`), takes its move
//!   without them ([`FirstMoves::take_holding`]), counted
//!   (`WorkCounts::private_textures_overwritten_unfilled`): the move still moves the image, and a
//!   batch on another `VkQueue` waits for that batch as for any first work, after which the
//!   texels are written. A submission not made gives the zeros back with the move.
//! - **Who records it.** The device's next submission that may reach the resource, before any of
//!   its commands: every batch of the scheduler, whose commands may reach every texture through
//!   its resource ID and every buffer through its address, unnamed (`heap_sets.rs`), and every
//!   internal submission (a host copy among them). It takes the work and holds what it works on
//!   until it completes, so nothing goes while it runs. An image bound to host pages, which a
//!   rebind may swap for another `VkImage` and park the old one with the old pages (`rebind.rs`),
//!   notes those pages with its move ([`FirstMove::new`]), without keeping them while it waits (a
//!   detach lets them go at once): the submission that takes the move holds them, and so the
//!   `VkImage`, until it completed, and drops a move whose pages went with its `VkImage`. A batch
//!   takes such a move whatever its
//!   commands name, so no command reads the image before its move; an internal submission, whose
//!   commands are the runtime's own, takes it only where they name the image's key (a host call
//!   on its texture, whose contract orders the program's GPU work on it). The work is recorded
//!   first in the command buffer that leads the submission's `VkSubmitInfo` -- a batch's own, or
//!   the one that holds the staged copies after it; an internal submission's one command buffer
//!   -- waiting for what the submission waits for: MoltenVK runs each `VkSubmitInfo` as a Metal
//!   command buffer of its own, so it adds none (`command_buffer/submit.rs`). On a device with a
//!   fill queue (a copy engine's second queue, `device/staging.rs`), a batch's buffers' fills of
//!   at least `FILL_QUEUE_LEAST` bytes in all go there instead ([`Taken::take_fills`]), in a
//!   submission of their own after the batches before it on its `VkQueue` and what it waits for,
//!   which the batch waits for before any of its commands; its moves and staged copies stay at
//!   its start, after them. There a batch takes only the fills of memory its commands may reach,
//!   of the device's own buffers, and every fill where it takes a move of an image in memory
//!   another resource owns, or a staged copy ([`FillTake::Reached`]); the rest wait for a batch
//!   that may reach them (`device/staging.rs`). On every device an upload into a buffer the
//!   program's command buffers may have read already (a `Managed` buffer's copy in,
//!   [`FirstMoves::register_upload`]) goes only with a batch whose commands may reach the buffer,
//!   or an internal submission: so it lands after the command buffers before that batch on its
//!   `VkQueue`, never under one of another queue still to read the bytes it replaces.
//! - **Staged copies**. `replace_region` of a texture whose `VkImage` is its own for good (in
//!   memory of its own, or a heap's own pages) and takes no host or staging-queue copies (those
//!   are queued for the staging queue instead, whose submission of them every submission of the
//!   device's makes before its own: `texture/staging_copy.rs`) stages the bytes
//!   ([`TexelCopy`], [`FirstMoves::register_texels`]): the next submission copies them in at
//!   the start of its work, after the first work's `VkSubmitInfo`, each copy after the writes before
//!   it, and holds the image and the staging buffer until it completed. A copy's decode into its
//!   texture's decoded twin (`texture/decoded_twin.rs`, a heap texture's whole decode alone)
//!   has its kernel made before the submission takes its lock (`Device::make_pending_decodes`);
//!   one staged after that walk began is left, with every staged copy after it, for a later
//!   submission, so the copies keep their order ([`FirstMoves::unmade_decodes`]).
//! - **Bounded**. Past [`STAGED_BYTES`] bytes, or [`STAGED_COPIES`] copies, staged for the next
//!   submission (copies into buffers from staging buffers of their own, and staged texel copies),
//!   the call that staged the last makes a submission for them and waits
//!   (`FirstMoves::register_bytes` and `register_texels` answer when). The bound is on the
//!   memory staged: a copy from a host copy the buffer holds whatever the copy (a `Managed`
//!   buffer's upload from its mirror, or from an import of the caller's pages: [`Keeps::HostCopy`])
//!   stages nothing, counts toward neither bound, and makes no submission, so a commit's uploads
//!   of the ranges `did_modify_range` named go with its own batch however many bytes they are.
//! - **Over host pages.** An image bound to imported host memory starts in `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`), and its move "may perform read and write accesses
//!   on all memory bound to the image subresource range": a no-copy heap's texture bound to the
//!   heap's bytes (`texture/heap.rs`) takes such a move. A texture over a no-copy buffer's host
//!   pages (or fresh pages) takes one only where the device's check showed that move keeping every
//!   byte over such pages (`in_place.rs`); elsewhere it is a shadow, whose copies address its texels
//!   alone (`texture.rs`'s `storage_over_buffer`).
//! - **The work.** A barrier from every command before (memory other commands wrote), then the
//!   buffers' fills with zeros (one `vkCmdFillBuffer` each, counted:
//!   `WorkCounts::new_buffer_zero_fills`; copies of the device's zeros for a range whose ends are
//!   not multiples of 4, `FirstFill::zeros`), then their copies of staged bytes
//!   after a barrier (a `Managed` buffer's small upload written inline instead, its bytes read
//!   from the host copy as the work is recorded: `vkCmdUpdateBuffer`s, transfers as the copies
//!   are, [`FirstMoves::register_inline_upload`]), then a
//!   barrier carrying each image's move (after the fills: an image placed in a new heap moves
//!   after the heap's fill, so the fill never overwrites what the move wrote), then the images'
//!   copies of zeros, then a barrier that makes them visible to every later command and to the
//!   host. Work that is a `Managed` buffer's uploads alone (copies or inline writes into its
//!   device copy, which the host never reads) waits for the commands before by a barrier to
//!   transfers alone, and is made visible by one to every later command but not to the host.
//!   A batch's such work on a device that orders its batches by their ends (`batch_order.rs`),
//!   where no internal submission may still run, takes no barrier from the commands before at
//!   all, nor does its GPU start ([`Earlier::Ended`], `WorkCounts::managed_uploads_unbarriered`):
//!   the end of the batch before it on its `VkQueue` already orders every command before it
//!   ahead of the uploads.
//!   Commands after it on the same `VkQueue`, in this submission or a later one, see each image
//!   in `GENERAL` and each fill done. On a device of several `VkQueue`s the batch that did it is
//!   kept with what owns the memory its work wrote ([`Taken::owners`]) until it completed
//!   ([`FirstMoves::waits`]), and a batch on another one waits for it where its commands, or the
//!   first work it took, may reach that memory: memory a resource its encoders' uses name or make
//!   resident owns, every memory where a command of it may reach memory no use names (a pointer
//!   loaded from memory, work recorded on the GPU alone) or where that work's owner is unknown
//!   (the device's own resources), and the memory of its own first work, which comes after that
//!   work's (`command_buffer/submit.rs`, `WorkCounts::first_work_waits_unused` counting the
//!   batches that waited for such work they reach none of). On the RTX 3090 a compositor's
//!   frame of one dispatch, beside an app on another queue that makes two 1024 by 1024 `Private`
//!   textures before each of its command buffers, takes 2.06 ms (p50), as when the app makes
//!   nothing (2.10 ms); where every batch waited for all such work it took 17.1 ms, and so on a
//!   device without a fill queue beside an app making buffers, 2.12 against 16.5 ms
//!   (`tests::fill_waits_bench`). An internal submission waits for every one. Fills made on the
//!   fill queue are waited for
//!   by every later batch whose commands may reach what they filled, and every internal
//!   submission, until the host saw them complete (`Device::fill_queue_wait_for`,
//!   `Device::fill_queue_wait`).
//! - **Order**. Each submission holds its place in the order from its take until its submission was
//!   made, whether or not it took anything: so no submission is made between another's take and its
//!   submission (one that found nothing to take would otherwise run before the moves another took,
//!   on the same `VkQueue`). No texture's elements are written between a batch's take and its
//!   submission: a texture registered after the take, whose move the batch did not take, would be
//!   in the batch's texture table in `UNDEFINED`. So a batch holds the texture slots' writes over
//!   that span ([`FirstMoves::take_holding`], `heap_sets::hold`): elements asked meanwhile are
//!   written once its submission was made, before the next submission takes its place, and a
//!   registration waits for nothing, as Metal's makes wait for nothing. Where registrations waited
//!   for the place instead, a texture made while a batch of 4096 to 32768 fills was submitted on
//!   the M3 Max waited up to 13 ms, in 20 of 60 makings: MoltenVK encodes a command buffer's Metal
//!   commands inside `vkQueueSubmit`, on the calling thread (4.5 to 17 ms for those batches); with
//!   the hold, no making of 60 took past 0.16 ms. A device whose heap sets take no writes while a
//!   command buffer that binds them is pending (no `descriptorBindingUpdateUnusedWhilePending`)
//!   holds no slots, as a write made after a command buffer committed misses the version it binds
//!   (`heap_sets.rs`): there each registration waits for its place ([`FirstMoves::ordered`]), so
//!   for no GPU work, only for the recording of a submission's own barriers and its
//!   `vkQueueSubmit`. An internal submission signals a timeline of its own, which a batch on
//!   another `VkQueue` waits for while one that did first work may not have completed
//!   ([`FirstMoves::internal_wait`]). Seen without it on RADV: a batch submitted with images in
//!   `UNDEFINED` in its texture table, under the validation layer.
//! - **Failure.** A submission that fails gives its work back, still to be done.
//! - **A resource gone.** A buffer's fill and copies and an image's staged copies go as the
//!   resource goes ([`FirstMoves::forget`]: a buffer's drop, an image's release), with the
//!   staging buffer or host copy they hold, each holding the device: left for the next
//!   submission, they would keep a device whose program let go of it alive for good, as no
//!   submission comes.
//! - **Out of reach of a submission.** A texture's `VkImage` handed out (`Texture::vk_image`)
//!   rests in `GENERAL` as its contract says: handing it out does its work first, by a
//!   submission waited for, which makes the fills of the buffer an image in place lies in too.
//!   So does a buffer's `VkBuffer` handed out (`Buffer::vk_buffer`) or its memory exported
//!   (`Buffer::export_memory_fd`) where a fill waits on that `VkBuffer`
//!   ([`FirstMoves::fills_wait`]): the caller's own work may write the bytes at once, and a fill
//!   made after it would wipe them. Each takes its own work alone ([`FirstMoves::take_mine`]),
//!   in its order, the rest waiting for the device's next submission, and waits for no other
//!   work of the device's: its sources are the host's and its destinations new
//!   (`Device::settle_mine`, on the staging queue for a buffer's fills where the device has
//!   one). A texture in a heap's memory, the caller's or a sparse one, takes the device's work
//!   as an internal submission does.

use super::super::creation;
use super::super::device::lock;
use super::super::error::{internal, vulkan, Error, MemoryKind};
use super::super::exit_guard::{self, Call};
use super::super::heap_sets::HeldSlots;
use super::super::host_bytes::HostBytes;
use super::super::image_barrier::{image_barrier, ImageDst};
use super::super::memory::{Memory, MemoryClass, Placement};
use super::super::owned_kernel::FirstUse;
use super::super::release;
use super::super::render_target::{FramebufferObject, RenderPassObject};
use super::super::resource::{ObjectId, ResourceKey};
use super::super::tracker::Point;
use super::super::work_counts::Counters;
use super::super::{Buffer, Device};
use super::decoded_twin::DecodePass;
use super::{buffer_image_copy, image_aspects, transfer_barrier, ImageRegion, ZERO_BYTES};
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::vk::Format;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};
use std::time::Instant;

/// One `VkImage`'s first move: from `old` to `GENERAL`, then, where it reads zero, the copies of
/// the device's zeros that write every texel, or a clear of them.
pub(crate) struct FirstMove {
    pub(super) image: vk::Image,
    aspects: vk::ImageAspectFlags,
    old: vk::ImageLayout,
    zero: Option<Zero>,
    /// The host pages the image is bound to, which a rebind may swap it out of (the module
    /// docs), not kept while the move waits (a detach lets them go at once, as Metal runs their
    /// deallocator then): the submission that takes the move holds them, and so the image, until
    /// it completed (`holding`), and drops a move whose pages went, with which its image went.
    /// `None` for an image bound to none.
    pages: Option<Weak<dyn Any + Send + Sync>>,
    /// The pages, held by the submission that took the move.
    holding: Option<Arc<dyn Any + Send + Sync>>,
    /// Metal's pattern over the image's first levels, which the zeros then leave
    /// ([`FirstMove::patterned`]).
    pattern: Option<Patterned>,
    /// The queue family the move acquires the image for from `VK_QUEUE_FAMILY_FOREIGN_EXT`, an
    /// image over dma-buf pages the CPU writes ([`FirstMove::acquired`]).
    acquire: Option<u32>,
    /// The bytes of a buffer the image is bound to that the move keeps by copies on the device
    /// around its layout transition ([`FirstMove::kept`]).
    keep: Option<GpuKeep>,
    /// Its zeros are a new texture's that a first use writing every texel stands for
    /// ([`FirstMove::mark_overwritable`]).
    overwritable: bool,
    /// The submission that took it records no zeros: its first use writes every texel first
    /// ([`FirstMoves::take_holding`]'s `overwrites`). Undone as it is given back.
    unzeroed: bool,
}

/// A first move's keep of the bytes its image is bound to, by copies on the device
/// ([`FirstMove::kept`]): the image, made `UNDEFINED`, aliases a buffer's bytes that only the
/// device's ordered commands write (`linear_plan::Between::Kept`), and of its move out of
/// `UNDEFINED` the Synchronization chapter says "the contents of that range may be discarded"
/// (Image Layout Transitions), the range its subresources are bound to and no more. So the move
/// copies that span of the buffer into `scratch`, moves the image to `GENERAL`, and copies the
/// span back through the buffer, in one ordered sequence ([`Taken::record`]). The image is a
/// `DRM_FORMAT_MOD_LINEAR` one, or the device's linear image in a block of the runtime's buffers
/// allocated exportable, made for its handle type. In `GENERAL` the image is a host-accessible
/// subresource (either is linear, the Glossary's Linear Resource), as the buffer is, and two such aliases "interpret the contents of
/// the memory in consistent ways, and data written to one alias can be read by the other alias"
/// (Memory Aliasing): the image reads the bytes the buffer held, the bytes between its rows among
/// them, and the buffer holds every byte it held. Its submission waits for every earlier
/// submission on the device's other `VkQueue`s, which may access the span ([`Taken::keeps`]), and
/// a later one there that may reach the buffer's memory waits for it, as for all first work. The
/// image is a buffer's alias, which copies on no staging queue, so no staging queue records it
/// (`record_several`). Counted: `WorkCounts::first_moves_kept`, `first_move_bytes_kept`.
pub(crate) struct GpuKeep {
    /// The buffer, held until the move's submission completed.
    buffer: Buffer,
    /// Where the span starts in the buffer, and its length.
    at: u64,
    length: u64,
    /// The scratch buffer of `length` bytes the span is copied into and back from, held likewise.
    scratch: Buffer,
}

impl GpuKeep {
    /// The keep of `length` bytes of `buffer` from its byte `at`, through `scratch`, a buffer of
    /// the runtime's own of at least `length` bytes (`Device::scratch_buffer`).
    pub(crate) fn new(buffer: Buffer, at: u64, length: u64, scratch: Buffer) -> GpuKeep {
        GpuKeep {
            buffer,
            at,
            length,
            scratch,
        }
    }

    /// The copy of the span out to the scratch buffer, or back with `back`: the source, the
    /// destination and the region.
    fn copy(&self, back: bool) -> (vk::Buffer, vk::Buffer, vk::BufferCopy) {
        let (buffer, at) = self.buffer.at(self.at);
        let (scratch, start) = self.scratch.at(0);
        let (src, dst, region) = match back {
            false => (buffer, scratch, (at, start)),
            true => (scratch, buffer, (start, at)),
        };
        let copy = vk::BufferCopy {
            src_offset: region.0,
            dst_offset: region.1,
            size: self.length,
        };
        (src, dst, copy)
    }

    /// Record the copy out, or back with `back`, into `command_buffer`.
    ///
    /// # Safety
    ///
    /// As [`Taken::record`]'s; both buffers are live (held) and have transfer usage.
    unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer, back: bool) {
        let (src, dst, copy) = self.copy(back);
        // SAFETY: the caller's contract; the span lies within the buffer's `VkBuffer` (the image
        // it is bound to lies within the buffer: `linear_plan::aliases`) and the scratch's.
        unsafe { device.cmd_copy_buffer(command_buffer, src, dst, &[copy]) };
        exit_guard::called(Call::CmdCopyBuffer);
    }
}

/// How a first move writes Metal's pattern into the first `levels` levels of its image, of
/// `total` (`heap_tiled::fresh_pattern`): every aspect and layer of them.
pub(crate) struct Patterned {
    pattern: Pattern,
    levels: u32,
    total: u32,
}

/// What writes the pattern.
pub(crate) enum Pattern {
    /// A clear to the color value that writes it exactly.
    Color(vk::ClearColorValue),
    /// A clear to the depth and stencil that write it exactly.
    DepthStencil(vk::ClearDepthStencilValue),
    /// Copies of the device's buffer of the pattern's texel ([`FirstMoves::pattern`]).
    Copies(vk::Buffer, Vec<vk::BufferImageCopy>),
    /// Copies of the device's multisampled scratch image of the texel's size
    /// ([`FirstMoves::scratch`]), of an unsigned-integer format, cleared to the texel's bits
    /// first: a multisampled image's pattern whose bits no clear of its own format writes (a
    /// NaN), which no buffer copy reaches.
    Scratch {
        image: vk::Image,
        bits: vk::ClearColorValue,
        copies: Vec<vk::ImageCopy>,
    },
}

impl Patterned {
    /// `pattern` over the first `levels` of the image's `total` levels, `levels` at least 1.
    pub(crate) fn new(pattern: Pattern, levels: u32, total: u32) -> Patterned {
        Patterned {
            pattern,
            levels,
            total,
        }
    }

    /// The pattern's copies, from the copies `copies` that write every texel
    /// (`texture::zero_copies`) of a buffer of the pattern.
    pub(crate) fn copies(pattern: vk::Buffer, copies: &[ImageRegion]) -> Pattern {
        Pattern::Copies(pattern, copies.iter().map(buffer_image_copy).collect())
    }

    /// The pattern of the scratch image `image` ([`FirstMoves::scratch`]) cleared to `bits`
    /// (`clear_value::uint_texel`), copied in tiles of [`SCRATCH_SIDE`] over the first level, `extent`, of
    /// each of an image's `layers`.
    pub(crate) fn scratch(
        image: vk::Image,
        bits: vk::ClearColorValue,
        [width, height]: [u32; 2],
        layers: u32,
    ) -> Pattern {
        let subresource = |layer| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        };
        let tiles = |side: u32| {
            (0..side.div_ceil(SCRATCH_SIDE)).map(move |tile| {
                let at = tile * SCRATCH_SIDE;
                (at, (side - at).min(SCRATCH_SIDE))
            })
        };
        let copies = (0..layers)
            .flat_map(|layer| {
                tiles(height).flat_map(move |(y, rows)| {
                    tiles(width).map(move |(x, columns)| vk::ImageCopy {
                        src_subresource: subresource(0),
                        src_offset: vk::Offset3D::default(),
                        dst_subresource: subresource(layer),
                        dst_offset: vk::Offset3D {
                            x: x as i32,
                            y: y as i32,
                            z: 0,
                        },
                        extent: vk::Extent3D {
                            width: columns,
                            height: rows,
                            depth: 1,
                        },
                    })
                })
            })
            .collect();
        Pattern::Scratch {
            image,
            bits,
            copies,
        }
    }
}

impl FirstMove {
    /// The move of `image`, of `format`, from `old`; `pages`, the host pages it is bound to, where
    /// it is bound to some.
    pub(crate) fn new(
        image: vk::Image,
        format: Format,
        old: vk::ImageLayout,
        pages: Option<Weak<dyn Any + Send + Sync>>,
    ) -> FirstMove {
        FirstMove {
            image,
            aspects: image_aspects(format).into(),
            old,
            zero: None,
            pages,
            holding: None,
            pattern: None,
            acquire: None,
            keep: None,
            overwritable: false,
            unzeroed: false,
        }
    }

    /// Whether the image is bound to host pages, which a rebind may swap it out of (the module
    /// docs).
    pub(super) fn rebindable(&self) -> bool {
        self.pages.is_some()
    }

    /// The move, taken by a submission: holding the host pages its image is bound to, where it
    /// is bound to some; `None` where they went, and its image with them.
    fn held(mut self) -> Option<FirstMove> {
        if let Some(pages) = &self.pages {
            self.holding = Some(pages.upgrade()?);
        }
        Some(self)
    }

    /// The move, given back by a submission not made: no longer holding its pages, its zeros
    /// still to be written.
    fn unheld(mut self) -> FirstMove {
        self.holding = None;
        self.unzeroed = false;
        self
    }

    /// The move, an acquire of the image from `VK_QUEUE_FAMILY_FOREIGN_EXT` for `family` (the
    /// device's one queue family, or `VK_QUEUE_FAMILY_IGNORED` for an image concurrent among
    /// several, `sharing.rs`): an image over dma-buf pages (`dma_buf.rs`), made `UNDEFINED` as
    /// Vulkan makes an image of external memory, whose bytes the CPU wrote outside Vulkan. It
    /// moves from `GENERAL`, the layout of the bytes as they lie, which an acquire keeps; a move
    /// from `UNDEFINED` may discard them, and so may a queue family that takes an exclusive
    /// image without one.
    pub(crate) fn acquired(mut self, family: u32) -> FirstMove {
        self.old = vk::ImageLayout::GENERAL;
        self.acquire = Some(family);
        self
    }

    /// The move, from `UNDEFINED`, keeping the bytes `keep` names by copies around it ([`GpuKeep`]).
    pub(crate) fn kept(mut self, keep: GpuKeep) -> FirstMove {
        self.keep = Some(keep);
        self
    }

    /// The move, which also writes every texel `copies` reach (`zero_copies`) with zero bytes
    /// from `zeros`.
    pub(crate) fn zeroed(mut self, zeros: vk::Buffer, copies: &[ImageRegion]) -> FirstMove {
        let copies = copies.iter().map(buffer_image_copy).collect();
        self.zero = Some(Zero::Copies(zeros, copies));
        self
    }

    /// The move, which also clears every texel of every aspect, level and layer to zero: for an
    /// image no copy writes, a multisampled one (a neutral image, `neutral.rs`), and a new
    /// texture's image of a format a clear writes (`Device::clears_to_zero`).
    pub(crate) fn cleared(mut self) -> FirstMove {
        self.zero = Some(Zero::Clear);
        self
    }

    /// Have the move also clear every texel of every aspect to zero in `passes`, which cover every
    /// level and layer a pass may attach: a new memoryless texture's image, a transient
    /// attachment, which no transfer writes (`Device::new_texture_image`).
    pub(crate) fn zero_in_passes(&mut self, passes: Vec<PassClear>) {
        self.zero = Some(Zero::Passes(passes));
    }

    /// Whether its zeros are written in render passes ([`zero_in_passes`](Self::zero_in_passes)):
    /// the barrier after them is from the attachments' stages.
    fn clears_in_passes(&self) -> bool {
        matches!(self.zero, Some(Zero::Passes(_)))
    }

    /// The move, which also writes `patterned`'s pattern into its levels; its zeros then write
    /// only the levels after them. Its zeros' copies, if any, cover every level from the first.
    pub(crate) fn patterned(mut self, patterned: Patterned) -> FirstMove {
        self.pattern = Some(patterned);
        self
    }

    /// Let a submission leave the move's zeros (a new `Private` texture's, which reads zero where
    /// Metal's reads what no write left, `Device::new_texture_image`) out where the first
    /// use of the image among its commands writes every texel before any command reads one
    /// ([`FirstMoves::take_holding`]).
    pub(crate) fn mark_overwritable(&mut self) {
        self.overwritable = matches!(self.zero, Some(Zero::Copies(..) | Zero::Clear))
            && self.pattern.is_none()
            && self.keep.is_none()
            && self.acquire.is_none();
    }

    /// Whether a submission may leave its zeros out ([`mark_overwritable`](Self::mark_overwritable)).
    pub(crate) fn may_go_unzeroed(&self) -> bool {
        self.overwritable
    }

    /// Whether it writes anything into its image: zeros or a pattern.
    fn writes(&self) -> bool {
        (self.zero.is_some() && !self.unzeroed) || self.pattern.is_some()
    }

    /// Levels `levels` of every aspect and layer of the image.
    fn levels(&self, levels: std::ops::Range<u32>) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            base_mip_level: levels.start,
            level_count: levels.end - levels.start,
            ..self.whole()
        }
    }

    /// Every aspect, level and layer of the image.
    fn whole(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspects,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }

    /// Record its writes of zero into `command_buffer`, of a family of `family`'s queue flags,
    /// the image in `GENERAL`.
    ///
    /// # Safety
    ///
    /// As [`Taken::record`]'s; the image is live, in `GENERAL`, has `TRANSFER_DST` usage, and
    /// its format the transfer destination feature.
    unsafe fn record_zero(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        family: vk::QueueFlags,
    ) {
        let general = vk::ImageLayout::GENERAL;
        // The zeros and the pattern write disjoint levels, so neither waits for the other.
        let (first, range) = match &self.pattern {
            None => (0, self.whole()),
            Some(patterned) => (
                patterned.levels,
                self.levels(patterned.levels..patterned.total),
            ),
        };
        let past_pattern = (self.pattern.as_ref()).is_none_or(|p| p.levels < p.total);
        match &self.zero {
            None => {}
            Some(_) if self.unzeroed => {}
            Some(Zero::Copies(zeros, copies)) => {
                let copies: Vec<vk::BufferImageCopy> = (copies.iter())
                    .filter(|copy| copy.image_subresource.mip_level >= first)
                    .copied()
                    .collect();
                if !copies.is_empty() {
                    // SAFETY: the caller's; every copy stays within the zeros
                    // (`FirstMoves::zeros`).
                    unsafe {
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            *zeros,
                            self.image,
                            general,
                            &copies,
                        );
                    }
                    exit_guard::called(Call::CmdCopyBufferToImage);
                }
            }
            Some(Zero::Passes(passes)) => {
                let attachments = [vk::ClearAttachment {
                    aspect_mask: self.aspects,
                    color_attachment: 0,
                    clear_value: vk::ClearValue::default(),
                }];
                for clear in passes {
                    let [width, height] = clear.extent;
                    let area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D { width, height },
                    };
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(clear.pass.raw)
                        .framebuffer(clear.framebuffer.raw)
                        .render_area(area);
                    let rects = [vk::ClearRect {
                        rect: area,
                        base_array_layer: 0,
                        layer_count: clear.layers,
                    }];
                    // SAFETY: the caller's; the command buffer is outside a render pass, of a
                    // family with graphics (every queue but the staging queue's, which records no
                    // such move: a memoryless image copies nothing there); the pass and its
                    // framebuffer, of a view of the image's level and layers in `GENERAL`, the
                    // pass's layout, live as long as the move, which the submission holds until
                    // it completed; the aspects are the image format's, the subpass's one
                    // attachment's; the rectangle is the framebuffer's area and its layers.
                    unsafe {
                        device.cmd_begin_render_pass(
                            command_buffer,
                            &begin,
                            vk::SubpassContents::INLINE,
                        );
                        device.cmd_clear_attachments(command_buffer, &attachments, &rects);
                        device.cmd_end_render_pass(command_buffer);
                    }
                    exit_guard::called(Call::CmdBeginRenderPass);
                    exit_guard::called(Call::CmdClearAttachments);
                    exit_guard::called(Call::CmdEndRenderPass);
                }
            }
            Some(Zero::Clear) if !past_pattern => {}
            Some(Zero::Clear) if self.aspects.contains(vk::ImageAspectFlags::COLOR) => {
                let black = vk::ClearColorValue::default();
                // SAFETY: the caller's; the range is the image's own.
                unsafe {
                    device.cmd_clear_color_image(
                        command_buffer,
                        self.image,
                        general,
                        &black,
                        &[range],
                    );
                }
                exit_guard::called(Call::CmdClearColorImage);
            }
            Some(Zero::Clear) => {
                let zero = vk::ClearDepthStencilValue::default();
                // SAFETY: as above; the image's aspects are depth and stencil ones.
                unsafe {
                    device.cmd_clear_depth_stencil_image(
                        command_buffer,
                        self.image,
                        general,
                        &zero,
                        &[range],
                    );
                }
                exit_guard::called(Call::CmdClearDepthStencilImage);
            }
        }
        let Some(patterned) = &self.pattern else {
            return;
        };
        let range = self.levels(0..patterned.levels);
        match &patterned.pattern {
            Pattern::Color(value) => {
                // SAFETY: the caller's; the range is the image's own, a color one.
                unsafe {
                    device.cmd_clear_color_image(
                        command_buffer,
                        self.image,
                        general,
                        value,
                        &[range],
                    );
                }
                exit_guard::called(Call::CmdClearColorImage);
            }
            Pattern::DepthStencil(value) => {
                // SAFETY: as above; the image's aspects are depth and stencil ones.
                unsafe {
                    device.cmd_clear_depth_stencil_image(
                        command_buffer,
                        self.image,
                        general,
                        value,
                        &[range],
                    );
                }
                exit_guard::called(Call::CmdClearDepthStencilImage);
            }
            Pattern::Copies(source, copies) => {
                // SAFETY: the caller's; every copy stays within the pattern's buffer
                // (`FirstMoves::pattern`) and writes the pattern's levels.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        *source,
                        self.image,
                        general,
                        copies,
                    );
                }
                exit_guard::called(Call::CmdCopyBufferToImage);
            }
            Pattern::Scratch {
                image,
                bits,
                copies,
            } => {
                // The scratch is cleared again at each first move that copies it (its contents
                // discarded, after the copies before it in the batch), then read.
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let barrier = |old, source, target| {
                    image_barrier(*image, range, (old, general), source, target, family)
                };
                let transfer = vk::PipelineStageFlags::TRANSFER;
                let flags = vk::DependencyFlags::empty();
                let cleared = barrier(
                    vk::ImageLayout::UNDEFINED,
                    vk::AccessFlags::TRANSFER_READ,
                    ImageDst::TransferWrite,
                );
                let copied = barrier(
                    general,
                    vk::AccessFlags::TRANSFER_WRITE,
                    ImageDst::TransferRead,
                );
                let pipeline_barrier = |barrier: vk::ImageMemoryBarrier<'_>| {
                    // SAFETY: the caller's; the barrier is of the scratch, the device's and live,
                    // whose whole range it names.
                    unsafe {
                        device.cmd_pipeline_barrier(
                            command_buffer,
                            transfer,
                            transfer,
                            flags,
                            &[],
                            &[],
                            &[barrier],
                        );
                    }
                    exit_guard::called(Call::CmdPipelineBarrier);
                };
                pipeline_barrier(cleared);
                // SAFETY: the caller's; the scratch is in `GENERAL` (above), of an
                // unsigned-integer color format, with `TRANSFER_DST` usage.
                unsafe {
                    device.cmd_clear_color_image(command_buffer, *image, general, bits, &[range]);
                }
                exit_guard::called(Call::CmdClearColorImage);
                pipeline_barrier(copied);
                // SAFETY: the caller's; the scratch (`TRANSFER_SRC`) and the image are in
                // `GENERAL`, of the same samples and texel size (`FirstMoves::scratch`), and every
                // copy lies within the scratch and the image's first level.
                unsafe {
                    device.cmd_copy_image(
                        command_buffer,
                        *image,
                        general,
                        self.image,
                        general,
                        copies,
                    );
                }
                exit_guard::called(Call::CmdCopyImage);
            }
        }
    }

    /// Record `moves` into `command_buffer`, of a family of `family`'s queue flags, on the staging
    /// queue (`texture/staging_copy.rs`): one barrier carrying every move, after every command
    /// before, then each one's writes of zero; the barrier that makes them visible after is the
    /// caller's.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, for a queue family each image is
    /// shared with, of `family`'s flags; each image is live, in the layout its move starts from,
    /// and nothing else uses it until the submission completed.
    pub(super) unsafe fn record_several(
        moves: &[&FirstMove],
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        family: vk::QueueFlags,
    ) {
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        let barriers: Vec<vk::ImageMemoryBarrier<'_>> =
            moves.iter().map(|first| first.barrier(family)).collect();
        // SAFETY: the caller's contract; each image is live, in the layout its move starts from.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        exit_guard::called(Call::CmdPipelineBarrier);
        for first in moves.iter().filter(|first| first.writes()) {
            // SAFETY: as above; each image is in `GENERAL` now and has `TRANSFER_DST` usage
            // where it reads zero.
            unsafe { first.record_zero(device, command_buffer, family) };
        }
    }

    /// The barrier of the move, recorded for a family of `family`'s queue flags, before every
    /// access of the device's there (`image_barrier.rs`): an acquire from
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` where it is one.
    pub(crate) fn barrier(&self, family: vk::QueueFlags) -> vk::ImageMemoryBarrier<'static> {
        let (source, target) = match self.acquire {
            Some(family) => (vk::QUEUE_FAMILY_FOREIGN_EXT, family),
            None => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
        };
        let general = vk::ImageLayout::GENERAL;
        let write = vk::AccessFlags::MEMORY_WRITE;
        image_barrier(
            self.image,
            self.whole(),
            (self.old, general),
            write,
            ImageDst::Every,
            family,
        )
        .src_queue_family_index(source)
        .dst_queue_family_index(target)
    }
}

/// How a first move writes zero into every texel of its image.
enum Zero {
    /// Copies of the device's zeros ([`Zeros`]).
    Copies(vk::Buffer, Vec<vk::BufferImageCopy>),
    /// A clear ([`FirstMove::cleared`]).
    Clear,
    /// Clears in render passes ([`FirstMove::zero_in_passes`]).
    Passes(Vec<PassClear>),
}

/// A clear of one level of an image, over `extent` and `layers` layers from its first, in a
/// render pass of the attachment alone and its framebuffer of a view of them
/// (`render_target::clear_alone`), which the move holds ([`FirstMove::zero_in_passes`]).
pub(crate) struct PassClear {
    pub(crate) pass: RenderPassObject,
    pub(crate) framebuffer: FramebufferObject,
    pub(crate) extent: [u32; 2],
    pub(crate) layers: u32,
}

/// The most bytes host calls' copies stage for the device's next submission before one is made
/// for them, waited for (the module docs).
pub(super) const STAGED_BYTES: u64 = 256 << 20;

/// The most copies host calls stage for the device's next submission before one is made for
/// them, waited for (the module docs): a bound on the work waiting, whatever its bytes.
pub(super) const STAGED_COPIES: usize = 4096;

/// What a copy of bytes into a buffer keeps until the submission that makes it completed
/// ([`FirstMoves::register_bytes`], [`FirstMoves::register_upload`]).
pub(crate) enum Keeps {
    /// A staging buffer of its own, which holds the bytes: memory that waits on the submission,
    /// which the bound counts ([`STAGED_BYTES`], [`STAGED_COPIES`]).
    Staging(Box<dyn Any + Send + Sync>),
    /// The host copy it reads, which the buffer holds whatever the copy (a `Managed` buffer's
    /// mirror, an import of the caller's pages): it stages nothing, and the bound counts none of
    /// it.
    HostCopy(Box<dyn Any + Send + Sync>),
}

/// A new buffer's fill: copies of the device's zeros onto `range` of `buffer`, or a fill of its
/// words with zeros (`word_fill.rs`), or copies of the bytes a staging buffer holds, or a
/// `Managed` buffer's upload from its host copy; or an exportable buffer's acquire from the
/// external agent that wrote its memory ([`FirstFill::acquired`]), which copies nothing.
pub(crate) struct FirstFill {
    buffer: vk::Buffer,
    source: vk::Buffer,
    copies: Vec<vk::BufferCopy>,
    /// The words one `vkCmdFillBuffer` sets to zero in place of the copies (none then)
    /// ([`FirstFill::zeros`]).
    filled: Option<std::ops::Range<u64>>,
    /// What keeps the staging buffer or host copy `source` is, held until the submission that
    /// makes the fill completed; `None` where `source` is the device's zeros.
    staging: Option<Box<dyn Any + Send + Sync>>,
    /// Whether `source` is a staging buffer of the fill's own ([`Keeps::Staging`]), whose bytes
    /// count toward the bound ([`past_bound`]); false for the device's zeros and a host copy.
    staged: bool,
    /// The queue families of an acquire of the whole `VkBuffer` (source, a family reserved for
    /// external memory; target), recorded in the barrier before every fill, in place of copies
    /// (none then).
    acquire: Option<(u32, u32)>,
    /// The key the program's uses of the memory it writes name ([`Filled::key`]), as it was
    /// registered.
    key: Option<ResourceKey>,
    /// What owns that memory (`used_resources::owner_of`), as the batch that took it last found
    /// it ([`FillTake::Reached`]); `None` where none did, or none owns it.
    owner: Option<ObjectId>,
    /// An upload into a buffer the program's command buffers may have read already
    /// ([`FirstMoves::register_upload`]), with the stamp of the commit that took its bytes
    /// (`managed.rs`'s `Modified`; 0 where none did): a batch takes it only where its commands
    /// may reach the memory and its command buffer was committed at or after that commit, so it
    /// lands on that batch's `VkQueue`, after the command buffers before it there, and under no
    /// command buffer committed before the bytes were taken; never because the batch takes every
    /// fill ([`Taken::every_fill`]). An internal submission's take ([`FillTake::All`]) takes it
    /// too. `None` for every other fill.
    upload: Option<u64>,
    /// Where an upload is written inline ([`FirstMoves::register_inline_upload`]): the host
    /// address of its first byte, read as the submission that takes it is recorded and written by
    /// `vkCmdUpdateBuffer`s in that submission's command stream, its one copy in `copies` naming
    /// where and how many (its `source` and `src_offset` unused). `None` for every other fill.
    inline: Option<HostBytes<'static>>,
}

/// The most bytes one `vkCmdUpdateBuffer` writes (`VUID-vkCmdUpdateBuffer-dataSize-00037`).
const UPDATE_BYTES: usize = 65536;

impl FirstFill {
    /// The fill, with the key of `keeper`'s memory ([`Filled::key`]).
    fn keyed(mut self, keeper: &Weak<dyn Filled>) -> FirstFill {
        self.key = keeper.upgrade().and_then(|filled| filled.key());
        self
    }

    /// The fill of `length` bytes of `buffer` from byte `start`, from `zeros`, the device's.
    fn new(buffer: vk::Buffer, start: u64, length: u64, zeros: vk::Buffer) -> FirstFill {
        let copies = (0..length.div_ceil(ZERO_BYTES))
            .map(|chunk| {
                let at = chunk * ZERO_BYTES;
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: start + at,
                    size: ZERO_BYTES.min(length - at),
                }
            })
            .collect();
        FirstFill {
            buffer,
            source: zeros,
            copies,
            filled: None,
            staging: None,
            staged: false,
            acquire: None,
            key: None,
            owner: None,
            upload: None,
            inline: None,
        }
    }

    /// The zeros of `range` of `buffer` on `device`: one `vkCmdFillBuffer` of its words
    /// (`word_fill.rs`), where the range's ends are multiples of 4; else copies of the device's
    /// zeros, made now if they are not yet. A copy reads the zeros, host-visible, as well as it
    /// writes the range: on an RX 7900 XTX (RADV) 512 KiB of them took 66 us to copy and 11 us
    /// more in the barrier after it, where a fill of the same bytes took 0.7 us and 2.5 us. On a
    /// device that shares the host's memory a fill costs no more: zeroing 17.9 MB in one blit took
    /// 0.53 ms of GPU time as a fill and 0.75 ms as copies on a Radeon 780M (RADV), and 0.52 ms and
    /// 0.53 ms on an Iris Xe (ANV), medians of 20.
    fn zeros(
        device: &Device,
        moves: &FirstMoves,
        buffer: vk::Buffer,
        range: std::ops::Range<u64>,
    ) -> Result<FirstFill, Error> {
        let words = range.start.is_multiple_of(4) && range.end.is_multiple_of(4);
        if words {
            return Ok(FirstFill {
                buffer,
                source: vk::Buffer::null(),
                copies: Vec::new(),
                filled: Some(range),
                staging: None,
                staged: false,
                acquire: None,
                key: None,
                owner: None,
                upload: None,
                inline: None,
            });
        }
        let zeros = moves.zeros(device, 0)?;
        Ok(FirstFill::new(
            buffer,
            range.start,
            range.end - range.start,
            zeros,
        ))
    }

    /// The copy of `length` bytes of `source` from byte `from` onto `buffer` from byte `start`,
    /// `keeps` keeping `source`.
    fn bytes(
        buffer: vk::Buffer,
        start: u64,
        (source, from): (vk::Buffer, u64),
        length: u64,
        keeps: Keeps,
    ) -> FirstFill {
        let (staging, staged) = match keeps {
            Keeps::Staging(staging) => (staging, true),
            Keeps::HostCopy(copy) => (copy, false),
        };
        FirstFill {
            buffer,
            source,
            copies: vec![vk::BufferCopy {
                src_offset: from,
                dst_offset: start,
                size: length,
            }],
            filled: None,
            staging: Some(staging),
            staged,
            acquire: None,
            key: None,
            owner: None,
            upload: None,
            inline: None,
        }
    }

    /// The acquire of `buffer`, whole, from `source`, a queue family reserved for external memory
    /// (`VK_QUEUE_FAMILY_EXTERNAL`, `VK_QUEUE_FAMILY_FOREIGN_EXT`), for `target`: the device's
    /// queue family, or `VK_QUEUE_FAMILY_IGNORED` where its buffers are concurrent among several.
    #[cfg(unix)]
    fn acquired(buffer: vk::Buffer, (source, target): (u32, u32)) -> FirstFill {
        FirstFill {
            buffer,
            source: vk::Buffer::null(),
            copies: Vec::new(),
            filled: None,
            staging: None,
            staged: false,
            acquire: Some((source, target)),
            key: None,
            owner: None,
            upload: None,
            inline: None,
        }
    }

    /// The bytes it writes: its copies' and its zeros'.
    fn written(&self) -> u64 {
        let copied: u64 = self.copies.iter().map(|copy| copy.size).sum();
        copied + (self.filled.as_ref()).map_or(0, |range| range.end - range.start)
    }

    /// The barrier of its acquire, where it is one: no access of the device's before it (an
    /// acquire's source access is ignored), every access after.
    fn acquire_barrier(&self) -> Option<vk::BufferMemoryBarrier<'static>> {
        let (source, target) = self.acquire?;
        Some(
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(source)
                .dst_queue_family_index(target)
                .buffer(self.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
        )
    }
}

/// A host call's copy of texels into an image, staged: `replace_region`'s
/// where the image takes no host copies (`Texture::replace_region`).
pub(crate) struct TexelCopy {
    source: vk::Buffer,
    /// Each image and its copy from `source`.
    copies: Vec<(vk::Image, vk::BufferImageCopy)>,
    /// What keeps `source`, held until the submission that makes the copy completed.
    staging: Box<dyn Any + Send + Sync>,
    /// The bytes `source` holds.
    bytes: u64,
    /// The decode of what the copies wrote into the image's decoded twin, after them
    /// (`texture/decoded_twin.rs`).
    decode: Option<DecodePass>,
    /// The hazard-tracking key the program's uses of the image name ([`Fresh::key`]).
    key: Option<ResourceKey>,
}

impl TexelCopy {
    /// The copies of `source`, `bytes` long, which `staging` keeps, into each image.
    pub(crate) fn new(
        source: vk::Buffer,
        bytes: u64,
        copies: Vec<(vk::Image, vk::BufferImageCopy)>,
        staging: Box<dyn Any + Send + Sync>,
    ) -> TexelCopy {
        TexelCopy {
            source,
            copies,
            staging,
            bytes,
            decode: None,
            key: None,
        }
    }

    /// The copies, into an image whose uses name `key` ([`Fresh::key`]).
    pub(crate) fn keyed(mut self, key: ResourceKey) -> TexelCopy {
        self.key = Some(key);
        self
    }

    /// The copies, then `pass`, their decode into the image's decoded twin (its scratch counted
    /// among the staged bytes).
    pub(crate) fn decoding(mut self, pass: DecodePass) -> TexelCopy {
        self.bytes += pass.bytes;
        self.decode = Some(pass);
        self
    }

    /// `pass` alone, the decode into a decoded twin of what is in its carrier already: a heap's
    /// bytes. (no longer a host call's copy's, which the CPU decodes.)
    pub(crate) fn decode_only(pass: DecodePass) -> TexelCopy {
        TexelCopy::new(vk::Buffer::null(), 0, Vec::new(), Box::new(())).decoding(pass)
    }
}

/// What holds first moves not yet made: an image of the runtime's, or a stand-in.
pub(crate) trait Fresh: Send + Sync {
    /// Its moves not yet taken, taken now.
    fn take(&self) -> Vec<FirstMove>;
    /// `moves`, which [`take`](Fresh::take) gave and no submission made, given back.
    fn give_back(&self, moves: Vec<FirstMove>);
    /// The key the commands that use it name, where a move of it waiting is of an image bound to
    /// host pages (the module docs): an internal submission takes that only where it names the
    /// key. `None` for every other.
    fn rebindable(&self) -> Option<ResourceKey>;
    /// Whether its image lies in memory another resource owns (a buffer's, a heap's), which that
    /// resource's fills write: a batch that takes its move takes every fill waiting (the module
    /// docs).
    fn in_other_memory(&self) -> bool {
        false
    }
    /// The hazard-tracking key the program's uses of its image name, whose owner
    /// (`used_resources::owner_of`) owns the memory its moves write. `None` for an image of the
    /// device's own, which no use of the program's names.
    fn key(&self) -> Option<ResourceKey> {
        None
    }
}

/// A new buffer whose fill waits: what a submission that takes the fill holds until it completes
/// is its memory alone (`Buffer::memory_hold`), so the buffer goes at the program's release, as
/// Metal frees it, its range of the address space and its range token with it.
pub(crate) trait Filled: Send + Sync {
    /// A hold on the memory the buffer lies in, kept and never read; `None` where it lies in
    /// none.
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>>;

    /// The hazard-tracking key the program's uses of the memory the fill writes name: the
    /// buffer's, or for a buffer placed in a heap the heap's own buffer's. `None` for a buffer
    /// of the device's own (a neutral), which no use of the program's names.
    fn key(&self) -> Option<ResourceKey> {
        None
    }
}

/// A resource whose first work waits.
enum Waiting {
    /// An image or a stand-in, which holds its moves.
    Moves(Weak<dyn Fresh>),
    /// A buffer and its fill.
    Fill(Weak<dyn Filled>, FirstFill),
    /// An image, which a submission holds while it copies, and a host call's copy into it.
    Texels(Weak<dyn Any + Send + Sync>, TexelCopy),
}

/// The fills waiting that a submission takes ([`FirstMoves::take_holding`]).
pub(crate) enum FillTake<'r> {
    /// Every one, what owns the memory of what it takes unknown: an internal submission's, a
    /// handle's.
    All,
    /// Every one, knowing what owns the memory of what it takes by `owner_of` (the owner under a
    /// key, `used_resources::owner_of`), but an upload whose memory its commands do not reach, as
    /// `reaches` answers for what owns it, or of a commit after `commit`, its command buffer's
    /// (`FirstFill::upload`): a batch's on a device without a fill queue.
    Owned {
        owner_of: &'r dyn Fn(ResourceKey) -> Option<ObjectId>,
        reaches: &'r dyn Fn(ObjectId) -> bool,
        commit: u64,
    },
    /// A batch's on a device with a fill queue (the module docs): the fills of memory its commands
    /// may reach, as `reaches` answers for what owns it (`owner_of` naming the owner under a
    /// key, `used_resources::owner_of`); every fill of memory no use of the program's names; and
    /// every fill where it takes a move of an image in memory another resource owns, or a staged
    /// copy ([`Taken::every_fill`]); but an upload of a commit after `commit`, its command
    /// buffer's (`FirstFill::upload`).
    Reached {
        owner_of: &'r dyn Fn(ResourceKey) -> Option<ObjectId>,
        reaches: &'r dyn Fn(ObjectId) -> bool,
        commit: u64,
    },
}

impl FillTake<'_> {
    /// The stamp of the commit of the batch's command buffer, where the take is a batch's: it
    /// takes no upload of a later commit (`FirstFill::upload`).
    fn commit(&self) -> Option<u64> {
        match self {
            FillTake::All => None,
            FillTake::Owned { commit, .. } | FillTake::Reached { commit, .. } => Some(*commit),
        }
    }

    /// What names the owner of the memory under a key, where the take knows it.
    fn owner_of(&self) -> Option<&dyn Fn(ResourceKey) -> Option<ObjectId>> {
        match self {
            FillTake::All => None,
            FillTake::Owned { owner_of, .. } | FillTake::Reached { owner_of, .. } => {
                Some(*owner_of)
            }
        }
    }
}

/// An entry of a take, in its order ([`FirstMoves::take_with`]).
enum Sorted {
    Kept(Waiting),
    Taken(Held),
    /// A fill whose turn waits until every move taken is known, and its buffer, held until the
    /// lock is free.
    Fill(Weak<dyn Filled>, FirstFill, Arc<dyn Filled>),
}

/// A resource whose first work a submission took, held; a move's and a staged copy's with what
/// owns the memory they write, as the take found it ([`Held::owner`]).
enum Held {
    Moves(Arc<dyn Fresh>, Vec<FirstMove>, Option<ObjectId>),
    /// The buffer, not kept, the hold on its memory, and its fill.
    Fill(Weak<dyn Filled>, Box<dyn Any + Send + Sync>, FirstFill),
    Texels(Arc<dyn Any + Send + Sync>, TexelCopy, Option<ObjectId>),
}

impl Held {
    /// What keeps the resource, or its memory (and a fill's staging buffer).
    fn keeper(self) -> Box<dyn Any + Send + Sync> {
        match self {
            // The moves keep the pages their images are bound to.
            Held::Moves(fresh, moves, _) => Box::new((fresh, moves)),
            Held::Fill(_, memory, fill) => match fill.staging {
                Some(staging) => Box::new((memory, staging)),
                None => memory,
            },
            Held::Texels(image, copy, _) => Box::new((image, copy.staging, copy.decode)),
        }
    }

    /// What owns the memory its work writes, by its identity, as the program's uses name it
    /// (`used_resources::owner_of`): `None` where the take did not know it, or where no use of
    /// the program's names that memory (the device's own resources).
    fn owner(&self) -> Option<ObjectId> {
        match self {
            Held::Moves(_, _, owner) | Held::Texels(_, _, owner) => *owner,
            Held::Fill(_, _, fill) => fill.owner,
        }
    }
}

/// What owns the memory the work of `held` writes, each once; `None` where one's owner is
/// unknown, or memory no use of the program's names, which any command may reach.
fn owners_of<'h>(held: impl IntoIterator<Item = &'h Held>) -> Option<Vec<ObjectId>> {
    let mut owners = Vec::new();
    for held in held {
        let owner = held.owner()?;
        if !owners.contains(&owner) {
            owners.push(owner);
        }
    }
    Some(owners)
}

/// Of `points`, the last on each `VkQueue`, in the order each queue first came: a wait for it
/// orders every one before it there.
pub(crate) fn last_of_each(points: impl IntoIterator<Item = Point>) -> Vec<Point> {
    let mut last: Vec<Point> = Vec::new();
    for at in points {
        match last.iter_mut().find(|point| point.queue == at.queue) {
            Some(point) => point.value = point.value.max(at.value),
            None => last.push(at),
        }
    }
    last
}

/// The device's buffer of [`ZERO_BYTES`] zero bytes, host-visible and written by the host once,
/// which the copies read: made at the first resource that reads zero, kept with the device.
/// A buffer of a pattern's texel repeated likewise ([`FirstMoves::pattern`]). And the device's
/// decode scratch, in its own memory ([`FirstMoves::decode_scratch`]).
struct Zeros {
    raw: vk::Buffer,
    placement: Placement,
}

/// The device's first work not yet done, and, on a device of several `VkQueue`s, the last batch
/// on each that did some and may not have completed (the module docs).
#[derive(Default)]
pub(crate) struct FirstMoves {
    /// The submissions' order: held by each from its take until its submission was made
    /// ([`Taken`]), so no submission is made between another's take and its submission; and by
    /// each registration where [`ordered`](Self::ordered), so no resource registers meanwhile
    /// (the module docs).
    order: Mutex<()>,
    /// Each registration waits for its place in the order: on a device whose heap sets take no
    /// writes while a command buffer that binds them is pending, where a batch cannot hold the
    /// texture slots' writes instead (`heap_sets::hold`).
    ordered: bool,
    /// The registrations that found a submission holding the order and waited for its place
    /// (`WorkCounts::registrations_waited`).
    waited: AtomicU64,
    /// The submissions that found another holding the order and waited for their place, and the
    /// nanoseconds they waited (`WorkCounts::first_work_order_waits`): a submission holds the
    /// order from its take to its `vkQueueSubmit`, and an internal submission
    /// (`device/submit.rs`) holds it while it waits for the GPU, so this is the scheduler's
    /// thread blocking before its own `vkQueueSubmit`.
    submissions_waited: AtomicU64,
    submission_wait_nanos: AtomicU64,
    waiting: Mutex<Vec<Waiting>>,
    /// The batches that did some and may not have completed, each at its point, with what owns
    /// the memory its work wrote ([`Taken::owners`]; `None`: memory any command may reach).
    done_at: Mutex<Vec<(Point, Option<Vec<ObjectId>>)>>,
    /// The internal submission that did some and may not have completed: the timeline it
    /// signals and the value (`device/submit.rs`).
    internal_at: Mutex<Option<(vk::Semaphore, u64)>>,
    zeros: Mutex<Option<Zeros>>,
    /// The device's buffers of Metal's patterns, each of [`PATTERN_BYTES`] bytes of its texel
    /// repeated, by texel: made at the first texture that reads it, kept with the device.
    patterns: Mutex<Vec<(&'static [u8], Zeros)>>,
    /// The device's multisampled scratch images of [`SCRATCH_SIDE`]², by the format of the images
    /// they write and samples: made at the first texture that copies one, kept with the device;
    /// `None` where the device makes no such image.
    scratches: Mutex<Vec<((vk::Format, vk::SampleCountFlags), Option<Scratch>)>>,
    /// The device's scratch buffer of the decodes into decoded twins that submissions take
    /// (`texture/decoded_twin.rs`, `PassScratch::Device`), and its bytes: made at the first such
    /// decode, kept with the device.
    decode_scratch: Mutex<Option<(Zeros, u64)>>,
    /// The work queued for the staging queue's next submission (`texture/staging_copy.rs`):
    /// copies into images that copy there, which every submission of the device's makes there
    /// before its own.
    pub(super) staging: Mutex<super::staging_copy::Queued>,
    /// The staging queue's buffer of [`ZERO_BYTES`] zero bytes, host-visible, which the moves
    /// queued there copy from (`texture/staging_copy.rs`): made at the first, kept with the
    /// device, and used by no other queue, as the device's own zeros are exclusive to its
    /// families.
    pub(super) staging_zeros: Mutex<Option<super::staging_copy::StagedBytes>>,
}

/// The side of each of the device's multisampled scratch images ([`FirstMoves::scratch`]): its
/// copies tile a texture's first level with it.
pub(crate) const SCRATCH_SIDE: u32 = 256;

/// A multisampled scratch image of the device's and its memory.
struct Scratch {
    image: vk::Image,
    placement: Placement,
}

/// The bytes of each of the device's buffers of a pattern ([`FirstMoves::pattern`]): the most one
/// copy of it reads, a whole number of every pattern's texels (of 1 to 16 bytes, powers of two),
/// and at least the widest level's row of the widest texel (16384 texels of 16 bytes), so each
/// copy is of whole rows.
pub(crate) const PATTERN_BYTES: u64 = 256 << 10;

/// Whether the host calls' copies `waiting` stages passed [`STAGED_BYTES`] or [`STAGED_COPIES`]:
/// copies into buffers from staging buffers of their own, and texel copies. A copy from a host
/// copy ([`Keeps::HostCopy`]) stages nothing and counts toward neither.
fn past_bound(waiting: &[Waiting]) -> bool {
    let staged = waiting.iter().filter_map(|entry| match entry {
        Waiting::Fill(_, fill) => {
            (fill.staged).then(|| fill.copies.iter().map(|copy| copy.size).sum())
        }
        Waiting::Texels(_, copy) => Some(copy.bytes),
        Waiting::Moves(_) => None,
    });
    let (copies, bytes) = staged.fold((0, 0), |(copies, bytes), size| (copies + 1, bytes + size));
    copies > STAGED_COPIES || bytes > STAGED_BYTES
}

/// The work a handle handed out settles ([`FirstMoves::take_mine`]): the fills whose destination
/// is `buffer`, and the moves of and copies into the image at `image` (an `Image`'s address).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mine {
    pub(crate) buffer: Option<vk::Buffer>,
    pub(crate) image: Option<*const ()>,
}

impl Mine {
    /// Whether `entry` is its work.
    fn names(self, entry: &Waiting) -> bool {
        let image = |at: *const ()| self.image == Some(at);
        match entry {
            Waiting::Fill(_, fill) => self.buffer == Some(fill.buffer),
            Waiting::Moves(weak) => image(weak.as_ptr().cast()),
            Waiting::Texels(weak, _) => image(weak.as_ptr().cast()),
        }
    }
}

impl FirstMoves {
    /// The first moves of a device whose registrations wait for their place in the order where
    /// `ordered` ([`FirstMoves::ordered`]).
    pub(crate) fn new(ordered: bool) -> FirstMoves {
        FirstMoves {
            ordered,
            ..FirstMoves::default()
        }
    }

    /// The submissions' order, for a registration: held where registrations wait for their
    /// place (the module docs), else not taken. One that finds a submission holding it is
    /// counted as it waits.
    fn registration_place(&self) -> Option<MutexGuard<'_, ()>> {
        self.ordered.then(|| match self.order.try_lock() {
            Ok(order) => order,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.waited.fetch_add(1, Ordering::Relaxed);
                lock(&self.order)
            }
        })
    }

    /// The registrations that waited for a submission's place in the order so far.
    pub(crate) fn registrations_waited(&self) -> u64 {
        self.waited.load(Ordering::Relaxed)
    }

    /// `fresh`, just shared, has moves waiting.
    pub(crate) fn register(&self, fresh: Weak<dyn Fresh>) {
        let _order = self.registration_place();
        lock(&self.waiting).push(Waiting::Moves(fresh));
    }

    /// Queue work for the staging queue's next submission (`texture/staging_copy.rs`) with
    /// `queue`, in its place in the order where registrations wait for it (the module docs): a
    /// new image's move, which a batch's texture table must not name before the batch, or one
    /// before it, took it.
    pub(super) fn queue_staging<R>(
        &self,
        queue: impl FnOnce(&mut super::staging_copy::Queued) -> R,
    ) -> R {
        let _order = self.registration_place();
        queue(&mut lock(&self.staging))
    }

    /// The new buffer `keeper`, whose bytes `range` of `buffer` the host cannot write, waits for
    /// its fill with zeros.
    pub(crate) fn register_fill(
        &self,
        device: &Device,
        keeper: Weak<dyn Filled>,
        buffer: vk::Buffer,
        range: std::ops::Range<u64>,
    ) -> Result<(), Error> {
        let fill = FirstFill::zeros(device, self, buffer, range)?.keyed(&keeper);
        let _order = self.registration_place();
        lock(&self.waiting).push(Waiting::Fill(keeper, fill));
        Ok(())
    }

    /// The exportable buffer `keeper` waits for the acquire of its `VkBuffer` `buffer`, whole,
    /// from `families`' source, a queue family reserved for external memory, for its target
    /// (`FirstFill::acquired`): what an external agent wrote to its memory, which the caller
    /// waited for, is the device's to read once a submission made it, before any of its commands
    /// (`Buffer::external_write_done`). Its barrier leads the submission's fills, so the caller
    /// settles the buffer's own fills first.
    #[cfg(unix)]
    pub(crate) fn register_acquire(
        &self,
        keeper: Weak<dyn Filled>,
        buffer: vk::Buffer,
        families: (u32, u32),
    ) {
        let fill = FirstFill::acquired(buffer, families).keyed(&keeper);
        let _order = self.registration_place();
        lock(&self.waiting).push(Waiting::Fill(keeper, fill));
    }

    /// The buffer `keeper`, whose bytes the host cannot write, waits for the copy of `length`
    /// bytes of `source` (a buffer and its byte the copy starts at), which `keeps` keeps, onto
    /// its `VkBuffer` `buffer` from byte `start`: after every fill with zeros waiting, its own
    /// among them where it is new, and after the copies registered before it. Whether the copies
    /// staged now passed their bound, so the caller makes a submission for them: never where it
    /// keeps a host copy ([`Keeps::HostCopy`]), which stages nothing.
    pub(crate) fn register_bytes(
        &self,
        keeper: Weak<dyn Filled>,
        (buffer, start): (vk::Buffer, u64),
        source: (vk::Buffer, u64),
        length: u64,
        keeps: Keeps,
    ) -> bool {
        let fill = FirstFill::bytes(buffer, start, source, length, keeps).keyed(&keeper);
        self.register_fill_of_bytes(keeper, fill)
    }

    /// As [`register_bytes`](Self::register_bytes), of an upload into a buffer the program's
    /// command buffers may have read already (a `Managed` buffer's copy in, `managed.rs`), of
    /// bytes taken at the commit stamped `commit` (0: at none): a batch takes it only where its
    /// commands may reach the buffer's memory and its command buffer was committed at or after
    /// that commit (`FirstFill::upload`), so it lands after the command buffers before that batch
    /// on its `VkQueue`, which may still read the bytes it replaces, and before that batch's
    /// commands, and a command buffer committed before the bytes were taken, submitted after,
    /// reads the bytes of its own commit, not these. A batch of another queue that reaches none
    /// of the buffer leaves it waiting.
    pub(crate) fn register_upload(
        &self,
        keeper: Weak<dyn Filled>,
        (buffer, start): (vk::Buffer, u64),
        source: (vk::Buffer, u64),
        length: u64,
        keeps: Keeps,
        commit: u64,
    ) -> bool {
        let fill = FirstFill::bytes(buffer, start, source, length, keeps).keyed(&keeper);
        let fill = FirstFill {
            upload: Some(commit),
            ..fill
        };
        self.register_fill_of_bytes(keeper, fill)
    }

    /// As [`register_upload`](Self::register_upload), of the bytes `from` written inline, which
    /// `keeps` keeps mapped until the submission that takes them completed: that submission reads
    /// them as it is recorded and writes them by `vkCmdUpdateBuffer`s in its command stream, in
    /// the order and with the barriers of a copy (`FirstFill::inline`). The caller's contract:
    /// `start` and `from`'s length are multiples of 4, and the bytes lie in the `VkBuffer`
    /// `buffer`.
    pub(crate) fn register_inline_upload(
        &self,
        keeper: Weak<dyn Filled>,
        (buffer, start): (vk::Buffer, u64),
        from: HostBytes<'static>,
        keeps: Box<dyn Any + Send + Sync>,
        commit: u64,
    ) {
        let (source, length) = ((vk::Buffer::null(), 0), from.len() as u64);
        let fill = FirstFill::bytes(buffer, start, source, length, Keeps::HostCopy(keeps));
        let fill = FirstFill {
            upload: Some(commit),
            inline: Some(from),
            ..fill.keyed(&keeper)
        };
        self.register_fill_of_bytes(keeper, fill);
    }

    /// Register `fill`, a copy of bytes into `keeper`'s buffer: whether it staged them
    /// ([`Keeps::Staging`]) and the copies staged now passed their bound.
    fn register_fill_of_bytes(&self, keeper: Weak<dyn Filled>, fill: FirstFill) -> bool {
        let _order = self.registration_place();
        let staged = fill.staged;
        let mut waiting = lock(&self.waiting);
        waiting.push(Waiting::Fill(keeper, fill));
        staged && past_bound(&waiting)
    }

    /// The host call's `copy` into the images of `keeper` waits for the device's next
    /// submission, after their first moves and every copy noted before it. Whether the copies
    /// staged now passed their bound, so the caller makes a submission for them.
    pub(crate) fn register_texels(
        &self,
        keeper: Weak<dyn Any + Send + Sync>,
        copy: TexelCopy,
    ) -> bool {
        let _order = self.registration_place();
        let mut waiting = lock(&self.waiting);
        waiting.push(Waiting::Texels(keeper, copy));
        past_bound(&waiting)
    }

    /// The kernels and sets of the staged copies' decode passes not made yet
    /// ([`DecodePass::unmade`]), for `Device::make_pending_decodes` to make once this lock is
    /// free.
    pub(crate) fn unmade_decodes(&self) -> Vec<Arc<FirstUse>> {
        let waiting = lock(&self.waiting);
        (waiting.iter())
            .filter_map(|entry| match entry {
                Waiting::Texels(_, copy) => copy.decode.as_ref().and_then(DecodePass::unmade),
                Waiting::Moves(_) | Waiting::Fill(..) => None,
            })
            .collect()
    }

    /// Take the work a submission does: all of it, but the moves of images bound to host pages
    /// whose key `named` does not name (an internal submission's; a batch names every key, the
    /// module docs). What goes is dropped once the lock is free.
    ///
    /// The submission's place in the order is held until it was made or the work given back
    /// ([`Taken`]): every submission takes, whether or not anything waits.
    pub(crate) fn take(&self, named: impl Fn(ResourceKey) -> bool) -> Taken<'_> {
        self.take_with(named, FillTake::All, || None, (|_| true, |_| false))
    }

    /// As [`take`](Self::take), for a handle handed out (`Buffer::vk_buffer`,
    /// `Buffer::export_memory_fd`, `Texture::vk_image`): only the work [`Mine`] names, in the order
    /// it was registered, the rest left waiting for the device's next submission. A move of an
    /// image bound to host pages is taken where `Mine` names its image, whatever its key.
    pub(crate) fn take_mine(&self, mine: Mine) -> Taken<'_> {
        let wanted = |entry: &Waiting| mine.names(entry);
        self.take_with(|_| true, FillTake::All, || None, (wanted, |_| false))
    }

    /// As [`take`](Self::take), for a submission whose commands may bind the heap sets (a
    /// batch's), taking the fills `fills` says: where registrations do not wait for their place
    /// ([`FirstMoves::ordered`]), it also holds the texture slots' writes (`heap_sets::hold`,
    /// which `hold` makes) from before it takes until its submission was made or the work given
    /// back (the module docs). The images whose key `overwrites` names are written whole by the
    /// first use among its commands before any of them reads a texel (`command_buffer/submit.rs`):
    /// their moves that may go so ([`FirstMove::mark_overwritable`]) record no zeros, counted
    /// (`WorkCounts::private_textures_overwritten_unfilled`).
    pub(crate) fn take_holding(
        &self,
        named: impl Fn(ResourceKey) -> bool,
        fills: FillTake<'_>,
        hold: impl FnOnce() -> HeldSlots,
        overwrites: impl Fn(ResourceKey) -> bool,
    ) -> Taken<'_> {
        let hold = || (!self.ordered).then(hold);
        self.take_with(named, fills, hold, (|_| true, overwrites))
    }

    /// As [`take`](Self::take), for a batch of a queue made for copies
    /// (`Device::new_command_queue_for_copies`), whose commands name every buffer they reach and
    /// no image, on a queue family of transfers alone: the fills `fills` says, and nothing else.
    /// The moves and the staged copies wait, in their order, for the device's next submission
    /// elsewhere: that family records no clear or pass a move may need, and the device's images
    /// are shared only among its own families (`sharing.rs`). It binds no heap set, so it holds
    /// no texture slot's writes.
    pub(crate) fn take_fills_alone(&self, fills: FillTake<'_>) -> Taken<'_> {
        let wanted = |entry: &Waiting| matches!(entry, Waiting::Fill(..));
        self.take_with(|_| true, fills, || None, (wanted, |_| false))
    }

    /// The submissions' order, for a submission: taken, and the wait counted where another
    /// submission holds it ([`FirstMoves::submissions_waited`]). Nothing is read from the host's
    /// clock where the order is free.
    fn submission_place(&self) -> MutexGuard<'_, ()> {
        match self.order.try_lock() {
            Ok(order) => order,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                let waited = Instant::now();
                let order = lock(&self.order);
                let nanos = u64::try_from(waited.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.submissions_waited.fetch_add(1, Ordering::Relaxed);
                self.submission_wait_nanos
                    .fetch_add(nanos, Ordering::Relaxed);
                order
            }
        }
    }

    /// The submissions that waited for another's place in the order so far, and the nanoseconds
    /// they waited.
    pub(crate) fn submissions_waited(&self) -> (u64, u64) {
        (
            self.submissions_waited.load(Ordering::Relaxed),
            self.submission_wait_nanos.load(Ordering::Relaxed),
        )
    }

    /// As [`take`](Self::take), holding the slots `hold` gives once its place is taken, taking
    /// only the entries `wanted` says, and of the fills those `fills` says: the rest stay waiting,
    /// in their order. A fill's turn is settled once every move taken is known: a move of an
    /// image in memory another resource owns, or a staged copy, takes every fill (the module
    /// docs). The moves of images whose key `overwrites` names record no zeros where they may go
    /// so ([`take_holding`](Self::take_holding)).
    fn take_with(
        &self,
        named: impl Fn(ResourceKey) -> bool,
        fills: FillTake<'_>,
        hold: impl FnOnce() -> Option<HeldSlots>,
        (wanted, overwrites): (impl Fn(&Waiting) -> bool, impl Fn(ResourceKey) -> bool),
    ) -> Taken<'_> {
        let order = self.submission_place();
        let slots = hold();
        let mut held = Vec::new();
        let mut left: Vec<Box<dyn Any + Send + Sync>> = Vec::new();
        let mut every_fill = matches!(fills, FillTake::All | FillTake::Owned { .. });
        {
            let mut waiting = lock(&self.waiting);
            if waiting.is_empty() {
                return Taken {
                    moves: self,
                    held,
                    every_fill,
                    _slots: slots,
                    _order: order,
                };
            }
            // Each entry in its order: kept, taken, or a fill whose turn waits for the moves.
            let mut sorted = Vec::with_capacity(waiting.len());
            // Whether a staged copy waits for a later submission: its decode's kernel is not
            // made, staged after this submission's walk (`Device::make_pending_decodes`), which
            // its maker's texture, made after that walk began, no command of this submission
            // reads. Every staged copy after it waits too, so the copies keep their order.
            let mut texels_kept = false;
            for entry in waiting.drain(..) {
                if !wanted(&entry) {
                    sorted.push(Sorted::Kept(entry));
                    continue;
                }
                match entry {
                    Waiting::Moves(weak) => {
                        let Some(fresh) = weak.upgrade() else {
                            continue;
                        };
                        if fresh.rebindable().is_some_and(|key| !named(key)) {
                            sorted.push(Sorted::Kept(Waiting::Moves(weak)));
                            left.push(Box::new(fresh));
                            continue;
                        }
                        let unzeroed = fresh.key().is_some_and(&overwrites);
                        let moves: Vec<FirstMove> = fresh
                            .take()
                            .into_iter()
                            .filter_map(FirstMove::held)
                            .map(|mut first| {
                                first.unzeroed = unzeroed && first.overwritable;
                                first
                            })
                            .collect();
                        if moves.is_empty() {
                            left.push(Box::new(fresh));
                        } else {
                            every_fill |= fresh.in_other_memory();
                            let owner = (fills.owner_of())
                                .and_then(|owner_of| fresh.key().and_then(owner_of));
                            sorted.push(Sorted::Taken(Held::Moves(fresh, moves, owner)));
                        }
                    }
                    Waiting::Fill(weak, fill) => {
                        let Some(buffer) = weak.upgrade() else {
                            continue;
                        };
                        sorted.push(Sorted::Fill(weak, fill, buffer));
                    }
                    // An image gone takes its copy and staging buffer with it.
                    Waiting::Texels(weak, copy) => {
                        texels_kept |= copy.decode.as_ref().is_some_and(|pass| !pass.is_made());
                        if texels_kept {
                            sorted.push(Sorted::Kept(Waiting::Texels(weak, copy)));
                            continue;
                        }
                        if let Some(image) = weak.upgrade() {
                            every_fill = true;
                            let owner =
                                (fills.owner_of()).and_then(|owner_of| copy.key.and_then(owner_of));
                            sorted.push(Sorted::Taken(Held::Texels(image, copy, owner)));
                        }
                    }
                }
            }
            let mut kept = Vec::new();
            for entry in sorted {
                match entry {
                    Sorted::Kept(entry) => kept.push(entry),
                    Sorted::Taken(entry) => held.push(entry),
                    Sorted::Fill(weak, mut fill, buffer) => {
                        fill.owner =
                            (fills.owner_of()).and_then(|owner_of| fill.key.and_then(owner_of));
                        let reached = match (&fills, fill.owner) {
                            (FillTake::Reached { reaches, .. }, Some(owner)) => reaches(owner),
                            (FillTake::Owned { reaches, .. }, Some(owner))
                                if fill.upload.is_some() =>
                            {
                                reaches(owner)
                            }
                            _ => true,
                        };
                        // An upload goes only with a batch that may reach its memory, whatever
                        // else the batch takes, of a command buffer committed at or after the
                        // commit that took its bytes.
                        let in_time = match (fill.upload, fills.commit()) {
                            (Some(upload), Some(commit)) => upload <= commit,
                            _ => true,
                        };
                        let taken = in_time && (reached || (every_fill && fill.upload.is_none()));
                        match taken {
                            // The submission holds the buffer's memory; the buffer goes as the
                            // program lets go of it, once the lock is free.
                            true => {
                                if let Some(memory) = buffer.memory() {
                                    held.push(Held::Fill(weak, memory, fill));
                                }
                            }
                            false => kept.push(Waiting::Fill(weak, fill)),
                        }
                        left.push(Box::new(buffer));
                    }
                }
            }
            waiting.extend(kept);
        }
        drop(left);
        Taken {
            moves: self,
            held,
            every_fill,
            _slots: slots,
            _order: order,
        }
    }

    /// Whether a batch whose commands name nothing takes no work waiting: none waits but fills
    /// of memory whose key `named_only` says only a batch that may reach it takes
    /// ([`FillTake::Reached`]), and uploads, which only such a batch takes on any device
    /// (`FirstFill::upload`). A command buffer that needs no GPU work then goes with no
    /// submission, and one that finds work waiting carries it as before
    /// (`command_buffer/submit.rs`).
    pub(crate) fn none_waiting_but(&self, named_only: impl Fn(ResourceKey) -> bool) -> bool {
        lock(&self.waiting).iter().all(|entry| {
            matches!(entry, Waiting::Fill(_, fill)
                if fill.upload.is_some() || fill.key.is_some_and(&named_only))
        })
    }

    /// Let go of the fills and copies that wait for what is at `keeper` (the address of what
    /// [`register_bytes`](Self::register_bytes), [`register_fill`](Self::register_fill),
    /// [`register_acquire`](Self::register_acquire) or [`register_texels`](Self::register_texels)
    /// was given), as it goes: a gone resource's work is no submission's, and what that work
    /// holds (a staging buffer, a host copy, each holding the device) goes with it, once the lock
    /// is free. Left waiting, it would hold the device past the program's last handle until a
    /// submission took it, and a device the program let go of makes none.
    pub(crate) fn forget(&self, keeper: *const ()) {
        let gone: Vec<Waiting> = {
            let mut waiting = lock(&self.waiting);
            let (gone, kept): (Vec<Waiting>, Vec<Waiting>) =
                waiting.drain(..).partition(|entry| match entry {
                    Waiting::Fill(weak, _) => weak.as_ptr().cast::<()>() == keeper,
                    Waiting::Texels(weak, _) => weak.as_ptr().cast::<()>() == keeper,
                    Waiting::Moves(_) => false,
                });
            waiting.extend(kept);
            gone
        };
        drop(gone);
    }

    /// Whether a host call's copy waits for the images of what is at `keeper` (the address of
    /// what [`register_texels`](Self::register_texels) was given). A decode alone waiting
    /// ([`TexelCopy::decode_only`]) is not: it writes the decoded twin, never the texels a copy
    /// reads or writes.
    pub(crate) fn texels_wait(&self, keeper: *const ()) -> bool {
        lock(&self.waiting).iter().any(|entry| {
            matches!(entry, Waiting::Texels(weak, copy)
                if weak.as_ptr().cast::<()>() == keeper && !copy.copies.is_empty())
        })
    }

    /// Whether first work waits on `buffer` (a `VkBuffer`): a new buffer's fill with zeros or
    /// its staged bytes, or a `Managed` buffer's upload, which the device's next submission makes
    /// before its commands, over whatever another writer wrote there meanwhile.
    pub(crate) fn fills_wait(&self, buffer: vk::Buffer) -> bool {
        lock(&self.waiting)
            .iter()
            .any(|entry| matches!(entry, Waiting::Fill(_, fill) if fill.buffer == buffer))
    }

    /// The batches on `VkQueue`s other than `queue` that did first work, where `completed` says
    /// they may not have completed, each at its point with what owns the memory its work wrote
    /// (`None`: memory any command may reach): what a submission on `queue` may wait for
    /// ([`last_of_each`] of their points).
    pub(crate) fn waits(
        &self,
        queue: usize,
        completed: impl Fn(Point) -> bool,
    ) -> Vec<(Point, Option<Vec<ObjectId>>)> {
        let mut done_at = lock(&self.done_at);
        done_at.retain(|&(point, _)| !completed(point));
        (done_at.iter())
            .filter(|(point, _)| point.queue != queue)
            .cloned()
            .collect()
    }

    /// The internal submission a batch on a `VkQueue` other than the first waits for: the last
    /// that did first work, while it may not have completed (the timeline it signals, and the
    /// value).
    pub(crate) fn internal_wait(&self) -> Option<(vk::Semaphore, u64)> {
        *lock(&self.internal_at)
    }

    /// The internal submission that signals `value` completed.
    pub(crate) fn internal_completed(&self, value: u64) {
        let mut internal_at = lock(&self.internal_at);
        if internal_at.is_some_and(|(_, at)| at <= value) {
            *internal_at = None;
        }
    }

    /// The device's buffer of zeros, made now if it is not yet: `ZERO_BYTES` long, which every
    /// copy of `length` bytes from it stays within.
    pub(crate) fn zeros(&self, device: &Device, length: u64) -> Result<vk::Buffer, Error> {
        if length > ZERO_BYTES {
            return Err(internal("a copy of zeros longer than the device's zeros"));
        }
        let mut zeros = lock(&self.zeros);
        if let Some(zeros) = &*zeros {
            return Ok(zeros.raw);
        }
        let made = Zeros::new(device, ZERO_BYTES, &[0])?;
        let raw = made.raw;
        *zeros = Some(made);
        Ok(raw)
    }

    /// The device's scratch buffer of the decodes into decoded twins that submissions take, made
    /// now if it is not yet: `bytes` long, every such decode's scratch (`texture/decoded_twin.rs`,
    /// `PassScratch::Device`). The submissions record their decodes one after another, each past
    /// a barrier from everything before it, and one on another queue waits for the last that did
    /// some (`waits`), so the decodes share it.
    pub(crate) fn decode_scratch(&self, device: &Device, bytes: u64) -> Result<vk::Buffer, Error> {
        let mut scratch = lock(&self.decode_scratch);
        if let Some((made, made_bytes)) = &*scratch {
            return match *made_bytes == bytes {
                true => Ok(made.raw),
                false => Err(internal(
                    "a decode scratch of another length than the device's",
                )),
            };
        }
        let made = Zeros::private(device, bytes)?;
        let raw = made.raw;
        *scratch = Some((made, bytes));
        Ok(raw)
    }

    /// The device's buffer of `texel` repeated, made now if it is not yet: [`PATTERN_BYTES`]
    /// long, which every copy of `length` bytes from it stays within.
    pub(crate) fn pattern(
        &self,
        device: &Device,
        texel: &'static [u8],
        length: u64,
    ) -> Result<vk::Buffer, Error> {
        if length > PATTERN_BYTES || !PATTERN_BYTES.is_multiple_of(texel.len() as u64) {
            return Err(internal(
                "a copy of a pattern past the device's buffer of it",
            ));
        }
        let mut patterns = lock(&self.patterns);
        if let Some((_, made)) = patterns.iter().find(|(own, _)| *own == texel) {
            return Ok(made.raw);
        }
        let made = Zeros::new(device, PATTERN_BYTES, texel)?;
        let raw = made.raw;
        patterns.push((texel, made));
        Ok(raw)
    }

    /// The device's multisampled scratch image of `format` (an unsigned-integer one, of the
    /// texel size of `target`) and `samples` that copies into images of `target`:
    /// [`SCRATCH_SIDE`]² of one level and layer, with transfer and color attachment usage (which
    /// a multisampled image takes), mutable to `target` (so a device that copies between formats
    /// through a view of the source in the destination's, MoltenVK, may); made at the first
    /// texture that copies it; `None` where the device makes no such image.
    pub(crate) fn scratch(
        &self,
        device: &Device,
        format: vk::Format,
        target: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Option<vk::Image>, Error> {
        let mut scratches = lock(&self.scratches);
        if let Some((_, made)) = scratches.iter().find(|(own, _)| *own == (target, samples)) {
            return Ok(made.as_ref().map(|made| made.image));
        }
        let made = Scratch::new(device, [format, target], samples)?;
        let image = made.as_ref().map(|made| made.image);
        scratches.push(((target, samples), made));
        Ok(image)
    }

    /// Release the buffer of zeros, the patterns' and the scratch images into `memory`; the
    /// device is idle.
    pub(crate) fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        lock(&self.staging).destroy(device, memory);
        if let Some(zeros) = lock(&self.staging_zeros).take() {
            zeros.destroy(device, memory);
        }
        if let Some(zeros) = &*lock(&self.zeros) {
            zeros.destroy(device, memory);
        }
        for (_, pattern) in lock(&self.patterns).iter() {
            pattern.destroy(device, memory);
        }
        if let Some((scratch, _)) = &*lock(&self.decode_scratch) {
            scratch.destroy(device, memory);
        }
        for (_, scratch) in lock(&self.scratches).iter() {
            if let Some(scratch) = scratch {
                // SAFETY: no submission uses the image.
                unsafe { release::object(device, scratch.image) };
                memory.free(device, &scratch.placement);
            }
        }
    }
}

impl Scratch {
    /// A scratch image of `formats`' first, viewed as either, and `samples`
    /// ([`FirstMoves::scratch`]), used by no submission yet; `None` where the device makes no
    /// such image.
    fn new(
        device: &Device,
        formats: [vk::Format; 2],
        samples: vk::SampleCountFlags,
    ) -> Result<Option<Scratch>, Error> {
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let shared = device.shared();
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .image_type(vk::ImageType::TYPE_2D)
                .format(formats[0])
                .extent(vk::Extent3D {
                    width: SCRATCH_SIDE,
                    height: SCRATCH_SIDE,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        );
        let limits = device.image_limits(&info, &formats)?;
        let fits =
            limits.max_extent.width >= SCRATCH_SIDE && limits.max_extent.height >= SCRATCH_SIDE;
        if !limits.sample_counts.contains(samples) || !fits {
            return Ok(None);
        }
        let raw_device = shared.vk();
        let info = info.push_next(&mut list);
        // SAFETY: the device is live and `info` valid, its combination supported.
        let image = unsafe { creation::image(raw_device, &info) }?;
        // SAFETY: `image` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
        let class = MemoryClass::Private;
        let placement =
            match crate::memory::allocate(&shared.memory, raw_device, &requirements, class) {
                Ok(placement) => placement,
                Err(error) => {
                    // SAFETY: the image is unused.
                    unsafe { release::object(raw_device, image) };
                    return Err(error);
                }
            };
        // SAFETY: the placement meets the image's requirements and is bound nowhere else.
        let bound = unsafe {
            crate::creation::bind_image(raw_device, image, placement.memory, placement.offset)
        };
        if let Err(error) = bound {
            // SAFETY: the image is unused.
            unsafe { release::object(raw_device, image) };
            lock(&shared.memory).free(raw_device, &placement);
            return Err(vulkan(error));
        }
        Ok(Some(Scratch { image, placement }))
    }
}

impl Zeros {
    /// A buffer of `length` bytes of `texel` repeated (`length` a multiple of it), written by
    /// the host now.
    fn new(device: &Device, length: u64, texel: &[u8]) -> Result<Zeros, Error> {
        let shared = device.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(length)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Shared,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let made = Zeros { raw, placement };
        let bound = match made.placement.mapped {
            None => Err(Error::OutOfMemory(MemoryKind::Host)),
            // SAFETY: the placement meets the buffer's requirements and is bound nowhere else.
            Some(mapped) => unsafe {
                crate::creation::bind_buffer(
                    raw_device,
                    raw,
                    made.placement.memory,
                    made.placement.offset,
                )
            }
            .map_err(vulkan)
            .map(|()| mapped),
        };
        let mapped = match bound {
            Ok(mapped) => mapped,
            Err(error) => {
                made.destroy(raw_device, &mut lock(&shared.memory));
                return Err(error);
            }
        };
        // SAFETY: the mapping covers the placement's `length` bytes, which only this buffer
        // uses and no submission reads yet; the memory is host-coherent.
        let bytes = unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), length as usize) };
        match texel {
            [byte] => bytes.fill(*byte),
            _ => {
                for chunk in bytes.chunks_mut(texel.len()) {
                    chunk.copy_from_slice(&texel[..chunk.len()]);
                }
            }
        }
        Ok(made)
    }

    /// A buffer of `length` bytes in the device's own memory, which a decode pass writes and
    /// reads ([`FirstMoves::decode_scratch`]): copied in and out of and a storage buffer, its
    /// bytes unwritten.
    fn private(device: &Device, length: u64) -> Result<Zeros, Error> {
        let shared = device.shared();
        let raw_device = shared.vk();
        let usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let info =
            (shared.sharing()).buffer(vk::BufferCreateInfo::default().size(length).usage(usage));
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let made = Zeros { raw, placement };
        // SAFETY: the placement meets the buffer's requirements and is bound nowhere else.
        let bound = unsafe {
            crate::creation::bind_buffer(
                raw_device,
                raw,
                made.placement.memory,
                made.placement.offset,
            )
        };
        if let Err(error) = bound {
            made.destroy(raw_device, &mut lock(&shared.memory));
            return Err(vulkan(error));
        }
        Ok(made)
    }

    /// Release the buffer and its memory into `memory`; nothing uses them.
    fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        // SAFETY: no submission uses the buffer.
        unsafe { release::object(device, self.raw) };
        memory.free(device, &self.placement);
    }
}

/// First work a submission took ([`FirstMoves::take`]): given back as this drops unless the
/// submission was made ([`submitted`](Taken::submitted)).
pub(crate) struct Taken<'a> {
    moves: &'a FirstMoves,
    held: Vec<Held>,
    /// It took every fill waiting ([`every_fill`](Self::every_fill)).
    every_fill: bool,
    /// The texture slots' writes it holds ([`FirstMoves::take_holding`]), made as this drops,
    /// before its place goes: the field before `_order`, which drops after it.
    _slots: Option<HeldSlots>,
    /// The submission's place in the order ([`FirstMoves::take`]), let go as this drops.
    _order: MutexGuard<'a, ()>,
}

/// What orders a submission's first work after the commands before it on its `VkQueue`
/// ([`Taken::record`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Earlier {
    /// The work's own barrier from every command before it.
    Barriered,
    /// The end of the batch before it on its `VkQueue`, for a batch's first work of uploads
    /// alone ([`Taken::uploads_only`]) on a device that orders its batches by their ends, where
    /// no internal submission may still run (`command_buffer/submit.rs`): the work records no
    /// barrier of its own from the commands before it. Every command before it on that
    /// `VkQueue` is a batch's, each followed in submission order by its batch's end,
    /// `Barrier::to_all` (`batch_order.rs`: `ALL_COMMANDS` and `MEMORY_WRITE` to `ALL_COMMANDS`
    /// and `MEMORY_READ | MEMORY_WRITE`), or an internal submission's that the host saw complete
    /// before the batch took its place in the order, which no internal submission can then
    /// take until the batch's submission was made (`DeviceShared::internal_running`). "If
    /// vkCmdPipelineBarrier was recorded outside a render pass instance, the first
    /// synchronization scope includes all commands that occur earlier in submission order" and
    /// "the second synchronization scope includes all commands that occur later in submission
    /// order" (the Vulkan specification, `vkCmdPipelineBarrier`), the uploads among them: every
    /// earlier read of the device copy runs before them ("Write-after-read hazards can be solved
    /// with just an execution dependency", Execution and Memory Dependencies), and every earlier
    /// write of it is made available and visible to their writes (`MEMORY_WRITE` "specifies all
    /// write accesses"), as the batch's own commands rely on it there
    /// (`Tracker::by_ends`). An internal submission seen complete was waited for on the host
    /// before the batch's `vkQueueSubmit`, whose first synchronization scope includes "anything
    /// that happened-before it". A command buffer of another `VkQueue` is ordered by the
    /// batch's semaphore waits, as before.
    Ended,
}

/// What a batch's first work of uploads alone is made visible to by the barrier after it
/// ([`Taken::record`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Later {
    /// Every later command, of every stage.
    All,
    /// The batch's own commands' uses of the device copies the uploads write: the stages and
    /// access types the batch's encoders' uses of those buffers name, on a device that orders
    /// its batches by their ends (`command_buffer/submit.rs`). The barrier's second scope is
    /// limited to them ("the second synchronization scope is limited to operations on the
    /// pipeline stages determined by the destination stage mask", and its second access scope to
    /// "memory accesses performed by pipeline stages explicitly specified in the destination stage
    /// mask", the Vulkan specification, `vkCmdPipelineBarrier` and Execution and Memory
    /// Dependencies), which holds every command of the batch that reads or writes them. Every
    /// later batch on its `VkQueue` comes after the batch's end, `Barrier::to_all`, whose first
    /// scope "includes all commands that occur earlier in submission order" with `ALL_COMMANDS`
    /// and `MEMORY_WRITE` (the uploads' transfers among them) and whose second includes every
    /// command after it; a command buffer of another `VkQueue` waits for the batch's signal,
    /// whose "first access scope includes all memory access performed by the device" (Semaphore
    /// Signaling).
    Batch(vk::PipelineStageFlags, vk::AccessFlags),
}

impl<'a> Taken<'a> {
    /// The keys of the buffers its uploads write (`FirstFill::upload`), each once; `None` where
    /// one names none.
    pub(crate) fn upload_keys(&self) -> Option<Vec<ResourceKey>> {
        let mut keys = Vec::new();
        for held in &self.held {
            if let Held::Fill(_, _, fill) = held {
                if fill.upload.is_some() {
                    let key = fill.key?;
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
            }
        }
        Some(keys)
    }

    /// Whether it took every fill waiting: it took them all ([`FillTake::All`]), or took a move of
    /// an image in memory another resource owns, or a staged copy, each of which comes after every
    /// fill of that memory. Its commands then may reach what any fill-queue submission not yet
    /// complete filled too (`command_buffer/submit.rs`).
    pub(crate) fn every_fill(&self) -> bool {
        self.every_fill
    }

    /// Whether it took first work ([`record`](Self::record)'s): a move or a fill.
    pub(crate) fn has_first(&self) -> bool {
        (self.held.iter()).any(|held| matches!(held, Held::Moves(..) | Held::Fill(..)))
    }

    /// Whether it took a move that keeps a buffer's bytes by copies around its layout transition
    /// ([`GpuKeep`]): the submission that records it waits for every earlier submission on the
    /// device's other `VkQueue`s, whose commands may access those bytes while the transition
    /// rewrites them.
    pub(crate) fn keeps(&self) -> bool {
        (self.held.iter()).any(|held| {
            matches!(held, Held::Moves(_, moves, _) if moves.iter().any(|first| first.keep.is_some()))
        })
    }

    /// Whether its work is uploads alone (`FirstFill::upload`): copies or inline writes of
    /// `Managed` buffers' host copies into their device copies, and nothing else.
    pub(crate) fn uploads_only(&self) -> bool {
        !self.held.is_empty()
            && (self.held.iter())
                .all(|held| matches!(held, Held::Fill(_, _, fill) if fill.upload.is_some()))
    }

    /// Whether it took host calls' staged copies ([`record_staged`](Self::record_staged)'s).
    pub(crate) fn has_staged(&self) -> bool {
        (self.held.iter()).any(|held| matches!(held, Held::Texels(..)))
    }

    /// Its buffers' fills, taken out for the fill queue (`device/staging.rs`), in the order they
    /// were noted; the moves and staged copies stay, recorded after them in submission order.
    /// `None` where it took no fill, its fills write fewer than `least` bytes in all, or it took
    /// an acquire from an external agent, whose barrier names the device's family as its target
    /// and leads every fill (all stay then).
    pub(crate) fn take_fills(&mut self, least: u64) -> Option<Fills<'a>> {
        let fill = |held: &Held| matches!(held, Held::Fill(..));
        let acquire =
            |held: &Held| matches!(held, Held::Fill(_, _, fill) if fill.acquire.is_some());
        let written: u64 = (self.held.iter())
            .map(|held| match held {
                Held::Fill(_, _, fill) => fill.written(),
                _ => 0,
            })
            .sum();
        if !self.held.iter().any(fill) || written < least || self.held.iter().any(acquire) {
            return None;
        }
        let (fills, rest): (Vec<Held>, Vec<Held>) = self.held.drain(..).partition(fill);
        self.held = rest;
        Some(Fills {
            moves: self.moves,
            held: fills,
        })
    }

    /// Record the host calls' staged copies it took into `command_buffer`, each after every
    /// write before it (the first work, which [`record`](Self::record) records into this command
    /// buffer before it or a submission before records; the copy before), then the barrier that
    /// makes them visible to every command after and to the host.
    ///
    /// # Safety
    ///
    /// As [`record`](Self::record)'s; every image's first move is recorded before it in
    /// submission order -- earlier in this command buffer, or in a submission before.
    pub(crate) unsafe fn record_staged(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        let staged = self.held.iter().filter_map(|held| match held {
            Held::Texels(_, copy, _) => Some(copy),
            Held::Moves(..) | Held::Fill(..) => None,
        });
        let mut copied = false;
        for copy in staged {
            transfer_barrier(device, command_buffer, true);
            exit_guard::called(Call::CmdPipelineBarrier);
            for (image, region) in &copy.copies {
                // SAFETY: the caller's contract; the image is live (held), in `GENERAL` (its
                // move was made before) and has `TRANSFER_DST` usage; the copy reads within the
                // staging buffer, which the copy keeps.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        copy.source,
                        *image,
                        vk::ImageLayout::GENERAL,
                        std::slice::from_ref(region),
                    );
                }
                exit_guard::called(Call::CmdCopyBufferToImage);
            }
            // The decoded twin's texels of what the copies wrote, after them.
            if let Some(decode) = &copy.decode {
                // SAFETY: the caller's contract; the copy holds the pass, and the image holds its
                // twin, both in `GENERAL` (their moves were made before).
                unsafe { decode.record(device, command_buffer) };
            }
            copied = true;
        }
        if copied {
            transfer_barrier(device, command_buffer, false);
            exit_guard::called(Call::CmdPipelineBarrier);
        }
    }

    /// Record the work into `command_buffer`, of a family of `family`'s queue flags: the
    /// buffers' fills with zeros, after every command before (memory other commands wrote, a
    /// heap's included), then their copies of staged bytes, after the zeros, then the copies out
    /// of the spans the moves keep ([`GpuKeep`]), after every command before, then a barrier
    /// carrying every image's move, after the fills (an image placed in a new heap moves after
    /// the heap's fill) and the copies out, then the copies of the kept spans back and the
    /// images' copies of zeros, then the barrier that makes every write visible to every command
    /// after and to the host; to every command after alone where the work is uploads alone,
    /// which write device copies the host never reads ([`uploaded_barrier`]), or to what `later`
    /// names ([`Later::Batch`]). Uploads alone take no barrier from the commands before where
    /// `earlier` is [`Earlier::Ended`].
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, for one of the device's queue
    /// families, of `family`'s flags; the submission that runs it is made before any that uses
    /// one of the resources, on the same `VkQueue` or waited for; [`Earlier::Ended`] and
    /// [`Later::Batch`] only as their docs say.
    pub(crate) unsafe fn record(
        &self,
        (device, counts): (&Vk, &Counters),
        command_buffer: vk::CommandBuffer,
        family: vk::QueueFlags,
        (earlier, later): (Earlier, Later),
    ) {
        if self.held.is_empty() {
            return;
        }
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        let written = || {
            [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)]
        };
        let fills: Vec<&FirstFill> = (self.held.iter())
            .filter_map(|held| match held {
                Held::Fill(_, _, fill) => Some(fill),
                Held::Moves(..) | Held::Texels(..) => None,
            })
            .collect();
        let moves = || {
            self.held.iter().flat_map(|held| match held {
                Held::Moves(_, moves, _) => moves.as_slice(),
                Held::Fill(..) | Held::Texels(..) => &[],
            })
        };
        // SAFETY: the caller's contract.
        unsafe { record_fills((device, counts), command_buffer, &fills, earlier) };
        let keeps: Vec<&GpuKeep> = moves().filter_map(|first| first.keep.as_ref()).collect();
        if !keeps.is_empty() {
            // The spans out, after every command before (the fills and staged copies above
            // among them), each before its image's move, which the move's barrier orders.
            // SAFETY: the caller's contract.
            unsafe {
                let flags = vk::DependencyFlags::empty();
                device.cmd_pipeline_barrier(command_buffer, all, all, flags, &written(), &[], &[]);
            }
            exit_guard::called(Call::CmdPipelineBarrier);
            for keep in &keeps {
                // SAFETY: as above; the move holds both buffers.
                unsafe { keep.record(device, command_buffer, false) };
            }
        }
        let images: Vec<vk::ImageMemoryBarrier<'_>> =
            moves().map(|first| first.barrier(family)).collect();
        if !images.is_empty() {
            // SAFETY: the caller's contract; every image is live (held) and in the layout its
            // move starts from.
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    all,
                    all,
                    vk::DependencyFlags::empty(),
                    &written(),
                    &[],
                    &images,
                );
            }
            exit_guard::called(Call::CmdPipelineBarrier);
        }
        let mut copied = !fills.is_empty();
        // The spans back through their buffers, after the moves, whose barrier makes the
        // transitions' writes and the copies out visible to them.
        for keep in &keeps {
            // SAFETY: as above.
            unsafe { keep.record(device, command_buffer, true) };
            counts.first_move_kept(keep.length);
            copied = true;
        }
        for first in moves() {
            if first.unzeroed {
                counts.private_texture_overwritten_unfilled();
            }
            if !first.writes() {
                continue;
            }
            // SAFETY: as above; the image is in `GENERAL` now and has `TRANSFER_DST` usage where
            // it reads zero by a transfer.
            unsafe { first.record_zero(device, command_buffer, family) };
            copied = true;
        }
        let uploads = (self.held.iter())
            .all(|held| matches!(held, Held::Fill(_, _, fill) if fill.upload.is_some()));
        if copied && uploads {
            // SAFETY: the caller's contract.
            unsafe { uploaded_barrier(device, command_buffer, later) };
            exit_guard::called(Call::CmdPipelineBarrier);
            if later != Later::All {
                counts.managed_upload_barrier_narrowed();
            }
        } else if copied {
            transfer_barrier(device, command_buffer, false);
            exit_guard::called(Call::CmdPipelineBarrier);
        }
        // Zeros cleared in render passes, from their attachments' stages, visible to every
        // command after.
        if moves().any(FirstMove::clears_in_passes) {
            // SAFETY: the caller's contract.
            unsafe {
                let flags = vk::DependencyFlags::empty();
                device.cmd_pipeline_barrier(command_buffer, all, all, flags, &written(), &[], &[]);
            }
            exit_guard::called(Call::CmdPipelineBarrier);
        }
    }

    /// What owns the memory its work writes ([`Held::owner`]), each once; `None` where one's
    /// owner is unknown, or memory no use of the program's names.
    pub(crate) fn owners(&self) -> Option<Vec<ObjectId>> {
        owners_of(&self.held)
    }

    /// The submission that records the work was made; on a device of several `VkQueue`s by a
    /// batch at `at` (`None` for a submission waited for, or a device of one), kept with what
    /// owns the memory its work writes ([`owners`](Self::owners)) until it completed
    /// ([`FirstMoves::waits`]). What the submission holds until it completes.
    pub(crate) fn submitted(mut self, at: Option<Point>) -> Vec<Box<dyn Any + Send + Sync>> {
        if let (Some(at), false) = (at, self.held.is_empty()) {
            let owners = self.owners();
            let mut done_at = lock(&self.moves.done_at);
            // One whose work any command may reach covers those before it on its `VkQueue`: a
            // wait for it orders them too.
            if owners.is_none() {
                done_at.retain(|(point, _)| point.queue != at.queue);
            }
            done_at.push((at, owners));
        }
        self.held.drain(..).map(Held::keeper).collect()
    }

    /// The internal submission that records the work, signalling `value` of `timeline`, was
    /// made: a batch on another `VkQueue` waits for it until it completed
    /// ([`FirstMoves::internal_completed`]). What the submission holds until it completes.
    pub(crate) fn submitted_internally(
        mut self,
        timeline: vk::Semaphore,
        value: u64,
    ) -> Vec<Box<dyn Any + Send + Sync>> {
        if !self.held.is_empty() {
            *lock(&self.moves.internal_at) = Some((timeline, value));
        }
        self.held.drain(..).map(Held::keeper).collect()
    }
}

/// Whether `fills` are uploads alone (`FirstFill::upload`): copies or inline writes of a
/// `Managed` buffer's host copy into its device copy, which the host never reads (`contents` maps
/// the host copy), and none an acquire.
fn uploads_only(fills: &[&FirstFill]) -> bool {
    fills.iter().all(|fill| fill.upload.is_some())
}

/// The barrier after uploads alone (`Taken::record`): every later command, of every stage, after
/// their transfers' writes, which it makes visible to it, or the commands `later` names
/// ([`Later::Batch`]). No host access: the uploads write device
/// copies, which the host never reads, so it needs no domain operation to the host ("If the
/// destination access mask includes `VK_ACCESS_HOST_WRITE_BIT` or `VK_ACCESS_HOST_READ_BIT`, a
/// memory domain operation is performed where available memory in the device domain is also made
/// available to the host domain", the Vulkan specification, Buffer Memory Barriers), as
/// `transfer_barrier`'s for writes the host may read performs.
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass; [`Later::Batch`] only as its docs say,
/// its access types those its stages make.
unsafe fn uploaded_barrier(device: &Vk, command_buffer: vk::CommandBuffer, later: Later) {
    let (stages, access) = match later {
        Later::All => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
        Later::Batch(stages, access) => (stages, access),
    };
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(access);
    // SAFETY: the caller's contract.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            stages,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Record `fills` into `command_buffer`: a barrier after every command before (memory other
/// commands wrote, a heap's included), carrying the acquires among them (to the transfers alone
/// where they are uploads alone, which carry none: [`uploads_only`]; none where `earlier`, the
/// end of the batch before, orders them: [`Earlier::Ended`]), then the fills with zeros, then,
/// after a barrier, their copies of staged bytes, each copy into bytes a copy before it wrote
/// after a barrier.
///
/// # Safety
///
/// As [`Taken::record`]'s; each buffer is live (held) and shared with the command buffer's family
/// (`sharing.rs`), and each acquire names that family as its target.
unsafe fn record_fills(
    (device, counts): (&Vk, &Counters),
    command_buffer: vk::CommandBuffer,
    fills: &[&FirstFill],
    earlier: Earlier,
) {
    let all = vk::PipelineStageFlags::ALL_COMMANDS;
    let written = || {
        [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)]
    };
    if !fills.is_empty() && uploads_only(fills) {
        // Uploads alone, transfers that write device copies and read host copies or staging
        // buffers, carry no acquire: their barrier waits for every command before and makes every
        // write before available, but visible to transfers alone, the only accesses in its second
        // scope ("its second access scope only includes memory accesses performed by pipeline
        // stages explicitly specified in the destination stage mask", the Vulkan specification,
        // Execution and Memory Dependencies). What runs after them waits for them by the barrier
        // after them (`Taken::record`) or by the submission's signal (`Fills::record`). None
        // where the end of the batch before them orders every command before them
        // ([`Earlier::Ended`]).
        match earlier {
            Earlier::Ended => counts.managed_upload_unbarriered(),
            Earlier::Barriered => {
                transfer_barrier(device, command_buffer, true);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
        }
    } else if !fills.is_empty() {
        let acquires: Vec<vk::BufferMemoryBarrier<'_>> = fills
            .iter()
            .filter_map(|fill| fill.acquire_barrier())
            .collect();
        // SAFETY: the caller's contract; each acquired buffer is live (held), and its acquire
        // names the device's family for it (`Device::external_acquirer`).
        unsafe {
            let flags = vk::DependencyFlags::empty();
            let (memory, buffers) = (&written(), &acquires);
            device.cmd_pipeline_barrier(command_buffer, all, all, flags, memory, buffers, &[]);
        }
        exit_guard::called(Call::CmdPipelineBarrier);
    }
    let (zeros, staged): (Vec<&FirstFill>, Vec<&FirstFill>) = fills
        .iter()
        .copied()
        .partition(|fill| fill.staging.is_none());
    // Copies into bytes a copy before them wrote land after it, in the order they were noted:
    // a barrier before each that meets one since the last barrier. A `Managed` no-copy
    // buffer's copies in (`managed_copy.rs`: its making's, a rebind's, each
    // `did_modify_range`'s) read its pages at different times.
    let mut landed: Vec<(vk::Buffer, std::ops::Range<u64>)> = Vec::new();
    for (at, fill) in zeros.iter().chain(&staged).enumerate() {
        if at == zeros.len() && at > 0 {
            // SAFETY: the caller's contract.
            unsafe {
                let flags = vk::DependencyFlags::empty();
                device.cmd_pipeline_barrier(command_buffer, all, all, flags, &written(), &[], &[]);
            }
            exit_guard::called(Call::CmdPipelineBarrier);
        }
        if let Some(words) = &fill.filled {
            // SAFETY: as above; the words lie in the buffer, held, which has `TRANSFER_DST`
            // usage as every buffer of the runtime's, their ends multiples of 4
            // (`FirstFill::zeros`).
            unsafe {
                let (buffer, words) = (fill.buffer, words.clone());
                crate::word_fill::record_command(device, command_buffer, buffer, words, 0);
            }
            exit_guard::called(Call::CmdFillBuffer);
            counts.new_buffer_zero_filled();
            continue;
        }
        // An acquire copies nothing: its barrier led the fills.
        if fill.copies.is_empty() {
            continue;
        }
        let ranges = fill
            .copies
            .iter()
            .map(|copy| copy.dst_offset..copy.dst_offset + copy.size);
        let meets = ranges.clone().any(|range| {
            (landed.iter()).any(|(buffer, before)| {
                *buffer == fill.buffer && range.start < before.end && before.start < range.end
            })
        });
        if meets {
            // SAFETY: the caller's contract.
            unsafe {
                let flags = vk::DependencyFlags::empty();
                let memory = &written();
                device.cmd_pipeline_barrier(command_buffer, all, all, flags, memory, &[], &[]);
            }
            exit_guard::called(Call::CmdPipelineBarrier);
            landed.clear();
        }
        landed.extend(ranges.map(|range| (fill.buffer, range)));
        if let Some(from) = fill.inline {
            // SAFETY: as above; the fill keeps the host copy `from` lies in mapped, and its copy's
            // bytes lie in the buffer, its ends multiples of 4
            // (`FirstMoves::register_inline_upload`).
            unsafe { record_inline(device, command_buffer, fill, from) };
            counts.managed_range_inlined();
            continue;
        }
        // SAFETY: as above; every copy lies within its source, the device's zeros (live and
        // `ZERO_BYTES` long) or a staging buffer the fill keeps, and within the buffer,
        // held, which has `TRANSFER_DST` usage as every buffer of the runtime's.
        unsafe {
            device.cmd_copy_buffer(command_buffer, fill.source, fill.buffer, &fill.copies);
        }
        exit_guard::called(Call::CmdCopyBuffer);
    }
}

/// Record `fill`'s upload written inline (`FirstFill::inline`): the bytes `from`, read now,
/// onto its copy's range of its buffer, by one `vkCmdUpdateBuffer` of each [`UPDATE_BYTES`] of
/// them, which the spec makes a transfer for barriers ("treated as a transfer operation for the
/// purposes of synchronization barriers"), as the copy it stands for.
///
/// # Safety
///
/// As [`record_fills`]'; `from` is mapped host memory (the fill keeps its host copy), and the
/// copy's `dst_offset` and `size` (`from`'s length) are multiples of 4 that lie in the buffer,
/// which has `TRANSFER_DST` usage (`VUID-vkCmdUpdateBuffer-dstOffset-00036`, `-dataSize-00038`,
/// `-dstBuffer-00034`).
unsafe fn record_inline(
    device: &Vk,
    command_buffer: vk::CommandBuffer,
    fill: &FirstFill,
    from: HostBytes<'static>,
) {
    let Some(copy) = fill.copies.first() else {
        return;
    };
    // Read once, by a copy: the program's CPU may write the host copy meanwhile.
    let bytes = from.to_vec();
    // Each piece at most `UPDATE_BYTES` (`VUID-vkCmdUpdateBuffer-dataSize-00037`), a multiple of
    // 4 from a multiple of 4, never empty (`-dataSize-arraylength`).
    for (index, piece) in bytes.chunks(UPDATE_BYTES).enumerate() {
        let at = copy.dst_offset + (index * UPDATE_BYTES) as u64;
        // SAFETY: the caller's contract; outside a render pass, recording.
        unsafe { device.cmd_update_buffer(command_buffer, fill.buffer, at, piece) };
        exit_guard::called(Call::CmdUpdateBuffer);
    }
}

/// `held`, which no submission made, given back to `moves`, still to be done.
fn give_back(moves: &FirstMoves, held: impl Iterator<Item = Held>) {
    let mut given = Vec::new();
    for held in held {
        match held {
            Held::Moves(fresh, moves, _) => {
                fresh.give_back(moves.into_iter().map(FirstMove::unheld).collect());
                given.push(Waiting::Moves(Arc::downgrade(&fresh)));
            }
            Held::Fill(buffer, _, fill) => {
                given.push(Waiting::Fill(buffer, fill));
            }
            Held::Texels(image, copy, _) => {
                given.push(Waiting::Texels(Arc::downgrade(&image), copy));
            }
        }
    }
    if !given.is_empty() {
        lock(&moves.waiting).extend(given);
    }
}

impl Drop for Taken<'_> {
    fn drop(&mut self) {
        give_back(self.moves, self.held.drain(..));
    }
}

/// A batch's buffers' fills, taken out of its first work for the fill queue
/// ([`Taken::take_fills`]): given back as this drops unless the submission that records them was
/// made ([`submitted`](Fills::submitted)).
pub(crate) struct Fills<'a> {
    moves: &'a FirstMoves,
    held: Vec<Held>,
}

impl Fills<'_> {
    /// Record the fills into `command_buffer`, as [`Taken::record`] records them: every write
    /// before them on the queue the command buffer goes to waited for by a barrier, the zeros,
    /// then the copies of staged bytes, in the order they were noted. No barrier follows them:
    /// the submission's signal makes their writes available ("The first access scope includes all
    /// memory access performed by the device", the Vulkan specification, 7.4.1 Semaphore
    /// Signaling), and the batch's wait for it makes them visible to its commands.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, for a family every buffer of the
    /// device's is shared with (`sharing.rs`: the staging queue's, which the fill queue's is); the
    /// submission that runs it is made before any that uses one of the buffers, which waits for
    /// it.
    pub(crate) unsafe fn record(
        &self,
        counted: (&Vk, &Counters),
        command_buffer: vk::CommandBuffer,
    ) {
        let fills: Vec<&FirstFill> = (self.held.iter())
            .filter_map(|held| match held {
                Held::Fill(_, _, fill) => Some(fill),
                Held::Moves(..) | Held::Texels(..) => None,
            })
            .collect();
        // SAFETY: the caller's contract; none is an acquire ([`Taken::take_fills`]).
        unsafe { record_fills(counted, command_buffer, &fills, Earlier::Barriered) };
    }

    /// What owns the memory the fills write, as the batch that took them found it
    /// ([`FillTake::Reached`]), each once; `None` where one writes memory no use of the
    /// program's names, or the take found no owner, which any command may reach.
    pub(crate) fn owners(&self) -> Option<Vec<ObjectId>> {
        owners_of(&self.held)
    }

    /// The submission that records the fills was made: what it holds until it completes (each
    /// buffer's memory, and a fill's staging buffer).
    pub(crate) fn submitted(mut self) -> Vec<Box<dyn Any + Send + Sync>> {
        self.held.drain(..).map(Held::keeper).collect()
    }
}

impl Drop for Fills<'_> {
    fn drop(&mut self) {
        give_back(self.moves, self.held.drain(..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    /// A resource holding moves, `key` its key where they are of an image bound to host pages.
    struct Resource {
        moves: Mutex<Vec<FirstMove>>,
        key: Option<ResourceKey>,
    }

    impl Fresh for Resource {
        fn take(&self) -> Vec<FirstMove> {
            lock(&self.moves).drain(..).collect()
        }

        fn give_back(&self, moves: Vec<FirstMove>) {
            lock(&self.moves).extend(moves);
        }

        fn rebindable(&self) -> Option<ResourceKey> {
            self.key.filter(|_| !lock(&self.moves).is_empty())
        }
    }

    /// A scratch's copies tile the first level of every layer, each from the scratch's origin,
    /// the last of a row or column cut to the level's edge.
    #[test]
    fn scratch_copies_tile_each_layer() {
        let side = SCRATCH_SIDE;
        let extent = [side * 2 + 3, side - 1];
        let Pattern::Scratch { copies, .. } =
            Patterned::scratch(vk::Image::null(), Default::default(), extent, 2)
        else {
            panic!("not a scratch pattern");
        };
        assert_eq!(copies.len(), 2 * 3);
        let mut covered = 0;
        for copy in &copies {
            let origin = copy.src_offset;
            assert_eq!([origin.x, origin.y, origin.z], [0; 3]);
            assert_eq!(copy.src_subresource.base_array_layer, 0);
            assert!(copy.extent.width <= side && copy.extent.height <= side);
            assert_eq!(copy.dst_subresource.layer_count, 1);
            assert_eq!(copy.dst_subresource.mip_level, 0);
            let right = copy.dst_offset.x as u32 + copy.extent.width;
            let bottom = copy.dst_offset.y as u32 + copy.extent.height;
            assert!(right <= extent[0] && bottom <= extent[1]);
            covered += u64::from(copy.extent.width * copy.extent.height);
        }
        let layers: Vec<u32> = copies
            .iter()
            .map(|copy| copy.dst_subresource.base_array_layer)
            .collect();
        assert_eq!(layers, [0, 0, 0, 1, 1, 1]);
        assert_eq!(covered, 2 * u64::from(extent[0] * extent[1]));
    }

    /// The host pages of the tests' images bound to some, which live for good.
    static PAGES: std::sync::LazyLock<Arc<dyn Any + Send + Sync>> =
        std::sync::LazyLock::new(|| Arc::new(()));

    /// The move of image `raw`, of one bound to host pages where `rebindable`.
    fn first(raw: u64, rebindable: bool) -> FirstMove {
        FirstMove::new(
            vk::Image::from_raw(raw),
            Format::R8G8B8A8_UNORM,
            vk::ImageLayout::UNDEFINED,
            rebindable.then(|| Arc::downgrade(&*PAGES)),
        )
    }

    /// A resource holding the move of image `raw`, registered with `moves`.
    fn registered(moves: &FirstMoves, raw: u64, key: Option<ResourceKey>) -> Arc<Resource> {
        let held = Arc::new(Resource {
            moves: Mutex::new(vec![first(raw, key.is_some())]),
            key,
        });
        let weak: Weak<Resource> = Arc::downgrade(&held);
        moves.register(weak);
        held
    }

    /// A buffer of the tests, whose memory is a unit.
    struct Unit;

    impl Filled for Unit {
        fn memory(&self) -> Option<Box<dyn Any + Send + Sync>> {
            Some(Box::new(()))
        }
    }

    /// A buffer's fill, registered with `moves`: what keeps the buffer.
    fn fill(moves: &FirstMoves, raw: u64, length: u64) -> Arc<Unit> {
        let buffer = Arc::new(Unit);
        let unit: Weak<Unit> = Arc::downgrade(&buffer);
        let weak: Weak<dyn Filled> = unit;
        let fill = FirstFill::new(vk::Buffer::from_raw(raw), 16, length, vk::Buffer::null());
        lock(&moves.waiting).push(Waiting::Fill(weak, fill));
        buffer
    }

    /// As [`fill`], of memory whose uses name `key`.
    fn keyed_fill(moves: &FirstMoves, raw: u64, key: Option<ResourceKey>) -> Arc<Unit> {
        let buffer = fill(moves, raw, 16);
        if let Some(Waiting::Fill(_, fill)) = lock(&moves.waiting).last_mut() {
            fill.key = key;
        }
        buffer
    }

    /// An image in memory another resource owns, holding its move.
    struct Placed(Mutex<Vec<FirstMove>>);

    impl Fresh for Placed {
        fn take(&self) -> Vec<FirstMove> {
            lock(&self.0).drain(..).collect()
        }

        fn give_back(&self, moves: Vec<FirstMove>) {
            lock(&self.0).extend(moves);
        }

        fn rebindable(&self) -> Option<ResourceKey> {
            None
        }

        fn in_other_memory(&self) -> bool {
            true
        }
    }

    /// The `VkBuffer`s of the fills `taken` holds, in order.
    fn filled(taken: &Taken<'_>) -> Vec<u64> {
        (taken.held.iter())
            .filter_map(|held| match held {
                Held::Fill(_, _, fill) => Some(fill.buffer.as_raw()),
                _ => None,
            })
            .collect()
    }

    /// A batch on a device with a fill queue (`FillTake::Reached`) takes the fills of memory its
    /// commands reach and of memory no use names, in their order, and leaves the rest waiting,
    /// which a batch whose commands name nothing leaves too; one that takes a move of an image in
    /// memory another resource owns, or a staged copy, takes every fill.
    #[test]
    fn a_batch_takes_the_fills_of_memory_it_reaches() {
        let moves = FirstMoves::default();
        let (reached, unreached) = (ResourceKey::new(), ResourceKey::new());
        let (owner, other) = (ObjectId::new(), ObjectId::new());
        let owner_of = |key: ResourceKey| match key == reached {
            true => Some(owner),
            false => Some(other),
        };
        let reaches = |of: ObjectId| of == owner;
        let none = |_: ObjectId| false;
        let _buffers = [
            keyed_fill(&moves, 1, Some(reached)),
            keyed_fill(&moves, 2, Some(unreached)),
            keyed_fill(&moves, 3, None),
            keyed_fill(&moves, 4, Some(reached)),
        ];
        let named_only = |key: ResourceKey| owner_of(key).is_some();
        assert!(!moves.none_waiting_but(named_only));
        let fills = FillTake::Reached {
            owner_of: &owner_of,
            reaches: &reaches,
            commit: 0,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert!(!taken.every_fill());
        assert_eq!(filled(&taken), [1, 3, 4]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(named_only));
        assert!(!moves.none_waiting_but(|_| false));
        // A move of an image in another resource's memory takes the fill no use reaches.
        let placed = Arc::new(Placed(Mutex::new(vec![first(5, false)])));
        let weak: Weak<Placed> = Arc::downgrade(&placed);
        moves.register(weak);
        let fills = FillTake::Reached {
            owner_of: &owner_of,
            reaches: &none,
            commit: 0,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert!(taken.every_fill());
        assert_eq!(filled(&taken), [2]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(|_| false));
        // So does a staged copy.
        let _later = keyed_fill(&moves, 6, Some(unreached));
        let image: Arc<u32> = Arc::new(0);
        let image_weak: Weak<u32> = Arc::downgrade(&image);
        let copy = (vk::Image::null(), vk::BufferImageCopy::default());
        let staged = TexelCopy::new(vk::Buffer::null(), 16, vec![copy], Box::new(()));
        assert!(!moves.register_texels(image_weak, staged));
        let fills = FillTake::Reached {
            owner_of: &owner_of,
            reaches: &none,
            commit: 0,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert!(taken.every_fill());
        assert_eq!(filled(&taken), [6]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(|_| false));
    }

    /// As [`keyed_fill`], an upload into a buffer the program may have read already, of bytes
    /// taken at the commit stamped `commit` (`FirstMoves::register_upload`).
    fn upload(moves: &FirstMoves, raw: u64, key: ResourceKey, commit: u64) -> Arc<Unit> {
        let buffer = keyed_fill(moves, raw, Some(key));
        if let Some(Waiting::Fill(_, fill)) = lock(&moves.waiting).last_mut() {
            fill.upload = Some(commit);
        }
        buffer
    }

    /// An upload goes only with a batch whose commands may reach its memory, on a device with a
    /// fill queue or without one, even where the batch takes every other fill (a move of an
    /// image in another resource's memory, or a device without a fill queue); a batch that
    /// reaches none of it leaves it waiting, and so does one that needs no GPU work; an internal
    /// submission takes it.
    #[test]
    fn an_upload_goes_only_with_a_batch_that_reaches_its_buffer() {
        let moves = FirstMoves::default();
        let (used, unused) = (ResourceKey::new(), ResourceKey::new());
        let (used_id, unused_id) = (ObjectId::new(), ObjectId::new());
        let owner_of = |key: ResourceKey| match key == used {
            true => Some(used_id),
            false => Some(unused_id),
        };
        let reaches = |of: ObjectId| of == used_id;
        let none = |_: ObjectId| false;
        // Without a fill queue: a batch that reaches the other buffer takes every fill but the
        // upload.
        let _buffers = [
            upload(&moves, 1, unused, 1),
            keyed_fill(&moves, 2, Some(used)),
            upload(&moves, 3, used, 1),
        ];
        let fills = FillTake::Owned {
            owner_of: &owner_of,
            reaches: &reaches,
            commit: 1,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert!(taken.every_fill());
        assert_eq!(filled(&taken), [2, 3]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(|_| false), "only an upload waits");
        // With a fill queue, a move of an image in another resource's memory takes every fill but
        // the upload no command of the batch reaches.
        let placed = Arc::new(Placed(Mutex::new(vec![first(4, false)])));
        let weak: Weak<Placed> = Arc::downgrade(&placed);
        moves.register(weak);
        let fills = FillTake::Reached {
            owner_of: &owner_of,
            reaches: &none,
            commit: 1,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert!(taken.every_fill());
        assert!(filled(&taken).is_empty());
        drop(taken.submitted(None));
        // A batch that reaches it takes it.
        let reaches_unused = |of: ObjectId| of == unused_id;
        let fills = FillTake::Owned {
            owner_of: &owner_of,
            reaches: &reaches_unused,
            commit: 1,
        };
        let taken = moves.take_with(|_| true, fills, || None, (|_| true, |_| false));
        assert_eq!(filled(&taken), [1]);
        drop(taken.submitted(None));
        // An internal submission takes one whatever reaches it.
        let _later = upload(&moves, 5, unused, 1);
        let taken = moves.take(|_| true);
        assert_eq!(filled(&taken), [5]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(|_| false));
    }

    /// An upload goes with no batch of a command buffer committed before the commit that took its
    /// bytes, though the batch reaches its buffer and is submitted after that commit: the batch of
    /// the commit takes its own upload and leaves the later ones, in their order, to the batches
    /// of their commits, on a device with a fill queue or without one. An upload no commit took
    /// (stamped 0) goes with any batch that reaches it.
    #[test]
    fn an_upload_goes_with_no_batch_committed_before_its_bytes_were_taken() {
        let moves = FirstMoves::default();
        let used = ResourceKey::new();
        let used_id = ObjectId::new();
        let owner_of = |key: ResourceKey| (key == used).then_some(used_id);
        let reaches = |of: ObjectId| of == used_id;
        let _buffers = [
            upload(&moves, 1, used, 2),
            upload(&moves, 2, used, 3),
            upload(&moves, 3, used, 4),
        ];
        let owned = |commit: u64| FillTake::Owned {
            owner_of: &owner_of,
            reaches: &reaches,
            commit,
        };
        let reached = |commit: u64| FillTake::Reached {
            owner_of: &owner_of,
            reaches: &reaches,
            commit,
        };
        let taken = moves.take_with(|_| true, owned(1), || None, (|_| true, |_| false));
        assert!(filled(&taken).is_empty(), "committed before every upload");
        drop(taken.submitted(None));
        let taken = moves.take_with(|_| true, reached(2), || None, (|_| true, |_| false));
        assert_eq!(filled(&taken), [1]);
        drop(taken.submitted(None));
        let taken = moves.take_with(|_| true, owned(4), || None, (|_| true, |_| false));
        assert_eq!(filled(&taken), [2, 3]);
        drop(taken.submitted(None));
        let _made = upload(&moves, 4, used, 0);
        let taken = moves.take_with(|_| true, reached(1), || None, (|_| true, |_| false));
        assert_eq!(filled(&taken), [4]);
        drop(taken.submitted(None));
        assert!(moves.none_waiting_but(|_| false));
    }

    /// An image of the program's whose uses name its key, holding its move.
    struct Owned(Mutex<Vec<FirstMove>>, ResourceKey);

    impl Fresh for Owned {
        fn take(&self) -> Vec<FirstMove> {
            lock(&self.0).drain(..).collect()
        }

        fn give_back(&self, moves: Vec<FirstMove>) {
            lock(&self.0).extend(moves);
        }

        fn rebindable(&self) -> Option<ResourceKey> {
            None
        }

        fn key(&self) -> Option<ResourceKey> {
            Some(self.1)
        }
    }

    /// A batch that knows what owns the memory of the first work it takes keeps that with its
    /// point: two batches of one `VkQueue` over different owners both stay, and one whose work's
    /// owner no use keeps, which any command may reach, covers those before it there.
    #[test]
    fn a_batch_keeps_what_owns_its_first_work_s_memory() {
        let moves = FirstMoves::default();
        let [texture, buffer, unkept] = [(); 3].map(|()| ResourceKey::new());
        let (texture_id, buffer_id) = (ObjectId::new(), ObjectId::new());
        let owner_of = |key: ResourceKey| {
            (key == texture)
                .then_some(texture_id)
                .or((key == buffer).then_some(buffer_id))
        };
        let owned = |raw: u64, key: ResourceKey| {
            let image = Arc::new(Owned(Mutex::new(vec![first(raw, false)]), key));
            let weak: Weak<Owned> = Arc::downgrade(&image);
            moves.register(weak);
            image
        };
        let take = || {
            let fills = FillTake::Owned {
                owner_of: &owner_of,
                reaches: &|_| true,
                commit: 0,
            };
            moves.take_with(|_| true, fills, || None, (|_| true, |_| false))
        };
        let image = owned(1, texture);
        let at = Point { queue: 1, value: 3 };
        let taken = take();
        assert_eq!(taken.owners(), Some(vec![texture_id]));
        drop(taken.submitted(Some(at)));
        let _buffer = keyed_fill(&moves, 2, Some(buffer));
        let later = Point { queue: 1, value: 4 };
        drop(take().submitted(Some(later)));
        assert_eq!(
            moves.waits(0, |_| false),
            [(at, Some(vec![texture_id])), (later, Some(vec![buffer_id]))]
        );
        assert_eq!(last_of_each([at, later]), [later]);
        let unknown = owned(3, unkept);
        let last = Point { queue: 1, value: 5 };
        let taken = take();
        assert_eq!(taken.owners(), None);
        drop(taken.submitted(Some(last)));
        assert_eq!(moves.waits(0, |_| false), [(last, None)]);
        drop((image, unknown));
    }

    #[test]
    fn a_fill_copies_the_zeros_in_pieces_of_the_zeros_length() {
        let fill = FirstFill::new(
            vk::Buffer::null(),
            16,
            2 * ZERO_BYTES + 4,
            vk::Buffer::null(),
        );
        let pieces: Vec<(u64, u64, u64)> = (fill.copies.iter())
            .map(|copy| (copy.src_offset, copy.dst_offset, copy.size))
            .collect();
        assert_eq!(
            pieces,
            [
                (0, 16, ZERO_BYTES),
                (0, 16 + ZERO_BYTES, ZERO_BYTES),
                (0, 16 + 2 * ZERO_BYTES, 4)
            ]
        );
    }

    #[test]
    fn a_submission_not_made_gives_its_work_back_for_the_next() {
        let moves = FirstMoves::default();
        let image = registered(&moves, 1, None);
        let buffer = fill(&moves, 2, 64);
        let taken = moves.take(|_| false);
        assert_eq!(taken.held.len(), 2);
        assert!(
            lock(&image.moves).is_empty(),
            "the move is the submission's"
        );
        drop(taken);
        assert_eq!(lock(&image.moves).len(), 1, "the move is the image's again");
        let taken = moves.take(|_| false);
        assert_eq!(taken.held.len(), 2, "the next submission takes both");
        assert_eq!(taken.submitted(None).len(), 2);
        assert!(
            moves.take(|_| false).held.is_empty(),
            "a made submission keeps them"
        );
        assert!(lock(&image.moves).is_empty());
        drop(buffer);
    }

    /// A resource registered on another thread while a submission holds its place: whether the
    /// registration returned within `within`, and the resource once it did.
    fn register_beside(
        moves: &Arc<FirstMoves>,
        within: std::time::Duration,
    ) -> (bool, std::thread::JoinHandle<Arc<Resource>>) {
        let (sent, got) = std::sync::mpsc::channel();
        let registering = Arc::clone(moves);
        let thread = crate::exit_guard::spawn(std::thread::Builder::new(), move || {
            let image = registered(&registering, 7, None);
            // Received, or not where the wait for it ended first.
            sent.send(()).ok();
            image
        })
        .unwrap();
        (got.recv_timeout(within).is_ok(), thread)
    }

    /// A submission that finds another holding its place in the submissions' order waits for it,
    /// counted with the nanoseconds it waited (`WorkCounts::first_work_order_waits`): a
    /// submission holds that place from its take of the first work to its `vkQueueSubmit`, and
    /// one of the device's own (`device/submit.rs`) holds it while it waits for its work to
    /// complete on the GPU, so a batch's submission waits here, on the scheduler's thread, before
    /// its own `vkQueueSubmit`. A submission that finds the order free waits for nothing and is
    /// not counted.
    #[test]
    fn a_submission_that_waits_for_another_s_place_is_counted_with_its_nanos() {
        let moves = Arc::new(FirstMoves::default());
        // One that finds the order free: not counted, and no clock read.
        drop(moves.take(|_| false));
        assert_eq!(moves.submissions_waited(), (0, 0));
        let taken = moves.take(|_| false);
        let (sent, got) = std::sync::mpsc::channel();
        let beside = Arc::clone(&moves);
        let thread = crate::exit_guard::spawn(std::thread::Builder::new(), move || {
            drop(beside.take(|_| false));
            // Received, or not where the wait for it ended first.
            sent.send(()).ok();
        })
        .unwrap();
        assert!(
            got.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a submission took its place while another held it"
        );
        drop(taken);
        thread.join().unwrap();
        let (waits, nanos) = moves.submissions_waited();
        assert_eq!(
            waits, 1,
            "the submission's wait for its place was not counted"
        );
        assert!(nanos > 0, "the wait was counted as taking no time");
    }

    /// Where a batch holds the texture slots' writes ([`FirstMoves::ordered`] false), a resource
    /// registers while a submission holds its place, without waiting for it, and the next
    /// submission takes its move.
    #[test]
    fn a_registration_waits_for_no_submission_where_batches_hold_the_slots() {
        let moves = Arc::new(FirstMoves::new(false));
        let taken = moves.take(|_| false);
        let (returned, thread) = register_beside(&moves, std::time::Duration::from_secs(30));
        assert!(
            returned,
            "the registration waited for the submission's place"
        );
        assert!(
            taken.submitted(None).is_empty(),
            "registered after the take"
        );
        let image = thread.join().unwrap();
        assert_eq!(moves.registrations_waited(), 0, "counted as waiting");
        let next = moves.take(|_| false);
        assert_eq!(next.held.len(), 1, "the next submission takes the move");
        assert!(lock(&image.moves).is_empty());
    }

    /// Where registrations wait for their place ([`FirstMoves::ordered`]), a resource registers
    /// only once the submission holding its place let go of it, counted as it waits, and the next
    /// submission takes its move.
    #[test]
    fn a_registration_waits_for_the_submission_where_registrations_are_ordered() {
        let moves = Arc::new(FirstMoves::new(true));
        let taken = moves.take(|_| false);
        let (returned, thread) = register_beside(&moves, std::time::Duration::from_millis(200));
        assert!(
            !returned,
            "the registration came between a take and its submission"
        );
        assert!(taken.submitted(None).is_empty());
        let image = thread.join().unwrap();
        assert_eq!(moves.registrations_waited(), 1, "the wait counted");
        let next = moves.take(|_| false);
        assert_eq!(next.held.len(), 1, "the next submission takes the move");
        assert!(lock(&image.moves).is_empty());
    }

    #[test]
    fn a_submission_holds_a_filled_buffer_s_memory_not_the_buffer() {
        let moves = FirstMoves::default();
        let buffer = fill(&moves, 2, 64);
        let taken = moves.take(|_| false);
        assert_eq!(taken.held.len(), 1);
        assert_eq!(
            Arc::strong_count(&buffer),
            1,
            "the submission holds the buffer"
        );
        drop(taken);
        assert_eq!(
            lock(&moves.waiting).len(),
            1,
            "a failed submission gives the fill back"
        );
        let taken = moves.take(|_| false);
        let held = taken.submitted(None);
        assert_eq!(held.len(), 1, "the submission holds the memory");
        assert_eq!(Arc::strong_count(&buffer), 1, "and not the buffer");
        drop((held, buffer));
    }

    #[test]
    fn a_fill_of_staged_bytes_holds_its_staging_until_its_submission_completed() {
        let moves = FirstMoves::default();
        let buffer = Arc::new(Unit);
        let unit: Weak<Unit> = Arc::downgrade(&buffer);
        let staging = Arc::new(());
        let raw = |n| vk::Buffer::from_raw(n);
        moves.register_bytes(
            unit,
            (raw(2), 16),
            (raw(3), 0),
            64,
            Keeps::Staging(Box::new(staging.clone())),
        );
        let taken = moves.take(|_| false);
        let copies: Vec<(vk::Buffer, u64, u64, u64)> = (taken.held.iter())
            .flat_map(|held| match held {
                Held::Fill(_, _, fill) => (fill.copies.iter())
                    .map(|c| (fill.source, c.src_offset, c.dst_offset, c.size))
                    .collect(),
                Held::Moves(..) | Held::Texels(..) => Vec::new(),
            })
            .collect();
        assert_eq!(copies, [(raw(3), 0, 16, 64)]);
        drop(taken);
        assert_eq!(
            Arc::strong_count(&staging),
            2,
            "a failed submission keeps it"
        );
        let held = moves.take(|_| false).submitted(None);
        assert_eq!(Arc::strong_count(&staging), 2, "the submission holds it");
        drop(held);
        assert_eq!(
            Arc::strong_count(&staging),
            1,
            "and lets go once it completed"
        );
        drop(buffer);
    }

    #[test]
    fn a_staged_texel_copy_holds_its_image_and_staging_until_its_submission_completed() {
        let moves = FirstMoves::default();
        let image: Arc<dyn Any + Send + Sync> = Arc::new(());
        let staging = Arc::new(());
        let copy = |bytes| {
            let region = vk::BufferImageCopy::default();
            let copies = vec![(vk::Image::from_raw(5), region)];
            TexelCopy::new(
                vk::Buffer::from_raw(6),
                bytes,
                copies,
                Box::new(staging.clone()),
            )
        };
        assert!(!moves.register_texels(Arc::downgrade(&image), copy(64)));
        assert!(
            !moves.register_texels(Arc::downgrade(&image), copy(32)),
            "within the bound"
        );
        drop(moves.take(|_| false));
        assert_eq!(
            lock(&moves.waiting).len(),
            2,
            "a failed submission gives both back"
        );
        let held = moves.take(|_| false).submitted(None);
        assert_eq!(held.len(), 2);
        assert_eq!(
            Arc::strong_count(&image),
            3,
            "the submission holds the image"
        );
        assert_eq!(Arc::strong_count(&staging), 3, "and the staging buffers");
        drop(held);
        assert_eq!(
            (Arc::strong_count(&image), Arc::strong_count(&staging)),
            (1, 1)
        );
        moves.register_texels(Arc::downgrade(&image), copy(16));
        drop(image);
        assert!(
            moves.take(|_| false).held.is_empty(),
            "an image gone drops its copy"
        );
        assert_eq!(Arc::strong_count(&staging), 1, "and its staging buffer");
    }

    #[test]
    fn staged_copies_pass_their_bound_by_bytes_or_by_count() {
        let image: Arc<dyn Any + Send + Sync> = Arc::new(());
        let copy = |bytes| {
            let region = vk::BufferImageCopy::default();
            let copies = vec![(vk::Image::from_raw(5), region)];
            TexelCopy::new(vk::Buffer::from_raw(6), bytes, copies, Box::new(()))
        };
        let moves = FirstMoves::default();
        assert!(!moves.register_texels(Arc::downgrade(&image), copy(STAGED_BYTES)));
        assert!(
            moves.register_texels(Arc::downgrade(&image), copy(1)),
            "one byte past the bytes' bound"
        );
        let moves = FirstMoves::default();
        let keeper: Weak<dyn Filled> = Weak::<Unit>::new();
        let bytes = |moves: &FirstMoves| {
            let to = (vk::Buffer::from_raw(7), 0);
            let from = (vk::Buffer::from_raw(8), 0);
            moves.register_bytes(keeper.clone(), to, from, 4, Keeps::Staging(Box::new(())))
        };
        for _ in 0..STAGED_COPIES {
            assert!(!bytes(&moves));
        }
        assert!(bytes(&moves), "one copy past the count's bound");
        // A fill of zeros stages nothing.
        let moves = FirstMoves::default();
        let held: Vec<Arc<Unit>> = (0..=STAGED_COPIES as u64)
            .map(|raw| fill(&moves, raw, 16))
            .collect();
        assert!(!past_bound(&lock(&moves.waiting)));
        drop(held);
    }

    /// A copy from a host copy the buffer holds (`Keeps::HostCopy`: a `Managed` buffer's upload
    /// from its mirror or its pages' import) stages nothing: past the bytes' bound, and past the
    /// count's, in copies at the making and uploads a commit took, it passes neither bound, and a
    /// staged copy after them is counted alone.
    #[test]
    fn copies_from_a_host_copy_pass_no_bound() {
        let moves = FirstMoves::default();
        let keeper: Weak<dyn Filled> = Weak::<Unit>::new();
        let to = (vk::Buffer::from_raw(7), 0);
        let from = (vk::Buffer::from_raw(8), 0);
        let host_copy = || Keeps::HostCopy(Box::new(()));
        assert!(!moves.register_bytes(keeper.clone(), to, from, STAGED_BYTES + 1, host_copy()));
        for commit in 1..=STAGED_COPIES as u64 + 1 {
            let upload =
                moves.register_upload(keeper.clone(), to, from, 1 << 20, host_copy(), commit);
            assert!(!upload, "an upload from a host copy passed the bound");
        }
        assert!(!past_bound(&lock(&moves.waiting)));
        assert!(!moves.register_bytes(
            keeper.clone(),
            to,
            from,
            STAGED_BYTES,
            Keeps::Staging(Box::new(()))
        ));
        assert!(
            moves.register_bytes(keeper.clone(), to, from, 1, Keeps::Staging(Box::new(()))),
            "one staged byte past the bytes' bound"
        );
    }

    #[test]
    fn work_of_resources_gone_is_dropped() {
        let moves = FirstMoves::default();
        drop(registered(&moves, 1, None));
        drop(fill(&moves, 2, 64));
        assert!(moves.take(|_| false).held.is_empty());
        assert!(lock(&moves.waiting).is_empty());
    }

    #[test]
    fn an_image_over_host_pages_waits_for_a_batch_naming_its_key() {
        let moves = FirstMoves::default();
        let key = ResourceKey::new();
        let over_pages = registered(&moves, 1, Some(key));
        let own = registered(&moves, 2, None);
        let taken = moves.take(|_| false);
        let images: Vec<vk::Image> = (taken.held.iter())
            .flat_map(|held| match held {
                Held::Moves(_, moves, _) => moves.iter().map(|m| m.image).collect(),
                Held::Fill(..) | Held::Texels(..) => Vec::new(),
            })
            .collect();
        assert_eq!(
            images,
            [vk::Image::from_raw(2)],
            "only the image of its own"
        );
        drop(taken.submitted(None));
        assert_eq!(lock(&over_pages.moves).len(), 1, "it still waits");
        assert!(moves.take(|named| named != key).held.is_empty());
        let taken = moves.take(|named| named == key);
        assert_eq!(taken.held.len(), 1, "a batch naming it takes it");
        drop(taken.submitted(None));
        drop(own);
    }

    #[test]
    fn another_queue_waits_for_the_batch_until_it_completed() {
        let moves = FirstMoves::default();
        let image = registered(&moves, 1, None);
        let at = Point { queue: 1, value: 5 };
        drop(moves.take(|_| false).submitted(Some(at)));
        assert_eq!(
            moves.waits(0, |_| false),
            [(at, None)],
            "what owns its memory unknown"
        );
        assert!(
            moves.waits(1, |_| false).is_empty(),
            "its own queue orders it"
        );
        let later = Point { queue: 1, value: 7 };
        let other = registered(&moves, 2, None);
        drop(moves.take(|_| false).submitted(Some(later)));
        assert_eq!(
            moves.waits(2, |_| false),
            [(later, None)],
            "the last batch of the queue, whose work any command may reach"
        );
        assert!(moves.waits(0, |point| point.value <= 7).is_empty());
        assert!(
            lock(&moves.done_at).is_empty(),
            "a completed batch is forgotten"
        );
        assert!(
            moves.take(|_| false).submitted(Some(at)).is_empty(),
            "a submission that did no first work is no wait"
        );
        assert!(lock(&moves.done_at).is_empty());
        drop((image, other));
    }

    /// A submission holds its place in the order from its take until it was made or gave its
    /// work back, whether or not it took anything: no other submission takes, and no resource
    /// registers, meanwhile.
    #[test]
    fn a_submission_holds_its_place_in_the_order_until_it_was_made() {
        let moves = FirstMoves::default();
        let free = |moves: &FirstMoves| moves.order.try_lock().is_ok();
        let nothing = moves.take(|_| false);
        assert!(nothing.held.is_empty());
        assert!(!free(&moves), "held by one that took nothing");
        drop(nothing.submitted(None));
        assert!(free(&moves), "let go once made");
        let image = registered(&moves, 1, None);
        let taken = moves.take(|_| false);
        assert!(!free(&moves));
        drop(taken);
        assert!(free(&moves), "let go as it gave its work back");
        assert_eq!(lock(&image.moves).len(), 1);
        drop(image);
    }

    /// A batch on another queue waits for the last internal submission that did first work
    /// until it completed; one that did none is no wait.
    #[test]
    fn another_queue_waits_for_the_internal_submission_until_it_completed() {
        let moves = FirstMoves::default();
        let timeline = vk::Semaphore::from_raw(9);
        drop(moves.take(|_| false).submitted_internally(timeline, 1));
        assert_eq!(moves.internal_wait(), None, "it did no first work");
        let image = registered(&moves, 1, None);
        drop(moves.take(|_| false).submitted_internally(timeline, 2));
        assert_eq!(moves.internal_wait(), Some((timeline, 2)));
        moves.internal_completed(1);
        assert_eq!(
            moves.internal_wait(),
            Some((timeline, 2)),
            "an earlier one completed"
        );
        moves.internal_completed(2);
        assert_eq!(moves.internal_wait(), None);
        drop(image);
    }
}
