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
//!   copied over, single- or multisampled. An image that takes a rebind's new `VkImage` notes that
//!   one's (`rebind.rs`). The image (or stand-in) is registered weakly once it is shared
//!   ([`FirstMoves::register`]). An image over a dma-buf of its own pages (`dma_buf.rs`) starts
//!   `UNDEFINED` too, and its move is an acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT` for the
//!   device's queue family (`VK_QUEUE_FAMILY_IGNORED` where its images are concurrent among
//!   several), from `GENERAL` (which keeps its bytes, as Vulkan's foreign images keep theirs only
//!   through such an acquire) to `GENERAL` ([`FirstMove::acquired`]); it rests there.
//!   A new buffer whose bytes the host cannot write (`Private`, and a
//!   `Managed` buffer's device copy) notes its fill with zeros ([`FirstFill`]) the same way
//!   (`Buffer::clear_first`), and a new `Private` buffer made with bytes then the copy of a staging
//!   buffer's (`Buffer::fill_first_from`). So do a sparse texture's image (its move; an emulated
//!   one's zeros too) and its tile map's bytes, and the device's neutrals: a neutral image's move
//!   and its clear ([`FirstMove::cleared`]: it may be multisampled, which no copy writes), and a
//!   neutral texel buffer's fill (`neutral.rs`).
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
//!   command buffer of its own, so it adds none (`command_buffer/submit.rs`).
//! - **Staged copies**. `replace_region` of a texture whose `VkImage` is its own for good (in
//!   memory of its own, or a heap's own pages) and takes no host copies stages the bytes
//!   ([`TexelCopy`], [`FirstMoves::register_texels`]): the next submission copies them in at
//!   the start of its work, after the first work's `VkSubmitInfo`, each copy after the writes before
//!   it, and holds the image and the staging buffer until it completed.
//! - **Bounded**. Past [`STAGED_BYTES`] bytes, or [`STAGED_COPIES`] copies, staged for the next
//!   submission (staged bytes' copies into buffers, a `Managed` buffer's uploads, and staged texel
//!   copies), the call that staged the last makes a submission for them and waits
//!   (`FirstMoves::register_bytes` and `register_texels` answer when).
//! - **Over host pages.** An image bound to imported host memory starts in `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`), and its move "may perform read and write accesses
//!   on all memory bound to the image subresource range": a no-copy heap's texture bound to the
//!   heap's bytes (`texture/heap.rs`) takes such a move. A texture over a no-copy buffer's host
//!   pages (or fresh pages) takes one only where the device's check showed that move keeping every
//!   byte over such pages (`in_place.rs`); elsewhere it is a shadow, whose copies address its texels
//!   alone (`texture.rs`'s `storage_over_buffer`).
//! - **The work.** A barrier from every command before (memory other commands wrote), then the
//!   buffers' fills with zeros (copies of the device's zeros; on a device without unified memory,
//!   which would read those over the bus, one `vkCmdFillBuffer` each, counted:
//!   `WorkCounts::new_buffer_zero_fills`, `FirstFill::zeros`), then their copies of staged bytes
//!   after a barrier, then a
//!   barrier carrying each image's move (after the fills: an image placed in a new heap moves
//!   after the heap's fill, so the fill never overwrites what the move wrote), then the images'
//!   copies of zeros, then a barrier that makes them visible to every later command and to the
//!   host.
//!   Commands after it on the same `VkQueue`, in this submission or a later one, see each image
//!   in `GENERAL` and each fill done. On a device of several `VkQueue`s a submission on another
//!   one waits for the batch that did it until that batch completed ([`FirstMoves::waits`]): any
//!   of its commands may reach the resource unnamed.
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
//! - **Out of reach of a submission.** A texture's `VkImage` handed out (`Texture::vk_image`)
//!   rests in `GENERAL` as its contract says: handing it out does its work first, by an internal
//!   submission waited for.

use super::super::creation;
use super::super::device::lock;
use super::super::error::{internal, vulkan, Error, MemoryKind};
use super::super::exit_guard::{self, Call};
use super::super::heap_sets::HeldSlots;
use super::super::image_barrier::{image_barrier, ImageDst};
use super::super::memory::{Memory, MemoryClass, Placement};
use super::super::release;
use super::super::resource::ResourceKey;
use super::super::tracker::Point;
use super::super::work_counts::Counters;
use super::super::Device;
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

    /// The move, given back by a submission not made: no longer holding its pages.
    fn unheld(mut self) -> FirstMove {
        self.holding = None;
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

    /// The move, which also writes `patterned`'s pattern into its levels; its zeros then write
    /// only the levels after them. Its zeros' copies, if any, cover every level from the first.
    pub(crate) fn patterned(mut self, patterned: Patterned) -> FirstMove {
        self.pattern = Some(patterned);
        self
    }

    /// Whether it writes anything into its image: zeros or a pattern.
    fn writes(&self) -> bool {
        self.zero.is_some() || self.pattern.is_some()
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

    /// Record the move alone into `command_buffer`, of a family of `family`'s queue flags: its
    /// barrier, after every command before, then its writes of zero (and pattern), then the
    /// barrier that makes them visible to every command after and to the host
    /// (`device/staging.rs`: a texture moved on the staging queue as it is made, which no batch
    /// then finds waiting).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, for a queue family the image is
    /// shared with, of `family`'s flags; the image is live, in the layout its move starts from,
    /// and nothing uses it until the submission completed.
    pub(crate) unsafe fn record_alone(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        family: vk::QueueFlags,
    ) {
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        // SAFETY: the caller's contract; the image is live, in the layout the move starts from.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[self.barrier(family)],
            );
        }
        exit_guard::called(Call::CmdPipelineBarrier);
        if self.writes() {
            // SAFETY: as above; the image is in `GENERAL` now and has `TRANSFER_DST` usage where
            // it reads zero.
            unsafe { self.record_zero(device, command_buffer, family) };
        }
        transfer_barrier(device, command_buffer, false);
        exit_guard::called(Call::CmdPipelineBarrier);
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
}

/// The most bytes host calls' copies stage for the device's next submission before one is made
/// for them, waited for (the module docs).
const STAGED_BYTES: u64 = 256 << 20;

/// The most copies host calls stage for the device's next submission before one is made for
/// them, waited for (the module docs): a bound on the work waiting, whatever its bytes.
const STAGED_COPIES: usize = 4096;

/// A new buffer's fill: copies of the device's zeros onto `range` of `buffer`, or a fill of its
/// words with zeros (`word_fill.rs`), or copies of the bytes a staging buffer holds, or a
/// `Managed` buffer's upload from its host copy.
pub(crate) struct FirstFill {
    buffer: vk::Buffer,
    source: vk::Buffer,
    copies: Vec<vk::BufferCopy>,
    /// The words one `vkCmdFillBuffer` sets to zero in place of the copies (none then): on a
    /// device without unified memory, whose zeros (`Zeros`, host-visible) a copy would read over
    /// the bus ([`FirstFill::zeros`]).
    filled: Option<std::ops::Range<u64>>,
    /// What keeps the staging buffer `source` is, held until the submission that makes the fill
    /// completed; `None` where `source` is the device's zeros.
    staging: Option<Box<dyn Any + Send + Sync>>,
}

impl FirstFill {
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
        }
    }

    /// The zeros of `range` of `buffer` on `device`: one `vkCmdFillBuffer` of its words on a
    /// device without unified memory whose fills are `vkCmdFillBuffer`'s (`word_fill.rs`), where
    /// the range's ends are multiples of 4; else copies of the device's zeros, made now if they
    /// are not yet. A discrete GPU reads those host-visible zeros over the bus: on the RX 7900 XTX
    /// (RADV) 512 KiB of them took 66 us to copy and 11 us more in the barrier after it, where a
    /// fill of the same bytes took 0.7 us and 2.5 us. The copy stays where the device shares the
    /// host's memory, whose zeros lie in the memory the GPU reads anyway (not measured there),
    /// and where fills go as transfers (`DriverFact::FillCompiledPerCommandPool`), whose copies
    /// within the buffer need `TRANSFER_SRC` usage that a neutral buffer lacks.
    fn zeros(
        device: &Device,
        moves: &FirstMoves,
        buffer: vk::Buffer,
        range: std::ops::Range<u64>,
    ) -> Result<FirstFill, Error> {
        let words = range.start.is_multiple_of(4) && range.end.is_multiple_of(4);
        let discrete = !device.properties().has_unified_memory;
        if words && discrete && !crate::word_fill::by_transfers(device) {
            return Ok(FirstFill {
                buffer,
                source: vk::Buffer::null(),
                copies: Vec::new(),
                filled: Some(range),
                staging: None,
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
    /// `staging` keeping `source`.
    fn bytes(
        buffer: vk::Buffer,
        start: u64,
        (source, from): (vk::Buffer, u64),
        length: u64,
        staging: Box<dyn Any + Send + Sync>,
    ) -> FirstFill {
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
        }
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
        }
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
}

/// A new buffer whose fill waits: what a submission that takes the fill holds until it completes
/// is its memory alone (`Buffer::memory_hold`), so the buffer goes at the program's release, as
/// Metal frees it, its range of the address space and its range token with it.
pub(crate) trait Filled: Send + Sync {
    /// A hold on the memory the buffer lies in, kept and never read; `None` where it lies in
    /// none.
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>>;
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

/// A resource whose first work a submission took, held.
enum Held {
    Moves(Arc<dyn Fresh>, Vec<FirstMove>),
    /// The buffer, not kept, the hold on its memory, and its fill.
    Fill(Weak<dyn Filled>, Box<dyn Any + Send + Sync>, FirstFill),
    Texels(Arc<dyn Any + Send + Sync>, TexelCopy),
}

impl Held {
    /// What keeps the resource, or its memory (and a fill's staging buffer).
    fn keeper(self) -> Box<dyn Any + Send + Sync> {
        match self {
            // The moves keep the pages their images are bound to.
            Held::Moves(fresh, moves) => Box::new((fresh, moves)),
            Held::Fill(_, memory, fill) => match fill.staging {
                Some(staging) => Box::new((memory, staging)),
                None => memory,
            },
            Held::Texels(image, copy) => Box::new((image, copy.staging, copy.decode)),
        }
    }
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
    /// The point of the last batch on each `VkQueue` that did some, by queue.
    done_at: Mutex<Vec<Point>>,
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
/// copies of staged bytes into buffers, uploads, and texel copies.
fn past_bound(waiting: &[Waiting]) -> bool {
    let staged = waiting.iter().filter_map(|entry| match entry {
        Waiting::Fill(_, fill) => {
            (fill.staging.as_ref()).map(|_| fill.copies.iter().map(|copy| copy.size).sum())
        }
        Waiting::Texels(_, copy) => Some(copy.bytes),
        Waiting::Moves(_) => None,
    });
    let (copies, bytes) = staged.fold((0, 0), |(copies, bytes), size| (copies + 1, bytes + size));
    copies > STAGED_COPIES || bytes > STAGED_BYTES
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

    /// The new buffer `keeper`, whose bytes `range` of `buffer` the host cannot write, waits for
    /// its fill with zeros.
    pub(crate) fn register_fill(
        &self,
        device: &Device,
        keeper: Weak<dyn Filled>,
        buffer: vk::Buffer,
        range: std::ops::Range<u64>,
    ) -> Result<(), Error> {
        let fill = FirstFill::zeros(device, self, buffer, range)?;
        let _order = self.registration_place();
        lock(&self.waiting).push(Waiting::Fill(keeper, fill));
        Ok(())
    }

    /// The buffer `keeper`, whose bytes the host cannot write, waits for the copy of `length`
    /// bytes of `source` (a buffer and its byte the copy starts at), which `staging` keeps, onto
    /// its `VkBuffer` `buffer` from byte `start`: after every fill with zeros waiting, its own
    /// among them where it is new, and after the copies registered before it. Whether the copies
    /// staged now passed their bound, so the caller makes a submission for them.
    pub(crate) fn register_bytes(
        &self,
        keeper: Weak<dyn Filled>,
        (buffer, start): (vk::Buffer, u64),
        source: (vk::Buffer, u64),
        length: u64,
        staging: Box<dyn Any + Send + Sync>,
    ) -> bool {
        let fill = FirstFill::bytes(buffer, start, source, length, staging);
        let _order = self.registration_place();
        let mut waiting = lock(&self.waiting);
        waiting.push(Waiting::Fill(keeper, fill));
        past_bound(&waiting)
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

    /// Take the work a submission does: all of it, but the moves of images bound to host pages
    /// whose key `named` does not name (an internal submission's; a batch names every key, the
    /// module docs). What goes is dropped once the lock is free.
    ///
    /// The submission's place in the order is held until it was made or the work given back
    /// ([`Taken`]): every submission takes, whether or not anything waits.
    pub(crate) fn take(&self, named: impl Fn(ResourceKey) -> bool) -> Taken<'_> {
        self.take_with(named, || None)
    }

    /// As [`take`](Self::take), for a submission whose commands may bind the heap sets (a
    /// batch's): where registrations do not wait for their place ([`FirstMoves::ordered`]), it
    /// also holds the texture slots' writes (`heap_sets::hold`, which `hold` makes) from before it
    /// takes until its submission was made or the work given back (the module docs).
    pub(crate) fn take_holding(
        &self,
        named: impl Fn(ResourceKey) -> bool,
        hold: impl FnOnce() -> HeldSlots,
    ) -> Taken<'_> {
        self.take_with(named, || (!self.ordered).then(hold))
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

    /// As [`take`](Self::take), holding the slots `hold` gives once its place is taken.
    fn take_with(
        &self,
        named: impl Fn(ResourceKey) -> bool,
        hold: impl FnOnce() -> Option<HeldSlots>,
    ) -> Taken<'_> {
        let order = self.submission_place();
        let slots = hold();
        let mut held = Vec::new();
        let mut left: Vec<Box<dyn Any + Send + Sync>> = Vec::new();
        {
            let mut waiting = lock(&self.waiting);
            if waiting.is_empty() {
                return Taken {
                    moves: self,
                    held,
                    _slots: slots,
                    _order: order,
                };
            }
            let mut kept = Vec::new();
            for entry in waiting.drain(..) {
                match entry {
                    Waiting::Moves(weak) => {
                        let Some(fresh) = weak.upgrade() else {
                            continue;
                        };
                        if fresh.rebindable().is_some_and(|key| !named(key)) {
                            kept.push(Waiting::Moves(weak));
                            left.push(Box::new(fresh));
                            continue;
                        }
                        let moves: Vec<FirstMove> = fresh
                            .take()
                            .into_iter()
                            .filter_map(FirstMove::held)
                            .collect();
                        if moves.is_empty() {
                            left.push(Box::new(fresh));
                        } else {
                            held.push(Held::Moves(fresh, moves));
                        }
                    }
                    Waiting::Fill(weak, fill) => {
                        let Some(buffer) = weak.upgrade() else {
                            continue;
                        };
                        // The submission holds the buffer's memory; the buffer goes as the
                        // program lets go of it, once the lock is free.
                        if let Some(memory) = buffer.memory() {
                            held.push(Held::Fill(weak, memory, fill));
                        }
                        left.push(Box::new(buffer));
                    }
                    // An image gone takes its copy and staging buffer with it.
                    Waiting::Texels(weak, copy) => {
                        if let Some(image) = weak.upgrade() {
                            held.push(Held::Texels(image, copy));
                        }
                    }
                }
            }
            waiting.extend(kept);
        }
        drop(left);
        Taken {
            moves: self,
            held,
            _slots: slots,
            _order: order,
        }
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

    /// The points a submission on `VkQueue` `queue` waits for: of the last batch on each other
    /// one that did first work, where `completed` says it may not have completed.
    pub(crate) fn waits(&self, queue: usize, completed: impl Fn(Point) -> bool) -> Vec<Point> {
        let mut done_at = lock(&self.done_at);
        done_at.retain(|&point| !completed(point));
        done_at
            .iter()
            .copied()
            .filter(|point| point.queue != queue)
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
    /// The texture slots' writes it holds ([`FirstMoves::take_holding`]), made as this drops,
    /// before its place goes: the field before `_order`, which drops after it.
    _slots: Option<HeldSlots>,
    /// The submission's place in the order ([`FirstMoves::take`]), let go as this drops.
    _order: MutexGuard<'a, ()>,
}

impl Taken<'_> {
    /// Whether it took first work ([`record`](Self::record)'s): a move or a fill.
    pub(crate) fn has_first(&self) -> bool {
        (self.held.iter()).any(|held| matches!(held, Held::Moves(..) | Held::Fill(..)))
    }

    /// Whether it took host calls' staged copies ([`record_staged`](Self::record_staged)'s).
    pub(crate) fn has_staged(&self) -> bool {
        (self.held.iter()).any(|held| matches!(held, Held::Texels(..)))
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
            Held::Texels(_, copy) => Some(copy),
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
    /// heap's included), then their copies of staged bytes, after the zeros, then a barrier
    /// carrying every image's move, after the fills (an image placed in a new heap moves after
    /// the heap's fill), then the images' copies of zeros, then the barrier that makes every write
    /// visible to every command after and to the host.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, for one of the device's queue
    /// families, of `family`'s flags; the submission that runs it is made before any that uses
    /// one of the resources, on the same `VkQueue` or waited for.
    pub(crate) unsafe fn record(
        &self,
        (device, counts): (&Vk, &Counters),
        command_buffer: vk::CommandBuffer,
        family: vk::QueueFlags,
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
                Held::Moves(_, moves) => moves.as_slice(),
                Held::Fill(..) | Held::Texels(..) => &[],
            })
        };
        if !fills.is_empty() {
            // SAFETY: the caller's contract.
            unsafe {
                let flags = vk::DependencyFlags::empty();
                device.cmd_pipeline_barrier(command_buffer, all, all, flags, &written(), &[], &[]);
            }
            exit_guard::called(Call::CmdPipelineBarrier);
        }
        let (zeros, staged): (Vec<&FirstFill>, Vec<&FirstFill>) = fills
            .iter()
            .copied()
            .partition(|fill| fill.staging.is_none());
        for (at, fill) in zeros.iter().chain(&staged).enumerate() {
            if at == zeros.len() && at > 0 {
                // SAFETY: the caller's contract.
                unsafe {
                    let flags = vk::DependencyFlags::empty();
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        all,
                        all,
                        flags,
                        &written(),
                        &[],
                        &[],
                    );
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
            // SAFETY: as above; every copy lies within its source, the device's zeros (live and
            // `ZERO_BYTES` long) or a staging buffer the fill keeps, and within the buffer,
            // held, which has `TRANSFER_DST` usage as every buffer of the runtime's.
            unsafe {
                device.cmd_copy_buffer(command_buffer, fill.source, fill.buffer, &fill.copies);
            }
            exit_guard::called(Call::CmdCopyBuffer);
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
        for first in moves().filter(|first| first.writes()) {
            // SAFETY: as above; the image is in `GENERAL` now and has `TRANSFER_DST` usage where
            // it reads zero.
            unsafe { first.record_zero(device, command_buffer, family) };
            copied = true;
        }
        if copied {
            transfer_barrier(device, command_buffer, false);
            exit_guard::called(Call::CmdPipelineBarrier);
        }
    }

    /// The submission that records the work was made; on a device of several `VkQueue`s by a
    /// batch at `at` (`None` for a submission waited for, or a device of one). What the
    /// submission holds until it completes.
    pub(crate) fn submitted(mut self, at: Option<Point>) -> Vec<Box<dyn Any + Send + Sync>> {
        if let (Some(at), false) = (at, self.held.is_empty()) {
            let mut done_at = lock(&self.moves.done_at);
            match done_at.iter_mut().find(|point| point.queue == at.queue) {
                Some(point) => point.value = point.value.max(at.value),
                None => done_at.push(at),
            }
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

impl Drop for Taken<'_> {
    fn drop(&mut self) {
        if self.held.is_empty() {
            return;
        }
        let mut given = Vec::new();
        for held in self.held.drain(..) {
            match held {
                Held::Moves(fresh, moves) => {
                    fresh.give_back(moves.into_iter().map(FirstMove::unheld).collect());
                    given.push(Waiting::Moves(Arc::downgrade(&fresh)));
                }
                Held::Fill(buffer, _, fill) => {
                    given.push(Waiting::Fill(buffer, fill));
                }
                Held::Texels(image, copy) => {
                    given.push(Waiting::Texels(Arc::downgrade(&image), copy));
                }
            }
        }
        lock(&self.moves.waiting).extend(given);
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
            Box::new(staging.clone()),
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
            moves.register_bytes(keeper.clone(), to, from, 4, Box::new(()))
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
                Held::Moves(_, moves) => moves.iter().map(|m| m.image).collect(),
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
        assert_eq!(moves.waits(0, |_| false), [at]);
        assert!(
            moves.waits(1, |_| false).is_empty(),
            "its own queue orders it"
        );
        let later = Point { queue: 1, value: 7 };
        let other = registered(&moves, 2, None);
        drop(moves.take(|_| false).submitted(Some(later)));
        assert_eq!(
            moves.waits(2, |_| false),
            [later],
            "the last batch of the queue"
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
