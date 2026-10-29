//! A shadowed texture's writes merged into its buffer's bytes (`shadow.rs`), where a copy back of
//! its shadow would write bytes Metal keeps:
//!
//! - **Two ways**: one command that writes through a shadowed texture and also writes the same
//!   buffer's bytes another way (through the buffer, a texture buffer, or another texture over
//!   them). A copy back would replace the other write's bytes with the shadow's, where Metal keeps
//!   both writes when they touch different bytes (Metal probe cases `k.two_ways_disjoint` and
//!   `k.two_ways_gap`, M3 Max and M4). Merged before the encoder's next command.
//! - **Merged back** (`ShadowCopies::merges_back`: a buffer's bytes but a tracked `Private`
//!   buffer's): a copy back would write the texels the encoder's commands left with what the copy
//!   in read, over another writer's writes to them while the command buffer ran (the CPU's, or
//!   another queue's commands' over an untracked buffer's), where Metal writes only the texels a
//!   command writes (`shadow.rs`). Merged in place of each copy back.
//!
//! So the bytes the commands changed through the texture are merged into the buffer's instead:
//!
//! - **Kept.** Before the command, after its copies in, the shadow's texels are copied into the
//!   *kept* part of a scratch buffer, laid out as in the buffer from the texture's first byte. A
//!   heap's tiled texture of one sample and colour aspects alone, kept right after its copy in
//!   with no copy of its shadow between, copies the staging buffer that copy filled with the
//!   heap's bytes: the bytes a move of its texels out of the image would leave there, without
//!   the move (`ShadowCopies::record_kept_from_staging`, `WorkCounts::shadow_keeps_from_staging`).
//! - **Merged in place**, where the texture's first byte is a whole word of its buffer, its last
//!   byte's word lies in the buffer, and the bytes lie in pages when the merge is planned (not
//!   detached, `detach.rs`): the shadow's texels are copied into the *changed* part, and the merge
//!   kernel binds the buffer's own bytes as its current buffer, from the texture's first byte
//!   rounded down to 256 (the largest `minStorageBufferOffsetAlignment`), its first word pushed,
//!   to its last byte's word. A word whose kept and changed words are equal is neither read nor
//!   written: the bytes the commands left, those between the rows and those past the last texel
//!   in its word keep whatever another writer wrote there meanwhile. In a word they changed, each
//!   byte they changed becomes the texture's, and the others keep the buffer's: another writer's
//!   write to the word's other bytes during the merge itself is a race the merge does not order.
//! - **Merged through a copy** elsewhere (a texture's first byte inside a word, through a staging
//!   buffer, or a last word past its buffer; never recorded over detached bytes): the shadow's
//!   texels are copied into the *changed* part, and the buffer's bytes from the texture's first to
//!   its last into the *current* part. One dispatch of the merge kernel then takes, in each byte of
//!   the current part, the changed part's byte where it differs from the kept one, and the current
//!   part is copied back into the buffer: another writer's write to the texture's bytes between
//!   that copy and the copy back is undone (`docs/FAST-PATHS.md`'s known gaps).
//! - **Cut.** Where a merge after commands that write through the texture goes in place over one
//!   region's texels, the keep holds the rectangle the encoder's commands write in, where they
//!   bound it (`ShadowUse::area`: a pass's attachment it loads, inside its draws' scissors), else
//!   every texel, each row widened to whole words of the buffer. The keep and the changed copy
//!   move that rectangle's texels, row after row, and the merge binds the buffer's rows it lies
//!   in, in bands of rows no binding of which passes `maxStorageBufferRange` and no dispatch of
//!   which passes `maxComputeWorkGroupCount[0]` workgroups, each band's kept rows from a multiple
//!   of 256 bytes.
//!   No command since the keep wrote a texel outside it: a later command that writes outside it
//!   takes the merge of what was kept first, and a keep of the rectangle bounding both
//!   (`ShadowSync::before`). (`WorkCounts::shadow_texels_kept`.)
//! - **Fetched.** Where the shadow has its unsigned-integer view (`shadow_copy_back.rs`: on a
//!   device that runs a copy back faster as a compute dispatch,
//!   `DriverFact::ShadowCopyBackDispatchFaster`, a one-slice image of 4-, 8- or 16-byte texels
//!   copied in place), or else, a one-slice `BGRA8Unorm` or `RGBA8Unorm` image copied in place,
//!   a view of its own format (`UnormView`, sampled usage on the image), a cut merge copies
//!   nothing of the texels the commands left: its kernel (`bind::shadow_merge_image_module`)
//!   fetches them from the shadow through the view, the same words a copy of them would hold
//!   (through the unsigned normalized view, each byte rebuilt from its value, which Vulkan
//!   defines as the byte over 255: `bind::ShadowMergeRead`), and compares them with the kept
//!   part's, so the scratch is the kept part alone (`WorkCounts::shadow_merges_fetched`). Every
//!   other format, sRGB and 16-bit float among them, copies them (`unorm_read`).
//! - **Kept in an image.** Where a merge fetching through the unsigned-integer view takes no
//!   keep in its copy in (the device copies in by no dispatch, no
//!   `DriverFact::ShadowCopyInDispatchFaster`), its keep is a `vkCmdCopyImage` of the cut
//!   rectangle from the shadow into a *kept image* at the same coordinates
//!   (`texture/kept_image.rs`: the shadow's extent, of its view's unsigned-integer format, a
//!   format of its compatibility class, so the copy holds the texels' bits), and its kernel
//!   (`bind::shadow_merge_kept_image_module`) fetches the kept texels from it beside the shadow's:
//!   no scratch (`WorkCounts::shadow_merges_kept_in_images`). On the Iris Xe of RPL-P such a copy
//!   takes under a third of the GPU time of a copy or dispatch of the same texels into a buffer's
//!   rows (`docs/FAST-PATHS.md`).
//! - **Dispatched copies.** Where the shadow has its unsigned-integer view, the copies of its
//!   texels into the kept part, and into the changed part where the merge does not fetch, are that
//!   dispatch, each storing the bytes the copy would (`WorkCounts::shadow_merge_copies_dispatched`).
//!   Where the view also stores (`DriverFact::ShadowCopyInDispatchFaster`), a cut keep in place
//!   whose texels the copy in before it moves, and no copy of the shadow between them, takes that
//!   copy in with it: one dispatch reads the texels' words from the buffer and stores them in the
//!   shadow, through the view, and in the kept part (`shadow_copy_in.rs`,
//!   `WorkCounts::shadow_copies_in_dispatched`).
//! - **The scratch** is the command's, or for a merge in place of a copy back the encoder's, kept
//!   in again at each keep after a merge (each merge ends with a barrier to everything after it),
//!   retained until its command buffer completes: one, two or three parts of the texture's bytes
//!   (a cut's rows) rounded up to 256, in one buffer where the device's buffers take it (else one
//!   each), or a kept image in place of them. A cut merge's copies write its parts' rows back to
//!   back, every byte its kernel reads, and a heap's tiled texture's merge in place copies the
//!   texture's bytes whole (`texture/heap_tiled.rs`), the bytes its kernel reads, so neither
//!   scratch is zeroed: one of those its shadow's
//!   earlier merges were done with, kept once their command buffers completed (at most two,
//!   `ScratchSpares`; `WorkCounts::shadow_merge_scratches_reused`), else a new one. A kept image
//!   is its queue's, of any merge of the queue of the same view and extent once the encoder that
//!   took it ended, whatever its shadow, without waiting for the command buffers that used it,
//!   ordered by its barriers and its tracked uses (`texture/kept_image.rs`,
//!   `WorkCounts::shadow_merge_kept_images_made` counting those made), its keep writing each
//!   texel its kernel reads. Any other merge's scratch is new and all zero, so the bytes between
//!   the texture's rows, which no copy of the shadow writes, are the same in the kept and changed
//!   parts (`WorkCounts::shadow_merge_scratches_zeroed`).
//! - **Whole blocks** for a command that replaces every texel of a shadow over logged host pages
//!   (`shadow.rs`, **Overwrites over logged pages**), and for a render pass that loads and stores
//!   its color attachment inside its draws' scissors (`shadow.rs`, **Stored attachments**), cut to
//!   the texels' rows: the kernel's `bind::ShadowMergeStores::Blocks` form stores each 256-byte
//!   block of the rows that holds a changed word whole, without gaps, and no other block
//!   (`bind::shadow_merge_module`'s module docs; a GPU storing into host memory over PCIe pays
//!   hundreds of times for gaps between a subgroup's words). Every word of those rows is bytes the
//!   command wrote, or bytes Metal's own store of the pass writes back, so a word it left is
//!   stored with the value the merge reads, Metal's answer. Every other merge stores the changed
//!   words alone (`ShadowMergeStores::Words`): its rows may hold bytes no command wrote, which
//!   another writer may write meanwhile (`WorkCounts::shadow_merge_dispatches_in_blocks`).
//! - **The kernel** (`bind::shadow_merge_module`, one per device and form, `owned_kernel.rs`): four
//!   4-byte words an invocation, each a workgroup's invocations apart from the next, so one
//!   store's words are consecutive across the invocations that run together (its module docs,
//!   which name the device that charges hundreds of times for the gaps otherwise), the parts
//!   bound as its kept, changed and current storage
//!   buffers, in windows of at most `maxStorageBufferRange` bytes and
//!   `maxComputeWorkGroupCount[0]` groups, one dispatch each; where it fetches
//!   (`bind::shadow_merge_image_module`, one per read of a texel), four words of a row an
//!   invocation, spread alike, the view, the kept part (or the kept image's view) and the bytes
//!   bound, in bands of rows each of
//!   whose bindings stays within `maxStorageBufferRange` and whose rows within
//!   `maxComputeWorkGroupCount[1]`. The dispatches of a merge and the words they examine are
//!   counted (`WorkCounts::shadow_merge_dispatches_planned`,
//!   `WorkCounts::shadow_merge_words_examined`).
//!
//! A byte a command wrote both ways is the texture's where its value changed: Metal leaves such a
//! race unordered. It costs the scratch (or a kept image), two copies of the shadow (one where the
//! merge fetches; dispatched where the copy back is, but into a kept image) and, through a copy,
//! one of the bytes, and the merge's dispatches, in place of one copy back. A merge of a command
//! writing two ways is announced once per device (`shadow-merge`, performance); one in place of a
//! copy back is how such bytes go back, announced with nothing.

use super::buffer::Buffer;
use super::degradation::Degradation;
use super::device::lock;
use super::error::{internal, Error};
use super::image_plan::{ImageFacts, ImagePlan};
use super::linear_merge::gcd;
use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use super::resource::ResourceKey;
use super::shadow::{ShadowCopies, ShadowSource, ShadowTarget};
use super::shadow_copy_back::{self, CopyBack, WordView};
use super::shadow_copy_in::CopyIn;
use super::texture::heap_tiled::SampleMoves;
use super::texture::kept_image::{KeptImage, KeptImages, KeptLease, TakenImage};
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::{HazardTrackingMode, ResourceOptions};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    ShadowMergeImageModule, ShadowMergeModule, ShadowMergeRead, ShadowMergeStores, ShadowTexelView,
};
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageUsageFlags};
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// Bytes one workgroup merges: `ShadowMergeModule::WORDS` words per invocation.
const GROUP_BYTES: u64 = 4 * (ShadowMergeModule::WORDS * ShadowMergeModule::LOCAL_SIZE) as u64;
/// What each part of the scratch is rounded up to: a multiple of every
/// `minStorageBufferOffsetAlignment` and of every texel's size.
const PART_ALIGNMENT: u64 = 256;

/// The most scratch buffers a shadow keeps for its next merges ([`ScratchSpares`]): one a frame's
/// command buffer holds while the next frame's is encoded, and one more.
const KEPT_SCRATCHES: usize = 2;

/// The scratch buffers a shadow's merges are done with, kept for its next merges: each comes back
/// as its merge goes, once the command buffers that recorded it completed (they retain it), so no
/// command a later merge records overlaps one of an earlier merge in it. At most
/// [`KEPT_SCRATCHES`], the smallest going first.
#[derive(Clone, Default)]
pub(super) struct ScratchSpares(Arc<Mutex<Vec<Buffer>>>);

impl ScratchSpares {
    /// The smallest kept buffer of at least `length` bytes, taken.
    fn take(&self, length: u64) -> Option<Buffer> {
        let spares = &mut *lock(&self.0);
        let at = (spares.iter().enumerate())
            .filter(|(_, buffer)| buffer.length() >= length)
            .min_by_key(|(_, buffer)| buffer.length())
            .map(|(at, _)| at)?;
        Some(spares.swap_remove(at))
    }

    /// Keep `buffer` for a later merge, the smallest kept going past [`KEPT_SCRATCHES`].
    fn give(&self, buffer: Buffer) {
        let mut buffers = lock(&self.0);
        buffers.push(buffer);
        let gone = match buffers.len() > KEPT_SCRATCHES {
            true => (buffers.iter().enumerate())
                .min_by_key(|(_, buffer)| buffer.length())
                .map(|(at, _)| at)
                .map(|at| buffers.swap_remove(at)),
            false => None,
        };
        drop(buffers);
        drop(gone);
    }
}

/// The buffers a merge's scratch lies in, or its kept image, held until its command buffers
/// completed.
enum Held {
    /// One buffer, back to its shadow's spares as the merge goes.
    Spare(Buffer, ScratchSpares),
    /// A buffer per part, where the device's buffers take no single one.
    Apart(Vec<Buffer>),
    /// The kept image of a merge that keeps the texels in one, its queue's (`KeptImages`); no
    /// scratch.
    Image(KeptImage),
}

/// Where the keep holds the shadow's texels before the commands (the module docs).
#[derive(Clone, Copy)]
enum KeptIn {
    /// The scratch's kept part, row after row (or laid out as in the buffer).
    Part(Part),
    /// A kept image of the unsigned-integer format of `texel`, at their own coordinates: `copy`
    /// copies the cut rectangle of the `shadow` into the `kept` image.
    Image {
        shadow: vk::Image,
        kept: vk::Image,
        copy: vk::ImageCopy,
        texel: ShadowTexelView,
    },
}

/// One part of the scratch: its `VkBuffer` and where it starts there.
#[derive(Clone, Copy)]
struct Part {
    raw: vk::Buffer,
    at: u64,
}

/// The bytes a merge writes: the texture's, in its buffer.
struct Merged {
    buffer: vk::Buffer,
    key: ResourceKey,
    bytes: Range<u64>,
    /// Where the texture's first byte lies in `buffer`.
    first: u64,
}

/// The rectangle of a shadow's texels a merge is cut to (the module docs).
struct Cut {
    /// `[x0, y0, x1, y1)`, its rows' first and last bytes whole words of the buffer.
    area: [u32; 4],
    /// Whether it is every texel, the cut made for a merge whose commands bound no area: its
    /// keep holds every texel, as one not cut does.
    whole: bool,
    /// The shadow's image, and the copy of the rectangle's texels into a part from its start, row
    /// after row (`bufferOffset` 0, `bufferRowLength` the rectangle's width).
    image: vk::Image,
    region: vk::BufferImageCopy,
}

/// How a merge reads the shadow's texels after the commands (the module docs).
enum Changed {
    /// Copied into the changed part (by the copy back's dispatch where the device runs one faster
    /// and the shadow takes it, `shadow_copy_back.rs`), which `bind::shadow_merge_module` reads.
    Copied {
        part: Part,
        dispatch: Option<CopyBack>,
    },
    /// Fetched from the shadow through its unsigned-integer view, or its own 8-bit unsigned
    /// normalized view, by `bind::shadow_merge_image_module`: no copy.
    Fetched,
}

/// The scratch and the dispatches of one merge (the module docs).
pub(super) struct ShadowMerge {
    /// The texture's bytes.
    merged: Merged,
    /// The rectangle the keep, the changed copy and the merge are cut to; `None` for every texel.
    cut: Option<Cut>,
    /// The texels the keep copies.
    texels: u64,
    /// Where the keep holds them.
    kept_in: KeptIn,
    /// How the merge reads the texels after the commands.
    changed: Changed,
    /// The current part, where the merge goes through a copy; `None` where it merges in place.
    current: Option<Part>,
    /// The kept part's (or kept image's) identity and range, for the tracker: kept before the
    /// command, read by the merge after it.
    kept: (ResourceKey, bool, UseRange),
    /// The keep's dispatch, where the device runs a copy back faster as one
    /// (`shadow_copy_back.rs`) and the shadow takes it into the kept part; `None` where it is a
    /// copy.
    keep_dispatch: Option<CopyBack>,
    /// The buffers the parts lie in, or the kept image, held until the command buffer completes.
    held: Held,
    kernel: OwnedKernel,
    /// Whether it is made for a command whose merge stores whole blocks (`MergeFor::blocks`: one
    /// that replaces every texel, or a render pass that stores the tiles of its area): its kernel
    /// stores whole blocks where it is cut to the texels' rows (the module docs), so another
    /// command's keep takes a merge of its own.
    blocks: bool,
    /// One set 0 per window, each binding the window of the parts (and the shadow's view, where
    /// the merge fetches, and the kept image's, where it keeps in one) and of the current bytes:
    /// pushed where the device pushes (`OwnedBinds`).
    binds: OwnedBinds,
    /// Each window's workgroups (along x and y) and push bytes.
    dispatches: Vec<([u32; 2], [u8; 16])>,
}

impl ShadowMerge {
    /// The scratch and the kernel's sets for a merge of the texture of `copies` on `device`, of a
    /// command that writes its bytes two ways where `two_ways` (announced once per device), else
    /// in place of a copy back; cut to the texels inside `area` (`[x0, y0, x1, y1)`), widened
    /// to whole words of the buffer, where given and the merge goes in place over one region's
    /// texels (the module docs), else of every texel. A kept image is one of `images`, its
    /// queue's, held by the lease returned, which the encoder keeps to its end, else one made
    /// and kept there. Refused as `new_buffer` refuses the scratch.
    pub(super) fn new(
        device: &Device,
        copies: &ShadowCopies,
        (two_ways, blocks, area): (bool, bool, Option<[u32; 4]>),
        images: &KeptImages,
    ) -> Result<(ShadowMerge, Option<KeptLease>), Error> {
        let (
            ShadowSource::Bytes {
                buffer, key, bytes, ..
            },
            Some(first),
        ) = (&copies.source, copies.first_byte())
        else {
            // A command writing an image two ways is refused before (`ShadowSync::before`), and
            // only a buffer's bytes merge back (`ShadowCopies::merges_back`).
            return Err(internal("a merge of an image's shadow"));
        };
        let merged = Merged {
            buffer: buffer.raw(),
            key: *key,
            bytes: bytes.clone(),
            first,
        };
        let span = merged.bytes.end - merged.bytes.start;
        // A window binds the bytes only while they lie in pages: a detached buffer's `VkBuffer` is
        // the device's stand-in, smaller than any window, and a binding past a buffer's size is
        // invalid (VUID-VkDescriptorBufferInfo-range-00342). The encoder records nothing of a
        // command over detached bytes (`EncoderCore::skips`), so their merge, planned through a
        // copy, whose windows bind the scratch alone, is never recorded.
        // In place, the merge reads and writes whole words, up to the last byte's word, which
        // must lie in the buffer's bytes.
        let rounded = span.next_multiple_of(4);
        let in_place = copies.merges_back()
            && first.is_multiple_of(4)
            && first + rounded <= buffer.end()
            && merged.buffer != vk::Buffer::null();
        let limits = &device.properties().description.limits;
        let range = u64::from(limits.max_storage_buffer_range);
        let [groups, groups_y] =
            [0, 1].map(|axis| u64::from(limits.max_compute_work_group_count[axis]));
        // A merge in place whose commands bound no area is cut to every texel where it can be:
        // its parts hold the texels row after row, as a cut's do, and need no zero (below). A cut
        // fetches the texels after the commands through the shadow's unsigned-integer view, or
        // its own 8-bit unsigned normalized view, where it has one (`fetch_view`), and the
        // rectangle lies within the kernel's origin and workgroups.
        let cut = in_place
            .then(|| cut_of(copies, area))
            .flatten()
            .and_then(|(cut, geometry)| {
                let view =
                    (fetch_view(copies)).filter(|&(_, read)| fetches(&cut, read, groups, groups_y));
                let rows = match view {
                    Some(_) => groups_y,
                    None => groups * GROUP_BYTES / geometry.row_bytes,
                };
                Some((cut, geometry, geometry.band(range, rows)?, view))
            });
        let part = match &cut {
            Some((_, geometry, ..)) => geometry.rows * geometry.row_bytes,
            None => rounded,
        }
        .next_multiple_of(PART_ALIGNMENT);
        let fetched = cut.as_ref().and_then(|(.., view)| *view);
        // A merge that fetches through the unsigned-integer view keeps the texels in an image of
        // its format at their own coordinates, copied from the shadow, where the copy in never
        // takes the keep (the device copies in by no dispatch, `shadow_copy_in.rs`): no scratch.
        // One that reads the shadow's own unsigned normalized view keeps them in the scratch.
        let kept_image = match (fetched, kept_extent(copies)) {
            (Some((_, ShadowMergeRead::Words(texel))), Some(extent))
                if !shadow_copy_back::dispatches_in(device) =>
            {
                Some((texel, extent))
            }
            _ => None,
        };
        let roles: u64 = match (in_place, fetched, kept_image) {
            (true, Some(_), Some(_)) => 0,
            (true, Some(_), None) => 1,
            (true, None, _) => 2,
            (false, ..) => 3,
        };
        let options = ResourceOptions::private();
        let place = |buffer: &Buffer, at: u64| {
            let (raw, at) = buffer.at(at);
            Part { raw, at }
        };
        let length = roles * part;
        let counts = &device.shared().counts;
        // The kept part is the first `part` bytes of the first buffer either way.
        let mut lease = None;
        let (held, placed) = if let Some((texel, extent)) = kept_image {
            let image = match images.take(texel, extent) {
                Some(TakenImage {
                    image,
                    lease: taken,
                    followed,
                }) => {
                    counts.shadow_merge_kept_in_image(false, followed);
                    lease = Some(taken);
                    image
                }
                None => {
                    // Made with no lock held, then kept.
                    let image = device.new_kept_image(texel, extent)?;
                    counts.shadow_merge_kept_in_image(true, false);
                    lease = Some(images.keep(image.clone()));
                    image
                }
            };
            (Held::Image(image), Vec::new())
        } else if length <= device.properties().max_buffer_length {
            // A cut's copies write every byte of its kept and changed parts the kernel reads (its
            // rows of whole words, back to back), and so do a heap's tiled texture's in place
            // (the texture's bytes whole, which the kernel reads alone), so their scratch needs
            // no zero: one of the shadow's spares, else a new one unzeroed. Every other merge's
            // scratch is new and zero.
            let written = cut.is_some() || (in_place && copies.copies_out_whole(rounded));
            let spares = copies.merge_scratch.clone();
            let one = match (written, written.then(|| spares.take(length)).flatten()) {
                (true, Some(spare)) => {
                    counts.shadow_merge_scratch_reused();
                    spare
                }
                (true, None) => device.unzeroed_buffer(length, HazardTrackingMode::Default)?,
                (false, _) => {
                    counts.shadow_merge_scratch_zeroed();
                    device.own_buffer(length, options)?
                }
            };
            let placed: Vec<Part> = (0..roles).map(|role| place(&one, role * part)).collect();
            (Held::Spare(one, spares), placed)
        } else {
            let held = (0..roles)
                .map(|_| device.own_buffer(part, options))
                .collect::<Result<Vec<Buffer>, Error>>()?;
            counts.shadow_merge_scratch_zeroed();
            let placed = held.iter().map(|buffer| place(buffer, 0)).collect();
            (Held::Apart(held), placed)
        };
        let (kept_in, changed, current) = match (&held, &placed[..]) {
            (Held::Image(image), []) => {
                let (
                    Some((
                        Cut {
                            image: shadow,
                            region,
                            ..
                        },
                        ..,
                    )),
                    Some((texel, _)),
                ) = (&cut, kept_image)
                else {
                    return Err(internal("an uncut merge keeping its texels in an image"));
                };
                // The rectangle at the same coordinates of the kept image's one level and slice.
                let copy = vk::ImageCopy {
                    src_subresource: region.image_subresource,
                    src_offset: region.image_offset,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offset: region.image_offset,
                    extent: region.image_extent,
                };
                let kept_in = KeptIn::Image {
                    shadow: *shadow,
                    kept: image.raw(),
                    copy,
                    texel,
                };
                (kept_in, None, None)
            }
            (_, &[kept]) => (KeptIn::Part(kept), None, None),
            (_, &[kept, changed]) => (KeptIn::Part(kept), Some(changed), None),
            (_, &[kept, changed, current]) => (KeptIn::Part(kept), Some(changed), Some(current)),
            _ => return Err(internal("a merge's scratch of another number of parts")),
        };
        let kept_part = |buffer: Option<&Buffer>| {
            let buffer = buffer.ok_or_else(|| internal("a merge without its scratch"))?;
            let (identity, range) = buffer.tracked(0..part);
            Ok::<_, Error>((identity.key, identity.tracked, UseRange::Bytes(range)))
        };
        let kept = match &held {
            Held::Spare(one, _) => kept_part(Some(one))?,
            Held::Apart(held) => kept_part(held.first())?,
            Held::Image(image) => {
                let identity = image.identity();
                let range = UseRange::Image {
                    aspects: vk::ImageAspectFlags::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                };
                (identity.key, identity.tracked, range)
            }
        };
        let u32_of = |value: u64| {
            u32::try_from(value).map_err(|_| internal("a merge window past u32 words"))
        };
        let mut sets = Vec::new();
        let mut dispatches = Vec::new();
        // The words the dispatches examine (`WorkCounts::shadow_merge_words_examined`).
        let mut examined = 0;
        // Each window's set: the shadow's view, where the merge fetches, and the kept image's,
        // where it keeps in one, then its buffers.
        let kept_view = match &held {
            Held::Image(image) => Some(image.view()),
            Held::Spare(..) | Held::Apart(_) => None,
        };
        let set = |buffers: Vec<(vk::Buffer, u64, u64)>| OwnedSet {
            images: (fetched.iter().map(|&(view, _)| view))
                .chain(kept_view)
                .collect(),
            storage_images: Vec::new(),
            buffers,
        };
        if let Some((cut, geometry, band, _)) = &cut {
            // The rectangle's rows in bands, each band's kept and changed rows from a multiple of
            // the alignment (`Geometry::band`), its current bytes bound from its first byte
            // rounded down to it.
            let Geometry {
                first,
                row_bytes,
                pitch,
                rows,
            } = *geometry;
            let words = row_bytes / 4;
            let [x0, y0] = [cut.region.image_offset.x, cut.region.image_offset.y].map(|o| o as u32);
            let width = cut.region.image_extent.width;
            let mut start = 0;
            while start < rows {
                let count = (*band).min(rows - start);
                let at = first + start * pitch;
                let bound = at / PART_ALIGNMENT * PART_ALIGNMENT;
                let skip = at - bound;
                let size = count * row_bytes;
                let kept_window = |kept: Part| (kept.raw, kept.at + start * row_bytes, size);
                let current_window = (merged.buffer, bound, skip + (count - 1) * pitch + row_bytes);
                let (row_words, first_word) = (u32_of(pitch / 4)?, u32_of(skip / 4)?);
                match (changed, kept_in) {
                    (Some(_), KeptIn::Image { .. }) => {
                        return Err(internal("a merge copying its texels kept in an image"));
                    }
                    (Some(changed), KeptIn::Part(kept)) => {
                        let changed_window = (changed.raw, changed.at + start * row_bytes, size);
                        sets.push(set(vec![kept_window(kept), changed_window, current_window]));
                        let total = count * words;
                        let shape = ShadowMergeModule::shape(
                            u32_of(words)?,
                            u32_of(total)?,
                            row_words,
                            first_word,
                        );
                        let groups = ShadowMergeModule::groups(u32_of(total)?);
                        dispatches.push(([groups, 1], shape));
                        examined += total;
                    }
                    (None, kept_in) => {
                        // The kept texels: the kept part's rows, or the kept image's, fetched at
                        // the shadow's coordinates.
                        sets.push(set(match kept_in {
                            KeptIn::Part(kept) => vec![kept_window(kept), current_window],
                            KeptIn::Image { .. } => vec![current_window],
                        }));
                        let origin = [x0, y0 + u32_of(start)?];
                        let shape =
                            ShadowMergeImageModule::shape(origin, width, row_words, first_word);
                        let read = (fetched.map(|(_, read)| read))
                            .ok_or_else(|| internal("a fetching merge without its read"))?;
                        let groups = ShadowMergeImageModule::groups(read, [width, u32_of(count)?]);
                        dispatches.push((groups, shape));
                        examined += u64::from(width) * u64::from(read.words()) * count;
                    }
                }
                start += count;
            }
        } else {
            // Only a cut fetches, and only a merge that fetches keeps in an image.
            let KeptIn::Part(kept_part) = kept_in else {
                return Err(internal("an uncut merge keeping its texels in an image"));
            };
            // In place, the current bytes' binding starts up to 255 bytes before each window.
            let room = match in_place {
                true => range.saturating_sub(PART_ALIGNMENT),
                false => range,
            };
            let window = (room / PART_ALIGNMENT * PART_ALIGNMENT).min(groups * GROUP_BYTES);
            if window == 0 {
                return Err(internal(
                    "a device whose storage buffers hold no merge window",
                ));
            }
            let mut start = 0;
            // In place, the windows cover the texture's bytes alone, which lie in the buffer;
            // through a copy, the parts whole.
            let total = if in_place { rounded } else { part };
            while start < total {
                let size = window.min(total - start);
                let words = u32_of(size / 4)?;
                let (current_window, first_word) = match current {
                    Some(current) => ((current.raw, current.at + start, size), 0),
                    None => {
                        let at = merged.first + start;
                        let bound = at / PART_ALIGNMENT * PART_ALIGNMENT;
                        let skip = at - bound;
                        ((merged.buffer, bound, skip + size), u32_of(skip / 4)?)
                    }
                };
                let changed =
                    changed.ok_or_else(|| internal("a merge without its changed part"))?;
                sets.push(set(vec![
                    (kept_part.raw, kept_part.at + start, size),
                    (changed.raw, changed.at + start, size),
                    current_window,
                ]));
                // The window's words, one row of them.
                let shape = ShadowMergeModule::shape(words, words, 0, first_word);
                dispatches.push(([ShadowMergeModule::groups(words), 1], shape));
                examined += u64::from(words);
                start += size;
            }
        }
        // Whole blocks where every word the dispatches examine is bytes the command wrote or
        // Metal's own store rewrites: a command's that replaces every texel, or a render pass's
        // that stores the tiles of its area, cut to the texels' rows (the module docs).
        let stores = match blocks && !two_ways && cut.is_some() {
            true => ShadowMergeStores::Blocks,
            false => ShadowMergeStores::Words,
        };
        counts.shadow_merge_planned(dispatches.len(), examined);
        if stores == ShadowMergeStores::Blocks {
            counts.shadow_merge_in_blocks(dispatches.len());
        }
        let kernel = match (fetched, kept_in) {
            (Some(_), KeptIn::Image { texel, .. }) => {
                device.owned_kernel(OwnedModule::ShadowMergeKeptImage(texel, stores))?
            }
            (Some((_, read)), KeptIn::Part(_)) => {
                device.owned_kernel(OwnedModule::ShadowMergeImage(read, stores))?
            }
            (None, _) => device.owned_kernel(OwnedModule::ShadowMerge(stores))?,
        };
        let binds = OwnedBinds::new(device, &kernel, &sets)?;
        let cut = cut.map(|(cut, ..)| cut);
        let keep_dispatch = match kept_in {
            KeptIn::Part(kept) => dispatched_out(device, copies, kept, cut.as_ref()),
            KeptIn::Image { .. } => None,
        };
        let changed = match changed {
            Some(part) => Changed::Copied {
                part,
                dispatch: dispatched_out(device, copies, part, cut.as_ref()),
            },
            None => {
                device.shared().counts.shadow_merge_fetched();
                Changed::Fetched
            }
        };
        let texels = match &cut {
            Some(Cut { region, .. }) => {
                let extent = region.image_extent;
                u64::from(extent.width) * u64::from(extent.height)
            }
            None => copies.texels(),
        };
        if two_ways {
            let bytes = merged.bytes.clone();
            device.degrade(Degradation::ShadowMerge, || {
                let object = format!("a command writing bytes {bytes:?} of a buffer two ways");
                let reason = "it writes them through a shadowed texture over the buffer and also \
                              another way, and a copy back of the shadow would replace the other \
                              write's bytes, which Metal keeps"
                    .to_owned();
                (object, reason)
            });
        }
        let merge = ShadowMerge {
            merged,
            cut,
            texels,
            kept_in,
            changed,
            current,
            kept,
            keep_dispatch,
            held,
            kernel,
            blocks,
            binds,
            dispatches,
        };
        Ok((merge, lease))
    }

    /// Where a copy writes the keep's kept part (the keep no dispatch, cut to no rectangle, into
    /// no kept image): its `VkBuffer` and where the part starts there; `None` for any other
    /// keep.
    pub(super) fn copied_kept_part(&self) -> Option<(vk::Buffer, u64)> {
        match (&self.keep_dispatch, self.kept_in, &self.cut) {
            (None, KeptIn::Part(kept), None) => Some((kept.raw, kept.at)),
            _ => None,
        }
    }

    /// Whether the keep is a dispatch (`ShadowMerge::keep_dispatch`), which runs in the compute
    /// stage.
    pub(super) fn dispatches(&self) -> bool {
        self.keep_dispatch.is_some()
    }

    /// The dispatch on `device` that takes the copy in of `region` of the shadow of `copies` (its
    /// texels in the buffer, as the copy in cuts it, `ShadowCopy::copied_region`) and this
    /// merge's keep together (`shadow_copy_in.rs`): where the keep is a dispatch of the
    /// rectangle the merge is cut to, in place, which is the copy in's, and the shadow's view
    /// stores; `None` where the copy in and the keep stay apart.
    pub(super) fn keep_in(
        &self,
        device: &Device,
        copies: &ShadowCopies,
        region: &vk::BufferImageCopy,
    ) -> Option<CopyIn> {
        let (Some(_), Some(cut), None, KeptIn::Part(kept)) =
            (&self.keep_dispatch, &self.cut, self.current, self.kept_in)
        else {
            return None;
        };
        // Both of the texture's one region, of one slice and level.
        let same = region.image_offset == cut.region.image_offset
            && region.image_extent == cut.region.image_extent;
        if !same {
            return None;
        }
        let view = word_view(copies)?;
        CopyIn::new(
            device,
            view,
            self.merged.buffer,
            region,
            (kept.raw, kept.at),
        )
    }

    /// The uses of the dispatch that takes the copy in of the shadow of `copies` and the keep
    /// ([`keep_in`](Self::keep_in)): a storage read of the texture's bytes, and storage writes of
    /// the shadow and of the kept part, in the compute stage.
    pub(super) fn keep_in_uses(&self, copies: &ShadowCopies) -> Vec<Use<ResourceKey>> {
        let storage = |access| Access::new(vk::PipelineStageFlags2::COMPUTE_SHADER, access);
        let (truth, range) = copies.truth();
        let (key, tracked, kept) = self.kept.clone();
        vec![
            Use::reads(
                truth,
                copies.tracked,
                range,
                storage(vk::AccessFlags2::SHADER_STORAGE_READ),
            ),
            Use::writes(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                storage(vk::AccessFlags2::SHADER_STORAGE_WRITE),
            ),
            Use::writes(
                key,
                tracked,
                kept,
                storage(vk::AccessFlags2::SHADER_STORAGE_WRITE),
            ),
        ]
    }

    /// Whether the merge's copy of the texels after the commands into the changed part is a
    /// dispatch (none where the merge fetches them).
    pub(super) fn changed_dispatches(&self) -> bool {
        matches!(
            self.changed,
            Changed::Copied {
                dispatch: Some(_),
                ..
            }
        )
    }

    /// The rectangle `[x0, y0, x1, y1)` of the shadow's texels the merge keeps and merges, `None`
    /// for every texel.
    pub(super) fn area(&self) -> Option<[u32; 4]> {
        (self.cut.as_ref())
            .filter(|cut| !cut.whole)
            .map(|cut| cut.area)
    }

    /// Whether it was made for a command whose merge stores whole blocks (`MergeFor::blocks`).
    pub(super) fn in_blocks(&self) -> bool {
        self.blocks
    }

    /// Whether the keep, the changed copy and the merge are cut to a rectangle.
    pub(super) fn cuts(&self) -> bool {
        self.cut.is_some()
    }

    /// The texels each keep copies (`WorkCounts::shadow_texels_kept`).
    pub(super) fn texels_kept(&self) -> u64 {
        self.texels
    }

    /// Record the copy of the shadow of `copies` into `part`: the cut rectangle's texels, row after
    /// row, or every texel laid out as in the buffer from the texture's first byte.
    ///
    /// # Safety
    ///
    /// As `record_keep`.
    unsafe fn record_out(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (copies, moves): (&ShadowCopies, Option<&SampleMoves>),
        part: Part,
    ) {
        // SAFETY: the caller's contract; the part holds the rectangle's texels row after row, or
        // the texture's bytes.
        unsafe {
            match &self.cut {
                Some(Cut { image, region, .. }) => device.cmd_copy_image_to_buffer(
                    command_buffer,
                    *image,
                    vk::ImageLayout::GENERAL,
                    part.raw,
                    &[vk::BufferImageCopy {
                        buffer_offset: part.at,
                        ..*region
                    }],
                ),
                None => copies.record_out(device, command_buffer, (part.raw, part.at), moves),
            }
        }
    }

    /// The use of the shadow of `copies` by the keep, the changed copy or the merge's fetches: a
    /// copy's transfer read, or a dispatch's sampled read where `dispatched`.
    fn shadow_read(copies: &ShadowCopies, dispatched: bool) -> Use<ResourceKey> {
        let read = match dispatched {
            true => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),
            false => transfer(vk::AccessFlags2::TRANSFER_READ),
        };
        Use::reads(
            copies.shadow_key,
            copies.tracked,
            copies.shadow_range.clone(),
            read,
        )
    }

    /// The uses of keeping the shadow of `copies`: a read of the shadow and a write of the kept
    /// part, by a copy or a dispatch.
    pub(super) fn keep_uses(&self, copies: &ShadowCopies) -> Vec<Use<ResourceKey>> {
        let (key, tracked, range) = self.kept.clone();
        let write = match self.dispatches() {
            true => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
            false => transfer(vk::AccessFlags2::TRANSFER_WRITE),
        };
        vec![
            ShadowMerge::shadow_read(copies, self.dispatches()),
            Use::writes(key, tracked, range, write),
        ]
    }

    /// The uses of merging the shadow of `copies`: a read of the shadow and of the kept part, and
    /// a read and a write of the texture's bytes (by the kernel where it merges in place, else by
    /// copies).
    pub(super) fn merge_uses(&self, copies: &ShadowCopies) -> Vec<Use<ResourceKey>> {
        let (key, tracked, range) = self.kept.clone();
        let storage = |access| Access::new(vk::PipelineStageFlags2::COMPUTE_SHADER, access);
        let (read, write) = match self.current {
            Some(_) => (
                transfer(vk::AccessFlags2::TRANSFER_READ),
                transfer(vk::AccessFlags2::TRANSFER_WRITE),
            ),
            None => (
                storage(vk::AccessFlags2::SHADER_STORAGE_READ),
                storage(vk::AccessFlags2::SHADER_STORAGE_WRITE),
            ),
        };
        let dispatched = match self.changed {
            Changed::Copied { ref dispatch, .. } => dispatch.is_some(),
            Changed::Fetched => true,
        };
        let kept_read = match self.kept_in {
            KeptIn::Part(_) => vk::AccessFlags2::SHADER_STORAGE_READ,
            KeptIn::Image { .. } => vk::AccessFlags2::SHADER_SAMPLED_READ,
        };
        vec![
            ShadowMerge::shadow_read(copies, dispatched),
            Use::reads(key, tracked, range, storage(kept_read)),
            Use {
                key: self.merged.key,
                tracked: copies.tracked,
                range: UseRange::Bytes(self.merged.bytes.clone()),
                read,
                write,
            },
        ]
    }

    /// Record the keep of the shadow of `copies` into `command_buffer`, a heap's multisample
    /// texture's through `moves` out of its image (`ShadowCopy::plan_moves`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the buffer, the shadow, the moves and
    /// the scratch are live until it completes, and an image is in the `GENERAL` layout.
    pub(super) unsafe fn record_keep(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (copies, moves): (&ShadowCopies, Option<&SampleMoves>),
    ) {
        // SAFETY: the caller's contract; the kept part holds the texture's bytes, and the
        // dispatch stores the shadow's texels there as the copy would; the kept image, of a
        // format of the shadow's compatibility class, holds the cut rectangle at its coordinates
        // (`kept_extent`), in the `GENERAL` layout where every image rests.
        unsafe {
            match (&self.keep_dispatch, self.kept_in) {
                (Some(keep), _) => keep.record(device, command_buffer),
                (None, KeptIn::Part(kept)) => match self.cut {
                    Some(_) => self.record_out(device, command_buffer, (copies, moves), kept),
                    None => copies.record_kept(device, command_buffer, (kept.raw, kept.at), moves),
                },
                (
                    None,
                    KeptIn::Image {
                        shadow, kept, copy, ..
                    },
                ) => device.cmd_copy_image(
                    command_buffer,
                    shadow,
                    vk::ImageLayout::GENERAL,
                    kept,
                    vk::ImageLayout::GENERAL,
                    &[copy],
                ),
            }
        }
    }

    /// Record the merge of the shadow of `copies` into its buffer's bytes (the module docs): a
    /// barrier between the copies of the texels into the changed and current parts, where it has
    /// them (a heap's multisample texture's through `moves` out of its image), and its
    /// dispatches; and one to everything after them. A merge that fetches the texels in place
    /// records no copy, and its dispatches follow the commands by the barrier every caller
    /// records before a copy (the encoder's, from the merge's uses).
    ///
    /// # Safety
    ///
    /// As `record_keep`, after the keep and the command it surrounds, and after a barrier from
    /// what precedes it.
    pub(super) unsafe fn record_merge(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (copies, moves): (&ShadowCopies, Option<&SampleMoves>),
    ) {
        let Merged {
            buffer,
            ref bytes,
            first,
            ..
        } = self.merged;
        let span = bytes.end - bytes.start;
        // SAFETY: the caller's contract; each part holds the texture's bytes, and the sets bind
        // windows inside the parts and, in place, inside the texture's bytes (and the shadow's
        // view, where the merge fetches its texels).
        unsafe {
            match &self.changed {
                Changed::Copied {
                    dispatch: Some(out),
                    ..
                } => out.record(device, command_buffer),
                Changed::Copied { part, .. } => {
                    self.record_out(device, command_buffer, (copies, moves), *part);
                }
                Changed::Fetched => {}
            }
            if let Some(current) = self.current {
                device.cmd_copy_buffer(
                    command_buffer,
                    buffer,
                    current.raw,
                    &[vk::BufferCopy {
                        src_offset: first,
                        dst_offset: current.at,
                        size: span,
                    }],
                );
            }
            let copied = !matches!(self.changed, Changed::Fetched) || self.current.is_some();
            if copied {
                Barrier::everything().record(device, command_buffer);
            }
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            for (window, (groups, shape)) in self.dispatches.iter().enumerate() {
                self.binds
                    .bind(device, command_buffer, &self.kernel, window);
                device.cmd_push_constants(
                    command_buffer,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    shape,
                );
                device.cmd_dispatch(command_buffer, groups[0], groups[1], 1);
            }
            // To everything after: the copy back, the next keep into the same scratch, and the
            // bytes' next use.
            Barrier::everything().record(device, command_buffer);
            if let Some(current) = self.current {
                device.cmd_copy_buffer(
                    command_buffer,
                    current.raw,
                    buffer,
                    &[vk::BufferCopy {
                        src_offset: current.at,
                        dst_offset: first,
                        size: span,
                    }],
                );
            }
        }
    }
}

impl Drop for ShadowMerge {
    /// Its scratch goes back to its shadow's spares: the merge goes once the command buffers that
    /// recorded it completed, which retain it.
    fn drop(&mut self) {
        match &self.held {
            Held::Spare(buffer, spares) => spares.give(buffer.clone()),
            Held::Image(_) | Held::Apart(_) => {}
        }
    }
}

/// The shadow's unsigned-integer view of `copies`, where it has one (`shadow_copy_back.rs`): a
/// one-region image of 4-, 8- or 16-byte texels copied in place.
fn word_view(copies: &ShadowCopies) -> Option<WordView> {
    match &copies.source {
        ShadowSource::Bytes {
            target: ShadowTarget::Image {
                words: Some(words), ..
            },
            ..
        } => Some(*words),
        _ => None,
    }
}

/// A shadow's sampled view of its own 8-bit unsigned normalized format, through which its cut
/// merges read the texels the commands left (`bind::ShadowMergeRead`) where it has no
/// unsigned-integer view (`shadow_copy_back.rs`): one of the shadow image's own views
/// (`Image::own_view`), which stays through a detach of the texture's bytes and goes as the image
/// drops.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnormView {
    pub(super) view: vk::ImageView,
    /// How the kernel reads it: `B8G8R8A8Unorm` or `R8G8B8A8Unorm` ([`unorm_read`]).
    pub(super) read: ShadowMergeRead,
}

/// How the kernel reads a shadow image of `format` through a view of its own format (the module
/// docs): `B8G8R8A8_UNORM` and `R8G8B8A8_UNORM` alone. Every other format's merges copy the texels
/// out: an sRGB format's fetch returns its bytes through the sRGB curve, not as `c / 255`, which
/// the kernel does not invert; a 16-bit float format's widens each value to `f32`, through which a
/// NaN's payload need not come back; and the kernel rebuilds no other format's bytes.
fn unorm_read(format: Format) -> Option<ShadowMergeRead> {
    match format {
        Format::B8G8R8A8_UNORM => Some(ShadowMergeRead::B8G8R8A8Unorm),
        Format::R8G8B8A8_UNORM => Some(ShadowMergeRead::R8G8B8A8Unorm),
        _ => None,
    }
}

/// `plan`, a shadow's image, made to take its [`UnormView`]: sampled usage, with the kernel's read
/// ([`unorm_read`]). `None`, `plan` unchanged, for an image of another format, of more than one
/// level or slice, or of a format the device does not sample. Sampled usage changes nothing of
/// the texture MoltenVK makes for the image: MoltenVK 1.4.1 gives a texture
/// `MTLTextureUsageShaderRead` for transfer-source usage, which every shadow has, as for sampled
/// usage (`MVKPixelFormats::getMTLTextureUsage`), and the view, of the image's own format and
/// mapping, takes no pixel-format view.
pub(super) fn with_unorm_view(
    mut plan: ImagePlan,
    facts: &ImageFacts<'_>,
) -> (ImagePlan, Option<ShadowMergeRead>) {
    let single = plan.levels == 1 && plan.layers == 1 && plan.extent[2] == 1;
    let sampled = facts
        .optimal(plan.format)
        .contains(FormatFeatureFlags2::SAMPLED_IMAGE);
    let read = unorm_read(plan.format).filter(|_| single && sampled);
    if read.is_some() {
        plan.usage |= ImageUsageFlags::SAMPLED;
    }
    (plan, read)
}

/// The view a cut merge of the shadow of `copies` fetches the texels the commands left through,
/// and how its kernel reads them: the shadow's unsigned-integer view where it has one
/// (`shadow_copy_back.rs`), else its own 8-bit unsigned normalized view ([`UnormView`]); `None`
/// where it has neither, and the merge copies them out.
fn fetch_view(copies: &ShadowCopies) -> Option<(vk::ImageView, ShadowMergeRead)> {
    let ShadowSource::Bytes {
        target: ShadowTarget::Image { words, unorm, .. },
        ..
    } = &copies.source
    else {
        return None;
    };
    match (words, unorm) {
        (Some(words), _) => Some((words.view, ShadowMergeRead::Words(words.texel))),
        (None, Some(unorm)) => Some((unorm.view, unorm.read)),
        (None, None) => None,
    }
}

/// The extent of a kept image of the shadow of `copies` (`texture/kept_image.rs`): every texel of
/// its one region of one level and slice at its own coordinates, so a kept image holds any cut of
/// it where the shadow does; `None` for another shadow.
fn kept_extent(copies: &ShadowCopies) -> Option<[u32; 2]> {
    let ShadowSource::Bytes {
        target: ShadowTarget::Image { regions, .. },
        ..
    } = &copies.source
    else {
        return None;
    };
    let [region] = regions.as_slice() else {
        return None;
    };
    let (offset, extent) = (region.image_offset, region.image_extent);
    let subresource = region.image_subresource;
    let one = extent.depth == 1
        && subresource.mip_level == 0
        && subresource.base_array_layer == 0
        && subresource.layer_count == 1;
    let at = |o: i32, n: u32| u32::try_from(o).ok()?.checked_add(n);
    match one {
        true => Some([at(offset.x, extent.width)?, at(offset.y, extent.height)?]),
        false => None,
    }
}

/// Whether `bind::shadow_merge_image_module` takes the rectangle of `cut`: its first texel's
/// coordinates and its last row's below 65536 (the shape's origin), and its row's workgroups
/// within `groups` (`maxComputeWorkGroupCount[0]`); its rows go in bands of at most
/// `groups_y` (`maxComputeWorkGroupCount[1]`).
fn fetches(cut: &Cut, read: ShadowMergeRead, groups: u64, groups_y: u64) -> bool {
    let (offset, extent) = (cut.region.image_offset, cut.region.image_extent);
    let row_groups = u64::from(ShadowMergeImageModule::groups(read, [extent.width, 1])[0]);
    let below = |o: i32, n: u32| u64::from(o as u32) + u64::from(n) <= 1 << 16;
    below(offset.x, 1) && below(offset.y, extent.height) && row_groups <= groups && groups_y > 0
}

/// The dispatch on `device` that stores the shadow of `copies` into `part`, as its copy out
/// (`ShadowMerge::record_out`) would: the texels of `cut` row after row, or every texel laid
/// out as in the buffer from the texture's first byte; where `device` runs a copy back faster as
/// a dispatch and the shadow is a one-region image copied in place with its integer view
/// (`shadow_copy_back.rs`); `None` where the part does not fit one (a copy takes it).
fn dispatched_out(
    device: &Device,
    copies: &ShadowCopies,
    part: Part,
    cut: Option<&Cut>,
) -> Option<CopyBack> {
    if !shadow_copy_back::dispatches(device) {
        return None;
    }
    let ShadowSource::Bytes {
        target:
            ShadowTarget::Image {
                regions,
                staging: None,
                words: Some(words),
                ..
            },
        ..
    } = &copies.source
    else {
        return None;
    };
    let [region] = regions.as_slice() else {
        return None;
    };
    let region = cut.map_or(region, |cut| &cut.region);
    let region = vk::BufferImageCopy {
        buffer_offset: part.at,
        ..*region
    };
    CopyBack::new(device, *words, part.raw, &region)
}

/// Where a cut rectangle's rows lie in the buffer's `VkBuffer`.
#[derive(Clone, Copy)]
struct Geometry {
    /// Its first byte, a whole word.
    first: u64,
    /// The bytes of each of its rows (whole words), their pitch in the buffer (whole words), and
    /// its rows.
    row_bytes: u64,
    pitch: u64,
    rows: u64,
}

impl Geometry {
    /// The rows of each band of the merge: all of them where one binding of each part and of the
    /// current bytes stays within `range` (`maxStorageBufferRange`) and one dispatch takes at most
    /// `dispatched` rows (as the device's workgroup counts bound its kernel), else the most that
    /// do in a whole number of steps, so each band's kept and changed rows start at a multiple of
    /// the alignment; `None` where not one row fits.
    fn band(&self, range: u64, dispatched: u64) -> Option<u64> {
        let Geometry {
            row_bytes,
            pitch,
            rows,
            ..
        } = *self;
        let step = PART_ALIGNMENT / gcd(row_bytes, PART_ALIGNMENT);
        let by_current = match range.checked_sub(PART_ALIGNMENT + row_bytes) {
            Some(rest) => rest / pitch + 1,
            None => 0,
        };
        let most = rows.min(by_current).min(range / row_bytes).min(dispatched);
        let band = match most == rows {
            true => rows,
            false => most / step * step,
        };
        (band > 0).then_some(band)
    }
}

/// The cut of `copies`'s shadow to the texels inside `area` (`[x0, y0, x1, y1)`; `None` every
/// texel), each row widened to whole words of the buffer, and where its rows lie: for one region
/// of one slice over the buffer's bytes in place, whose rows' pitch is whole words and whose
/// texels inside `area` take whole words from the first of a row to the last; `None` otherwise,
/// or where `area` holds no texel.
fn cut_of(copies: &ShadowCopies, area: Option<[u32; 4]>) -> Option<(Cut, Geometry)> {
    let [x0, y0, x1, y1] = area.unwrap_or([0, 0, u32::MAX, u32::MAX]);
    let ShadowSource::Bytes {
        target:
            ShadowTarget::Image {
                image,
                regions,
                staging: None,
                texel: Some(texel),
                ..
            },
        ..
    } = &copies.source
    else {
        return None;
    };
    let [region] = regions.as_slice() else {
        return None;
    };
    let extent = region.image_extent;
    if extent.depth != 1 || region.image_subresource.layer_count != 1 {
        return None;
    }
    let row = match region.buffer_row_length {
        0 => extent.width,
        row => row,
    };
    let pitch = u64::from(row) * texel;
    // The texels of a whole number of words: a row's first texel's byte a whole word, as the
    // texture's first byte and its pitch are.
    let step = (4 / gcd(*texel, 4)) as u32;
    let x0 = x0 / step * step;
    let x1 = x1.min(extent.width).next_multiple_of(step);
    let y1 = y1.min(extent.height);
    if x1 > extent.width || x0 >= x1 || y0 >= y1 || !pitch.is_multiple_of(4) {
        return None;
    }
    let first = region.buffer_offset + (u64::from(y0) * u64::from(row) + u64::from(x0)) * texel;
    let geometry = Geometry {
        first,
        row_bytes: u64::from(x1 - x0) * texel,
        pitch,
        rows: u64::from(y1 - y0),
    };
    if !first.is_multiple_of(4) || !geometry.row_bytes.is_multiple_of(4) {
        return None;
    }
    let cut = Cut {
        area: [x0, y0, x1, y1],
        whole: area.is_none(),
        image: *image,
        region: vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: x1 - x0,
            buffer_image_height: y1 - y0,
            image_subresource: region.image_subresource,
            image_offset: vk::Offset3D {
                x: region.image_offset.x + x0 as i32,
                y: region.image_offset.y + y0 as i32,
                z: region.image_offset.z,
            },
            image_extent: vk::Extent3D {
                width: x1 - x0,
                height: y1 - y0,
                depth: 1,
            },
        },
    };
    Some((cut, geometry))
}

fn transfer(access: vk::AccessFlags2) -> Access {
    Access::new(vk::PipelineStageFlags2::COPY, access)
}
