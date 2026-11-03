//! Sparse textures over ordinary images, the zeroing of unmapped texels, and tile maps.
//!
//! - **Ordinary images.** On a device that binds no sparse memory for a texture (MoltenVK binds
//!   none), where no function reads the image's own residency (the device does not enable
//!   `shaderResourceResidency`), a sparse texture is an ordinary image in memory of its own: its
//!   units are Metal's own (a tile, a slice's tail), so every mapping update is exact, and its
//!   memory is its image's whatever is mapped.
//! - **Zeroing.** Every sparse texture keeps the texels of each unmapped Metal unit inside memory
//!   it holds at zero: an ordinary image's from when it is made, a sparse image's in each device
//!   unit bound while some Metal unit it holds texels of is mapped. A batch zeroes each device unit
//!   it binds and the texels of each Metal unit it unmaps inside a unit that stays bound
//!   ([`Mappings`]), and each blit or render pass that writes the texture is followed by the
//!   zeroing of the unmapped Metal units it wrote there (its scrub, `Deferred::SparseScrub`), so a
//!   read of an unmapped texel reads what Metal's reads, zero, and a write to one is dropped as
//!   Metal drops it (a blit into unmapped tiles leaves them reading zero). A single-sampled image
//!   is zeroed by copies of zeros; a multisampled one, which no copy writes, by clears
//!   (`vkCmdClearAttachments`) in a render pass of its level and layer ([`Zeroing`]), as Metal's
//!   multisampled sparse textures read zero where unmapped (a Metal probe, the M3 Max at 2x and
//!   4x: a tile mapped, unmapped and mapped again, or mapped new, reads zero, and a clear's stores
//!   into unmapped tiles are dropped). A sparse image's device unit that holds no mapped tile
//!   reads the device's value, and a write into one is the device's. A function's writes to a
//!   sparse texture never land (the module drops them, as Metal does).
//! - **Tile maps.** A function reads a sparse texture's residency from a tile map
//!   (`bind::SparseTileMap`) on every device, at each slot whose type has one and whose observed
//!   accesses each name their texels: a buffer with a bit per tile, which each batch that changes
//!   the texture's mappings rewrites after it applies them, before its work, so a command reads the
//!   mappings Metal's would. Every texture whose type has a map keeps one.
//!
//! A batch's zeroing and map writes are one command buffer ([`Mappings`]), the first of its work
//! in place of its first resource state encoder's commands, after everything submitted before it
//! on every queue, as a bind is (`bind.rs`); the hazard tracker sees it write the textures and
//! maps. A scrub ([`Scrub`]) is a segment of the encoder that wrote, recorded as its batch is
//! submitted.

use super::bind::Moved;
use super::mapping::{MetalUnit, SparseState, TexelBox, VkUnit};
use super::texture::{SparseImage, SparseShape, Zeroing};
use crate::buffer::Buffer;
use crate::copy_plan::ImageRegion;
use crate::device::lock;
use crate::error::{internal, Error};
use crate::exit_guard::{self, Call};
use crate::memory::Placement;
use crate::render_pass::Target;
use crate::render_target::{self, FramebufferObject, RenderPassObject, ATTACHMENT_STAGES};
use crate::resource::ResourceKey;
use crate::texture::{buffer_image_copy, copy_units, image_aspects, Texture};
use crate::tracker::{Access, Use};
use crate::upload::Uploads;
use crate::values::ResourceOptions;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{SparseTileMap, TileMapping};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{ImageAspectFlags, ImageUsageFlags};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Range;

/// How a sparse texture's texels are held.
pub(crate) enum Backing {
    /// A sparse resident image, its memory bound by device unit.
    Bound,
    /// An ordinary image in memory of its own, its unmapped tiles kept at zero.
    Emulated { memory: Placement },
}

/// A sparse texture's tile map: the buffer a function reads its residency from, and the map
/// with nothing mapped, which each rewrite starts from.
pub(crate) struct TileMapBuffer {
    pub(crate) buffer: Buffer,
    empty: SparseTileMap,
}

impl TileMapBuffer {
    /// The tile map of a sparse texture of `shape` with `layers` slices, tiles of `tile` texels
    /// and its tail from `first_tail`, on `device`, or `None` where its type has none (a cube or
    /// multisampled texture, which the format does not lay out). Its buffer holds no bytes yet:
    /// [`SparseImage::tile_map_bytes`] are staged when the texture is made, for the device's next
    /// submission to copy in.
    pub(crate) fn new(
        device: &Device,
        shape: &SparseShape,
        layers: u32,
        tile: [u64; 3],
        first_tail: u64,
    ) -> Result<Option<TileMapBuffer>, Error> {
        let Some(empty) = tile_map_layout(shape, layers, tile, first_tail) else {
            return Ok(None);
        };
        let buffer = device.own_buffer(empty.byte_len(), ResourceOptions::private())?;
        Ok(Some(TileMapBuffer { buffer, empty }))
    }
}

/// The tile map, nothing mapped, of a sparse texture of `shape` with `layers` slices, tiles of
/// `tile` texels and its tail from `first_tail`, or `None` where its type has none (a cube or
/// multisampled texture, which the format does not lay out).
fn tile_map_layout(
    shape: &SparseShape,
    layers: u32,
    tile: [u64; 3],
    first_tail: u64,
) -> Option<SparseTileMap> {
    let n = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
    let slices = match shape.texture_type {
        TextureType::Type2DArray => layers,
        _ => 1,
    };
    SparseTileMap::new(
        shape.texture_type,
        shape.extent.map(n),
        n(shape.levels),
        slices,
        tile.map(n),
        n(first_tail),
    )
    .ok()
}

impl SparseImage {
    /// Whether the texture is an ordinary image whose unmapped tiles the runtime zeroes.
    pub(crate) fn emulated(&self) -> bool {
        matches!(self.backing, Backing::Emulated { .. })
    }

    /// Its tile map, where its type has one.
    pub(crate) fn tile_map(&self) -> Option<&TileMapBuffer> {
        self.tile_map.as_ref()
    }

    /// The bytes of its tile map with `mapped` mapped (none for a texture without one).
    pub(crate) fn tile_map_bytes(&self, mapped: &HashSet<MetalUnit>) -> Vec<u8> {
        let Some(tile_map) = self.tile_map() else {
            return Vec::new();
        };
        let mut map = tile_map.empty.clone();
        for &unit in mapped {
            let (level, slice, at) = match unit {
                MetalUnit::Tile { level, slice, at } => (level, slice, at),
                MetalUnit::Tail { slice } => (self.grid.first_tail(), slice, [0; 3]),
            };
            // The units are the texture's own (an update checked them), so the map takes them.
            let _ = map.update(TileMapping::Map, level, slice, at, [1, 1, 1]);
        }
        map.to_bytes()
    }

    /// Whether device unit `unit` holds memory in `state`: an ordinary image's every unit, a
    /// sparse image's while bound.
    pub(crate) fn held(&self, unit: VkUnit, state: &SparseState) -> bool {
        self.emulated() || state.bound.contains_key(&unit)
    }
}

/// The render passes and framebuffers clears recorded with ([`record_zeroing`] of a texture the
/// runtime zeroes by clears), held until their command buffer completes.
#[derive(Default)]
pub(crate) struct ClearObjects(Vec<(RenderPassObject, FramebufferObject)>);

impl ClearObjects {
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Record into `command_buffer` the zeroing of `boxes` of `texture`, a sparse texture (nothing
/// for another), in the `GENERAL` layout, as it is zeroed ([`Zeroing`]): copies of zeros, each box
/// in pieces of at most one Metal tile, whose bytes the zeros hold; or clears in render passes
/// ([`record_clears`]), whose objects go into `held`.
///
/// # Safety
///
/// `command_buffer` is a command buffer of `device` recording outside a render pass; the texture
/// is live and the memory of each box bound; the caller orders the zeroing after what used the
/// image before and before what uses it after.
pub(crate) unsafe fn record_zeroing(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    boxes: impl IntoIterator<Item = TexelBox>,
    held: &mut ClearObjects,
) -> Result<(), Error> {
    let Some(image) = texture.sparse_image() else {
        return Ok(());
    };
    match image.zeroing {
        // SAFETY: the caller's.
        Zeroing::Copies => unsafe { record_copies(device, command_buffer, texture, boxes) },
        // SAFETY: the caller's.
        Zeroing::Clears => unsafe { record_clears(device, command_buffer, texture, boxes, held) },
    }
}

/// [`record_zeroing`] by copies of zeros, of a sparse texture whose image a copy writes.
///
/// # Safety
///
/// As [`record_zeroing`]'s.
unsafe fn record_copies(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    boxes: impl IntoIterator<Item = TexelBox>,
) -> Result<(), Error> {
    let Some(image) = texture.sparse_image() else {
        return Ok(());
    };
    let raw = texture.raw_image()?;
    let plan = texture.image_plan()?;
    let zeros = &image.zeros;
    let (source, start) = zeros.at(0);
    let copy_units = copy_units(plan.format, plan.pixel_format);
    let tile = image.grid.tile32();
    let mut copies = Vec::new();
    for piece in boxes.into_iter().flat_map(|b| b.pieces(tile)) {
        for &(aspect, bytes, [bw, bh]) in &copy_units {
            let length = u64::from(piece.size[0].div_ceil(bw))
                * u64::from(piece.size[1].div_ceil(bh))
                * u64::from(piece.size[2])
                * u64::from(bytes);
            if length > zeros.length() {
                return Err(internal("a sparse tile larger than its heap's zeros"));
            }
            let region = ImageRegion {
                aspect,
                level: piece.level,
                layer: piece.layer,
                layers: 1,
                origin: piece.origin,
                size: piece.size,
            };
            let mut copy = buffer_image_copy(&region);
            copy.buffer_offset = start;
            copies.push(copy);
        }
    }
    if copies.is_empty() {
        return Ok(());
    }
    // SAFETY: the caller's; the zeros are a live buffer of the device holding every copy's
    // bytes at `start`, and each region lies in the texture's image.
    unsafe {
        device.shared().vk().cmd_copy_buffer_to_image(
            command_buffer,
            source,
            raw,
            vk::ImageLayout::GENERAL,
            &copies,
        );
    }
    exit_guard::called(Call::CmdCopyBufferToImage);
    Ok(())
}

/// [`record_zeroing`] by clears, of a multisampled sparse texture, which no copy writes: for each
/// level and layer a box lies in, a render pass of that level and layer alone, every aspect of
/// the image's loaded and stored (`render_target::reach_pass`: the device's kept pass, and a
/// framebuffer made now, which go into `held`), over the rectangle that bounds its boxes, in
/// which `vkCmdClearAttachments` writes zero into each box. Zero bits are zero in every format's
/// clear value.
///
/// # Safety
///
/// As [`record_zeroing`]'s.
unsafe fn record_clears(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    boxes: impl IntoIterator<Item = TexelBox>,
    held: &mut ClearObjects,
) -> Result<(), Error> {
    let mut slices: BTreeMap<(u32, u32), Vec<vk::Rect2D>> = BTreeMap::new();
    for b in boxes {
        let rect = vk::Rect2D {
            offset: vk::Offset2D {
                x: b.origin[0] as i32,
                y: b.origin[1] as i32,
            },
            extent: vk::Extent2D {
                width: b.size[0],
                height: b.size[1],
            },
        };
        slices.entry((b.level, b.layer)).or_default().push(rect);
    }
    if slices.is_empty() {
        return Ok(());
    }
    let plan = texture.image_plan()?;
    let aspect = image_aspects(plan.format);
    let attachments = [vk::ClearAttachment {
        aspect_mask: aspect.into(),
        color_attachment: 0,
        clear_value: vk::ClearValue::default(),
    }];
    let raw = device.shared().vk();
    for ((level, layer), rects) in slices {
        let extent = [plan.extent[0], plan.extent[1]].map(|side| (side >> level).max(1));
        let target = Target {
            texture: texture.clone(),
            level,
            layer,
            layers: 1,
            format: plan.format,
            samples: plan.samples.as_raw(),
            extent,
        };
        let depth_stencil = !aspect.contains(ImageAspectFlags::COLOR);
        let (pass, framebuffer, _) =
            render_target::reach_pass(device, &target, depth_stencil, extent)?;
        let area = bounding(&rects);
        let clears: Vec<vk::ClearRect> = rects
            .iter()
            .map(|&rect| vk::ClearRect {
                rect,
                base_array_layer: 0,
                layer_count: 1,
            })
            .collect();
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass.raw)
            .framebuffer(framebuffer.raw)
            .render_area(area);
        // SAFETY: the caller's; the pass and its framebuffer are live until the command buffer
        // completes (held), the framebuffer's one attachment the image's level and layer, in
        // `GENERAL`, the pass's layout; the area and every rectangle lie in the level, and each
        // rectangle in the area, of the framebuffer's one layer; the aspects are the image's.
        unsafe {
            raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            raw.cmd_clear_attachments(command_buffer, &attachments, &clears);
            raw.cmd_end_render_pass(command_buffer);
        }
        held.0.push((pass, framebuffer));
    }
    exit_guard::called(Call::CmdBeginRenderPass);
    exit_guard::called(Call::CmdClearAttachments);
    exit_guard::called(Call::CmdEndRenderPass);
    Ok(())
}

/// The rectangle that bounds `rects` (at least one).
fn bounding(rects: &[vk::Rect2D]) -> vk::Rect2D {
    let start = |r: &vk::Rect2D| [r.offset.x, r.offset.y];
    let end = |r: &vk::Rect2D| {
        [
            r.offset.x + r.extent.width as i32,
            r.offset.y + r.extent.height as i32,
        ]
    };
    let low = rects
        .iter()
        .map(start)
        .fold([i32::MAX; 2], |a, b| [a[0].min(b[0]), a[1].min(b[1])]);
    let high = rects
        .iter()
        .map(end)
        .fold([i32::MIN; 2], |a, b| [a[0].max(b[0]), a[1].max(b[1])]);
    vk::Rect2D {
        offset: vk::Offset2D {
            x: low[0],
            y: low[1],
        },
        extent: vk::Extent2D {
            width: (high[0] - low[0]) as u32,
            height: (high[1] - low[1]) as u32,
        },
    }
}

/// A barrier from every command before to every command after: all memory writes made
/// available and visible.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass.
pub(crate) unsafe fn record_full_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // SAFETY: the caller's.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl Device {
    /// Whether a function may read a sparse texture's residency from its image (the device
    /// enables `shaderResourceResidency`, so a slot without a tile map takes the sparse arm): an
    /// ordinary image would answer resident everywhere there, so every sparse texture must be a
    /// sparse image.
    pub(crate) fn reads_image_residency(&self) -> bool {
        self.compile_device()
            .profile()
            .enabled()
            .features
            .contains(&VkFeature::ShaderResourceResidency)
    }
}

/// A box of a texture's image a command writes: a level, its layers, and the texels from `origin`
/// of `size`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Written {
    pub(crate) level: u32,
    pub(crate) layers: Range<u32>,
    pub(crate) origin: [u32; 3],
    pub(crate) size: [u32; 3],
}

impl Written {
    /// The box a copy region covers.
    pub(crate) fn of(region: &ImageRegion) -> Written {
        Written {
            level: region.level,
            layers: region.layer..region.layer + region.layers,
            origin: region.origin,
            size: region.size,
        }
    }
}

/// A blit's or a render pass's writes of a sparse texture: Metal drops a blit's write to an unmapped
/// tile and a pass's store (a Metal sparse probe on the M3 Max and an M4, single-sampled and
/// multisampled), so where the command wrote an unmapped tile inside memory the texture holds the
/// runtime zeroes it again right after, recorded as the batch is submitted, with the mappings the
/// batch applied (`Deferred::SparseScrub`). A write into a device unit that holds no memory is the
/// device's (dropped: Vulkan discards a write to an unbound block).
pub(crate) struct Scrub {
    texture: Texture,
    written: Vec<Written>,
}

impl Scrub {
    /// The scrub of `written` of `texture` (made only for one), or `None` for a texture that is
    /// not sparse.
    pub(crate) fn of(texture: &Texture, written: impl FnOnce() -> Vec<Written>) -> Option<Scrub> {
        texture.sparse_image().map(|_| Scrub {
            texture: texture.clone(),
            written: written(),
        })
    }

    /// The scrubs of a render pass's writes of `targets` (its attachments, resolve targets and the
    /// textures copied into after it) over its render area `extent` from the origin: Metal drops a
    /// pass's stores into unmapped tiles, of a clear, a draw and a resolve alike (a Metal sparse
    /// probe on the M3 Max and an M4), as a blit's, so each sparse texture is scrubbed after the
    /// pass. Inside the pass, the draws see what the draws before them stored,
    /// as in Metal's tile memory.
    pub(crate) fn of_pass<'a>(
        targets: impl IntoIterator<Item = &'a Target>,
        extent: [u32; 2],
    ) -> Vec<Scrub> {
        targets
            .into_iter()
            .filter_map(|target| {
                Scrub::of(&target.texture, || {
                    vec![Written {
                        level: target.level,
                        layers: target.layer..target.layer + target.layers,
                        origin: [0; 3],
                        size: [extent[0], extent[1], 1],
                    }]
                })
            })
            .collect()
    }

    /// Record into `command_buffer` the zeroing of the unmapped units the writes touched inside
    /// memory the texture holds, as mapped now, between barriers from everything before and to
    /// everything after: what its clears recorded with, for the command buffer to hold until it
    /// completes.
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `device` recording outside a render pass; the
    /// texture is live in the `GENERAL` layout, and its mappings are those the batch the scrub is
    /// in applied.
    pub(crate) unsafe fn record(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<ClearObjects, Error> {
        let mut held = ClearObjects::default();
        let Some(image) = self.texture.sparse_image() else {
            return Ok(held);
        };
        let boxes = {
            let state = lock(&image.state);
            let unmapped: BTreeSet<MetalUnit> = self
                .written
                .iter()
                .flat_map(|written| image.grid.units_within(written))
                .filter(|unit| !state.mapped.contains(unit))
                .collect();
            image
                .grid
                .unmapped_boxes(unmapped, |unit| image.held(unit, &state))
        };
        let raw = device.shared().vk();
        // SAFETY: the caller's; each box lies in memory the texture holds.
        unsafe {
            record_full_barrier(raw, command_buffer);
            record_zeroing(device, command_buffer, &self.texture, boxes, &mut held)?;
            record_full_barrier(raw, command_buffer);
        }
        exit_guard::called(Call::CmdPipelineBarrier);
        Ok(held)
    }
}

/// One texture's work after a batch's updates applied (indexed as the batch's textures): the
/// texels of the device units it bound to new memory, zeroed whole first; the texels of the Metal
/// units it unmapped inside memory the texture still holds, zeroed after the moves' copies; and
/// its tile map's bytes.
struct TextureWork {
    texture: Texture,
    fresh: BTreeSet<TexelBox>,
    unmapped: BTreeSet<TexelBox>,
    map: Vec<u8>,
}

/// One copy of a move's texels, from a box of the batch's texture `from` to one of `to`.
struct MoveCopy {
    from: usize,
    source: TexelBox,
    to: usize,
    destination: TexelBox,
}

/// A batch's work on its sparse textures and tile maps after its updates applied: each device unit
/// the batch bound to new memory zeroed whole; each move's texels copied from its source tile to
/// its destination, one tile at a time in the batch's order, as Metal's move takes them; each Metal
/// unit it unmapped inside memory the texture still holds zeroed; and each changed tile map
/// rewritten.
#[derive(Default)]
pub(crate) struct Mappings {
    work: Vec<TextureWork>,
    copies: Vec<MoveCopy>,
}

impl Mappings {
    /// The work of `textures` once the units `changed` names changed, the device units `fresh`
    /// names were bound to new memory, the steps `moved` moved Metal's units in order, and the
    /// device units `kept` stay bound through the batch for the copies.
    pub(crate) fn of(
        textures: &[&Texture],
        changed: impl IntoIterator<Item = (usize, MetalUnit)>,
        fresh: impl IntoIterator<Item = (usize, VkUnit)>,
        (moved, kept): (&[Moved], &BTreeSet<(usize, VkUnit)>),
    ) -> Mappings {
        let mut touched: Vec<BTreeSet<MetalUnit>> = vec![BTreeSet::new(); textures.len()];
        for (index, unit) in changed {
            touched[index].insert(unit);
        }
        let mut binds: Vec<BTreeSet<VkUnit>> = vec![BTreeSet::new(); textures.len()];
        for (index, unit) in fresh {
            binds[index].insert(unit);
        }
        let images: Vec<Option<&SparseImage>> = textures
            .iter()
            .map(|texture| texture.sparse_image())
            .collect();
        let mut work = Vec::new();
        for (index, (texture, units)) in textures.iter().zip(touched).enumerate() {
            let (fresh, unmapped, map) = match texture.sparse_image() {
                Some(image) if !units.is_empty() => {
                    let state = lock(&image.state);
                    let bound = &binds[index];
                    // Held once the batch's binds apply: what the plan bound (`bind.rs`).
                    let held = |unit: VkUnit| {
                        !bound.contains(&unit)
                            && (image.emulated()
                                || image.grid.wanted(unit, &state.mapped)
                                || kept.contains(&(index, unit)))
                    };
                    let unmapped = units
                        .iter()
                        .copied()
                        .filter(|unit| !state.mapped.contains(unit));
                    let fresh = image.grid.bound_boxes(bound);
                    let unmapped = image.grid.unmapped_boxes(unmapped, held);
                    (fresh, unmapped, image.tile_map_bytes(&state.mapped))
                }
                _ => (BTreeSet::new(), BTreeSet::new(), Vec::new()),
            };
            work.push(TextureWork {
                texture: (*texture).clone(),
                fresh,
                unmapped,
                map,
            });
        }
        let mut copies = Vec::new();
        for step in moved {
            let (Some(from), Some(to)) = (images[step.from], images[step.to]) else {
                continue;
            };
            copies.extend(
                from.grid
                    .move_boxes(step.source, &to.grid, step.destination)
                    .into_iter()
                    .map(|(source, destination)| MoveCopy {
                        from: step.from,
                        source,
                        to: step.to,
                        destination,
                    }),
            );
        }
        Mappings { work, copies }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.copies.is_empty()
            && self.work.iter().all(|work| {
                work.fresh.is_empty() && work.unmapped.is_empty() && work.map.is_empty()
            })
    }

    /// What the work uses, for the hazard tracker: each texture it zeroes or copies into written
    /// whole, each it only copies from read whole, in the transfer stage, or in the attachment
    /// stages where its clears zero it; and each tile map it rewrites written whole.
    pub(crate) fn uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        let mut reads = vec![false; self.work.len()];
        let mut moved_into = vec![false; self.work.len()];
        for copy in &self.copies {
            reads[copy.from] = true;
            moved_into[copy.to] = true;
        }
        let mut uses = Vec::new();
        for ((work, read), moved_into) in self.work.iter().zip(reads).zip(moved_into) {
            let zeroes = !work.fresh.is_empty() || !work.unmapped.is_empty();
            let mut stages = vk::PipelineStageFlags2::empty();
            let mut access = vk::AccessFlags2::empty();
            if read || moved_into {
                stages |= vk::PipelineStageFlags2::TRANSFER;
            }
            if read {
                access |= vk::AccessFlags2::TRANSFER_READ;
            }
            if moved_into {
                access |= vk::AccessFlags2::TRANSFER_WRITE;
            }
            let image = work.texture.sparse_image();
            let plan = work.texture.image_plan()?;
            match image.map(|image| image.zeroing) {
                Some(Zeroing::Clears) if zeroes => {
                    stages |= ATTACHMENT_STAGES;
                    access |= match image_aspects(plan.format).contains(ImageAspectFlags::COLOR) {
                        true => {
                            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        }
                        false => {
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        }
                    };
                }
                _ if zeroes => {
                    stages |= vk::PipelineStageFlags2::TRANSFER;
                    access |= vk::AccessFlags2::TRANSFER_WRITE;
                }
                _ => {}
            }
            if !access.is_empty() {
                let access = Access::new(stages, access);
                let write = moved_into || zeroes;
                uses.push(
                    work.texture
                        .image_use(0..plan.levels, 0..plan.layers, access, write)?,
                );
            }
            let tile_map = work.texture.sparse_image().and_then(SparseImage::tile_map);
            if let (false, Some(tile_map)) = (work.map.is_empty(), tile_map) {
                let length = tile_map.buffer.length();
                uses.push(crate::blit::writes(&tile_map.buffer, 0..length));
            }
        }
        Ok(uses)
    }

    /// Record the work into `command_buffer`, between barriers from everything before and to
    /// everything after, each step after the one before it: the new units zeroed, each move's
    /// copy in order, the unmapped units zeroed, and each tile map's bytes copied from `uploads`,
    /// the command buffer's arena (inside the scheduler's section, whose entry points hold
    /// copies, not buffer updates). What its clears recorded with, for the command buffer to hold
    /// until it completes.
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `device` recording outside a render pass; every
    /// texture is live in the `GENERAL` layout with the batch's binds applied, and the caller
    /// orders the command buffer after everything submitted before the batch on every queue.
    pub(crate) unsafe fn record(
        &self,
        device: &Device,
        uploads: &mut Uploads,
        command_buffer: vk::CommandBuffer,
    ) -> Result<ClearObjects, Error> {
        let raw = device.shared().vk();
        let mut held = ClearObjects::default();
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        exit_guard::called(Call::CmdPipelineBarrier);
        for work in &self.work {
            let fresh = work.fresh.iter().copied();
            // SAFETY: the caller's; each box lies in a unit the batch bound.
            unsafe { record_zeroing(device, command_buffer, &work.texture, fresh, &mut held)? };
        }
        for copy in &self.copies {
            // SAFETY: the caller's; each step's writes finish before the next reads them.
            unsafe {
                record_full_barrier(raw, command_buffer);
                record_move_copy(
                    raw,
                    command_buffer,
                    (&self.work[copy.from].texture, copy.source),
                    (&self.work[copy.to].texture, copy.destination),
                )?;
            }
            exit_guard::called(Call::CmdPipelineBarrier);
            exit_guard::called(Call::CmdCopyImage);
        }
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        for work in &self.work {
            let unmapped = work.unmapped.iter().copied();
            // SAFETY: the caller's; each box lies in memory the texture holds after the binds.
            unsafe { record_zeroing(device, command_buffer, &work.texture, unmapped, &mut held)? };
            let tile_map = work.texture.sparse_image().and_then(SparseImage::tile_map);
            if let (false, Some(tile_map)) = (work.map.is_empty(), tile_map) {
                let bytes = &work.map;
                let upload = uploads.upload(device, bytes, bytes.len() as u64)?;
                let (target, start) = tile_map.buffer.at(0);
                let copy = vk::BufferCopy {
                    src_offset: upload.offset(),
                    dst_offset: start,
                    size: bytes.len() as u64,
                };
                // SAFETY: the caller's; the upload holds the bytes and the map's buffer has
                // room for them; the command buffer holds the arena until it completes.
                unsafe { raw.cmd_copy_buffer(command_buffer, upload.raw(), target, &[copy]) };
                exit_guard::called(Call::CmdCopyBuffer);
            }
        }
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        Ok(held)
    }
}

/// Record into `command_buffer` the copy of box `source` of texture `from` to box `destination`
/// of texture `to`, both sparse textures, of one format and sample count, in the `GENERAL`
/// layout, each of the format's aspects; nothing where `from`'s image is not a copy's source.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass; both textures are live and both boxes
/// lie in their images, of one extent; the caller orders the copy.
unsafe fn record_move_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    (from, source): (&Texture, TexelBox),
    (to, destination): (&Texture, TexelBox),
) -> Result<(), Error> {
    let plan = from.image_plan()?;
    if !plan.usage.contains(ImageUsageFlags::TRANSFER_SRC) {
        return Ok(());
    }
    let offset = |origin: [u32; 3]| vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: origin[2] as i32,
    };
    let subresource = |aspect: ImageAspectFlags, level, layer| vk::ImageSubresourceLayers {
        aspect_mask: aspect.into(),
        mip_level: level,
        base_array_layer: layer,
        layer_count: 1,
    };
    let copies: Vec<vk::ImageCopy> = copy_units(plan.format, plan.pixel_format)
        .into_iter()
        .map(|(aspect, _, _)| vk::ImageCopy {
            src_subresource: subresource(aspect, source.level, source.layer),
            src_offset: offset(source.origin),
            dst_subresource: subresource(aspect, destination.level, destination.layer),
            dst_offset: offset(destination.origin),
            extent: vk::Extent3D {
                width: source.size[0],
                height: source.size[1],
                depth: source.size[2],
            },
        })
        .collect();
    // SAFETY: the caller's; each region lies in its image, which a copy reads (checked) or
    // writes (every sparse texture's image takes transfer writes, `Device::plan_sparse_texture`).
    unsafe {
        device.cmd_copy_image(
            command_buffer,
            from.raw_image()?,
            vk::ImageLayout::GENERAL,
            to.raw_image()?,
            vk::ImageLayout::GENERAL,
            &copies,
        );
    }
    Ok(())
}
