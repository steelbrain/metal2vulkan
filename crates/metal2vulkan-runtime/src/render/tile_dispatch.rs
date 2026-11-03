//! Tile dispatches in a render encoder (`dispatchThreadsPerTile:`) and the encoder's tile stage
//! table.
//!
//! - **Checks** (measured by our probe on an M3 Max and an M4): the state set is
//!   a tile pipeline's ("the renderPipelineState was not created with a tileDescriptor"); the
//!   threads per tile are not empty and fit the pass's tile ("threadsPerTile.width(32) must be
//!   <= 16"). Threads per tile past Metal's limit for the state, which Metal's validation stops,
//!   run nothing, as plain Metal runs them (an M4, and an M3 Max for threads also past the tile),
//!   and that limit is read first, so a dispatch past both it and the tile runs nothing rather
//!   than being refused; those within it but past the state's
//!   `max_total_threads_per_threadgroup`, where this device's limit is lower, are refused typed by
//!   the binder. A depth other than 1, which Metal's validation stops ("threadsPerTile.depth(2)
//!   must be 1"), runs as 1, as plain Metal runs it (a Metal probe on an M4: 8x8 threads per tile
//!   of depth 2 and 4 ran 64 threads a tile, each `z` 0 and `threads_per_threadgroup.z` 1). Threads
//!   per tile past the tile but within Metal's limit stay refused: plain Metal runs them at the
//!   pixels of the tiles that follow, so a pixel within the threads per tile of more than one
//!   tile's origin runs once for each of them, on overlapping threadgroups the runtime does not
//!   model (`DrawProblem::ThreadsPerTilePastTile` holds the counted reading). Serving such a
//!   dispatch as the tile's own threads would store once where Metal stores repeatedly. A pass is
//!   single-sampled: the compile layer stages one sample a pixel, so a
//!   tile state of more samples is not made (`Unsupported`), and the encoder takes no state of
//!   another sample count than its pass's. The state's color formats may differ from the pass's
//!   past the last color its function reaches through its implicit imageblock, which plain Metal
//!   runs as it runs matching ones (a Metal probe on an M4); before it, or with an explicit
//!   imageblock, the encoder refuses the state (`DrawProblem::AttachmentFormat`).
//! - **What runs** is what Metal runs: one threadgroup of the threads per tile on each tile of
//!   the pass's tile size covering the render area, `ceil(area / tile)` of them, whose stores
//!   reach each tile's pixels past the render area as far as the attachments go (measured: a
//!   40x24 area of 16x16 tiles in 64x64 attachments stores 48x32 pixels). Scissor rectangles and
//!   viewports do not apply. The binder plans it as a compute dispatch of the tile function
//!   (`BindState::render_area`, the whole tiles covered, and `imageblock_size`, the tile).
//! - **Cells past the attachments** (measured by the command replays on an M4: `RGBA16Float`
//!   attachments of 282x88 in 32x32 tiles, `RGBA8Unorm` ones of 10x20 in 16x16 tiles). When the
//!   pass begins, a partial tile's cells past an attachment hold the attachment's texel nearest
//!   each, clamped to its edge, where the pass loads it (as loaded: a draw before the dispatch that
//!   rewrites the edge leaves them), its clear color where the pass clears it, and zero where it
//!   loads it `DontCare`. A tile function's writes to them last, and the pass's later dispatches
//!   read them. Each scratch covers the whole tiles: its cells past its attachment are set so
//!   before the pass's first instance ([`ScratchEdges`]), and never copied to or from the
//!   attachment, so the pass's dispatches carry them along.
//! - **Over a region** (`dispatchThreadsPerTile:inRegion:` and its form with a render target
//!   array index, SPI; measured by our probes on an M3 Max and an M4,
//!   identically, with no validation stop but the threads per tile's, as above). The tiles that
//!   run are those from `x / tile` to the end `(x + w + tile - 1) / tile`, in 32-bit arithmetic
//!   that wraps, each origin and size cut to its low 32 bits, and cut to the pass's tiles
//!   ([`region_tiles`]): a region of no width runs the tile its origin lies inside unless the
//!   origin is on a tile's edge, and an end that wraps runs to the pass's last tile or nothing.
//!   Each runs as in the whole dispatch: its `threadgroup_position_in_grid` is its tile's, the
//!   grid is the whole pass's, and its imageblock is the whole tile. The region's `z` and depth
//!   are not read. A layered pass's region form runs every layer and the indexed form the layer
//!   of its index, or layer 0 for an index at or past the layers; the planner refuses a dispatch
//!   in a layered pass (below), so the passes it runs in have one layer. The binder plans it as the range of the whole dispatch's threadgroups
//!   (`Dispatch::ThreadgroupRange`), and the scratch copies cover its tiles alone.
//! - **Order.** A dispatch sees the pass's loads and clears and every draw and dispatch before
//!   it, and the draws after it see what it stored (measured). The render pass instance before
//!   it ends there, and the pass's next draw records into a new one that loads what the dispatch
//!   stored (`render.rs`'s instances); `end_encoding` records each dispatch between the
//!   instances it falls between, and a pass whose last command is a dispatch ends with one more
//!   instance, of no draws, which runs the pass's store actions.
//! - **Its pipelines.** A region's `VkPipeline` the tile function's variant has not made is made
//!   on the device's pool while the dispatch goes on, and the pass records once it is made
//!   (`render/pending_tile.rs`).
//! - **Attachments.** The tile function stages each attachment it reaches through a storage view
//!   in its staging format (an integer one, `WriteSource::TileAttachment`), which an
//!   attachment's own image may lack: the dispatch reads and writes a scratch image of the
//!   attachment's format, viewed in the staging format, which holds the covered tiles' cells.
//!   The pixels the attachment holds are copied in from it before the dispatch and back after
//!   it, between barriers from everything before and to everything after. A pass reuses one
//!   scratch per attachment and format across its dispatches. Both costs are announced (`tile-dispatch-scratch`, and
//!   `render-pass-split` for the instances).
//!
//! - **Argument buffers** (`render/arguments.rs`): the function reaches what the encoder made
//!   resident in the tile stage, and its variant's set 1 is the command buffer's heap set of the
//!   variant, as a compute dispatch's, bound with set 0 when the dispatch is recorded.
//! - **Shadowed textures** (`shadow.rs`): a dispatch is a command of its own between the instances
//!   before and after it, so it gathers the shadowed textures it uses, in the tile table and
//!   through the residency, and its uses of buffers' bytes, as an instance does
//!   (`render/shadows.rs`), and `end_encoding` plans their copies in the pass's order, instances
//!   and dispatches alike: what a dispatch needs copied in or back goes between the instance before
//!   it and it, each copy between barriers.
//!
//! Not built yet (`NotImplemented`): a tile function whose explicit imageblock is aliased onto
//! its implicit one past the bytes the attachments pack to, in a pass that keeps its imageblock
//! storage (`TileAndFragmentImageblocks`), and one whose explicit imageblock is of another size
//! than the storage's (`ImageblockMasters`, `fragment_imageblock.rs`). The planner refuses a
//! dispatch in a layered pass, what Metal's tile function reads there being unread
//! (`NotImplemented(TileShading)`).
//!
//! - **Threadgroup memory** (`setThreadgroupMemoryLength:offset:atIndex:`): a tile function's
//!   threadgroup arguments are its tile's memory, which the pass keeps across its tile dispatches
//!   and draws in one buffer of every tile's, all zero when the pass begins
//!   (`render/tile_memory.rs`).
//! - **The imageblock**: a tile function's explicit imageblock that stands alone is the pass's
//!   imageblock storage, which its draws' fragment imageblocks are in, made at the pass's first
//!   draw or dispatch that reaches it, all zero when the pass begins: the dispatch stages its
//!   tiles' cells from the storage and back (`fragment_imageblock.rs`), between the barriers every
//!   dispatch records between. In a pass of fewer imageblock bytes per sample than the state's,
//!   which Metal's validation stops, plain Metal keeps the imageblock's bytes below the pass's
//!   rounded up to 8 and reads every other word as zero, dropping its writes; the dispatch runs the
//!   variant that does so (`BindState::kept_imageblock_bytes`).

use super::pending_tile::{TileCall, TilePipelines};
use super::replay::Samplers;
use super::shadows::InstanceShadows;
use super::RenderCommandEncoder;
use super::{draw_problem, DrawProblem, RenderLowering};
use super::{RenderCall, Table};
use crate::address_map;
use crate::attachment_stand_in;
use crate::blit::TRANSFER;
use crate::buffer::Buffer;
use crate::clear_reach;
use crate::command_buffer::CommandBuffer;
use crate::degradation::{BoundFormats, CompiledPipeline};
use crate::device::functions::Vk;
use crate::encoder::EncoderCore;
use crate::error::{internal, invalid, Error, Problem, RuntimeGap};
use crate::execution::{ExecutedPipeline, Execution};
use crate::fragment_imageblock::PassImageblock;
use crate::function_table::VisibleFunctionTable;
use crate::heap_sets;
use crate::image_plan::ImagePlan;
use crate::pipeline::{BufferForms, RawPipeline, RenderPipelineState, RenderStateKind};
use crate::render_pass::{PassLayout, Target};
use crate::render_values::LoadAction;
use crate::resource::ResourceKey;
use crate::sampler::SamplerState;
use crate::shadow;
use crate::texture::Texture;
use crate::texture_descriptor::TextureDescriptor;
use crate::tracker::{Access, Barrier, Use};
use crate::values::{Region, ResourceOptions, Size};
use crate::{binding, Device};
use ash::vk;
use metal2vulkan::bind::{
    BindState, Dispatch, HeapPlacement, ImageblockSize, ViewSetup, WriteSource,
};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::{imageblock_sample_length, FormatMapping};
use metal2vulkan::metal::{RenderStages, TextureType, TextureUsage};
use metal2vulkan::pipeline::Stage;
use metal2vulkan::vk::{ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags};
use std::sync::{Arc, OnceLock};

/// One tile dispatch of a pass, made at its call; `end_encoding` records it after the render
/// pass instance before it.
pub(super) struct TileDispatch {
    /// The pass's render pass instances before it.
    pub(super) after: usize,
    /// The color index and scratch of each attachment the dispatch stages.
    staged: Vec<(u32, Target)>,
    /// The pixels the dispatch's tiles cover, copied into each scratch and back where its
    /// attachment holds them: their origin and extent, inside the scratch images' extent (the
    /// pixels every dispatch of the pass covers, from the origin).
    copied: [[u32; 2]; 2],
    layout: vk::PipelineLayout,
    sets: Vec<vk::DescriptorSet>,
    /// The dynamic offsets set 0 is bound with (`binding::dynamic_writes`).
    offsets: Vec<u32>,
    /// Set 0's writes, where the variant pushes it (`pipeline::objects::pushes_set_zero`):
    /// pushed with the device's push command (`Vk::push_descriptor`).
    pushed: Vec<binding::SetWrite>,
    /// Set 1, the command buffer's heap set of the variant, when the variant has one.
    heap: Option<vk::DescriptorSet>,
    /// Each region's push constants and group count.
    regions: Vec<(Vec<binding::PushWrite>, [u32; 3])>,
    /// Its regions' `VkPipeline`s, made at the call or being made (`render/pending_tile.rs`).
    pub(super) pipelines: TilePipelines,
    /// The shadowed textures it uses and its uses of buffers' bytes (the module docs).
    pub(super) shadows: InstanceShadows,
}

/// One staged attachment's copies, ready to record: its image, the copy into the scratch and
/// back, and the scratch's image.
pub(super) type Staging = (vk::Image, (vk::ImageCopy, vk::ImageCopy), vk::Image);

impl TileDispatch {
    /// The copies of each staged attachment of `layout` (the pass's, as it ends: a stand-in that
    /// arrived after the dispatch is what the pass renders) into its scratch and back, over the
    /// covered pixels it holds (none where the dispatch's tiles lie past it).
    pub(super) fn copies(&self, layout: &PassLayout) -> Result<Vec<Staging>, Error> {
        let mut copies = Vec::with_capacity(self.staged.len());
        for (color, scratch) in &self.staged {
            let own = layout
                .colors
                .get(*color as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a staged attachment the pass does not have"))?;
            if let Some(pair) = region(&own.target, scratch, self.copied) {
                copies.push((
                    own.target.texture.raw_image()?,
                    pair,
                    scratch.texture.raw_image()?,
                ));
            }
        }
        Ok(copies)
    }

    /// Record the dispatch in `command_buffer` with `pipelines`, its regions' (its
    /// [`TilePipelines`] made), outside any render pass instance: each staged attachment copied
    /// into its scratch (`copies`, from [`TileDispatch::copies`]), the dispatch, and the scratch
    /// copied back, between barriers from everything before and to everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass; every object named lives until it
    /// completes; the images are in the `GENERAL` layout; `pipelines` are made with the
    /// variant's layout, one per region.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        copies: &[Staging],
        pipelines: &[Arc<RawPipeline>],
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            for (own, (into, _), scratch) in copies {
                raw.cmd_copy_image(
                    command_buffer,
                    *own,
                    vk::ImageLayout::GENERAL,
                    *scratch,
                    vk::ImageLayout::GENERAL,
                    std::slice::from_ref(into),
                );
            }
            Barrier::everything().record(raw, command_buffer);
            if let Some(push) = raw.push_descriptor().filter(|_| !self.pushed.is_empty()) {
                binding::push_set(
                    push,
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    &self.pushed,
                );
            } else if !self.sets.is_empty() {
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &self.sets,
                    &self.offsets,
                );
            }
            if let Some(heap) = self.heap {
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    1,
                    &[heap],
                    &[],
                );
            }
            let regions = self.regions.iter().zip(pipelines);
            for (index, ((push, [x, y, z]), pipeline)) in regions.enumerate() {
                if index > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                raw.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.raw);
                for (stages, offset, bytes) in push {
                    raw.cmd_push_constants(command_buffer, self.layout, *stages, *offset, bytes);
                }
                raw.cmd_dispatch(command_buffer, *x, *y, *z);
            }
            Barrier::everything().record(raw, command_buffer);
            for (own, (_, back), scratch) in copies {
                raw.cmd_copy_image(
                    command_buffer,
                    *scratch,
                    vk::ImageLayout::GENERAL,
                    *own,
                    vk::ImageLayout::GENERAL,
                    std::slice::from_ref(back),
                );
            }
            Barrier::everything().record(raw, command_buffer);
        }
    }
}

/// The copies between attachment `own` and its `scratch` over the pixels `[origin, extent]` the
/// attachment holds: into the scratch, then back; `None` where it holds none of them.
// Cut to the attachment: the scratch's cells past it are the pass's ([`ScratchEdges`]).
fn region(
    own: &Target,
    scratch: &Target,
    [origin, extent]: [[u32; 2]; 2],
) -> Option<(vk::ImageCopy, vk::ImageCopy)> {
    let layers = |target: &Target| vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: target.level,
        base_array_layer: target.layer,
        layer_count: 1,
    };
    let [width, height] = [0, 1].map(|axis| {
        let end = origin[axis].saturating_add(extent[axis]);
        end.min(own.extent[axis]).saturating_sub(origin[axis])
    });
    if width == 0 || height == 0 {
        return None;
    }
    // Inside the attachment, whose extent fits an `i32`.
    let offset = vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: 0,
    };
    let copy = |from: &Target, to: &Target| vk::ImageCopy {
        src_subresource: layers(from),
        src_offset: offset,
        dst_subresource: layers(to),
        dst_offset: offset,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    Some((copy(own, scratch), copy(scratch, own)))
}

/// The cells of a pass's tile scratches past their attachments, set before the pass as Metal's
/// tile memory holds them when the pass begins (the module docs): an attachment the pass loads
/// (`Load`, as the layout has it once the pass's clears done ahead of it load), its texels at its
/// edges repeated outward as they stand before the pass's first instance; one it clears, the
/// clear value; any other, its format's zero (`clear_reach.rs`). Only the pass's tile dispatches
/// write them after, and each later dispatch reads what the one before left: a scratch serves
/// every dispatch of its pass.
pub(super) struct ScratchEdges {
    /// The copies of each loaded attachment's edge texels into its scratch: the attachment's
    /// image, the scratch's, and the regions.
    edges: Vec<(vk::Image, vk::Image, Vec<vk::ImageCopy>)>,
    /// The copies within each such scratch of its cells past both edges, from those the edges'
    /// copies filled.
    corners: Vec<(vk::Image, Vec<vk::ImageCopy>)>,
    /// The clears of each other scratch: its image and value.
    clears: Vec<(vk::Image, vk::ClearColorValue)>,
    uses: Vec<Use<ResourceKey>>,
}

impl ScratchEdges {
    /// The settings of the cells past their attachments of each of a pass's tile `scratch`
    /// images (by color index, as the encoder made them) in `layout`, the pass's as its
    /// instances' passes are made from it; `None` when no scratch has such cells.
    pub(super) fn plan(
        layout: &PassLayout,
        scratch: &[(u32, Target, metal2vulkan::vk::Format)],
    ) -> Result<Option<ScratchEdges>, Error> {
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let both = Access::new(
            TRANSFER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        let mut settings = ScratchEdges {
            edges: Vec::new(),
            corners: Vec::new(),
            clears: Vec::new(),
            uses: Vec::new(),
        };
        // The colors whose attachment the copies read, each image one use.
        let mut read_colors = Vec::new();
        for (color, held, _) in scratch {
            let own = layout
                .colors
                .get(*color as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a staged attachment the pass does not have"))?;
            let [width, height] = own.target.extent;
            let [across, down] = held.extent;
            if width >= across && height >= down {
                continue;
            }
            let image = held.texture.raw_image()?;
            settings
                .uses
                .push(held.texture.image_use(0..1, 0..1, both, true)?);
            match own.load {
                LoadAction::Load => {
                    let target = &own.target;
                    let (edges, corner) = edge_copies(target, held);
                    if !read_colors.contains(color) {
                        read_colors.push(*color);
                        let level = target.level..target.level + 1;
                        let layer = target.layer..target.layer + 1;
                        let read = target.texture.image_use(level, layer, read, false)?;
                        settings.uses.push(read);
                    }
                    settings
                        .edges
                        .push((target.texture.raw_image()?, image, edges));
                    if !corner.is_empty() {
                        settings.corners.push((image, corner));
                    }
                }
                LoadAction::Clear => settings.clears.push((image, own.clear)),
                _ => {
                    let zero = clear_reach::dont_care_clear(own.target.texture.pixel_format());
                    settings.clears.push((image, zero));
                }
            }
        }
        let none = settings.edges.is_empty() && settings.clears.is_empty();
        Ok((!none).then_some(settings))
    }

    /// Record the settings in `core`'s command buffer, between barriers from everything before
    /// and to everything after: the edges' copies and the clears, then the corners' copies.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ScratchEdges {
            edges,
            corners,
            clears,
            uses,
        } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            let whole = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            // SAFETY: the encoder's command buffer is recording outside a render pass; the images
            // are live until it completes (the encoder retains the attachments and the scratch
            // images); every image rests in `GENERAL` and has the transfer usage its copy or clear
            // needs (an attachment's checked as its scratch was made, `scratch`); each region lies
            // in both its images, and a corner's source row lies above its destination rows.
            unsafe {
                for (from, to, regions) in &edges {
                    device.cmd_copy_image(raw, *from, general, *to, general, regions);
                }
                for (image, value) in &clears {
                    device.cmd_clear_color_image(raw, *image, general, value, &[whole]);
                }
                if !corners.is_empty() {
                    Barrier::everything().record(device, raw);
                    for (image, regions) in &corners {
                        device.cmd_copy_image(raw, *image, general, *image, general, regions);
                    }
                }
            }
        });
        core.barrier(Barrier::everything());
    }
}

/// The copies of loaded attachment `own`'s texels at its edges into the cells of `scratch` past
/// it, each cell taking the texel nearest it (clamped to the edge): its last column across the
/// cells past its right edge and its last row down those past its bottom edge; then the copies
/// within the scratch of the cells past both, each row from the row above them the first copies
/// filled.
fn edge_copies(own: &Target, scratch: &Target) -> (Vec<vk::ImageCopy>, Vec<vk::ImageCopy>) {
    let [width, height] = own.extent;
    let [across, down] = scratch.extent;
    let layers = |target: &Target| vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: target.level,
        base_array_layer: target.layer,
        layer_count: 1,
    };
    // Inside the scratch, whose extent (whole tiles over the render area) fits an `i32`.
    let offset = |[x, y]: [u32; 2]| vk::Offset3D {
        x: x as i32,
        y: y as i32,
        z: 0,
    };
    let copy = |from: &Target, to: &Target, at: [u32; 2], into: [u32; 2], [w, h]: [u32; 2]| {
        vk::ImageCopy {
            src_subresource: layers(from),
            src_offset: offset(at),
            dst_subresource: layers(to),
            dst_offset: offset(into),
            extent: vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            },
        }
    };
    let [held_width, held_height] = [width.min(across), height.min(down)];
    let mut edges = Vec::new();
    for x in width..across {
        edges.push(copy(own, scratch, [width - 1, 0], [x, 0], [1, held_height]));
    }
    for y in height..down {
        edges.push(copy(own, scratch, [0, height - 1], [0, y], [held_width, 1]));
    }
    let mut corner = Vec::new();
    if width < across {
        for y in height..down {
            let row = [across - width, 1];
            corner.push(copy(scratch, scratch, [width, height - 1], [width, y], row));
        }
    }
    (edges, corner)
}

/// The scratch a dispatch stages attachment `own` in, viewed in `staging`, over `area`: an image
/// of the attachment's format, storage in the staging view, copied to and from (the module
/// docs). Refused (`Unsupported`) where the device lacks storage images of the staging format,
/// and (`NotImplemented(Render)`) where the attachment's image does not copy both ways, or is not
/// 2D (`render_pass::not_2d`).
fn scratch(
    maker: &CommandBuffer,
    own: &Target,
    staging: metal2vulkan::vk::Format,
    area: [u32; 2],
) -> Result<Target, Error> {
    let device = maker.device();
    // A 3D attachment's copies address its planes by depth, and a 1D one's a 1D image, which
    // these do not.
    if crate::render_pass::not_2d(own) {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    let profile = device.compile_device().profile();
    if !profile.supports_format(staging, FormatUsage::StorageImage) {
        return Err(Error::Unsupported(Unsupported::needs(
            Feature::TileShading,
            Capability::Format {
                format: staging,
                usage: FormatUsage::StorageImage,
            },
        )));
    }
    attachment_stand_in::check_copies(own)?;
    let format = own.format;
    let mut flags = ImageCreateFlags::empty();
    let mut view_formats = Vec::new();
    if staging != format {
        flags |= ImageCreateFlags::MUTABLE_FORMAT;
        view_formats = vec![format, staging];
        if !profile.supports_format(format, FormatUsage::StorageImage) {
            flags |= ImageCreateFlags::EXTENDED_USAGE;
        }
    }
    let pixel_format = own.texture.pixel_format();
    let [width, height] = area;
    let descriptor = TextureDescriptor {
        texture_type: TextureType::Type2D,
        pixel_format,
        width: width.into(),
        height: height.into(),
        usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags,
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers: 1,
        samples: metal2vulkan::vk::SampleCountFlags::from_raw(1),
        usage: ImageUsageFlags::STORAGE
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST,
        view_formats,
        mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers: 1,
        format,
        samples: 1,
        extent: area,
    })
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::tile_width`].
    pub(crate) fn tile_width(&self) -> Result<u64, Error> {
        self.layout.tile_size().map(|[width, _]| width.into())
    }

    /// The lowering of [`RenderCommandEncoder::tile_height`].
    pub(crate) fn tile_height(&self) -> Result<u64, Error> {
        self.layout.tile_size().map(|[_, height]| height.into())
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_buffer`].
    pub(crate) fn set_tile_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::Buffer(Table::Tile, buffer, offset, index)
        });
        self.tile
            .set_buffer(&mut self.core, "set_tile_buffer", buffer, offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_buffers`].
    pub(crate) fn set_tile_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            RenderCall::Buffers(
                Table::Tile,
                log.buffer_span(buffers),
                log.word_span(offsets),
                first,
            )
        });
        self.tile
            .set_buffers(&mut self.core, "set_tile_buffers", buffers, offsets, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_buffer_offset`].
    pub(crate) fn set_tile_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        self.log(|_| RenderCall::BufferOffset(Table::Tile, offset, index));
        self.tile
            .set_buffer_offset(&mut self.core, "set_tile_buffer_offset", offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_bytes`].
    pub(crate) fn set_tile_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.log(|log| RenderCall::Bytes(Table::Tile, log.byte_span(bytes), index));
        self.tile
            .set_bytes(&mut self.core, "set_tile_bytes", bytes, None, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_texture`].
    pub(crate) fn set_tile_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let texture = texture.map(|texture| log.hold_texture(texture));
            RenderCall::Texture(Table::Tile, texture, index)
        });
        self.tile
            .set_textures(&mut self.core, "set_tile_texture", &[texture], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_textures`].
    pub(crate) fn set_tile_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::Textures(Table::Tile, log.texture_span(textures), first));
        self.tile
            .set_textures(&mut self.core, "set_tile_textures", textures, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_sampler_state`].
    pub(crate) fn set_tile_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
            RenderCall::Sampler(Table::Tile, sampler, index)
        });
        self.tile.set_samplers(
            &mut self.core,
            "set_tile_sampler_state",
            &[sampler],
            None,
            index,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_sampler_states`].
    pub(crate) fn set_tile_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Samplers(Table::Tile, Samplers::of(samplers, clamps, first)));
        self.tile.set_samplers(
            &mut self.core,
            "set_tile_sampler_states",
            samplers,
            clamps,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::dispatch_threads_per_tile`].
    pub(crate) fn dispatch_threads_per_tile(&mut self, threads: Size) -> Result<(), Error> {
        self.log(|_| RenderCall::ThreadsPerTile(Box::new(threads)));
        self.dispatch_tiles("dispatch_threads_per_tile", threads, None)
    }

    /// The lowering of [`RenderCommandEncoder::dispatch_threads_per_tile_in_region`].
    pub(crate) fn dispatch_threads_per_tile_in_region(
        &mut self,
        threads: Size,
        region: Region,
        render_target_array_index: Option<u32>,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::ThreadsPerTileInRegion(Box::new((
                threads,
                region,
                render_target_array_index,
            )))
        });
        // Metal draws into layer `index` of a layered pass and into layer 0 for an index at or
        // past its layers; the planner refuses a dispatch in a layered pass, so every pass one runs
        // in has one layer, which every index draws into.
        let _ = render_target_array_index;
        self.dispatch_tiles("dispatch_threads_per_tile_in_region", threads, Some(region))
    }

    /// A tile dispatch of `threads` per tile on every tile of the pass, or on those `region`
    /// touches.
    fn dispatch_tiles(
        &mut self,
        call: &'static str,
        threads: Size,
        region: Option<Region>,
    ) -> Result<(), Error> {
        self.sync_heaps();
        let reads = match self.pipeline.as_ref().map(RenderPipelineState::kind) {
            Some(RenderStateKind::Tile(state)) if state.kernel().reads_texture_handles() => {
                RenderStages::TILE
            }
            _ => RenderStages::empty(),
        };
        self.dispatch_tiles_resident(call, threads, region)?;
        self.core.device().shared().counts.tile_dispatch_recorded();
        self.tile_dispatch_calls += 1;
        self.command_encoded(reads);
        Ok(())
    }

    /// [`dispatch_tiles`](Self::dispatch_tiles) with the textures the encoder's heaps hold
    /// resident.
    fn dispatch_tiles_resident(
        &mut self,
        call: &'static str,
        threads: Size,
        region: Option<Region>,
    ) -> Result<(), Error> {
        let state = match self.pipeline.as_ref().map(RenderPipelineState::kind) {
            None => return Err(invalid(call, Problem::NoPipelineState)),
            Some(RenderStateKind::Graphics(_)) => {
                return Err(draw_problem(call, DrawProblem::NotTileState))
            }
            Some(RenderStateKind::Tile(state)) => state.clone(),
        };
        let [width, height, depth] = threads
            .axes()
            .ok_or_else(|| invalid(call, Problem::SizeOutOfRange))?;
        if width == 0 || height == 0 || depth == 0 {
            return Err(invalid(call, Problem::EmptyThreadgroup));
        }
        // Any other depth runs as 1, as plain Metal runs it: the threadgroups below are one deep.
        let tile = self.layout.tile_size()?;
        let kernel = state.kernel().clone();
        // A tile function that statically uses a texture slot that holds a memoryless texture
        // fails the command buffer as plain Metal's does, from its pass on, and records nothing
        // (`EncoderCore::fail_memoryless_use`; an M3 Max under macOS 27.0.1).
        if (self.tile).uses_memoryless(kernel.interface(), Stage::Kernel) {
            self.core.fail_memoryless_use();
            return Ok(());
        }
        // A tile function's explicit imageblock aliased onto the attachments is threadgroup
        // memory of its dispatch, staged from and to them, where Metal's is the tile memory the
        // pass's imageblock storage stands in for (`fragment_imageblock.rs`); an explicit one
        // alone is that storage (below), and an implicit one alone the attachments, which the
        // dispatch stages. An aliased layout no larger than the attachments pack to reaches them
        // alone, as Metal's does; a larger one reaches the explicit imageblock after them.
        let colors = imageblock_sample_length(&state.descriptor().color_attachment_formats);
        let aliased_past = kernel
            .aliased_imageblock_bytes()
            .is_some_and(|bytes| colors.is_none_or(|colors| bytes > colors));
        if aliased_past && self.imageblock.is_some() {
            return Err(Error::NotImplemented(
                RuntimeGap::TileAndFragmentImageblocks,
            ));
        }
        // Threads per tile past Metal's limit for the state run nothing, as plain Metal runs them
        // (a state of 256 threads dispatched 32x32 per tile left every pixel
        // and record as the pass had them; Metal's validation stops it), whether or not they are
        // also past the pass's tile: Metal's limit is read first, so a dispatch past both runs
        // nothing rather than being refused below (measured on an Apple M3 Max, macOS 27.0.1,
        // probe `refusals_without_a_named_device` shapes `tilec-32x64` and `tilec-64x64` over a 16
        // by 16 tile: no thread ran at all, with no error). Threads within it but
        // past this device's lower limit (`max_total_threads_per_threadgroup`) are the device's
        // gap, which Metal runs: the binder refuses them typed below (`Bind`, `Unsupported`), and
        // never runs nothing for them.
        let metal = kernel.metal_max_total_threads_per_threadgroup();
        if u64::from(width) * u64::from(height) > u64::from(metal) {
            return Ok(());
        }
        // Threads per tile wider or taller than the pass's tile, which Metal's limit above leaves,
        // are refused: plain Metal takes such a dispatch, but it runs the threads past the tile at
        // the pixels of the tiles that follow, which the runtime does not model
        // (`DrawProblem::ThreadsPerTilePastTile`).
        if width > tile[0] || height > tile[1] {
            return Err(draw_problem(
                call,
                DrawProblem::ThreadsPerTilePastTile {
                    threads: [width, height],
                    tile,
                },
            ));
        }
        // A size other than the state's required one runs as given, as plain Metal runs it
        // (Metal's API validation stops it).
        let area = self.layout.tile_area(tile);
        // The tile function reaches what the encoder made resident in the tile stage, the
        // device's sampler states its set-1 sampler arrays hold, and the live tables its state
        // made (`render/arguments.rs`).
        let used_resources = self.residency.tile_resources();
        let function_tables = kernel.function_tables();
        // A buffer where the tile function reads a table reads as a null table, and a table
        // where it reads a buffer it does not write as an unbound buffer, as a kernel's and a
        // drawing stage's do.
        let views = kernel.at_bind_views();
        let stage_table = (self.tile).stage_table_reading(Stage::Kernel, kernel.interface(), views);
        // Its runtime samplers' address modes as the textures they sample map them.
        let sampler_maps = address_map::sampler_maps(
            kernel.base_layout(),
            &[(Stage::Kernel, &self.tile)],
            Some(&self.residency),
        )
        .map_err(Error::NotImplemented)?;
        // The bytes of the explicit imageblock's cells the pass keeps, where it keeps fewer.
        let kept_imageblock_bytes = self.kept_imageblock_bytes(&state)?;
        let bind_state = |argument_samplers: &[SamplerState]| {
            let mut bind = BindState::default();
            bind.stages.insert(Stage::Kernel, stage_table.clone());
            bind.view_base_alignment = views.alignment();
            bind.imageblock_size = ImageblockSize::new(tile[0], tile[1]);
            bind.render_area = area;
            // The planner refuses a layered pass's dispatch (`NotImplemented(TileShading)`).
            bind.render_target_array_length = self.render_target_array_length;
            bind.used_resources = used_resources.clone();
            bind.used_in_token_order = true;
            bind.argument_samplers = heap_sets::listed(argument_samplers);
            bind.function_tables = function_tables
                .iter()
                .map(VisibleFunctionTable::listed)
                .collect();
            bind.heap_placement = HeapPlacement::ResourceId;
            bind.tile_threadgroup = self
                .tile_memory
                .bind_state(self.layout.tile.threadgroup_memory_length);
            bind.kept_imageblock_bytes = kept_imageblock_bytes;
            sampler_maps.apply(&mut bind);
            bind
        };
        let groups = [area[0].div_ceil(tile[0]), area[1].div_ceil(tile[1])];
        let threads_per_group = [width, height, 1];
        // The tiles that run, and the pixels of the scratch they cover.
        let (dispatch, copied) = match region {
            None => (
                Dispatch::Threadgroups {
                    groups: [groups[0], groups[1], 1],
                    threads_per_group,
                },
                [[0, 0], area],
            ),
            Some(region) => {
                let [(x, w), (y, h)] = region_tiles(&region, tile, groups);
                let rect = |first: u32, count: u32, axis: usize| {
                    let from = first * tile[axis];
                    let to = ((first + count) * tile[axis]).min(area[axis]);
                    (from, to.saturating_sub(from))
                };
                let ((px, pw), (py, ph)) = (rect(x, w, 0), rect(y, h, 1));
                (
                    Dispatch::ThreadgroupRange {
                        groups: [groups[0], groups[1], 1],
                        first: [x, y, 0],
                        count: [w, h, 1],
                        threads_per_group,
                    },
                    [[px, py], [pw, ph]],
                )
            }
        };
        let mut argument_samplers = self.core.device().argument_samplers();
        let mut unlisted = None;
        let (bind, variant, plan) = loop {
            let bind = bind_state(&argument_samplers);
            let (variant, plan) = kernel.settle(&bind.shape(), &dispatch, BufferForms::Bound)?;
            let set = variant.pipeline().layout.sets.get(1);
            if heap_sets::fit_samplers(set, &mut argument_samplers, &mut unlisted) {
                break (bind, variant, plan);
            }
        };
        self.core.device().degrade_argument_pixel_samplers(
            std::slice::from_ref(&variant.pipeline().stage),
            &argument_samplers,
        );
        self.tile
            .check_function_tables(call, Stage::Kernel, &variant.pipeline().interface)?;
        // The function's explicit imageblock standing alone is the pass's imageblock storage,
        // which the pass's draws' fragment imageblocks are in (`fragment_imageblock.rs`): of the
        // storage's cell, and beside no aliased layout.
        let pass_cells = plan
            .descriptors
            .iter()
            .any(|write| matches!(write.source, WriteSource::Imageblock));
        // Its bytes, and the words its cells are staged as.
        let cells = match pass_cells {
            true => {
                let bytes = kernel.tile_imageblock().map(|(data_size, _)| data_size);
                let words = kernel.imageblock_cell_bytes().map(|cell| cell / 4);
                match bytes.zip(words) {
                    Some(cells) => Some(cells),
                    None => return Err(internal("a tile function's cells of no imageblock")),
                }
            }
            false => None,
        };
        if let Some((bytes, words)) = cells {
            if self.tile_imageblock {
                return Err(Error::NotImplemented(
                    RuntimeGap::TileAndFragmentImageblocks,
                ));
            }
            if self
                .imageblock
                .as_ref()
                .is_some_and(|held| !held.takes_tile(bytes, words))
            {
                return Err(Error::NotImplemented(RuntimeGap::ImageblockMasters));
            }
        }
        let mut commands = plan
            .instantiate_dispatch(&bind, &dispatch)
            .map_err(Error::Bind)?;
        variant.keyed_regions(self.core.device(), &mut commands);
        self.reaches_unnamed |= commands.root_table.is_some() || super::reads_handles(&plan);
        if commands.indirect_clamp.is_some()
            || commands
                .regions
                .iter()
                .any(|region| region.indirect.is_some())
        {
            return Err(internal(
                "a tile dispatch's commands read indirect arguments",
            ));
        }
        // Set 1, the heap: its writes at the elements of their resource IDs, and the device's
        // texture table at the texture handle tables.
        let heap = heap_sets::heap_elements(&variant.pipeline().layout, &plan, &bind)?;
        let device = self.core.device().clone();
        let texture_table = heap_sets::texture_table(&device, &variant.pipeline().layout)?;
        let sampler_table = heap_sets::sampler_table(
            &device,
            &variant.pipeline().layout,
            &variant.pipeline().dependencies,
        )?;
        if commands.regions.is_empty() {
            self.core.executed(|| {
                Execution::new(
                    ExecutedPipeline::Compute(Arc::clone(variant.shared_pipeline())),
                    Arc::new(bind.shape()),
                    Vec::new(),
                )
            });
            return Ok(());
        }
        // Each staged attachment's scratch: the pass's, or a new one.
        let mut staged: Vec<(u32, Target)> = Vec::new();
        let mut made = Vec::new();
        for write in &plan.descriptors {
            let WriteSource::TileAttachment {
                color_index,
                format,
                ..
            } = write.source
            else {
                continue;
            };
            if staged.iter().any(|(color, _)| *color == color_index) {
                continue;
            }
            let held = self.scratch.iter().find(|(color, target, view)| {
                *color == color_index && *view == format && target.extent == area
            });
            let target = match held {
                Some((_, target, _)) => target.clone(),
                None => {
                    let own = self
                        .layout
                        .colors
                        .get(color_index as usize)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| internal("a staged attachment the pass does not have"))?;
                    let target = scratch(self.core.command_buffer(), &own.target, format, area)?;
                    made.push((color_index, target.clone(), format));
                    target
                }
            };
            staged.push((color_index, target));
        }
        // The pass's tile threadgroup memory, which the function's threadgroup arguments read
        // (`render/tile_memory.rs`).
        let tile_memory = plan
            .descriptors
            .iter()
            .find_map(|write| match write.source {
                WriteSource::TileMemory { bytes } => Some(bytes),
                _ => None,
            })
            .map(|bytes| self.tile_memory.buffer(&device, bytes))
            .transpose()?;
        // The pass's imageblock storage, made at its first draw or dispatch that reaches it.
        let made_imageblock = match (&self.imageblock, cells) {
            (None, Some((bytes, words))) => Some(PassImageblock::for_tile(
                self.core.command_buffer(),
                bytes,
                words,
                &self.layout,
            )?),
            _ => None,
        };
        let imageblock = match cells {
            Some(_) => made_imageblock.as_ref().or(self.imageblock.as_ref()),
            None => None,
        };
        let pipeline_stage = &variant.pipeline().stage;
        let command = binding::Command {
            descriptor_bytes: &commands.descriptor_bytes,
            descriptor_ranges: &commands.descriptor_ranges,
            root_tables: commands
                .root_table
                .iter()
                .map(|table| (metal2vulkan::vk::ShaderStageFlags::COMPUTE, table))
                .collect(),
            push_holes: &commands.push_holes,
            uses: &commands.uses,
            blobs: pipeline_stage
                .constant_blob
                .iter()
                .map(|blob| (Stage::Kernel, blob))
                .collect(),
            mesh: None,
            tessellation: None,
        };
        let stages = [(Stage::Kernel, &self.tile)];
        let mut tables = binding::Tables::new(&stages)
            .with_pass(&self.layout, ViewSetup::default())
            .with_tile_scratch(&staged)
            .with_residency(&self.residency)
            .with_texture_table(texture_table.as_ref())
            .with_sampler_table(sampler_table.as_ref())
            .with_sampler_maps(sampler_maps);
        // The listed function tables: set 1's elements, or the snapshot a variant without the
        // function-table array reads in set 0, which has no heap of its own.
        tables = tables.with_function_tables(&function_tables);
        if let Some(heap) = &heap {
            tables = tables.with_heap(&argument_samplers, heap);
        }
        if let Some(buffer) = &tile_memory {
            tables = tables.with_tile_memory(buffer);
        }
        if let Some(imageblock) = imageblock {
            tables = tables.with_imageblock(imageblock.view()?);
        }
        let mut uses = binding::command_uses(call, &plan, &command, &tables, None, None)?;
        if let Some(imageblock) = imageblock {
            uses.extend(imageblock.dispatch_uses()?);
        }
        // A dispatch that names a detached buffer records nothing: its submission fails.
        let named = uses
            .iter()
            .map(|used| used.key)
            .filter(|key| !uses.resident().contains(key));
        if self.core.skips(named) {
            self.tile.retain_exclusive(&mut self.core);
            return Ok(());
        }
        // The shadowed textures the dispatch uses, copied around it (the module docs).
        let textures = self.tile.textures.values().chain(self.residency.textures());
        let mut shadows = InstanceShadows::default();
        shadows.add(&shadow::shadow_uses(textures, &uses), &uses);
        let mut writes = Vec::new();
        let bound = {
            let mut state = self.core.command_buffer().state();
            let into = (None, &mut writes);
            binding::bind(call, &device, &mut state, &plan, &command, &tables, into)?
        };
        // The formats its notices name, where the device announces any of its variant's kinds
        // at every occurrence (announced as the dispatch records, below), taken while its
        // tables live.
        let bound_each = device.binds_each(variant.objects().kinds()).then(|| {
            variant
                .objects()
                .tested_formats(&|slot| tables.bound_format(slot))
        });
        device.degrade_bound_formats(
            std::slice::from_ref(&variant.pipeline().stage),
            Some(&|slot| tables.bound_format(slot)),
        );
        // Each specialization the dispatch's regions take, once.
        for (n, region) in commands.regions.iter().enumerate() {
            if commands.regions[..n]
                .iter()
                .all(|earlier| earlier.specialization != region.specialization)
            {
                device.degrade_bound_values(
                    std::slice::from_ref(&variant.pipeline().stage),
                    &region.specialization,
                    &|k| plan.specialized_value(k),
                    Some(&|slot| tables.bound_format(slot)),
                );
            }
        }
        let mut regions = Vec::with_capacity(commands.regions.len());
        for region in &commands.regions {
            let push = binding::push_writes(region.push_constants.clone(), &bound.holes)?;
            regions.push((push, region.group_count));
        }
        // Each region's `VkPipeline`: made, or handed to the device's pool while the dispatch goes
        // on, its pass recording once it is made (`render/pending_tile.rs`). An encoder encoded
        // again (`replay.rs`) waits for no making either, and leaves the same pass to record
        // later: the making is its submission's to wait for, as the first encoding left it, never
        // the rebind's call on the program's thread, as for a compute dispatch.
        let formats: BoundFormats<'_> = &|slot| tables.bound_format(slot);
        let creation = self
            .core
            .records_executions()
            .then(|| Arc::new(OnceLock::new()));
        let pipelines = {
            let call = TileCall {
                state: (self.pipeline.clone())
                    .ok_or_else(|| internal("a tile dispatch without its pipeline state"))?,
                encoder: self.core.command_buffer().state().open_encoder_place(),
                dispatch: self.tile_dispatch_calls,
            };
            let regions = &commands.regions;
            TilePipelines::begin(
                (&kernel, &variant),
                regions,
                formats,
                call,
                creation.clone(),
            )
        };
        let objects = Arc::clone(variant.objects());
        // Set 0 per dispatch; set 1 the device's (written below, once nothing refuses).
        let (heap_writes, mut writes): (Vec<_>, Vec<_>) =
            writes.into_iter().partition(|write| write.set > 0);
        let set = objects.set_zero(&mut self.core.command_buffer().state().descriptors)?;
        let mut offsets = Vec::new();
        binding::dynamic_writes(&mut writes, objects.dynamic_bindings(), &mut offsets);
        let pushed = match objects.pushes_set_zero() {
            true => binding::SetWrite::pushed(&writes, objects.immutable_pushes()),
            false => {
                // SAFETY: the set was just allocated and no command uses it yet; every buffer,
                // view and sampler written is live (the command buffer retains them below, and a
                // refusal leaves the set unused in its pool).
                unsafe { binding::write_sets(&device, set.as_slice(), &writes) }?;
                Vec::new()
            }
        };
        let stages = uses
            .iter()
            .fold(vk::PipelineStageFlags2::empty(), |all, used| {
                all | used.read.stages | used.write.stages
            });
        self.core
            .note_reaching(stages, (&*uses, uses.resident()), None)?;
        self.unordered.add(&uses);
        self.unordered_dispatches.add(&uses);
        let heap_set = match heap {
            Some(_) => Some(heap_sets::bind(
                &device,
                &objects,
                &mut self.core.command_buffer().state().heaps,
                heap_writes,
                &uses,
            )?),
            None => None,
        };
        // Recording starts here: nothing below can be refused.
        self.scratch.extend(made);
        self.tile_imageblock |= aliased_past;
        if let Some(made) = made_imageblock {
            self.imageblock = Some(made);
        }
        let first = staged.first().and_then(|(color, _)| {
            let color = self.layout.colors.get(*color as usize)?.as_ref()?;
            Some(color.stand_in.as_ref().map_or(&color.target, |s| &s.own))
        });
        if let Some(own) = first {
            device.degrade_tile_scratch(own, staged.len(), area);
        }
        // The pipelines made at the call, held until the command buffer completes.
        let made_pipelines = match &pipelines {
            TilePipelines::Made(made) => Some(made.clone()),
            TilePipelines::Making(_) => None,
        };
        self.dispatches.push(TileDispatch {
            after: self.earlier.len() + 1,
            staged,
            copied,
            layout: objects.layout(),
            sets: set.into_iter().collect(),
            offsets,
            pushed,
            heap: heap_set,
            regions,
            pipelines,
            shadows,
        });
        self.dispatched = true;
        // Its writes of an extended-range color are the codes' alone: the shadows end.
        self.end_extended_shadows();
        self.called = true;
        self.drew = true;
        // Its pixels are not bounded by a viewport the host knows of.
        self.drawn.everywhere();
        self.coverage.everywhere();
        let executed = || ExecutedPipeline::Compute(Arc::clone(variant.shared_pipeline()));
        match (&made_pipelines, creation) {
            (Some(made), _) => self.core.executed(|| {
                let times = made.iter().map(|made| made.created_in).collect();
                Execution::new(executed(), Arc::new(bind.shape()), times)
            }),
            // Its times set once its pipelines are made.
            (None, Some(creation)) => self
                .core
                .executed(|| Execution::made_later(executed(), Arc::new(bind.shape()), creation)),
            (None, None) => {}
        }
        // The dispatch binds the variant: the kinds its making announces at every occurrence.
        if let Some(tested) = &bound_each {
            let formats = |slot| tested.iter().find(|(s, _)| *s == slot)?.1;
            let formats: BoundFormats<'_> = &formats;
            objects.announce_bound(
                (
                    std::slice::from_ref(&variant.pipeline().stage),
                    CompiledPipeline::Kernel(variant.pipeline()),
                ),
                commands
                    .regions
                    .iter()
                    .map(|region| (&region.specialization, Some(formats))),
            );
        }
        self.core.retain(kernel);
        self.core.retain(objects);
        if let Some(made) = made_pipelines {
            self.core.retain(made);
        }
        self.tile.retain_exclusive(&mut self.core);
        self.core.retain(bound.samplers);
        // The pass's tile memory, untracked: live until the command buffer completes.
        if let Some(buffer) = tile_memory {
            self.core.retain(buffer);
        }
        // The tables the heap's function-table array may name: their IDs stay theirs until the
        // command buffer completes.
        if heap.is_some() && !function_tables.is_empty() {
            self.core.retain(function_tables);
        }
        if let Some(unlisted) = unlisted {
            heap_sets::degrade_unlisted(&device, unlisted, "a tile pipeline state's dispatches");
            // It runs with the states its sampler arrays hold.
            let mut held = self.core.command_buffer().state();
            held.unlisted_argument_sampler(&device);
        }
        Ok(())
    }
}

impl RenderCommandEncoder {
    /// `tileWidth`: the pass's tile width, or the default for its samples (32; `tile_pass.rs`).
    /// Not measured (`NotImplemented(TileAtSampleCount)`) in a pass of a sample count Apple's
    /// GPUs lack.
    pub fn tile_width(&self) -> Result<u64, Error> {
        self.lowered(RenderLowering::tile_width)
    }

    /// `tileHeight`: the pass's tile height, or the default for its samples (32, and 16 at 4
    /// samples; `tile_pass.rs`). Not measured as [`RenderCommandEncoder::tile_width`] is not.
    pub fn tile_height(&self) -> Result<u64, Error> {
        self.lowered(RenderLowering::tile_height)
    }

    /// `setTileBuffer:offset:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_buffer`] is.
    pub fn set_tile_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_buffer(buffer, offset, index)),
            false => self.defer(|log| {
                let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                RenderCall::Buffer(Table::Tile, buffer, offset, index)
            }),
        }
    }

    /// `setTileBuffers:offsets:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_buffers`] is.
    pub fn set_tile_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_buffers(buffers, offsets, first)),
            false => self.defer(|log| {
                RenderCall::Buffers(
                    Table::Tile,
                    log.buffer_span(buffers),
                    log.word_span(offsets),
                    first,
                )
            }),
        }
    }

    /// `setTileBufferOffset:atIndex:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_buffer_offset`] is.
    pub fn set_tile_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_buffer_offset(offset, index)),
            false => self.defer(|_| RenderCall::BufferOffset(Table::Tile, offset, index)),
        }
    }

    /// `setTileBytes:length:atIndex:`: as
    /// [`RenderCommandEncoder::set_vertex_bytes`], in the tile stage's table, which the pass's
    /// later tile dispatches read. Refused as that is.
    pub fn set_tile_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_bytes(bytes, index)),
            false => self.defer(|log| RenderCall::Bytes(Table::Tile, log.byte_span(bytes), index)),
        }
    }

    /// `setTileTexture:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_texture`] is.
    pub fn set_tile_texture(&mut self, texture: Option<&Texture>, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_texture(texture, index)),
            false => self.defer(|log| {
                let texture = texture.map(|texture| log.hold_texture(texture));
                RenderCall::Texture(Table::Tile, texture, index)
            }),
        }
    }

    /// `setTileTextures:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_textures`] is.
    pub fn set_tile_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_textures(textures, first)),
            false => self
                .defer(|log| RenderCall::Textures(Table::Tile, log.texture_span(textures), first)),
        }
    }

    /// `setTileSamplerState:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_sampler_state`] is.
    pub fn set_tile_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_sampler_state(sampler, index)),
            false => self.defer(|log| {
                let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                RenderCall::Sampler(Table::Tile, sampler, index)
            }),
        }
    }

    /// `setTileSamplerStates:withRange:`, or with `clamps`
    /// `setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_sampler_states`] is.
    pub fn set_tile_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_tile_sampler_states(samplers, clamps, first))
            }
            false => self.defer(|_| {
                RenderCall::Samplers(Table::Tile, Samplers::of(samplers, clamps, first))
            }),
        }
    }

    /// `dispatchThreadsPerTile:`: run the tile pipeline's state set on every tile of the pass
    /// (the module docs). Threads per tile of a depth other than 1 run as 1, and threads per tile
    /// past Metal's limit for the state run nothing, as plain Metal runs them, whether or not
    /// they are also past the pass's tile (Metal's limit is read first). Refused
    /// (`Invalid`), as Metal's validation refuses it: no state set, a drawing state
    /// (`DrawProblem::NotTileState`), threads per tile that are empty or, within Metal's limit,
    /// wider or taller than the pass's tile (`DrawProblem::ThreadsPerTilePastTile`, whose doc
    /// says what plain Metal runs instead). What the binder refuses is
    /// refused (`Bind`, `Unsupported`), threads per tile within Metal's limit but past the
    /// state's `max_total_threads_per_threadgroup` (this device's lower limit) among them. A
    /// refused dispatch records nothing and changes nothing.
    pub fn dispatch_threads_per_tile(&mut self, threads: Size) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.dispatch_threads_per_tile(threads)),
            false => self.defer(|_| RenderCall::ThreadsPerTile(Box::new(threads))),
        }
    }

    /// `dispatchThreadsPerTile:inRegion:`, or with `render_target_array_index`
    /// `dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` (SPI): run the tile
    /// pipeline's state set on the tiles of the pass that `region`'s pixels touch, each as
    /// [`RenderCommandEncoder::dispatch_threads_per_tile`] runs it (the module docs). Refused as
    /// that is; a region that touches no tile runs nothing.
    pub fn dispatch_threads_per_tile_in_region(
        &mut self,
        threads: Size,
        region: Region,
        render_target_array_index: Option<u32>,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.dispatch_threads_per_tile_in_region(
                    threads,
                    region,
                    render_target_array_index,
                )
            }),
            false => self.defer(|_| {
                RenderCall::ThreadsPerTileInRegion(Box::new((
                    threads,
                    region,
                    render_target_array_index,
                )))
            }),
        }
    }
}

/// The tiles `region`'s pixels touch in a pass of `groups` tiles of `tile` pixels, per axis
/// `(first, count)`, as Metal takes the region (the module docs): in 32-bit arithmetic that wraps,
/// the first tile is `x / tile`, the end `(x + w + tile - 1) / tile`, and the count `end - first`,
/// each origin and size cut to its low 32 bits; the tiles from the first, cut to the pass's. The
/// region's `z` and depth are not read.
fn region_tiles(region: &Region, tile: [u32; 2], groups: [u32; 2]) -> [(u32, u32); 2] {
    let axis = |origin: u64, size: u64, i: usize| {
        // Metal's 32-bit origin and size (measured: an origin of 2^32 + 16 is 16).
        let (origin, size) = (origin as u32, size as u32);
        let first = origin / tile[i];
        let end = origin.wrapping_add(size).wrapping_add(tile[i] - 1) / tile[i];
        let count = end.wrapping_sub(first);
        match groups[i].checked_sub(first) {
            Some(left) if left > 0 => (first, count.min(left)),
            _ => (0, 0),
        }
    };
    [
        axis(region.origin.x, region.size.width, 0),
        axis(region.origin.y, region.size.height, 1),
    ]
}
