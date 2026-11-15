//! A rate-mapped pass's executed commands at a host step, through the execution's GPU route
//! (`render/executions/gpu_route.rs`): the records the GPU may write, of a buffer that inherits
//! the encoder's pipeline state, are decoded on the GPU, and the host reads none of them
//! (`WorkCounts::indirect_records_read`), only each command's corners, which it probes and routes
//! as it routes the commands it reads (`render/fan_out.rs`). Every command but three draws a grid
//! of small triangles twice, one of them in two instances; one draws a triangle over the whole
//! target, another a triangle over four tiles a side, and two the grid with two of its triangles
//! far apart, one with indices and one without, so the probe flags three. Each command's positions
//! lie nearer than the one's before it, so each passes the `LessEqual` depth test over the earlier
//! ones and blends its colour at half alpha over them. The target's colour and depth bytes are
//! those of the same commands drawn on the encoder in the same pass: a command or a triangle lost,
//! drawn twice or out of order, a copy drawn into another region, or a vertex placed otherwise,
//! changes them.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are `cfb_depth_vertex` of the owned fixture
//! `validation/fixtures/public/engine/cfb_sample_depth.ll`, which draws each vertex where its
//! buffer holds it, and `tob_solid` of `engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use super::rate_mapped_fan_out::{device_under, differing};
use crate::write_routes::GpuWriter;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, DepthStencilState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, Origin,
    PipelineOptions, RasterizationRateMap, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, IndexType, IndirectCommandType, PixelFormat, PrimitiveType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The grid's quads on a side, the commands, the one without indices, the one in two instances,
/// the large and the medium triangles', and the grid with wide triangles', the triangles they
/// replace, in blocks far apart.
const GRID: u32 = 16;
const COMMANDS: u32 = 8;
const FLAT: u32 = 0;
const TWICE: u32 = 2;
const LARGE: u32 = 3;
const MEDIUM: u32 = 5;
const MIXED: u32 = 7;
const WIDE_AT: u32 = 300;
const WIDE_FAR: u32 = 900;
/// A grid command's indices: the grid's triangles, twice.
const INDICES: u32 = 2 * GRID * GRID * 6;
/// The map's screen side, and its quality samples on both axes.
const SCREEN: u32 = 512;
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// The device, its queue, and what the frames draw with: each shape's positions, a copy for each
/// command one after another, each copy nearer than the one before it.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    depth: DepthStencilState,
    grid: Buffer,
    grid_indices: Buffer,
    mixed_indices: Buffer,
    mixed_vertices: Buffer,
    large: Buffer,
    large_indices: Buffer,
    medium_indices: Buffer,
    colors: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
    /// The bytes of one command's copy of the grid's, the wide grid's and the triangles'
    /// positions.
    copies: [u64; 3],
}

/// What a command draws: its positions and the offset of its copy of them, its indices (`None`:
/// none, its vertices in order), their count, and its instances.
type Drawn<'a> = (&'a Buffer, u64, Option<&'a Buffer>, u32, u32);

impl Workload {
    fn new(device: Device) -> Workload {
        let (vertex, solid) = (
            library(&device, "engine/cfb_sample_depth.ll"),
            library(&device, "engine/tob_renders.ll"),
        );
        let function = |library: &Library, name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(&vertex, "cfb_depth_vertex"));
        descriptor.fragment = Some(function(&solid, "tob_solid"));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::SourceAlpha;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut depth = DepthStencilDescriptor::default();
        depth.depth_compare_function = CompareFunction::LessEqual;
        depth.depth_write_enabled = true;
        let depth = device.new_depth_stencil_state(&depth).unwrap();
        // Each of `corners` at the depth of command `k`'s copy: nearer for each later command,
        // and sloped along `x`, so the depth varies over each triangle.
        let copies = |corners: &[[f32; 2]]| -> Vec<u8> {
            (0..COMMANDS)
                .flat_map(|k| {
                    corners
                        .iter()
                        .flat_map(move |&[x, y]| [x, y, 0.5 + 0.1 * x - 0.04 * k as f32, 1.0])
                })
                .flat_map(f32::to_le_bytes)
                .collect()
        };
        // The grid's corners over the target, and its triangles twice.
        let at = |i: u32| i as f32 * 2.0 / GRID as f32 - 1.0;
        let grid: Vec<[f32; 2]> = (0..=GRID)
            .flat_map(|y| (0..=GRID).map(move |x| [at(x), at(y)]))
            .collect();
        let corner = |x: u32, y: u32| y * (GRID + 1) + x;
        let triangles: Vec<u32> = (0..GRID * GRID)
            .flat_map(|quad| {
                let (x, y) = (quad % GRID, quad / GRID);
                let [a, b, c, d] = [
                    corner(x, y),
                    corner(x + 1, y),
                    corner(x, y + 1),
                    corner(x + 1, y + 1),
                ];
                [a, b, c, b, d, c]
            })
            .collect();
        let twice = [&triangles[..], &triangles[..]].concat();
        let index_bytes =
            |indices: &[u32]| -> Vec<u8> { indices.iter().flat_map(|i| i.to_le_bytes()).collect() };
        // The grid with triangle `WIDE_AT` over the target's lower left half, and `WIDE_FAR` over
        // its upper right half.
        let mut mixed = twice.clone();
        let wide = 3 * WIDE_AT as usize;
        mixed[wide..wide + 3].copy_from_slice(&[corner(0, 0), corner(GRID, 0), corner(0, GRID)]);
        let far = 3 * WIDE_FAR as usize;
        mixed[far..far + 3].copy_from_slice(&[
            corner(GRID, GRID),
            corner(0, GRID),
            corner(GRID, 0),
        ]);
        let mixed_vertices: Vec<[f32; 2]> = (mixed.iter())
            .map(|&i| [at(i % (GRID + 1)), at(i / (GRID + 1))])
            .collect();
        // One triangle over the whole target, and one from the centre of a tile of the map
        // (32 screen pixels) to the centre of the fourth after it, on each axis.
        let large = [
            [-1.0f32, -1.0],
            [3.0, -1.0],
            [-1.0, 3.0],
            [-0.4375, -0.4375],
            [-0.0625, -0.4375],
            [-0.4375, -0.0625],
        ];
        // Each command's colour at half alpha, 256 bytes apart.
        let colors: Vec<u8> = (0..COMMANDS)
            .flat_map(|k| {
                let k = k as f32;
                let mut slot = [k / 8.0, 1.0 - k / 8.0, 0.25 + k / 16.0, 0.5]
                    .map(f32::to_le_bytes)
                    .concat();
                slot.resize(256, 0);
                slot
            })
            .collect();
        let buffer = |bytes: &[u8]| {
            (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
        };
        let per_copy = |corners: usize| 16 * corners as u64;
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
        icb.inherit_pipeline_state = true;
        icb.inherit_buffers = false;
        icb.max_vertex_buffer_bind_count = 1;
        icb.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&icb, u64::from(COMMANDS), ResourceOptions::shared())
            .unwrap();
        let mut map = RateMapDescriptor::new([SCREEN, SCREEN]);
        map.layers = vec![RateMapLayerDescriptor::new(
            QUALITY.to_vec(),
            QUALITY.to_vec(),
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        let workload = Workload {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            depth,
            grid: buffer(&copies(&grid)),
            grid_indices: buffer(&index_bytes(&twice)),
            mixed_indices: buffer(&index_bytes(&mixed)),
            mixed_vertices: buffer(&copies(&mixed_vertices)),
            large: buffer(&copies(&large)),
            large_indices: buffer(&index_bytes(&[0, 1, 2])),
            medium_indices: buffer(&index_bytes(&[3, 4, 5])),
            colors: buffer(&colors),
            icb,
            map,
            device,
            copies: [
                per_copy(grid.len()),
                per_copy(mixed_vertices.len()),
                per_copy(large.len()),
            ],
        };
        for k in 0..COMMANDS {
            let command = workload
                .icb
                .indirect_render_command_at(u64::from(k))
                .unwrap();
            let (positions, offset, indices, count, instances) = workload.drawn(k);
            command
                .set_vertex_buffer(Some(positions), offset, 0)
                .unwrap();
            command
                .set_fragment_buffer(Some(&workload.colors), u64::from(k) * 256, 0)
                .unwrap();
            let (count, instances) = (u64::from(count), u64::from(instances));
            match indices {
                Some(indices) => command.draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    count,
                    IndexType::UInt32,
                    indices,
                    0,
                    instances,
                    0,
                    0,
                ),
                None => command.draw_primitives(PrimitiveType::Triangle, 0, count, instances, 0),
            }
            .unwrap();
        }
        // The GPU may write the records: the host step records the pass.
        workload.icb.written_by_gpu(GpuWriter::Reset);
        drop(workload.icb.gpu_write());
        workload
    }

    /// What command `k` draws.
    fn drawn(&self, k: u32) -> Drawn<'_> {
        let copy = |bytes: u64| u64::from(k) * bytes;
        let [grid, mixed, large] = self.copies;
        match k {
            FLAT => (&self.mixed_vertices, copy(mixed), None, INDICES, 1),
            LARGE => (&self.large, copy(large), Some(&self.large_indices), 3, 1),
            MEDIUM => (&self.large, copy(large), Some(&self.medium_indices), 3, 1),
            MIXED => (
                &self.grid,
                copy(grid),
                Some(&self.mixed_indices),
                INDICES,
                1,
            ),
            TWICE => (&self.grid, copy(grid), Some(&self.grid_indices), INDICES, 2),
            _ => (&self.grid, copy(grid), Some(&self.grid_indices), INDICES, 1),
        }
    }

    /// One frame into a colour target of the map's physical size cleared to opaque black and a
    /// depth target cleared to 1: the commands executed from the indirect command buffer where
    /// `executed`, else drawn on the encoder. The colour's bytes, the depth's, and the device's
    /// counts over the frame.
    fn frame(&self, executed: bool) -> (Vec<u8>, Vec<u8>, WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let (width, height) = (size.width, size.height);
        let texture = |format| {
            let mut d = TextureDescriptor::texture_2d(format, width, height, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            self.device.new_texture(&d).unwrap()
        };
        let (target, depth) = (
            texture(PixelFormat::RGBA8Unorm),
            texture(PixelFormat::Depth32Float),
        );
        let bytes = width * height * 4;
        let back = || (self.device.new_buffer(bytes, ResourceOptions::shared())).unwrap();
        let (color_back, depth_back) = (back(), back());
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.rasterization_rate_map = Some(self.map.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder.set_depth_stencil_state(&self.depth).unwrap();
        match executed {
            true => encoder
                .execute_commands_in_buffer(&self.icb, 0..u64::from(COMMANDS))
                .unwrap(),
            false => {
                for k in 0..COMMANDS {
                    let (positions, offset, indices, count, instances) = self.drawn(k);
                    encoder
                        .set_vertex_buffer(Some(positions), offset, 0)
                        .unwrap();
                    encoder
                        .set_fragment_buffer(Some(&self.colors), u64::from(k) * 256, 0)
                        .unwrap();
                    match indices {
                        Some(indices) => encoder.draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            count,
                            IndexType::UInt32,
                            indices,
                            0,
                            instances,
                            0,
                            0,
                        ),
                        None => {
                            encoder.draw_primitives(PrimitiveType::Triangle, 0, count, instances, 0)
                        }
                    }
                    .unwrap();
                }
            }
        }
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (texture, back) in [(&target, &color_back), (&depth, &depth_back)] {
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(width, height, 1),
                back,
                0,
                width * 4,
                bytes,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(command_buffer);
        let after = self.device.work_counts();
        let read = |back: &Buffer| {
            let at = back.contents().unwrap().as_ptr();
            // SAFETY: the read-back buffer's bytes, which no command writes now.
            unsafe { std::slice::from_raw_parts(at, bytes as usize) }.to_vec()
        };
        (read(&color_back), read(&depth_back), before, after)
    }
}

/// Whether `device` enables the features the fan-out and its probe take.
fn fans_out(device: &Device) -> bool {
    let features = &device.compile_device().profile().enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// The records read, the GPU routes, the probes and the draws they flagged between two counts.
fn route_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.indirect_records_read - before.indirect_records_read,
        after.rate_map_gpu_routes - before.rate_map_gpu_routes,
        after.rate_map_probes - before.rate_map_probes,
        after.rate_map_flagged_draws - before.rate_map_flagged_draws,
    ]
}

/// On a device that fans out, the host step records the execution through its GPU route: it
/// reads no record, probes the commands' corners once and flags the three draws that touch more
/// regions than the fan-out's copies. Elsewhere it reads the records. Either way the colour and the
/// depth are the encoder's draws', bit for bit.
#[test]
fn a_gpu_route_draws_the_encoder_s_pixels_and_depth_reading_no_record() {
    let scratch = Scratch::new("rate-mapped-gpu-route");
    let workload = Workload::new(device_under(&scratch.0, RateMapRegionPolicy::FanOut));
    let (drawn, drawn_depth, before, after) = workload.frame(false);
    assert_eq!(
        route_counts(&before, &after)[1..],
        [0; 3],
        "the encoder's draws"
    );
    let (executed, executed_depth, before, after) = workload.frame(true);
    let counts = route_counts(&before, &after);
    let wait = after.rate_map_gpu_route_wait_nanos - before.rate_map_gpu_route_wait_nanos;
    let fanned = after.rate_map_fanned_draws - before.rate_map_fanned_draws;
    let parted = after.rate_map_parted_draws - before.rate_map_parted_draws;
    let copies = after.rate_map_region_copy_draws - before.rate_map_region_copy_draws;
    let boxed = after.rate_map_boxed_draws - before.rate_map_boxed_draws;
    eprintln!(
        "RATE-MAP-GPU-ROUTE records read, routes, probes, flagged {counts:?}; waited {wait} ns; \
         {fanned} fanned, {parted} in parts, {copies} through the region-copy variant, {boxed} \
         boxed"
    );
    match fans_out(&workload.device) {
        true => {
            assert_eq!(
                counts,
                [0, 1, 1, 3],
                "records read, routes, probes, flagged"
            );
            assert!(wait > 0, "the host's wait for the route's prework");
        }
        false => assert_eq!(counts[1..], [0; 3], "routes, probes, flagged"),
    }
    // Every pixel took the commands' blends: none is the clear's black.
    assert!(
        drawn.chunks(4).all(|texel| texel[..3] != [0, 0, 0]),
        "a pixel the commands did not draw"
    );
    let (differ, first) = differing(&drawn, &executed);
    assert_eq!(
        differ, 0,
        "colour texels the execution draws otherwise, the first {first:?}"
    );
    let (differ, first) = differing(&drawn_depth, &executed_depth);
    assert_eq!(
        differ, 0,
        "depth texels the execution draws otherwise, the first {first:?}"
    );
}

/// The host step of an execution its GPU route takes makes no `VkPipeline` itself
/// (`WorkCounts::host_step_pipelines_made`), at the execution's first frame or after: `commit`'s
/// settle began the merged variants' of the step's draws, the probe's and the fan-out's
/// (`DeferredPass::settle`). On a device that does not fan out, the step reads the records, and
/// the settle's large-buffer form is what it draws in.
#[test]
fn a_gpu_route_s_host_step_makes_no_pipeline() {
    let scratch = Scratch::new("rate-mapped-gpu-route-made");
    let workload = Workload::new(device_under(&scratch.0, RateMapRegionPolicy::FanOut));
    let fans = fans_out(&workload.device);
    for frame in ["first", "second"] {
        let (_, _, before, after) = workload.frame(true);
        let routes = route_counts(&before, &after)[1];
        let made = after.host_step_pipelines_made - before.host_step_pipelines_made;
        let ready =
            after.rate_map_fan_out_pipelines_ready - before.rate_map_fan_out_pipelines_ready;
        let waited =
            after.rate_map_fan_out_pipelines_waited - before.rate_map_fan_out_pipelines_waited;
        eprintln!(
            "RATE-MAP-GPU-ROUTE-MADE {frame} frame: {routes} routes; the step made {made} \
             pipelines, found {ready} of the fan-out's ready and waited for {waited}"
        );
        assert_eq!(routes, u64::from(fans), "{frame} frame: executions routed");
        assert_eq!(made, 0, "{frame} frame: pipelines the host step made");
    }
}
