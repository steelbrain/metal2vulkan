//! A rate-mapped pass's executed commands at a host step, fanned out to the map's regions
//! (`rate_mapped_pass.rs`, **Fan-out**), under both settings of `RateMapRegionPolicy`. Every
//! command but five draws a grid of small triangles, twice, blended at half alpha in its own
//! colour; one between them draws a triangle over the whole target, which touches more regions
//! than the fan-out's copies, another a triangle over the whole target one of whose corners lies
//! behind the eye (`w` of -1), and two the grid with two of its triangles far apart, [`WIDE_AT`]
//! and [`WIDE_FAR`], each over half the target, one with indices and one without, so the probe
//! flags these four; and another a triangle over four tiles a side, which touches more regions
//! than few copies (`bind::fan_out::FEW_COPIES`) and no more than the copies. Under `FanOut` the
//! probe runs once, the grid commands are fanned out, and the four it flagged are drawn in parts
//! (`render/fan_out.rs`, **Parts**), each in its place: each grid with the wide triangles in five,
//! the blocks of `bind::fan_out::BLOCK_TRIANGLES` that hold them region by region, from each wide
//! triangle to its block's end and from its block's start, and the triangles before, between and
//! after them fanned out; and each single triangle region by region, through its region-copy
//! variant (`render/fan_out.rs`, **Region-copy variant**), as the encoder draws every command;
//! under `Exact` no probe runs and every command is drawn region by region, each region through its
//! own viewport. Either way the target's bytes are those of the same commands drawn on the encoder:
//! a command or a triangle lost, drawn twice or out of order, or a copy drawn into another region
//! or left out of its own, changes the pixels. The probe's and the fanned-out draws' pipelines are
//! begun at commit (`RenderExecution::settle`): the host step takes them made or waits for them,
//! and makes no pipeline itself, on a device's first frame too.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_triangle` and
//! `tob_solid` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`, and for
//! the triangle behind the eye `cfb_depth_vertex` of `engine/cfb_sample_depth.ll`, which draws
//! each vertex where its buffer holds it (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::write_routes::GpuWriter;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RasterizationRateMap, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::{
    BlendFactor, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The grid's quads on a side, the commands, the one that draws the large triangle, the medium
/// one, the one across `w = 0`, and the grid with wide triangles without indices and with
/// them, the triangles they replace, in blocks far apart.
const GRID: u32 = 16;
const COMMANDS: u32 = 8;
/// The commands of a frame whose probe is submitted in parts (`render/executions.rs`,
/// `DeferredPass::probe_parts`: at least 32 commands a part): the eight's pattern nine times, in
/// parts of 32, 32 and 8 commands.
const PARTED: u32 = 9 * COMMANDS;
const FLAT: u32 = 0;
const LARGE: u32 = 3;
const MEDIUM: u32 = 5;
const BEHIND: u32 = 6;
const MIXED: u32 = 7;
const WIDE_AT: u32 = 300;
const WIDE_FAR: u32 = 900;
/// A grid command's indices: the grid's triangles, twice.
const INDICES: u32 = 2 * GRID * GRID * 6;
/// The map's screen side, and its quality samples on both axes.
const SCREEN: u32 = 512;
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// The device, its queue, and what the frames draw with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    /// The pipeline of the command across `w = 0`, its positions four floats a vertex.
    behind_pipeline: RenderPipelineState,
    behind: Buffer,
    grid: Buffer,
    grid_indices: Buffer,
    mixed_indices: Buffer,
    /// The vertices of the grid with a wide triangle, each triangle's three in order.
    mixed_vertices: Buffer,
    large: Buffer,
    large_indices: Buffer,
    medium_indices: Buffer,
    colors: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
    /// Its commands, each drawing as its place in the eight's pattern ([`draw_of`]).
    commands: u32,
}

/// What a command draws, by its place in the eight's pattern: its pipeline, its positions, its
/// indices (`None`: none, its vertices in order) and their count.
fn draw_of(
    workload: &Workload,
    command: u32,
) -> (&RenderPipelineState, &Buffer, Option<&Buffer>, u32) {
    let (plain, behind) = (&workload.pipeline, &workload.behind_pipeline);
    match command % COMMANDS {
        LARGE => (plain, &workload.large, Some(&workload.large_indices), 3),
        MEDIUM => (plain, &workload.large, Some(&workload.medium_indices), 3),
        BEHIND => (behind, &workload.behind, Some(&workload.large_indices), 3),
        MIXED => (
            plain,
            &workload.grid,
            Some(&workload.mixed_indices),
            INDICES,
        ),
        FLAT => (plain, &workload.mixed_vertices, None, INDICES),
        _ => (plain, &workload.grid, Some(&workload.grid_indices), INDICES),
    }
}

impl Workload {
    /// The workload of `commands` commands on `device`.
    fn new(device: Device, commands: u32) -> Workload {
        let renders = library(&device, "engine/tob_renders.ll");
        let function = |name| {
            StageSource::Function(
                renders
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_triangle"));
        descriptor.fragment = Some(function("tob_solid"));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::SourceAlpha;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        // The command across `w = 0`: the same colour stage, each vertex where its buffer holds it.
        let depth = library(&device, "engine/cfb_sample_depth.ll");
        let vertex = depth.new_function_with_name("cfb_depth_vertex").unwrap();
        descriptor.vertex = StageSource::Function(vertex.function_descriptor());
        let behind_pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        // The grid's corners over the target, and its triangles twice.
        let at = |i: u32| i as f32 * 2.0 / GRID as f32 - 1.0;
        let grid: Vec<u8> = (0..=GRID)
            .flat_map(|y| (0..=GRID).map(move |x| [at(x), at(y)]))
            .flatten()
            .flat_map(f32::to_le_bytes)
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
        let grid_indices: Vec<u8> = twice.iter().flat_map(|i| i.to_le_bytes()).collect();
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
        let mixed_indices: Vec<u8> = mixed.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mixed_vertices: Vec<u8> = (mixed.iter())
            .flat_map(|&i| [at(i % (GRID + 1)), at(i / (GRID + 1))])
            .flat_map(f32::to_le_bytes)
            .collect();
        // One triangle over the whole target, and one from the centre of a tile of the map
        // (32 screen pixels) to the centre of the fourth after it, on each axis.
        let large: Vec<u8> = [
            [-1.0f32, -1.0],
            [3.0, -1.0],
            [-1.0, 3.0],
            [-0.4375, -0.4375],
            [-0.0625, -0.4375],
            [-0.4375, -0.0625],
        ]
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
        // A triangle whose third corner lies behind the eye, on the plane `w = 1 + y / 2`, whose
        // part in view spans the target (`w` from 2/3 to 2 there, `|x|` at most 2, `y` from -2/3
        // to 2): the device clips it off the target, and its edges lie off the target too, so no
        // pixel centre is near one.
        let behind: Vec<u8> = [
            [-12.0f32, 4.0, 0.25, 3.0],
            [12.0, 4.0, 0.25, 3.0],
            [0.0, -4.0, 0.25, -1.0],
        ]
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
        let indices =
            |at: [u32; 3]| -> Vec<u8> { at.iter().flat_map(|i| i.to_le_bytes()).collect() };
        let (large_indices, medium_indices) = (indices([0, 1, 2]), indices([3, 4, 5]));
        // Each command's colour at half alpha, by its place in the pattern, 256 bytes apart.
        let colors: Vec<u8> = (0..commands)
            .flat_map(|k| {
                let k = (k % COMMANDS) as f32;
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
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
        icb.inherit_buffers = false;
        icb.max_vertex_buffer_bind_count = 1;
        icb.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&icb, u64::from(commands), ResourceOptions::shared())
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
            behind_pipeline,
            behind: buffer(&behind),
            grid: buffer(&grid),
            grid_indices: buffer(&grid_indices),
            mixed_indices: buffer(&mixed_indices),
            mixed_vertices: buffer(&mixed_vertices),
            large: buffer(&large),
            large_indices: buffer(&large_indices),
            medium_indices: buffer(&medium_indices),
            colors: buffer(&colors),
            icb,
            map,
            device,
            commands,
        };
        for k in 0..commands {
            let command = workload
                .icb
                .indirect_render_command_at(u64::from(k))
                .unwrap();
            let (pipeline, positions, indices, count) = draw_of(&workload, k);
            command.set_render_pipeline_state(pipeline).unwrap();
            command.set_vertex_buffer(Some(positions), 0, 0).unwrap();
            command
                .set_fragment_buffer(Some(&workload.colors), u64::from(k) * 256, 0)
                .unwrap();
            match indices {
                Some(indices) => command.draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    u64::from(count),
                    IndexType::UInt32,
                    indices,
                    0,
                    1,
                    0,
                    0,
                ),
                None => command.draw_primitives(PrimitiveType::Triangle, 0, u64::from(count), 1, 0),
            }
            .unwrap();
        }
        // The GPU may write the records: the host step records the pass.
        workload.icb.written_by_gpu(GpuWriter::Reset);
        drop(workload.icb.gpu_write());
        workload
    }

    /// One frame into a target of the map's physical size cleared to opaque black: the commands
    /// executed from the indirect command buffer where `executed`, else drawn on the encoder.
    /// The target's bytes, and the device's counts over the frame.
    fn frame(&self, executed: bool) -> (Vec<u8>, WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&d).unwrap();
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        pass.rasterization_rate_map = Some(self.map.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        match executed {
            true => encoder
                .execute_commands_in_buffer(&self.icb, 0..u64::from(self.commands))
                .unwrap(),
            false => {
                for k in 0..self.commands {
                    let (pipeline, positions, indices, count) = draw_of(self, k);
                    encoder.set_render_pipeline_state(pipeline).unwrap();
                    encoder.set_vertex_buffer(Some(positions), 0, 0).unwrap();
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
                            1,
                            0,
                            0,
                        ),
                        None => encoder.draw_primitives(PrimitiveType::Triangle, 0, count, 1, 0),
                    }
                    .unwrap();
                }
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(command_buffer);
        let after = self.device.work_counts();
        let (width, height) = (size.width, size.height);
        let mut bytes = vec![0u8; (width * height * 4) as usize];
        let region = Region::new_2d(0, 0, width, height);
        target
            .get_bytes(&mut bytes, width * 4, 0, region, 0, 0)
            .unwrap();
        (bytes, before, after)
    }
}

/// The device the cfg selects under `policy`, its shader cache in `cache`.
pub(super) fn device_under(cache: &Path, policy: RateMapRegionPolicy) -> Device {
    let mut options = options(cache);
    options.policy.rate_map_regions = policy;
    Device::system_default_with(&options).unwrap()
}

/// Whether `device` enables the features the fan-out and its probe take.
fn fans_out(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let features = &profile.enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// The probes, flagged draws, fanned-out draws and draws in parts between two counts.
fn fan_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.rate_map_probes - before.rate_map_probes,
        after.rate_map_flagged_draws - before.rate_map_flagged_draws,
        after.rate_map_fanned_draws - before.rate_map_fanned_draws,
        after.rate_map_parted_draws - before.rate_map_parted_draws,
    ]
}

/// The draws through the region-copy variant and their recordings between two counts
/// (`render/fan_out.rs`, **Region-copy variant**).
fn copy_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 2] {
    [
        after.rate_map_region_copy_draws - before.rate_map_region_copy_draws,
        after.rate_map_region_copy_recordings - before.rate_map_region_copy_recordings,
    ]
}

/// The vertex work between two counts (`WorkCounts::rate_map_whole_copy_vertex_regions`): through
/// the region-copy variant into the whole table, of it the encoder's own, in runs; fanned out,
/// and that into the boxes its blocks' bounds reach.
fn vertex_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.rate_map_whole_copy_vertex_regions - before.rate_map_whole_copy_vertex_regions,
        after.rate_map_own_copy_vertex_regions - before.rate_map_own_copy_vertex_regions,
        after.rate_map_run_vertex_regions - before.rate_map_run_vertex_regions,
        after.rate_map_fanned_vertices - before.rate_map_fanned_vertices,
        after.rate_map_fanned_box_vertex_regions - before.rate_map_fanned_box_vertex_regions,
    ]
}

/// The fanned-out vertices whose boxes hold at most 16, 64 and 256 regions between two counts.
fn boxed_counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.rate_map_fanned_vertices_boxed_16 - before.rate_map_fanned_vertices_boxed_16,
        after.rate_map_fanned_vertices_boxed_64 - before.rate_map_fanned_vertices_boxed_64,
        after.rate_map_fanned_vertices_boxed_256 - before.rate_map_fanned_vertices_boxed_256,
    ]
}

/// The texels of `a` and `b` that differ: how many, and the first's index and its bytes in each.
pub(super) fn differing<'a>(
    a: &'a [u8],
    b: &'a [u8],
) -> (usize, Option<(usize, &'a [u8], &'a [u8])>) {
    let mut differ = (a.chunks(4).zip(b.chunks(4)).enumerate()).filter(|(_, (a, b))| a != b);
    let first = differ.next().map(|(at, (a, b))| (at, a, b));
    (first.map_or(0, |_| 1 + differ.count()), first)
}

/// Under `FanOut` the host step probes once, fans the grid commands out and draws the four it
/// flagged in parts between them, each grid's two wide triangles in runs of their own blocks
/// region by region and the triangles between them fanned out, with the pipelines `commit` began:
/// the fan-out's, its runs region by region through the region-copy variant, so the step makes
/// none. The grid commands are drawn through the variant of few copies and the medium one
/// through the most copies'. Its probe and its fanned-out draws each upload the region table
/// once, the commit's settle none. The encoder draws every command through the region-copy
/// variant into the whole table, its vertices times the table's regions counted as its own; the
/// host step none, its runs counted apart, and its fanned-out vertices (the grids', the medium
/// triangle's and the parts' between the wide ones) each into at least one region and at most the
/// table's. The pixels are the encoder's.
#[test]
fn a_host_step_fans_out_the_commands_the_probe_does_not_flag() {
    let scratch = Scratch::new("rate-mapped-fan-out");
    let workload = Workload::new(
        device_under(&scratch.0, RateMapRegionPolicy::FanOut),
        COMMANDS,
    );
    let (drawn, before, after) = workload.frame(false);
    assert_eq!(fan_counts(&before, &after), [0; 4], "the encoder's draws");
    let encoder_copies = copy_counts(&before, &after);
    let encoder_vertices = vertex_counts(&before, &after);
    let (fanned, before, after) = workload.frame(true);
    let vertices = vertex_counts(&before, &after);
    let boxed_by_size = boxed_counts(&before, &after);
    let counts = fan_counts(&before, &after);
    let copies = copy_counts(&before, &after);
    let probe_nanos = after.rate_map_probe_gpu_nanos - before.rate_map_probe_gpu_nanos;
    let made = after.host_step_pipelines_made - before.host_step_pipelines_made;
    let ready = after.rate_map_fan_out_pipelines_ready - before.rate_map_fan_out_pipelines_ready;
    let waited = after.rate_map_fan_out_pipelines_waited - before.rate_map_fan_out_pipelines_waited;
    let tables = after.rate_map_fan_tables_uploaded - before.rate_map_fan_tables_uploaded;
    let few = after.rate_map_fanned_few_draws - before.rate_map_fanned_few_draws;
    let by_regions =
        after.rate_map_parted_region_triangles - before.rate_map_parted_region_triangles;
    let wait = after.rate_map_probe_wait_nanos - before.rate_map_probe_wait_nanos;
    let parts = after.rate_map_probe_parts - before.rate_map_probe_parts;
    let flagged = after.rate_map_flagged_triangles - before.rate_map_flagged_triangles;
    let runs = after.rate_map_region_runs - before.rate_map_region_runs;
    let run_regions = after.rate_map_run_region_draws - before.rate_map_run_region_draws;
    let every_region = after.rate_map_runs_every_region - before.rate_map_runs_every_region;
    let straddling = after.rate_map_straddling_triangles - before.rate_map_straddling_triangles;
    eprintln!(
        "RATE-MAP-FAN-OUT counts {counts:?} probe_gpu_nanos {probe_nanos}, waited {wait} ns in \
         {parts} parts; \
         {flagged} flagged triangles, {straddling} straddling; runs drawn into {run_regions} \
         regions, {every_region} into every one; \
         fan-out pipelines {ready} ready, {waited} waited for; {made} made at the step; {tables} \
         region tables uploaded; {few} drawn of few copies; {by_regions} triangles of draws in \
         parts region by region, in {runs} runs; draws through the region-copy variant and \
         their recordings: the encoder's {encoder_copies:?}, the execution's {copies:?}; vertex \
         work (whole table, own, runs, fanned, fanned boxes): the encoder's \
         {encoder_vertices:?}, the execution's {vertices:?}, fanned of boxes of at most 16, 64 \
         and 256 regions {boxed_by_size:?}"
    );
    assert_eq!(made, 0, "pipelines the host step made");
    match fans_out(&workload.device) {
        true => {
            assert_eq!(
                encoder_copies[0],
                u64::from(COMMANDS),
                "the encoder's draws"
            );
            assert!(
                encoder_copies[1] >= encoder_copies[0],
                "the encoder's recordings"
            );
            assert_eq!(
                copies[0], runs,
                "the runs drawn through the region-copy variant"
            );
            assert_eq!(counts, [1, 4, u64::from(COMMANDS - 4), 4]);
            assert_eq!(
                parts, 1,
                "the probe's parts: its commands are fewer than a part holds"
            );
            assert_eq!(tables, 2, "region tables uploaded");
            assert!(
                ready + waited >= 3,
                "the commit began no probe or draw pipeline"
            );
            assert_eq!(few, u64::from(COMMANDS - 5), "draws of few copies");
            // Each single triangle, and in each grid from each wide triangle to its block's end and
            // from the far one's block's start to it.
            let block = metal2vulkan::bind::fan_out::BLOCK_TRIANGLES;
            let near = WIDE_AT.next_multiple_of(block) - WIDE_AT;
            let far = WIDE_FAR + 1 - WIDE_FAR / block * block;
            let grid_runs = u64::from(near + far);
            assert_eq!(
                by_regions,
                2 + 2 * grid_runs,
                "triangles of draws in parts region by region"
            );
            assert_eq!(runs, 6, "runs region by region");
            assert_eq!(flagged, 6, "the triangles the probe flagged");
            assert!(wait > 0, "the host's wait for the probe");
            // The encoder's draws: five of the grid's indices and three single triangles, each
            // into every region of the table.
            let drawn = u64::from(5 * INDICES + 9);
            let [whole, own, in_runs, fanned_out, boxed] = encoder_vertices;
            let table = whole / drawn;
            assert!(
                table > 1 && whole == drawn * table && own == whole,
                "the encoder's vertex work {encoder_vertices:?}"
            );
            assert_eq!((in_runs, fanned_out, boxed), (0, 0, 0), "the encoder's");
            // The step's: its runs alone through the region-copy variant, the grids and the
            // medium triangle fanned out, and the parts of the wide grids between their runs.
            let [whole, own, in_runs, fanned_out, boxed] = vertices;
            assert_eq!(
                (whole, own),
                (0, 0),
                "the step's draws into the whole table"
            );
            assert!(
                (3 * by_regions..=3 * by_regions * table).contains(&in_runs),
                "the runs' vertex work {in_runs}"
            );
            let between = 2 * (u64::from(INDICES) - 3 * grid_runs);
            assert_eq!(
                fanned_out,
                3 * u64::from(INDICES) + 3 + between,
                "fanned-out vertices"
            );
            assert!(
                (fanned_out..=fanned_out * table).contains(&boxed),
                "the fanned-out vertices into their boxes {boxed}"
            );
            let [at_16, at_64, at_256] = boxed_by_size;
            assert!(
                at_16 <= at_64 && at_64 <= at_256 && at_256 <= fanned_out,
                "fanned-out vertices by their boxes' regions {boxed_by_size:?}"
            );
        }
        false => {
            assert_eq!((counts, parts), ([0; 4], 0));
            assert_eq!(
                [encoder_copies, copies],
                [[0; 2]; 2],
                "draws through the region-copy variant"
            );
            assert_eq!([encoder_vertices, vertices], [[0; 5]; 2], "vertex work");
            assert_eq!(boxed_by_size, [0; 3], "fanned-out vertices by their boxes");
        }
    }
    // Every pixel took the commands' blends: none is the clear's black.
    assert!(
        drawn.chunks(4).all(|texel| texel[..3] != [0, 0, 0]),
        "a pixel the commands did not draw"
    );
    let (differ, first) = differing(&drawn, &fanned);
    assert_eq!(
        differ, 0,
        "texels the fanned-out execution draws otherwise, the first {first:?}"
    );
}

/// A host step of more commands than a part of its probe holds probes them in parts
/// (`render/executions.rs`, `DeferredPass::probe_parts`): [`PARTED`] commands in three parts,
/// each submitted once the host recorded it, its GPU time counted, and one wait. The probe is one,
/// and it flags, fans out and draws in parts the eight's pattern's draws, each time it comes: its
/// words are those one part would write. The pixels are the encoder's.
#[test]
fn a_host_step_probes_many_commands_in_parts() {
    let scratch = Scratch::new("rate-mapped-fan-out-parts");
    let device = device_under(&scratch.0, RateMapRegionPolicy::FanOut);
    let workload = Workload::new(device, PARTED);
    let (drawn, _, _) = workload.frame(false);
    let (fanned, before, after) = workload.frame(true);
    let counts = fan_counts(&before, &after);
    let parts = after.rate_map_probe_parts - before.rate_map_probe_parts;
    let wait = after.rate_map_probe_wait_nanos - before.rate_map_probe_wait_nanos;
    let probe_nanos = after.rate_map_probe_gpu_nanos - before.rate_map_probe_gpu_nanos;
    let submitted = after.queue_submissions - before.queue_submissions;
    eprintln!(
        "RATE-MAP-FAN-OUT-PARTS counts {counts:?} in {parts} parts, probe_gpu_nanos \
         {probe_nanos}, waited {wait} ns; {submitted} queue submissions"
    );
    match fans_out(&workload.device) {
        true => {
            let flagged = u64::from(PARTED / COMMANDS * 4);
            assert_eq!(counts, [1, flagged, u64::from(PARTED) - flagged, flagged]);
            assert_eq!(parts, 3, "the probe's parts of 32, 32 and 8 commands");
            assert!(
                submitted >= parts,
                "{submitted} submissions for {parts} parts"
            );
            assert!(wait > 0, "the host's wait for the probe");
        }
        false => assert_eq!((counts, parts), ([0; 4], 0)),
    }
    let (differ, first) = differing(&drawn, &fanned);
    assert_eq!(
        differ, 0,
        "texels the fanned-out execution draws otherwise, the first {first:?}"
    );
}

/// With the encoder trail on (`encoder_trail.rs`), the host step notes the states its commands
/// name and its probe is a waited submission the trail watches; the execution is fanned out as
/// without, to the same pixels.
#[test]
fn the_encoder_trail_fans_out_the_same_pixels() {
    let scratch = Scratch::new("rate-mapped-fan-out-trail");
    let mut options = options(&scratch.0);
    options.policy.rate_map_regions = RateMapRegionPolicy::FanOut;
    options.encoder_trail = true;
    let workload = Workload::new(Device::system_default_with(&options).unwrap(), COMMANDS);
    assert!(workload.device.shared().encoder_trail);
    let (drawn, _, _) = workload.frame(false);
    let (fanned, before, after) = workload.frame(true);
    let counts = fan_counts(&before, &after);
    eprintln!("RATE-MAP-FAN-OUT-TRAIL counts {counts:?}");
    match fans_out(&workload.device) {
        true => assert_eq!(counts, [1, 4, u64::from(COMMANDS - 4), 4]),
        false => assert_eq!(counts, [0; 4]),
    }
    let (differ, first) = differing(&drawn, &fanned);
    assert_eq!(
        differ, 0,
        "texels the fanned-out execution draws otherwise, the first {first:?}"
    );
}

/// Under `Exact` no probe runs and every command is drawn region by region, each region through
/// its own viewport, none through the region-copy variant, with the regions variant's pipelines
/// `commit` began: the host step makes none. The pixels are the encoder's draws'.
#[test]
fn exact_regions_draw_every_command_region_by_region() {
    let scratch = Scratch::new("rate-mapped-fan-out-exact");
    let workload = Workload::new(
        device_under(&scratch.0, RateMapRegionPolicy::Exact),
        COMMANDS,
    );
    let (drawn, before, after) = workload.frame(false);
    assert_eq!(copy_counts(&before, &after), [0; 2], "the encoder's draws");
    let (exact, before, after) = workload.frame(true);
    assert_eq!(fan_counts(&before, &after), [0; 4]);
    assert_eq!(
        copy_counts(&before, &after),
        [0; 2],
        "the execution's draws"
    );
    let made = after.host_step_pipelines_made - before.host_step_pipelines_made;
    eprintln!("RATE-MAP-EXACT {made} made at the step");
    assert_eq!(made, 0, "pipelines the host step made");
    let (differ, first) = differing(&drawn, &exact);
    assert_eq!(
        differ, 0,
        "texels the execution draws otherwise, the first {first:?}"
    );
}

/// The first frame of a device that drew none of the commands' states before: its host step makes
/// no `VkPipeline` itself (`WorkCounts::host_step_pipelines_made`), under `FanOut` with the
/// eight's pattern and with [`PARTED`] commands probed in parts, and under `Exact`. `commit`'s
/// settle began each one the step's draws take, keyed as the step keys them: the flagged draws'
/// region by region through their region-copy variant, or through as many viewports a command as
/// a multi-draw takes (`render.rs`, `region_batch`), and the fan-out's. No encoder's draw made
/// one before. The pixels are those of the same commands drawn on the encoder after.
#[test]
fn a_host_steps_first_frame_makes_no_pipeline() {
    let shapes = [
        ("fan-out", RateMapRegionPolicy::FanOut, COMMANDS),
        ("fan-out-parted", RateMapRegionPolicy::FanOut, PARTED),
        ("exact", RateMapRegionPolicy::Exact, COMMANDS),
    ];
    let mut made = Vec::new();
    for (name, policy, commands) in shapes {
        let scratch = Scratch::new(&format!("rate-mapped-first-frame-{name}"));
        let workload = Workload::new(device_under(&scratch.0, policy), commands);
        let (executed, before, after) = workload.frame(true);
        let at_step = after.host_step_pipelines_made - before.host_step_pipelines_made;
        let counts = fan_counts(&before, &after);
        let ready =
            after.rate_map_fan_out_pipelines_ready - before.rate_map_fan_out_pipelines_ready;
        let waited =
            after.rate_map_fan_out_pipelines_waited - before.rate_map_fan_out_pipelines_waited;
        let boxed = after.rate_map_boxed_draws - before.rate_map_boxed_draws;
        let copies = copy_counts(&before, &after);
        let ranges = after.rate_map_index_ranges - before.rate_map_index_ranges;
        eprintln!(
            "RATE-MAP-FIRST-FRAME {name}: {at_step} made at the step; counts {counts:?}; \
             fan-out pipelines {ready} ready, {waited} waited for; {boxed} boxed draws; \
             region-copy draws and recordings {copies:?}; {ranges} index ranges"
        );
        let (drawn, _, _) = workload.frame(false);
        let (differ, first) = differing(&drawn, &executed);
        assert_eq!(
            differ, 0,
            "{name}: texels the execution draws otherwise, the first {first:?}"
        );
        made.push((name, at_step));
    }
    assert!(
        made.iter().all(|&(_, at_step)| at_step == 0),
        "pipelines the host steps made {made:?}"
    );
}
