//! The pieces a rate-mapped pass's executed commands are recorded in on the device the cfg
//! selects (`rate_mapped_pass.rs`, **Pieces**), at the density of a guest's rate-mapped frame:
//! report and bound, in release builds.
//!
//! The workload is one execution of the shape of the consumer's (macOS guests over the runtime):
//! a map of one layer over a [`SCREEN`] screen whose quality falls from 1 at the centre to 0.3 at
//! the edges on both axes ([`QUALITY`]); a pass into a target of the map's physical size; and an
//! indirect command buffer's [`COUNTS`] commands executed, each indexed triangles of one of the
//! index counts a guest's executed commands drew (their 10th, 50th, 90th and 99th percentiles,
//! their mean and their most), drawn from a grid of small triangles over the screen
//! (`tob_triangle`, `tob_solid`). Drawn region by region, its command buffer runs well under a
//! second, so a device that cut nothing would still not reach amdgpu's 2 s job timeout: at the
//! host step, cold, 514 ms of an Iris Xe's GPU, in 20 pieces of at most 32 ms, and 299 ms of a
//! Radeon 780M's, in 20 pieces of at most 36 ms (Mesa 26.2.6). Fanned out it is 47 ms and 42 ms
//! cold, 41 ms and 29 ms warm, after probes of 1.3 ms and 0.4 ms. On an RTX 3090 (NVIDIA
//! 620.70.06) it is 71 ms, in 20 pieces of at most 2.7 ms, and fanned out 44 ms cold, the step
//! waiting for the fan-out's pipelines `commit` began, and 7.5 ms warm, after a probe of 1.5 ms.
//! The frame runs twice, cold, then warm, the pipelines it draws with made, each printed as a
//! `RATE-MAP-PIECES` line: the map's regions, the command buffer's GPU time, the pieces and
//! ranges it was cut into, and its pieces' GPU times where the device times them, and the host
//! step's time and the pipelines it made
//! (`WorkCounts::host_step_nanos`, `host_step_pipelines_made`, `render_pipeline_create_nanos`),
//! and the fan-out's pipelines `commit` began that it found made or waited for
//! (`rate_map_fan_out_pipelines_ready`, `rate_map_fan_out_pipelines_waited`), and the program's
//! time in `commit`, on the host's clock.
//! Checked, each time: the command buffer was cut into pieces and its largest command into
//! ranges, and no piece ran past [`PIECE_LIMIT`]. Twice: over a range the call
//! names, from the CPU's records, which `commit` records (`rate_mapped_pieces`), and over a range
//! read from a buffer, from records a blit encoder of the command buffer may write before the
//! pass (a reset of a command past the range), which its host step records
//! (`rate_mapped_pieces_host_step`). At the host step the default policy fans the commands'
//! triangles out to the regions (`RateMapRegionPolicy::FanOut`, `rate_mapped_pass.rs`,
//! **Fan-out**), after a probe; under `Exact` (`rate_mapped_pieces_host_step_exact`) every draw
//! is drawn region by region, as at `commit`. The line then states the probes, the draws they
//! flagged, the draws fanned out, the flagged ones drawn in parts, their triangles drawn region
//! by region and the runs of blocks those were drawn in (`rate_map_parted_draws`,
//! `rate_map_parted_region_triangles`, `rate_map_region_runs`, at least one a draw) and how many
//! of those the probe flagged (`rate_map_flagged_triangles`), and the
//! probes' GPU time and the host's wait for them (`rate_map_probe_wait_nanos`); fanned out, a
//! draw is never cut into ranges, so only the region-by-region frames are checked for ranges and
//! pieces, and a fanned-out frame for its probe, the host's wait for it and its fanned-out draws.
//!
//! With wide triangles among the grid's (`rate_mapped_pieces_host_step_wide` and its `_exact`
//! twin), one triangle of every [`WIDE_EVERY`] is replaced by one [`WIDE`] quads a side, which
//! touches more regions than the fan-out's copies: the probe flags each command whose indices
//! reach one, as a guest's frame had about a quarter of its probed draws flagged
//! (`rate_map_flagged_draws`).
//!
//! With patches (`rate_mapped_pieces_host_step_patches` and its `_exact` twin), each command draws
//! a patch of small triangles of its own, as many quads of [`PATCH_QUAD`] screen pixels a side as
//! its indices fill, at a place of its own, where the grid's commands each draw a band across the
//! screen: a patch's triangles lie in a few regions, as an object's of a guest's frame do, and the
//! host step boxes the commands whose region draws into them hold no more work than their
//! geometry stage (`render/fan_out.rs`, **Boxed draws**; `rate_map_boxed_draws`).
//!
//! Without a map (`rate_mapped_pieces_host_step_wide_unmapped`), the wide frame's pass renders
//! into a target of the screen's size: no probe, no fan-out and no piece, what the same commands
//! cost where nothing is drawn by regions.
//!
//! At a guest frame's scale ([`GUEST_SCALE`] times the commands, about the 126 pieces a frame of
//! the consumer's was recorded in on an RTX 3090), the host step's frame is
//! `rate_mapped_pieces_guest_scale` (fanned out) and `rate_mapped_pieces_guest_scale_exact`: a GPU
//! hazard, whose command buffer may run for a second or more, compiled only under
//! `m2v_gpu_hazards`. They print the same line and check the same bounds. On the RTX 3090 the
//! region-by-region frame is 332 ms, in 117 pieces of at most 2.7 ms.
//!
//! Ignored by default: a measurement on a GPU, far too slow on a CPU device. Run it by name with
//! `--run-ignored all`. The fixture's text is compiled in, so the binary also runs on a machine
//! other than the one that built it.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{
    RasterizationRateMap as CompiledMap, RateMapDescriptor, RateMapLayerDescriptor, RegionLimits,
};
use metal2vulkan::device::{PipelineWaitPolicy, RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// The map's screen, and its quality samples on both axes.
const SCREEN: [u32; 2] = [3648, 1734];
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// The commands' index counts, as many commands of each.
const COUNTS: [(u32, u32); 6] = [
    (768, 6),
    (2_814, 26),
    (8_436, 19),
    (20_187, 8),
    (89_184, 4),
    (161_820, 1),
];

/// The commands of a guest frame's scale, as many times [`COUNTS`]': about the 126 pieces a frame
/// of the consumer's was recorded in (`rate_map_pieces`) on an RTX 3090.
#[cfg(m2v_gpu_hazards)]
const GUEST_SCALE: u32 = 6;

/// The grid's quads on a side: its triangles' indices hold the most indices a command draws.
const GRID: u32 = 165;

/// With wide triangles, the grid's triangles each wide one replaces one of, and its quads a side.
const WIDE_EVERY: u32 = 8192;
const WIDE: u32 = 48;

/// With patches, the screen pixels a patch's quad spans on a side.
const PATCH_QUAD: f32 = 2.0;

/// The grid's triangles: all small, or one of every [`WIDE_EVERY`] wide; or each command's own
/// patch of small triangles, as many quads a side as its indices fill, [`PATCH_QUAD`] screen
/// pixels each, at a place of its own.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Triangles {
    Small,
    Wide,
    Patches,
}

/// The longest a piece may run: half amdgpu's job timeout.
const PIECE_LIMIT: Duration = Duration::from_secs(1);

/// The counts of `work`: the pieces past the first, the index ranges past the first, the pieces
/// timed, their GPU time and the longest's; the probes, the draws they flagged, the draws fanned
/// out and the probes' GPU time; the host step's time and making; the fan-out's pipelines; the
/// draws of few copies; the draws in parts and their triangles drawn region by region; the
/// host's wait for the probes; the triangles the probes flagged; their runs drawn region by region;
/// the regions the runs were drawn into and the runs that reached every region; the flagged
/// triangles that straddle `w = 0`; the recordings through the region-copy variant; the host
/// steps recorded; the draws boxed and their vertex work.
fn counts(work: &WorkCounts) -> [u64; 27] {
    [
        work.rate_map_pieces,
        work.rate_map_index_ranges,
        work.rate_map_pieces_timed,
        work.rate_map_piece_gpu_nanos,
        work.rate_map_piece_gpu_nanos_most,
        work.rate_map_probes,
        work.rate_map_flagged_draws,
        work.rate_map_fanned_draws,
        work.rate_map_probe_gpu_nanos,
        work.host_step_nanos,
        work.host_step_pipelines_made,
        work.render_pipeline_create_nanos,
        work.rate_map_fan_out_pipelines_ready,
        work.rate_map_fan_out_pipelines_waited,
        work.rate_map_fanned_few_draws,
        work.rate_map_parted_draws,
        work.rate_map_parted_region_triangles,
        work.rate_map_probe_wait_nanos,
        work.rate_map_flagged_triangles,
        work.rate_map_region_runs,
        work.rate_map_run_region_draws,
        work.rate_map_runs_every_region,
        work.rate_map_straddling_triangles,
        work.rate_map_region_copy_recordings,
        work.host_steps_recorded,
        work.rate_map_boxed_draws,
        work.rate_map_boxed_vertex_regions,
    ]
}

/// One frame of the commands, cut into pieces at `commit`: printed and bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces() {
    pieces_frame(false, RateMapRegionPolicy::FanOut, 1, Triangles::Small);
}

/// One frame of the commands, their records ones the GPU may write, cut into pieces at the host
/// step: printed and bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step() {
    pieces_frame(true, RateMapRegionPolicy::FanOut, 1, Triangles::Small);
}

/// One frame of the commands at the host step, every draw drawn region by region and cut into
/// pieces: printed and bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_exact() {
    pieces_frame(true, RateMapRegionPolicy::Exact, 1, Triangles::Small);
}

/// The host step's frame with wide triangles among the grid's, fanned out: printed and bounded
/// (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_wide() {
    pieces_frame(true, RateMapRegionPolicy::FanOut, 1, Triangles::Wide);
}

/// The host step's frame with wide triangles among the grid's, every draw region by region:
/// printed and bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_wide_exact() {
    pieces_frame(true, RateMapRegionPolicy::Exact, 1, Triangles::Wide);
}

/// The host step's frame of each command's own patch, fanned out where its patch reaches many
/// regions and boxed where few (`render/fan_out.rs`, **Boxed draws**): printed and bounded (the
/// module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_patches() {
    pieces_frame(true, RateMapRegionPolicy::FanOut, 1, Triangles::Patches);
}

/// The host step's frame of each command's own patch, every draw region by region: printed and
/// bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_patches_exact() {
    pieces_frame(true, RateMapRegionPolicy::Exact, 1, Triangles::Patches);
}

/// The host step's frame with wide triangles among the grid's, without a map, into a target of
/// the screen's size: what the pass costs where nothing is drawn by regions (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_host_step_wide_unmapped() {
    pieces_frame_of(true, None, 1, Triangles::Wide);
}

/// The host step's frame at a guest frame's scale, fanned out: printed and bounded (the module
/// docs). A GPU hazard.
#[cfg(m2v_gpu_hazards)]
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_guest_scale() {
    pieces_frame(
        true,
        RateMapRegionPolicy::FanOut,
        GUEST_SCALE,
        Triangles::Small,
    );
}

/// The host step's frame at a guest frame's scale, every draw region by region: printed and
/// bounded (the module docs). A GPU hazard.
#[cfg(m2v_gpu_hazards)]
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces_guest_scale_exact() {
    pieces_frame(
        true,
        RateMapRegionPolicy::Exact,
        GUEST_SCALE,
        Triangles::Small,
    );
}

/// One frame of the commands, `scale` times [`COUNTS`]', recorded at the host step where a blit
/// encoder may write their records before the pass (`step`), else at `commit`, the device's draws
/// reaching the regions as `policy` says, the grid's triangles `triangles`.
fn pieces_frame(step: bool, policy: RateMapRegionPolicy, scale: u32, triangles: Triangles) {
    pieces_frame_of(step, Some(policy), scale, triangles);
}

/// [`pieces_frame`]'s frame, its pass over the map where `policy` is some, else over none.
fn pieces_frame_of(
    step: bool,
    policy: Option<RateMapRegionPolicy>,
    scale: u32,
    triangles: Triangles,
) {
    let mapped = policy.is_some();
    let policy = policy.unwrap_or_default();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record, and makes the pipelines a command
        // buffer waits for on the device's pool, the bound its default.
        options.record_executions = false;
        options.policy.pipeline_waits = PipelineWaitPolicy::default();
        options.policy.rate_map_regions = policy;
    })
    .unwrap();
    let features = &device.compile_device().profile().enabled().features;
    let fans = step
        && policy == RateMapRegionPolicy::FanOut
        && [
            VkFeature::GeometryShader,
            VkFeature::ShaderClipDistance,
            VkFeature::VertexPipelineStoresAndAtomics,
        ]
        .iter()
        .all(|feature| features.contains(feature));
    let name = device.name().to_owned();
    let ll = include_str!("../../fixtures/public/engine/tob_renders.ll");
    let library = device
        .new_library_with_air(&llvm_assemble_text(ll).unwrap())
        .unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_triangle"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    let pipeline = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    // The grid's corners over the screen, and its triangles' indices.
    let at = |i: u32| i as f32 * 2.0 / GRID as f32 - 1.0;
    let positions: Vec<u8> = (0..=GRID)
        .flat_map(|y| (0..=GRID).map(move |x| [at(x), at(y)]))
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
    let corner = |x: u32, y: u32| y * (GRID + 1) + x;
    let indices: Vec<u8> = (0..GRID * GRID)
        .flat_map(|quad| {
            let (x, y) = (quad % GRID, quad / GRID);
            let [a, b, c, d] = [
                corner(x, y),
                corner(x + 1, y),
                corner(x, y + 1),
                corner(x + 1, y + 1),
            ];
            // A wide triangle in place of the quad's first, every `WIDE_EVERY` triangles.
            let (b0, c0) = match triangles == Triangles::Wide && (2 * quad) % WIDE_EVERY == 0 {
                true => (
                    corner((x + WIDE).min(GRID), y),
                    corner(x, (y + WIDE).min(GRID)),
                ),
                false => (b, c),
            };
            [a, b0, c0, b, d, c]
        })
        .flat_map(u32::to_le_bytes)
        .collect();
    let grid_indices = GRID * GRID * 6;
    // The commands in an order that spreads the large ones, each its counts' indices from a
    // place of the grid, a whole quad's, and its own colour; or each its own patch's indices.
    let mut draws: Vec<u32> = (COUNTS.iter())
        .flat_map(|&(count, commands)| std::iter::repeat_n(count, (commands * scale) as usize))
        .collect();
    let spread = draws.len();
    draws = (0..spread).map(|k| draws[k * 37 % spread]).collect();
    let (positions, indices, starts): (Vec<u8>, Vec<u8>, Vec<u32>) = match triangles {
        Triangles::Patches => {
            let (positions, indices, ranges) = patches(&draws);
            draws = ranges.iter().map(|&(_, count)| count).collect();
            let starts = ranges.iter().map(|&(start, _)| start).collect();
            (positions, indices, starts)
        }
        Triangles::Small | Triangles::Wide => {
            let starts = (draws.iter().enumerate())
                .map(|(k, &count)| {
                    let room = (grid_indices - count) / 6;
                    (k as u32 * 7_919 % (room + 1)) * 6
                })
                .collect();
            (positions, indices, starts)
        }
    };
    let positions = buffer_with(&device, &positions).unwrap();
    let index_buffer = buffer_with(&device, &indices).unwrap();
    let colors: Vec<u8> = (0..spread)
        .flat_map(|k| {
            let k = k as f32 / spread as f32;
            let mut slot = [k, 1.0 - k, 0.5, 1.0].map(f32::to_le_bytes).concat();
            slot.resize(256, 0);
            slot
        })
        .collect();
    let colors = buffer_with(&device, &colors).unwrap();
    let mut icb = IndirectCommandBufferDescriptor::new();
    icb.command_types = IndirectCommandType::DRAW_INDEXED;
    icb.inherit_buffers = false;
    icb.max_vertex_buffer_bind_count = 1;
    icb.max_fragment_buffer_bind_count = 1;
    // One command past the range, which a blit encoder resets where the host step records.
    let icb = device
        .new_indirect_command_buffer(&icb, spread as u64 + 1, ResourceOptions::shared())
        .unwrap();
    for (k, &count) in draws.iter().enumerate() {
        let command = icb.indirect_render_command_at(k as u64).unwrap();
        command.set_render_pipeline_state(&pipeline).unwrap();
        command.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
        command
            .set_fragment_buffer(Some(&colors), k as u64 * 256, 0)
            .unwrap();
        let start: u32 = starts[k];
        command
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                u64::from(count),
                IndexType::UInt32,
                &index_buffer,
                u64::from(start) * 4,
                1,
                0,
                0,
            )
            .unwrap();
    }
    // At the host step the range is read from a buffer, as the guest's are; `commit` records
    // only a range the call names from the CPU's records.
    let range = [0u32, spread as u32].map(u32::to_le_bytes).concat();
    let range = buffer_with(&device, &range).unwrap();
    let mut map = RateMapDescriptor::new(SCREEN);
    map.layers = vec![RateMapLayerDescriptor::new(
        QUALITY.to_vec(),
        QUALITY.to_vec(),
    )];
    // The regions a draw over the whole screen records, by the compile layer's plan.
    let limits = RegionLimits::new([16384, 16384], [-32768.0, 32767.0]);
    let whole = [0.0, 0.0, f64::from(SCREEN[0]), f64::from(SCREEN[1])];
    let regions = match mapped {
        true => (CompiledMap::new(&map).unwrap())
            .regions(0, whole, None, &limits)
            .unwrap()
            .regions()
            .len(),
        false => 1,
    };
    let map = device.new_rasterization_rate_map(&map).unwrap();
    let size = match mapped {
        true => map.physical_size_for_layer(0),
        false => Size::new(u64::from(SCREEN[0]), u64::from(SCREEN[1]), 1),
    };
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let queue = device.new_command_queue().unwrap();
    for frame in ["cold", "warm"] {
        let before = counts(&device.work_counts());
        let command_buffer = queue.command_buffer().unwrap();
        if step {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let past = spread as u64;
            blit.reset_commands_in_buffer(&icb, past..past + 1).unwrap();
            blit.end_encoding().unwrap();
        }
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        pass.rasterization_rate_map = mapped.then(|| map.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        match step {
            true => encoder.execute_commands_in_buffer_indirect(&icb, &range, 0),
            false => encoder.execute_commands_in_buffer(&icb, 0..spread as u64),
        }
        .unwrap();
        encoder.end_encoding().unwrap();
        let committing = Instant::now();
        command_buffer.commit().unwrap();
        let commit = committing.elapsed();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        drop(command_buffer);
        let after = counts(&device.work_counts());
        let delta = |i: usize| after[i] - before[i];
        let [pieces, ranges, timed, nanos] = [0, 1, 2, 3].map(delta);
        let [probes, flagged, fanned, probe_nanos] = [5, 6, 7, 8].map(delta);
        let [stepped, made, made_nanos, ready, waited] = [9, 10, 11, 12, 13].map(delta);
        let few = delta(14);
        let [parted, by_regions, probe_wait, flagged_triangles] = [15, 16, 17, 18].map(delta);
        let runs = delta(19);
        let [run_regions, every_region, straddling, copy_recordings] = [20, 21, 22, 23].map(delta);
        let steps = delta(24);
        let [boxed, boxed_work] = [25, 26].map(delta);
        let most = Duration::from_nanos(after[4]);
        let indices: u64 = draws.iter().map(|&count| u64::from(count)).sum();
        let mean = Duration::from_nanos(nanos / timed.max(1));
        let at = match (step, mapped.then_some(policy), triangles) {
            (true, None, Triangles::Small) => "host step, no map",
            (true, None, Triangles::Wide) => "host step, no map, wide",
            (true, Some(RateMapRegionPolicy::FanOut), Triangles::Small) => "host step, fan-out",
            (true, Some(RateMapRegionPolicy::FanOut), Triangles::Wide) => {
                "host step, fan-out, wide"
            }
            (true, Some(_), Triangles::Small) => "host step, exact",
            (true, Some(_), Triangles::Wide) => "host step, exact, wide",
            (true, None, Triangles::Patches) => "host step, no map, patches",
            (true, Some(RateMapRegionPolicy::FanOut), Triangles::Patches) => {
                "host step, fan-out, patches"
            }
            (true, Some(_), Triangles::Patches) => "host step, exact, patches",
            (false, _, _) => "commit",
        };
        eprintln!(
            "RATE-MAP-PIECES {name} ({at}, {frame}): screen {}x{}, physical {}x{}, {regions} \
             regions, {spread} commands of {indices} indices: the command buffer {:.1} ms of GPU \
             time, {} pieces ({ranges} index ranges past the first); {timed} pieces timed, the \
             longest {:.2} ms, the mean {:.2} ms; {probes} probes of {:.2} ms, waited for {:.2} ms, \
             {flagged} draws flagged, {fanned} fanned out ({few} of few copies), {parted} drawn in \
             parts ({by_regions} triangles region by region in {runs} runs, {flagged_triangles} of \
             them flagged, {straddling} straddling w = 0; the runs drawn into {run_regions} \
             regions, {every_region} into every region; {copy_recordings} region-copy \
             recordings); {boxed} draws boxed, of {boxed_work} vertices times regions; \
             {steps} host steps of {:.1} ms, \
             {made} pipelines made there, \
             {:.1} ms creating render pipelines; the fan-out's pipelines {ready} ready at the \
             step, {waited} waited for; the commit {:.1} ms",
            SCREEN[0],
            SCREEN[1],
            size.width,
            size.height,
            gpu * 1e3,
            pieces + 1,
            most.as_secs_f64() * 1e3,
            mean.as_secs_f64() * 1e3,
            probe_nanos as f64 * 1e-6,
            probe_wait as f64 * 1e-6,
            stepped as f64 * 1e-6,
            made_nanos as f64 * 1e-6,
            commit.as_secs_f64() * 1e3,
        );
        match (mapped, fans) {
            (false, _) => {
                assert_eq!((probes, fanned, pieces), (0, 0, 0), "a pass without a map");
            }
            (true, true) => {
                assert_eq!(probes, 1, "the host step probed the pass once");
                assert!(fanned + boxed > 0, "no draw was fanned out or boxed");
                assert_eq!(
                    boxed > 0,
                    triangles == Triangles::Patches,
                    "draws boxed: {boxed}"
                );
                assert_eq!(
                    flagged > 0,
                    triangles == Triangles::Wide,
                    "draws flagged: {flagged}"
                );
                assert_eq!(parted, flagged, "flagged draws drawn in parts");
                assert_eq!(flagged_triangles > 0, flagged > 0, "triangles flagged");
                assert!(
                    flagged_triangles <= by_regions,
                    "{flagged_triangles} triangles flagged of {by_regions} region by region"
                );
                assert!(
                    runs >= parted,
                    "{runs} runs region by region of {parted} draws in parts"
                );
                assert!(
                    probe_wait > 0,
                    "the host's wait for the probe was not counted"
                );
            }
            (true, false) => {
                assert_eq!((probes, fanned), (0, 0), "a draw was fanned out");
                assert!(pieces > 0, "the frame was not cut into pieces");
                assert!(ranges > 0, "its largest command was not cut into ranges");
            }
        }
        assert!(most < PIECE_LIMIT, "a piece ran {most:?}");
        assert!(
            Duration::from_nanos(probe_nanos) < PIECE_LIMIT,
            "the probe ran {probe_nanos} ns"
        );
    }
}

/// Each of `draws`' index counts' own patch of small triangles (`Triangles::Patches`): as many
/// quads a side as its indices fill, at least one, [`PATCH_QUAD`] screen pixels each, at a place of
/// the screen of its own. The patches' corners in normalized device coordinates, one after
/// another, their triangles' indices, and each command's first index and indices.
fn patches(draws: &[u32]) -> (Vec<u8>, Vec<u8>, Vec<(u32, u32)>) {
    let (mut positions, mut indices, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
    let ndc = |pixels: f32, side: u32| pixels * 2.0 / side as f32 - 1.0;
    for (k, &count) in draws.iter().enumerate() {
        let n = ((f64::from(count) / 6.0).sqrt() as u32).max(1);
        let side = n as f32 * PATCH_QUAD;
        let room = |screen: u32| (screen as f32 - side).max(1.0) as u32;
        let [x, y] = [
            (k as u32 * 7_919 % room(SCREEN[0])) as f32,
            (k as u32 * 104_729 % room(SCREEN[1])) as f32,
        ];
        let base = (positions.len() / 8) as u32;
        for j in 0..=n {
            for i in 0..=n {
                let corner = [
                    ndc(x + i as f32 * PATCH_QUAD, SCREEN[0]),
                    ndc(y + j as f32 * PATCH_QUAD, SCREEN[1]),
                ];
                positions.extend(corner.iter().flat_map(|f| f.to_le_bytes()));
            }
        }
        let corner = |i: u32, j: u32| base + j * (n + 1) + i;
        ranges.push(((indices.len() / 4) as u32, 6 * n * n));
        for quad in 0..n * n {
            let (i, j) = (quad % n, quad / n);
            let [a, b, c, d] = [
                corner(i, j),
                corner(i + 1, j),
                corner(i, j + 1),
                corner(i + 1, j + 1),
            ];
            indices.extend([a, b, c, b, d, c].iter().flat_map(|i| i.to_le_bytes()));
        }
    }
    (positions, indices, ranges)
}
