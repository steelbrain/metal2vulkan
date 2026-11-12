//! The pieces a rate-mapped pass's executed commands are recorded in on the device the cfg
//! selects (`rate_mapped_pass.rs`, **Pieces**), at the density of a guest's rate-mapped frame:
//! report and bound, in release builds.
//!
//! The workload is one execution of the shape of the consumer's (macOS guests over the runtime):
//! a map of one layer over a [`SCREEN`] screen whose quality falls from 1 at the centre to 0.3 at
//! the edges on both axes ([`QUALITY`]), 2,074 regions; a pass into a target of the map's
//! physical size; and an indirect command buffer's [`COUNTS`] commands executed over a range read
//! from a buffer, each indexed triangles of one of the index counts a guest's executed commands
//! drew (their 10th, 50th, 90th and 99th percentiles, their mean and their most), drawn from a
//! grid of small triangles over the screen (`tob_triangle`, `tob_solid`). That holds about
//! 1.9 G region-indices of work, 28 times the piece work: about 0.8 s of an Iris Xe's GPU at its
//! trivial vertex function, under a second, so a device that cut nothing would still not reach
//! amdgpu's 2 s job timeout. Printed as a `RATE-MAP-PIECES` line: the frame's GPU time, the
//! pieces and ranges it was cut into, and its pieces' GPU times where the device times them.
//! Checked: the frame was cut into pieces and its largest command into ranges, and no piece ran
//! past [`PIECE_LIMIT`].
//!
//! Ignored by default: a measurement on a GPU, far too slow on a CPU device. Run it by name with
//! `--run-ignored all`. The fixture's text is compiled in, so the binary also runs on a machine
//! other than the one that built it.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, StoreAction, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

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

/// The grid's quads on a side: its triangles' indices hold the most indices a command draws.
const GRID: u32 = 165;

/// The longest a piece may run: half amdgpu's job timeout.
const PIECE_LIMIT: Duration = Duration::from_secs(1);

/// The counts of `work`: the pieces past the first, the index ranges past the first, the pieces
/// timed, their GPU time and the longest's.
fn counts(work: &WorkCounts) -> [u64; 5] {
    [
        work.rate_map_pieces,
        work.rate_map_index_ranges,
        work.rate_map_pieces_timed,
        work.rate_map_piece_gpu_nanos,
        work.rate_map_piece_gpu_nanos_most,
    ]
}

/// One frame of the commands, cut into pieces: printed and bounded (the module docs).
#[test]
#[ignore = "a measurement on a GPU: run by name with --run-ignored all"]
fn rate_mapped_pieces() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
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
            [a, b, c, b, d, c]
        })
        .flat_map(u32::to_le_bytes)
        .collect();
    let grid_indices = GRID * GRID * 6;
    let positions = buffer_with(&device, &positions).unwrap();
    let index_buffer = buffer_with(&device, &indices).unwrap();
    // The commands in an order that spreads the large ones, each its counts' indices from a
    // place of the grid, a whole quad's, and its own colour.
    let mut draws: Vec<u32> = (COUNTS.iter())
        .flat_map(|&(count, commands)| std::iter::repeat_n(count, commands as usize))
        .collect();
    let spread = draws.len();
    draws = (0..spread).map(|k| draws[k * 37 % spread]).collect();
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
    let icb = device
        .new_indirect_command_buffer(&icb, spread as u64, ResourceOptions::shared())
        .unwrap();
    for (k, &count) in draws.iter().enumerate() {
        let command = icb.indirect_render_command_at(k as u64).unwrap();
        command.set_render_pipeline_state(&pipeline).unwrap();
        command.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
        command
            .set_fragment_buffer(Some(&colors), k as u64 * 256, 0)
            .unwrap();
        let room = (grid_indices - count) / 6;
        let start = (k as u32 * 7_919 % (room + 1)) * 6;
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
    let range = [0u32, spread as u32].map(u32::to_le_bytes).concat();
    let range = buffer_with(&device, &range).unwrap();
    let mut map = RateMapDescriptor::new(SCREEN);
    map.layers = vec![RateMapLayerDescriptor::new(
        QUALITY.to_vec(),
        QUALITY.to_vec(),
    )];
    let map = device.new_rasterization_rate_map(&map).unwrap();
    let size = map.physical_size_for_layer(0);
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let queue = device.new_command_queue().unwrap();
    let before = counts(&device.work_counts());
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::DontCare;
    color.store_action = StoreAction::Store;
    pass.rasterization_rate_map = Some(map.clone());
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder
        .execute_commands_in_buffer_indirect(&icb, &range, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    drop(command_buffer);
    let after = counts(&device.work_counts());
    let [pieces, ranges, timed, nanos] = [0, 1, 2, 3].map(|i| after[i] - before[i]);
    let most = Duration::from_nanos(after[4]);
    let indices: u64 = draws.iter().map(|&count| u64::from(count)).sum();
    let mean = Duration::from_nanos(nanos / timed.max(1));
    eprintln!(
        "RATE-MAP-PIECES {name}: screen {}x{}, physical {}x{}, {spread} commands of {indices} \
         indices: the command buffer {:.1} ms of GPU time, {} pieces ({ranges} index ranges past \
         the first); {timed} pieces timed, the longest {:.2} ms, the mean {:.2} ms",
        SCREEN[0],
        SCREEN[1],
        size.width,
        size.height,
        gpu * 1e3,
        pieces + 1,
        most.as_secs_f64() * 1e3,
        mean.as_secs_f64() * 1e3,
    );
    assert!(pieces > 0, "the frame was not cut into pieces");
    assert!(ranges > 0, "its largest command was not cut into ranges");
    assert!(most < PIECE_LIMIT, "a piece ran {most:?}");
}
