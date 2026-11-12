//! What a render pass over a rasterization-rate map costs to encode on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the rate-mapped
//! draw path before and after a change to it.
//!
//! The workload is a rate-mapped frame of the consumer's shape (macOS guests over the runtime): a
//! map of one layer over a [`SCREEN`] screen whose quality falls from 1 at the centre to 0.3 at
//! the edges on both axes ([`QUALITY`]), a pass into a target of the map's physical size, and
//! [`DRAWS`] indexed draws of a small rectangle each (`tob_rect`, `tob_solid`): half encoded on
//! the encoder, each its own rectangle's bytes, and half through an indirect command buffer the CPU
//! encoded, its buffers inherited, executed once. Each frame is a command buffer of its own, and
//! [`WARM`] untimed frames precede [`TIMED`] timed ones. The same frame over no map is the
//! baseline. Each is printed as a `RATE-MAP-COST` line: the draws the device recorded a frame
//! (`WorkCounts::commands_recorded`: draws, viewports and scissors set) and those per guest draw
//! (each draw's regions, in a rate-mapped pass), and the median of the encoding thread's own CPU
//! time and of the wall time a frame, from the encoder's creation through `commit`, per guest
//! draw and per recorded draw.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::draw_costs::{percentiles, thread_cpu};
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RasterizationRateMap,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// The untimed and the timed frames of each pass.
const WARM: usize = 1;
const TIMED: usize = 3;
/// The guest draws of a frame: half on the encoder, half through the indirect command buffer.
const DRAWS: usize = 256;
/// The map's screen.
const SCREEN: [u32; 2] = [3840, 2160];
/// The map's quality samples, on both axes.
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// The device, its queue, and what the frames draw with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    indices: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
}

/// One frame's times and the device's counts over it ([`Workload::frame`]).
struct Frame {
    cpu: Duration,
    wall: Duration,
    counts: [u64; 3],
}

impl Workload {
    fn new(device: Device) -> Workload {
        let ll = engine_dir().join("tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
        descriptor.fragment = Some(function("tob_solid"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let indices = [0u16, 1, 2, 3, 4, 5].map(u16::to_le_bytes).concat();
        let indices = buffer_with(&device, &indices).unwrap();
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW_INDEXED;
        icb.inherit_buffers = true;
        let commands = (DRAWS / 2) as u64;
        let icb = device
            .new_indirect_command_buffer(&icb, commands, ResourceOptions::shared())
            .unwrap();
        for index in 0..commands {
            let command = icb.indirect_render_command_at(index).unwrap();
            command.set_render_pipeline_state(&pipeline).unwrap();
            command
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    6,
                    IndexType::UInt16,
                    &indices,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
        }
        let mut map = RateMapDescriptor::new(SCREEN);
        map.layers = vec![RateMapLayerDescriptor::new(
            QUALITY.to_vec(),
            QUALITY.to_vec(),
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        Workload {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            indices,
            icb,
            map,
            device,
        }
    }

    /// A private target of the map's physical size.
    fn target(&self) -> Texture {
        let size = self.map.physical_size_for_layer(0);
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        self.device.new_texture(&target).unwrap()
    }

    /// Encode one frame into `target`, over the map where `mapped`, commit it and wait for it.
    fn frame(&self, target: &Texture, mapped: bool) -> Frame {
        let before = counts(&self.device.work_counts());
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        if mapped {
            pass.rasterization_rate_map = Some(self.map.clone());
        }
        let start = Instant::now();
        let cpu_start = thread_cpu();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        let color = [0.25f32, 0.5, 0.75, 1.0].map(f32::to_le_bytes).concat();
        encoder.set_fragment_bytes(&color, 0).unwrap();
        // Each rectangle one of a 32 by 24 grid over the screen; the commands take the last.
        let rect = |draw: usize| {
            let (x, y) = ((draw % 32) as f32, (draw / 32 % 24) as f32);
            let at = |v: f32, n: f32| v * 2.0 / n - 1.0;
            [
                at(x, 32.0),
                at(y, 24.0),
                at(x + 1.0, 32.0),
                at(y + 1.0, 24.0),
            ]
            .map(f32::to_le_bytes)
            .concat()
        };
        for draw in 0..DRAWS / 2 {
            encoder.set_vertex_bytes(&rect(draw), 0).unwrap();
            encoder
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    6,
                    IndexType::UInt16,
                    &self.indices,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer(&self.icb, 0..self.icb.size())
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let wall = start.elapsed();
        let cpu = thread_cpu().saturating_sub(cpu_start);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = counts(&self.device.work_counts());
        Frame {
            cpu,
            wall,
            counts: [0, 1, 2].map(|i| after[i] - before[i]),
        }
    }
}

/// The draws, viewports and scissors the device recorded so far.
fn counts(work: &WorkCounts) -> [u64; 3] {
    assert_eq!(
        work.commands_uncounted, 0,
        "the device's commands are counted"
    );
    let recorded = &work.commands_recorded;
    [
        recorded.cmd_draw_indexed + recorded.cmd_draw,
        recorded.cmd_set_viewport,
        recorded.cmd_set_scissor,
    ]
}

/// The rate-mapped pass's costs, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn rate_map_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let work = Workload::new(device);
    let target = work.target();
    let size = work.map.physical_size_for_layer(0);
    eprintln!(
        "RATE-MAP-COST {name}: screen {}x{}, quality {QUALITY:?} on both axes, physical {}x{}; \
         {DRAWS} indexed draws a frame ({} encoded, {} through an indirect command buffer), \
         {WARM} untimed and {TIMED} timed frames",
        SCREEN[0],
        SCREEN[1],
        size.width,
        size.height,
        DRAWS / 2,
        DRAWS / 2,
    );
    for mapped in [false, true] {
        for _ in 0..WARM {
            work.frame(&target, mapped);
        }
        let frames: Vec<Frame> = (0..TIMED).map(|_| work.frame(&target, mapped)).collect();
        let [draws, viewports, scissors] = frames[TIMED - 1].counts;
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let [_, cpu, _] = percentiles(frames.iter().map(|f| ms(f.cpu)).collect());
        let [_, wall, _] = percentiles(frames.iter().map(|f| ms(f.wall)).collect());
        eprintln!(
            "RATE-MAP-COST {name}: {}: {draws} draws, {viewports} viewports and {scissors} \
             scissors set a frame ({:.1} draws a guest draw); encode through commit {cpu:.2} ms \
             of the thread's CPU, {wall:.2} ms wall a frame: {:.2} us CPU a guest draw, {:.0} ns \
             CPU a recorded draw",
            match mapped {
                true => "rate-mapped",
                false => "no map",
            },
            draws as f64 / DRAWS as f64,
            cpu * 1e3 / DRAWS as f64,
            cpu * 1e6 / draws.max(1) as f64,
        );
    }
}
