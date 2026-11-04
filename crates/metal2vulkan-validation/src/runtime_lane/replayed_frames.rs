//! A page's frame whose indirect command buffer its own vertex function writes, on the device the
//! cfg selects, without the validation layer, while two other queues commit work: what the host
//! step of the execution costs, and how long the other queues' command buffers wait behind it.
//!
//! The owned fixture `icb_vertex_encode_indexed`: a pass with rasterization off draws one point
//! per instance [`DRAWS`] times, setting `ive_encode`'s state before each, as a page's validation
//! pass does; the draw of instance `i` encodes command `i`, a draw of 6 16-bit indices of
//! primitive type `i % 5` (`ive_encode` may call `draw_indexed_primitives` alone). The pass makes
//! the buffer resident for writing, the indices for reading, and a texture the presenter samples
//! for reading. A second pass executes the buffer's [`DRAWS`] commands, which inherit its state
//! (`ive_vertex`, `ive_fragment`) and buffers: the GPU may write the records, so a host step
//! records the execution from them.
//!
//! Meanwhile a presenter queue commits a blit that reads the texture, and a bystander queue one
//! that fills a buffer of its own, each one at a time, again as each completes, until the frame
//! completes. [`replayed_frames`] prints, per frame (the first cold, the rest warm), as
//! `REPLAYED-FRAMES` lines: the frame's time from commit to completion, the counters' moves that
//! split it (host steps and their time, pipelines made and their time, records read), and the
//! presenter's and the bystander's command buffers' times from commit to completion, largest and
//! count. [`FRAMES_VAR`] sets the frames (4 by default), and [`RESIDENT_VAR`] how many buffers
//! and textures, one of each, the execution's pass makes resident besides, as a page's bind
//! groups do (none by default): the step's time should grow with the records alone, not with
//! them times the resident resources. It is ignored by default: run it by name with
//! `--run-ignored all`.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// The variable that sets the frames timed.
const FRAMES_VAR: &str = "REPLAYED_FRAMES_FRAMES";

/// The variable that sets how many buffers and textures the execution's pass makes resident
/// besides (none by default).
const RESIDENT_VAR: &str = "REPLAYED_FRAMES_RESIDENT";

/// The commands the pass encodes, one a draw.
const DRAWS: u32 = 5000;

/// The target's width and height.
const SIDE: u64 = 64;

/// The fixture's states and buffers on a device.
struct Page {
    encode: RenderPipelineState,
    draw: RenderPipelineState,
    icb: IndirectCommandBuffer,
    table: Buffer,
    indices: Buffer,
    target: Texture,
    surface: Texture,
    /// The buffers and textures the execution's pass makes resident besides, as a page's bind
    /// groups, one of each a draw, do ([`RESIDENT_VAR`]).
    resident: Vec<(Buffer, Texture)>,
}

impl Page {
    fn new(device: &Device) -> Page {
        let bitcode = llvm_assemble(&engine_dir().join("icb_vertex_encode_indexed.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |library: &Library, name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let rgba = PixelFormat::RGBA8Unorm;
        let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ive_encode"));
        descriptor.rasterization_enabled = false;
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        let options = PipelineOptions::default();
        let encode = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ive_vertex"));
        descriptor.fragment = Some(function(&library, "ive_fragment"));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        descriptor.support_indirect_command_buffers = true;
        let draw = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW_INDEXED;
        commands.inherit_pipeline_state = true;
        commands.inherit_buffers = true;
        let icb = device
            .new_indirect_command_buffer(&commands, u64::from(DRAWS), ResourceOptions::shared())
            .unwrap();
        let table = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let indices: Vec<u8> = (0..64u16 * 6)
            .map(|at| (at * 37 + 11) % 64)
            .flat_map(u16::to_le_bytes)
            .collect();
        let indices = buffer_with(device, &indices).unwrap();
        let texture = |usage: TextureUsage| {
            let mut descriptor = TextureDescriptor::texture_2d(rgba, SIDE, SIDE, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::shared();
            device.new_texture(&descriptor).unwrap()
        };
        let resident = std::env::var(RESIDENT_VAR).map_or(0, |count| count.parse().unwrap());
        let resident = (0..resident)
            .map(|_| {
                let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
                let mut descriptor = TextureDescriptor::texture_2d(rgba, 4, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                (buffer, device.new_texture(&descriptor).unwrap())
            })
            .collect();
        Page {
            encode,
            draw,
            icb,
            table,
            indices,
            target: texture(TextureUsage::RENDER_TARGET),
            surface: texture(TextureUsage::SHADER_READ),
            resident,
        }
    }

    fn pass(&self) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        pass
    }

    /// One frame on `queue`, committed: the encoding pass, then the execution's.
    fn frame(&self, queue: &CommandQueue) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        let vertex = RenderStages::VERTEX;
        let uses = [
            (self.icb.as_resource(), ResourceUsage::WRITE),
            (self.indices.as_resource(), ResourceUsage::READ),
            (self.table.as_resource(), ResourceUsage::READ),
            (self.surface.as_resource(), ResourceUsage::READ),
        ];
        for (resource, usage) in uses {
            encoder.use_resource(resource, usage, vertex).unwrap();
        }
        encoder.set_vertex_buffer(Some(&self.table), 0, 0).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.indices), 0, 1)
            .unwrap();
        for draw in 0..DRAWS {
            encoder.set_render_pipeline_state(&self.encode).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Point, 0, 1, 1, draw)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        encoder.set_render_pipeline_state(&self.draw).unwrap();
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        for (buffer, texture) in &self.resident {
            for resource in [buffer.as_resource(), texture.as_resource()] {
                encoder
                    .use_resource(resource, ResourceUsage::READ, stages)
                    .unwrap();
            }
        }
        encoder
            .execute_commands_in_buffer(&self.icb, 0..u64::from(DRAWS))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    }
}

/// What another queue's command buffers took, one at a time from commit to completion, while
/// `done` is unset.
#[derive(Debug, Default)]
struct Waits {
    count: usize,
    largest: Duration,
    total: Duration,
}

/// Commit `encode`'s command buffers on `queue`, each after the last completed, until `done`.
fn commit_until(queue: &CommandQueue, done: &AtomicBool, encode: impl Fn(&CommandBuffer)) -> Waits {
    let mut waits = Waits::default();
    while !done.load(Ordering::Acquire) {
        let command_buffer = queue.command_buffer().unwrap();
        encode(&command_buffer);
        let start = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let took = start.elapsed();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        waits.count += 1;
        waits.total += took;
        waits.largest = waits.largest.max(took);
    }
    waits
}

/// The frames to time: 4 unless [`FRAMES_VAR`] sets another count.
fn frames() -> usize {
    std::env::var(FRAMES_VAR).map_or(4, |count| count.trim().parse().expect(FRAMES_VAR))
}

/// What `after` moved past `before` of the counters that split a frame.
fn split(before: &WorkCounts, after: &WorkCounts) -> String {
    let ms = |nanos: u64| nanos as f64 / 1e6;
    format!(
        "host_steps {} host_step_ms {:.1} render_pipelines_made_ms {:.1} \
         compute_pipelines_made_ms {:.1} variant_pipelines_made {} records_read {}",
        after.host_steps_recorded - before.host_steps_recorded,
        ms(after.host_step_nanos - before.host_step_nanos),
        ms(after.render_pipeline_create_nanos - before.render_pipeline_create_nanos),
        ms(after.compute_pipeline_create_nanos - before.compute_pipeline_create_nanos),
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.indirect_records_read - before.indirect_records_read,
    )
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn replayed_frames() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let page = Page::new(&device);
    let (page_queue, presenter, bystander) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let read = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let own = device
        .new_buffer(1 << 16, ResourceOptions::shared())
        .unwrap();
    for frame in 0..frames() {
        let done = AtomicBool::new(false);
        let before = device.work_counts();
        let (took, presented, stood) = thread::scope(|scope| {
            let presented = scope.spawn(|| {
                commit_until(&presenter, &done, |command_buffer| {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    let size = Size::new(SIDE, SIDE, 1);
                    blit.copy_from_texture_to_buffer(
                        &page.surface,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        size,
                        &read,
                        0,
                        SIDE * 4,
                        SIDE * SIDE * 4,
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                })
            });
            let stood = scope.spawn(|| {
                commit_until(&bystander, &done, |command_buffer| {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&own, 0..1 << 16, 7).unwrap();
                    blit.end_encoding().unwrap();
                })
            });
            let start = Instant::now();
            let command_buffer = page.frame(&page_queue);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let took = start.elapsed();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            done.store(true, Ordering::Release);
            (took, presented.join().unwrap(), stood.join().unwrap())
        });
        let after = device.work_counts();
        println!(
            "REPLAYED-FRAMES {name} frame {frame}: {:.1} ms (encoding to completion), {}; \
             presenter {} command buffers, largest {:.1} ms, mean {:.1} ms; bystander {}, \
             largest {:.1} ms, mean {:.1} ms",
            took.as_secs_f64() * 1e3,
            split(&before, &after),
            presented.count,
            presented.largest.as_secs_f64() * 1e3,
            presented.total.as_secs_f64() * 1e3 / presented.count.max(1) as f64,
            stood.count,
            stood.largest.as_secs_f64() * 1e3,
            stood.total.as_secs_f64() * 1e3 / stood.count.max(1) as f64,
        );
    }
}
