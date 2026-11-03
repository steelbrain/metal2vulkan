//! What a pass's draws cost on the device the cfg selects when each binds a resource of its own
//! and the pass makes every one resident, as a page's render bundle does with a bind group a
//! draw, without the validation layer: report only, in release builds, to find what grows with
//! the draws and with the resources made resident.
//!
//! Two shapes of draw, `resident_draws.metal`, each placing one small triangle into a 512x512
//! `RGBA8Unorm` target:
//! - **slot**: draw `i` binds position `i` (16 bytes) as vertex buffer 0 (`rd_slot_vertex`,
//!   `rd_color_fragment`); the pass makes the positions resident for its vertex stage.
//! - **entry**: draw `i` binds entry `i` of one argument table (24 bytes: the position's address,
//!   a texture's resource ID and a sampler's) as vertex and fragment buffer 0 (`rd_entry_vertex`
//!   reads the position through the pointer, `rd_entry_fragment` samples the texture); the pass
//!   makes the positions, the table and the texture resident.
//!
//! Each with the positions in a buffer of their own each (`own`), or at their offsets in one
//! (`one`). Three paths for each: the encoder's own draws (`encoder`), an indirect command
//! buffer's execution of the same draws made resident for reading (`read`: `commit` records the
//! CPU's records), and the same execution after a compute encoder made the buffer resident for
//! writing (`write`: a host step records it).
//!
//! Printed as `RESIDENT-DRAWS` lines: for each shape, binding, draw count and path, the frame's
//! time (encoding to completion) over [`frames`] frames after one to warm, median and largest,
//! per draw, and the device's counters over those frames.
//!
//! [`COUNTS_VAR`] lists the draw counts (500, 1000, 2000 and 5000 by default), [`FRAMES_VAR`] the
//! frames timed, and [`ONLY_VAR`] keeps the cases whose name (`shape binding path`, such as
//! `slot own write`) contains it.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ResourceRef, SamplerState,
    SamplerStateOptions, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Instant;

/// The variable that sets [`counts`].
const COUNTS_VAR: &str = "RESIDENT_DRAWS_COUNTS";
/// The variable that sets [`frames`].
const FRAMES_VAR: &str = "RESIDENT_DRAWS_FRAMES";
/// The variable that keeps the cases whose name contains it.
const ONLY_VAR: &str = "RESIDENT_DRAWS_ONLY";

/// The draw counts: 500, 1000, 2000 and 5000 unless [`COUNTS_VAR`] lists others, comma-separated.
fn counts() -> Vec<u64> {
    std::env::var(COUNTS_VAR).map_or(vec![500, 1000, 2000, 5000], |list| {
        list.split(',')
            .map(|count| count.trim().parse().expect(COUNTS_VAR))
            .collect()
    })
}

/// The frames timed after the one that warms: 10 unless [`FRAMES_VAR`] says.
fn frames() -> usize {
    std::env::var(FRAMES_VAR).map_or(10, |frames| frames.parse().expect(FRAMES_VAR))
}

/// What a draw binds (the module docs).
#[derive(Clone, Copy, Debug)]
enum Shape {
    Slot,
    Entry,
}

/// How a pass's draws reach the device (the module docs).
#[derive(Clone, Copy, Debug)]
enum Path {
    Encoder,
    Read,
    Write,
}

/// The bytes of an argument-table entry.
const ENTRY: u64 = 24;

/// The functions' states, the texture and sampler the entries name, the target and a queue.
struct Scene {
    device: Device,
    slot: RenderPipelineState,
    entry: RenderPipelineState,
    touch: ComputePipelineState,
    count: Buffer,
    sampled: Texture,
    sampler: SamplerState,
    target: Texture,
    queue: CommandQueue,
}

impl Scene {
    fn new(device: Device) -> Scene {
        let bitcode = llvm_assemble(&engine_dir().join("resident_draws.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let state = |vertex: &str, fragment: &str| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = true;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let touch = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(function("rd_touch")),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        sampled.usage = TextureUsage::SHADER_READ;
        sampled.resource_options = ResourceOptions::shared();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let sampler = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 512, 512, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        Scene {
            slot: state("rd_slot_vertex", "rd_color_fragment"),
            entry: state("rd_entry_vertex", "rd_entry_fragment"),
            touch,
            count: buffer_with(&device, &[0; 4]).unwrap(),
            sampled: device.new_texture(&sampled).unwrap(),
            sampler,
            target: device.new_texture(&target).unwrap(),
            queue: device.new_command_queue().unwrap(),
            device,
        }
    }
}

/// One shape's draws of one binding and count: the positions, the argument table (the entry
/// shape's), and the indirect command buffer that holds the same draws.
struct Draws<'s> {
    scene: &'s Scene,
    shape: Shape,
    positions: Vec<Buffer>,
    /// Where position `i` lies: its buffer's index among `positions` and its offset there.
    at: Vec<(usize, u64)>,
    table: Option<Buffer>,
    icb: IndirectCommandBuffer,
}

impl<'s> Draws<'s> {
    fn new(scene: &'s Scene, shape: Shape, own: bool, n: u64) -> Draws<'s> {
        let device = &scene.device;
        let position = |i: u64| -> Vec<u8> {
            [
                (i % 71) as f32 / 36.0 - 1.0,
                (i / 71) as f32 / 36.0 - 1.0,
                0.0,
                1.0,
            ]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect()
        };
        let (positions, at): (Vec<Buffer>, Vec<(usize, u64)>) = match own {
            true => (
                (0..n)
                    .map(|i| buffer_with(device, &position(i)).unwrap())
                    .collect(),
                (0..n as usize).map(|i| (i, 0)).collect(),
            ),
            false => (
                vec![buffer_with(device, &(0..n).flat_map(position).collect::<Vec<_>>()).unwrap()],
                (0..n).map(|i| (0, 16 * i)).collect(),
            ),
        };
        let table = matches!(shape, Shape::Entry).then(|| {
            let texture = scene.sampled.gpu_resource_id().0;
            let sampler = scene.sampler.gpu_resource_id().0;
            let bytes: Vec<u8> = at
                .iter()
                .flat_map(|&(buffer, offset)| {
                    let address = positions[buffer].gpu_address().0 + offset;
                    [address, texture, sampler]
                })
                .flat_map(u64::to_le_bytes)
                .collect();
            buffer_with(device, &bytes).unwrap()
        });
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&descriptor, n, ResourceOptions::shared())
            .unwrap();
        let draws = Draws {
            scene,
            shape,
            positions,
            at,
            table,
            icb,
        };
        for i in 0..n {
            let command = draws.icb.indirect_render_command_at(i).unwrap();
            command.set_render_pipeline_state(draws.state()).unwrap();
            let (buffer, offset) = draws.binding(i);
            command.set_vertex_buffer(Some(buffer), offset, 0).unwrap();
            if let Shape::Entry = shape {
                command
                    .set_fragment_buffer(Some(buffer), offset, 0)
                    .unwrap();
            }
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        draws
    }

    /// The draws' state.
    fn state(&self) -> &'s RenderPipelineState {
        match self.shape {
            Shape::Slot => &self.scene.slot,
            Shape::Entry => &self.scene.entry,
        }
    }

    /// The buffer and offset draw `i` binds.
    fn binding(&self, i: u64) -> (&Buffer, u64) {
        match &self.table {
            Some(table) => (table, ENTRY * i),
            None => {
                let (buffer, offset) = self.at[i as usize];
                (&self.positions[buffer], offset)
            }
        }
    }

    /// Make the pass's resources resident (the module docs).
    fn resident(&self, encoder: &mut RenderCommandEncoder) {
        let positions: Vec<ResourceRef> = self.positions.iter().map(ResourceRef::Buffer).collect();
        encoder
            .use_resources(&positions, ResourceUsage::READ, RenderStages::VERTEX)
            .unwrap();
        if let Some(table) = &self.table {
            let both = RenderStages::VERTEX | RenderStages::FRAGMENT;
            encoder
                .use_resource(table.as_resource(), ResourceUsage::READ, both)
                .unwrap();
            encoder
                .use_resource(
                    self.scene.sampled.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::FRAGMENT,
                )
                .unwrap();
        }
    }

    /// One frame of `path`, committed and waited for: its time from encoding to completion, in
    /// milliseconds.
    fn frame(&self, path: Path) -> f64 {
        let scene = self.scene;
        let started = Instant::now();
        let command_buffer = scene.queue.command_buffer().unwrap();
        if let Path::Write = path {
            let mut writing = command_buffer.compute_command_encoder().unwrap();
            writing.set_compute_pipeline_state(&scene.touch).unwrap();
            writing.set_buffer(Some(&scene.count), 0, 0).unwrap();
            writing
                .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            let one = Size::new(1, 1, 1);
            writing.dispatch_threads(one, one).unwrap();
            writing.end_encoding().unwrap();
        }
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(scene.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        self.resident(&mut encoder);
        let n = self.at.len() as u64;
        match path {
            Path::Encoder => {
                encoder.set_render_pipeline_state(self.state()).unwrap();
                for i in 0..n {
                    let (buffer, offset) = self.binding(i);
                    encoder.set_vertex_buffer(Some(buffer), offset, 0).unwrap();
                    if let Shape::Entry = self.shape {
                        encoder
                            .set_fragment_buffer(Some(buffer), offset, 0)
                            .unwrap();
                    }
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                }
            }
            Path::Read | Path::Write => {
                encoder.execute_commands_in_buffer(&self.icb, 0..n).unwrap();
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        started.elapsed().as_secs_f64() * 1e3
    }
}

/// The counters `after` minus `before`, each a frame's over `frames`, printed.
fn counters(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let per = |f: fn(&WorkCounts) -> u64| (f(after) - f(before)) as f64 / frames as f64;
    format!(
        "a frame's host steps {:.0} taking {:.2} ms, records read {:.0}, chunks kept {:.0}, \
         graphics pipelines looked up {:.0}, heap tables worked out {:.0}, resident resources \
         joined {:.0}",
        per(|c| c.host_steps_recorded),
        per(|c| c.host_step_nanos) / 1e6,
        per(|c| c.indirect_records_read),
        per(|c| c.indirect_chunks_kept),
        per(|c| c.graphics_pipelines_looked_up),
        per(|c| c.heap_tables_worked_out),
        per(|c| c.resident_resources_joined),
    )
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn resident_draws() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let scene = Scene::new(device);
    let frames = frames();
    let only = std::env::var(ONLY_VAR).ok();
    for shape in [Shape::Slot, Shape::Entry] {
        for own in [true, false] {
            let binding = if own { "own" } else { "one" };
            for n in counts() {
                let draws = Draws::new(&scene, shape, own, n);
                // In this order: a buffer made resident for writing stays written by the GPU.
                for path in [Path::Encoder, Path::Read, Path::Write] {
                    let case = format!("{shape:?} {binding} {path:?}").to_lowercase();
                    if only.as_deref().is_some_and(|only| !case.contains(only)) {
                        continue;
                    }
                    draws.frame(path);
                    let before = scene.device.work_counts();
                    let mut times: Vec<f64> = (0..frames).map(|_| draws.frame(path)).collect();
                    let after = scene.device.work_counts();
                    times.sort_by(f64::total_cmp);
                    let median = times[times.len() / 2];
                    eprintln!(
                        "RESIDENT-DRAWS {name}: {case} N {n}: frame {median:.2} ms median, \
                         {:.2} largest, {:.2} us a draw; {}",
                        times[times.len() - 1],
                        median * 1e3 / n as f64,
                        counters(&before, &after, frames)
                    );
                }
            }
        }
    }
}
