//! What a stream of draws that each bind vertex buffer 0 at the next offset of one buffer costs the
//! thread that encodes them, by how the vertex function reads the slot: through a constant
//! reference (`constant float4 &`, `cof_constant`) or a device pointer at an index it cannot bound
//! (`const device float4 *`, `cof_device`, both of `cof_offsets.metal`), on the device the cfg
//! selects, without the validation layer: report only, in release builds.
//!
//! The stream is a guest's small draws as a consumer that stages each draw's `setVertexBytes` in
//! upload chunks sends them: each draw of 3 vertices binds a chunk of [`CHUNK`] bytes at vertex
//! index 0, at the next [`STEP`]-byte offset of the chunk, the next of [`CHUNKS`] chunks once one
//! is used up; the fragment function's colour is bound once a pass (`setFragmentBytes`). The
//! chunks are shared buffers, or no-copy buffers over host pages, as a guest's memory is
//! (`over pages`; a device that imports no host memory runs no such shape, and says so). A
//! command buffer is one render pass of [`DRAWS`] draws into a `RGBA8Unorm` target of
//! [`TARGET_SIDE`] square, committed and waited for. Each shape runs [`WARM`] command buffers
//! untimed, then [`TIMED`], printed as a `CONSTANT-OFFSETS` line (the medians of the encoding
//! thread's CPU time a draw, from the encoder's creation through `end_encoding`) and a
//! `CONSTANT-OFFSETS-MOVED` line (every work count the timed command buffers moved, a draw, by
//! name: the draws moved and moved inside, and the set-0 writes pushed, among them).
//!
//! `constant_offsets` measures every shape in turn; `constant_offsets_profiled_constant` and
//! `constant_offsets_profiled_device` encode [`PROFILED`] command buffers of one form over host
//! pages alone (shared chunks where the device imports none), for a profiler.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::draw_costs::{percentiles, thread_cpu};
use super::host_memory::{host_page, no_copy, not_importable, Mapping};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};
use std::time::Duration;

/// The untimed and the timed command buffers of each shape.
const WARM: usize = 3;
const TIMED: usize = 9;
/// The command buffers a profiled run encodes.
const PROFILED: usize = 40;
/// The draws of a command buffer's pass.
const DRAWS: usize = 10_000;
/// The bytes of an upload chunk, the offset step between two draws' bytes in it, and the chunks
/// the draws take in turn.
const CHUNK: u64 = 64 * 1024;
const STEP: u64 = 256;
const CHUNKS: usize = 4;
/// The target's side.
const TARGET_SIDE: u64 = 256;

/// How the vertex function reads its float4 (the module docs).
#[derive(Clone, Copy, Debug)]
enum Form {
    Constant,
    Device,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Constant => "constant reference",
            Form::Device => "device pointer",
        }
    }

    fn function(self) -> &'static str {
        match self {
            Form::Constant => "cof_constant",
            Form::Device => "cof_device",
        }
    }
}

/// The chunks of a [`Workload`], over host pages where the device imports them.
struct Chunks {
    shared: Vec<Buffer>,
    over_pages: Option<Vec<Buffer>>,
    /// The host pages the chunks over pages lie in, which outlive them (`Workload`'s order).
    _pages: Vec<Mapping>,
}

/// The device, its queue, the pipeline state of each form, the chunks and the target.
struct Workload {
    queue: CommandQueue,
    /// The constant reference's pipeline state, and the device pointer's.
    pipelines: [RenderPipelineState; 2],
    chunks: Chunks,
    target: Texture,
    device: Device,
}

/// Chunk `n`'s bytes: at each step a small triangle's float4, one of a grid over the target.
fn chunk_bytes(n: usize) -> Vec<u8> {
    let per_chunk = (CHUNK / STEP) as usize;
    let mut bytes = vec![0u8; CHUNK as usize];
    for slot in 0..per_chunk {
        let at = n * per_chunk + slot;
        let (x, y) = ((at % 32) as f32, ((at / 32) % 32) as f32);
        let q = [x / 16.0 - 1.0, y / 16.0 - 1.0, 1.0 / 16.0, 1.0 / 16.0];
        let words: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
        let start = slot * STEP as usize;
        bytes[start..start + words.len()].copy_from_slice(&words);
    }
    bytes
}

impl Workload {
    fn new(device: Device) -> Workload {
        let ll = engine_dir().join("cof_offsets.ll");
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
        let pipeline = |form: Form| {
            let mut descriptor = RenderPipelineDescriptor::new(function(form.function()));
            descriptor.fragment = Some(function("cof_flat"));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let shared = (0..CHUNKS)
            .map(|n| super::buffer_with(&device, &chunk_bytes(n)).unwrap())
            .collect();
        let len = (CHUNK as usize).next_multiple_of(host_page());
        let pages: Vec<Mapping> = (0..CHUNKS).map(|_| Mapping::anonymous(len)).collect();
        let over_pages = (pages.iter().enumerate())
            .map(|(n, mapping)| {
                mapping.write(0, &chunk_bytes(n));
                // SAFETY: the pages are this process's, mapped read and write, and outlive the
                // buffer (`Chunks`' order).
                unsafe {
                    no_copy(
                        &device,
                        mapping.pointer(),
                        len,
                        ResourceOptions::shared(),
                        None,
                    )
                }
            })
            .collect::<Result<Vec<Buffer>, _>>();
        let over_pages = match over_pages {
            Ok(buffers) => Some(buffers),
            Err(error) if not_importable(&error) => None,
            Err(error) => panic!("chunks over host pages: {error:?}"),
        };
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET_SIDE, TARGET_SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        Workload {
            queue: device.new_command_queue().unwrap(),
            pipelines: [pipeline(Form::Constant), pipeline(Form::Device)],
            chunks: Chunks {
                shared,
                over_pages,
                _pages: pages,
            },
            target: device.new_texture(&target).unwrap(),
            device,
        }
    }

    /// The chunks over host pages where `over_pages` and the device imports them, else the shared
    /// ones; `None` for chunks over pages the device does not import.
    fn chunks(&self, over_pages: bool) -> Option<&[Buffer]> {
        match over_pages {
            true => self.chunks.over_pages.as_deref(),
            false => Some(&self.chunks.shared),
        }
    }

    /// Encode one command buffer of `form` over `chunks`, commit it and wait for it: the encoding
    /// thread's CPU time its encoding took.
    fn run(&self, form: Form, chunks: &[Buffer]) -> Duration {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        let cpu_start = thread_cpu();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let pipeline = match form {
            Form::Constant => &self.pipelines[0],
            Form::Device => &self.pipelines[1],
        };
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let colour = [0.25f32, 0.5, 0.75, 1.0].map(f32::to_le_bytes).concat();
        encoder.set_fragment_bytes(&colour, 0).unwrap();
        let per_chunk = (CHUNK / STEP) as usize;
        for draw in 0..DRAWS {
            let chunk = &chunks[(draw / per_chunk) % chunks.len()];
            let offset = (draw % per_chunk) as u64 * STEP;
            encoder.set_vertex_buffer(Some(chunk), offset, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let cpu = thread_cpu().saturating_sub(cpu_start);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        cpu
    }
}

/// The workload on the cfg's device, as a consumer opens it: no layer, no execution records,
/// the default policy.
fn workload(cache: &ScratchCache, messages: &Messages) -> Workload {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        options.record_executions = false;
        options.policy = DeviceOptions::default().policy;
    })
    .unwrap();
    Workload::new(device)
}

/// A shape's name: its form, and where its chunks lie.
fn shape_name(form: Form, over_pages: bool) -> String {
    match over_pages {
        true => format!("{} over pages", form.name()),
        false => form.name().to_owned(),
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn constant_offsets() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let name = work.device.name().to_owned();
    eprintln!(
        "CONSTANT-OFFSETS {name}: {DRAWS} draws a pass, each binding vertex buffer 0 at the next \
         {STEP}-byte offset of a {CHUNK}-byte chunk ({CHUNKS} in turn), {WARM} untimed and \
         {TIMED} timed command buffers a shape"
    );
    let shapes = [
        (Form::Constant, false),
        (Form::Device, false),
        (Form::Constant, true),
        (Form::Device, true),
    ];
    for (form, over_pages) in shapes {
        let shape = shape_name(form, over_pages);
        let Some(chunks) = work.chunks(over_pages) else {
            eprintln!("CONSTANT-OFFSETS {name} {shape}: the device imports no host memory");
            continue;
        };
        for _ in 0..WARM {
            work.run(form, chunks);
        }
        let before = work.device.work_counts();
        let runs: Vec<f64> = (0..TIMED)
            .map(|_| work.run(form, chunks).as_secs_f64() * 1e9 / DRAWS as f64)
            .collect();
        let after = work.device.work_counts();
        let [low, cpu, high] = percentiles(runs);
        eprintln!(
            "CONSTANT-OFFSETS {name} {shape}: {cpu:.1} ns of the thread's CPU a draw (p10 \
             {low:.1}, p90 {high:.1})"
        );
        let moved = super::video_frames::moved_per_frame(&before, &after, TIMED * DRAWS);
        eprintln!("CONSTANT-OFFSETS-MOVED {name} {shape}: a draw: {moved}");
    }
}

/// Encode [`PROFILED`] command buffers of `form` over host pages alone (shared chunks where the
/// device imports none), for a profiler.
fn profiled(form: Form) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let over_pages = work.chunks(true).is_some();
    let chunks = work.chunks(over_pages).unwrap();
    for _ in 0..WARM {
        work.run(form, chunks);
    }
    let runs: Vec<f64> = (0..PROFILED)
        .map(|_| work.run(form, chunks).as_secs_f64() * 1e9 / DRAWS as f64)
        .collect();
    let [_, cpu, _] = percentiles(runs);
    eprintln!(
        "CONSTANT-OFFSETS {} {} profiled: {cpu:.1} ns of the thread's CPU a draw",
        work.device.name(),
        shape_name(form, over_pages)
    );
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn constant_offsets_profiled_constant() {
    profiled(Form::Constant);
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn constant_offsets_profiled_device() {
    profiled(Form::Device);
}
