//! What encoding render passes that each end with a draw reading its own color attachment through
//! a snapshot (`render/snapshot.rs`) costs the thread that encodes them, on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the kept,
//! rebound and carried draw paths (`render/draw_again.rs`, `render/carried.rs`) before and after a
//! change to them.
//!
//! The workload is shaped as the consumer's guest Launchpad walk measured it: render encoders of a
//! few draws each, nearly every one holding a draw that reads its own color through a snapshot,
//! most of them followed by a render encoder of the same pass shape. A frame is one command
//! buffer of [`PASSES`] render encoders into one private `BGRA8Unorm` target of
//! [`TARGET_WIDTH`]x[`TARGET_HEIGHT`], each loading and storing it: the reader (`tob_read`, which
//! writes the texel of its fragment texture 0, the target, under each fragment) over a rectangle
//! of its own, then [`PLAIN`] draws of `tob_sample` (fragment texture 0 one of [`TEXTURES`]
//! textures, sampler 0) over rectangles of their own, then the reader again over another
//! rectangle. Each draw sets its rectangle's corners as inline vertex bytes; the reader's two
//! draws set the pipeline state and bind the target at fragment texture 0, the first plain draw
//! the pipeline state, and each plain draw its texture. The first reader keeps the pass's first
//! instance out of a primary (`render/inline_first.rs`), so every draw records into that
//! instance, and the reader is the encoder's last draw. Through the whole target's scissor the
//! area rule takes the snapshot for a reader of a 4-byte color this large, and a device that
//! measures readers keeps the snapshot for one that covers a rectangle this small
//! (`render/coverage.rs`). Each frame is committed and waited for.
//!
//! [`WARM`] frames untimed, then [`TIMED`] frames, printed as a `READER-COST` line: the medians of
//! the encoding thread's CPU time and wall time a frame, from the first encoder's creation through
//! the last one's `end_encoding`, and per draw; then a `READER-COUNTS` line with the work counts
//! the timed frames moved a frame (`Device::work_counts`): the kept and carried draws' counts and
//! the readers'.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::draw_costs::{percentiles, thread_cpu};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, CommandQueue, Device, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// The untimed and the timed frames.
const WARM: usize = 20;
const TIMED: usize = 200;
/// The render encoders of a frame.
const PASSES: usize = 2;
/// The plain draws between an encoder's two readers.
const PLAIN: usize = 4;
/// The draws of an encoder: its two readers and its plain draws.
const DRAWS: usize = PLAIN + 2;
/// The textures the plain draws take in turn, and their side.
const TEXTURES: usize = 4;
const TEXTURE_SIDE: u64 = 64;
/// The target's size.
const TARGET_WIDTH: u64 = 1920;
const TARGET_HEIGHT: u64 = 1080;

/// The device, its queue, and what a frame draws with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    reader: RenderPipelineState,
    plain: RenderPipelineState,
    sampler: SamplerState,
    textures: Vec<Texture>,
    target: Texture,
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
        let pipeline = |fragment| {
            let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
            descriptor.fragment = Some(function(fragment));
            let color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
            descriptor.color_attachments[0] = Some(color);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let texture = |n: usize| {
            let mut descriptor = TextureDescriptor::texture_2d(
                PixelFormat::RGBA8Unorm,
                TEXTURE_SIDE,
                TEXTURE_SIDE,
                false,
            );
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&descriptor).unwrap();
            let texel = [n as u8 * 16, 255 - n as u8 * 16, 128, 255];
            let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
                .flat_map(|_| texel)
                .collect();
            let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
            texture
                .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
                .unwrap();
            texture
        };
        let mut target = TextureDescriptor::texture_2d(
            PixelFormat::BGRA8Unorm,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            false,
        );
        target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        target.resource_options = ResourceOptions::private();
        Workload {
            queue: device.new_command_queue().unwrap(),
            reader: pipeline("tob_read"),
            plain: pipeline("tob_sample"),
            sampler,
            textures: (0..TEXTURES).map(texture).collect(),
            target: device.new_texture(&target).unwrap(),
            device,
        }
    }

    /// Encode one frame (the module docs), commit it and wait for it: the wall time and the
    /// encoding thread's CPU time its encoding took.
    fn run(&self) -> (Duration, Duration) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let start = Instant::now();
        let cpu_start = thread_cpu();
        for at in 0..PASSES {
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&self.sampler), 0)
                .unwrap();
            self.read(&mut encoder, at * DRAWS);
            encoder.set_render_pipeline_state(&self.plain).unwrap();
            for draw in 1..=PLAIN {
                let texture = &self.textures[draw % TEXTURES];
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                encoder
                    .set_vertex_bytes(&rectangle(at * DRAWS + draw), 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            self.read(&mut encoder, at * DRAWS + DRAWS - 1);
            encoder.end_encoding().unwrap();
        }
        let encoded = start.elapsed();
        let cpu = thread_cpu().saturating_sub(cpu_start);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        (encoded, cpu)
    }

    /// The reader's draw over rectangle `at`, its pipeline state set and the target bound at
    /// fragment texture 0.
    fn read(&self, encoder: &mut RenderCommandEncoder, at: usize) {
        encoder.set_render_pipeline_state(&self.reader).unwrap();
        encoder.set_fragment_texture(Some(&self.target), 0).unwrap();
        encoder.set_vertex_bytes(&rectangle(at), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
}

/// The corners of rectangle `at`, in normalized device coordinates: one of a 10 by 4 grid of
/// rectangles of 64 by 64 pixels over the target's top-left.
fn rectangle(at: usize) -> Vec<u8> {
    let (x, y) = ((at % 10) as f32 * 64.0, (at / 10 % 4) as f32 * 64.0);
    let ndc_x = |p: f32| p / TARGET_WIDTH as f32 * 2.0 - 1.0;
    let ndc_y = |p: f32| 1.0 - p / TARGET_HEIGHT as f32 * 2.0;
    [ndc_x(x), ndc_y(y + 64.0), ndc_x(x + 64.0), ndc_y(y)]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// A work count a `READER-COUNTS` line prints, and its name.
type Count = (&'static str, fn(&WorkCounts) -> u64);

/// The work counts a `READER-COUNTS` line prints, by name.
const COUNTS: [Count; 13] = [
    ("reading_draws_snapshotted", |c| c.reading_draws_snapshotted),
    ("snapshot_inline_splits", |c| c.snapshot_inline_splits),
    ("coverage_queries", |c| c.coverage_queries),
    ("draws_as_before", |c| c.draws_as_before),
    ("draws_rebound", |c| c.draws_rebound),
    ("draws_rebound_carried", |c| c.draws_rebound_carried),
    ("kept_draws_carried", |c| c.kept_draws_carried),
    ("draws_rebind_stopped", |c| c.draws_rebind_stopped),
    ("draws_carry_stopped", |c| c.draws_carry_stopped),
    ("draws_none_kept", |c| c.draws_none_kept),
    ("draws_unkept", |c| c.draws_unkept),
    ("encoders_none_kept_next_fits", |c| {
        c.encoders_none_kept_next_fits
    }),
    ("inline_passes", |c| c.inline_passes),
];

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn reader_costs() {
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
    eprintln!(
        "READER-COST {name}: {PASSES} encoders a frame of {DRAWS} draws each (a reader, {PLAIN} \
         plain draws, a reader), target {TARGET_WIDTH}x{TARGET_HEIGHT}, {WARM} untimed and \
         {TIMED} timed frames"
    );
    for _ in 0..WARM {
        work.run();
    }
    let before = work.device.work_counts();
    let runs: Vec<(Duration, Duration)> = (0..TIMED).map(|_| work.run()).collect();
    let after = work.device.work_counts();
    let micros = |d: Duration| d.as_secs_f64() * 1e6;
    let [_, wall, _] = percentiles(runs.iter().map(|run| micros(run.0)).collect());
    let [low, cpu, high] = percentiles(runs.iter().map(|run| micros(run.1)).collect());
    let draws = (PASSES * DRAWS) as f64;
    eprintln!(
        "READER-COST {name}: a frame {cpu:.1} us of the thread's CPU (p10 {low:.1}, p90 \
         {high:.1}), {wall:.1} us wall; a draw {:.2} us CPU, {:.2} us wall",
        cpu / draws,
        wall / draws
    );
    let counts: Vec<String> = COUNTS
        .iter()
        .map(|(name, count)| {
            let moved = (count(&after) - count(&before)) as f64 / TIMED as f64;
            format!("{name} {moved:.2}")
        })
        .collect();
    eprintln!("READER-COUNTS {name}: a frame: {}", counts.join(", "));
}
