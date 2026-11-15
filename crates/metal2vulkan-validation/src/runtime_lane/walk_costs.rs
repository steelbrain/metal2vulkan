//! What encoding render encoders shaped as the consumer's guest Launchpad walk costs the thread
//! that encodes them, on the device the cfg selects, without the validation layer: report only, in
//! release builds, to weigh the every-step, kept, rebound and carried draw paths
//! (`render/draw_again.rs`, `render/carried.rs`) by the mix of draws the walk takes.
//!
//! The mix follows the walk's work counts a present (the consumer's XTX walk at pin 75): three
//! render encoders a command buffer of about 4.5 draws each, nearly every encoder holding one draw
//! that reads its own color attachment through a snapshot (`render/snapshot.rs`) after a memory
//! barrier over the render targets from the fragment stage to the fragment stage, every draw
//! indexed (`UInt16`) after its scissor rectangle and its buffers' offsets moved, the encoder's
//! work behind a fence wait and ahead of a fence update. A frame is one command buffer of
//! [`ENCODERS`] render encoders into one `BGRA8Unorm` target of
//! [`TARGET_WIDTH`]x[`TARGET_HEIGHT`] (private, or over host pages: below), each loading and
//! storing it, each:
//!
//! - `wait_for_fence` before the fragment stage, the sampler at fragment index 0, then the draws:
//! - **A** (`tob_sample`, opaque): fragment texture 0 one of [`TEXTURES`] textures;
//! - **B** (`tob_solid`, blended): fragment buffer 0 at a slot of its own;
//! - with `barriers`, `memory_barrier_with_scope` over the render targets, fragment to fragment;
//! - **R** (`tob_read`, the reader): fragment texture 0 the target, read at each fragment's pixel;
//! - **A** and **B** again;
//! - `update_fence` after the fragment stage and `end_encoding`.
//!
//! Each draw sets its pipeline state where the last draw's was another, its scissor rectangle
//! (the next of a grid over the target) and vertex buffer 0's offset (its rectangle's corners, at
//! a slot of its own), then draws 6 indices at an index buffer offset of its own; vertex buffer 0
//! and fragment buffer 0 lie in a no-copy buffer over host pages, as a guest's do. Each frame is
//! committed and waited for.
//!
//! Four shapes: `barriers` as above, and `no barrier`, the same calls without the barrier, so the
//! two frames' difference is what the barrier's effect on the draws after it costs; and each of
//! the two `over pages`, the target a texture over a no-copy buffer over host pages, as the
//! guest's surface is ([`Target::OverPages`]), made once the device's in-place checks answered
//! (`Device::settle_in_place_checks`), as a consumer's surface is: one made before takes a shadow
//! for good (`in_place.rs`). The device takes a consumer's policy, the default
//! (`DeviceOptions::policy`), under which such a texture renders in place where the check passed
//! for it; the validation lanes' exact policy makes every one a shadow
//! (`HostPageTexturePolicy::Shadow`). Over a buffer's bytes R reads through a snapshot
//! whatever it covers (`render/snapshot.rs`'s `over_bytes`), as the consumer's readers mostly do
//! (about ten snapshots to one split by coverage over its Iris Xe walks); into the private
//! target its learned count takes the split, which copies only the scissor's texels
//! (`reading_draws_split_coverage`). Each runs [`WARM`] frames untimed, then [`TIMED`] frames,
//! printed as a `WALK-COST` line (the medians of the encoding thread's CPU time a frame, from the
//! first encoder's creation through the last one's `end_encoding`, and a draw), a `WALK-COUNTS`
//! line (the listed work counts the timed frames moved, a frame) and a `WALK-MOVED` line (every
//! work count the timed frames moved, a frame, by name: each reason a draw took every step, was
//! let go or stopped its rebind among them). `walk_costs_profiled`
//! encodes [`PROFILED`] frames of `barriers` alone, for a profiler,
//! `walk_costs_profiled_no_barrier` as many of `no barrier`, and `walk_costs_profiled_over_pages`
//! as many of `barriers over pages`. A device that imports no host memory runs no `over pages`
//! shape, and says so.
//!
//! `walk_costs_frame` encodes the same encoders [`FRAME_ENCODERS`] to a command buffer, as many as
//! a compositor's desktop frame of the consumer's guest walks holds in one command buffer (about
//! 107 render encoders over its 3090 walk at pin 92), and commits it with `commit_and_submit`, as
//! the consumer's encoding thread does: printed per shape as a `WALK-FRAME` line, the medians of
//! the encoding thread's CPU time a frame through the last `end_encoding`, a draw and an encoder,
//! and of its `commit_and_submit`, with the frame's `WALK-MOVED` counts. `walk_costs_frame_profiled`
//! encodes [`FRAME_PROFILED`] such frames of `barriers over pages`, for a profiler.
//!
//! `walk_reader_forms` runs `barriers` with each [`Reader`] as draw R: the snapshot reader above;
//! one whose read texel decides a branch (`wlk_read_branch` of `wlk_readers.metal`), a slot no
//! snapshot serves (`SnapshotRefusal::Branch`), read through a split and a copy; and the same
//! arithmetic over the color it fetches (`wlk_fetch_branch`, `[[color(0)]]`), read in place
//! inside the fragment shader's pixel interlock where the device fetches so. Printed per form
//! (`WALK-READER`): the medians of the encoding thread's CPU time a frame and of the frame's GPU
//! time (its command buffer's GPU end less its start), and the work counts a frame.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::draw_costs::{percentiles, thread_cpu};
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{host_page, no_copy, not_importable, Mapping};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, IndexType, PixelFormat, PrimitiveType, RenderStages, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions, Fence, Library, LoadAction,
    PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, ScissorRect, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::time::Duration;

/// The untimed and the timed frames, and the frames a profiling run encodes.
const WARM: usize = 20;
const TIMED: usize = 200;
const PROFILED: usize = 4000;
/// The render encoders of a frame.
const ENCODERS: usize = 3;
/// The render encoders of a compositor's frame (`walk_costs_frame`), its timed frames, and the
/// frames its profiling run encodes.
const FRAME_ENCODERS: usize = 107;
const FRAME_TIMED: usize = 60;
const FRAME_PROFILED: usize = 400;
/// The draws of an encoder, in order.
const DRAWS: [Kind; 5] = [Kind::A, Kind::B, Kind::Reader, Kind::A, Kind::B];
/// The textures draw A takes in turn, and their side.
const TEXTURES: usize = 4;
const TEXTURE_SIDE: u64 = 64;
/// The target's size.
const TARGET_WIDTH: u64 = 1920;
const TARGET_HEIGHT: u64 = 1080;
/// The slots of the chunk over host pages, each a draw's rectangle and constants, and their size.
const SLOTS: usize = 64;
const SLOT: usize = 256;

/// A draw's pipeline state.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    A,
    B,
    Reader,
}

/// What draw R reads the target with.
#[derive(Clone, Copy, PartialEq)]
enum Reader {
    /// `tob_read`: the target at fragment texture 0, read at the fragment's pixel.
    Snapshot,
    /// `wlk_read_branch`: the target at fragment texture 0, its texel deciding a loop's trip count.
    Branch,
    /// `wlk_fetch_branch`: the same arithmetic over the color the fragment fetches.
    Fetch,
}

/// What the frame renders into, and draw R reads.
#[derive(Clone, Copy, PartialEq)]
enum Target {
    /// A private texture.
    Private,
    /// A texture over a no-copy buffer over host pages, its rows packed.
    OverPages,
}

/// One frame's costs: the encoding thread's CPU time through the last `end_encoding`, that of
/// its `commit_and_submit` where it submits so (else zero), and the GPU time in seconds.
struct Frame {
    cpu: Duration,
    commit: Duration,
    gpu: f64,
}

/// The device, its queue, and what a frame draws with.
struct Workload {
    device: Device,
    /// The device's in-place checks answered before the targets were made.
    settled: bool,
    queue: CommandQueue,
    a: RenderPipelineState,
    b: RenderPipelineState,
    reader: RenderPipelineState,
    branch: RenderPipelineState,
    fetch: RenderPipelineState,
    sampler: SamplerState,
    textures: Vec<Texture>,
    target: Texture,
    /// The target over host pages and its buffer, where the device imports host memory.
    over_pages: Option<(Texture, Buffer)>,
    indices: Buffer,
    fence: Fence,
    chunk: Buffer,
    /// The host pages under `chunk` and under `over_pages`, which outlive them (dropped after
    /// them).
    _pages: Mapping,
    _target_pages: Mapping,
}

impl Workload {
    /// The workload on `device`, whose in-place checks answered first where `settled`.
    fn new(device: Device, settled: bool) -> Workload {
        let library = |name: &str| {
            let ll = engine_dir().join(format!("{name}.ll"));
            device
                .new_library_with_air(&llvm_assemble(&ll).unwrap())
                .unwrap()
        };
        let (renders, readers) = (library("tob_renders"), library("wlk_readers"));
        let function = |library: &Library, name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |fragment: &str, blended: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function(&renders, "tob_rect"));
            let library = if fragment.starts_with("wlk_") {
                &readers
            } else {
                &renders
            };
            descriptor.fragment = Some(function(library, fragment));
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
            if blended {
                color.blending_enabled = true;
                color.source_rgb_blend_factor = BlendFactor::One;
                color.source_alpha_blend_factor = BlendFactor::One;
                color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            }
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
        // The target over host pages: its texels' rows packed, the guest's surface's.
        let row = TARGET_WIDTH as usize * 4;
        let target_len = (row * TARGET_HEIGHT as usize).next_multiple_of(host_page());
        let target_pages = Mapping::anonymous(target_len);
        // SAFETY: the pages are this process's, mapped read and write, and outlive the buffer
        // (`Workload`'s order).
        let target_chunk = unsafe {
            no_copy(
                &device,
                target_pages.pointer(),
                target_len,
                ResourceOptions::shared(),
                None,
            )
        };
        let over_pages = match target_chunk {
            Ok(buffer) => {
                let texture = (buffer.new_texture(&target, 0, row as u64)).unwrap();
                Some((texture, buffer))
            }
            Err(error) if not_importable(&error) => None,
            Err(error) => panic!("the target's no-copy buffer: {error:?}"),
        };
        target.resource_options = ResourceOptions::private();
        // Each draw's 6 indices `0..6` at an offset of its own.
        let indices: Vec<u8> = (0..SLOTS)
            .flat_map(|_| 0..6u16)
            .flat_map(u16::to_le_bytes)
            .collect();
        let indices = device
            .new_buffer_with_bytes(&indices, ResourceOptions::shared())
            .unwrap();
        // Each slot: the corners of rectangle `slot` (vertex buffer 0), then a tint (fragment
        // buffer 0) 128 bytes on.
        let len = (SLOTS * SLOT).next_multiple_of(host_page());
        let pages = Mapping::anonymous(len);
        for slot in 0..SLOTS {
            pages.write(slot * SLOT, &rectangle(slot));
            let tint = [0.25f32, 0.5, slot as f32 / SLOTS as f32, 0.5];
            let tint: Vec<u8> = tint.iter().flat_map(|v| v.to_le_bytes()).collect();
            pages.write(slot * SLOT + 128, &tint);
        }
        // SAFETY: the pages are this process's, mapped read and write, and outlive the buffer
        // (`Workload`'s order).
        let chunk = unsafe {
            no_copy(
                &device,
                pages.pointer(),
                len,
                ResourceOptions::shared(),
                None,
            )
        }
        .unwrap();
        Workload {
            settled,
            queue: device.new_command_queue().unwrap(),
            a: pipeline("tob_sample", false),
            b: pipeline("tob_solid", true),
            reader: pipeline("tob_read", false),
            branch: pipeline("wlk_read_branch", false),
            fetch: pipeline("wlk_fetch_branch", false),
            sampler,
            textures: (0..TEXTURES).map(texture).collect(),
            target: device.new_texture(&target).unwrap(),
            over_pages,
            indices,
            fence: device.new_fence(),
            chunk,
            _pages: pages,
            _target_pages: target_pages,
            device,
        }
    }

    /// The texture `target` names: `None` for [`Target::OverPages`] where the device imports no
    /// host memory.
    fn target(&self, target: Target) -> Option<&Texture> {
        match target {
            Target::Private => Some(&self.target),
            Target::OverPages => self.over_pages.as_ref().map(|(texture, _)| texture),
        }
    }

    /// Encode one frame (the module docs) into `target`, with the barrier before each reader
    /// where `barriers` and `reader` as draw R, commit it and wait for it: the encoding thread's
    /// CPU time its encoding took and its GPU time.
    fn run(&self, frame: (bool, Target), reader: Reader) -> Frame {
        self.encode(frame, reader, (ENCODERS, false))
    }

    /// [`run`](Self::run) of `encoders` encoders, committed with `commit_and_submit` where
    /// `submits`, else with `commit`.
    fn encode(
        &self,
        (barriers, target): (bool, Target),
        reader: Reader,
        (encoders, submits): (usize, bool),
    ) -> Frame {
        let target = self.target(target).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let cpu_start = thread_cpu();
        for at in 0..encoders {
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder
                .wait_for_fence(&self.fence, RenderStages::FRAGMENT)
                .unwrap();
            encoder
                .set_fragment_sampler_state(Some(&self.sampler), 0)
                .unwrap();
            encoder.set_vertex_buffer(Some(&self.chunk), 0, 0).unwrap();
            encoder
                .set_fragment_buffer(Some(&self.chunk), 128, 0)
                .unwrap();
            let mut last = None;
            for (n, &kind) in DRAWS.iter().enumerate() {
                let slot = (at * DRAWS.len() + n) % SLOTS;
                if kind == Kind::Reader && barriers {
                    encoder.memory_barrier_with_scope(
                        BarrierScope::RENDER_TARGETS,
                        RenderStages::FRAGMENT,
                        RenderStages::FRAGMENT,
                    );
                }
                self.draw(
                    &mut encoder,
                    (kind, reader),
                    target,
                    last != Some(kind),
                    slot,
                );
                last = Some(kind);
            }
            encoder
                .update_fence(&self.fence, RenderStages::FRAGMENT)
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        let cpu = thread_cpu().saturating_sub(cpu_start);
        let commit = match submits {
            true => {
                let commit_start = thread_cpu();
                command_buffer.commit_and_submit().unwrap();
                thread_cpu().saturating_sub(commit_start)
            }
            false => {
                command_buffer.commit().unwrap();
                Duration::ZERO
            }
        };
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        Frame { cpu, commit, gpu }
    }

    /// A draw of `kind` over rectangle `slot`, draw R's state `reader`'s reading `target`, its
    /// pipeline state set where `set`.
    fn draw(
        &self,
        encoder: &mut RenderCommandEncoder,
        (kind, reader): (Kind, Reader),
        target: &Texture,
        set: bool,
        slot: usize,
    ) {
        let state = match (kind, reader) {
            (Kind::A, _) => &self.a,
            (Kind::B, _) => &self.b,
            (Kind::Reader, Reader::Snapshot) => &self.reader,
            (Kind::Reader, Reader::Branch) => &self.branch,
            (Kind::Reader, Reader::Fetch) => &self.fetch,
        };
        if set {
            encoder.set_render_pipeline_state(state).unwrap();
        }
        match kind {
            Kind::A => {
                let texture = &self.textures[slot % TEXTURES];
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
            }
            Kind::B => encoder
                .set_fragment_buffer_offset((slot * SLOT + 128) as u64, 0)
                .unwrap(),
            Kind::Reader if reader == Reader::Fetch => {}
            Kind::Reader => encoder.set_fragment_texture(Some(target), 0).unwrap(),
        }
        encoder
            .set_vertex_buffer_offset((slot * SLOT) as u64, 0)
            .unwrap();
        let (x, y) = cell(slot);
        let rect = ScissorRect {
            x,
            y,
            width: 64,
            height: 64,
        };
        encoder.set_scissor_rect(rect).unwrap();
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                6,
                IndexType::UInt16,
                &self.indices,
                (slot * 12) as u64,
                1,
                0,
                0,
            )
            .unwrap();
    }
}

/// The top-left pixel of rectangle `slot`: one of a 16 by 4 grid of 64 by 64 pixels over the
/// target's top-left.
fn cell(slot: usize) -> (u64, u64) {
    ((slot % 16) as u64 * 64, (slot / 16 % 4) as u64 * 64)
}

/// The corners of rectangle `slot` ([`cell`]), in normalized device coordinates.
fn rectangle(slot: usize) -> Vec<u8> {
    let (x, y) = cell(slot);
    let (x, y) = (x as f32, y as f32);
    let ndc_x = |p: f32| p / TARGET_WIDTH as f32 * 2.0 - 1.0;
    let ndc_y = |p: f32| 1.0 - p / TARGET_HEIGHT as f32 * 2.0;
    [ndc_x(x), ndc_y(y + 64.0), ndc_x(x + 64.0), ndc_y(y)]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// A work count a `WALK-COUNTS` line prints, and its name.
type Count = (&'static str, fn(&WorkCounts) -> u64);

/// The work counts a `WALK-COUNTS` line prints, by name.
const COUNTS: [Count; 24] = [
    ("draws_none_kept", |c| c.draws_none_kept),
    ("draws_as_before", |c| c.draws_as_before),
    ("draws_rebound", |c| c.draws_rebound),
    ("draws_rebound_carried", |c| c.draws_rebound_carried),
    ("draws_moved", |c| c.draws_moved),
    ("draws_restated", |c| c.draws_restated),
    ("draws_rebind_stopped", |c| c.draws_rebind_stopped),
    ("draws_carry_stopped", |c| c.draws_carry_stopped),
    ("draws_unkept", |c| c.draws_unkept),
    ("kept_draws_carried", |c| c.kept_draws_carried),
    ("kept_draws_carried_aside", |c| c.kept_draws_carried_aside),
    ("kept_draws_let_go_barrier", |c| c.kept_draws_let_go_barrier),
    ("kept_draws_kept_across_fence", |c| {
        c.kept_draws_kept_across_fence
    }),
    ("reading_draws_snapshotted", |c| c.reading_draws_snapshotted),
    ("reading_draws_snapshotted_again", |c| {
        c.reading_draws_snapshotted_again
    }),
    ("inline_passes", |c| c.inline_passes),
    ("attachment_read_splits", |c| c.attachment_read_splits),
    ("attachment_reads_through_copies", |c| {
        c.attachment_reads_through_copies
    }),
    ("reading_draws_split_area", |c| c.reading_draws_split_area),
    ("reading_draws_split_coverage", |c| {
        c.reading_draws_split_coverage
    }),
    ("reading_draws_split_stand_in", |c| {
        c.reading_draws_split_stand_in
    }),
    ("reading_draws_split_not_in_place", |c| {
        c.reading_draws_split_not_in_place
    }),
    ("reading_draws_split_device_lacks", |c| {
        c.reading_draws_split_device_lacks
    }),
    ("reading_draws_split_unreadable_own_pixel", |c| {
        c.reading_draws_split_unreadable_own_pixel
    }),
];

/// The workload on the cfg's device, as a consumer opens it.
fn workload(cache: &ScratchCache, messages: &Messages) -> Workload {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record, and takes the default policy.
        options.record_executions = false;
        options.policy = DeviceOptions::default().policy;
    })
    .unwrap();
    // A consumer's surface over host pages is made once the device's in-place checks answered
    // (`in_place.rs`); one made before takes a shadow for good.
    let settled = device.settle_in_place_checks(CHECK_WAIT);
    Workload::new(device, settled)
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn walk_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let name = work.device.name().to_owned();
    let draws = (ENCODERS * DRAWS.len()) as f64;
    eprintln!(
        "WALK-COST {name}: {ENCODERS} encoders a frame of {} draws each (A, B, a reader, A, B), \
         target {TARGET_WIDTH}x{TARGET_HEIGHT}, {WARM} untimed and {TIMED} timed frames",
        DRAWS.len()
    );
    let shapes = [
        ("barriers", (true, Target::Private)),
        ("no barrier", (false, Target::Private)),
        ("barriers over pages", (true, Target::OverPages)),
        ("no barrier over pages", (false, Target::OverPages)),
    ];
    for (shape, frame) in shapes {
        if work.target(frame.1).is_none() {
            eprintln!("WALK-COST {name} {shape}: the device imports no host memory");
            continue;
        }
        for _ in 0..WARM {
            work.run(frame, Reader::Snapshot);
        }
        let before = work.device.work_counts();
        let runs: Vec<f64> = (0..TIMED)
            .map(|_| work.run(frame, Reader::Snapshot).cpu.as_secs_f64() * 1e6)
            .collect();
        let after = work.device.work_counts();
        let [low, cpu, high] = percentiles(runs);
        eprintln!(
            "WALK-COST {name} {shape}: a frame {cpu:.1} us of the thread's CPU (p10 {low:.1}, p90 \
             {high:.1}); a draw {:.2} us",
            cpu / draws
        );
        let counts: Vec<String> = COUNTS
            .iter()
            .map(|(count_name, count)| {
                let moved = (count(&after) - count(&before)) as f64 / TIMED as f64;
                format!("{count_name} {moved:.2}")
            })
            .collect();
        eprintln!("WALK-COUNTS {name} {shape}: a frame: {}", counts.join(", "));
        let moved = super::video_frames::moved_per_frame(&before, &after, TIMED);
        eprintln!("WALK-MOVED {name} {shape}: a frame: {moved}");
    }
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn walk_costs_profiled() {
    profiled((true, Target::Private));
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn walk_costs_profiled_no_barrier() {
    profiled((false, Target::Private));
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn walk_costs_profiled_over_pages() {
    profiled((true, Target::OverPages));
}

/// [`PROFILED`] frames of the shape `frame` names (barriers or none, and the target), after
/// [`WARM`] untimed ones.
fn profiled(frame: (bool, Target)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let (barriers, over_pages) = (frame.0, frame.1 == Target::OverPages);
    if work.target(frame.1).is_none() {
        eprintln!(
            "WALK-PROFILED {}: the device imports no host memory",
            work.device.name()
        );
        return;
    }
    for _ in 0..WARM {
        work.run(frame, Reader::Snapshot);
    }
    let cpu: Duration = (0..PROFILED)
        .map(|_| work.run(frame, Reader::Snapshot).cpu)
        .sum();
    let draws = (PROFILED * ENCODERS * DRAWS.len()) as f64;
    eprintln!(
        "WALK-PROFILED {} (barriers {barriers}, over pages {over_pages}): {PROFILED} frames, \
         {:.2} us of the thread's CPU a draw",
        work.device.name(),
        cpu.as_secs_f64() * 1e6 / draws
    );
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn walk_costs_frame() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let name = work.device.name().to_owned();
    let draws = (FRAME_ENCODERS * DRAWS.len()) as f64;
    eprintln!(
        "WALK-FRAME {name}: {FRAME_ENCODERS} encoders a command buffer of {} draws each, \
         committed with commit_and_submit, {WARM} untimed and {FRAME_TIMED} timed frames; \
         in-place checks settled {}, textures in place over host pages {}",
        DRAWS.len(),
        work.settled,
        work.device.work_counts().host_textures_in_place,
    );
    let shapes = [
        ("barriers over pages", (true, Target::OverPages)),
        ("no barrier over pages", (false, Target::OverPages)),
        ("barriers", (true, Target::Private)),
    ];
    let encode = (FRAME_ENCODERS, true);
    for (shape, frame) in shapes {
        if work.target(frame.1).is_none() {
            eprintln!("WALK-FRAME {name} {shape}: the device imports no host memory");
            continue;
        }
        for _ in 0..WARM {
            work.encode(frame, Reader::Snapshot, encode);
        }
        let before = work.device.work_counts();
        let frames: Vec<Frame> = (0..FRAME_TIMED)
            .map(|_| work.encode(frame, Reader::Snapshot, encode))
            .collect();
        let after = work.device.work_counts();
        let [low, cpu, high] =
            percentiles(frames.iter().map(|f| f.cpu.as_secs_f64() * 1e6).collect());
        let [_, commit, _] = percentiles(
            frames
                .iter()
                .map(|f| f.commit.as_secs_f64() * 1e6)
                .collect(),
        );
        eprintln!(
            "WALK-FRAME {name} {shape}: a frame {cpu:.1} us of the thread's CPU through its last \
             end_encoding (p10 {low:.1}, p90 {high:.1}), a draw {:.2} us, an encoder {:.2} us; \
             commit_and_submit {commit:.1} us",
            cpu / draws,
            cpu / FRAME_ENCODERS as f64
        );
        let moved = super::video_frames::moved_per_frame(&before, &after, FRAME_TIMED);
        eprintln!("WALK-MOVED {name} frame {shape}: a frame: {moved}");
    }
}

#[test]
#[ignore = "a profiling run, not a check: run by name with --run-ignored all"]
fn walk_costs_frame_profiled() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let frame = (true, Target::OverPages);
    if work.target(frame.1).is_none() {
        eprintln!(
            "WALK-FRAME-PROFILED {}: the device imports no host memory",
            work.device.name()
        );
        return;
    }
    let encode = (FRAME_ENCODERS, true);
    for _ in 0..WARM {
        work.encode(frame, Reader::Snapshot, encode);
    }
    let frames: Vec<Frame> = (0..FRAME_PROFILED)
        .map(|_| work.encode(frame, Reader::Snapshot, encode))
        .collect();
    let cpu: Duration = frames.iter().map(|f| f.cpu + f.commit).sum();
    eprintln!(
        "WALK-FRAME-PROFILED {} (barriers over pages): {FRAME_PROFILED} frames of \
         {FRAME_ENCODERS} encoders, {:.1} us of the thread's CPU a frame with its commit",
        work.device.name(),
        cpu.as_secs_f64() * 1e6 / FRAME_PROFILED as f64
    );
}

/// The work counts a `WALK-READER` line prints, by name.
const READER_COUNTS: [Count; 12] = [
    ("render_passes_begun_instances", |c| {
        c.render_passes_begun_instances
    }),
    ("instance_splits_barrier", |c| c.instance_splits_barrier),
    ("instance_splits_attached_read", |c| {
        c.instance_splits_attached_read
    }),
    ("instance_splits_feedback", |c| c.instance_splits_feedback),
    ("instance_splits_shape", |c| c.instance_splits_shape),
    ("reading_draws_snapshotted", |c| c.reading_draws_snapshotted),
    ("reading_draws_split_unreadable_own_pixel_branch", |c| {
        c.reading_draws_split_unreadable_own_pixel_branch
    }),
    ("attached_read_slots", |c| c.attached_read_slots),
    ("storage_forms_taken", |c| c.storage_forms_taken),
    ("fetch_draw_barriers", |c| c.fetch_draw_barriers),
    ("draws_none_kept", |c| c.draws_none_kept),
    ("draws_rebound", |c| c.draws_rebound),
];

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn walk_reader_forms() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let work = workload(&cache, &messages);
    let name = work.device.name().to_owned();
    eprintln!(
        "WALK-READER {name}: {ENCODERS} encoders a frame of {} draws each (A, B, a barrier, a \
         reader, A, B), target {TARGET_WIDTH}x{TARGET_HEIGHT}, {WARM} untimed and {TIMED} timed \
         frames",
        DRAWS.len()
    );
    let forms = [
        ("snapshot", Reader::Snapshot),
        ("branch, split and copy", Reader::Branch),
        ("branch, fetched", Reader::Fetch),
    ];
    for (form, reader) in forms {
        let frame = (true, Target::Private);
        for _ in 0..WARM {
            work.run(frame, reader);
        }
        let before = work.device.work_counts();
        let frames: Vec<Frame> = (0..TIMED).map(|_| work.run(frame, reader)).collect();
        let after = work.device.work_counts();
        let [cpu_low, cpu, cpu_high] =
            percentiles(frames.iter().map(|f| f.cpu.as_secs_f64() * 1e6).collect());
        let [gpu_low, gpu, gpu_high] = percentiles(frames.iter().map(|f| f.gpu * 1e6).collect());
        let counts: Vec<String> = READER_COUNTS
            .iter()
            .map(|(count_name, count)| {
                let moved = (count(&after) - count(&before)) as f64 / TIMED as f64;
                format!("{count_name} {moved:.2}")
            })
            .collect();
        eprintln!(
            "WALK-READER {name} {form}: a frame {cpu:.1} us of the thread's CPU (p10 {cpu_low:.1}, \
             p90 {cpu_high:.1}), {gpu:.1} us of GPU (p10 {gpu_low:.1}, p90 {gpu_high:.1}); a \
             frame: {}",
            counts.join(", ")
        );
    }
}
