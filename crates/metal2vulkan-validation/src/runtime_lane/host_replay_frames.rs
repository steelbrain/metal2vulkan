//! What frames that execute render commands a kernel may write cost on the device the cfg
//! selects, and what they cost another queue that presents meanwhile, without the validation
//! layer: report only, in release builds, to weigh the host step before and after a change to it.
//!
//! The workload is a browser's render bundle under a compositor: [`live_states`] render pipeline
//! states made with `supportIndirectCommandBuffers`, each with a fragment function of its own (a
//! constant color, or [`fragment_ops`] dependent operations on the position), and a buffer of [`COMMANDS`] draws of one triangle each (`rt6_resolve_half`)
//! that sets the first [`NAMED_STATES`] of them in turn, inheriting nothing. Each frame, on one
//! queue, a compute encoder makes the buffer resident for writing (`use_resource` with `WRITE`)
//! and dispatches a kernel that writes no command (`a4_index_word`, over buffers of its own), and
//! a render pass into a 256x256 `BGRA8Unorm` target executes every command; the frame is
//! committed and waited for. Meanwhile another queue presents: every [`PRESENT_NANOS`] a thread
//! commits a clear of a target of its own and waits for it.
//!
//! Printed as `HOST-REPLAY` lines: each of the first [`COLD`] frames, then the median, the 90th
//! percentile and the largest of the [`FRAMES`] after them, of the frame's time (encoding to
//! completion), its `commit`, and its time from the first submission's kernel start to the last
//! one's kernel end (its gate, its host step and its GPU work); the presents' times from commit to
//! completion over the same frames, and the presents missed (ticks that came while one was still
//! out); and the device's counters over them.
//!
//! [`STATES_VAR`], [`OPS_VAR`] and [`DROP_VAR`] change the workload: more states, fragment
//! functions that take the driver longer to compile, and the program's release of every state
//! but the named ones a while into the first frame.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The draws the buffer holds.
const COMMANDS: u64 = 5000;
/// The variable that sets [`live_states`].
const STATES_VAR: &str = "HOST_REPLAY_STATES";
/// The variable that sets [`fragment_ops`].
const OPS_VAR: &str = "HOST_REPLAY_OPS";
/// The variable that sets [`drop_after`].
const DROP_VAR: &str = "HOST_REPLAY_DROP_MS";

/// A count the variable `name` sets, else `default`.
fn knob(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |value| value.parse().expect(name))
}

/// The live states made with `supportIndirectCommandBuffers`: 256 unless [`STATES_VAR`] says.
fn live_states() -> usize {
    knob(STATES_VAR, 256) as usize
}

/// The dependent operations of each state's fragment function: none (a constant color) unless
/// [`OPS_VAR`] says.
fn fragment_ops() -> usize {
    knob(OPS_VAR, 0) as usize
}

/// How long after the first frame begins the program releases every state but the named ones,
/// where [`DROP_VAR`] says.
fn drop_after() -> Option<Duration> {
    std::env::var(DROP_VAR)
        .ok()
        .map(|ms| Duration::from_millis(ms.parse().expect(DROP_VAR)))
}
/// The states the CPU sets on the commands, in turn.
const NAMED_STATES: usize = 4;
/// The frames printed one by one, then those measured together.
const COLD: usize = 4;
const FRAMES: usize = 60;
/// The presenting queue's period: 120 Hz.
const PRESENT_NANOS: u64 = 8_333_333;
/// The targets' side.
const SIDE: u64 = 256;

/// A fragment function of its own for state `k`: a constant color whose red is `k / 1024`, or,
/// with `ops` operations, a chain of that many dependent multiplies and adds on the position.
fn fragment(k: usize, ops: usize) -> String {
    let red = (k as f64 / 1024.0).to_bits();
    if ops > 0 {
        let mut body = format!(
            "  %x = extractelement <4 x float> %pos, i64 0\n  \
             %y = extractelement <4 x float> %pos, i64 1\n  \
             %a0 = fadd float %x, 0x{red:016X}\n"
        );
        for i in 1..=ops {
            let step = ((i as f64) / 4096.0 + 1.0).to_bits();
            let (op, with) = match i % 2 {
                0 => ("fmul", "%y".to_string()),
                _ => ("fadd", format!("0x{step:016X}")),
            };
            body.push_str(&format!("  %a{i} = {op} float %a{}, {with}\n", i - 1));
        }
        return format!(
            "source_filename = \"host_replay_frames\"\n\
             define <4 x float> @host_replay_fragment(<4 x float> %pos) {{\n{body}  \
             %c = insertelement <4 x float> <float 0.0, float 5.000000e-01, float 7.500000e-01, \
             float 1.000000e+00>, float %a{ops}, i64 0\n  \
             ret <4 x float> %c\n\
             }}\n\
             !air.fragment = !{{!0}}\n\
             !0 = !{{ptr @host_replay_fragment, !1, !3}}\n\
             !1 = !{{!2}}\n\
             !2 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", \
             !\"air.arg_name\", !\"color\"}}\n\
             !3 = !{{!4}}\n\
             !4 = !{{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", \
             !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"pos\"}}\n"
        );
    }
    format!(
        "source_filename = \"host_replay_frames\"\n\
         define <4 x float> @host_replay_fragment() {{\n  \
           ret <4 x float> <float 0x{red:016X}, float 5.000000e-01, float 7.500000e-01, \
         float 1.000000e+00>\n\
         }}\n\
         !air.fragment = !{{!0}}\n\
         !0 = !{{ptr @host_replay_fragment, !1, !3}}\n\
         !1 = !{{!2}}\n\
         !2 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", \
         !\"air.arg_name\", !\"color\"}}\n\
         !3 = !{{}}\n"
    )
}

/// A frame's times.
struct Frame {
    total: Duration,
    commit: Duration,
    /// From the first submission's kernel start to the last's kernel end.
    held: f64,
}

/// The device, its queues, the states and the buffer.
struct Workload {
    device: Device,
    queue: CommandQueue,
    /// Held under a lock, so the program may release some ([`drop_after`]).
    states: Mutex<Vec<RenderPipelineState>>,
    writer: ComputePipelineState,
    words: (Buffer, Buffer),
    target: Texture,
    icb: IndirectCommandBuffer,
}

fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A pass that clears `target` and stores it.
fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    pass
}

impl Workload {
    fn new(device: Device) -> Workload {
        let resolves = device
            .new_library_with_air(&llvm_assemble(&engine_dir().join("rt6_resolves.ll")).unwrap())
            .unwrap();
        let vertex = resolves.new_function_with_name("rt6_resolve_half").unwrap();
        let made = Instant::now();
        let (count, ops) = (live_states(), fragment_ops());
        let states: Vec<RenderPipelineState> = (0..count)
            .map(|k| {
                let library = device
                    .new_library_with_air(&llvm_assemble_text(&fragment(k, ops)).unwrap())
                    .unwrap();
                let function = library
                    .new_function_with_name("host_replay_fragment")
                    .unwrap();
                let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                    vertex.function_descriptor(),
                ));
                descriptor.fragment = Some(StageSource::Function(function.function_descriptor()));
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
                descriptor.support_indirect_command_buffers = true;
                device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .unwrap()
            })
            .collect();
        eprintln!(
            "HOST-REPLAY {}: {count} states of {ops} fragment operations made in {:.1} ms",
            device.name(),
            made.elapsed().as_secs_f64() * 1e3
        );
        let kernel = device
            .new_library_with_air(&llvm_assemble(&engine_dir().join("a4_index_word.ll")).unwrap())
            .unwrap()
            .new_function_with_name("a4_index_word")
            .unwrap();
        let writer = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    kernel.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        let icb = device
            .new_indirect_command_buffer(&descriptor, COMMANDS, ResourceOptions::shared())
            .unwrap();
        for index in 0..COMMANDS {
            let command = icb.indirect_render_command_at(index).unwrap();
            command
                .set_render_pipeline_state(&states[index as usize % NAMED_STATES])
                .unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        Workload {
            queue: device.new_command_queue().unwrap(),
            words: (
                buffer_with(&device, &[0; 64]).unwrap(),
                buffer_with(&device, &[0; 64]).unwrap(),
            ),
            target: target(&device),
            states: Mutex::new(states),
            writer,
            icb,
            device,
        }
    }

    /// One frame (the module docs), committed and waited for.
    fn frame(&self) -> Frame {
        let start = Instant::now();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&self.writer).unwrap();
        compute.set_buffer(Some(&self.words.0), 0, 0).unwrap();
        compute.set_buffer(Some(&self.words.1), 0, 1).unwrap();
        compute
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        compute
            .dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        compute.end_encoding().unwrap();
        let mut render = command_buffer
            .render_command_encoder(&pass(&self.target))
            .unwrap();
        render
            .execute_commands_in_buffer(&self.icb, 0..COMMANDS)
            .unwrap();
        render.end_encoding().unwrap();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        let commit = committed.elapsed();
        complete(&command_buffer);
        Frame {
            total: start.elapsed(),
            commit,
            held: command_buffer.kernel_end_time() - command_buffer.kernel_start_time(),
        }
    }
}

fn complete(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// What the presenting queue saw: each present's time from commit to completion, and the ticks
/// that came while one was still out.
#[derive(Default)]
struct Presents {
    times: Vec<f64>,
    missed: u64,
}

/// Present on `queue` into `target` every [`PRESENT_NANOS`] until `stop`, into `seen`.
fn present(queue: &CommandQueue, target: &Texture, stop: &AtomicBool, seen: &Mutex<Presents>) {
    let period = Duration::from_nanos(PRESENT_NANOS);
    let mut next = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        if now < next {
            std::thread::sleep(next - now);
        }
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer
            .render_command_encoder(&pass(target))
            .unwrap();
        encoder.end_encoding().unwrap();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        complete(&command_buffer);
        let took = committed.elapsed();
        let mut seen = seen.lock().unwrap();
        seen.times.push(took.as_secs_f64() * 1e3);
        next += period;
        while next < Instant::now() {
            seen.missed += 1;
            next += period;
        }
    }
}

/// `values` as "median / 90th percentile / largest".
fn spread(mut values: Vec<f64>) -> String {
    if values.is_empty() {
        return "none".into();
    }
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    format!(
        "{:.2} / {:.2} / {:.2}",
        at(50),
        at(90),
        values[values.len() - 1]
    )
}

/// The counters `after` minus `before`, printed.
fn counters(before: &WorkCounts, after: &WorkCounts) -> String {
    let delta = |f: fn(&WorkCounts) -> u64| f(after) - f(before);
    format!(
        "host steps {} taking {:.2} ms, making {} pipelines; heads held for host steps {}, at \
         their gate {}, behind earlier commits {}; variant pipelines made {}, render pipeline \
         create {:.1} ms, graphics pipelines looked up {}, draws settled {}, draws dropped still \
         making {}; records read {}",
        delta(|c| c.host_steps_recorded),
        delta(|c| c.host_step_nanos) as f64 / 1e6,
        delta(|c| c.host_step_pipelines_made),
        delta(|c| c.heads_held_for_host_steps),
        delta(|c| c.heads_waiting_at_their_gate),
        delta(|c| c.heads_held_for_earlier_commits),
        delta(|c| c.variant_pipelines_made),
        delta(|c| c.render_pipeline_create_nanos) as f64 / 1e6,
        delta(|c| c.graphics_pipelines_looked_up),
        delta(|c| c.draws_settled),
        delta(|c| c.draws_dropped_still_making),
        delta(|c| c.indirect_records_read),
    )
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn host_replay_frames() {
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
    let presenter = work.device.new_command_queue().unwrap();
    let drawable = target(&work.device);
    let stop = AtomicBool::new(false);
    let seen = Mutex::new(Presents::default());
    std::thread::scope(|scope| {
        scope.spawn(|| present(&presenter, &drawable, &stop, &seen));
        // Let the presenter settle into its period.
        std::thread::sleep(Duration::from_millis(200));
        let take = || std::mem::take(&mut *seen.lock().unwrap());
        take();
        eprintln!(
            "HOST-REPLAY {name}: {} live states, {COMMANDS} commands naming {NAMED_STATES}",
            work.states.lock().unwrap().len()
        );
        // The program's release of every state but the named ones, a while into the first frame.
        if let Some(after) = drop_after() {
            let (states, name) = (&work.states, &name);
            scope.spawn(move || {
                std::thread::sleep(after);
                let dropped = Instant::now();
                let released: Vec<_> = states.lock().unwrap().drain(NAMED_STATES..).collect();
                let count = released.len();
                drop(released);
                eprintln!(
                    "HOST-REPLAY {name}: released {count} states {:.0} ms into the first frame, \
                     in {:.1} ms",
                    after.as_secs_f64() * 1e3,
                    dropped.elapsed().as_secs_f64() * 1e3
                );
            });
        }
        for index in 0..COLD {
            let before = work.device.work_counts();
            let frame = work.frame();
            let after = work.device.work_counts();
            let presents = take();
            eprintln!(
                "HOST-REPLAY {name}: frame {index}: {:.2} ms, commit {:.2} ms, first submission \
                 to last {:.2} ms; presents meanwhile {} (commit to completion {} ms), missed {}; \
                 {}",
                frame.total.as_secs_f64() * 1e3,
                frame.commit.as_secs_f64() * 1e3,
                frame.held * 1e3,
                presents.times.len(),
                spread(presents.times),
                presents.missed,
                counters(&before, &after),
            );
        }
        let before = work.device.work_counts();
        let frames: Vec<Frame> = (0..FRAMES).map(|_| work.frame()).collect();
        let after = work.device.work_counts();
        let presents = take();
        stop.store(true, Ordering::Release);
        let ms = |f: fn(&Frame) -> f64| spread(frames.iter().map(f).collect());
        eprintln!(
            "HOST-REPLAY {name}: {FRAMES} frames (median / p90 / largest): frame {} ms, commit {} \
             ms, first submission to last {} ms; presents {} (commit to completion {} ms), \
             missed {} of {} ticks; {}",
            ms(|f| f.total.as_secs_f64() * 1e3),
            ms(|f| f.commit.as_secs_f64() * 1e3),
            ms(|f| f.held * 1e3),
            presents.times.len(),
            spread(presents.times.clone()),
            presents.missed,
            presents.times.len() as u64 + presents.missed,
            counters(&before, &after),
        );
    });
}
