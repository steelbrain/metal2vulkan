//! What the fill queue's waits (`device/staging.rs`) cost a compositor's frames while an app
//! makes new memory, as a guest does. The app, on a queue of its own, commits command buffers of
//! [`LOAD_DISPATCHES`] dispatches of `scale_and_offset` (the owned fixture
//! `examples/runtime_examples.ll`) over 64 MiB `Private` buffers, two in flight; where it makes
//! new memory, each one first makes [`NEW_BUFFERS`] `Private` buffers of [`NEW_BYTES`] and a
//! `Private` texture of 1024 x 1024 `RGBA8Unorm`, and copies each new buffer into a buffer of its
//! own after its kernels. The compositor, on another queue, commits one dispatch over buffers it
//! made before and waits for it, frame after frame. Each phase runs [`SPAN`]: the app not making
//! memory, then making it, on the default device (its fill queue where it has one) and on one
//! made with an extra queue of transfers that takes a copy engine's second queue, so it has no
//! fill queue and fills run at the start of the batch that takes them.
//!
//! Printed as `FILL-WAITS-BENCH` lines: the compositor's frames (wall time from commit to the
//! wait's return, and GPU time, p50, p90, p99 and the most), the app's command buffers a second,
//! and the work counts (`fill_queue_submissions`, `fill_queue_waits`,
//! `fill_queue_waits_unused`). Ignored: run it in release, under the GPU safety cfg.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceKind, DeviceOptions,
    PipelineOptions, ResourceOptions, Size, Texture, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// How long a phase runs.
const SPAN: Duration = Duration::from_secs(4);
/// The app's load: its buffers' bytes, and its dispatches a command buffer.
const LOAD_BYTES: u64 = 64 << 20;
const LOAD_DISPATCHES: usize = 48;
/// What the app makes before each command buffer where it makes new memory.
const NEW_BUFFERS: usize = 2;
const NEW_BYTES: u64 = 4 << 20;
const NEW_SIDE: u64 = 1024;
/// The floats the compositor's frame reads.
const FRAME_FLOATS: u64 = 1 << 18;

/// The device the cfg selects; with an extra queue of transfers where `inline` (the module docs).
fn device(inline: bool) -> Device {
    let options = DeviceOptions {
        extra_queues: match inline {
            true => vec![vk::QueueFlags::TRANSFER],
            false => Vec::new(),
        },
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// `count` floats of `scale_and_offset` from `input` into `output`, `times` over.
fn scale(
    command_buffer: &crate::CommandBuffer,
    state: &ComputePipelineState,
    (input, output): (&Buffer, &Buffer),
    count: u64,
    times: usize,
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    for _ in 0..times {
        (encoder.dispatch_threads(Size::new(count, 1, 1), Size::new(256, 1, 1))).unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// The 50th, 90th and 99th percentiles of `values` and the most, nearest rank.
fn percentiles(mut values: Vec<f64>) -> [f64; 4] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[((values.len() * p).div_ceil(100)).clamp(1, values.len()) - 1];
    [at(50), at(90), at(99), values[values.len() - 1]]
}

/// The app (the module docs): its command buffers on a queue of its own until it is stopped,
/// making new memory before each where `making`; the command buffers completed.
struct App {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl App {
    fn start(device: &Device, state: &ComputePipelineState, making: bool) -> App {
        let (stop, done) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
        );
        let (device, state) = (device.clone(), state.clone());
        let (thread_stop, thread_done) = (stop.clone(), done.clone());
        let thread = std::thread::spawn(move || {
            let queue = device.new_command_queue().unwrap();
            let private = |bytes| {
                device
                    .new_buffer(bytes, ResourceOptions::private())
                    .unwrap()
            };
            let (input, output) = (private(LOAD_BYTES), private(LOAD_BYTES));
            let copies = private(NEW_BYTES);
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, NEW_SIDE, NEW_SIDE, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            type Made = (Vec<Buffer>, Option<Texture>);
            let mut flight: VecDeque<(crate::CommandBuffer, Made)> = VecDeque::new();
            while !thread_stop.load(Ordering::Relaxed) {
                if flight.len() == 2 {
                    let (oldest, _made) = flight.pop_front().unwrap();
                    oldest.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(oldest.status(), CommandBufferStatus::Completed);
                    thread_done.fetch_add(1, Ordering::Relaxed);
                }
                let made: Made = match making {
                    true => (
                        (0..NEW_BUFFERS).map(|_| private(NEW_BYTES)).collect(),
                        Some(device.new_texture(&descriptor).unwrap()),
                    ),
                    false => (Vec::new(), None),
                };
                let command_buffer = queue.command_buffer().unwrap();
                let count = LOAD_BYTES / 4;
                scale(
                    &command_buffer,
                    &state,
                    (&input, &output),
                    count,
                    LOAD_DISPATCHES,
                );
                if !made.0.is_empty() {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    for new in &made.0 {
                        (blit.copy_from_buffer_to_buffer(new, 0, &copies, 0, NEW_BYTES)).unwrap();
                    }
                    blit.end_encoding().unwrap();
                }
                command_buffer.commit().unwrap();
                flight.push_back((command_buffer, made));
            }
            for (command_buffer, _made) in flight {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
        });
        App {
            stop,
            done,
            thread: Some(thread),
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The compositor's and the app's numbers in each phase (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn fill_waits_bench() {
    for inline in [false, true] {
        let device = device(inline);
        let name = device.name().to_owned();
        let variant = match (inline, device.fill_queue_family()) {
            (false, Some(_)) => "fill queue",
            (false, None) => "no fill queue on this device",
            (true, _) => "fills at each batch's start",
        };
        let settle = Duration::from_secs(120);
        assert!(device.settle_in_place_checks(settle), "{name}");
        assert!(device.settle_queue_switch_check(settle), "{name}");
        let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
            .new_function_with_name("scale_and_offset")
            .unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let shared = |bytes| device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
        let (frame_in, frame_out) = (shared(FRAME_FLOATS * 4), shared(FRAME_FLOATS * 4));
        for making in [false, true] {
            let app = App::start(&device, &state, making);
            // The app's first command buffers, its buffers' fills among them, before the phase.
            while app.done.load(Ordering::Relaxed) < 2 {
                std::thread::sleep(Duration::from_millis(5));
            }
            let (before, done) = (device.work_counts(), app.done.load(Ordering::Relaxed));
            let began = Instant::now();
            let (mut walls, mut gpus) = (Vec::new(), Vec::new());
            while began.elapsed() < SPAN {
                let command_buffer = queue.command_buffer().unwrap();
                scale(
                    &command_buffer,
                    &state,
                    (&frame_in, &frame_out),
                    FRAME_FLOATS,
                    1,
                );
                let at = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                walls.push(at.elapsed().as_secs_f64() * 1e3);
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                let (start, end) = (
                    command_buffer.gpu_start_time().unwrap(),
                    command_buffer.gpu_end_time().unwrap(),
                );
                gpus.push((end - start) * 1e3);
            }
            let seconds = began.elapsed().as_secs_f64();
            let app_rate = (app.done.load(Ordering::Relaxed) - done) as f64 / seconds;
            let after = device.work_counts();
            drop(app);
            let frames = walls.len();
            let [w50, w90, w99, wmax] = percentiles(walls);
            let [g50, g90, g99, gmax] = percentiles(gpus);
            eprintln!(
                "FILL-WAITS-BENCH {name} ({variant}): app {}: frames {frames}, frame_ms p50 \
                 {w50:.3} p90 {w90:.3} p99 {w99:.3} max {wmax:.3}, gpu_ms p50 {g50:.3} p90 \
                 {g90:.3} p99 {g99:.3} max {gmax:.3}, app {app_rate:.1}/s, fill_queue_submissions \
                 {}, fill_queue_waits {}, fill_queue_waits_unused {}",
                match making {
                    true => "making new memory",
                    false => "making none",
                },
                after.fill_queue_submissions - before.fill_queue_submissions,
                after.fill_queue_waits - before.fill_queue_waits,
                after.fill_queue_waits_unused - before.fill_queue_waits_unused,
            );
        }
    }
}
