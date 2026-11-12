//! What the runtime's own copy of a `Managed` buffer into its device copy costs a frame on a device
//! without unified memory (`managed.rs`: `did_modify_range` copies the range by the device's next
//! submission, before its commands), alone and beside another queue's GPU work. Each frame the CPU
//! writes the whole buffer and tells it (`did_modify_range`), then a command buffer on its own
//! queue runs `scale_and_offset` of the owned fixture `examples/runtime_examples.ll` over the
//! buffer's first floats (at most [`READ`]) into a `Shared` buffer, committed and waited for; its
//! answer is checked, so the copy landed before the kernel read it. The load: another queue's
//! command buffers, each [`LOAD_DISPATCHES`] dispatches of the same kernel over 64 MiB `Private`
//! buffers, two in flight. Each size runs on two devices: the default one, whose copies in go to
//! its fill queue where it has one (`device/staging.rs`), and one made with an extra queue of
//! transfers that takes a copy engine's second queue, so it has no fill queue and its copies in
//! run at the start of each batch's own work. Printed as `MANAGED-COPIES` lines: the frame's wall
//! time from the `did_modify_range` call to the wait's return, the load's command buffers a second
//! against the load alone, and the work counts that say where the copy ran, with each phase's
//! span on the host's clock (`from`, `to`, Unix milliseconds) for a sampler's lines (`nvidia-smi
//! dmon`); and each device's making time. Ignored: run it in release, under the GPU safety cfg.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device with unified memory, whose `Managed` buffers are one copy,
//! says so and passes.

use crate::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, CpuCacheMode, Device,
    DeviceKind, DeviceOptions, HazardTrackingMode, PipelineOptions, ResourceOptions, Size,
    StorageMode, WorkCounts,
};
use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const WAIT: Duration = Duration::from_secs(60);
/// The floats each frame's kernel reads: the copy moves the whole buffer, the check this much.
const READ: u64 = 1 << 18;
/// The load's buffers' bytes, and its dispatches a command buffer.
const LOAD_BYTES: u64 = 64 << 20;
const LOAD_DISPATCHES: usize = 24;
/// How long a phase runs, and its fewest frames.
const SPAN: Duration = Duration::from_secs(4);
const LEAST: usize = 5;

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

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis())
}

/// The 50th and 90th percentiles of `values`, nearest rank.
fn percentiles(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[((values.len() * p).div_ceil(100)).clamp(1, values.len()) - 1];
    [at(50), at(90)]
}

/// Encode `scale_and_offset` over `count` floats of `input` into `output`, `times` over.
fn encode(
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
        encoder
            .dispatch_threads(Size::new(count, 1, 1), Size::new(256, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// The load: command buffers of [`LOAD_DISPATCHES`] dispatches on a queue of its own, two in
/// flight, until it is stopped; the command buffers completed.
struct Load {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Load {
    fn start(device: &Device, state: &ComputePipelineState) -> Load {
        let (stop, done) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
        );
        let (device, state) = (device.clone(), state.clone());
        let (thread_stop, thread_done) = (stop.clone(), done.clone());
        let thread = std::thread::spawn(move || {
            let queue = device.new_command_queue().unwrap();
            let input = device
                .new_buffer(LOAD_BYTES, ResourceOptions::private())
                .unwrap();
            let output = device
                .new_buffer(LOAD_BYTES, ResourceOptions::private())
                .unwrap();
            let mut flight = std::collections::VecDeque::new();
            while !thread_stop.load(Ordering::Relaxed) {
                if flight.len() == 2 {
                    let oldest: crate::CommandBuffer = flight.pop_front().unwrap();
                    oldest.wait_until_completed_for(WAIT).unwrap();
                    thread_done.fetch_add(1, Ordering::Relaxed);
                }
                let command_buffer = queue.command_buffer().unwrap();
                let count = LOAD_BYTES / 4;
                encode(
                    &command_buffer,
                    &state,
                    (&input, &output),
                    count,
                    LOAD_DISPATCHES,
                );
                command_buffer.commit().unwrap();
                flight.push_back(command_buffer);
            }
            for command_buffer in flight {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
        });
        Load {
            stop,
            done,
            thread: Some(thread),
        }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// Frames over a `Managed` buffer of `bytes` for [`SPAN`] and at least [`LEAST`] (the module
/// docs): their wall times in milliseconds, and the span.
fn frames(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    managed: &Buffer,
    output: &Buffer,
) -> (Vec<f64>, u128, u128) {
    let bytes = managed.length();
    let count = READ.min(bytes / 4);
    let (from, began) = (unix_ms(), Instant::now());
    let mut walls = Vec::new();
    let mut frame = 0u32;
    while began.elapsed() < SPAN || walls.len() < LEAST {
        frame += 1;
        let contents = managed.contents().unwrap().cast::<f32>();
        // SAFETY: the buffer's host copy, `bytes` long, which no command buffer in flight
        // reads (each frame's was waited for).
        let floats =
            unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), (bytes / 4) as usize) };
        floats.fill(frame as f32);
        let started = Instant::now();
        managed.did_modify_range(0..bytes).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        encode(&command_buffer, state, (managed, output), count, 1);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        walls.push(started.elapsed().as_secs_f64() * 1e3);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let read = output.contents().unwrap().cast::<f32>();
        // SAFETY: `count` floats of a `Shared` buffer the completed command buffer wrote.
        let read = unsafe { std::slice::from_raw_parts(read.as_ptr(), count as usize) };
        for (index, &value) in read.iter().enumerate().step_by(4099) {
            assert_eq!(
                value,
                frame as f32 + index as f32,
                "{}: frame {frame} read float {index} before its copy in landed",
                device.name()
            );
        }
    }
    (walls, from, unix_ms())
}

/// Where the copies ran, from the work counts' difference over a phase.
fn counted(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "staging_submissions {}, fill_queue_submissions {}",
        after.staging_submissions - before.staging_submissions,
        after.fill_queue_submissions - before.fill_queue_submissions,
    )
}

/// Frames over `Managed` buffers of 64 KiB to 256 MiB, alone and beside the load, on the default
/// device and on one without a fill queue (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn managed_copies_bench() {
    for inline in [false, true] {
        let made = Instant::now();
        let device = device(inline);
        let made_ms = made.elapsed().as_secs_f64() * 1e3;
        let name = device.name().to_owned();
        if device.properties().has_unified_memory {
            eprintln!("MANAGED-COPIES {name}: unified memory, one copy; nothing to measure");
            return;
        }
        let variant = match (inline, device.fill_queue_family()) {
            (false, Some(_)) => "fill queue",
            (false, None) => "no fill queue on this device",
            (true, _) => "copies at each batch's start",
        };
        eprintln!("MANAGED-COPIES {name} ({variant}): made in {made_ms:.1} ms");
        let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
            .new_function_with_name("scale_and_offset")
            .unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let output = device
            .new_buffer(READ * 4, ResourceOptions::shared())
            .unwrap();
        for bytes in [64u64 << 10, 1 << 20, 64 << 20, 256 << 20] {
            let options = ResourceOptions::new(
                StorageMode::Managed,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            let managed = device.new_buffer(bytes, options).unwrap();
            for load in [false, true] {
                let running = load.then(|| Load::start(&device, &state));
                let alone = running.as_ref().map(|load| {
                    let (before, began) = (load.done.load(Ordering::Relaxed), Instant::now());
                    std::thread::sleep(SPAN);
                    let done = load.done.load(Ordering::Relaxed) - before;
                    done as f64 / began.elapsed().as_secs_f64()
                });
                let before = device.work_counts();
                let done = running
                    .as_ref()
                    .map(|load| load.done.load(Ordering::Relaxed));
                let began = Instant::now();
                let (walls, from, to) = frames(&device, &queue, &state, &managed, &output);
                let seconds = began.elapsed().as_secs_f64();
                let rate = (running.as_ref().zip(done)).map(|(load, done)| {
                    (load.done.load(Ordering::Relaxed) - done) as f64 / seconds
                });
                let after = device.work_counts();
                drop(running);
                let [p50, p90] = percentiles(walls.clone());
                let load_text = match (alone, rate) {
                    (Some(alone), Some(rate)) => format!(
                        "load on, its command buffers {rate:.1}/s ({:.0}% of {alone:.1}/s alone)",
                        100.0 * rate / alone
                    ),
                    _ => "load off".to_owned(),
                };
                let line = format!(
                    "MANAGED-COPIES {name} ({variant}): {} KiB a frame, {load_text}: n {} \
                     frame_ms p50 {p50:.3} p90 {p90:.3} ({:.2} GB/s), {} from {from} to {to}",
                    bytes >> 10,
                    walls.len(),
                    bytes as f64 / (p50 * 1e6),
                    counted(&before, &after),
                );
                eprintln!("{line}");
            }
        }
    }
}
