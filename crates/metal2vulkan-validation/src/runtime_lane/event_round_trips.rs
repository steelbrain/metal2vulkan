//! The round trip of a checked command buffer that waits for an event value another queue's
//! command buffer signals, on the device the cfg selects: a measurement, not a check (run by name
//! with `--run-ignored all`).
//!
//! The consumer's rings wait for each other: a command buffer of one waits for the value the
//! other's last one signalled, and signals the value the other's next one waits for. Each is
//! checked (`CommandBuffer::set_submission_check`), so the scheduler holds it until the signal it
//! waits for landed, runs its check, and only then submits it (`command_buffer/checks.rs`): every
//! step of the chain is a round trip through the host, the GPU idle meanwhile.
//!
//! [`HOPS`] such command buffers alternate between two queues, each filling [`FILL`] bytes of a
//! buffer of its own between its wait and its signal of one shared event, the first behind a gate
//! the host raises once all are committed. For each one after the first [`WARM`], the line prints
//! the time from the previous one's `GPUEndTime` to its check (the signal landing, seen by a
//! queue's waiter, the scheduler woken and its pass), from its check to its `GPUStartTime` (the
//! pass that submits it, the submission, and the GPU picking it up), the round trip between the two
//! GPU times, and its GPU time, as medians and 90th percentiles, under each `CompletionPolicy`,
//! twice each in turn, on a device without the validation layer whose GPU times are the device's
//! calibrated timestamps. Beside them: the scheduler thread's sleeps, those it slept waiting on the
//! GPU itself, and its wakings by completions, per command buffer.
//!
//! Checked beside it: that thread waits for such a signal on the GPU itself, the signal held on the
//! GPU behind a gate the test opens once the thread waits there, and the check still runs only
//! once the signal landed; and a host signal wakes it from that wait, whether of the event it
//! waits for there or of another head's (no head waits for the GPU's signal once the host met its
//! wait).

use super::{
    layer_reported_nothing_at_all, loader, open_device, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::device::CompletionPolicy;
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, DeviceOptions, EventRef, ResourceOptions,
    ShaderCacheOptions, VkEventSignal,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The command buffers of the chain.
const HOPS: usize = 200;
/// The first ones, not measured.
const WARM: usize = 10;
/// The bytes each one fills: a few microseconds of GPU time, so the round trip is the host's.
const FILL: u64 = 64 << 10;
/// The GPU time, in seconds, a long command buffer lasts at least ([`long_fills`]): the window
/// for a host signal, the scheduler's waking and a check before its end, which a loaded host's
/// delays of the test's thread or the scheduler's must not close.
const LONG_FOR: f64 = 0.3;
/// The fills of the command buffer [`long_fills`] times.
const TIMED_FILLS: usize = 32;
/// The fewest and the most fills of a long command buffer ([`long_fills`]).
const LONG_FILLS: (usize, usize) = (256, 8192);

/// The host's clock, in seconds, as the runtime's GPU times read it (`gpu_time.rs`).
fn host_seconds() -> f64 {
    #[cfg(target_os = "macos")]
    let clock = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_os = "macos"))]
    let clock = libc::CLOCK_MONOTONIC_RAW;
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(clock, &mut now) };
    assert_eq!(read, 0, "the host's clock");
    now.tv_sec as f64 + now.tv_nsec as f64 / 1e9
}

/// How many fills of all of `buffer` a command buffer of `queue` takes to last [`LONG_FOR`] of
/// GPU time, from the GPU time of one of [`TIMED_FILLS`] fills (its wall time from its commit to
/// its completion, where the device answers no GPU time), within [`LONG_FILLS`].
fn long_fills(queue: &CommandQueue, buffer: &Buffer) -> usize {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for at in 0..TIMED_FILLS {
        blit.fill_buffer(buffer, 0..buffer.length(), at as u8)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    let start = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wall = start.elapsed().as_secs_f64();
    let gpu = command_buffer.gpu_end_time().unwrap_or(0.0)
        - command_buffer.gpu_start_time().unwrap_or(0.0);
    let took = if gpu > 0.0 { gpu } else { wall };
    let fills = (TIMED_FILLS as f64 * LONG_FOR / took).ceil() as usize;
    let fills = fills.clamp(LONG_FILLS.0, LONG_FILLS.1);
    println!(
        "long command buffer: {fills} fills of {} bytes, {TIMED_FILLS} taking {:.2} ms",
        buffer.length(),
        took * 1e3
    );
    fills
}

/// The value at `at` (0 to 1) of `values`, sorted.
fn percentile(mut values: Vec<f64>, at: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * at).round() as usize]
}

/// The module docs' measurement.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn checked_event_round_trips() {
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped].repeat(2) {
        let cache = ScratchCache::new().unwrap();
        let mut options = DeviceOptions::default();
        options.policy.completion = completion;
        options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
        options.loader = loader();
        let device = open_device(TEST_DEVICE, options).unwrap();
        let name = device.name().to_owned();
        // Every command buffer is committed before the gate opens, more than a queue's default
        // 64 uncompleted command buffers.
        let queues = [
            device
                .new_command_queue_with_max_command_buffer_count(HOPS)
                .unwrap(),
            device
                .new_command_queue_with_max_command_buffer_count(HOPS)
                .unwrap(),
        ];
        let (event, gate) = (
            device.new_shared_event().unwrap(),
            device.new_shared_event().unwrap(),
        );
        let buffers: Vec<Buffer> = (0..HOPS)
            .map(|_| device.new_buffer(FILL, ResourceOptions::private()).unwrap())
            .collect();
        let checked: Vec<Arc<Mutex<f64>>> = buffers.iter().map(|_| Arc::default()).collect();
        let before = device.work_counts();
        let committed: Vec<CommandBuffer> = (buffers.iter().zip(&checked).enumerate())
            .map(|(at, (buffer, checked))| {
                let command_buffer = queues[at % 2].command_buffer().unwrap();
                let checked = Arc::clone(checked);
                command_buffer
                    .set_submission_check(Arc::new(move |_| {
                        let mut first = checked.lock().unwrap();
                        if *first == 0.0 {
                            *first = host_seconds();
                        }
                        None
                    }))
                    .unwrap();
                let (waited, value) = match at {
                    0 => (&gate, 1),
                    _ => (&event, at as u64),
                };
                command_buffer
                    .encode_wait_for_event(EventRef::from(waited), value)
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(buffer, 0..FILL, at as u8).unwrap();
                blit.end_encoding().unwrap();
                command_buffer
                    .encode_signal_event(EventRef::from(&event), at as u64 + 1)
                    .unwrap();
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        gate.set_signaled_value(1);
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "command buffer {at}");
        }
        let after = device.work_counts();
        let times: Vec<(f64, f64, f64)> = (committed.iter().zip(&checked))
            .map(|(command_buffer, checked)| {
                (
                    *checked.lock().unwrap(),
                    command_buffer.gpu_start_time().unwrap(),
                    command_buffer.gpu_end_time().unwrap(),
                )
            })
            .collect();
        let hops = || (WARM.max(1)..HOPS).map(|at| (times[at - 1], times[at]));
        let us = |values: Vec<f64>| -> String {
            let values: Vec<f64> = values.into_iter().map(|value| value * 1e6).collect();
            format!(
                "{:.0} us (p90 {:.0})",
                percentile(values.clone(), 0.5),
                percentile(values, 0.9)
            )
        };
        let to_check = us(hops()
            .map(|(before, (check, _, _))| check - before.2)
            .collect());
        let to_start = us(hops().map(|(_, (check, start, _))| start - check).collect());
        let round_trip = us(hops()
            .map(|(before, (_, start, _))| start - before.2)
            .collect());
        let gpu = us(hops().map(|(_, (_, start, end))| end - start).collect());
        let each = |count: u64| count as f64 / HOPS as f64;
        eprintln!(
            "EVENT-ROUND-TRIP {name} {completion:?}: from the signal's GPUEndTime to the waiting \
             command buffer's check {to_check}; from its check to its GPUStartTime {to_start}; \
             the round trip {round_trip}; its GPU time {gpu}; a command buffer: {:.2} scheduler \
             sleeps, {:.2} of them on the GPU, {:.2} wakings by completions",
            each(after.scheduler_sleeps - before.scheduler_sleeps),
            each(after.scheduler_gpu_waits - before.scheduler_gpu_waits),
            each(after.scheduler_wakes_completion - before.scheduler_wakes_completion),
        );
    }
}

/// A plain event's signal the test reserved (`Event::reserve_vk_signal`) and gives from the host
/// ([`open`](Self::open)), which a command buffer's wait for it holds on the GPU: such a wait goes
/// to its `VkQueue` at once, so the command buffer is submitted and its signals are in flight
/// until the test opens the gate, however fast the GPU and however loaded the host. Given on drop
/// where not yet, so no command buffer waits for good after a failed assertion.
struct GpuGate {
    device: metal2vulkan_runtime::Device,
    event: metal2vulkan_runtime::Event,
    counter: u64,
    opened: bool,
}

impl GpuGate {
    /// A fresh event of `device`, its first signal reserved.
    fn new(device: &metal2vulkan_runtime::Device) -> GpuGate {
        let event = device.new_event().unwrap();
        // SAFETY: the reservation's one signal is given once, from the host, by `open` (or the
        // drop): a fresh event's first, so after nothing.
        let reserved = unsafe { event.reserve_vk_signal(1) }.unwrap();
        let VkEventSignal::Counter {
            counter,
            after: None,
        } = reserved
        else {
            panic!("a fresh event's first signal reserved as {reserved:?}");
        };
        GpuGate {
            device: device.clone(),
            event,
            counter,
            opened: false,
        }
    }

    /// Give the reserved signal from the host, once: the work waiting for it runs on.
    fn open(&mut self) {
        if self.opened {
            return;
        }
        self.opened = true;
        let info = vk::SemaphoreSignalInfo::default()
            // SAFETY: the semaphore the event holds while it lives, signalled only as reserved.
            .semaphore(unsafe { self.event.vk_semaphore() })
            .value(self.counter);
        // SAFETY: the reservation's one signal operation, from the host, of a semaphore of the
        // live device; none of the device's objects is destroyed.
        unsafe { self.device.raw().device.signal_semaphore(&info) }.unwrap();
    }
}

impl Drop for GpuGate {
    fn drop(&mut self) {
        self.open();
    }
}

/// Where the one head waits for an event value another queue's submitted command buffer signals,
/// the scheduler's thread waits for it on the GPU itself (`scheduler/mod.rs`; not on MoltenVK,
/// which keeps the condition variable), and the head's check, run as the signal landed, reads it.
/// The signalling command buffer waits at a [`GpuGate`] before its fill and its signal: it is
/// submitted, its signal in flight and held on the GPU, while the test waits for the scheduler's
/// thread to wait on the GPU (`WorkCounts::scheduler_gpu_waits`), the check not yet run and the
/// event still 0, and only then opens the gate. So the GPU wait is met however late the
/// scheduler's thread comes, where before a 64 MiB fill held the signal for some 70 us of an RTX
/// 5080's time, which a loaded host's scheduler could come after. On MoltenVK the test opens the
/// gate after 2 ms instead.
#[test]
fn a_checked_head_waits_for_another_queue_s_signal_on_the_gpu() {
    let messages = Messages::default();
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped] {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = completion;
        })
        .unwrap();
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let queues = [
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        ];
        let buffers = [
            device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
            device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
        ];
        let before = device.work_counts();
        for round in 0..3_u8 {
            let what = format!("{completion:?} round {round}");
            let event = device.new_shared_event().unwrap();
            let mut gate = GpuGate::new(&device);
            let signalling = queues[0].command_buffer().unwrap();
            signalling
                .encode_wait_for_event(EventRef::from(&gate.event), 1)
                .unwrap();
            let mut blit = signalling.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffers[0], 0..FILL, round).unwrap();
            blit.end_encoding().unwrap();
            signalling
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap();
            let checked = queues[1].command_buffer().unwrap();
            let seen: Arc<Mutex<Option<u64>>> = Arc::default();
            let (seen_by_check, read) = (Arc::clone(&seen), event.clone());
            checked
                .set_submission_check(Arc::new(move |_| {
                    seen_by_check
                        .lock()
                        .unwrap()
                        .get_or_insert(read.signaled_value());
                    None
                }))
                .unwrap();
            checked
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut blit = checked.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffers[1], 0..FILL, round).unwrap();
            blit.end_encoding().unwrap();
            let waits = device.work_counts().scheduler_gpu_waits;
            signalling.commit().unwrap();
            checked.commit().unwrap();
            // The scheduler's thread holds the checked one, waiting on the GPU for the signal the
            // gate holds.
            if moltenvk {
                std::thread::sleep(Duration::from_millis(2));
            } else {
                let start = Instant::now();
                while device.work_counts().scheduler_gpu_waits == waits {
                    assert!(
                        start.elapsed() < WAIT,
                        "{what}: the scheduler never waited for the signal on the GPU"
                    );
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
            assert_eq!(
                *seen.lock().unwrap(),
                None,
                "{what}: checked before the gate"
            );
            assert_eq!(
                event.signaled_value(),
                0,
                "{what}: signalled before the gate"
            );
            gate.open();
            for command_buffer in [&signalling, &checked] {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{what}");
            }
            assert_eq!(
                *seen.lock().unwrap(),
                Some(1),
                "{what}: the check ran before the signal landed"
            );
        }
        if moltenvk {
            let waits = device.work_counts().scheduler_gpu_waits - before.scheduler_gpu_waits;
            assert_eq!(
                waits, 0,
                "{completion:?}: MoltenVK keeps the condition variable"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// Where the scheduler's thread waits on the GPU for a signal a long command buffer gives at its
/// end, a host signal that meets a checked head's wait lets that head go at once, before the GPU's
/// signal lands: the host signals the event the thread waits for (`same`), or the event of another
/// head, which waits for no GPU signal. The head's check runs while the long one still runs (it
/// reads another event the long one signals at its end, still 0), and reads the value the host
/// signalled; nothing is left waiting. The host signals once the thread waits on the GPU
/// (`WorkCounts::scheduler_gpu_waits`), which it enters only after every commit was taken in, and
/// its signal wakes the thread from there. On MoltenVK
/// the thread never waits on the GPU, and sleeps on its condition variable instead
/// (`scheduler/mod.rs`), which the host's signal may find it out of (a waking is counted only
/// where it slept): the head is held to the same letting go, before the long one's end. The long
/// one's fills are as many as last [`LONG_FOR`] on the device ([`long_fills`]), its signals
/// landing well after the host's even where the host delays the test's thread or the scheduler's.
#[test]
fn a_host_signal_wakes_the_scheduler_from_its_wait_on_the_gpu() {
    /// The bytes of each of the long command buffer's fills.
    const LONG: u64 = 64 << 20;
    let messages = Messages::default();
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped] {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = completion;
        })
        .unwrap();
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let queues = [
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        ];
        let long = device.new_buffer(LONG, ResourceOptions::private()).unwrap();
        let fills = long_fills(&queues[0], &long);
        for same in [true, false] {
            let what = format!("{completion:?}, the host signalling the same event: {same}");
            let before = device.work_counts();
            let (event, host, done) = (
                device.new_shared_event().unwrap(),
                device.new_shared_event().unwrap(),
                device.new_shared_event().unwrap(),
            );
            // The long one, signalling `done` and the event at its end.
            let signalling = queues[0].command_buffer().unwrap();
            let mut blit = signalling.blit_command_encoder().unwrap();
            for at in 0..fills {
                blit.fill_buffer(&long, 0..LONG, at as u8).unwrap();
            }
            blit.end_encoding().unwrap();
            signalling
                .encode_signal_event(EventRef::from(&done), 1)
                .unwrap();
            signalling
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap();
            // A checked one waiting for `waited`'s 1, its check reading it and `done`; filling a
            // buffer of its own, so neither waits behind the other (`scheduler/order.rs`).
            let checked = |queue: usize, waited: &metal2vulkan_runtime::SharedEvent| {
                let short = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
                let command_buffer = queues[queue].command_buffer().unwrap();
                let seen: Arc<Mutex<Option<(u64, u64)>>> = Arc::default();
                let (seen_by_check, read, long_done) =
                    (Arc::clone(&seen), waited.clone(), done.clone());
                command_buffer
                    .set_submission_check(Arc::new(move |_| {
                        seen_by_check
                            .lock()
                            .unwrap()
                            .get_or_insert((read.signaled_value(), long_done.signaled_value()));
                        None
                    }))
                    .unwrap();
                command_buffer
                    .encode_wait_for_event(EventRef::from(waited), 1)
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&short, 0..FILL, queue as u8).unwrap();
                blit.end_encoding().unwrap();
                (command_buffer, seen)
            };
            // The head the thread waits on the GPU for.
            let (gpu_waited, gpu_seen) = checked(1, &event);
            // Another, which only the host's signal meets.
            let (host_waited, host_seen) = checked(2, &host);
            // The head the thread waits on the GPU for last: its first wait there comes only once
            // a pass took it in, and with it every commit before it (a pass drains them all), so
            // no commit wakes the thread from that wait, and the host's signal finds it there.
            host_waited.commit().unwrap();
            signalling.commit().unwrap();
            gpu_waited.commit().unwrap();
            // The thread holds both heads, waiting on the GPU for the first's signal.
            if moltenvk {
                std::thread::sleep(Duration::from_millis(2));
            } else {
                let start = std::time::Instant::now();
                while device.work_counts().scheduler_gpu_waits == before.scheduler_gpu_waits {
                    assert!(
                        start.elapsed() < WAIT,
                        "{what}: the scheduler never waited on the GPU"
                    );
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
            let (signalled, let_go, seen) = match same {
                true => (&event, &gpu_waited, &gpu_seen),
                false => (&host, &host_waited, &host_seen),
            };
            signalled.set_signaled_value(1);
            let_go.wait_until_completed_for(WAIT).unwrap();
            // The other head's wait: the long one's signal meets the one the thread waited for.
            if same {
                host.set_signaled_value(1);
            }
            for command_buffer in [&signalling, &gpu_waited, &host_waited] {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{what}");
            }
            assert_eq!(
                *seen.lock().unwrap(),
                Some((1, 0)),
                "{what}: the check's values of the event the host signalled and of the long \
                 one's end"
            );
            let after = device.work_counts();
            if moltenvk {
                let waits = after.scheduler_gpu_waits - before.scheduler_gpu_waits;
                assert_eq!(waits, 0, "{what}: MoltenVK keeps the condition variable");
            } else {
                assert!(
                    after.scheduler_wakes_host > before.scheduler_wakes_host,
                    "{what}: no host signal woke the scheduler from its wait on the GPU"
                );
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
