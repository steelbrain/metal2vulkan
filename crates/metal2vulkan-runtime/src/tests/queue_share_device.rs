//! Command queues sharing `VkQueue`s (`queue_share.rs`) on a device, under both values of
//! `QueueSharingPolicy`: what the moves keep and what they never do.
//!
//! - **Order across moves.** A queue's command buffers each copy the word the one before wrote
//!   into the next word, and write their sequence number, while two other queues keep their
//!   `VkQueue`s busy in turn, so the queue moves between them where the device shares: every word
//!   reads as written in commit order, and the queue moved (`queue_moves` above 0) where the
//!   device shares, never elsewhere.
//! - **No deadlock.** A queue joins another's busy `VkQueue` and commits a command buffer that
//!   waits for a shared event, before the other commits the one that signals it: both complete.
//! - **Pinned and presenting queues never move**, where an unpinned one in the same place does.
//! - **Unshared queues.** No queue moves onto an unshared queue's `VkQueue` (each command buffer's
//!   `VkQueue` read, `joins_withheld_from_unshared` above 0 where the device shares), a queue
//!   placed on one leaves it at its next idle point (`queue_moves_off_unshared`), and under
//!   `PerQueue` an unshared queue takes the round robin's next `VkQueue`, as any queue does.
//! - **Isolation.** A compositor's small command buffers, committed at 4,800 a second and never
//!   waited for, beside an application's command buffers of about 30 ms each: once the
//!   application's queue went long (where the device shares), the compositor's 99th percentile
//!   over 2 s stays under 5 ms with at most one stall over 10 ms; on an unshared queue, from the
//!   start, with none.
//!
//! Where the device does not share (`PerQueue`, its check found no idling, or it has no two
//! `VkQueue`s of one family: MoltenVK, RADV, lavapipe) each test asserts that no queue moves, and
//! the isolation tests run only where the compositor's and the application's queues take
//! `VkQueue`s of their own. A device test: compiled only under the validation crate's GPU safety
//! cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::Scratch;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions,
    ShaderCacheOptions,
};
use metal2vulkan::device::{DevicePolicy, QueueSharingPolicy};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest any wait of these tests lasts.
const WAIT: Duration = Duration::from_secs(20);

/// The bytes one fill of a long command buffer writes.
const FILL_BYTES: u64 = 64 << 20;

/// The device the cfg selects under `queue_sharing`, its shader cache in `cache`, its check
/// settled.
fn device(cache: &Scratch, queue_sharing: QueueSharingPolicy) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    options.policy = DevicePolicy::UNTRUSTED;
    options.policy.queue_sharing = queue_sharing;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_queue_switch_check(WAIT), "the check settles");
    println!("{queue_sharing:?}: {}", device.queue_switch_check_report());
    device
}

/// Both values, each with a device of its own.
fn each_policy(leg: &str, test: impl Fn(&Device, bool)) {
    for policy in [QueueSharingPolicy::Adaptive, QueueSharingPolicy::PerQueue] {
        let scratch = Scratch::new(&format!("queue-share-{leg}-{policy:?}"));
        let device = device(&scratch, policy);
        let shares = device.shares_queues();
        assert!(
            !shares || policy == QueueSharingPolicy::Adaptive,
            "no device shares under PerQueue"
        );
        test(&device, shares);
        if !shares {
            assert_eq!(device.work_counts().queue_moves, 0, "{policy:?}");
        }
    }
}

/// Commit `count` command buffers of one 1 MiB fill each on `queue`, not waited for: its
/// `VkQueue` busy, its oldest batch young.
fn burst(queue: &CommandQueue, target: &Buffer, count: usize) -> Vec<CommandBuffer> {
    (0..count)
        .map(|_| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(target, 0..(1 << 20), 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect()
}

fn wait_all(command_buffers: Vec<CommandBuffer>) {
    for command_buffer in command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
}

/// The words of a `Shared` buffer.
fn words(buffer: &Buffer, count: usize) -> Vec<u32> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `count` words, and no command writes it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), count) }.to_vec()
}

#[test]
fn a_queue_s_command_buffers_run_in_commit_order_across_its_moves() {
    const STEPS: usize = 48;
    each_policy("order", |device, shares| {
        let busy = [
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        ];
        let scratch = device
            .new_buffer(1 << 20, ResourceOptions::private())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let chain = device
            .new_buffer(8 * STEPS as u64, ResourceOptions::shared())
            .unwrap();
        let mut seen = Vec::new();
        let mut kept = Vec::new();
        for step in 0..STEPS {
            // Every fourth step one of the other queues is busy, in turn; the rest the queue
            // makes its next command buffer with its last one in flight, so with no move.
            let idle = step % 4 == 0;
            if idle {
                kept.extend(burst(&busy[step / 4 % 2], &scratch, 24));
            }
            let command_buffer = queue.command_buffer().unwrap();
            seen.push(queue.vk_queue());
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let at = 4 * step as u64;
            match step {
                0 => blit.fill_buffer(&chain, 0..4, 0x7e).unwrap(),
                _ => blit
                    .copy_from_buffer_to_buffer(&chain, at - 4, &chain, at, 4)
                    .unwrap(),
            }
            let sequence = 4 * (STEPS + step) as u64;
            blit.fill_buffer(&chain, sequence..sequence + 4, step as u8)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            if (step + 1) % 4 == 0 {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            } else {
                kept.push(command_buffer);
            }
        }
        wait_all(kept);
        let read = words(&chain, 2 * STEPS);
        assert!(
            read[..STEPS].iter().all(|&word| word == 0x7e7e_7e7e),
            "a command buffer read its predecessor's word before it was written: {:x?}",
            &read[..STEPS]
        );
        for (step, &word) in read[STEPS..].iter().enumerate() {
            assert_eq!(word, u32::from_ne_bytes([step as u8; 4]), "sequence {step}");
        }
        let moved = seen.windows(2).filter(|pair| pair[0] != pair[1]).count();
        println!("VkQueues {seen:?}: {moved} moves");
        match shares {
            true => {
                assert!(moved > 0, "the queue moved where the device shares");
                assert!(device.work_counts().queue_moves > 0);
            }
            false => assert_eq!(moved, 0),
        }
    });
}

#[test]
fn a_shared_event_wait_on_a_shared_vk_queue_committed_before_its_signal_completes() {
    each_policy("event", |device, shares| {
        let signaller = device.new_command_queue().unwrap();
        let waiter = device.new_command_queue().unwrap();
        let scratch = device
            .new_buffer(1 << 20, ResourceOptions::private())
            .unwrap();
        let event = device.new_shared_event().unwrap();
        let mut joined = false;
        for round in 1..=8u64 {
            let busy = burst(&signaller, &scratch, 24);
            let waiting = waiter.command_buffer().unwrap();
            joined |= waiter.vk_queue() == signaller.vk_queue();
            waiting
                .encode_wait_for_event((&event).into(), round)
                .unwrap();
            let mut blit = waiting.blit_command_encoder().unwrap();
            blit.fill_buffer(&scratch, 0..4096, 1).unwrap();
            blit.end_encoding().unwrap();
            waiting.commit().unwrap();
            let signalling = signaller.command_buffer().unwrap();
            let mut blit = signalling.blit_command_encoder().unwrap();
            blit.fill_buffer(&scratch, 4096..8192, 2).unwrap();
            blit.end_encoding().unwrap();
            signalling
                .encode_signal_event((&event).into(), round)
                .unwrap();
            signalling.commit().unwrap();
            waiting.wait_until_completed_for(WAIT).unwrap();
            signalling.wait_until_completed_for(WAIT).unwrap();
            wait_all(busy);
        }
        if shares {
            assert!(joined, "the waiter joined the signaller's VkQueue");
        }
    });
}

#[test]
fn pinned_and_presenting_queues_never_move_where_an_unpinned_one_does() {
    each_policy("pinned", |device, shares| {
        let busy = device.new_command_queue().unwrap();
        let scratch = device
            .new_buffer(1 << 20, ResourceOptions::private())
            .unwrap();
        let pinned = device.new_command_queue().unwrap();
        // SAFETY: the closure keeps no handle and submits nothing.
        unsafe { pinned.with_vk_queue(|_, _| {}) };
        let presenting = device.new_command_queue_for_presenting().unwrap();
        let free = device.new_command_queue().unwrap();
        let (pinned_at, presenting_at) = (pinned.vk_queue(), presenting.vk_queue());
        let mut free_moved = false;
        for _ in 0..8 {
            for queue in [&pinned, &presenting, &free] {
                let kept = burst(&busy, &scratch, 24);
                let before = queue.vk_queue();
                wait_all(burst(queue, &scratch, 1));
                free_moved |= std::ptr::eq(queue, &free) && queue.vk_queue() != before;
                wait_all(kept);
            }
        }
        assert_eq!(pinned.vk_queue(), pinned_at, "a pinned queue moved");
        assert_eq!(
            presenting.vk_queue(),
            presenting_at,
            "a presenting queue moved"
        );
        if shares {
            assert!(
                free_moved,
                "the unpinned queue moved where the device shares"
            );
        }
    });
}

#[test]
fn no_queue_joins_an_unshared_queue_s_vk_queue_and_one_placed_there_leaves() {
    each_policy("unshared", |device, shares| {
        let scratch = device
            .new_buffer(1 << 20, ResourceOptions::private())
            .unwrap();
        let unshared = device.new_command_queue_unshared(64).unwrap();
        let home = unshared.vk_queue();
        let others: Vec<CommandQueue> = (0..3)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        for _ in 0..16 {
            let kept = burst(&unshared, &scratch, 24);
            for queue in &others {
                let made = queue.command_buffer().unwrap();
                assert_ne!(
                    queue.vk_queue(),
                    home,
                    "a queue joined the unshared queue's VkQueue"
                );
                drop(made);
            }
            wait_all(kept);
        }
        assert_eq!(unshared.vk_queue(), home, "the unshared queue moved");
        if shares {
            assert!(device.work_counts().joins_withheld_from_unshared > 0);
            // Every round robin's `VkQueue` holds a queue, so the next unshared queue shares
            // one with a queue, which leaves at its next idle point.
            let round_robin = device.shared().round_robin();
            let filled: Vec<CommandQueue> = (0..round_robin)
                .map(|_| device.new_command_queue().unwrap())
                .collect();
            let second = device.new_command_queue_unshared(64).unwrap();
            let there: Vec<&CommandQueue> = (filled.iter().chain(&others))
                .filter(|queue| queue.vk_queue() == second.vk_queue())
                .collect();
            assert!(
                !there.is_empty(),
                "the second unshared queue shares a VkQueue"
            );
            let before = device.work_counts().queue_moves_off_unshared;
            for queue in &there {
                drop(queue.command_buffer().unwrap());
                assert_ne!(queue.vk_queue(), second.vk_queue());
            }
            assert_eq!(
                device.work_counts().queue_moves_off_unshared - before,
                there.len() as u64
            );
        }
    });
}

#[test]
fn under_per_queue_an_unshared_queue_takes_the_round_robin_s_next_vk_queue() {
    let scratch = Scratch::new("queue-share-per-queue");
    let device = device(&scratch, QueueSharingPolicy::PerQueue);
    let round_robin = device.shared().round_robin();
    let queues: Vec<CommandQueue> = (0..2 * round_robin + 1)
        .map(|made| match made % 2 {
            0 => device.new_command_queue().unwrap(),
            _ => device.new_command_queue_unshared(64).unwrap(),
        })
        .collect();
    for (made, queue) in queues.iter().enumerate() {
        assert_eq!(queue.vk_queue(), made % round_robin, "queue {made}");
    }
    assert_eq!(
        device.work_counts().queues_sharing_a_vk_queue,
        (queues.len() - round_robin) as u64
    );
    let refused = device.new_command_queue_unshared(0).unwrap_err();
    assert!(
        matches!(refused, crate::Error::Invalid(ref invalid) if invalid.call == "new_command_queue_unshared"),
        "{refused:?}"
    );
}

/// What one isolation run measured: the compositor's latencies in milliseconds.
struct Isolation {
    latencies: Vec<f64>,
}

impl Isolation {
    fn p99(&self) -> f64 {
        let mut sorted = self.latencies.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)]
    }

    fn stalls(&self) -> usize {
        self.latencies.iter().filter(|&&ms| ms > 10.0).count()
    }
}

/// The fills of one long command buffer of about 30 ms of GPU time over `target`, measured.
fn long_fills(queue: &CommandQueue, target: &Buffer) -> usize {
    let mut fills = 64;
    for _ in 0..3 {
        let command_buffer = long(queue, target, fills);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        fills = ((fills as f64 * 0.030 / gpu.max(1e-6)) as usize).clamp(8, 4096);
    }
    fills
}

fn long(queue: &CommandQueue, target: &Buffer, fills: usize) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for fill in 0..fills {
        blit.fill_buffer(target, 0..FILL_BYTES, fill as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer
}

/// A compositor on `compositor` committing one small fill at 4,800 a second, never waiting,
/// beside an application on its own queue committing long command buffers back to back, each
/// waited for: the compositor's latencies over 2 s from the moment `ready` answers (bounded).
/// `None` where the two queues share a `VkQueue` whatever the policy (one queue on the device).
fn isolation(
    device: &Device,
    compositor: CommandQueue,
    ready: impl Fn(&Device) -> bool,
) -> Option<Isolation> {
    let app = device.new_command_queue().unwrap();
    if app.vk_queue() == compositor.vk_queue() && !device.shares_queues() {
        println!("the device gives the two queues one VkQueue: no isolation to measure");
        return None;
    }
    let big = device
        .new_buffer(FILL_BYTES, ResourceOptions::private())
        .unwrap();
    let fills = long_fills(&app, &big);
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let application = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = long(&app, &big, fills);
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
        })
    };
    let latencies = Arc::new(Mutex::new(Vec::new()));
    let recording = Arc::new(AtomicBool::new(false));
    let composing = {
        let (stop, latencies, recording) = (
            Arc::clone(&stop),
            Arc::clone(&latencies),
            Arc::clone(&recording),
        );
        std::thread::spawn(move || {
            let period = Duration::from_secs_f64(1.0 / 4800.0);
            let mut next = Instant::now();
            let mut last = None;
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = compositor.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&small, 0..256, 3).unwrap();
                blit.end_encoding().unwrap();
                let committed = Instant::now();
                let (latencies, recording) = (Arc::clone(&latencies), Arc::clone(&recording));
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        if recording.load(Ordering::Relaxed) {
                            let ms = committed.elapsed().as_secs_f64() * 1e3;
                            latencies.lock().unwrap().push(ms);
                        }
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
                last = Some(command_buffer);
                next += period;
                let now = Instant::now();
                match next > now {
                    true => std::thread::sleep(next - now),
                    false => next = now,
                }
            }
            if let Some(last) = last {
                last.wait_until_completed_for(WAIT).unwrap();
            }
        })
    };
    let began = Instant::now();
    while !ready(device) && began.elapsed() < WAIT {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(ready(device), "the run did not reach its measured part");
    recording.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(2));
    recording.store(false, Ordering::Relaxed);
    stop.store(true, Ordering::Relaxed);
    composing.join().unwrap();
    application.join().unwrap();
    let latencies = latencies.lock().unwrap().clone();
    assert!(
        latencies.len() > 1000,
        "{} compositor samples",
        latencies.len()
    );
    Some(Isolation { latencies })
}

#[test]
fn a_compositor_beside_a_long_queue_keeps_its_latency_once_that_queue_went_long() {
    each_policy("isolation", |device, shares| {
        let compositor = device.new_command_queue().unwrap();
        let ready = move |device: &Device| !shares || device.work_counts().queues_gone_long >= 1;
        let Some(run) = isolation(device, compositor, ready) else {
            return;
        };
        println!(
            "shares {shares}: p99 {:.2} ms, {} stalls of {} samples; {:?}",
            run.p99(),
            run.stalls(),
            run.latencies.len(),
            device.work_counts().queues_gone_long
        );
        assert!(run.p99() < 5.0, "p99 {:.2} ms", run.p99());
        assert!(run.stalls() <= 1, "{} stalls", run.stalls());
    });
}

#[test]
fn an_unshared_compositor_beside_a_long_queue_never_stalls() {
    each_policy("isolation-unshared", |device, shares| {
        let compositor = device.new_command_queue_unshared(64).unwrap();
        let Some(run) = isolation(device, compositor, |_| true) else {
            return;
        };
        println!(
            "shares {shares}: p99 {:.2} ms, {} stalls of {} samples",
            run.p99(),
            run.stalls(),
            run.latencies.len()
        );
        assert!(run.p99() < 5.0, "p99 {:.2} ms", run.p99());
        assert_eq!(run.stalls(), 0, "the unshared compositor stalled");
    });
}
