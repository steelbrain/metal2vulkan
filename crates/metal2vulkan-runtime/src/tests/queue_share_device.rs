//! Command queues sharing `VkQueue`s (`queue_share.rs`) on a device, under both values of
//! `QueueSharingPolicy`: what the moves keep and what they never do.
//!
//! - **Order across moves.** A queue's command buffers each copy the word the one before wrote
//!   into the next word, and write their sequence number, while two other queues keep their
//!   `VkQueue`s busy in turn, so the queue moves between them where the device shares: every word
//!   reads as written in commit order, and the queue moved where the device shares at every idle
//!   point where the rule calls for a move, never elsewhere. A choice that stays there is the
//!   rule's where it found no other `VkQueue` busy or passed over the busy ones for their oldest
//!   batches' age (`picks_with_none_busy`, `joins_withheld_from_old_heads`): on a busy host or
//!   beside another process's GPU work every choice can.
//! - **No deadlock.** A queue joins another's busy `VkQueue` and commits a command buffer that
//!   waits for a shared event, before the other commits the one that signals it: both complete.
//!   The join is asserted where the rule calls for it, as above.
//! - **Pinned and presenting queues never move**, where an unpinned one in the same place does.
//! - **Unshared queues.** No queue moves onto an unshared queue's `VkQueue` (each command buffer's
//!   `VkQueue` read, `joins_withheld_from_unshared` above 0 where the device shares), and none
//!   takes it as it is made while another is free of unshared queues; a queue placed on one where
//!   none was free leaves it at its next idle point where the device shares
//!   (`queue_moves_off_unshared`).
//! - **Placement at creation**, under `Adaptive` (sharing or not, as the device's check found),
//!   `PerQueue` and an untimed device: over a churn of queues made and dropped well past the round
//!   robin's `VkQueue`s, as a guest's processes make them, each new queue takes the `VkQueue` the
//!   fewest live queues hold of those no unshared queue holds, and an unshared queue one no queue
//!   holds where one is free; the most live at once is counted (`command_queues_live_most`), and
//!   an unshared queue that found none free (`unshared_queues_sharing_a_vk_queue`).
//! - **Untimed devices.** A device that answers GPU times from the host (`gpu_times_from_host`)
//!   does not share, and counts it (`queue_sharing_untimed`) where it has two `VkQueue`s of one
//!   family to share: a queue is known long only by its command buffers' timestamps.
//! - **Isolation.** A compositor's small command buffers, committed at 4,800 a second and never
//!   waited for, beside an application's command buffers of about 30 ms each: once the
//!   application's queue went long, the compositor's 99th percentile over 2 s stays under 5 ms
//!   with at most one stall over 10 ms; on an unshared queue, from the start, with none.
//! - **Other queues' work held back** (`unshared_hold.rs`). On a `VkQueue` an unshared queue
//!   shares with another queue, three long command buffers of that queue committed at once:
//!   under `UnsharedIsolationPolicy::Bounded` the second is held until the first is nearly done,
//!   so the unshared queue's command buffer committed meanwhile runs before it; under `Off` all
//!   three go at once and it runs after the third. A pass that finds the second let go and the
//!   unshared queue's command buffer ready together (the scheduler's thread late, held off by the
//!   test) sends the unshared one first. Every device, a `VkQueue` shared on purpose (`PerQueue`,
//!   the round robin wrapped).
//!
//! Where the device does not share (`PerQueue`, its check found no idling, or it has no two
//! `VkQueue`s of one family: MoltenVK, RADV, lavapipe) each test asserts that no queue moves, and
//! the isolation tests measure nothing: the runtime moves no queue there, so the latencies are the
//! driver's own (MoltenVK on an M3 Max, the same workload on two `VkQueue`s of their own: a 99th
//! percentile of 57 to 60 ms). A device test: compiled only under the validation crate's GPU
//! safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::Scratch;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions, WorkCounts,
};
use metal2vulkan::device::{DevicePolicy, QueueSharingPolicy, UnsharedIsolationPolicy};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest any wait of these tests lasts.
const WAIT: Duration = Duration::from_secs(20);

/// How long an isolation run waits past its start before it records: a queue's backlog behind a
/// long command buffer drains within about 1 ms of its end.
const SETTLING: Duration = Duration::from_millis(50);

/// The bytes one fill of a long command buffer writes.
const FILL_BYTES: u64 = 64 << 20;

/// The options of the device the cfg selects under `queue_sharing`, its shader cache in `cache`.
fn options(cache: &Scratch, queue_sharing: QueueSharingPolicy) -> DeviceOptions {
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
    options
}

/// That device, its check settled.
fn device(cache: &Scratch, queue_sharing: QueueSharingPolicy) -> Device {
    isolated(cache, queue_sharing, UnsharedIsolationPolicy::Bounded)
}

/// That device under `isolation`, its check settled.
fn isolated(
    cache: &Scratch,
    queue_sharing: QueueSharingPolicy,
    isolation: UnsharedIsolationPolicy,
) -> Device {
    let mut options = options(cache, queue_sharing);
    options.policy.unshared_isolation = isolation;
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
        assert_eq!(device.work_counts().queue_sharing_untimed, 0, "{policy:?}");
        test(&device, shares);
        if !shares {
            assert_eq!(device.work_counts().queue_moves, 0, "{policy:?}");
        }
    }
}

/// Commit `count` command buffers of one 1 MiB fill each on `queue`, not waited for, the first
/// scheduled: its `VkQueue` busy as this returns, its oldest batch young unless the host was slow
/// to come back. A command buffer committed and not yet submitted is in no `VkQueue`'s flight,
/// which is all a queue's choice looks at (`queue_share.rs`).
fn burst(queue: &CommandQueue, target: &Buffer, count: usize) -> Vec<CommandBuffer> {
    let command_buffers: Vec<CommandBuffer> = (0..count)
        .map(|_| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(target, 0..(1 << 20), 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect();
    if let Some(first) = command_buffers.first() {
        first.wait_until_scheduled().unwrap();
    }
    command_buffers
}

/// Whether a queue's choice made between `before` and `after`, which moved it nowhere, stayed as
/// the sharing rule says (`queue_share.rs`): it found no other `VkQueue` with batches in flight,
/// or passed over the busy ones for their oldest batches' age, or a queue went long. On a busy
/// host or beside another process's GPU work every choice can stay so.
fn stayed_by_the_rule(before: &WorkCounts, after: &WorkCounts) -> bool {
    after.picks_with_none_busy > before.picks_with_none_busy
        || after.joins_withheld_from_old_heads > before.joins_withheld_from_old_heads
        || after.queues_gone_long > 0
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
        // The idle points where the queue stayed though the rule called for a move.
        let mut unexplained = Vec::new();
        for step in 0..STEPS {
            // Every fourth step one of the other queues is busy, in turn; the rest the queue
            // makes its next command buffer with its last one in flight, so with no move.
            let idle = step % 4 == 0;
            let other = &busy[step / 4 % 2];
            if idle {
                kept.extend(burst(other, &scratch, 24));
            }
            let (at, before) = (queue.vk_queue_sharing().0, device.work_counts());
            let command_buffer = queue.command_buffer().unwrap();
            seen.push(queue.vk_queue_sharing().0);
            // A stay where the busy queue's `VkQueue` is its own is the rule's too.
            if idle
                && shares
                && seen[step] == at
                && other.vk_queue_sharing().0 != at
                && !stayed_by_the_rule(&before, &device.work_counts())
            {
                unexplained.push(step);
            }
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
        let counts = device.work_counts();
        println!(
            "VkQueues {seen:?}: {moved} moves; choices with none busy {}, old heads passed over \
             {}, queues gone long {}",
            counts.picks_with_none_busy,
            counts.joins_withheld_from_old_heads,
            counts.queues_gone_long
        );
        match shares {
            // Where the device shares, the queue moved at every idle point where the rule called
            // for a move: where another `VkQueue` had batches in flight whose oldest was young.
            true => {
                assert!(
                    unexplained.is_empty(),
                    "the queue stayed where the device shares, at idle steps {unexplained:?}"
                );
                assert!(moved == 0 || counts.queue_moves > 0);
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
        // The rounds where the waiter stayed though the rule called for a join.
        let mut unexplained = Vec::new();
        for round in 1..=8u64 {
            let busy = burst(&signaller, &scratch, 24);
            let before = device.work_counts();
            let waiting = waiter.command_buffer().unwrap();
            let shared = waiter.vk_queue_sharing().0 == signaller.vk_queue_sharing().0;
            if shares && !shared && !stayed_by_the_rule(&before, &device.work_counts()) {
                unexplained.push(round);
            }
            joined |= shared;
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
        println!("the waiter joined the signaller's VkQueue: {joined}");
        if shares {
            assert!(
                unexplained.is_empty(),
                "the waiter stayed off the signaller's busy VkQueue where the device shares, in \
                 rounds {unexplained:?}"
            );
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
        unsafe { pinned.with_vk_queue(|_, _, _| {}) };
        let presenting = device.new_command_queue_for_presenting().unwrap();
        let free = device.new_command_queue().unwrap();
        let (pinned_at, presenting_at) =
            (pinned.vk_queue_sharing().0, presenting.vk_queue_sharing().0);
        let mut free_moved = false;
        for _ in 0..8 {
            for queue in [&pinned, &presenting, &free] {
                let kept = burst(&busy, &scratch, 24);
                let before = queue.vk_queue_sharing().0;
                wait_all(burst(queue, &scratch, 1));
                free_moved |= std::ptr::eq(queue, &free) && queue.vk_queue_sharing().0 != before;
                wait_all(kept);
            }
        }
        assert_eq!(
            pinned.vk_queue_sharing().0,
            pinned_at,
            "a pinned queue moved"
        );
        assert_eq!(
            presenting.vk_queue_sharing().0,
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
        let home = unshared.vk_queue_sharing().0;
        let others: Vec<CommandQueue> = (0..3)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        // Each queue after the unshared one took a `VkQueue` no unshared queue holds where the
        // round robin has another (`Device::new_command_queue_unshared`), so none is on its
        // `VkQueue` but on a device of one round-robin queue, where each is.
        let round_robin = device.shared().round_robin();
        // A choice withholds a join from the unshared queue's `VkQueue` only while that
        // `VkQueue` is busy and its oldest batch in flight young (`queue_share.rs`), as a choice
        // right after a burst finds it, unless the host was slow to come back from the burst
        // (its batches complete or seen complete long before): where the device shares, the
        // bursts go on past the 16 until a choice withheld one, for at most `WAIT`.
        let began = Instant::now();
        let mut bursts = 0;
        while bursts < 16
            || (shares
                && device.work_counts().joins_withheld_from_unshared == 0
                && began.elapsed() < WAIT)
        {
            bursts += 1;
            let kept = burst(&unshared, &scratch, 24);
            for (after, queue) in others.iter().enumerate() {
                let made = queue.command_buffer().unwrap();
                assert_eq!(
                    queue.vk_queue_sharing().0 == home,
                    round_robin == 1,
                    "queue {after} after the unshared one, of a round robin of {round_robin} \
                     VkQueues (shares: {shares}): on the unshared queue's VkQueue"
                );
                drop(made);
            }
            wait_all(kept);
        }
        assert_eq!(
            unshared.vk_queue_sharing().0,
            home,
            "the unshared queue moved"
        );
        let counts = device.work_counts();
        println!(
            "{bursts} bursts: {} choices withheld a join from the unshared queue's VkQueue, {} \
             found no other VkQueue busy",
            counts.joins_withheld_from_unshared, counts.picks_with_none_busy
        );
        if shares {
            assert!(counts.joins_withheld_from_unshared > 0);
            // Every round robin's `VkQueue` holds a queue, so the next unshared queue shares
            // one with a queue, which leaves at its next idle point.
            let filled: Vec<CommandQueue> = (0..round_robin)
                .map(|_| device.new_command_queue().unwrap())
                .collect();
            let second = device.new_command_queue_unshared(64).unwrap();
            let there: Vec<&CommandQueue> = (filled.iter().chain(&others))
                .filter(|queue| queue.vk_queue_sharing().0 == second.vk_queue_sharing().0)
                .collect();
            assert!(
                !there.is_empty(),
                "the second unshared queue shares a VkQueue"
            );
            let before = device.work_counts().queue_moves_off_unshared;
            for queue in &there {
                drop(queue.command_buffer().unwrap());
                assert_ne!(queue.vk_queue_sharing().0, second.vk_queue_sharing().0);
            }
            assert_eq!(
                device.work_counts().queue_moves_off_unshared - before,
                there.len() as u64
            );
        }
    });
}

/// A placement test's device under each configuration: `Adaptive` (sharing or not, as its check
/// found), `PerQueue`, and `Adaptive` answering GPU times from the host (untimed, `Unchecked`),
/// each with a device of its own, its check settled.
fn each_placement(leg: &str, test: impl Fn(&Device)) {
    let configurations = [
        (QueueSharingPolicy::Adaptive, false),
        (QueueSharingPolicy::PerQueue, false),
        (QueueSharingPolicy::Adaptive, true),
    ];
    for (policy, untimed) in configurations {
        let scratch = Scratch::new(&format!("queue-place-{leg}-{policy:?}-{untimed}"));
        let mut options = options(&scratch, policy);
        options.gpu_times_from_host = untimed;
        let device = Device::system_default_with(&options).unwrap();
        assert!(device.settle_queue_switch_check(WAIT), "the check settles");
        println!(
            "PLACEMENT {leg} {policy:?}, untimed {untimed}, round robin {}: {}",
            device.shared().round_robin(),
            device.queue_switch_check_report()
        );
        assert!(
            !untimed || !device.shares_queues(),
            "an untimed device shares"
        );
        test(&device);
    }
}

/// A deterministic sequence's next value of `seed`, below `below`.
fn next(seed: &mut u64, below: u64) -> u64 {
    *seed = (seed.wrapping_mul(6_364_136_223_846_793_005)).wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) % below
}

/// A new command queue (`unshared`, an unshared one) of `device`, asserted to take, of the
/// round robin's `VkQueue`s as they stood just before, the one the fewest live queues hold of
/// those no unshared queue holds (a command queue), or one no queue holds where one is free (an
/// unshared queue).
fn placed(device: &Device, unshared: bool) -> CommandQueue {
    let before = device.round_robin_held();
    let queue = match unshared {
        true => device.new_command_queue_unshared(64).unwrap(),
        false => device.new_command_queue().unwrap(),
    };
    let at = queue.vk_queue_sharing().0;
    let free = || (before.iter()).filter(|&&(_, held_by_unshared)| !held_by_unshared);
    match unshared {
        true => {
            let empty = free().any(|&(homes, _)| homes == 0);
            assert!(
                !empty || before[at] == (0, false),
                "an unshared queue took VkQueue {at} of {before:?}, where one was free"
            );
        }
        false => {
            if let Some(least) = free().map(|&(homes, _)| homes).min() {
                assert_eq!(
                    before[at],
                    (least, false),
                    "a queue took VkQueue {at} of {before:?}"
                );
            }
        }
    }
    queue
}

/// Under every configuration, over a churn shaped like a guest's (an unshared queue and three
/// kept for the whole run, then four times the round robin's `VkQueue`s and 40 more made one at a
/// time, each dropped after 1 to `round_robin / 2` more makes): each queue takes the `VkQueue` the
/// fewest live queues hold of those no unshared queue holds, the unshared one an empty one, so no
/// queue shares while fewer are live than `VkQueue`s; a second unshared queue, with a `VkQueue`
/// still free, takes it, and one with none free shares, counted.
#[test]
fn over_a_churn_queues_take_the_least_held_vk_queue_and_unshared_ones_an_empty_one() {
    each_placement("churn", |device| {
        let round_robin = device.shared().round_robin();
        let before = device.work_counts();
        let unshared = placed(device, true);
        let kept: Vec<CommandQueue> = (0..3).map(|_| placed(device, false)).collect();
        let lifetime = (round_robin as u64 / 2).max(1);
        let mut seed = 0x2545_f491_4f6c_dd1d;
        let mut live: Vec<(CommandQueue, usize)> = Vec::new();
        let makes = 4 * round_robin + 40;
        for made in 0..makes {
            live.retain(|(_, until)| *until > made);
            let queue = placed(device, false);
            let until = made + 1 + next(&mut seed, lifetime) as usize;
            live.push((queue, until));
        }
        let home = unshared.vk_queue_sharing().0;
        let shared_home = (kept.iter().chain(live.iter().map(|(queue, _)| queue)))
            .filter(|queue| queue.vk_queue_sharing().0 == home)
            .count();
        let counts = device.work_counts();
        println!(
            "PLACEMENT churn: {makes} made, at most {} live, {shared_home} beside the unshared \
             queue; {} past the round robin",
            counts.command_queues_live_most,
            counts.queues_sharing_a_vk_queue - before.queues_sharing_a_vk_queue
        );
        assert_eq!(shared_home > 0, round_robin == 1);
        // The unshared queue, the kept ones and one of the churn's at least.
        assert!(counts.command_queues_live_most >= 5);
        drop(live);
        // A second unshared queue: on a `VkQueue` no queue holds where one is free, else beside
        // another, counted, never the first unshared queue's but on a device of one.
        let free = (device.round_robin_held().iter()).any(|&held| held == (0, false));
        let sharing = || device.work_counts().unshared_queues_sharing_a_vk_queue;
        let first = sharing();
        let second = placed(device, true);
        assert!(second.vk_queue_sharing().0 != home || round_robin == 1);
        assert_eq!(sharing() - first, u64::from(!free));
    });
}

/// What one isolation run measured: the compositor's latencies in milliseconds.
struct Isolation {
    latencies: Vec<f64>,
    /// Each queue's changes of `VkQueue` or long mark (`CommandQueue::vk_queue_sharing`) over the
    /// run, in words.
    moves: String,
    /// The device's queues marked long by the run's end (`WorkCounts::queues_gone_long`).
    gone_long: u64,
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
/// waited for: the compositor's latencies over 2 s from [`SETTLING`] after `ready` answers
/// (bounded), so that what queued behind a long command buffer before then has completed.
///
/// Where `waiting`, each long command buffer signals an event's next value at its end, and a
/// third queue commits, once each long one is committed, a command buffer of one small fill that
/// waits for that value, each waited for. A plain event (`MTLEvent`), not a shared one: no host
/// call sets its value, so the scheduler submits its wait once the signal is submitted, chained
/// on the waiter's `VkQueue` (`event_timeline.rs`), so it waits on the GPU, its own GPU work
/// short, where a shared event's wait is held on the host until the value is reached.
fn isolation(
    device: &Device,
    compositor: CommandQueue,
    waiting: bool,
    ready: impl Fn(&Device) -> bool,
) -> Isolation {
    let app = device.new_command_queue().unwrap();
    let big = device
        .new_buffer(FILL_BYTES, ResourceOptions::private())
        .unwrap();
    let fills = long_fills(&app, &big);
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let began = Instant::now();
    let event = device.new_event().unwrap();
    // The last value a committed long command buffer signals.
    let signalled = Arc::new(AtomicU64::new(0));
    let application = {
        let (stop, event, signalled) = (Arc::clone(&stop), event.clone(), Arc::clone(&signalled));
        std::thread::spawn(move || {
            let mut seen = Seen::default();
            let mut value = 0;
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = long(&app, &big, fills);
                if waiting {
                    value += 1;
                    command_buffer
                        .encode_signal_event((&event).into(), value)
                        .unwrap();
                }
                seen.note(&app, began);
                command_buffer.commit().unwrap();
                signalled.store(value, Ordering::SeqCst);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
            seen.0
        })
    };
    let waiter = waiting.then(|| {
        let (stop, signalled) = (Arc::clone(&stop), Arc::clone(&signalled));
        let (queue, scratch) = (
            device.new_command_queue().unwrap(),
            device.new_buffer(4096, ResourceOptions::private()).unwrap(),
        );
        std::thread::spawn(move || {
            let mut seen = Seen::default();
            let mut waited = 0;
            while !stop.load(Ordering::Relaxed) {
                let value = signalled.load(Ordering::SeqCst);
                if value == waited {
                    std::thread::sleep(Duration::from_micros(100));
                    continue;
                }
                let command_buffer = queue.command_buffer().unwrap();
                seen.note(&queue, began);
                command_buffer
                    .encode_wait_for_event((&event).into(), value)
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&scratch, 0..256, 1).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                waited = value;
            }
            seen.0
        })
    });
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
            let mut seen = Seen::default();
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = compositor.command_buffer().unwrap();
                seen.note(&compositor, began);
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
            seen.0
        })
    };
    let waiting = Instant::now();
    while !ready(device) && waiting.elapsed() < WAIT {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(ready(device), "the run did not reach its measured part");
    std::thread::sleep(SETTLING);
    recording.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(2));
    recording.store(false, Ordering::Relaxed);
    stop.store(true, Ordering::Relaxed);
    let counts = device.work_counts();
    let waiter = match waiter {
        Some(thread) => format!("waiter {}; ", thread.join().unwrap()),
        None => String::new(),
    };
    let moves = format!(
        "compositor {}; application {}; {waiter}moves {} joins {} gone long {} settles {}",
        composing.join().unwrap(),
        application.join().unwrap(),
        counts.queue_moves,
        counts.queue_joins,
        counts.queues_gone_long,
        counts.long_queue_settles
    );
    let latencies = latencies.lock().unwrap().clone();
    assert!(
        latencies.len() > 1000,
        "{} compositor samples",
        latencies.len()
    );
    Isolation {
        latencies,
        moves,
        gone_long: counts.queues_gone_long,
    }
}

/// A queue's changes of `VkQueue` or long mark, each at its command buffer's count and the
/// milliseconds into the run, in words.
#[derive(Default)]
struct Seen(String, u64, Option<(usize, bool)>);

impl Seen {
    fn note(&mut self, queue: &CommandQueue, began: Instant) {
        let now = queue.vk_queue_sharing();
        if self.2 != Some(now) {
            let long = if now.1 { " long" } else { "" };
            let at = began.elapsed().as_secs_f64() * 1e3;
            self.0 += &format!("#{} {at:.1} ms VkQueue {}{long}, ", self.1, now.0);
            self.2 = Some(now);
        }
        self.1 += 1;
    }
}

#[test]
fn a_device_that_answers_gpu_times_from_the_host_does_not_share_and_counts_it() {
    let scratch = Scratch::new("queue-share-untimed");
    let mut options = options(&scratch, QueueSharingPolicy::Adaptive);
    options.gpu_times_from_host = true;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_queue_switch_check(WAIT), "the check settles");
    let counts = device.work_counts();
    let could = crate::queue_switch_check::pair(&device).is_some();
    println!(
        "{}; could share: {could}",
        device.queue_switch_check_report()
    );
    assert!(!device.shares_queues());
    assert_eq!(counts.queue_sharing_untimed, u64::from(could));
    assert_eq!(counts.queue_switch_checks_run, 0);
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(1 << 20, ResourceOptions::private())
        .unwrap();
    wait_all(burst(&queue, &target, 8));
    assert_eq!(device.work_counts().queue_moves, 0);
}

#[test]
fn a_compositor_beside_a_long_queue_keeps_its_latency_once_that_queue_went_long() {
    each_policy("isolation", |device, shares| {
        if !shares {
            return;
        }
        let compositor = device.new_command_queue().unwrap();
        let ready = |device: &Device| device.work_counts().queues_gone_long >= 1;
        let run = isolation(device, compositor, false, ready);
        println!(
            "p99 {:.2} ms, {} stalls of {} samples; {}",
            run.p99(),
            run.stalls(),
            run.latencies.len(),
            run.moves
        );
        assert!(run.p99() < 5.0, "p99 {:.2} ms", run.p99());
        assert!(run.stalls() <= 1, "{} stalls", run.stalls());
    });
}

#[test]
fn an_unshared_compositor_beside_a_long_queue_never_stalls() {
    each_policy("isolation-unshared", |device, shares| {
        if !shares {
            return;
        }
        let compositor = device.new_command_queue_unshared(64).unwrap();
        let run = isolation(device, compositor, false, |_| true);
        println!(
            "p99 {:.2} ms, {} stalls of {} samples; {}",
            run.p99(),
            run.stalls(),
            run.latencies.len(),
            run.moves
        );
        assert!(run.p99() < 5.0, "p99 {:.2} ms", run.p99());
        assert_eq!(run.stalls(), 0, "the unshared compositor stalled");
    });
}

/// A compositor beside a queue whose command buffers each wait on the GPU for the end of another
/// queue's long command buffer: a wait the long mark must see, as it sees a long command buffer of
/// the queue's own, since the compositor's work behind it on a shared `VkQueue` waits as long.
/// The waiting queue goes long as well as the application's, and once the application's did,
/// the compositor's 99th percentile over 2 s stays under 5 ms with at most one stall over 10 ms,
/// as beside the long queue alone.
#[test]
fn a_compositor_beside_a_queue_waiting_on_the_gpu_for_long_work_keeps_its_latency() {
    each_policy("isolation-gpu-wait", |device, shares| {
        if !shares {
            return;
        }
        let compositor = device.new_command_queue().unwrap();
        let ready = |device: &Device| device.work_counts().queues_gone_long >= 1;
        let run = isolation(device, compositor, true, ready);
        println!(
            "p99 {:.2} ms, {} stalls of {} samples; {}",
            run.p99(),
            run.stalls(),
            run.latencies.len(),
            run.moves
        );
        assert!(
            run.gone_long >= 2,
            "{} queue(s) went long: the waiting one did not",
            run.gone_long
        );
        assert!(run.p99() < 5.0, "p99 {:.2} ms", run.p99());
        assert!(run.stalls() <= 1, "{} stalls", run.stalls());
    });
}

/// What one run of [`behind`] saw: the counts it added, the long command buffer's GPU work left
/// at the short one's submission (seconds of the host's clock, its GPU end less the short one's
/// kernel end), and the host's time from before the short one's commit to both completed.
struct Behind {
    before: WorkCounts,
    after: WorkCounts,
    left: f64,
    elapsed: Duration,
}

/// A long command buffer of `fills` on `long_queue`, scheduled, then a short one on
/// `short_queue`, both waited for.
fn behind(
    device: &Device,
    (long_queue, big, fills): (&CommandQueue, &Buffer, usize),
    short_queue: &CommandQueue,
    small: &Buffer,
) -> Behind {
    let before = device.work_counts();
    let long = long(long_queue, big, fills);
    long.commit().unwrap();
    long.wait_until_scheduled().unwrap();
    let began = Instant::now();
    let short = short_queue.command_buffer().unwrap();
    let mut blit = short.blit_command_encoder().unwrap();
    blit.fill_buffer(small, 0..256, 1).unwrap();
    blit.end_encoding().unwrap();
    short.commit().unwrap();
    short.wait_until_completed_for(WAIT).unwrap();
    long.wait_until_completed_for(WAIT).unwrap();
    Behind {
        before,
        after: device.work_counts(),
        left: long.gpu_end_time().unwrap() - short.kernel_end_time(),
        elapsed: began.elapsed(),
    }
}

impl Behind {
    /// Its wait, the longest of the tally's (`longest`), held to the host's view: at least the
    /// long command buffer's work left at its submission, less 1 ms for the clocks' calibration,
    /// at most the host's time around it.
    fn wait(&self, longest: impl Fn(&WorkCounts) -> u64) -> u64 {
        let wait = longest(&self.after);
        println!(
            "waited {:.2} ms behind {:.2} ms of the long one's work left, {:.2} ms around",
            wait as f64 / 1e6,
            self.left * 1e3,
            self.elapsed.as_secs_f64() * 1e3
        );
        assert!(
            self.left > 0.0,
            "the long one ended before the short one went"
        );
        assert!(wait as f64 >= (self.left - 0.001) * 1e9, "{wait} ns");
        assert!(wait as u128 <= self.elapsed.as_nanos(), "{wait} ns");
        wait
    }
}

/// A queue `make` makes on `queue`'s `VkQueue` (`PerQueue`: no queue moves): the queues it makes
/// are kept until one takes it, as a new queue takes the `VkQueue` the fewest live queues hold,
/// once the others hold as many.
fn beside(queue: &CommandQueue, make: &dyn Fn() -> CommandQueue) -> CommandQueue {
    let at = queue.vk_queue_sharing().0;
    let mut kept = Vec::new();
    for _ in 0..64 {
        let made = make();
        if made.vk_queue_sharing().0 == at {
            return made;
        }
        kept.push(made);
    }
    panic!(
        "no queue on the other queue's VkQueue of {} made",
        kept.len()
    )
}

/// Two command queues on one `VkQueue` (`PerQueue`: no queue moves), a long command buffer of
/// about 30 ms of GPU time, then a short one (`queue_wait.rs`). From the same queue the short one
/// waits behind nothing of another's; from another queue it counts one wait, as long as the long
/// one's work left at its submission at least and the host's time around it at most, over 16 and
/// 33 ms as it is; from an unshared queue, one in the unshared tallies too. Under
/// `UnsharedIsolationPolicy::Off`: the short ones go at once, behind the whole long one.
#[test]
fn a_command_buffer_behind_another_queue_s_long_one_counts_its_wait() {
    let scratch = Scratch::new("queue-wait");
    let off = UnsharedIsolationPolicy::Off;
    let device = isolated(&scratch, QueueSharingPolicy::PerQueue, off);
    let long_queue = device.new_command_queue().unwrap();
    let other = beside(&long_queue, &|| device.new_command_queue().unwrap());
    let unshared = beside(&long_queue, &|| {
        device.new_command_queue_unshared(64).unwrap()
    });
    let big = device
        .new_buffer(FILL_BYTES, ResourceOptions::private())
        .unwrap();
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let fills = long_fills(&long_queue, &big);
    let longs = (&long_queue, &big, fills);
    let tally = |counts: &WorkCounts| {
        [
            counts.command_buffers_behind_other_queues,
            counts.command_buffers_behind_other_queues_over_16ms,
            counts.command_buffers_behind_other_queues_over_33ms,
            counts.wait_behind_other_queues_longest_nanos,
        ]
    };
    let unshared_tally = |counts: &WorkCounts| {
        [
            counts.unshared_command_buffers_behind_other_queues,
            counts.unshared_command_buffers_behind_other_queues_over_16ms,
            counts.unshared_command_buffers_behind_other_queues_over_33ms,
            counts.unshared_wait_behind_other_queues_longest_nanos,
        ]
    };
    let added = |run: &Behind, of: &dyn Fn(&WorkCounts) -> [u64; 4]| {
        let (before, after) = (of(&run.before), of(&run.after));
        [
            after[0] - before[0],
            after[1] - before[1],
            after[2] - before[2],
        ]
    };
    let over = |wait: u64| {
        [
            1,
            u64::from(wait > 16_000_000),
            u64::from(wait > 33_000_000),
        ]
    };

    let same = behind(&device, longs, &long_queue, &small);
    assert_eq!(tally(&same.after), [0; 4], "behind its own queue's");
    assert_eq!(unshared_tally(&same.after), [0; 4]);

    let run = behind(&device, longs, &other, &small);
    let wait = run.wait(|counts| counts.wait_behind_other_queues_longest_nanos);
    assert_eq!(added(&run, &tally), over(wait));
    assert_eq!(
        unshared_tally(&run.after),
        [0; 4],
        "not an unshared queue's"
    );

    let run = behind(&device, longs, &unshared, &small);
    let wait = run.wait(|counts| counts.unshared_wait_behind_other_queues_longest_nanos);
    assert_eq!(added(&run, &unshared_tally), over(wait));
    assert_eq!(
        added(&run, &tally),
        over(wait),
        "counted among every queue's"
    );
    let longest = run.after.wait_behind_other_queues_longest_nanos;
    assert_eq!(
        longest,
        wait.max(run.before.wait_behind_other_queues_longest_nanos)
    );
}

/// The held-work tests' device under `isolation` (`PerQueue`), an application's queue, an
/// unshared queue on its `VkQueue`, and the fills of a long command buffer of about 30 ms.
struct HeldWork {
    device: Device,
    app: CommandQueue,
    unshared: CommandQueue,
    big: Buffer,
    small: Buffer,
    fills: usize,
}

/// One run of [`HeldWork`]: the long command buffers' and the unshared one's GPU starts and ends,
/// in seconds of the host's clock.
struct HeldRun {
    longs: [(f64, f64); 3],
    short: (f64, f64),
    /// The host time the unshared one's commit returned.
    committed: f64,
    /// The second long one's kernel start time: when a pass took it to submit.
    let_go: f64,
    /// The looks that held a head back for the unshared queue
    /// (`WorkCounts::heads_held_for_unshared_isolation`).
    held: u64,
}

impl HeldWork {
    fn new(scratch: &Scratch, isolation: UnsharedIsolationPolicy) -> HeldWork {
        let device = isolated(scratch, QueueSharingPolicy::PerQueue, isolation);
        let app = device.new_command_queue().unwrap();
        let unshared = beside(&app, &|| device.new_command_queue_unshared(64).unwrap());
        let big = device
            .new_buffer(FILL_BYTES, ResourceOptions::private())
            .unwrap();
        let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let fills = long_fills(&app, &big);
        HeldWork {
            device,
            app,
            unshared,
            big,
            small,
            fills,
        }
    }

    /// A long command buffer run alone (its queue's expected run a long one's), then three
    /// committed at once, the first scheduled, then the unshared queue's small one committed.
    /// Where `late`, it is committed while the test holds the scheduler's lock
    /// (`Scheduler::under_lock`), kept until the first long one completed: the pass that takes it
    /// in finds the second let go too, as the pass of a scheduler thread the host ran late does.
    fn run(&self, late: bool) -> HeldRun {
        let warm = long(&self.app, &self.big, self.fills);
        warm.commit().unwrap();
        warm.wait_until_completed_for(WAIT).unwrap();
        let before = self.device.work_counts();
        let longs: Vec<CommandBuffer> = (0..3)
            .map(|_| {
                let command_buffer = long(&self.app, &self.big, self.fills);
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        longs[0].wait_until_scheduled().unwrap();
        let short = self.unshared.command_buffer().unwrap();
        let mut blit = short.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.small, 0..256, 1).unwrap();
        blit.end_encoding().unwrap();
        let (short, committed) = match late {
            false => {
                short.commit().unwrap();
                (short, crate::host_time())
            }
            true => self.commit_late(short, &longs[0]),
        };
        short.wait_until_completed_for(WAIT).unwrap();
        for command_buffer in &longs {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        let after = self.device.work_counts();
        let times = |command_buffer: &CommandBuffer| {
            let start = command_buffer.gpu_start_time().unwrap();
            (start, command_buffer.gpu_end_time().unwrap())
        };
        HeldRun {
            longs: [times(&longs[0]), times(&longs[1]), times(&longs[2])],
            short: times(&short),
            committed,
            let_go: longs[1].kernel_start_time(),
            held: after.heads_held_for_unshared_isolation
                - before.heads_held_for_unshared_isolation,
        }
    }

    /// Commit `short` on another thread while this one holds the scheduler's lock, kept until
    /// `first` completed (a commit takes no lock of the state's, `scheduler/inbox.rs`, nor does
    /// a completion turn); with the host time the commit returned.
    fn commit_late(&self, short: CommandBuffer, first: &CommandBuffer) -> (CommandBuffer, f64) {
        let (committer, returned) = self.device.shared().scheduler.under_lock(|| {
            let (done, returns) = mpsc::channel();
            let committer = std::thread::spawn(move || {
                short.commit().unwrap();
                // Gone where the test stopped waiting for it.
                let _ = done.send(crate::host_time());
                short
            });
            let returned = returns.recv_timeout(WAIT);
            let began = Instant::now();
            while first.status() != CommandBufferStatus::Completed && began.elapsed() < WAIT {
                std::thread::sleep(Duration::from_millis(1));
            }
            (committer, returned)
        });
        let committed = returned.expect("the commit returned while the lock was held");
        assert_eq!(
            first.status(),
            CommandBufferStatus::Completed,
            "the first long one completed while the lock was held"
        );
        (committer.join().unwrap(), committed)
    }
}

impl HeldRun {
    fn print(&self, label: &str) {
        let [(s0, e0), (s1, e1), (s2, e2)] = self.longs;
        let (ss, se) = self.short;
        println!(
            "{label}: long ones {:.1}-{:.1}, {:.1}-{:.1}, {:.1}-{:.1} ms, the unshared one \
             {:.1}-{:.1} ms, committed at {:.1} ms, the second let go at {:.1} ms; GPU idle \
             between the long ones {:.2} and {:.2} ms; {} looks held",
            0.0,
            (e0 - s0) * 1e3,
            (s1 - s0) * 1e3,
            (e1 - s0) * 1e3,
            (s2 - s0) * 1e3,
            (e2 - s0) * 1e3,
            (ss - s0) * 1e3,
            (se - s0) * 1e3,
            (self.committed - s0) * 1e3,
            (self.let_go - s0) * 1e3,
            // The unshared one ends the first gap only where it ran in it.
            (s1 - if se <= s1 { e0.max(se) } else { e0 }) * 1e3,
            (s2 - e1) * 1e3,
            self.held,
        );
    }

    /// The unshared one ran before the second long one.
    fn ahead_of_the_second(&self) -> bool {
        self.short.1 <= self.longs[1].0 + 0.001
    }
}

/// The runs [`an_unshared_queue_s_command_buffer_goes_ahead_of_other_queues_held_work`] makes
/// under `Bounded` for one whose unshared command buffer was committed before the second long
/// one was let go.
const HELD_RUNS: usize = 5;

/// Under `UnsharedIsolationPolicy::Bounded` (the default), three long command buffers of about
/// 30 ms committed at once on a `VkQueue` an unshared queue shares with theirs go one at a time,
/// each held until the one ahead is nearly done (`unshared_hold.rs`), so the unshared queue's
/// small command buffer committed once the first was scheduled runs before the second; under
/// `Off` all three go at once, and it runs after the third. Their queue's expected run is a long
/// one's first: four run one at a time beforehand.
///
/// The second is let go about 28 ms after the first went: a run whose unshared command buffer was
/// committed only after that (the test's thread late on a loaded host: inside a whole suite's run
/// on an RTX 5080 Laptop, this test took 35 s where it takes 2 s alone) shows nothing of the
/// hold, and is made again, up to [`HELD_RUNS`] runs.
#[test]
fn an_unshared_queue_s_command_buffer_goes_ahead_of_other_queues_held_work() {
    for isolation in [
        UnsharedIsolationPolicy::Bounded,
        UnsharedIsolationPolicy::Off,
    ] {
        let scratch = Scratch::new(&format!("unshared-hold-{isolation:?}"));
        let work = HeldWork::new(&scratch, isolation);
        match isolation {
            UnsharedIsolationPolicy::Bounded => {
                let run = (0..HELD_RUNS)
                    .map(|_| work.run(false))
                    .inspect(|run| run.print("Bounded"))
                    .find(|run| run.committed < run.let_go)
                    .expect("a run committed the unshared one before the second was let go");
                assert!(
                    run.ahead_of_the_second(),
                    "the unshared one ran before the second"
                );
                assert!(run.held > 0, "a long one was held");
            }
            _ => {
                let run = work.run(false);
                run.print("Off");
                assert!(
                    run.short.0 >= run.longs[2].1 - 0.001,
                    "the unshared one ran after the third"
                );
                assert_eq!(run.held, 0, "none held");
            }
        }
    }
}

/// Under `Bounded`, a pass that finds the unshared queue's command buffer ready together with the
/// second long one, which the hold let go meanwhile, sends the unshared one first
/// (`scheduler/order.rs`): its commit made while the test holds the scheduler's lock, kept until
/// the first long one completed, as a scheduler thread the host ran late takes both in one pass.
/// It runs before the second.
#[test]
fn a_late_pass_sends_an_unshared_queue_s_command_buffer_before_work_let_go_with_it() {
    let scratch = Scratch::new("unshared-hold-late");
    let work = HeldWork::new(&scratch, UnsharedIsolationPolicy::Bounded);
    let run = work.run(true);
    run.print("Bounded, a late pass");
    assert!(
        run.committed < run.let_go,
        "the second was let go after the commit"
    );
    assert!(
        run.ahead_of_the_second(),
        "the unshared one ran before the second"
    );
    assert!(run.held > 0, "a long one was held");
}
