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
//!   shares with another queue, three long command buffers of that queue committed at once, the
//!   unshared queue having committed just before (a compositor drawing): under
//!   `UnsharedIsolationPolicy::Bounded` the second is held until the first is nearly done, so the
//!   unshared queue's command buffer committed meanwhile runs before it; under `Off` all three go
//!   at once and it runs after the third. A pass that finds the second let go and the unshared
//!   queue's command buffer ready together (the scheduler's thread late, held off by the test)
//!   sends the unshared one first. With the unshared queue quiet for longer than
//!   `unshared_hold::ACTIVE` before, the second goes once the first has under 16 ms left (long
//!   before it ends, where 2 ms would hold it, `heads_let_go_unshared_idle`), the third is held,
//!   and the unshared queue's first command buffer, committed once the second went, runs before
//!   the third: behind the idle budget and one batch, not all three. Every device, a `VkQueue`
//!   shared on purpose (`PerQueue`, the round robin wrapped).
//! - **Commits not yet taken in** (`unshared_hold.rs`). A scheduler pass takes commits in at its
//!   start, the test's scheduler pausing after each submission: the unshared queue's command
//!   buffer committed during the pass that submits another queue's first goes before that
//!   queue's next, whose room came in that pass, and the unshared queue's next commit, posted
//!   after that room came, does not hold it.
//! - **Passes of other threads** (`unshared_hold.rs`). A thread about to block, and a committing
//!   thread (`commit_and_submit`), each pass for itself with the scheduler's thread held in a
//!   submission handler, and take in an application's command buffer beside the unshared queue's
//!   head, which that pass may not submit (a scheduled handler; a host step recorded, the steps'
//!   thread held at trace points): the unshared one goes first, as the scheduler's own pass sends
//!   it.
//!
//! Where the device does not share (`PerQueue`, its check found no idling, or it has no two
//! `VkQueue`s of one family: MoltenVK, RADV, lavapipe) each test asserts that no queue moves, and
//! the isolation tests measure nothing: the runtime moves no queue there, so the latencies are the
//! driver's own (MoltenVK on an M3 Max, the same workload on two `VkQueue`s of their own: a 99th
//! percentile of 57 to 60 ms). A device test: compiled only under the validation crate's GPU
//! safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::Scratch;
use super::dispatch_pending::{self, Buffers, Grid};
use super::waiter_submits::Held as HeldThread;
use crate::command_buffer::completion::trace;
use crate::tests::SETTLE;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions, Size, WorkCounts,
};
use metal2vulkan::device::{
    CompletionPolicy, DevicePolicy, PipelineWaitPolicy, QueueSharingPolicy, UnsharedIsolationPolicy,
};
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
    println!(
        "{queue_sharing:?}: {}; late rounds {}",
        device.queue_switch_check_report(),
        device.shared().queue_switch.late_rounds()
    );
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
        let free = device.round_robin_held().contains(&(0, false));
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
    /// The host time the unshared queue went quiet: its last commit, the unshared one's, as the
    /// runtime noted it (`SlotShare::unshared_commit`), and `unshared_hold::ACTIVE`.
    quiet_from: f64,
    /// The long ones' kernel start times: when a pass took each to submit.
    let_go: [f64; 3],
    /// The looks that held a head back for the unshared queue
    /// (`WorkCounts::heads_held_for_unshared_isolation`).
    held: u64,
    /// The looks that let a head go under the idle budget where the active one would hold it
    /// (`WorkCounts::heads_let_go_unshared_idle`).
    let_go_idle: u64,
    /// The application queue's expected run as the long ones were committed, in milliseconds.
    expected: f64,
}

/// What the unshared queue did before a [`HeldWork`] run's long command buffers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Compositor {
    /// It committed a small command buffer just before them, as a compositor drawing does: the
    /// 2 ms budget holds them back. Its command buffer of the run is committed once the first
    /// long one is scheduled.
    Active,
    /// It committed nothing for longer than `unshared_hold::ACTIVE` before them: the 16 ms
    /// budget holds them back. Its command buffer of the run, its first after the quiet stretch,
    /// is committed once the second long one is scheduled.
    Quiet,
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
        // The calibration's first command buffer ran far longer than one of `fills` (on lavapipe
        // its expected run stayed past the first's end at the next ones' commit), and a queue's
        // expected run halves toward a shorter one (`unshared_hold::Expected`): ten run alone take
        // it to within a thousandth of that excess of a long one's.
        for _ in 0..10 {
            let command_buffer = long(&app, &big, fills);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        HeldWork {
            device,
            app,
            unshared,
            big,
            small,
            fills,
        }
    }

    /// The unshared queue's small command buffer.
    fn small(&self) -> CommandBuffer {
        let command_buffer = self.unshared.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.small, 0..256, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    }

    /// A long command buffer run alone (its queue's expected run a long one's), then, as
    /// `compositor` says, three committed at once and the unshared queue's small one committed
    /// once the first (`Active`) or second (`Quiet`) is scheduled. Where `late`, it is committed
    /// while the test holds the scheduler's lock (`Scheduler::under_lock`), kept until the first
    /// long one completed: the pass that takes it in finds the second let go too, as the pass of
    /// a scheduler thread the host ran late does.
    fn run(&self, late: bool, compositor: Compositor) -> HeldRun {
        let warm = long(&self.app, &self.big, self.fills);
        warm.commit().unwrap();
        warm.wait_until_completed_for(WAIT).unwrap();
        let share = &self.device.shared().queues[self.unshared.vk_queue_sharing().0].share;
        let active =
            || crate::unshared_hold::active(share.unshared_commit(), crate::gpu_time::host_nanos());
        let drawing = match compositor {
            Compositor::Active => {
                let drawing = self.small();
                drawing.commit().unwrap();
                Some(drawing)
            }
            Compositor::Quiet => {
                let began = Instant::now();
                while active() {
                    assert!(
                        began.elapsed() < WAIT,
                        "the unshared queue never went quiet"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                None
            }
        };
        assert_eq!(active(), compositor == Compositor::Active);
        let expected = self.app.serving().expected().nanos().unwrap_or(0) as f64 / 1e6;
        let before = self.device.work_counts();
        let longs: Vec<CommandBuffer> = (0..3)
            .map(|_| {
                let command_buffer = long(&self.app, &self.big, self.fills);
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        match compositor {
            Compositor::Active => longs[0].wait_until_scheduled().unwrap(),
            Compositor::Quiet => longs[1].wait_until_scheduled().unwrap(),
        }
        let short = self.small();
        let (short, committed) = match late {
            false => {
                short.commit().unwrap();
                (short, crate::host_time())
            }
            true => self.commit_late(short, &longs[0]),
        };
        short.wait_until_completed_for(WAIT).unwrap();
        for command_buffer in longs.iter().chain(&drawing) {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        let after = self.device.work_counts();
        let quiet_from = (share.unshared_commit() + crate::unshared_hold::ACTIVE) as f64 / 1e9;
        let times = |command_buffer: &CommandBuffer| {
            let start = command_buffer.gpu_start_time().unwrap();
            (start, command_buffer.gpu_end_time().unwrap())
        };
        HeldRun {
            longs: [times(&longs[0]), times(&longs[1]), times(&longs[2])],
            short: times(&short),
            committed,
            quiet_from,
            let_go: [
                longs[0].kernel_start_time(),
                longs[1].kernel_start_time(),
                longs[2].kernel_start_time(),
            ],
            held: after.heads_held_for_unshared_isolation
                - before.heads_held_for_unshared_isolation,
            let_go_idle: after.heads_let_go_unshared_idle - before.heads_let_go_unshared_idle,
            expected,
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
             {:.1}-{:.1} ms, committed at {:.1} ms, the second let go at {:.1} ms, the third at \
             {:.1} ms; GPU idle between the long ones {:.2} and {:.2} ms; {} looks held, {} let \
             go under the idle budget; their queue's expected run {:.1} ms; the unshared queue \
             quiet from {:.1} ms",
            0.0,
            (e0 - s0) * 1e3,
            (s1 - s0) * 1e3,
            (e1 - s0) * 1e3,
            (s2 - s0) * 1e3,
            (e2 - s0) * 1e3,
            (ss - s0) * 1e3,
            (se - s0) * 1e3,
            (self.committed - s0) * 1e3,
            (self.let_go[1] - s0) * 1e3,
            (self.let_go[2] - s0) * 1e3,
            // The unshared one ends the first gap only where it ran in it.
            (s1 - if se <= s1 { e0.max(se) } else { e0 }) * 1e3,
            (s2 - e1) * 1e3,
            self.held,
            self.let_go_idle,
            self.expected,
            (self.quiet_from - s0) * 1e3,
        );
    }

    /// The unshared one ran before the second long one.
    fn ahead_of_the_second(&self) -> bool {
        self.short.1 <= self.longs[1].0 + 0.001
    }

    /// The unshared one ran before the third long one.
    fn ahead_of_the_third(&self) -> bool {
        self.short.1 <= self.longs[2].0 + 0.001
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
/// `Off` all three go at once, and it runs after the third (ends with or after it, on a device
/// that answers GPU times from the host, MoltenVK 1.4.1 on Apple's GPUs among them, whose GPU
/// start is the command buffer's kernel end). Their queue's expected run is a long
/// one's first: four run one at a time beforehand.
///
/// The second is let go about 28 ms after the first went: a run whose unshared command buffer was
/// committed only after that (the test's thread late on a loaded host: inside a whole suite's run
/// on an RTX 5080 Laptop, this test took 35 s where it takes 2 s alone) shows nothing of the
/// hold, and is made again, up to [`HELD_RUNS`] runs.
///
/// The 2 ms budget holds while the unshared queue's last commit is less than
/// `unshared_hold::ACTIVE` (50 ms) old; then the 16 ms one does (`unshared_hold.rs`, **Two
/// budgets**). Where the long ones ran far past 30 ms (lavapipe on a loaded host: 149 ms), the
/// unshared queue went quiet before the second was let go, and the 16 ms budget let it go, as it
/// lets a quiet queue's go: the 2 ms budget is checked on a second let go while the queue was
/// active (no more than 5 ms early, where the 16 ms budget lets it go about 16 ms early), and a
/// head let go under the 16 ms one only once it went quiet.
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
                    .map(|_| work.run(false, Compositor::Active))
                    .inspect(|run| run.print("Bounded"))
                    .find(|run| run.committed < run.let_go[1])
                    .expect("a run committed the unshared one before the second was let go");
                assert!(
                    run.ahead_of_the_second(),
                    "the unshared one ran before the second"
                );
                assert!(run.held > 0, "a long one was held");
                // The 2 ms budget lets the second go at the first's end, or 2 ms before its
                // expected end (its kernel start and its queue's expected run), whichever comes
                // first; the 16 ms one about 16 ms before.
                if run.let_go[1] < run.quiet_from {
                    let expected_end = run.let_go[0] + run.expected / 1e3;
                    let bound = run.longs[0].1.min(expected_end - 0.002);
                    assert!(
                        run.let_go[1] > bound - 0.005,
                        "the second was let go {:.2} ms before the first's end or 2 ms before its \
                         expected end, while the unshared queue was active: not under the 2 ms \
                         budget",
                        (bound - run.let_go[1]) * 1e3
                    );
                }
                assert!(
                    run.let_go_idle == 0 || run.let_go[1..].iter().any(|&at| at >= run.quiet_from),
                    "a head let go under the idle budget while the unshared queue was active"
                );
            }
            _ => {
                let run = work.run(false, Compositor::Active);
                run.print("Off");
                assert!(
                    run.short.1 >= run.longs[2].1 - 0.001,
                    "the unshared one ended with or after the third"
                );
                // A device that answers GPU times from the host starts it at its kernel end
                // (`gpu_time.rs`), not where the GPU took it up.
                if !work.device.shared().timing.answers_from_host() {
                    assert!(
                        run.short.0 >= run.longs[2].1 - 0.001,
                        "the unshared one ran after the third"
                    );
                }
                assert_eq!(run.held, 0, "none held");
                assert_eq!(run.let_go_idle, 0, "none let go under the idle budget");
            }
        }
    }
}

/// Under `Bounded`, beside an unshared queue quiet for longer than `unshared_hold::ACTIVE`, three
/// long command buffers of about 30 ms committed at once go under the 16 ms budget
/// (`unshared_hold::IDLE_BUDGET`): the second once the first has under 16 ms of its expected run
/// left (`unshared_hold::Flight::ahead`), so more than 7 ms before the first's expected end (its
/// submission and its queue's expected run), where the 2 ms budget lets it go about 2 ms before
/// (`heads_let_go_unshared_idle` counts it), and the third is held. The bound is on the expected
/// run, the host's view of the queue's recent runs: on a loaded host one run may end well before
/// it, so its own end bounds nothing. The unshared queue's first command buffer after its quiet
/// stretch, committed once the second went, runs behind the first and second and before the third:
/// behind at most the idle budget's work and one batch, not all three. A run whose unshared command
/// buffer was committed only after the third was let go (the test's thread late on a loaded host)
/// shows nothing of the bound, and is made again, up to [`HELD_RUNS`] runs.
#[test]
fn a_quiet_unshared_queue_s_first_command_buffer_waits_behind_the_idle_budget_and_one_batch() {
    let scratch = Scratch::new("unshared-hold-quiet");
    let work = HeldWork::new(&scratch, UnsharedIsolationPolicy::Bounded);
    let run = (0..HELD_RUNS)
        .map(|_| work.run(false, Compositor::Quiet))
        .inspect(|run| run.print("Bounded, quiet"))
        .find(|run| run.committed < run.let_go[2])
        .expect("a run committed the unshared one before the third was let go");
    let early = run.let_go[0] + run.expected / 1e3 - run.let_go[1];
    assert!(
        early > 0.007,
        "the second was let go {:.2} ms before the first's expected end, not under the idle \
         budget",
        early * 1e3
    );
    assert!(run.let_go_idle > 0, "a head let go under the idle budget");
    assert!(run.held > 0, "the idle budget held a long one");
    assert!(
        run.ahead_of_the_third(),
        "the unshared one ran before the third"
    );
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
    let run = work.run(true, Compositor::Active);
    run.print("Bounded, a late pass");
    assert!(
        run.committed < run.let_go[1],
        "the second was let go after the commit"
    );
    assert!(
        run.ahead_of_the_second(),
        "the unshared one ran before the second"
    );
    assert!(run.held > 0, "a long one was held");
}

/// How long the scheduler pauses after each `vkQueueSubmit` returned in
/// [`an_unshared_commit_posted_during_a_pass_goes_before_work_let_go_after_it`]: the test's thread
/// commits within it, on a loaded host too.
const PASS_PAUSE: Duration = Duration::from_millis(250);

/// Lets the scheduler go on without pausing as it drops, a failing test's unwinding too.
struct NoPause;

impl Drop for NoPause {
    fn drop(&mut self) {
        trace::pause_after_submit(None);
    }
}

/// Under `Bounded`, a scheduler pass takes commits in at its start (`scheduler/inbox.rs`), so an
/// unshared queue's commit posted during a pass waits for the next one. Another queue's batch
/// whose room comes during that pass is held until a pass took the commit in, and that pass sends
/// the unshared one first (`unshared_hold.rs`, "Commits not yet taken in"); a commit posted after
/// its room came never holds it. The scheduler pauses [`PASS_PAUSE`] each time a `vkQueueSubmit`
/// returned (`trace::pause_after_submit`, a scheduler thread the host preempted there), and the
/// test commits during those pauses: the application's `first` and `next` during its `earlier`
/// one's, the unshared queue's `drawn` during `first`'s, and its `later` during `drawn`'s. Each
/// batch goes alone (`CompletionPolicy::Exact`), so each takes its own pause. Without the hold,
/// `next` goes in the pass that submitted `first`, before `drawn`; a hold that waited for every
/// commit not taken in would hold `next` behind `later` too, and behind every commit of a
/// compositor that posts during each pass.
#[test]
fn an_unshared_commit_posted_during_a_pass_goes_before_work_let_go_after_it() {
    let scratch = Scratch::new("unshared-hold-untaken");
    let mut options = options(&scratch, QueueSharingPolicy::PerQueue);
    options.policy.unshared_isolation = UnsharedIsolationPolicy::Bounded;
    options.policy.completion = CompletionPolicy::Exact;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_queue_switch_check(WAIT), "the check settles");
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    println!("device: {}", device.name());
    let app = device.new_command_queue().unwrap();
    let unshared = beside(&app, &|| device.new_command_queue_unshared(64).unwrap());
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let fill = |queue: &CommandQueue, buffer: &Buffer| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, 0..256, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    };
    // The application's expected run a small fill's (`unshared_hold::Expected`), so its batches
    // find their room at once; and the unshared queue's first command buffer run.
    for (queue, buffer) in [(&app, &target), (&app, &target), (&unshared, &small)] {
        let command_buffer = fill(queue, buffer);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    let began = |command_buffer: &CommandBuffer, what: &str| {
        let start = Instant::now();
        while !command_buffer.submission_began() {
            assert!(start.elapsed() < WAIT, "{what}'s submission never began");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let before = device.work_counts();
    trace::pause_after_submit(Some(PASS_PAUSE));
    let no_pause = NoPause;
    let earlier = fill(&app, &target);
    earlier.commit().unwrap();
    began(&earlier, "earlier");
    let (first, next) = (fill(&app, &target), fill(&app, &target));
    first.commit().unwrap();
    next.commit().unwrap();
    began(&first, "first");
    let drawn = fill(&unshared, &small);
    drawn.commit().unwrap();
    began(&drawn, "drawn");
    let later = fill(&unshared, &small);
    later.commit().unwrap();
    let all = [&earlier, &first, &next, &drawn, &later];
    for command_buffer in all {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    drop(no_pause);
    let after = device.work_counts();
    let held = after.heads_held_for_unshared_commits - before.heads_held_for_unshared_commits;
    let base = earlier.kernel_start_time();
    let [_, first, next, drawn, later] =
        all.map(|command_buffer| (command_buffer.kernel_start_time() - base) * 1e3);
    println!(
        "kernel starts from the earlier one's: first {first:.1}, next {next:.1}, drawn \
         {drawn:.1}, later {later:.1} ms; {held} looks held for an unshared commit not taken in"
    );
    assert!(
        drawn < next,
        "the unshared one went before the batch whose room came after its commit"
    );
    assert!(
        next < later,
        "a commit posted after that room came did not hold the batch"
    );
    assert!(held > 0, "a look held it for the unshared commit");
}

/// The threads one threadgroup of the held dispatch runs in
/// [`a_committing_thread_s_pass_leaves_work_behind_an_unshared_head_with_its_step_recorded`].
const STEP_THREADS: u64 = 64;

/// A device for [`unshared_went_first`]'s tests: `PerQueue` (no queue moves), the unshared hold
/// `Bounded`, `completion` its completion policy, every call waiting for a pipeline still being
/// made until it is made (`PipelineWaitPolicy::Exact`: a bound passed would drop a dispatch whose
/// making a test holds); with an application's queue and an unshared queue on its `VkQueue`, each
/// queue's expected run a small fill's (`unshared_hold::Expected`), so the application's batches
/// find their room at once.
fn beside_unshared(
    scratch: &Scratch,
    completion: CompletionPolicy,
) -> (Device, CommandQueue, CommandQueue) {
    let mut options = options(scratch, QueueSharingPolicy::PerQueue);
    options.policy.unshared_isolation = UnsharedIsolationPolicy::Bounded;
    options.policy.completion = completion;
    options.policy.pipeline_waits = PipelineWaitPolicy::Exact;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_queue_switch_check(WAIT), "the check settles");
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    println!("device: {}", device.name());
    let app = device.new_command_queue().unwrap();
    let unshared = beside(&app, &|| device.new_command_queue_unshared(64).unwrap());
    let scratch = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    for queue in [&app, &app, &unshared] {
        let command_buffer = small_fill(queue, &scratch);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    (device, app, unshared)
}

/// A command buffer of `queue` that fills `buffer`'s first 256 bytes.
fn small_fill(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..256, 1).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Both completed, the unshared queue's command buffer `drawn` began before another queue's
/// `other` (their kernel starts, the scheduler's calls of their first submissions).
fn unshared_went_first(leg: &str, drawn: &CommandBuffer, other: &CommandBuffer) {
    for command_buffer in [drawn, other] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{leg}: {:?}",
            command_buffer.error()
        );
    }
    let ahead = (other.kernel_start_time() - drawn.kernel_start_time()) * 1e3;
    println!("{leg}: the other queue's command buffer began {ahead:.3} ms after the unshared one");
    assert!(
        ahead > 0.0,
        "{leg}: another queue's batch went ahead of the unshared queue's head, which the \
         scheduler's pass sends first"
    );
}

/// Lets go of the threads held at the steps' trace points as it drops, a failing test's
/// unwinding too.
struct StepsLetGo;

impl Drop for StepsLetGo {
    fn drop(&mut self) {
        trace::release(trace::Point::Step);
        trace::release(trace::Point::Stepped);
    }
}

/// A thread about to block passes for itself (`Scheduler::submit_before_waiting`) with the
/// scheduler's thread held in a submission handler (`waiter_submits::Held`), and takes in an
/// application's command buffer and, committed after it, the unshared queue's, which has a
/// scheduled handler: such a pass never submits that one, where the scheduler's own pass would
/// send it first (`order.rs`, "Heads put first"). So the waiting thread's pass leaves the
/// application's to the scheduler's thread too, whose next pass sends the unshared one first
/// (`unshared_hold.rs`, "Passes of other threads"). Under `CompletionPolicy::Exact`, where a
/// waiting thread's pass submits any queue's batch on every driver; without the rule it submits
/// the application's, ahead of the unshared one.
#[test]
fn a_waiting_thread_s_pass_leaves_work_behind_an_unshared_head_it_cannot_send() {
    let scratch = Scratch::new("unshared-hold-waiter");
    let (device, app, unshared) = beside_unshared(&scratch, CompletionPolicy::Exact);
    let [target, small, holder] =
        [(); 3].map(|()| (device.new_buffer(4096, ResourceOptions::private())).unwrap());
    let held = HeldThread::new(&app, &holder);
    let other = small_fill(&app, &target);
    other.commit().unwrap();
    let drawn = small_fill(&unshared, &small);
    drawn
        .add_scheduled_handler(Box::new(|_: &CommandBuffer| {}))
        .unwrap();
    drawn.commit().unwrap();
    let before = device.work_counts();
    // Completed or not, the pass was made.
    let _ = other.wait_until_completed_for(Duration::ZERO);
    let after = device.work_counts();
    drop(held);
    let passes = after.scheduler_passes_by_waiters - before.scheduler_passes_by_waiters;
    let submitted = after.batches_submitted_by_waiters - before.batches_submitted_by_waiters;
    println!("the waiting thread's passes {passes}, batches it submitted {submitted}");
    assert_eq!(passes, 1, "the waiting thread passed for the commits");
    unshared_went_first("a waiting thread's pass", &drawn, &other);
}

/// A committing thread's own pass (`CommandBuffer::commit_and_submit`) takes in an application's
/// command buffer while the unshared queue's head waits in the order with its host step recorded
/// (a dispatch whose pipeline's making the device's pool held, `pool::held`), which such a pass
/// never submits, where the scheduler's own pass would send it first: the steps' thread is held
/// before the step (`trace::Point::Step`) while the scheduler's thread is held in a submission
/// handler, its pass finding the dispatch stepping, then let go to record it. So the committing
/// thread's pass leaves the application's command buffer to the scheduler's thread too
/// (`unshared_hold.rs`, "Passes of other threads"). Under the default `CompletionPolicy`: where
/// the device's batches group, the committing thread's pass sends the application's batch alone,
/// its `VkQueue` running at most one earlier submission; without the rule it sends it ahead of
/// the dispatch.
#[test]
fn a_committing_thread_s_pass_leaves_work_behind_an_unshared_head_with_its_step_recorded() {
    let scratch = Scratch::new("unshared-hold-committer");
    let (device, app, unshared) = beside_unshared(&scratch, CompletionPolicy::default());
    let state = dispatch_pending::state(&device);
    let buffers = Buffers::new(&device);
    let [target, holder] =
        [(); 2].map(|()| (device.new_buffer(4096, ResourceOptions::private())).unwrap());
    let let_go = StepsLetGo;
    let making = crate::pipeline::pool::held::hold();
    let grid = Grid::One(Size::new(STEP_THREADS, 1, 1));
    let drawn = dispatch_pending::dispatch(&unshared, &state, &buffers, grid);
    drawn.commit().unwrap();
    trace::hold_at(trace::Point::Step);
    drop(making);
    assert!(
        trace::reached(trace::Point::Step, WAIT),
        "the dispatch's host step was never handed over"
    );
    let held = HeldThread::new(&app, &holder);
    trace::hold_at(trace::Point::Stepped);
    trace::release(trace::Point::Step);
    assert!(
        trace::reached(trace::Point::Stepped, WAIT),
        "the dispatch's host step was never recorded"
    );
    trace::release(trace::Point::Stepped);
    let before = device.work_counts();
    let other = small_fill(&app, &target);
    other.commit_and_submit().unwrap();
    let after = device.work_counts();
    drop(held);
    drop(let_go);
    let passes = after.scheduler_passes_at_commit - before.scheduler_passes_at_commit;
    let submitted = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!("the committing thread's passes {passes}, batches it submitted {submitted}");
    assert_eq!(passes, 1, "the committing thread passed for its commit");
    unshared_went_first("a committing thread's pass", &drawn, &other);
    assert!(
        buffers.written(STEP_THREADS),
        "the dispatch did not write what Metal's writes"
    );
}
