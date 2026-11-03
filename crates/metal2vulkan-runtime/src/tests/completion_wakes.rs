//! The queue waiters and the scheduler's thread sleep with no timeout while idle
//! (`command_buffer/completion.rs`, "An idle waiter sleeps until it is woken"; `scheduler/mod.rs`,
//! "Idle, it sleeps until it is woken"), so every change they wait for has to wake them.
//!
//! - **Idle, nothing wakes.** Over [`IDLE`] with nothing committed, no waiter parks again
//!   (`WorkCounts::completion_waiter_parks`), none of their sleeps ends at a slice, and the
//!   scheduler's thread neither sleeps again nor wakes at its slice
//!   (`WorkCounts::scheduler_sleeps`, `WorkCounts::scheduler_sleep_slices`).
//! - **Every change wakes its waiter.** With every park made unbounded, a pending batch or not
//!   (`completion::park_unbounded`), work over several queues that reaches each change a parked
//!   waiter waits for (a batch watched on a parked or a waiting queue, a turn's take that leaves a
//!   queue a batch, a command buffer needing no GPU work alone and behind work, a head held for a
//!   host signal) completes within a bound, and so does the device's drop (a stop): a change that
//!   woke no waiter it should would leave its command buffer, or the drop, waiting for good.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::completion;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, EventRef,
    ResourceOptions,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// How long the idle device is watched.
const IDLE: Duration = Duration::from_millis(350);

/// How long a command buffer, or the device's drop, may take before the test reads it as a wake
/// that never came.
const BOUND: Duration = Duration::from_secs(20);

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
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

/// A command buffer of `queue` that fills `target` whole with `value`.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..target.length(), value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Wait for `command_buffer` within [`BOUND`]: a wake that never came fails here.
fn settle(command_buffer: &CommandBuffer, what: &str) {
    command_buffer
        .wait_until_completed_for(BOUND)
        .unwrap_or_else(|_| panic!("{what}: not completed within {BOUND:?}"));
    assert_eq!(command_buffer.error(), None, "{what}");
}

/// An idle device's queue waiters and scheduler stay asleep: after work on two queues and an
/// empty command buffer, nothing parks or wakes at a slice for [`IDLE`].
#[test]
fn an_idle_device_s_waiters_and_scheduler_stay_asleep() {
    let device = device();
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    let target = device
        .new_buffer(1 << 16, ResourceOptions::private())
        .unwrap();
    for (at, queue) in queues.iter().enumerate() {
        let filled = fill(queue, &target, at as u8);
        filled.commit().unwrap();
        settle(&filled, "a fill");
        let empty = queue.command_buffer().unwrap();
        empty.commit().unwrap();
        settle(&empty, "an empty command buffer");
    }
    // What the work set going (a handler thread's turn, a spare pool's begin) settles first.
    std::thread::sleep(Duration::from_millis(50));
    let before = device.work_counts();
    std::thread::sleep(IDLE);
    let after = device.work_counts();
    let moved = |name: &str, field: fn(&crate::WorkCounts) -> u64| {
        let moved = field(&after) - field(&before);
        println!("{name}: {moved} over {IDLE:?} idle");
        moved
    };
    let parks = moved("completion_waiter_parks", |c| c.completion_waiter_parks);
    let timed_out = moved("completion_waiter_parks_timed_out", |c| {
        c.completion_waiter_parks_timed_out
    });
    let sleeps = moved("scheduler_sleeps", |c| c.scheduler_sleeps);
    let slices = moved("scheduler_sleep_slices", |c| c.scheduler_sleep_slices);
    assert_eq!(
        (parks, timed_out),
        (0, 0),
        "an idle queue waiter woke and parked again"
    );
    assert_eq!(
        (sleeps, slices),
        (0, 0),
        "the idle scheduler's thread woke and slept again"
    );
}

/// With every waiter's park unbounded, work that reaches each change a parked waiter waits for
/// completes within [`BOUND`], and so does the device's drop.
#[test]
fn with_parks_unbounded_every_change_still_wakes_its_waiter() {
    completion::park_unbounded();
    let device = device();
    let queues: Vec<CommandQueue> = (0..3)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let small = device
        .new_buffer(1 << 16, ResourceOptions::private())
        .unwrap();
    let large = device
        .new_buffer(16 << 20, ResourceOptions::private())
        .unwrap();
    let event = device.new_shared_event().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    for round in 0..12u8 {
        // Parked waiters: an idle gap.
        if round % 3 == 0 {
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut committed = Vec::new();
        // Back to back on every queue, each with a completed handler: batches watched on parked
        // and on waiting queues, turns that take one queue's batches and leave it more.
        for (at, queue) in queues.iter().enumerate() {
            for value in 0..3u8 {
                let target = if value == 0 { &large } else { &small };
                let command_buffer = fill(queue, target, round ^ value ^ at as u8);
                let counted = Arc::clone(&completed);
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        counted.fetch_add(1, Ordering::SeqCst);
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
                committed.push(command_buffer);
            }
            // Behind the fills: watched at a value not reached yet.
            let behind = queue.command_buffer().unwrap();
            behind.commit().unwrap();
            committed.push(behind);
        }
        // A head held for a host signal while the others complete, then let go.
        let held = queues[round as usize % queues.len()]
            .command_buffer()
            .unwrap();
        let value = u64::from(round) + 1;
        held.encode_wait_for_event(EventRef::Shared(&event), value)
            .unwrap();
        held.commit().unwrap();
        for (at, command_buffer) in committed.iter().enumerate() {
            settle(
                command_buffer,
                &format!("round {round}, command buffer {at}"),
            );
        }
        event.set_signaled_value(value);
        settle(&held, &format!("round {round}, the held one"));
        // Alone, on an idle queue: the turn owed to the thread that took it.
        let alone = queues[0].command_buffer().unwrap();
        alone.commit().unwrap();
        settle(&alone, &format!("round {round}, an empty one alone"));
    }
    assert_eq!(completed.load(Ordering::SeqCst), 12 * 3 * 3);
    // The stop: every waiter and the scheduler's thread wake and end.
    let (dropped, done) = mpsc::channel();
    drop(queues);
    std::thread::spawn(move || {
        drop((small, large, event, device));
        let _ = dropped.send(());
    });
    assert!(
        done.recv_timeout(BOUND).is_ok(),
        "the device's drop did not end within {BOUND:?}"
    );
}
