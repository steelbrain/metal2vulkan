//! A waker wakes the scheduler's thread once a sleep (`scheduler/mod.rs`, "Handed to the thread";
//! `State::mark`): the first marks it awake as it notifies it, so a commit made before the thread
//! ran makes no system call, and no commit's wake is lost for it.
//!
//! - **Once a sleep.** The notifies counted as wakings by a commit, a completion or a host signal
//!   (`WorkCounts::scheduler_wakes_commit`, `scheduler_wakes_completion`, `scheduler_wakes_host`)
//!   are at most the thread's sleeps (`WorkCounts::scheduler_sleeps`), on a device whose heads
//!   never wait for the GPU, after bursts of commits made faster than the thread wakes.
//! - **None lost.** Threads committing with idle gaps between some commits, with `commit` and
//!   `commit_and_submit` in turn, never waiting for a command buffer (a wait would submit for
//!   itself), see every command buffer complete within a bound: a commit whose wake was lost
//!   would leave its batches in the inbox of an idle thread that sleeps until woken.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions,
    WorkCounts,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long the command buffers may go without one completing before the test reads it as a wake
/// that never came.
const BOUND: Duration = Duration::from_secs(20);

/// The command buffers each burst commits back to back.
const BURST: u64 = 48;

/// The command buffers each stress thread commits.
const PER_THREAD: u64 = 400;

/// The command buffers a stress thread keeps uncompleted at most.
const IN_FLIGHT: u64 = 8;

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

/// A command buffer of `queue` that fills `target` with `value`, counting its completion into
/// `completed`.
fn counted_fill(
    queue: &CommandQueue,
    target: &Buffer,
    value: u8,
    completed: &Arc<AtomicU64>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..target.length(), value).unwrap();
    blit.end_encoding().unwrap();
    let counted = Arc::clone(completed);
    command_buffer
        .add_completed_handler(Box::new(move |_: &CommandBuffer| {
            counted.fetch_add(1, Ordering::SeqCst);
        }))
        .unwrap();
    command_buffer
}

/// Wait until `completed` reaches `count`, failing where it goes [`BOUND`] without moving.
fn settle(completed: &AtomicU64, count: u64, what: &str) {
    let (mut seen, mut since) = (completed.load(Ordering::SeqCst), Instant::now());
    while seen < count {
        std::thread::sleep(Duration::from_micros(50));
        let now = completed.load(Ordering::SeqCst);
        if now != seen {
            (seen, since) = (now, Instant::now());
        }
        assert!(
            since.elapsed() < BOUND,
            "{what}: {seen} of {count} completed, none for {BOUND:?}: a lost wake"
        );
    }
}

/// The wakings the thread's notifies and nudges count, and its sleeps, between `before` and
/// `after`.
fn wakings_and_sleeps(before: &WorkCounts, after: &WorkCounts) -> (u64, u64) {
    let wakings = |counts: &WorkCounts| {
        counts.scheduler_wakes_commit
            + counts.scheduler_wakes_completion
            + counts.scheduler_wakes_host
    };
    (
        wakings(after) - wakings(before),
        after.scheduler_sleeps - before.scheduler_sleeps,
    )
}

/// Bursts of commits made faster than the scheduler's thread wakes, each after an idle gap: the
/// thread is notified at most once a sleep.
#[test]
fn a_burst_of_commits_wakes_the_sleeping_thread_once() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let completed = Arc::new(AtomicU64::new(0));
    let before = device.work_counts();
    for burst in 0..10u64 {
        // The thread sleeps by now.
        std::thread::sleep(Duration::from_millis(5));
        let committed: Vec<CommandBuffer> = (0..BURST)
            .map(|at| counted_fill(&queue, &target, at as u8, &completed))
            .collect();
        for command_buffer in &committed {
            command_buffer.commit().unwrap();
        }
        settle(&completed, (burst + 1) * BURST, &format!("burst {burst}"));
    }
    let after = device.work_counts();
    let (wakings, sleeps) = wakings_and_sleeps(&before, &after);
    let by_commit = after.scheduler_wakes_commit - before.scheduler_wakes_commit;
    println!(
        "{wakings} wakings ({by_commit} by a commit) over {sleeps} sleeps, {} commits",
        10 * BURST
    );
    assert!(
        wakings <= sleeps,
        "{wakings} wakings over {sleeps} sleeps: a sleep was notified more than once"
    );
    assert!(by_commit > 0, "no commit woke the sleeping thread");
}

/// Four threads, each with a queue, commit with idle gaps of 0, 200 us and 2 ms between some
/// commits, with `commit` and `commit_and_submit` in turn, never waiting for a command buffer:
/// every one completes, none going [`BOUND`] without a completion, and the thread is notified at
/// most once a sleep.
#[test]
fn commits_with_idle_gaps_lose_no_wake() {
    let device = device();
    let before = device.work_counts();
    std::thread::scope(|scope| {
        for thread in 0..4u64 {
            let device = &device;
            scope.spawn(move || {
                let queue = device.new_command_queue().unwrap();
                let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
                let completed = Arc::new(AtomicU64::new(0));
                for at in 0..PER_THREAD {
                    if at % 7 == 3 {
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    if at % 31 == 11 + thread {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    if at >= IN_FLIGHT {
                        settle(&completed, at - IN_FLIGHT + 1, &format!("thread {thread}"));
                    }
                    let command_buffer = counted_fill(&queue, &target, at as u8, &completed);
                    match (at + thread) % 2 {
                        0 => command_buffer.commit().unwrap(),
                        _ => command_buffer.commit_and_submit().unwrap(),
                    }
                }
                settle(
                    &completed,
                    PER_THREAD,
                    &format!("thread {thread}, the last"),
                );
            });
        }
    });
    let after = device.work_counts();
    let (wakings, sleeps) = wakings_and_sleeps(&before, &after);
    println!(
        "{wakings} wakings over {sleeps} sleeps; {} sleeps ended at the slice; {} passes at commit",
        after.scheduler_sleep_slices - before.scheduler_sleep_slices,
        after.scheduler_passes_at_commit - before.scheduler_passes_at_commit
    );
    assert!(
        wakings <= sleeps,
        "{wakings} wakings over {sleeps} sleeps: a sleep was notified more than once"
    );
}
