//! A waker wakes the scheduler's thread once a sleep (`scheduler/mod.rs`, "Handed to the thread";
//! `Sleep::mark`): the first marks it awake as it notifies it, so a commit made before the thread
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
//! - **Woken by a committing thread's pass as by the batch's hand-off.** A `commit_and_submit`
//!   that submits its batch itself, after an idle gap, wakes the sleeping thread once after its
//!   submission where the batch may wait in a group, which every other pass leaves to the thread,
//!   and leaves it asleep elsewhere (`scheduler/mod.rs`, "A committing thread may submit for
//!   itself"): the thread sleeps and wakes as it does where such batches are its own.
//! - **Never behind a submission.** A commit that finds no pass under way wakes the thread without
//!   the lock a pass holds across its `vkQueueSubmit`: with the `VkQueue` held and that lock held,
//!   no pass counted, a commit from another thread returns at once.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::tests::SETTLE;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions,
    WorkCounts,
};
use metal2vulkan::device::CompletionPolicy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
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

/// Wait until the scheduler's thread sleeps (`Scheduler::asleep`), failing where it stays awake
/// for [`BOUND`]: what the last commit or completion set going is over, and no pass is under way.
fn asleep(device: &Device, what: &str) {
    let since = Instant::now();
    while !device.shared().scheduler.asleep() {
        assert!(
            since.elapsed() < BOUND,
            "{what}: the scheduler's thread stayed awake for {BOUND:?}"
        );
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// The wakings the thread's notifies and nudges count between `before` and `after`, and the
/// sleeps they may end: those the thread began between the two, and the one it may have been in
/// as `before` was read, counted as it began, before `before`. A waking of that one is counted in
/// the window, its sleep not: the wakings so exceed the sleeps begun by one wherever the thread
/// was asleep at `before` and awake at `after`, as each test's device, idle at `before`, leaves
/// it as its last command buffer completes.
fn wakings_and_sleeps(before: &WorkCounts, after: &WorkCounts) -> (u64, u64) {
    let wakings = |counts: &WorkCounts| {
        counts.scheduler_wakes_commit
            + counts.scheduler_wakes_completion
            + counts.scheduler_wakes_host
    };
    (
        wakings(after) - wakings(before),
        after.scheduler_sleeps - before.scheduler_sleeps + 1,
    )
}

/// Bursts of commits made faster than the scheduler's thread wakes, each once the thread sleeps
/// ([`asleep`]): the thread is notified at most once a sleep.
#[test]
fn a_burst_of_commits_wakes_the_sleeping_thread_once() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let completed = Arc::new(AtomicU64::new(0));
    let before = device.work_counts();
    for burst in 0..10u64 {
        asleep(&device, &format!("burst {burst}"));
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

/// Commits made with `commit_and_submit` alone, each once the thread sleeps ([`asleep`], not after
/// a time, which the last commit's waking and pass may outlast under load) and each submitted by
/// its committing thread: where the device's batches may wait in a group
/// (`CompletionPolicy::Grouped` and its check found the merge, `command_buffer/group.rs`), each
/// wakes the sleeping thread once, counted as a commit's waking; elsewhere none wakes it.
#[test]
fn a_commit_and_submit_wakes_the_sleeping_thread_only_for_a_batch_that_may_group() {
    const COMMITS: u64 = 10;
    let device = device();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    let groups = device.compile_device().policy().completion == CompletionPolicy::Grouped
        && device.submissions_merge();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let completed = Arc::new(AtomicU64::new(0));
    // The first use of the queue and the target, untimed and uncounted.
    counted_fill(&queue, &target, 0, &completed)
        .commit()
        .unwrap();
    settle(&completed, 1, "the first");
    let before = device.work_counts();
    for at in 1..=COMMITS {
        asleep(&device, &format!("commit {at}"));
        let command_buffer = counted_fill(&queue, &target, at as u8, &completed);
        command_buffer.commit_and_submit().unwrap();
        settle(&completed, at + 1, &format!("commit {at}"));
    }
    let after = device.work_counts();
    let by_commit = after.scheduler_wakes_commit - before.scheduler_wakes_commit;
    let at_commit = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!(
        "batches grouping {groups}: {by_commit} wakings by a commit, {at_commit} batches \
         submitted at commit, {COMMITS} commits"
    );
    assert_eq!(at_commit, COMMITS, "a batch was left to the thread");
    match groups {
        true => assert_eq!(
            by_commit, COMMITS,
            "a commit_and_submit of a batch that may group left the sleeping thread asleep"
        ),
        false => assert_eq!(by_commit, 0, "a commit_and_submit woke the sleeping thread"),
    }
}

/// A commit that finds no pass under way wakes the scheduler's thread without the lock a pass
/// holds across its `vkQueueSubmit` (`scheduler/inbox.rs`, "Posted, never passed"). A pass
/// counted only after the commit looked for one can take that lock before the commit's wake, and
/// hold it in a submission that waits for a held `VkQueue`. Here the `VkQueue` is held, and the
/// lock is held as such a pass holds it, with no pass counted: a commit from another thread
/// returns at once. Once both are let go, the thread's pass submits it and it completes.
#[test]
fn a_commit_returns_while_the_lock_a_submission_holds_is_held() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let completed = Arc::new(AtomicU64::new(0));
    let command_buffer = counted_fill(&queue, &target, 1, &completed);
    let (held, holding) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let holder = {
        let queue = queue.clone();
        std::thread::spawn(move || {
            // SAFETY: the closure keeps no handle and calls nothing of the runtime's; it waits
            // only for this test's release, past the commit's bound.
            unsafe {
                queue.with_vk_queue(|_, _, _| {
                    held.send(()).unwrap();
                    let _ = released.recv_timeout(2 * BOUND);
                })
            }
        })
    };
    holding.recv_timeout(BOUND).unwrap();
    // No pass is under way.
    asleep(&device, "before the lock");
    let (returned, committer) = device.shared().scheduler.under_lock(|| {
        let (done, returns) = mpsc::channel();
        let committer = std::thread::spawn(move || {
            command_buffer.commit().unwrap();
            // Gone where the test stopped waiting for it.
            let _ = done.send(());
            command_buffer
        });
        (returns.recv_timeout(BOUND).is_ok(), committer)
    });
    release.send(()).unwrap();
    holder.join().unwrap();
    let command_buffer = committer.join().unwrap();
    assert!(
        returned,
        "the commit waited for the lock a pass holds across its submission"
    );
    settle(&completed, 1, "the commit made while the lock was held");
    assert_eq!(command_buffer.error(), None);
}
