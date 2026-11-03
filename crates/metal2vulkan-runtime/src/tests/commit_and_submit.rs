//! `CommandBuffer::commit_and_submit` (`scheduler/mod.rs`, "A committing thread may submit for
//! itself"): where no other pass is under way, the committing thread submits the ready heads
//! committed first itself and runs their submission handlers inside the call, before the command
//! buffer reads `Scheduled` (`WorkCounts::scheduler_passes_at_commit`,
//! `WorkCounts::batches_submitted_at_commit`); submission handlers keep their submission order
//! whichever thread runs them; and a head with a submission check, with what is behind it on its
//! queue, is left to the scheduler's thread.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    ResourceOptions, Submission,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// The command buffers committed and waited for, one at a time.
const ROUNDS: u64 = 40;

/// The command buffers each of two threads commits in the order test.
const PER_THREAD: u64 = 500;

/// How long each submission handler of the order test runs.
const HANDLER: Duration = Duration::from_micros(30);

/// How long the scheduler's thread may take to reach a submission check.
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

/// Where a handler ran, and the status it read.
type Seen = Arc<Mutex<Option<(ThreadId, CommandBufferStatus)>>>;

/// A handler that records where it ran into `seen`.
fn recording(seen: &Seen) -> Box<dyn FnOnce(&CommandBuffer) + Send> {
    let seen = Arc::clone(seen);
    Box::new(move |buffer: &CommandBuffer| {
        *seen.lock().unwrap() = Some((std::thread::current().id(), buffer.status()));
    })
}

/// On an idle device, a command buffer committed with `commit_and_submit` is submitted by the
/// committing thread: its submission handler runs on it, inside the call, reading `Committed`,
/// and its scheduled handler runs on its queue's handler thread after it. Counted: a pass and a
/// batch at commit for each.
#[test]
fn a_submission_handler_runs_on_the_committing_thread_before_scheduled() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let caller = std::thread::current().id();
    let before = device.work_counts();
    let mut here = 0;
    for round in 0..ROUNDS {
        let command_buffer = fill(&queue, &target, round as u8);
        let (submitted, scheduled): (Seen, Seen) = Default::default();
        command_buffer
            .add_submission_handler(recording(&submitted))
            .unwrap();
        command_buffer
            .add_scheduled_handler(recording(&scheduled))
            .unwrap();
        command_buffer.commit_and_submit().unwrap();
        let at_return = *submitted.lock().unwrap();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let (thread, status) = submitted
            .lock()
            .unwrap()
            .expect("the submission handler ran");
        assert_eq!(status, CommandBufferStatus::Committed, "round {round}");
        let (scheduled_on, read) = scheduled
            .lock()
            .unwrap()
            .expect("the scheduled handler ran");
        assert_eq!(read, CommandBufferStatus::Scheduled, "round {round}");
        assert_ne!(
            scheduled_on, caller,
            "a scheduled handler ran on the committing thread"
        );
        if thread == caller {
            assert!(
                at_return.is_some(),
                "round {round}: the handler ran on the caller after the call returned"
            );
            here += 1;
        }
    }
    let after = device.work_counts();
    let passes = after.scheduler_passes_at_commit - before.scheduler_passes_at_commit;
    let batches = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!(
        "{here} of {ROUNDS} handlers on the caller; passes at commit {passes}, batches {batches}"
    );
    // A pass of the scheduler's thread under way at the commit (a completion's nudge) takes it
    // in instead: rarely, on an idle device.
    assert!(
        here >= ROUNDS / 2,
        "{here} of {ROUNDS} submission handlers ran on the committing thread"
    );
    assert!(
        passes >= here && batches >= here,
        "passes {passes}, batches {batches}"
    );
}

/// Two threads commit 1000 command buffers to one queue, each in turn with `commit` and with
/// `commit_and_submit`, their commits in one order (a lock held across each call): every
/// submission handler runs in that order, some on a committing thread and some on the
/// scheduler's.
#[test]
fn submission_handlers_keep_commit_order_across_commit_and_commit_and_submit() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let next = Mutex::new(0u64);
    let ran: Arc<Mutex<Vec<(u64, ThreadId)>>> = Arc::default();
    let before = device.work_counts();
    let (committers, committed): (Vec<ThreadId>, Vec<Vec<CommandBuffer>>) =
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..2u64)
                .map(|thread| {
                    let (queue, target, next, ran) = (&queue, &target, &next, &ran);
                    scope.spawn(move || {
                        let mut committed = Vec::new();
                        for at in 0..PER_THREAD {
                            let command_buffer = fill(queue, target, at as u8);
                            let sequence = Arc::new(AtomicU64::new(u64::MAX));
                            let (read, ran) = (Arc::clone(&sequence), Arc::clone(ran));
                            command_buffer
                                .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                                    let sequence = read.load(Ordering::SeqCst);
                                    let me = std::thread::current().id();
                                    // Long enough that a pass's handlers still run when the
                                    // next pass, on another thread, takes its own.
                                    let started = Instant::now();
                                    while started.elapsed() < HANDLER {
                                        std::hint::spin_loop();
                                    }
                                    ran.lock().unwrap().push((sequence, me));
                                }))
                                .unwrap();
                            let mut next = next.lock().unwrap();
                            sequence.store(*next, Ordering::SeqCst);
                            *next += 1;
                            match (at + thread) % 2 {
                                0 => command_buffer.commit_and_submit().unwrap(),
                                _ => command_buffer.commit().unwrap(),
                            }
                            drop(next);
                            committed.push(command_buffer);
                        }
                        (std::thread::current().id(), committed)
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .unzip()
        });
    for command_buffer in committed.iter().flatten() {
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let after = device.work_counts();
    let ran = ran.lock().unwrap();
    let total = 2 * PER_THREAD;
    assert_eq!(ran.len() as u64, total);
    let order: Vec<u64> = ran.iter().map(|&(sequence, _)| sequence).collect();
    let first_out = order.iter().zip(0..).position(|(&ran, at)| ran != at);
    assert_eq!(
        first_out,
        None,
        "submission handlers out of commit order near {:?}",
        first_out.map(|at| &order[at.saturating_sub(3)..(at + 4).min(order.len())])
    );
    let on_committer = ran
        .iter()
        .filter(|(_, thread)| committers.contains(thread))
        .count() as u64;
    let passes = after.scheduler_passes_at_commit - before.scheduler_passes_at_commit;
    let batches = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!(
        "{on_committer} of {total} on a committing thread; passes at commit {passes}, batches \
         {batches}"
    );
    assert!(
        on_committer > 0 && on_committer < total,
        "{on_committer} of {total} submission handlers ran on a committing thread"
    );
    assert!(passes > 0 && batches >= on_committer);
}

/// A command buffer with a submission check, committed with `commit_and_submit`, is left to the
/// scheduler's thread, which runs the check, and so is the one committed behind it on its queue
/// while the check runs (held until then): its submission handler runs there too, after the
/// check. The committing thread's passes submit neither.
#[test]
fn a_checked_head_and_what_is_behind_it_are_left_to_the_scheduler() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let caller = std::thread::current().id();
    for round in 0..ROUNDS as u8 {
        let before = device.work_counts();
        let events: Arc<Mutex<Vec<(&'static str, ThreadId)>>> = Arc::default();
        let checked = fill(&queue, &target, round);
        let checks = Arc::clone(&events);
        let (entered, checking) = mpsc::channel();
        let (open, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        checked
            .set_submission_check(Arc::new(move |_: &Submission<'_>| {
                let me = std::thread::current().id();
                checks.lock().unwrap().push(("check", me));
                let _ = entered.send(());
                // Until the test drops its end, once the one behind is committed.
                let _ = gate.lock().unwrap().recv();
                None
            }))
            .unwrap();
        let behind = fill(&queue, &target, round ^ 1);
        let handled = Arc::clone(&events);
        behind
            .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                let me = std::thread::current().id();
                handled.lock().unwrap().push(("handler", me));
            }))
            .unwrap();
        checked.commit_and_submit().unwrap();
        checking
            .recv_timeout(BOUND)
            .expect("the scheduler's thread checks the submission");
        behind.commit_and_submit().unwrap();
        drop(open);
        behind.wait_until_completed().unwrap();
        checked.wait_until_completed().unwrap();
        assert_eq!(checked.status(), CommandBufferStatus::Completed);
        assert_eq!(behind.status(), CommandBufferStatus::Completed);
        let after = device.work_counts();
        let events = events.lock().unwrap();
        let names: Vec<&str> = events.iter().map(|&(name, _)| name).collect();
        assert_eq!(names, ["check", "handler"], "round {round}");
        assert!(
            events.iter().all(|&(_, thread)| thread != caller),
            "round {round}: the committing thread ran {events:?}"
        );
        assert_eq!(
            after.batches_submitted_at_commit - before.batches_submitted_at_commit,
            0,
            "round {round}: a committing thread submitted a checked head or what is behind it"
        );
    }
}
