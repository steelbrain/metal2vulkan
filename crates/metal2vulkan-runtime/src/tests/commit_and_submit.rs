//! `CommandBuffer::commit_and_submit` (`scheduler/mod.rs`, "A committing thread may submit for
//! itself"): where no other pass is under way, the committing thread submits the ready heads
//! itself, in commit order, and runs their submission handlers inside the call, before the command
//! buffer reads `Scheduled` (`WorkCounts::scheduler_passes_at_commit`,
//! `WorkCounts::batches_submitted_at_commit`); submission handlers keep their submission order
//! whichever thread runs them; and a head with a submission check is checked by the committing
//! thread too, just before it submits it (`WorkCounts::batches_checked_at_commit`): a check that
//! answers a code fails its submission as on the scheduler's thread, and checks keep commit order
//! whichever thread runs them. A head another queue holds for its check holds back only the
//! commits that wait behind it: one of a third queue is checked and submitted at its commit.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device,
    DeviceKind, DeviceOptions, EncoderKind, EventRef, ResourceOptions, Submission,
};
use std::num::NonZeroU64;
use std::ops::Range;
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

/// What ran for a command buffer, in order, and on which thread.
type Events = Arc<Mutex<Vec<(&'static str, ThreadId)>>>;

/// Note `name` and this thread into `events`.
fn note(events: &Events, name: &'static str) {
    events
        .lock()
        .unwrap()
        .push((name, std::thread::current().id()));
}

/// On an idle device, a command buffer with a submission check committed with
/// `commit_and_submit` is checked and submitted by the committing thread: its check runs on it,
/// inside the call, then its submission handler, in that order, before the call returns, and it
/// completes. Counted: a batch checked at commit, and submitted at commit, for each.
#[test]
fn a_checked_command_buffer_is_checked_and_submitted_on_the_committing_thread() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let caller = std::thread::current().id();
    let before = device.work_counts();
    let mut here = 0;
    for round in 0..ROUNDS {
        let command_buffer = fill(&queue, &target, round as u8);
        let events: Events = Arc::default();
        let (checks, handled) = (Arc::clone(&events), Arc::clone(&events));
        command_buffer
            .set_submission_check(Arc::new(move |_: &Submission<'_>| {
                note(&checks, "check");
                None
            }))
            .unwrap();
        command_buffer
            .add_submission_handler(Box::new(move |_: &CommandBuffer| note(&handled, "handler")))
            .unwrap();
        command_buffer.commit_and_submit().unwrap();
        let at_return = events.lock().unwrap().len();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let events = events.lock().unwrap();
        let names: Vec<&str> = events.iter().map(|&(name, _)| name).collect();
        assert_eq!(names, ["check", "handler"], "round {round}");
        if events.iter().all(|&(_, thread)| thread == caller) {
            assert_eq!(
                at_return, 2,
                "round {round}: the check or the handler ran on the caller after the call returned"
            );
            here += 1;
        }
    }
    let after = device.work_counts();
    let checked = after.batches_checked_at_commit - before.batches_checked_at_commit;
    let batches = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!(
        "{here} of {ROUNDS} checked and submitted on the caller; batches checked at commit \
         {checked}, submitted {batches}"
    );
    // A pass of the scheduler's thread under way at the commit (a completion's nudge) takes it
    // in instead: rarely, on an idle device.
    assert!(
        here >= ROUNDS / 2,
        "{here} of {ROUNDS} command buffers checked and submitted on the committing thread"
    );
    assert!(
        checked >= here && batches >= here,
        "checked {checked}, submitted {batches}"
    );
}

/// The initial bytes of the buffers the refusal test fills, which a fill that runs overwrites.
const POISON: u8 = 0xee;

/// What a command buffer whose second submission its check failed left: its status and error,
/// the status its completed handler read, the submissions the check saw, each target's first
/// byte, and its event's value.
#[derive(Debug, PartialEq)]
struct Failed {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    handler_read: Option<CommandBufferStatus>,
    seen: Vec<(EncoderKind, Range<usize>)>,
    bytes: [u8; 2],
    signaled: u64,
}

/// The first byte of a `Shared` buffer, which no command uses now.
fn first_byte(buffer: &Buffer) -> u8 {
    let contents = buffer.contents().expect("a Shared buffer's contents");
    // SAFETY: the buffer is `Shared`, a byte long at least, and no command uses it now.
    unsafe { *contents.as_ptr() }
}

/// Commit, with `commit_and_submit` where `at_commit`, else `commit`, a command buffer of
/// `queue` of two submissions, two fills split by an event signal, whose check fails the second
/// with code 8; wait for it: what it left, and the threads its check ran on.
fn fail_the_second(
    device: &Device,
    queue: &CommandQueue,
    at_commit: bool,
) -> (Failed, Vec<ThreadId>) {
    let targets = [0, 1].map(|_| {
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer is `Shared` and 256 bytes long, and no command uses it yet.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), POISON, 256) };
        buffer
    });
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for (at, target) in targets.iter().enumerate() {
        if at == 1 {
            command_buffer
                .encode_signal_event(EventRef::from(&event), 1)
                .unwrap();
        }
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(target, 0..target.length(), 0x11 * (at as u8 + 1))
            .unwrap();
        blit.end_encoding().unwrap();
    }
    let seen: Arc<Mutex<Vec<(EncoderKind, Range<usize>, ThreadId)>>> = Arc::default();
    let checks = Arc::clone(&seen);
    command_buffer
        .set_submission_check(Arc::new(move |submission: &Submission<'_>| {
            let me = std::thread::current().id();
            checks
                .lock()
                .unwrap()
                .push((submission.kind(), submission.encoders(), me));
            (submission.encoders().start == 1).then(|| NonZeroU64::new(8).unwrap())
        }))
        .unwrap();
    let read: Arc<Mutex<Option<CommandBufferStatus>>> = Arc::default();
    let reads = Arc::clone(&read);
    command_buffer
        .add_completed_handler(Box::new(move |buffer: &CommandBuffer| {
            *reads.lock().unwrap() = Some(buffer.status());
        }))
        .unwrap();
    match at_commit {
        true => command_buffer.commit_and_submit().unwrap(),
        false => command_buffer.commit().unwrap(),
    }
    command_buffer.wait_until_completed().unwrap();
    // Its completed handler runs before its waiters return.
    let seen = seen.lock().unwrap();
    let failed = Failed {
        status: command_buffer.status(),
        error: command_buffer.error(),
        handler_read: *read.lock().unwrap(),
        seen: (seen.iter())
            .map(|(kind, encoders, _)| (*kind, encoders.clone()))
            .collect(),
        bytes: targets.each_ref().map(first_byte),
        signaled: event.signaled_value(),
    };
    (failed, seen.iter().map(|&(_, _, thread)| thread).collect())
}

/// A check that answers a code fails its submission alike whether the scheduler's thread runs it
/// (`commit`) or the committing thread (`commit_and_submit`): of two submissions split by an event
/// signal, the second's check fails it with code 8, so the first's fill runs and the second's does
/// not, the signal fires, the completed handler reads `Error`, and the command buffer completes
/// with `CommandBufferError::Injected(8)`; the check saw each submission once, in order. The
/// committing thread ran both checks in some rounds at least (`batches_checked_at_commit`).
#[test]
fn a_refusing_check_fails_its_submission_alike_at_commit_and_on_the_scheduler() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let caller = std::thread::current().id();
    let expected = Failed {
        status: CommandBufferStatus::Error,
        error: Some(CommandBufferError::Injected(8)),
        handler_read: Some(CommandBufferStatus::Error),
        seen: vec![(EncoderKind::Blit, 0..1), (EncoderKind::Blit, 1..2)],
        bytes: [0x11, POISON],
        signaled: 1,
    };
    let before = device.work_counts();
    let mut here = 0;
    for round in 0..ROUNDS / 4 {
        let (scheduled, threads) = fail_the_second(&device, &queue, false);
        assert_eq!(
            scheduled, expected,
            "round {round}, by the scheduler's thread"
        );
        assert!(
            threads.iter().all(|&thread| thread != caller),
            "round {round}: a check of `commit`'s ran on the caller"
        );
        let (committed, threads) = fail_the_second(&device, &queue, true);
        assert_eq!(committed, expected, "round {round}, at commit");
        if threads.iter().all(|&thread| thread == caller) {
            here += 1;
        }
    }
    let after = device.work_counts();
    let checked = after.batches_checked_at_commit - before.batches_checked_at_commit;
    println!(
        "{here} of {} checked on the caller; batches checked at commit {checked}",
        ROUNDS / 4
    );
    assert!(
        here >= ROUNDS / 8,
        "{here} command buffers checked on the committing thread"
    );
    assert!(checked >= 2 * here, "batches checked at commit {checked}");
}

/// Two threads commit 1000 command buffers with submission checks to one queue, each in turn
/// with `commit` and with `commit_and_submit`, their commits in one order (a lock held across
/// each call): every check, and every submission handler, runs in that order, each check once,
/// some on a committing thread and some on the scheduler's.
#[test]
fn checked_command_buffers_keep_commit_order_across_commit_and_commit_and_submit() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let next = Mutex::new(0u64);
    let checks: Arc<Mutex<Vec<(u64, ThreadId)>>> = Arc::default();
    let handled: Arc<Mutex<Vec<u64>>> = Arc::default();
    let before = device.work_counts();
    let spin = |length: Duration| {
        let started = Instant::now();
        while started.elapsed() < length {
            std::hint::spin_loop();
        }
    };
    let (committers, committed): (Vec<ThreadId>, Vec<Vec<CommandBuffer>>) =
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..2u64)
                .map(|thread| {
                    let (queue, target, next) = (&queue, &target, &next);
                    let (checks, handled) = (&checks, &handled);
                    scope.spawn(move || {
                        let mut committed = Vec::new();
                        for at in 0..PER_THREAD {
                            let command_buffer = fill(queue, target, at as u8);
                            let sequence = Arc::new(AtomicU64::new(u64::MAX));
                            let (read, checks) = (Arc::clone(&sequence), Arc::clone(checks));
                            command_buffer
                                .set_submission_check(Arc::new(move |_: &Submission<'_>| {
                                    let me = std::thread::current().id();
                                    // Long enough that another thread passes while it runs.
                                    spin(HANDLER);
                                    checks
                                        .lock()
                                        .unwrap()
                                        .push((read.load(Ordering::SeqCst), me));
                                    None
                                }))
                                .unwrap();
                            let (read, handled) = (Arc::clone(&sequence), Arc::clone(handled));
                            command_buffer
                                .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                                    // Long enough that a pass's handlers still run when the
                                    // next pass, on another thread, takes its own.
                                    spin(HANDLER);
                                    handled.lock().unwrap().push(read.load(Ordering::SeqCst));
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
    let total = 2 * PER_THREAD;
    let in_order = |order: &[u64], what: &str| {
        let first_out = order.iter().zip(0..).position(|(&ran, at)| ran != at);
        assert_eq!(
            first_out,
            None,
            "{what} out of commit order near {:?}",
            first_out.map(|at| &order[at.saturating_sub(3)..(at + 4).min(order.len())])
        );
        assert_eq!(order.len() as u64, total, "{what}");
    };
    let checks = checks.lock().unwrap();
    let order: Vec<u64> = checks.iter().map(|&(sequence, _)| sequence).collect();
    in_order(&order, "checks");
    in_order(&handled.lock().unwrap(), "submission handlers");
    let on_committer = checks
        .iter()
        .filter(|(_, thread)| committers.contains(thread))
        .count() as u64;
    let checked = after.batches_checked_at_commit - before.batches_checked_at_commit;
    let batches = after.batches_submitted_at_commit - before.batches_submitted_at_commit;
    println!(
        "{on_committer} of {total} checks on a committing thread; batches checked at commit \
         {checked}, submitted {batches}"
    );
    assert!(
        on_committer > 0 && on_committer < total,
        "{on_committer} of {total} checks ran on a committing thread"
    );
    assert!(checked >= on_committer && batches >= checked);
}

/// How long the held-check test waits for a check to start, and for its command buffers.
const BOUND: Duration = Duration::from_secs(20);

/// Give `command_buffer` a submission check and a submission handler that note `check` and
/// `submitted` into `events`.
fn noting(
    command_buffer: &CommandBuffer,
    events: &Events,
    (check, submitted): (&'static str, &'static str),
) {
    let (checks, handled) = (Arc::clone(events), Arc::clone(events));
    command_buffer
        .set_submission_check(Arc::new(move |_: &Submission<'_>| {
            note(&checks, check);
            None
        }))
        .unwrap();
    command_buffer
        .add_submission_handler(Box::new(move |_: &CommandBuffer| note(&handled, submitted)))
        .unwrap();
}

/// While the scheduler's thread runs the submission check of queue A's head, committed with
/// `commit` (the check held until the test lets it go), the heads of two other queues are
/// committed with `commit_and_submit`: B's fills A's buffer, so it waits behind A (its footprint
/// runs in commit order with A's, which keeps its place, `scheduler/order.rs`) and its check and
/// submission handler run on the scheduler's thread once A was submitted; C's fills another, so
/// the committing thread checks and submits it inside the call while A's check still runs
/// (`WorkCounts::batches_checked_at_commit`, `batches_submitted_at_commit`), as the scheduler's
/// own pass would. Submission handlers keep submission order: C's, A's, then B's. All complete.
#[test]
fn a_head_held_for_its_check_holds_back_only_the_commits_behind_it() {
    let device = device();
    let queues: Vec<CommandQueue> = (0..3)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let [shared, other] = [(); 2].map(|()| {
        device
            .new_buffer(1 << 16, ResourceOptions::private())
            .unwrap()
    });
    let caller = std::thread::current().id();
    let events: Events = Arc::default();
    let (entered, enters) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let held = fill(&queues[0], &shared, 1);
    let (checks, handled) = (Arc::clone(&events), Arc::clone(&events));
    let released = Mutex::new(released);
    held.set_submission_check(Arc::new(move |_: &Submission<'_>| {
        note(&checks, "a check");
        let _ = entered.send(());
        let _ = released.lock().unwrap().recv_timeout(BOUND);
        None
    }))
    .unwrap();
    held.add_submission_handler(Box::new(move |_: &CommandBuffer| {
        note(&handled, "a submitted")
    }))
    .unwrap();
    held.commit().unwrap();
    enters
        .recv_timeout(BOUND)
        .expect("the scheduler's thread runs A's check");
    let before = device.work_counts();
    let behind = fill(&queues[1], &shared, 2);
    noting(&behind, &events, ("b check", "b submitted"));
    behind.commit_and_submit().unwrap();
    let apart = fill(&queues[2], &other, 3);
    noting(&apart, &events, ("c check", "c submitted"));
    apart.commit_and_submit().unwrap();
    let while_held = events.lock().unwrap().clone();
    let after = device.work_counts();
    release.send(()).unwrap();
    for command_buffer in [&held, &behind, &apart] {
        command_buffer.wait_until_completed_for(BOUND).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let on_caller = |events: &[(&'static str, ThreadId)]| -> Vec<(&'static str, bool)> {
        (events.iter())
            .map(|&(name, thread)| (name, thread == caller))
            .collect()
    };
    assert_eq!(
        on_caller(&while_held[..]),
        [("a check", false), ("c check", true), ("c submitted", true)],
        "while A's check ran: C checked and submitted inside its call, B left"
    );
    let events = events.lock().unwrap().clone();
    assert_eq!(
        on_caller(&events[while_held.len()..]),
        [
            ("a submitted", false),
            ("b check", false),
            ("b submitted", false)
        ],
        "once A's check returned: A submitted, then B checked and submitted by the scheduler"
    );
    let moved = |field: fn(&crate::WorkCounts) -> u64| field(&after) - field(&before);
    assert_eq!(
        (
            moved(|c| c.batches_checked_at_commit),
            moved(|c| c.batches_submitted_at_commit),
        ),
        (1, 1),
        "C alone checked and submitted at commit"
    );
}
