//! When a command buffer's submission handlers run (`scheduler/mod.rs`, **Submission handlers**;
//! `scheduler/inbox.rs`, "Who runs the after-work, and when"): never after a later pass of the
//! thread that took it. A pass of the scheduler's thread is held in a submission
//! (`trace::Point::FirstWork`) while another queue commits, so the thread's look after that pass
//! takes the second commit in, and each submission is followed by a pause
//! (`trace::pause_after_submit`): the first command buffer's submission handlers started before
//! the pause after the second's submission ended, where they once ran only after every pass the
//! commits kept the thread making. Under either `CompletionPolicy`: a batch the default `Grouped`
//! holds in a group is taken (its handlers taken with it) at the hold, which the first work's take
//! is reached at too.
//!
//! And where its transition to `Scheduled` waits for them (`CommandBufferShared::schedule`): only
//! once they ran does it take its place in its queue's context, so the completion of the queue's
//! command buffer before it never waits for them. A queue's waiter is held before it publishes a
//! first command buffer's completion (`trace::Point::Publish`) while the queue's next one is
//! submitted, whose submission handler then waits for the first one's completed handler; the
//! waiter let go, that handler runs, and the second's handler returns. Where the second's
//! transition to `Scheduled` was posted as its batch went, it held the queue's context, the first
//! one's completion parked behind it, for as long as its submission handler waited.
//!
//! And where the scheduler's thread runs none of them (`scheduler/mod.rs`, `HandOff`): it hands
//! each command buffer's submission handlers to the device's handler threads before it looks for
//! its pass's next batch. A pass is held while two command buffers of two queues are committed,
//! so the next one submits both, pausing after each submission (`trace::pause_after_submit`):
//! the first one's submission handler starts before the pause after the second one's submission
//! ended, where it once ran only after the pass's last submission and pause.
//!
//! And that no handler thread waits for a submission handler (`CommandBufferShared::
//! gate_for_handlers`): with the device's handler pool capped at two threads, the first of
//! several queues' command buffers has a submission handler that holds its context of the pool
//! until every command buffer's work completed and was seen, so each one's transition to
//! `Completed` comes before its submission handlers ran. Each waits behind a gate in its queue's
//! context, holding no thread, and all complete once that handler returns; a transition that
//! waited on a thread for them would take both, and the submission handlers would find none.
//!
//! And that a committing thread's own pass (`CommandBuffer::commit_and_submit`) keeps inline only
//! its own command buffer's submission handlers: with the scheduler's thread held after a pass
//! (`waiter_submits::Held`), a command buffer committed on one queue waits in the inbox, and a
//! `commit_and_submit` on another submits both. The earlier one's handler runs on the device's
//! handler threads, where it once ran on the committing thread after its whole pass; the
//! committing one's runs inside its call, after the earlier one's.
//!
//! And that such a pass ends at the committing command buffer's own submission handlers: with the
//! scheduler's thread held so, a `commit_and_submit` is held before its pass
//! (`trace::Point::AtCommitPass`) while three more command buffers are committed, ready, after it
//! (one on another queue, one behind it on its own, one behind the first on the other), and its
//! pass takes all four in. It submits its own and ends: its handler runs inside its call before
//! any later batch's submission, where it once ran after the pass submitted all four. The
//! scheduler's thread submits the three, each queue's in its commit order.
//!
//! And that such a gated transition to `Completed` takes one look for a thread to run it
//! (`Contexts::post_gated`): a command buffer's submission handler waits until its work completed
//! and its transition was posted, behind the gate, with one look, where the push behind the gate
//! and the poster's opening once made one each.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::completion::trace;
use crate::contexts;
use crate::tests::SETTLE;
use crate::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, ResourceOptions,
};
use metal2vulkan::device::CompletionPolicy;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// The bytes each command buffer fills.
const FILL: u64 = 64 << 10;

/// How long a step may take before the test fails rather than hangs.
const WAIT: Duration = Duration::from_secs(10);

/// How long the scheduler pauses after each submission in the hand-off check: a submission handler
/// that started while the pass went on is told from one that started after it by a margin this
/// long.
const PASS_PAUSE: Duration = Duration::from_millis(200);

/// The threads of the handler pool, and the queues, of the check at the pool's cap.
const POOL: usize = 2;
const QUEUES: usize = 6;

/// The device the cfg selects (a CPU device under `m2v_cpu_icd`, else a GPU) under `completion`.
fn device(completion: CompletionPolicy) -> Device {
    device_with(completion, DeviceOptions::default())
}

/// [`device`], from `options`.
fn device_with(completion: CompletionPolicy, options: DeviceOptions) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..options
    };
    options.policy.completion = completion;
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    device
}

/// A command buffer of `queue` that fills `target` with `value`.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..FILL, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Wait for `command_buffer`, which completes.
fn completes(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The module docs' check, under `completion`.
fn handlers_run_before_the_next_pass(completion: CompletionPolicy) {
    let device = device(completion);
    // Each command buffer's submission's return (`CommandBuffer::submit_returned_time`).
    device.shared().timing.note_queue_times();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let targets = [
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
    ];
    // Each queue's first work done, and each target's, before the measured commits.
    for (queue, target) in [(&first, &targets[0]), (&second, &targets[1])] {
        let warm = fill(queue, target, 1);
        warm.commit().unwrap();
        completes(&warm);
    }
    let ran = Arc::new(Mutex::new(None));
    let held = fill(&first, &targets[0], 2);
    let noted = Arc::clone(&ran);
    held.add_submission_handler(Box::new(move |_: &CommandBuffer| {
        *noted.lock().unwrap() = Some(host_time());
    }))
    .unwrap();
    // The scheduler's thread takes it in its pass and is held in its submission, the scheduler's
    // lock held and the thread counted as the passer.
    trace::hold_at(trace::Point::FirstWork);
    held.commit().unwrap();
    let reached = trace::reached(trace::Point::FirstWork, WAIT);
    // Posted while that pass is under way: left to the thread's look once the pass ends.
    let later = fill(&second, &targets[1], 3);
    if reached {
        later.commit().unwrap();
        trace::pause_after_submit(Some(PASS_PAUSE));
    }
    trace::release(trace::Point::FirstWork);
    assert!(reached, "the scheduler's pass never reached its submission");
    completes(&held);
    completes(&later);
    trace::pause_after_submit(None);
    let ran = (*ran.lock().unwrap()).expect("its submission handler ran");
    let paused = later.submit_returned_time() + PASS_PAUSE.as_secs_f64();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: the held pass's handler started {:.1} ms before the \
         pause after the later commit's submission ended",
        device.name(),
        (paused - ran) * 1e3,
    );
    assert!(
        ran < paused,
        "the held pass's submission handler started {:.1} ms after the pause after the next \
         pass's submission of the commit posted meanwhile ended ({completion:?})",
        (ran - paused) * 1e3,
    );
}

#[test]
fn a_pass_s_submission_handlers_run_before_the_thread_passes_again_exact() {
    handlers_run_before_the_next_pass(CompletionPolicy::Exact);
}

#[test]
fn a_pass_s_submission_handlers_run_before_the_thread_passes_again_grouped() {
    handlers_run_before_the_next_pass(CompletionPolicy::Grouped);
}

/// The module docs' second check, under `completion`.
fn a_completion_waits_for_no_later_submission_handler(completion: CompletionPolicy) {
    let device = device(completion);
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    let warm = fill(&queue, &target, 1);
    warm.commit().unwrap();
    completes(&warm);
    let (completed, first_completed) = mpsc::channel();
    let first = fill(&queue, &target, 2);
    first
        .add_completed_handler(Box::new(move |_: &CommandBuffer| {
            let _ = completed.send(());
        }))
        .unwrap();
    // Its completion seen and held before it is published.
    trace::hold_at(trace::Point::Publish);
    first.commit().unwrap();
    let reached = trace::reached(trace::Point::Publish, WAIT);
    let (started, handler_started) = mpsc::channel();
    let (answered, handler_answered) = mpsc::channel();
    let second = fill(&queue, &target, 3);
    second
        .add_submission_handler(Box::new(move |_: &CommandBuffer| {
            let _ = started.send(());
            let _ = answered.send(first_completed.recv_timeout(WAIT).is_ok());
        }))
        .unwrap();
    if reached {
        second.commit().unwrap();
    }
    // The second's submission handler runs, waiting for the first's completion, before the
    // waiter publishes that completion.
    let began = reached && handler_started.recv_timeout(WAIT).is_ok();
    trace::release(trace::Point::Publish);
    assert!(
        reached,
        "the first command buffer's completion was never seen"
    );
    assert!(
        began,
        "the second command buffer's submission handler never ran"
    );
    let answered = handler_answered.recv_timeout(WAIT + WAIT).unwrap();
    completes(&first);
    completes(&second);
    assert!(
        answered,
        "the first command buffer's completed handler did not run while the second's submission \
         handler waited {WAIT:?} for it ({completion:?})"
    );
}

#[test]
fn a_completion_waits_for_no_later_submission_handler_exact() {
    a_completion_waits_for_no_later_submission_handler(CompletionPolicy::Exact);
}

#[test]
fn a_completion_waits_for_no_later_submission_handler_grouped() {
    a_completion_waits_for_no_later_submission_handler(CompletionPolicy::Grouped);
}

/// The module docs' third check, under `completion`.
fn a_pass_s_handlers_run_while_it_goes_on(completion: CompletionPolicy) {
    let device = device(completion);
    // Each command buffer's submission's return (`CommandBuffer::submit_returned_time`).
    device.shared().timing.note_queue_times();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let targets = [
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
    ];
    for (queue, target) in [(&first, &targets[0]), (&second, &targets[1])] {
        let warm = fill(queue, target, 1);
        warm.commit().unwrap();
        completes(&warm);
    }
    // A pass of the scheduler's thread is held in a submission while the two are committed: its
    // next pass takes both in and submits them in commit order.
    let held = fill(&first, &targets[0], 2);
    trace::hold_at(trace::Point::FirstWork);
    held.commit().unwrap();
    let reached = trace::reached(trace::Point::FirstWork, WAIT);
    let started = Arc::new(Mutex::new(None));
    let earlier = fill(&first, &targets[0], 3);
    let noted = Arc::clone(&started);
    earlier
        .add_submission_handler(Box::new(move |_: &CommandBuffer| {
            *noted.lock().unwrap() = Some(host_time());
        }))
        .unwrap();
    let later = fill(&second, &targets[1], 4);
    if reached {
        earlier.commit().unwrap();
        later.commit().unwrap();
        trace::pause_after_submit(Some(PASS_PAUSE));
    }
    trace::release(trace::Point::FirstWork);
    assert!(reached, "the scheduler's pass never reached its submission");
    completes(&held);
    completes(&earlier);
    completes(&later);
    trace::pause_after_submit(None);
    let started = (*started.lock().unwrap()).expect("its submission handler ran");
    let paused = later.submit_returned_time() + PASS_PAUSE.as_secs_f64();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: the earlier command buffer's handler started {:.1} ms \
         before the pause after the later one's submission ended",
        device.name(),
        (paused - started) * 1e3,
    );
    assert!(
        started < paused,
        "the earlier command buffer's submission handler started {:.1} ms after the pause after \
         the later one's submission, made by the same pass, ended ({completion:?})",
        (started - paused) * 1e3,
    );
}

#[test]
fn a_pass_s_submission_handlers_run_while_it_goes_on_exact() {
    a_pass_s_handlers_run_while_it_goes_on(CompletionPolicy::Exact);
}

#[test]
fn a_pass_s_submission_handlers_run_while_it_goes_on_grouped() {
    a_pass_s_handlers_run_while_it_goes_on(CompletionPolicy::Grouped);
}

/// The module docs' fourth check, under `completion`.
fn completions_before_their_handlers_hold_no_thread(completion: CompletionPolicy) {
    // The device's handler pool, capped at `POOL` threads; its queues, `QUEUES`, each on the
    // device's plan.
    contexts::TEST_LIMIT.store(POOL, std::sync::atomic::Ordering::Relaxed);
    let options = DeviceOptions {
        queue_limits: None,
        ..DeviceOptions::default()
    };
    let device = device_with(completion, options);
    contexts::TEST_LIMIT.store(0, std::sync::atomic::Ordering::Relaxed);
    let queues: Vec<CommandQueue> = (0..QUEUES)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let targets: Vec<Buffer> = (0..QUEUES)
        .map(|_| device.new_buffer(FILL, ResourceOptions::private()).unwrap())
        .collect();
    for (queue, target) in queues.iter().zip(&targets) {
        let warm = fill(queue, target, 1);
        warm.commit().unwrap();
        completes(&warm);
    }
    // The first one's submission handler holds the context of submission handlers, the later
    // ones' behind it, until every command buffer's work completed and its completion was seen.
    let (go, release) = mpsc::channel::<()>();
    let mut release = Some(release);
    let committed: Vec<CommandBuffer> = (queues.iter().zip(&targets))
        .map(|(queue, target)| {
            let command_buffer = fill(queue, target, 2);
            let release = release.take();
            command_buffer
                .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                    if let Some(release) = release {
                        let _ = release.recv_timeout(WAIT);
                    }
                }))
                .unwrap();
            command_buffer
        })
        .collect();
    trace::start();
    for command_buffer in &committed {
        command_buffer.commit().unwrap();
    }
    let deadline = std::time::Instant::now() + WAIT;
    while trace::finished_count() < QUEUES && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let seen = trace::finished_count();
    // The completions seen are posted right after.
    std::thread::sleep(Duration::from_millis(50));
    let _ = go.send(());
    let completed = (committed.iter())
        .filter(|command_buffer| {
            command_buffer.wait_until_completed_for(WAIT).is_ok()
                && command_buffer.status() == CommandBufferStatus::Completed
        })
        .count();
    let _ = trace::stop();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: {seen} of {QUEUES} completions seen before their \
         submission handlers ran, {completed} completed, with {POOL} handler threads",
        device.name(),
    );
    assert_eq!(
        seen, QUEUES,
        "not every command buffer's work completed in time"
    );
    assert_eq!(
        completed,
        QUEUES,
        "with {POOL} handler threads, {} of {QUEUES} command buffers whose completion came before \
         their submission handlers never completed ({completion:?})",
        QUEUES - completed,
    );
}

#[test]
fn completions_before_their_submission_handlers_hold_no_handler_thread_exact() {
    completions_before_their_handlers_hold_no_thread(CompletionPolicy::Exact);
}

#[test]
fn completions_before_their_submission_handlers_hold_no_handler_thread_grouped() {
    completions_before_their_handlers_hold_no_thread(CompletionPolicy::Grouped);
}

/// What a submission handler of [`a_committing_pass_keeps_only_its_own`] noted as it ran: whose,
/// on which thread, and that thread's name.
type Ran = Arc<Mutex<Vec<(&'static str, std::thread::ThreadId, Option<String>)>>>;

/// The module docs' check of a committing thread's pass, under `completion`.
fn a_committing_pass_keeps_only_its_own(completion: CompletionPolicy) {
    // Its three queues, each on the device's plan.
    let options = DeviceOptions {
        queue_limits: None,
        ..DeviceOptions::default()
    };
    let device = device_with(completion, options);
    let [first, second, side] = [(); 3].map(|()| device.new_command_queue().unwrap());
    let targets = [(); 3].map(|()| (device.new_buffer(FILL, ResourceOptions::private())).unwrap());
    // Each queue's first work done, and each target's, before the measured commits.
    for (queue, target) in [&first, &second, &side].into_iter().zip(&targets) {
        let warm = fill(queue, target, 0);
        warm.commit().unwrap();
        completes(&warm);
    }
    let ran: Ran = Arc::default();
    let noting = |command_buffer: &CommandBuffer, name: &'static str| {
        let ran = Arc::clone(&ran);
        let handler = Box::new(move |_: &CommandBuffer| {
            let thread = std::thread::current();
            let name_of = thread.name().map(str::to_owned);
            ran.lock().unwrap().push((name, thread.id(), name_of));
        });
        command_buffer.add_submission_handler(handler).unwrap();
    };
    // The scheduler's thread held after a pass, counted out: the earlier commit waits in the
    // inbox, and the committing thread's pass takes it in with its own.
    let held = super::waiter_submits::Held::new(&side, &targets[2]);
    let earlier = fill(&first, &targets[0], 1);
    noting(&earlier, "earlier");
    earlier.commit().unwrap();
    let ours = fill(&second, &targets[1], 2);
    noting(&ours, "ours");
    let before = device.work_counts();
    ours.commit_and_submit().unwrap();
    let after = device.work_counts();
    let at_return = ran.lock().unwrap().clone();
    drop(held);
    for command_buffer in [&earlier, &ours] {
        completes(command_buffer);
    }
    let ran = ran.lock().unwrap().clone();
    let caller = std::thread::current().id();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: a committing pass's handlers {:?}",
        device.name(),
        (ran.iter())
            .map(|(name, thread, named)| (name, *thread == caller, named))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        after.batches_submitted_at_commit - before.batches_submitted_at_commit,
        2,
        "the committing thread's pass did not submit both ({completion:?})"
    );
    assert!(
        (at_return.iter()).any(|&(name, thread, _)| name == "ours" && thread == caller),
        "the committing command buffer's own submission handler did not run inside its call \
         ({completion:?}): {at_return:?}"
    );
    let order: Vec<&str> = ran.iter().map(|&(name, ..)| name).collect();
    assert_eq!(
        order,
        ["earlier", "ours"],
        "submission handlers out of submission order ({completion:?})"
    );
    assert_eq!(
        ran[0].2.as_deref(),
        Some("metal2vulkan-handlers"),
        "the handler of a command buffer the committing thread's pass took, not its own, ran on \
         {:?}, not a handler thread ({completion:?})",
        ran[0].2,
    );
}

#[test]
fn a_committing_pass_keeps_only_its_own_submission_handlers_exact() {
    a_committing_pass_keeps_only_its_own(CompletionPolicy::Exact);
}

#[test]
fn a_committing_pass_keeps_only_its_own_submission_handlers_grouped() {
    a_committing_pass_keeps_only_its_own(CompletionPolicy::Grouped);
}

/// What a submission handler of [`a_committing_pass_ends_at_its_own`] noted as it ran: whose, and
/// the device's kernel submissions by then.
type Noted = Arc<Mutex<Vec<(&'static str, u64)>>>;

/// The module docs' check of a committing pass's end, under `completion`.
fn a_committing_pass_ends_at_its_own(completion: CompletionPolicy) {
    // Its three queues, each on the device's plan.
    let options = DeviceOptions {
        queue_limits: None,
        ..DeviceOptions::default()
    };
    let device = device_with(completion, options);
    let [first, second, side] = [(); 3].map(|()| device.new_command_queue().unwrap());
    let targets = [(); 3].map(|()| (device.new_buffer(FILL, ResourceOptions::private())).unwrap());
    // Each queue's first work done, and each target's, before the measured commits.
    for (queue, target) in [&first, &second, &side].into_iter().zip(&targets) {
        let warm = fill(queue, target, 0);
        warm.commit().unwrap();
        completes(&warm);
    }
    let noted: Noted = Arc::default();
    let noting = |command_buffer: &CommandBuffer, name: &'static str| {
        let noted = Arc::clone(&noted);
        let handler = Box::new(move |command_buffer: &CommandBuffer| {
            let submissions = command_buffer.device().work_counts().queue_submissions;
            noted.lock().unwrap().push((name, submissions));
        });
        command_buffer.add_submission_handler(handler).unwrap();
    };
    // The scheduler's thread held after a pass, counted out: what is committed waits in the
    // inbox, and the committing thread's pass takes it in.
    let held = super::waiter_submits::Held::new(&side, &targets[2]);
    let ours = fill(&first, &targets[0], 1);
    noting(&ours, "ours");
    let before = device.work_counts();
    trace::hold_at(trace::Point::AtCommitPass);
    let committing = std::thread::spawn(move || ours.commit_and_submit().map(|()| ours));
    assert!(
        trace::reached(trace::Point::AtCommitPass, WAIT),
        "the committing thread never came to its pass ({completion:?})"
    );
    // Committed after it, each ready: one on another queue, one behind it on its own queue, and
    // one behind the first on the other.
    let later = [
        (&second, 1, "second's first"),
        (&first, 0, "first's next"),
        (&second, 1, "second's next"),
    ]
    .map(|(queue, target, name)| {
        let command_buffer = fill(queue, &targets[target], 2);
        noting(&command_buffer, name);
        command_buffer.commit().unwrap();
        command_buffer
    });
    trace::release(trace::Point::AtCommitPass);
    let ours = committing.join().unwrap().unwrap();
    let after = device.work_counts();
    let at_return = noted.lock().unwrap().clone();
    drop(held);
    for command_buffer in std::iter::once(&ours).chain(&later) {
        completes(command_buffer);
    }
    let noted = noted.lock().unwrap().clone();
    let moved = |field: fn(&crate::WorkCounts) -> u64| field(&after) - field(&before);
    let submitted: Vec<(&str, u64)> = (noted.iter())
        .map(|&(name, submissions)| (name, submissions - before.queue_submissions))
        .collect();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: a committing pass submitted {} batches, ended early          {}; handlers in their turn, with the kernel submissions made by then: {submitted:?}",
        device.name(),
        moved(|c| c.batches_submitted_at_commit),
        moved(|c| c.scheduler_passes_at_commit_ended_early),
    );
    assert_eq!(
        (at_return.len(), submitted.first().copied()),
        (1, Some(("ours", 1))),
        "the committing command buffer's own submission handler did not run inside its call          before any later batch's submission ({completion:?}): {submitted:?}"
    );
    assert_eq!(
        (
            moved(|c| c.batches_submitted_at_commit),
            moved(|c| c.scheduler_passes_at_commit_ended_early),
        ),
        (1, 1),
        "the committing thread's pass went on past its own command buffer ({completion:?})"
    );
    let order: Vec<&str> = submitted.iter().map(|&(name, _)| name).collect();
    assert_eq!(
        order,
        ["ours", "second's first", "first's next", "second's next"],
        "submission handlers out of submission order, or a queue's batches out of commit order          ({completion:?})"
    );
}

#[test]
fn a_committing_pass_ends_at_its_own_submission_handlers_exact() {
    a_committing_pass_ends_at_its_own(CompletionPolicy::Exact);
}

#[test]
fn a_committing_pass_ends_at_its_own_submission_handlers_grouped() {
    a_committing_pass_ends_at_its_own(CompletionPolicy::Grouped);
}

/// The module docs' check of a gated completion's post, under `completion`.
fn a_gated_completion_is_posted_with_one_look(completion: CompletionPolicy) {
    let device = device(completion);
    let queue = device.new_command_queue().unwrap();
    let target = (device.new_buffer(FILL, ResourceOptions::private())).unwrap();
    let warm = fill(&queue, &target, 0);
    warm.commit().unwrap();
    completes(&warm);
    let command_buffer = fill(&queue, &target, 1);
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(Some(released));
    command_buffer
        .add_submission_handler(Box::new(move |_: &CommandBuffer| {
            if let Some(released) = released.lock().unwrap().take() {
                let _ = released.recv_timeout(WAIT);
            }
        }))
        .unwrap();
    trace::start();
    command_buffer.commit().unwrap();
    // Its work completes while its submission handler waits: its transition to `Completed` is
    // posted behind the gate the handler opens.
    let deadline = std::time::Instant::now() + WAIT;
    while trace::finished_posts().is_empty() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let _ = release.send(());
    completes(&command_buffer);
    let posts = trace::finished_posts();
    let _ = trace::stop();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: a completion before its submission handler posted \
         (gated, looks for a thread) {posts:?}",
        device.name(),
    );
    assert_eq!(
        posts,
        [(true, 1)],
        "the completion that came before its submission handler ran was not posted behind its \
         gate with one look for a thread ({completion:?})"
    );
}

#[test]
fn a_gated_completion_is_posted_with_one_look_for_a_thread_exact() {
    a_gated_completion_is_posted_with_one_look(CompletionPolicy::Exact);
}

#[test]
fn a_gated_completion_is_posted_with_one_look_for_a_thread_grouped() {
    a_gated_completion_is_posted_with_one_look(CompletionPolicy::Grouped);
}
