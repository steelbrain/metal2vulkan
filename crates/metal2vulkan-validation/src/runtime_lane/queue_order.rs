//! Tracked order across command queues (the scheduler's `order.rs`) on the device the cfg selects,
//! under the validation layer. Metal tracks a resource across queues in the order it schedules the
//! command buffers, their commit order (its tracking across queues), whatever order they were made
//! in. Each test makes queue B's command buffer first and encodes its copy of a tracked buffer,
//! then commits queue A's write of the buffer and B's copy, in that order: B's copy holds A's
//! write.
//!
//! - **Ready at its commit.** A's write is a blit fill.
//! - **Held for its host step.** A's write is an execution of commands a kernel of A wrote
//!   (`indirect_host_steps.rs`), behind fills of a ballast: its batch waits at its gate for A's
//!   earlier work to complete, and for its host step, while B commits.
//! - **Held for its submission check.** A's fill has a check that waits until B committed.
//!
//! And within one queue, commits from two threads while the scheduler thread's submission waits
//! for the `VkQueue` (the test holds it): every commit returns at once, left to that thread
//! (`scheduler/inbox.rs`), and every command buffer runs in commit order. A checked command buffer
//! committed meanwhile is checked and taken up by that thread as soon as its pass ends: its
//! submission begins then, the time the thread spent in the driver's `vkQueueSubmit` before it
//! apart (`WorkCounts::batch_submit_nanos`).

use super::degrade_notices::on_device_with;
use super::degrade_notices::{assert_made_once, on_device};
use super::indirect_host_steps::{expected, Setup, HOST_STEPS};
use super::shared_kinds;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions,
    Submission, SubmissionCheck,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// The bytes A's fill writes.
const FILLED: u8 = 0x5a;

/// The bytes of the buffers A writes and B copies.
const BYTES: u64 = 4096;

/// The ballast before the host step's execution, and how many times A fills it.
const BALLAST: u64 = 16 << 20;
const BALLAST_FILLS: usize = 8;

/// Queue B's command buffer, made first: a copy of `source` into a new buffer of 0xff bytes,
/// encoded before anything of queue A's. The copy's destination with it.
fn copy_first(queue: &CommandQueue, source: &Buffer) -> (CommandBuffer, Buffer) {
    let copy = buffer_with(queue.device(), &vec![0xff; source.length() as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, &copy, 0, source.length())
        .unwrap();
    blit.end_encoding().unwrap();
    (command_buffer, copy)
}

/// A fill of `buffer` with [`FILLED`] in `command_buffer`.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), FILLED)
        .unwrap();
    blit.end_encoding().unwrap();
}

/// Wait for each committed command buffer, and require it completed.
fn completed(command_buffers: &[&CommandBuffer]) {
    for command_buffer in command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

/// Two queues of `device`: A's and B's.
fn queues(device: &Device) -> (CommandQueue, CommandQueue) {
    (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    )
}

/// A's fill, submitted at its commit, comes before B's copy, committed after it: B's command
/// buffer was made and encoded first.
#[test]
fn a_copy_committed_after_a_write_on_another_queue_holds_it() {
    on_device_with(at_plan, |device, _| {
        let (a, b) = queues(device);
        let written = buffer_with(device, &[0; BYTES as usize]).unwrap();
        let (reader, copy) = copy_first(&b, &written);
        let writer = a.command_buffer().unwrap();
        fill(&writer, &written);
        writer.commit().unwrap();
        reader.commit().unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(
            read_buffer(&copy, 0, BYTES).unwrap(),
            vec![FILLED; BYTES as usize],
            "on {}",
            device.name()
        );
    });
}

/// A's execution of commands its kernel wrote waits for its host step, behind fills of a
/// ballast; B's copy of what the commands write, committed meanwhile, holds their writes: the
/// held batch keeps its place.
#[test]
fn a_copy_committed_while_a_write_waits_for_its_host_step_holds_it() {
    on_device_with(at_plan, |device, messages| {
        let (a, b) = queues(device);
        let setup = Setup::new(device, 2);
        let (reader, copy) = copy_first(&b, setup.out());
        let ballast = device
            .new_buffer(BALLAST, ResourceOptions::private())
            .unwrap();
        let writer = a.command_buffer().unwrap();
        let mut blit = writer.blit_command_encoder().unwrap();
        for fill in 0..BALLAST_FILLS {
            blit.fill_buffer(&ballast, 0..BALLAST, fill as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        let mut encoder = writer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut encoder = writer.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        writer.commit().unwrap();
        reader.commit().unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(setup.output(), expected());
        let copied: Vec<u32> = read_buffer(&copy, 0, copy.length())
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(copied, expected(), "on {}", device.name());
        assert_made_once(messages, &HOST_STEPS, &shared_kinds(device));
    });
}

/// A's fill has a submission check that returns only once B committed; B's copy, committed
/// meanwhile, holds the fill: the held batch keeps its place.
#[test]
fn a_copy_committed_while_a_write_waits_for_its_check_holds_it() {
    on_device_with(at_plan, |device, _| {
        let (a, b) = queues(device);
        let written = buffer_with(device, &[0; BYTES as usize]).unwrap();
        let (reader, copy) = copy_first(&b, &written);
        let (entered, check_entered) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (entered, released) = (Mutex::new(entered), Mutex::new(released));
        let check: SubmissionCheck = Arc::new(move |_: &Submission<'_>| {
            let _ = entered.lock().unwrap().send(());
            let _ = released
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
            None
        });
        let writer = a.command_buffer().unwrap();
        fill(&writer, &written);
        writer.set_submission_check(check).unwrap();
        writer.commit().unwrap();
        check_entered
            .recv_timeout(Duration::from_secs(30))
            .expect("the check runs on the scheduler thread");
        reader.commit().unwrap();
        release.send(()).unwrap();
        completed(&[&writer, &reader]);
        assert_eq!(
            read_buffer(&copy, 0, BYTES).unwrap(),
            vec![FILLED; BYTES as usize],
            "on {}",
            device.name()
        );
    });
}

/// Command buffer `number` (from 1) of `queue`: a copy of `tracked`'s 4 bytes into `seen` at
/// `4 * (number - 1)`, then a fill of `tracked` with `number`. Its copy holds the number of the
/// command buffer that ran before it, 0 for the first.
fn numbered(queue: &CommandQueue, tracked: &Buffer, seen: &Buffer, number: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let at = 4 * u64::from(number - 1);
    blit.copy_from_buffer_to_buffer(tracked, 0, seen, at, 4)
        .unwrap();
    blit.fill_buffer(tracked, 0..4, number).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Two threads each commit [`EACH`] command buffers of one queue while the test holds its
/// `VkQueue`: every commit returns at once, its batches left to the scheduler's thread, whose
/// `vkQueueSubmit` waits for the `VkQueue`, as Metal's `commit` returns at once. Once the `VkQueue`
/// is free, the command buffers run in commit order: each that returned before another started
/// runs before it, as each one's copy of the number the one before it wrote shows.
#[test]
fn commits_made_while_the_vk_queue_is_held_return_at_once_and_run_in_commit_order() {
    const EACH: u8 = 4;
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let tracked = buffer_with(device, &[0; 4]).unwrap();
        let seen = buffer_with(device, &[0xff; 4 * 2 * EACH as usize]).unwrap();
        // Made and encoded first: the threads only commit them.
        let lines: Vec<Vec<(u8, CommandBuffer)>> = (0..2)
            .map(|line| {
                (1..=EACH)
                    .map(|k| line * EACH + k)
                    .map(|number| (number, numbered(&queue, &tracked, &seen, number)))
                    .collect()
            })
            .collect();
        let handed_off = || device.work_counts().commits_handed_off;
        let before = handed_off();
        let (held, holding) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let holder = {
            let queue = queue.clone();
            std::thread::spawn(move || {
                // SAFETY: the closure keeps no handle and calls nothing of the runtime's; it waits
                // only for this test's release.
                unsafe {
                    queue.with_vk_queue(|_, _, _| {
                        held.send(()).unwrap();
                        let _ = released.recv_timeout(WAIT);
                    })
                }
            })
        };
        holding.recv_timeout(WAIT).unwrap();
        // Each commit's start and return on one clock.
        let clock = Arc::new(AtomicU64::new(0));
        let (returned, returns) = mpsc::channel();
        let committers: Vec<_> = lines
            .into_iter()
            .map(|line| {
                let (clock, returned) = (Arc::clone(&clock), returned.clone());
                std::thread::spawn(move || {
                    let mut times = Vec::new();
                    for (number, command_buffer) in &line {
                        let start = clock.fetch_add(1, Ordering::SeqCst);
                        command_buffer.commit().unwrap();
                        let end = clock.fetch_add(1, Ordering::SeqCst);
                        times.push((*number, start, end));
                        returned.send(()).unwrap();
                    }
                    (line, times)
                })
            })
            .collect();
        drop(returned);
        for commit in 1..=2 * EACH {
            returns.recv_timeout(WAIT).unwrap_or_else(|_| {
                panic!("commit {commit} waited for the scheduler thread's submission")
            });
        }
        assert_eq!(
            handed_off() - before,
            u64::from(2 * EACH),
            "every commit was left to the scheduler's thread"
        );
        release.send(()).unwrap();
        holder.join().unwrap();
        let (mut command_buffers, mut times) = (Vec::new(), Vec::new());
        for committer in committers {
            let (line, line_times) = committer.join().unwrap();
            command_buffers.extend(line.into_iter().map(|(_, command_buffer)| command_buffer));
            times.extend(line_times);
        }
        completed(&command_buffers.iter().collect::<Vec<_>>());
        // The order they ran in: each copy names the one before it.
        let before_of = read_buffer(&seen, 0, seen.length()).unwrap();
        let mut ran = vec![0u8];
        while ran.len() <= 2 * EACH as usize {
            let last = *ran.last().unwrap();
            let next: Vec<u8> = (1..=2 * EACH)
                .filter(|&number| before_of[4 * usize::from(number - 1)] == last)
                .collect();
            assert_eq!(next.len(), 1, "one ran after {last}: {before_of:?}");
            ran.push(next[0]);
        }
        let place = |number: u8| ran.iter().position(|&n| n == number).unwrap();
        for &(first, _, returned) in &times {
            for &(later, started, _) in &times {
                if returned < started {
                    assert!(
                        place(first) < place(later),
                        "{first} returned before {later} was committed, so runs first: {ran:?}"
                    );
                }
            }
        }
    });
}

/// A check that lets every submission run.
fn passes() -> SubmissionCheck {
    Arc::new(|_: &Submission<'_>| None)
}

/// The nanoseconds the batches' `vkQueueSubmit` calls of `device` spent in the driver so far
/// (`WorkCounts::batch_submit_nanos`).
fn in_driver(device: &Device) -> u64 {
    device.work_counts().batch_submit_nanos
}

/// A check that lets every submission run, and notes into `noted`, as it first runs, the time the
/// batches' `vkQueueSubmit` calls of `device` had spent in the driver by then ([`in_driver`]). It
/// runs on the scheduler's thread, which takes the command buffer up (begins its submission) after
/// it with no `vkQueueSubmit` between: a commit made while that thread submits waits in the inbox
/// until the pass ends (`scheduler/inbox.rs`), and the thread runs a check without its lock once
/// that pass and the next, which took the commit in, ended (`scheduler/mod.rs`).
fn noting(device: &Device, noted: &Arc<OnceLock<u64>>) -> SubmissionCheck {
    let (device, noted) = (device.clone(), Arc::clone(noted));
    Arc::new(move |_: &Submission<'_>| {
        noted.get_or_init(|| in_driver(&device));
        None
    })
}

/// How soon after the `VkQueue` is free the scheduler's thread takes up the left command buffer
/// (runs its check and begins its submission), at most, its own time: well under the 100 ms it
/// sleeps for at a time. The time it spent meanwhile in the driver's `vkQueueSubmit` is not its
/// own: an RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside the full suite once took 243 ms in one
/// call, and a Radeon 780M about a second.
const PROMPTLY: Duration = Duration::from_millis(40);

/// A command buffer with a submission check, committed while the scheduler's thread submits
/// another (the test holds the `VkQueue`), is taken in as that thread looks at its inbox again once
/// that pass ends: the thread runs the check and begins its submission at once, [`PROMPTLY`] after
/// the `VkQueue` is free less the time it spent in the driver's `vkQueueSubmit` before the check
/// ([`noting`]), not when it looks for work by itself. The scheduler's thread is made to sleep
/// afresh first (it checks and submits a command buffer), so that it would look by itself only
/// about 100 ms later.
#[test]
fn a_checked_command_buffer_committed_during_a_submission_is_taken_up_promptly() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let tracked = buffer_with(device, &[0; 4]).unwrap();
        let seen = buffer_with(device, &[0xff; 12]).unwrap();
        let first = numbered(&queue, &tracked, &seen, 1);
        first.set_submission_check(passes()).unwrap();
        let submitting = numbered(&queue, &tracked, &seen, 2);
        let left = numbered(&queue, &tracked, &seen, 3);
        let noted = Arc::new(OnceLock::new());
        left.set_submission_check(noting(device, &noted)).unwrap();
        // The scheduler's thread checks and submits it, then sleeps afresh, its pass over.
        first.commit().unwrap();
        first.wait_until_scheduled().unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let (held, holding) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let holder = {
            let queue = queue.clone();
            std::thread::spawn(move || {
                // SAFETY: the closure keeps no handle and calls nothing of the runtime's; it waits
                // only for this test's release.
                unsafe {
                    queue.with_vk_queue(|_, _, _| {
                        held.send(()).unwrap();
                        let _ = released.recv_timeout(WAIT);
                    })
                }
            })
        };
        holding.recv_timeout(WAIT).unwrap();
        let handed_off = || device.work_counts().commits_handed_off;
        let before = handed_off();
        submitting.commit().unwrap();
        // The scheduler thread's pass waits for the `VkQueue` under the scheduler's lock.
        std::thread::sleep(Duration::from_millis(20));
        left.commit().unwrap();
        assert_eq!(
            handed_off() - before,
            2,
            "both commits were left to the scheduler's thread"
        );
        // No `vkQueueSubmit` is in the driver while the test holds the `VkQueue`.
        let driven = in_driver(device);
        let free = host_time();
        release.send(()).unwrap();
        holder.join().unwrap();
        completed(&[&first, &submitting, &left]);
        let taken_up = left.kernel_start_time() - free;
        let driver = 1e-9 * noted.get().expect("its check ran").saturating_sub(driven) as f64;
        assert!(
            taken_up - driver < PROMPTLY.as_secs_f64(),
            "taken up (its kernel start) {taken_up:.4} s after the VkQueue was free, on {}, \
             {driver:.4} s of it in the driver's vkQueueSubmit; its submission {:.4} s",
            device.name(),
            left.kernel_end_time() - left.kernel_start_time()
        );
        // In commit order: each copied the number the one before it filled.
        let ran = read_buffer(&seen, 0, 12).unwrap();
        assert_eq!(ran, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
    });
}

/// How many rounds [`checked_commits_during_submissions_are_taken_up_promptly`] runs.
const ROUNDS: usize = 40;

/// The numbered command buffer `number`, carrying a submission handler and `check`, which lets it
/// run, as the consumer's queue gives every command buffer.
fn carried(
    queue: &CommandQueue,
    (tracked, seen): (&Buffer, &Buffer),
    number: u8,
    check: SubmissionCheck,
) -> CommandBuffer {
    let command_buffer = numbered(queue, tracked, seen, number);
    command_buffer.set_submission_check(check).unwrap();
    command_buffer
        .add_submission_handler(Box::new(|_: &CommandBuffer| {}))
        .unwrap();
    command_buffer
}

/// What one round of [`checked_commits_during_submissions_are_taken_up_promptly`] saw of the two
/// command buffers committed while the scheduler's submission waited for the `VkQueue`, in
/// seconds.
#[derive(Debug)]
struct Round {
    /// How long after the `VkQueue` was free the scheduler's thread took up each (its kernel
    /// start).
    taken_up: [f64; 2],
    /// The thread's own time to take up each: the earlier one's less the time the batches'
    /// `vkQueueSubmit` calls spent in the driver from the release until its check ran
    /// ([`noting`]); the later one's from the later of the release and the earlier one's kernel
    /// end, after which the thread made no `vkQueueSubmit` before taking it up (it submitted the
    /// earlier one, its call in the driver among it, or held it for one call with later ones).
    ours: [f64; 2],
    /// How long each one's own submission took, from its kernel start to its kernel end.
    submissions: [f64; 2],
    /// The time the batches' `vkQueueSubmit` calls spent in the driver from the release until the
    /// earlier one's check ran, and until the round's command buffers completed.
    in_driver: [f64; 2],
    /// The thread's sleeps that ended at their time limit between the `VkQueue`'s release and the
    /// later kernel start.
    slept: u64,
}

/// One round of [`checked_commits_during_submissions_are_taken_up_promptly`].
fn taken_up(device: &Device, queue: &CommandQueue) -> Round {
    let tracked = buffer_with(device, &[0; 4]).unwrap();
    let seen = buffer_with(device, &[0xff; 16]).unwrap();
    let buffers = (&tracked, &seen);
    let first = carried(queue, buffers, 1, passes());
    let submitting = carried(queue, buffers, 2, passes());
    let noted = Arc::new(OnceLock::new());
    let left = [
        carried(queue, buffers, 3, noting(device, &noted)),
        carried(queue, buffers, 4, passes()),
    ];
    // The scheduler's thread checks and submits it, then sleeps afresh, its pass over.
    first.commit().unwrap();
    first.wait_until_scheduled().unwrap();
    std::thread::sleep(Duration::from_millis(5));
    let (held, holding) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let holder = {
        let queue = queue.clone();
        std::thread::spawn(move || {
            // SAFETY: the closure keeps no handle and calls nothing of the runtime's; it waits
            // only for this round's release.
            unsafe {
                queue.with_vk_queue(|_, _, _| {
                    held.send(()).unwrap();
                    let _ = released.recv_timeout(WAIT);
                })
            }
        })
    };
    holding.recv_timeout(WAIT).unwrap();
    submitting.commit().unwrap();
    // The scheduler thread's pass runs the check, then waits for the `VkQueue` under the
    // scheduler's lock.
    std::thread::sleep(Duration::from_millis(20));
    for command_buffer in &left {
        command_buffer.commit().unwrap();
    }
    let slices = || device.work_counts().scheduler_sleep_slices;
    let before = slices();
    // No `vkQueueSubmit` is in the driver while the round holds the `VkQueue`.
    let driven = in_driver(device);
    let free = host_time();
    release.send(()).unwrap();
    holder.join().unwrap();
    // Read the thread's sleeps as soon as the later one is taken up: a sleep that ended at its
    // limit takes 100 ms, and these reads take microseconds.
    let deadline = std::time::Instant::now() + WAIT;
    while left[1].kernel_start_time() == 0.0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the later command buffer was not taken up"
        );
        std::hint::spin_loop();
    }
    let slept = slices() - before;
    completed(&[&first, &submitting, &left[0], &left[1]]);
    let since = |nanos: u64| 1e-9 * nanos.saturating_sub(driven) as f64;
    let in_driver = [
        since(*noted.get().expect("the earlier one's check ran")),
        since(in_driver(device)),
    ];
    // In commit order: each copied the number the one before it filled.
    let ran = read_buffer(&seen, 0, 16).unwrap();
    assert_eq!(ran, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    let taken_up = left.each_ref().map(|c| c.kernel_start_time() - free);
    Round {
        taken_up,
        ours: [
            taken_up[0] - in_driver[0],
            left[1].kernel_start_time() - left[0].kernel_end_time().max(free),
        ],
        submissions: (left.each_ref()).map(|c| c.kernel_end_time() - c.kernel_start_time()),
        in_driver,
        slept,
    }
}

/// The round of [`a_checked_command_buffer_committed_during_a_submission_is_taken_up_promptly`],
/// [`ROUNDS`] times, every command buffer carrying a submission handler and a check (as the
/// consumer's queue gives every one), and two committed while the scheduler's thread waits for
/// the `VkQueue` to submit another: each time, the thread takes up both (runs their checks and
/// begins their submissions, or holds them for one with later ones: their kernel starts) in
/// [`PROMPTLY`] of its own time ([`Round::ours`]): the earlier one after the `VkQueue` is free,
/// the time its `vkQueueSubmit` calls spent in the driver meanwhile apart; the later one after
/// the earlier one's submission, the driver's call among it, ended. And none of its sleeps between
/// the release and the later one's start ended at its time limit: it was woken for them, or found
/// them as its pass ended, never by itself. Each submission's own time, and the driver's, is
/// printed, not bounded: on an RTX 5080 Laptop GPU beside the full suite, the earlier one's
/// submission took 168 ms and 243 ms in two runs, and the later one was taken up 0.1 ms after it.
#[test]
fn checked_commits_during_submissions_are_taken_up_promptly() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let rounds: Vec<_> = (0..ROUNDS).map(|_| taken_up(device, &queue)).collect();
        let most =
            |of: fn(&Round) -> [f64; 2]| (rounds.iter().flat_map(of)).fold(f64::MIN, f64::max);
        let ms = |of: fn(&Round) -> [f64; 2]| 1e3 * most(of);
        let slept: u64 = rounds.iter().map(|round| round.slept).sum();
        eprintln!(
            "TAKEN-UP {}: {ROUNDS} rounds, taken up at most {:.3} ms after the VkQueue was free, \
             the thread's own time at most {:.3} ms; a submission at most {:.3} ms; the driver's \
             vkQueueSubmit at most {:.3} ms before the earlier one's check, {:.3} ms in a round; \
             sleeps ended at their limit {slept}",
            device.name(),
            ms(|round| round.taken_up),
            ms(|round| round.ours),
            ms(|round| round.submissions),
            1e3 * (rounds.iter().map(|round| round.in_driver[0])).fold(f64::MIN, f64::max),
            1e3 * (rounds.iter().map(|round| round.in_driver[1])).fold(f64::MIN, f64::max),
        );
        for (number, round) in rounds.iter().enumerate() {
            assert!(
                round.ours.iter().all(|&at| at < PROMPTLY.as_secs_f64()) && round.slept == 0,
                "round {number} on {}: {round:?}",
                device.name()
            );
        }
    });
}
