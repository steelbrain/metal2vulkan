//! `command_buffer/completion.rs`: a command buffer's completion is seen as its work ends, while
//! other queues are in use and waited for; and `commit` returns while they are.
//!
//! The consumer's guest presented at about a quarter of Metal's rate under MoltenVK 1.4.1: the
//! completion waited for any of the timelines of the busy queue's first pending batch and
//! of every idle `VkQueue` in use, and MoltenVK's wait for any of several timelines loses a signal
//! that lands while it still registers a later one, so the batch was seen only when the 100 ms
//! wait ran out. These tests make that shape: four queues, three of them in use and idle, and the
//! fourth committing a few fills back to back each round, then waiting for the last. The
//! completion finishes the first and waits again while the next one's signal lands, its
//! moment swept across rounds by the fills' sizes, which is when a wait for any loses it.
//!
//! **The completion** (`a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for`).
//! No completion waiter misses a wake: none of its waits runs out its 100 ms slice while the fence
//! or the timeline value it waits for is signaled (`WorkCounts::missed_completion_wakes`, read
//! after each round), which is how a lost wake shows, a command buffer then seen about a slice
//! late. A count is the runtime's own observation, which a loaded host does not make. On a device
//! that places its timestamps on the host's clock by a calibration (`gpu_time.rs`), a count in a
//! round whose command buffers were all seen within half a slice of when they could first be
//! seen is a signal in the slice's last moment, not a missed wake: printed, not failed.
//!
//! How late each command buffer was seen after it could first be seen is printed, not judged. It is
//! seen completed when its completed handler runs, which the runtime's threads reach from the
//! completion alone; it could first be seen at the later of its GPU end time and its kernel end
//! time where the device calibrates its timestamps, and at its kernel end time elsewhere (a device
//! that answers its GPU times from the host, or calibrates none, ends them where the completion was
//! seen). Under a test suite's load that delay is the host's scheduling, which holds Metal's own
//! handlers as long: on the M3 Max (macOS 27.0.1, at the utility quality of service the Mac lane
//! runs at) beside the runtime lane's tests at nextest's default parallelism, Metal's completed
//! handlers of this test's shape ran up to 82 ms after their `GPUEndTime` (15 of 1800 at 50 ms or
//! more), and the runtime's up to 57 ms after their kernel end, against 0.47 ms and 0.89 ms alone.
//! Neither the round's own time nor when the test's wait returned measures the completion: the
//! test's thread commits the round's command buffers before it waits for the first, so a commit the
//! driver holds holds that wait too, and neither the GPU's other work nor the driver's submission
//! is the completion's. On an RTX 3090 (NVIDIA 615.71.09) beside the lane tests' busy
//! kernels, rounds of 190 to 202 ms were submitted within 0.1 ms, their GPU work begun as long
//! after the commit, as another process's busy kernel runs, and each command buffer seen within 0.7
//! ms of its GPU end; and rounds of 57 to 64 ms ran their first command buffer's work at the commit
//! while the driver's submission of it returned 57 to 64 ms later, each command buffer seen within
//! 0.3 ms of that.
//!
//! **The commit** (`a_commit_returns_while_other_queues_are_waited_for`). Metal's `commit`
//! returns at once, and so does the runtime's: it hands the command buffer's batches to the
//! scheduler's thread and submits nothing itself (`scheduler/inbox.rs`), so a `vkQueueSubmit` the
//! driver holds never holds it. While commits submitted on the committing thread where the
//! scheduler was free, on the RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside about 20 processes
//! opening devices 1 or 2 of a run's 1,800 commits took 58 to 226 ms, the GPU having run the work
//! within 0.1 ms, and on the M3 Max beside the full suite one took 51.4 ms.
//!
//! A commit's time is its wall time less the committing thread's waits for a CPU meanwhile, where
//! the kernel counts them (`timed_commit`): a wait for a lock, and work, stay in it, and the host's
//! preemption of the thread does not. Measured on lavapipe (Mesa 26.2.4): beside a batch's suites
//! (a load of about 48 on 16 hardware threads) one commit of a run's 1,800 took 117.8 ms of wall
//! time; beside 64 spinning threads on 24 hardware threads (a load up to 40), 20 runs of 20 passed,
//! the commits' waits for a CPU up to 2.9 ms and their times less those at most 85 us; and with
//! the scheduler's state lock held 120 ms by another thread while a round committed (a diagnostic
//! build), that round's first commit read 120.2 ms, none of it a wait for a CPU, and the test
//! failed, as it did beside 48 spinning threads (123.1 ms).
//!
//! Where the kernel counts no wait for a CPU (macOS: no `/proc`), a commit's time is its wall
//! time, the host's preemption of the thread in it: there a round with a commit that slow runs
//! again, timed afresh, and the test fails only where a commit of the round run again takes that
//! long too. A lock a commit waits for, held that long, is held again; a preemption seldom is: on
//! the M3 Max beside the full suite at the utility quality of service, two commits of a run's
//! 1,800 took 69 and 97 ms of wall time after handing off their batches, which the scheduler
//! submitted and the completion saw within 8 ms, while 20 runs alone held none over 63 us. With
//! the scheduler's inbox lock held 100 ms by another thread during one commit of every 100th
//! round and of its run again (a diagnostic build), the test failed on the M3 Max and on
//! lavapipe.
//!
//! **The wakes** (`a_queue_s_batches_wake_no_other_queue_s_waiter`). A queue's command buffers
//! wake their own `VkQueue`'s waiter alone, where it sleeps, and no idle queue's: over
//! [`WAKE_ROUNDS`] rounds of [`IN_FLIGHT`] fills on one queue, the notifies to the waiters
//! (`WorkCounts::wakes_completion_waiters`) number at most the command buffers, where waking the
//! idle `VkQueue`s' waiters at each batch's watch, wait and turn makes more than one a command
//! buffer. On a device of one `VkQueue` there is no idle waiter to wake, and it holds either way.
//!
//! A summary of the rounds' times, of how late each round's command buffers were seen, of how long
//! each wait returned after its command buffer was seen (what the wait adds), of the commits' times
//! and their waits for a CPU, and of the missed wakes is printed (`ROUNDS`), and each round that
//! took 50 ms or more, or saw a command buffer late, or held a commit that long, or counted a
//! missed wake, with its command buffers' commit times and waits for a CPU, kernel start and end,
//! GPU start and end, completed handler and wait's return, from the round's commit
//! (`SLOW-ROUND`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::plan_room::device_with_room;
use crate::{
    host_time, CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// How many rounds the busy queue runs.
const ROUNDS: usize = 600;

/// The command buffers of a round, committed back to back.
const IN_FLIGHT: usize = 3;

/// How many rounds the wakes are counted over (`a_queue_s_batches_wake_no_other_queue_s_waiter`).
const WAKE_ROUNDS: usize = 100;

/// The fills' sizes, in turn over the rounds: their GPU times move each signal's moment.
const SIZES: [u64; 7] = [
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    8 << 20,
];

/// Half the completion wait's 100 ms slice: the longest a commit may take; the time from which a
/// round, or how late a command buffer was seen after it could first be seen, prints as slow; and
/// how late a round that counted a missed wake must have seen a command buffer, at least, to fail
/// on a device that calibrates its timestamps (the module docs).
const STALL: Duration = Duration::from_millis(50);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-completion-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        // Each test here keeps more than two command queues live at once: the queue plan, not
        // the runtime tests' default queue limits (`device/create.rs`).
        queue_limits: None,
        ..DeviceOptions::default()
    };
    device_with_room(&options).unwrap()
}

/// Whether `device`'s GPU end times are when the GPU ended the work: its timestamps, placed on
/// the host's clock by a calibration (`gpu_time.rs`).
fn calibrated(device: &Device) -> bool {
    let timing = &device.shared().timing;
    !timing.answers_from_host()
        && timing
            .clock()
            .is_some_and(|clock| clock.calibration.is_some())
}

/// One command buffer of a round, as the round saw it: host times in seconds.
struct Seen {
    /// Its `commit` call's time, less the committing thread's wait for a CPU meanwhile
    /// ([`timed_commit`]).
    commit: Duration,
    /// That wait.
    queued: Duration,
    /// When its completed handler ran.
    handled: f64,
    /// When the test's wait for it began.
    asked: f64,
    /// When that wait returned.
    returned: f64,
}

/// One round: its fills' size, its time from the first commit until the last wait returned, how
/// late its latest command buffer was seen completed after it could first be seen, and its line
/// for `SLOW-ROUND`.
struct Round {
    size: u64,
    took: Duration,
    late: Duration,
    /// The completion waiters' missed wakes counted while it ran
    /// (`WorkCounts::missed_completion_wakes`).
    missed: u64,
    /// Each command buffer's, in commit order.
    seen: Vec<Seen>,
    /// Its command buffers' times from the round's commit.
    line: String,
}

impl Round {
    /// Its longest commit.
    fn longest_commit(&self) -> Duration {
        self.seen.iter().map(|seen| seen.commit).max().unwrap()
    }
}

/// A completed command buffer's times as `seen` saw them, in milliseconds from `from`, a host
/// time: its commit's wall time, its kernel start and end, GPU start and end, its completed
/// handler, and its wait's return.
fn times(command_buffer: &CommandBuffer, seen: &Seen, from: f64) -> String {
    let ms = |at: f64| (at - from) * 1e3;
    let gpu = |at: Result<f64, _>| at.map_or("-".to_owned(), |at| format!("{:.1}", ms(at)));
    format!(
        "[commit {:.1} queued {:.1} ks {:.1} ke {:.1} gs {} ge {} handled {:.1} returned {:.1}]",
        seen.commit.as_secs_f64() * 1e3,
        seen.queued.as_secs_f64() * 1e3,
        ms(command_buffer.kernel_start_time()),
        ms(command_buffer.kernel_end_time()),
        gpu(command_buffer.gpu_start_time()),
        gpu(command_buffer.gpu_end_time()),
        ms(seen.handled),
        ms(seen.returned)
    )
}

/// A device of its own (its shader cache in a scratch directory named for the test) with four
/// queues, three of them idle and in use, the fourth the one each round commits on, and the buffer
/// the rounds fill: [`Rounds::round`] runs one round.
struct Rounds {
    device: Device,
    queues: Vec<crate::CommandQueue>,
    buffer: crate::Buffer,
    /// Whether the device calibrates its timestamps.
    calibrated: bool,
    /// Dropped after the device.
    _scratch: Scratch,
}

impl Rounds {
    fn new(test: &str) -> Rounds {
        let scratch = Scratch::new(test);
        let device = device(&scratch.0);
        let calibrated = calibrated(&device);
        let buffer = device
            .new_buffer(SIZES[SIZES.len() - 1], ResourceOptions::private())
            .unwrap();
        // Four queues, round robin over the device's `VkQueue`s: each is in use.
        let queues: Vec<_> = (0..4)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        let rounds = Rounds {
            device,
            queues,
            buffer,
            calibrated,
            _scratch: scratch,
        };
        // Each idle queue runs one first, so its `VkQueue` has a value its waits name.
        for (value, queue) in rounds.queues[1..].iter().enumerate() {
            let command_buffer = rounds.fill(queue, SIZES[0], value as u8);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        rounds
    }

    /// A short command buffer of `queue`: one fill of `size` bytes of `value`.
    fn fill(&self, queue: &crate::CommandQueue, size: u64, value: u8) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.buffer, 0..size, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    }

    /// Round `round`: [`IN_FLIGHT`] fills of its size committed back to back on the busy queue,
    /// each with a completed handler that notes the time, then a wait for each.
    fn round(&self, round: usize) -> Round {
        let size = SIZES[round % SIZES.len()];
        // Each with the slot its completed handler notes the host time in.
        let command_buffers: Vec<(CommandBuffer, Arc<Mutex<Option<f64>>>)> = (0..IN_FLIGHT)
            .map(|_| {
                let command_buffer = self.fill(&self.queues[0], size, round as u8);
                let handled = Arc::new(Mutex::new(None));
                let noted = Arc::clone(&handled);
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        *noted.lock().unwrap() = Some(host_time());
                    }))
                    .unwrap();
                (command_buffer, handled)
            })
            .collect();
        let missed_before = self.device.work_counts().missed_completion_wakes;
        let start = Instant::now();
        let committed = host_time();
        let commits: Vec<(Duration, Duration)> = command_buffers
            .iter()
            .map(|(command_buffer, _)| timed_commit(command_buffer))
            .collect();
        let mut seen = Vec::with_capacity(IN_FLIGHT);
        for ((command_buffer, handled), (commit, queued)) in command_buffers.iter().zip(commits) {
            let asked = host_time();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let returned = host_time();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let handled = handled
                .lock()
                .unwrap()
                .expect("its completed handler ran before its wait returned");
            seen.push(Seen {
                commit,
                queued,
                handled,
                asked,
                returned,
            });
        }
        let took = start.elapsed();
        let missed = self.device.work_counts().missed_completion_wakes - missed_before;
        // How late each was seen after it could first be seen: the latest of the round.
        let late = (command_buffers.iter().zip(&seen))
            .map(|((command_buffer, _), seen)| {
                let seeable = match self.calibrated {
                    true => (command_buffer.gpu_end_time().unwrap())
                        .max(command_buffer.kernel_end_time()),
                    false => command_buffer.kernel_end_time().max(committed),
                };
                Duration::from_secs_f64((seen.handled - seeable).max(0.0))
            })
            .max()
            .unwrap();
        let line = (command_buffers.iter().zip(&seen))
            .map(|((command_buffer, _), seen)| self::times(command_buffer, seen, committed))
            .collect::<Vec<String>>()
            .join(" ");
        Round {
            size,
            took,
            late,
            missed,
            seen,
            line,
        }
    }
}

/// Run the rounds ([`Rounds`]), and print what they saw: the rounds, and what ran them, to run a
/// round again.
fn rounds(test: &str) -> (Vec<Round>, Rounds) {
    let runner = Rounds::new(test);
    let calibrated = runner.calibrated;
    let rounds: Vec<Round> = (0..ROUNDS).map(|round| runner.round(round)).collect();
    let summary = |durations: Vec<Duration>| {
        let mut sorted = durations;
        sorted.sort();
        let at = |fraction: usize| sorted[(sorted.len() - 1) * fraction / 100];
        format!(
            "p50 {:?} p90 {:?} p99 {:?} max {:?}",
            at(50),
            at(90),
            at(99),
            sorted[sorted.len() - 1],
        )
    };
    let after = match calibrated {
        true => "the later of its GPU end and kernel end",
        false => "its kernel end",
    };
    for (n, round) in rounds.iter().enumerate() {
        let slow = round.took >= STALL || round.late >= STALL || round.longest_commit() >= STALL;
        if slow || round.missed > 0 {
            println!(
                "SLOW-ROUND {n} ({} bytes a fill): {:?}, seen {:?} late, {} missed wake(s); ms \
                 from the commit: {}",
                round.size, round.took, round.late, round.missed, round.line
            );
        }
    }
    let seen = || rounds.iter().flat_map(|round| &round.seen);
    // What a wait adds: from when its command buffer was seen, or the wait began if later, until
    // it returned.
    let added = seen()
        .map(|seen| {
            Duration::from_secs_f64((seen.returned - seen.handled.max(seen.asked)).max(0.0))
        })
        .collect();
    println!(
        "ROUNDS {ROUNDS}: {}, {} at or over {STALL:?}; seen (completed handler) after {after}: \
         {}, {} at or over {STALL:?}; wait returned after seen: {}; commit time less the wait for \
         a CPU: {}, {} at or over {STALL:?}; the commits' waits for a CPU: {}; missed wakes: {}",
        summary(rounds.iter().map(|round| round.took).collect()),
        rounds.iter().filter(|round| round.took >= STALL).count(),
        summary(rounds.iter().map(|round| round.late).collect()),
        rounds.iter().filter(|round| round.late >= STALL).count(),
        summary(added),
        summary(seen().map(|seen| seen.commit).collect()),
        seen().filter(|seen| seen.commit >= STALL).count(),
        summary(seen().map(|seen| seen.queued).collect()),
        rounds.iter().map(|round| round.missed).sum::<u64>(),
    );
    (rounds, runner)
}

/// `command_buffer`'s `commit`, timed: its wall time less the committing thread's wait for a CPU
/// meanwhile, and that wait. Linux counts each thread's wait: the second field of
/// `/proc/thread-self/schedstat`, "time spent waiting on a runqueue" in nanoseconds (the kernel's
/// `Documentation/scheduler/sched-stats.rst`); where it is not counted, the time is the wall time.
/// The instant is read before the count at the start and after it at the end, so a wait between
/// the two reads stays in the time, which never reads less than the call's time on a CPU and
/// blocked: a wait for a lock stays in it.
fn timed_commit(command_buffer: &CommandBuffer) -> (Duration, Duration) {
    let start = Instant::now();
    let before = run_queue_wait();
    command_buffer.commit().unwrap();
    let after = run_queue_wait();
    let wall = start.elapsed();
    let queued = match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => Duration::ZERO,
    };
    (wall.saturating_sub(queued), queued)
}

/// The calling thread's wait for a CPU so far, where the kernel counts it ([`timed_commit`]).
fn run_queue_wait() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let waited = stat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(Duration::from_nanos(waited))
}

#[test]
fn a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for() {
    let (rounds, runner) = rounds("seen");
    let calibrated = runner.calibrated;
    let missed: Vec<(usize, u64, Duration)> = (rounds.iter().enumerate())
        .filter(|(_, round)| round.missed > 0 && (!calibrated || round.late >= STALL))
        .map(|(n, round)| (n, round.missed, round.late))
        .collect();
    assert!(
        missed.is_empty(),
        "completion waits that ran out their slice while what they waited for was signaled \
         (round, missed wakes, latest seen after it could be): {missed:?}"
    );
}

#[test]
fn a_commit_returns_while_other_queues_are_waited_for() {
    let (rounds, runner) = rounds("commit");
    // Where the kernel counts no wait for a CPU, a round with a commit that slow runs again,
    // timed afresh: held only where a commit takes that long again (the module docs).
    let counted = run_queue_wait().is_some();
    let held: Vec<(usize, Duration, Option<Duration>)> = (rounds.iter().enumerate())
        .filter(|(_, round)| round.longest_commit() >= STALL)
        .map(|(n, round)| {
            let again = (!counted).then(|| runner.round(n).longest_commit());
            (n, round.longest_commit(), again)
        })
        .filter(|&(_, _, again)| again.is_none_or(|again| again >= STALL))
        .collect();
    assert!(
        held.is_empty(),
        "commits that took {STALL:?} or more, less the committing thread's wait for a CPU \
         where the kernel counts it, and again in the round run again where it does not, where \
         Metal's returns at once (round, longest, longest again): {held:?}. A commit only hands \
         its batches to the scheduler's thread (scheduler/inbox.rs): one this slow waited for a \
         lock or worked that long, not for a submission"
    );
}

#[test]
fn a_queue_s_batches_wake_no_other_queue_s_waiter() {
    let scratch = Scratch::new("wakes");
    let device = device(&scratch.0);
    let buffer = device
        .new_buffer(SIZES[0], ResourceOptions::private())
        .unwrap();
    // Four queues, round robin over the device's `VkQueue`s, three of them idle.
    let queues: Vec<_> = (0..4)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let fill = |value: u8| {
        let command_buffer = queues[0].command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..SIZES[0], value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    // The first makes what the first commit makes (a pool, a kept barrier command buffer).
    fill(0).wait_until_completed_for(WAIT).unwrap();
    let before = device.work_counts().wakes_completion_waiters;
    for round in 0..WAKE_ROUNDS {
        let command_buffers: Vec<CommandBuffer> =
            (0..IN_FLIGHT).map(|_| fill(round as u8)).collect();
        for command_buffer in &command_buffers {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
    }
    let woke = device.work_counts().wakes_completion_waiters - before;
    let committed = (WAKE_ROUNDS * IN_FLIGHT) as u64;
    assert!(
        woke <= committed,
        "{woke} notifies to the completion waiters for {committed} command buffers of one queue, \
         on a device of {} VkQueues: a batch watched wakes its own queue's waiter alone, where it \
         sleeps, and no turn of that queue's wakes another (command_buffer/completion.rs)",
        device.shared().queues.len()
    );
}
