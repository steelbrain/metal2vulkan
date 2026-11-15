//! A command buffer whose second batch waits at its host step's gate for the GPU's long work
//! before it (a rate-mapped pass executing an indirect command buffer whose records that work
//! resets, `tests/host_step_rules.rs`'s workload), and an unshared queue's command buffer
//! committed while that work runs (`scheduler/order.rs`, **Held batches give way**):
//!
//! - one small fill of a buffer of its own, which shares nothing with the pass: its submission
//!   waits for no host step, whether the pass's uses are bounded (it counts no visibility results,
//!   `DeferredPass::bound`) or not known until its step records it (it does: `tracker::ordered`'s
//!   `None`). On an RTX 3090 a step so held every queue's commits, a compositor's unshared ones
//!   included, until an 89 ms pass before it ended; on lavapipe, before the step gave way, the
//!   fill of the second kind went 111 ms after its commit, as the pass's batch did;
//! - a reset of the records the pass's step reads: it waits behind the step's batch, as the gate
//!   covers no write submitted after it.
//!
//! And the long command buffer itself, and a small fill with a submission handler committed after
//! it on its queue, read `Scheduled` while the step waits (`scheduler/mod.rs`, **Scheduled**), as
//! Metal's do once their dependencies are resolved: before, they read so with their last batches,
//! once the long work ended. The long one's commit call returns within a frame at least once of
//! the runs, as Metal's returns at once: its first translates the variants the pass's host step
//! draws with ([`COMMIT`]).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::host_step_rules::{Heard, Workload};
use crate::{
    Buffer, CommandBuffer, CommandQueue, LoadAction, RenderPassDescriptor, ResourceOptions,
    StoreAction, TextureDescriptor, VisibilityResultMode, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The bytes one fill of the long work writes: few fills make it on a fast GPU, so the host's
/// share of the long command buffer (its fills' encoding, planning and submission, before its
/// pass's batch reaches its gate) stays small whatever the GPU's speed. At 64 MiB the RX 7900 XTX's
/// calibration took about 1400 fills, and on that loaded suite leg the long one read `Scheduled`
/// 12.6 and 17.1 ms after its commit, its follower 0.35 and 2.2 ms after its own.
const FILL_BYTES: u64 = 256 << 20;

/// The long work's length the fills are calibrated to, in seconds.
const LONG: f64 = 0.1;

/// The runs a leg makes, at most, for one whose fill was committed before the pass's batch went
/// (the test's thread late on a loaded host shows nothing).
const RUNS: usize = 3;

/// The bound on the time from a commit to `Scheduled` behind the step, in seconds: a scheduler's
/// pass and a handler thread's wake; a tenth of the long work, which held it before.
const SCHEDULED: f64 = LONG / 10.0;

/// The bound on the long command buffer's commit call by its own clock, in seconds, at least once
/// of the runs: a frame's 16 ms, past which the call holds the program's thread where Metal's
/// returns at once. Apart from [`SCHEDULED`], which runs from the commit's return: the first
/// commit of the pass translates the variants its host step draws with on the program's thread
/// (`DeferredPass::settle`), which a later one finds made. The whole call is that recording, on a
/// CPU: no pipeline wait, no wait for the scheduler's lock, no submission. On a Radeon 780M
/// (Ryzen 7 7840HS) with the GPU to itself a release build's first call took 3.6 to 4.2 ms and a
/// later one 0.10 to 0.26 ms, this unoptimized test build's 27 to 30 ms and 1.1 to 1.2 ms. Beside
/// the 15 other processes of the device suite (`.config/nextest.toml` runs this test alone) the
/// same work took many times the CPU time and waited for a CPU too: a release build's later calls
/// 5.8 and 7.0 ms (4.7 and 7.3 ms on a CPU), the test build's 10.7 to 38 ms.
const COMMIT: f64 = 0.016;

/// What the unshared queue's command buffer does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unshared {
    /// One small fill of a buffer of its own.
    Fill,
    /// A reset of the pass's empty command, in the records its step reads.
    Reset,
}

/// One run's host times, in seconds of `host_time`'s clock, and its holds.
struct Run {
    /// The unshared queue's commit returned.
    committed: f64,
    /// The scheduler called its submission (its kernel end).
    called: f64,
    /// The long command buffer's GPU end, after the pass's batch went: once the long work
    /// completed and the step recorded the pass.
    long_end: f64,
    /// Looks that held a head behind an earlier commit
    /// (`WorkCounts::heads_held_for_earlier_commits`).
    held: u64,
}

impl Run {
    /// Whether the unshared one was committed at least half the long work before it ended: else
    /// the test's thread came late (a loaded host), and the run shows nothing.
    fn early(&self) -> bool {
        self.long_end - self.committed > LONG / 2.0
    }

    /// Whether the unshared one was submitted past half of the long work, from its commit to the
    /// long work's end, by this run's own times: once the fills ran, which are most of the long
    /// work, the pass's own batch going after them. One that went at its commit, as the fills of
    /// the other legs do, went within a millisecond of it. Not a bound of the long work's length:
    /// on a GPU the suite shares, the long work ended 220 to 460 ms after the commit, and the
    /// pass's batch waited its turn too, ending 35 ms and 118 ms after the reset went on two
    /// Radeon 780M suite legs, past the quarter of the calibrated length this asked before.
    fn past_the_fills(&self) -> bool {
        self.called - self.committed > (self.long_end - self.committed) / 2.0
    }
}

/// The fills that take about [`LONG`] of `queue`'s GPU.
fn long_fills(queue: &CommandQueue, target: &Buffer) -> usize {
    let mut fills = 16;
    for _ in 0..3 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for fill in 0..fills {
            blit.fill_buffer(target, 0..FILL_BYTES, fill as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        fills = ((fills as f64 * LONG / gpu.max(1e-6)) as usize).clamp(2, 4096);
    }
    fills
}

/// The long command buffer: `fills` fills of `big`, a reset of the buffer's empty command after
/// them, then the pass executing both commands, counting visibility results where `counted`.
fn long(work: &Workload, big: &Buffer, fills: usize, counted: bool) -> CommandBuffer {
    let size = work.map.physical_size_for_layer(0);
    let mut d =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    d.usage = TextureUsage::RENDER_TARGET;
    let target = work.device.new_texture(&d).unwrap();
    let command_buffer = work.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for fill in 0..fills {
        blit.fill_buffer(big, 0..FILL_BYTES, fill as u8).unwrap();
    }
    // The GPU writes the records last: the pass's step waits for the whole batch.
    blit.reset_commands_in_buffer(&work.icb, 1..2).unwrap();
    blit.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    pass.rasterization_rate_map = Some(work.map.clone());
    pass.visibility_result_buffer = Some(work.visibility.clone());
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&work.pipeline).unwrap();
    if counted {
        (encoder.set_visibility_result_mode(VisibilityResultMode::Boolean, 0)).unwrap();
    }
    encoder.execute_commands_in_buffer(&work.icb, 0..2).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// A commit call's time by its own clock (`WorkCounts::commit_nanos`) and where it went, in
/// seconds: its phases (`commit_times.rs`), and the calling thread's time on a CPU
/// ([`thread_cpu`]) and on a run queue ([`run_queue_wait`]) over the call where the host reports
/// them, which tell the call's own work from a loaded host's.
#[derive(Clone, Copy)]
struct Call {
    whole: f64,
    /// Lowering, recording, pipeline waits, the scheduler's lock, its pass, submissions.
    phases: [f64; 6],
    /// On a CPU.
    cpu: Option<f64>,
    /// Runnable on a run queue.
    queued: Option<f64>,
}

impl std::fmt::Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [lowering, recording, waits, lock, pass, submit] = self.phases.map(|at| at * 1e3);
        write!(
            f,
            "{:.2} ms by its own clock: lowering {lowering:.2}, recording {recording:.2}, \
             pipeline waits {waits:.2}, the scheduler's lock {lock:.2}, its pass {pass:.2}, \
             submissions {submit:.2}",
            self.whole * 1e3
        )?;
        if let Some(cpu) = self.cpu {
            write!(f, "; the thread on a CPU {:.2}", cpu * 1e3)?;
        }
        match self.queued {
            Some(queued) => write!(f, ", on a run queue {:.2}", queued * 1e3),
            None => Ok(()),
        }
    }
}

/// The calling thread's CPU time so far, in seconds (`CLOCK_THREAD_CPUTIME_ID`, which counts the
/// time it is running now too).
#[cfg(unix)]
fn thread_cpu() -> Option<f64> {
    let mut at = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `at` is a valid `timespec` for the call to write.
    let answer = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) };
    (answer == 0).then_some(at.tv_sec as f64 + at.tv_nsec as f64 * 1e-9)
}

/// The calling thread's CPU time so far: not read here.
#[cfg(not(unix))]
fn thread_cpu() -> Option<f64> {
    None
}

/// The calling thread's time runnable on a run queue so far, in seconds, where the host reports
/// it (Linux's `/proc/thread-self/schedstat`, whose first field, the time on a CPU, stands as of
/// the thread's last scheduler event and leaves out the slice it runs now).
fn run_queue_wait() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let waited = stat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(waited as f64 * 1e-9)
}

/// The commit times `WorkCounts` sums phase by phase, in nanoseconds, in [`Call::phases`]'s order.
fn commit_phases(counts: &WorkCounts) -> [u64; 6] {
    [
        counts.commit_lowering_nanos,
        counts.commit_recording_nanos,
        counts.commit_pipeline_wait_nanos,
        counts.commit_lock_wait_nanos,
        counts.commit_pass_nanos,
        counts.commit_submit_nanos,
    ]
}

/// Commit `long`, then wait until its pass's batch waits at its gate: the host times its commit
/// was called and returned, and the call by its own clock.
fn commit_to_its_gate(work: &Workload, long: &CommandBuffer) -> (f64, f64, Call) {
    let before = work.device.work_counts();
    let (cpu, queued) = (thread_cpu(), run_queue_wait());
    let called = crate::host_time();
    long.commit().unwrap();
    let returned = crate::host_time();
    let since = |was: Option<f64>, now: Option<f64>| Some(now? - was?);
    let (cpu, queued) = (since(cpu, thread_cpu()), since(queued, run_queue_wait()));
    let after = work.device.work_counts();
    let (was, now) = (commit_phases(&before), commit_phases(&after));
    let call = Call {
        whole: (after.commit_nanos - before.commit_nanos) as f64 * 1e-9,
        phases: std::array::from_fn(|at| (now[at] - was[at]) as f64 * 1e-9),
        cpu,
        queued,
    };
    let began = Instant::now();
    while work.device.work_counts().heads_waiting_at_their_gate
        == before.heads_waiting_at_their_gate
    {
        assert!(
            began.elapsed() < WAIT,
            "the pass's batch waited at its gate"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    (called, returned, call)
}

/// Commit the long command buffer, wait until its pass's batch waits at its gate, then commit the
/// unshared queue's command buffer, `what`.
fn run(
    work: &Workload,
    unshared: &CommandQueue,
    (big, small): (&Buffer, &Buffer),
    fills: usize,
    (counted, what): (bool, Unshared),
) -> Run {
    let long = long(work, big, fills, counted);
    let before = work.device.work_counts();
    commit_to_its_gate(work, &long);
    let other = unshared.command_buffer().unwrap();
    let mut blit = other.blit_command_encoder().unwrap();
    match what {
        Unshared::Fill => blit.fill_buffer(small, 0..256, 1).unwrap(),
        Unshared::Reset => blit.reset_commands_in_buffer(&work.icb, 1..2).unwrap(),
    }
    blit.end_encoding().unwrap();
    other.commit().unwrap();
    let committed = crate::host_time();
    other.wait_until_completed_for(WAIT).unwrap();
    long.wait_until_completed_for(WAIT).unwrap();
    let after = work.device.work_counts();
    Run {
        committed,
        called: other.kernel_end_time(),
        long_end: long.gpu_end_time().unwrap(),
        held: after.heads_held_for_earlier_commits - before.heads_held_for_earlier_commits,
    }
}

/// The unshared queue's fill, the pass's uses bounded or not known, is submitted with no look
/// holding it behind an earlier commit, before half the long work ran; its reset of the records
/// the step reads waits behind the pass's batch, held so, until the long work's fills ran: past
/// half of the long work, by the run's own times ([`Run::past_the_fills`]).
#[test]
fn an_unshared_queue_s_fill_does_not_wait_for_another_queue_s_host_step() {
    let scratch = Scratch::new("held-step-order");
    let work = Workload::new(&scratch, &Heard::default());
    let unshared = work.device.new_command_queue_unshared(64).unwrap();
    let private = ResourceOptions::private();
    let big = work.device.new_buffer(FILL_BYTES, private).unwrap();
    let small = work.device.new_buffer(4096, private).unwrap();
    let fills = long_fills(&work.queue, &big);
    let legs = [
        ("a fill, the pass bounded", false, Unshared::Fill),
        ("a fill, the pass's uses not known", true, Unshared::Fill),
        ("a reset of the records", false, Unshared::Reset),
    ];
    for (leg, counted, what) in legs {
        let run = (0..RUNS)
            .map(|_| run(&work, &unshared, (&big, &small), fills, (counted, what)))
            .inspect(|run| {
                println!(
                    "{leg}: committed at 0, called at {:.2} ms; the long work's end {:.2} ms; {} \
                     looks held behind an earlier commit",
                    (run.called - run.committed) * 1e3,
                    (run.long_end - run.committed) * 1e3,
                    run.held
                )
            })
            .find(Run::early)
            .expect("a run committed the unshared one before half the long work ran");
        match what {
            Unshared::Fill => {
                assert_eq!(run.held, 0, "{leg}: no look held it behind the pass");
                assert!(
                    run.called - run.committed < LONG / 2.0,
                    "{leg}: it went before half the long work ran"
                );
            }
            Unshared::Reset => {
                assert!(run.held > 0, "{leg}: a look held it behind the pass");
                assert!(
                    run.past_the_fills(),
                    "{leg}: it went once the long work's fills ran"
                );
            }
        }
    }
}

/// One run of [`command_buffers_behind_a_waiting_step_read_scheduled`]'s: host times, in seconds
/// of `host_time`'s clock.
struct Early {
    /// The long command buffer's commit was called, returned, and its kernel end (made
    /// `Scheduled`).
    long: (f64, f64, f64),
    /// Its commit call, by the call's own clock (`WorkCounts::commit_nanos`).
    long_commit: Call,
    /// The follower's commit was called, its submission handler ran, and its
    /// `wait_until_scheduled` returned.
    follower: (f64, f64, f64),
    /// The follower's kernel start and end, and GPU start and end.
    times: [f64; 4],
    /// The long command buffer's GPU end.
    long_end: f64,
    /// Command buffers made `Scheduled` behind a host step
    /// (`WorkCounts::scheduled_behind_host_steps`).
    made: u64,
}

impl Early {
    /// Whether the follower was committed at least twice [`SCHEDULED`] before the long work
    /// ended, by this run's own times: a command buffer that read `Scheduled` only once the long
    /// work ended would then read it past the bound, so the run tells the two apart. Else the
    /// test's thread came late (a loaded host), or the long work ran short of its calibration,
    /// and the run shows nothing.
    fn early(&self) -> bool {
        self.long_end - self.follower.0 > 2.0 * SCHEDULED
    }

    /// The time from the long one's commit's return to its `Scheduled`.
    fn long_scheduled(&self) -> f64 {
        self.long.2 - self.long.1
    }

    /// The time the long one's commit call took on the host's clock.
    fn long_call(&self) -> f64 {
        self.long.1 - self.long.0
    }

    /// The time from the follower's commit to its `Scheduled`.
    fn follower_scheduled(&self) -> f64 {
        self.follower.2 - self.follower.0
    }

    /// The later of the two.
    fn scheduled(&self) -> f64 {
        self.long_scheduled().max(self.follower_scheduled())
    }
}

/// Commit the long command buffer, wait until its pass's batch waits at its gate, then commit the
/// follower on its queue, a small fill of `small` with a submission handler, and wait for both.
fn early_run(work: &Workload, (big, small): (&Buffer, &Buffer), fills: usize) -> Early {
    let long = long(work, big, fills, false);
    let before = work.device.work_counts();
    let (long_committed, long_returned, long_commit) = commit_to_its_gate(work, &long);
    let follower = work.queue.command_buffer().unwrap();
    let mut blit = follower.blit_command_encoder().unwrap();
    blit.fill_buffer(small, 0..256, 1).unwrap();
    blit.end_encoding().unwrap();
    let handled = Arc::new(Mutex::new(None));
    let at = Arc::clone(&handled);
    (follower.add_submission_handler(Box::new(move |_: &CommandBuffer| {
        *at.lock().unwrap() = Some(crate::host_time());
    })))
    .unwrap();
    let committed = crate::host_time();
    follower.commit().unwrap();
    follower.wait_until_scheduled().unwrap();
    let scheduled = crate::host_time();
    follower.wait_until_completed_for(WAIT).unwrap();
    long.wait_until_completed_for(WAIT).unwrap();
    let after = work.device.work_counts();
    let handler = (*handled.lock().unwrap()).expect("the follower's submission handler ran");
    Early {
        long: (long_committed, long_returned, long.kernel_end_time()),
        long_commit,
        follower: (committed, handler, scheduled),
        times: [
            follower.kernel_start_time(),
            follower.kernel_end_time(),
            follower.gpu_start_time().unwrap(),
            follower.gpu_end_time().unwrap(),
        ],
        long_end: long.gpu_end_time().unwrap(),
        made: after.scheduled_behind_host_steps - before.scheduled_behind_host_steps,
    }
}

/// The long command buffer, its pass's batch waiting at its step's gate for the long work, and a
/// follower committed after it on its queue, a small fill with a submission handler, read
/// `Scheduled` within [`SCHEDULED`] of their commits (the long one's return, the follower's
/// call), the follower's handler first, while the long work runs (`scheduler/mod.rs`,
/// **Scheduled**; the hang's thread makes the command buffers behind a GPU hang so,
/// `command_buffer/hang.rs`): Metal's are `Scheduled` once their dependencies are resolved.
/// Before, each read so with its last batch, after the long work. The follower's times stay in its
/// life's order: its commit, kernel start and end, GPU start and end (`gpu_time.rs`). Of [`RUNS`]
/// runs that each tell the two apart ([`Early::early`]), the one whose later of the two read
/// `Scheduled` soonest: on lavapipe the first run's long one read it 25.5 ms after its commit, the
/// later runs' 1.9 ms. And the long one's commit call returned within [`COMMIT`] in one of the
/// runs at least.
#[test]
fn command_buffers_behind_a_waiting_step_read_scheduled() {
    let scratch = Scratch::new("held-step-scheduled");
    let work = Workload::new(&scratch, &Heard::default());
    let private = ResourceOptions::private();
    let big = work.device.new_buffer(FILL_BYTES, private).unwrap();
    let small = work.device.new_buffer(4096, private).unwrap();
    let fills = long_fills(&work.queue, &big);
    let runs: Vec<Early> = (0..RUNS)
        .map(|_| early_run(&work, (&big, &small), fills))
        .inspect(|run| {
            let since = |at: f64| (at - run.follower.0) * 1e3;
            println!(
                "{fills} fills: the long one's commit returned {:.2} ms after its call ({}), \
                 Scheduled {:.2} ms after its return; the follower committed at 0, its handler at \
                 {:.2} ms, Scheduled at {:.2} ms, its kernel end {:.2} ms; the long work's end \
                 {:.2} ms; {} made Scheduled behind the step",
                run.long_call() * 1e3,
                run.long_commit,
                run.long_scheduled() * 1e3,
                since(run.follower.1),
                since(run.follower.2),
                since(run.times[1]),
                since(run.long_end),
                run.made
            )
        })
        .collect();
    let commit = (runs.iter().map(|run| run.long_commit.whole)).fold(f64::INFINITY, f64::min);
    assert!(
        commit < COMMIT,
        "the long one's commit call returned within a frame in a run"
    );
    let run = (runs.into_iter())
        .filter(Early::early)
        .min_by(|a, b| a.scheduled().total_cmp(&b.scheduled()))
        .expect("a run committed the follower twice the bound before the long work ended");
    assert!(
        run.long_scheduled() < SCHEDULED,
        "the long one read Scheduled while its step waited"
    );
    assert!(
        run.follower_scheduled() < SCHEDULED,
        "the follower read Scheduled while the step waited"
    );
    let (committed, handler, scheduled) = run.follower;
    assert!(
        committed <= handler && handler <= scheduled,
        "the follower's submission handler ran before it read Scheduled"
    );
    assert_eq!(run.made, 2, "both were made Scheduled behind the step");
    let [kernel_start, kernel_end, gpu_start, gpu_end] = run.times;
    assert!(
        committed <= kernel_start
            && kernel_start <= kernel_end
            && kernel_end <= gpu_start
            && gpu_start <= gpu_end,
        "the follower's times in its life's order: {:?}",
        run.times
    );
}
