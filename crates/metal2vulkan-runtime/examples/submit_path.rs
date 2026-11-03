//! Not a consumer's example: a command buffer's fixed cost on the submit and completion path, the
//! shape of a consumer that records a command buffer, adds handlers, commits and never waits.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example submit_path [-- latency|cpu|threads|empty|idle|all [--seen] [--bare] [--at-commit]]
//! ```
//!
//! - **latency**: one thread commits a tiny blit command buffer (a fill of 16 bytes) with a
//!   scheduled handler and a completed handler, or a completion-seen handler instead of the
//!   completed one (`seen`), and does not wait for it: it blocks on a condition variable the
//!   handler signals as it starts. Timed: `commit` to the handler's start, and the command
//!   buffer's `GPUEndTime` to the handler's start (both on the runtime's host clock,
//!   `host_time`), at idle gaps of 0, 100 us, 300 us and 2 ms between command buffers. With
//!   `--bare`, also with no scheduled handler: a lone fill and its one callback.
//! - **cpu**: the same command buffers committed as fast as the queue's slots allow, never waited
//!   for: command buffers a second, and the CPU per command buffer of the runtime's threads (by
//!   thread on Linux, from `/proc/self/task/*/schedstat`; elsewhere the process's CPU less the
//!   committing thread's).
//! - **threads**: the cpu shape on 1 and on 8 threads at once, each with a queue of its own:
//!   command buffers a second in all, and the CPU per command buffer.
//! - **empty**: a command buffer with nothing encoded, committed and waited for, back to back and
//!   at a 300 us gap; and committed as fast as the slots allow, never waited for.
//! - **idle**: after one fill committed and waited for, wakes a second over [`IDLE_SECONDS`] with
//!   nothing committed: on Linux the voluntary context switches by thread group, from
//!   `/proc/self/task/*/status`; everywhere the scheduler's sleeps and the sleeps that ended at
//!   its slice, from the device's work counts (macOS's per-process switch counts read back
//!   inconsistently, so none is printed there).
//!
//! With `--at-commit`, latency, cpu and threads commit with `CommandBuffer::commit_and_submit`
//! (a runtime extension: the committing thread submits where no pass is under way) instead of
//! `commit`.

use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandQueue, Device, ResourceOptions,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Whether command buffers are committed with `commit_and_submit` (`--at-commit`).
static AT_COMMIT: AtomicBool = AtomicBool::new(false);

/// Commit `command_buffer` as `--at-commit` says.
fn commit(command_buffer: &CommandBuffer) {
    match AT_COMMIT.load(Ordering::Relaxed) {
        true => command_buffer.commit_and_submit().unwrap(),
        false => command_buffer.commit().unwrap(),
    }
}

/// Command buffers per measured row.
const ROUNDS: usize = 2000;

/// How long a full-rate row runs.
const RATE_SECONDS: f64 = 3.0;

/// How long the idle row watches the process.
const IDLE_SECONDS: f64 = 3.0;

/// The idle gaps of the latency rows.
const GAPS_US: [u64; 4] = [0, 100, 300, 2000];

/// Which handler the latency and cpu rows signal from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Callback {
    /// A completed handler, with a scheduled handler before it.
    Completed,
    /// A completion-seen handler, with a scheduled handler before it.
    Seen,
}

impl Callback {
    fn name(self) -> &'static str {
        match self {
            Callback::Completed => "completed",
            Callback::Seen => "seen",
        }
    }
}

/// A one-shot signal from a handler to the committing thread: the instant the handler started,
/// and the seconds from its command buffer's `GPUEndTime` to then on the host clock.
#[derive(Default)]
struct Signal {
    at: Mutex<Option<(Instant, f64)>>,
    changed: Condvar,
}

impl Signal {
    fn set(&self, after_gpu_end: f64) {
        let now = Instant::now();
        *self.at.lock().unwrap() = Some((now, after_gpu_end));
        self.changed.notify_one();
    }

    fn take(&self) -> (Instant, f64) {
        let mut at = self.at.lock().unwrap();
        loop {
            if let Some(at) = at.take() {
                return at;
            }
            at = self.changed.wait(at).unwrap();
        }
    }
}

/// A tiny blit command buffer: a fill of 16 bytes of `target`, with a scheduled handler where
/// `scheduled`, and the handler `callback` names, which runs `then` with the seconds from the
/// command buffer's `GPUEndTime` to its start.
fn tiny(
    queue: &CommandQueue,
    target: &Buffer,
    (callback, scheduled): (Callback, bool),
    then: impl FnOnce(f64) + Send + 'static,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..16, 0xa5).unwrap();
    blit.end_encoding().unwrap();
    if scheduled {
        command_buffer
            .add_scheduled_handler(Box::new(|_: &CommandBuffer| {}))
            .unwrap();
    }
    let handler = Box::new(move |buffer: &CommandBuffer| {
        let now = host_time();
        then(now - buffer.gpu_end_time().unwrap_or(now))
    });
    match callback {
        Callback::Completed => command_buffer.add_completed_handler(handler).unwrap(),
        Callback::Seen => command_buffer.add_completion_seen_handler(handler).unwrap(),
    }
    command_buffer
}

/// Sleep `gap` microseconds (none for 0).
fn idle(gap: u64) {
    if gap > 0 {
        std::thread::sleep(Duration::from_micros(gap));
    }
}

/// The p50, p90 and p99 of `samples`, in microseconds.
fn percentiles(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let at = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)];
    (at(0.5), at(0.9), at(0.99))
}

fn latency(device: &Device, callback: Callback, scheduled: bool) {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let name = match scheduled {
        true => callback.name().to_owned(),
        false => format!("{} bare", callback.name()),
    };
    for gap in GAPS_US {
        let signal = Arc::new(Signal::default());
        let (mut samples, mut after_end) = (Vec::with_capacity(ROUNDS), Vec::new());
        for round in 0..ROUNDS + 50 {
            idle(gap);
            let signalled = Arc::clone(&signal);
            let command_buffer = tiny(&queue, &target, (callback, scheduled), move |after| {
                signalled.set(after)
            });
            let committed = Instant::now();
            commit(&command_buffer);
            drop(command_buffer);
            let (started, after) = signal.take();
            if round >= 50 {
                samples.push(started.duration_since(committed).as_secs_f64() * 1e6);
                after_end.push(after * 1e6);
            }
        }
        let (p50, p90, p99) = percentiles(&mut samples);
        let (e50, e90, e99) = percentiles(&mut after_end);
        println!(
            "latency {name:<14} gap {gap:>4} us: commit to handler p50 {p50:7.1} p90 {p90:7.1} \
             p99 {p99:7.1} us; GPU end to handler p50 {e50:7.1} p90 {e90:7.1} p99 {e99:7.1} us"
        );
    }
}

/// The group a thread named `comm` is counted in.
fn group_of(comm: &str) -> &'static str {
    match comm {
        c if c.starts_with("metal2vulkan-sc") => "scheduler",
        c if c.starts_with("metal2vulkan-qu") => "queue-waiters",
        c if c.starts_with("metal2vulkan-ha") => "handlers",
        c if c.starts_with("metal2vulkan") => "other-m2v",
        c if c.starts_with("submit_path") => "committer+main",
        _ => "driver/other",
    }
}

/// Voluntary context switches by thread group, each of the process's threads by its name's
/// prefix: Linux only, none elsewhere.
fn switches_by_group() -> Vec<(String, u64)> {
    let mut groups: Vec<(String, u64)> = Vec::new();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let path = task.path();
            let comm = std::fs::read_to_string(path.join("comm")).unwrap_or_default();
            let status = std::fs::read_to_string(path.join("status")).unwrap_or_default();
            let switches: u64 = status
                .lines()
                .find_map(|line| line.strip_prefix("voluntary_ctxt_switches:"))
                .and_then(|n| n.trim().parse().ok())
                .unwrap_or(0);
            let group = group_of(comm.trim()).to_owned();
            match groups.iter_mut().find(|(n, _)| *n == group) {
                Some((_, total)) => *total += switches,
                None => groups.push((group, switches)),
            }
        }
    }
    groups
}

fn idle_wakes(device: &Device) {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let signal = Arc::new(Signal::default());
    let signalled = Arc::clone(&signal);
    let command_buffer = tiny(&queue, &target, (Callback::Completed, true), move |after| {
        signalled.set(after)
    });
    command_buffer.commit().unwrap();
    signal.take();
    command_buffer.wait_until_completed().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let (before, counts) = (switches_by_group(), device.work_counts());
    std::thread::sleep(Duration::from_secs_f64(IDLE_SECONDS));
    let (after, later) = (switches_by_group(), device.work_counts());
    if !after.is_empty() {
        let mut line = "idle wakes a second:".to_owned();
        for (name, count) in &after {
            let earlier = before
                .iter()
                .find(|(n, _)| n == name)
                .map_or(0, |(_, n)| *n);
            let per = count.saturating_sub(earlier) as f64 / IDLE_SECONDS;
            line.push_str(&format!(" {name} {per:.1}"));
        }
        println!("{line} (voluntary context switches, the main thread's one sleep among them)");
    }
    println!(
        "idle scheduler a second: {:.1} sleeps, {:.1} ended at the slice",
        (later.scheduler_sleeps - counts.scheduler_sleeps) as f64 / IDLE_SECONDS,
        (later.scheduler_sleep_slices - counts.scheduler_sleep_slices) as f64 / IDLE_SECONDS,
    );
}

/// Nanoseconds of CPU by thread group: on Linux each of the process's threads by its name's
/// prefix; elsewhere `other` alone (the process less the calling thread).
fn cpu_by_group() -> Vec<(String, u64)> {
    let mut groups: Vec<(String, u64)> = Vec::new();
    let mut add = |name: String, nanos: u64| match groups.iter_mut().find(|(n, _)| *n == name) {
        Some((_, total)) => *total += nanos,
        None => groups.push((name, nanos)),
    };
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let path = task.path();
            let comm = std::fs::read_to_string(path.join("comm")).unwrap_or_default();
            let stat = std::fs::read_to_string(path.join("schedstat")).unwrap_or_default();
            let nanos: u64 = stat
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            add(group_of(comm.trim()).to_owned(), nanos);
        }
        return groups;
    }
    // The process's CPU less this thread's.
    // SAFETY: plain calls with out-parameters of the right types.
    let (process, own) = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let process = (usage.ru_utime.tv_sec as u64 * 1_000_000
            + usage.ru_utime.tv_usec as u64
            + usage.ru_stime.tv_sec as u64 * 1_000_000
            + usage.ru_stime.tv_usec as u64)
            * 1000;
        let mut spec: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut spec);
        (
            process,
            spec.tv_sec as u64 * 1_000_000_000 + spec.tv_nsec as u64,
        )
    };
    add("committer".to_owned(), own);
    add("all-but-committer".to_owned(), process.saturating_sub(own));
    groups
}

/// This thread's CPU so far, in nanoseconds.
fn thread_cpu_nanos() -> u64 {
    // SAFETY: a plain call with an out-parameter of the right type.
    let spec = unsafe {
        let mut spec: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut spec);
        spec
    };
    spec.tv_sec as u64 * 1_000_000_000 + spec.tv_nsec as u64
}

/// The CPU each group spent between `before` and `after`, per command buffer, in microseconds.
fn print_cpu(label: &str, before: &[(String, u64)], after: &[(String, u64)], count: usize) {
    let mut line = format!("{label}:");
    for (name, nanos) in after {
        let earlier = before
            .iter()
            .find(|(n, _)| n == name)
            .map_or(0, |(_, n)| *n);
        let per = (nanos.saturating_sub(earlier)) as f64 / 1e3 / count as f64;
        line.push_str(&format!(" {name} {per:.2}"));
    }
    println!("{line} (us of CPU per command buffer)");
}

fn cpu(device: &Device, callback: Callback) {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    // Warm.
    for _ in 0..200 {
        commit(&tiny(&queue, &target, (callback, true), |_| {}));
    }
    std::thread::sleep(Duration::from_millis(200));
    let before = cpu_by_group();
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < RATE_SECONDS {
        commit(&tiny(&queue, &target, (callback, true), |_| {}));
        count += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let after = cpu_by_group();
    println!(
        "cpu {:<9} {:.0} command buffers a second ({count} in {elapsed:.2} s)",
        callback.name(),
        count as f64 / elapsed
    );
    print_cpu(
        &format!("cpu {:<9}", callback.name()),
        &before,
        &after,
        count,
    );
    let last = queue.command_buffer().unwrap();
    last.commit().unwrap();
    last.wait_until_completed().unwrap();
}

fn empty(device: &Device) {
    let queue = device.new_command_queue().unwrap();
    for gap in [0, 300] {
        let mut samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS + 50 {
            idle(gap);
            let command_buffer = queue.command_buffer().unwrap();
            let committed = Instant::now();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed().unwrap();
            if round >= 50 {
                samples.push(committed.elapsed().as_secs_f64() * 1e6);
            }
        }
        let (p50, p90, p99) = percentiles(&mut samples);
        println!(
            "empty commit and wait, gap {gap:>4} us: p50 {p50:7.1} p90 {p90:7.1} p99 {p99:7.1} us"
        );
    }
    let before_counts = device.work_counts();
    let before = cpu_by_group();
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < RATE_SECONDS {
        queue.command_buffer().unwrap().commit().unwrap();
        count += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let after = cpu_by_group();
    let last = queue.command_buffer().unwrap();
    last.commit().unwrap();
    last.wait_until_completed().unwrap();
    let after_counts = device.work_counts();
    println!(
        "empty never waited: {:.0} command buffers a second ({count} in {elapsed:.2} s), {} queue submissions",
        count as f64 / elapsed,
        after_counts.queue_submissions - before_counts.queue_submissions
    );
    print_cpu("empty never waited", &before, &after, count);
}

/// The cpu shape on `threads` threads at once, each with a queue of its own.
fn threads(device: &Device, threads: usize) {
    let queues: Vec<(CommandQueue, Buffer)> = (0..threads)
        .map(|_| {
            let queue = device.new_command_queue().unwrap();
            let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            (queue, target)
        })
        .collect();
    for (queue, target) in &queues {
        for _ in 0..200 {
            commit(&tiny(queue, target, (Callback::Completed, true), |_| {}));
        }
    }
    std::thread::sleep(Duration::from_millis(200));
    let before = cpu_by_group();
    let start = Instant::now();
    // Each committing thread's own CPU, read before it ends (its schedstat goes with it).
    let (count, committers): (usize, u64) = std::thread::scope(|scope| {
        let running: Vec<_> = (queues.iter())
            .map(|(queue, target)| {
                scope.spawn(move || {
                    let (mut count, began) = (0, thread_cpu_nanos());
                    while start.elapsed().as_secs_f64() < RATE_SECONDS {
                        commit(&tiny(queue, target, (Callback::Completed, true), |_| {}));
                        count += 1;
                    }
                    (count, thread_cpu_nanos() - began)
                })
            })
            .collect();
        (running.into_iter())
            .map(|thread| thread.join().unwrap())
            .fold((0, 0), |(count, nanos), (more, spent)| {
                (count + more, nanos + spent)
            })
    });
    let elapsed = start.elapsed().as_secs_f64();
    let after = cpu_by_group();
    println!(
        "threads {threads}: {:.0} command buffers a second ({count} in {elapsed:.2} s), \
         committing threads {:.2} us of CPU per command buffer",
        count as f64 / elapsed,
        committers as f64 / 1e3 / count as f64
    );
    print_cpu(&format!("threads {threads}"), &before, &after, count);
    for (queue, _) in &queues {
        let last = queue.command_buffer().unwrap();
        last.commit().unwrap();
        last.wait_until_completed().unwrap();
    }
}

fn main() {
    let device = Device::system_default().unwrap();
    println!("device: {}", device.name());
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".to_owned());
    let seen = std::env::args().any(|arg| arg == "--seen");
    let bare = std::env::args().any(|arg| arg == "--bare");
    let at_commit = std::env::args().any(|arg| arg == "--at-commit");
    AT_COMMIT.store(at_commit, Ordering::Relaxed);
    println!(
        "committed with {}",
        if at_commit {
            "commit_and_submit"
        } else {
            "commit"
        }
    );
    let callbacks: &[Callback] = match seen {
        true => &[Callback::Completed, Callback::Seen],
        false => &[Callback::Completed],
    };
    if what == "latency" || what == "all" {
        for &callback in callbacks {
            latency(&device, callback, true);
        }
        if bare {
            for &callback in callbacks {
                latency(&device, callback, false);
            }
        }
    }
    if what == "cpu" || what == "all" {
        for &callback in callbacks {
            cpu(&device, callback);
        }
    }
    if what == "threads" || what == "all" {
        for count in [1, 8] {
            threads(&device, count);
        }
    }
    if what == "empty" || what == "all" {
        empty(&device);
    }
    if what == "idle" || what == "all" {
        idle_wakes(&device);
    }
}
