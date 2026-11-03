//! Not a consumer's example: a command buffer's fixed cost on the submit and completion path, the
//! shape of a consumer that records a command buffer, adds handlers, commits and never waits.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example submit_path [-- latency|cpu|empty|all]
//! ```
//!
//! - **latency**: one thread commits a tiny blit command buffer (a fill of 16 bytes) with a
//!   scheduled handler and a completed handler, or a completion-seen handler instead of the
//!   completed one (`seen`), and does not wait for it: it blocks on a condition variable the
//!   handler signals as it starts. Timed: `commit` to the handler's start, at idle gaps of 0,
//!   100 us, 300 us and 2 ms between command buffers.
//! - **cpu**: the same command buffers committed as fast as the queue's slots allow, never waited
//!   for: command buffers a second, and the CPU per command buffer of the runtime's threads (by
//!   thread on Linux, from `/proc/self/task/*/schedstat`; elsewhere the process's CPU less the
//!   committing thread's).
//! - **empty**: a command buffer with nothing encoded, committed and waited for, back to back and
//!   at a 300 us gap; and committed as fast as the slots allow, never waited for.

use metal2vulkan_runtime::{Buffer, CommandBuffer, CommandQueue, Device, ResourceOptions};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Command buffers per measured row.
const ROUNDS: usize = 2000;

/// How long a full-rate row runs.
const RATE_SECONDS: f64 = 3.0;

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

/// A one-shot signal from a handler to the committing thread: the instant the handler started.
#[derive(Default)]
struct Signal {
    at: Mutex<Option<Instant>>,
    changed: Condvar,
}

impl Signal {
    fn set(&self) {
        let now = Instant::now();
        *self.at.lock().unwrap() = Some(now);
        self.changed.notify_one();
    }

    fn take(&self) -> Instant {
        let mut at = self.at.lock().unwrap();
        loop {
            if let Some(at) = at.take() {
                return at;
            }
            at = self.changed.wait(at).unwrap();
        }
    }
}

/// A tiny blit command buffer: a fill of 16 bytes of `target`, with a scheduled handler and the
/// handler `callback` names, which runs `then`.
fn tiny(
    queue: &CommandQueue,
    target: &Buffer,
    callback: Callback,
    then: impl FnOnce() + Send + 'static,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..16, 0xa5).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
        .add_scheduled_handler(Box::new(|_: &CommandBuffer| {}))
        .unwrap();
    let handler = Box::new(move |_: &CommandBuffer| then());
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

fn latency(device: &Device, callback: Callback) {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    for gap in GAPS_US {
        let signal = Arc::new(Signal::default());
        let mut samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS + 50 {
            idle(gap);
            let signalled = Arc::clone(&signal);
            let command_buffer = tiny(&queue, &target, callback, move || signalled.set());
            let committed = Instant::now();
            command_buffer.commit().unwrap();
            drop(command_buffer);
            let started = signal.take();
            if round >= 50 {
                samples.push(started.duration_since(committed).as_secs_f64() * 1e6);
            }
        }
        let (p50, p90, p99) = percentiles(&mut samples);
        println!(
            "latency {:<9} gap {:>4} us: commit to handler p50 {p50:7.1} p90 {p90:7.1} p99 {p99:7.1} us",
            callback.name(),
            gap
        );
    }
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
            let comm = comm.trim();
            let group = match comm {
                c if c.starts_with("metal2vulkan-sc") => "scheduler",
                c if c.starts_with("metal2vulkan-qu") => "queue-waiters",
                c if c.starts_with("metal2vulkan-ha") => "handlers",
                c if c.starts_with("metal2vulkan") => "other-m2v",
                "submit_path" => "committer+main",
                _ => "driver/other",
            };
            add(group.to_owned(), nanos);
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
        tiny(&queue, &target, callback, || {}).commit().unwrap();
    }
    std::thread::sleep(Duration::from_millis(200));
    let before = cpu_by_group();
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < RATE_SECONDS {
        tiny(&queue, &target, callback, || {}).commit().unwrap();
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

fn main() {
    let device = Device::system_default().unwrap();
    println!("device: {}", device.name());
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".to_owned());
    let seen = std::env::args().any(|arg| arg == "--seen");
    let callbacks: &[Callback] = match seen {
        true => &[Callback::Completed, Callback::Seen],
        false => &[Callback::Completed],
    };
    if what == "latency" || what == "all" {
        for &callback in callbacks {
            latency(&device, callback);
        }
    }
    if what == "cpu" || what == "all" {
        for &callback in callbacks {
            cpu(&device, callback);
        }
    }
    if what == "empty" || what == "all" {
        empty(&device);
    }
}
