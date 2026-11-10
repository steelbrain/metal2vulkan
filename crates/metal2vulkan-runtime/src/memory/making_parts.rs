//! The parts of each block's making, for the reclaim bench (`tests::reclaim_submit_bench`): how
//! long the buffer of addresses, `vkAllocateMemory`, the mapping and the buffer's binding took,
//! and how much of `vkAllocateMemory` the making thread spent on a CPU
//! (`/proc/thread-self/schedstat`), so that the bench can line each making up with the
//! submissions it held.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A block's making ([`super::BlockMaker::make_as`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MakingParts {
    /// The block's bytes.
    pub(crate) bytes: u64,
    /// Whether the device's thread that makes blocks ahead made it (`memory/ready.rs`).
    pub(crate) ahead: bool,
    /// When its `vkAllocateMemory` began.
    pub(crate) allocating: Instant,
    /// How long its buffer of addresses' making, `vkAllocateMemory`, the mapping and the
    /// buffer's binding took.
    pub(crate) took: [Duration; 4],
    /// The time the making thread ran on a CPU during `vkAllocateMemory`, and the time it waited
    /// for one.
    pub(crate) on_cpu: Duration,
    pub(crate) runnable: Duration,
}

/// Every making since the bench last drained it.
pub(crate) static MAKINGS: Mutex<Vec<MakingParts>> = Mutex::new(Vec::new());

/// The calling thread's time on a CPU and waiting for one so far (`/proc/thread-self/schedstat`),
/// zero where it is not readable.
pub(super) fn thread_times() -> (Duration, Duration) {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    let mut fields = text
        .split_whitespace()
        .map(|field| field.parse().unwrap_or(0));
    let mut next = || Duration::from_nanos(fields.next().unwrap_or(0));
    (next(), next())
}

/// `CLOCK_MONOTONIC` now, in seconds: what a kernel trace's `mono` clock reads.
pub(crate) fn monotonic() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock and a timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as f64 + now.tv_nsec as f64 * 1e-9
}

/// A line of `making`: when its `vkAllocateMemory` began, in `CLOCK_MONOTONIC` seconds from
/// `base` (an `Instant` and its [`monotonic`] reading), and how long it took (on a CPU, waiting
/// for one).
pub(crate) fn making_line(making: &MakingParts, base: (Instant, f64)) -> String {
    let ms = |took: Duration| took.as_secs_f64() * 1e3;
    let at = |instant: Instant| {
        base.1 + instant.saturating_duration_since(base.0).as_secs_f64()
            - base.0.saturating_duration_since(instant).as_secs_f64()
    };
    format!(
        "{} MiB {} at {:.4}: vkAllocateMemory {:.2} ms (on a CPU {:.2}, waiting for one {:.2})",
        making.bytes >> 20,
        if making.ahead { "ahead" } else { "in place" },
        at(making.allocating),
        ms(making.took[1]),
        ms(making.on_cpu),
        ms(making.runnable),
    )
}
