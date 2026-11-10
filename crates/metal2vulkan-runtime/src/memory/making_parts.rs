//! The parts of each block's making, for the reclaim bench (`tests::reclaim_submit_bench`): how
//! long the buffer of addresses, the fill, `vkAllocateMemory`, the mapping and the buffer's
//! binding took,
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
    /// How long its buffer of addresses' making, its pages' fill through an exportable
    /// allocation (`memory.rs`, "Blocks filled exportable first"), `vkAllocateMemory`, the
    /// mapping and the buffer's binding took.
    pub(crate) took: [Duration; 5],
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
