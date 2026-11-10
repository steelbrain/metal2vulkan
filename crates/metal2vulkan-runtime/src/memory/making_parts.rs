//! The parts of each block's making, for the reclaim bench (`tests::reclaim_submit_bench`) and the
//! fills' test (`tests::block_fills`): how long the buffer of addresses, the fill, `vkAllocateMemory`,
//! the mapping and the buffer's binding took, how much of `vkAllocateMemory` the making thread
//! spent on a CPU (`/proc/thread-self/schedstat`), and, where the block was filled through an
//! exportable allocation first, that allocation's and its free's parts and the CPUs the making
//! ran on, so that the bench can line each making up with the submissions it held and a missed
//! fill shows where its time went.

use std::cell::Cell;
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
    /// Where its pages were filled through an exportable allocation first, that fill's parts.
    pub(crate) fill: Option<FillParts>,
}

/// The parts of a block's fill through an exportable allocation (`memory.rs`'s
/// `BlockMaker::fill_first`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FillParts {
    /// How long the exportable allocation took and its free took.
    pub(crate) allocate: Duration,
    pub(crate) free: Duration,
    /// The time the making thread ran on a CPU during the exportable allocation, and during its
    /// free.
    pub(crate) allocate_on_cpu: Duration,
    pub(crate) free_on_cpu: Duration,
    /// The CPUs the making thread ran on as the exportable allocation began, as it ended, as its
    /// free ended and as the block's own allocation ended (`sched_getcpu`).
    pub(crate) cpus: [i32; 4],
}

/// Every making since the bench last drained it.
pub(crate) static MAKINGS: Mutex<Vec<MakingParts>> = Mutex::new(Vec::new());

thread_local! {
    /// The parts of the fill the calling thread made last, taken by its block's making.
    static FILL: Cell<Option<FillParts>> = const { Cell::new(None) };
}

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

/// The CPU the calling thread runs on.
pub(super) fn cpu() -> i32 {
    // SAFETY: no argument; it reads the calling thread's CPU.
    unsafe { libc::sched_getcpu() }
}

/// A fill's moments ([`FILL`]): as its allocation began, as it ended and as its free ended, each
/// with the thread's times and CPU.
pub(super) type FillMoment = (Instant, (Duration, Duration), i32);

/// The calling thread's moment now, for a fill's parts.
pub(super) fn moment() -> FillMoment {
    (Instant::now(), thread_times(), cpu())
}

/// Record the calling thread's fill of these moments, for its block's making to take.
pub(super) fn filled(began: FillMoment, allocated: FillMoment, freed: FillMoment) {
    FILL.set(Some(FillParts {
        allocate: allocated.0 - began.0,
        free: freed.0 - allocated.0,
        allocate_on_cpu: allocated.1 .0.saturating_sub(began.1 .0),
        free_on_cpu: freed.1 .0.saturating_sub(allocated.1 .0),
        cpus: [began.2, allocated.2, freed.2, -1],
    }));
}

/// The calling thread's last fill, taken, the block's own allocation having ended on CPU `cpu`.
pub(super) fn take_fill(cpu: i32) -> Option<FillParts> {
    FILL.take().map(|mut fill| {
        fill.cpus[3] = cpu;
        fill
    })
}

/// A line of `making`'s fill, where it was filled first: the exportable allocation's and its
/// free's times (on a CPU), the block's own allocation's (on a CPU, waiting for one), and the CPUs
/// each ended on.
pub(crate) fn fill_line(making: &MakingParts) -> Option<String> {
    let fill = making.fill?;
    let ms = |took: Duration| took.as_secs_f64() * 1e3;
    Some(format!(
        "{} MiB {}: fill allocated {:.2} ms (on a CPU {:.2}), freed {:.2} ms (on a CPU {:.2}); \
         own vkAllocateMemory {:.2} ms (on a CPU {:.2}, waiting for one {:.2}), {}; CPUs {:?}",
        making.bytes >> 20,
        if making.ahead { "ahead" } else { "in place" },
        ms(fill.allocate),
        ms(fill.allocate_on_cpu),
        ms(fill.free),
        ms(fill.free_on_cpu),
        ms(making.took[2]),
        ms(making.on_cpu),
        ms(making.runnable),
        if making.took[2] < super::FILL_SAVED_BELOW {
            "saved"
        } else {
            "missed"
        },
        fill.cpus,
    ))
}
