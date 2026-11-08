//! The device's checks step aside for its other work: before each step that allocates or frees
//! memory, maps or unmaps pages, or submits, a check's thread (`submit_check.rs`,
//! `queue_switch_check.rs`, `in_place_check.rs`) waits until no allocation or free of the
//! device's is in progress on another thread and none ended in the last [`QUIET`].
//!
//! - **Why.** A device whose checks have no record runs them on threads of their own as it is
//!   made (the submission checks) or at the first use that asks (the others), beside the program's
//!   first makings. Their imports, frees, mappings and submissions meet the program's own in the
//!   driver and the kernel. Measured on the RTX 5080 Laptop GPU (NVIDIA 615.71.09, Linux 7.3.1), in
//!   the consumer's orders of `tests/host_ram_window_bench.rs` over a guest RAM in 2 MiB pages
//!   (each take a heap and a buffer in it, its run imported alone): with the submission checks
//!   running beside them (a shader cache directory of the device's own, fresh), 11 of 36 runs'
//!   takes reached 5 to 10.7 ms, every one in the first two runs, while the checks ran (87 to 115
//!   ms, against 22 to 26 ms alone); with the checks read from a record, 2 of 36, at 5.1 and 5.2
//!   ms.
//! - **What counts.** The device's allocations and frees of memory, at their two doors
//!   (`creation::allocate_memory`, `release::object`), on every thread but a check's
//!   ([`mark_a_check_s_thread`]): the program's makings, their imports, and the runtime's own
//!   threads' work for them (the guest RAM's imports ahead, the free thread's frees). A making
//!   that allocates nothing (a buffer in a block already allocated) is not seen, nor waits on a
//!   check in the driver.
//! - **Never for a waiter.** A step steps aside for no one while a program's call waits for the
//!   check's verdicts (`SubmitChecks::wait`, `InPlaceChecks::wait`), so such a wait is the check's
//!   own run alone.
//! - **Never for long.** A step waits at most [`MOST_ASIDE`]: a program whose allocations and
//!   frees never pause for [`QUIET`] (one that allocates every few milliseconds) still sees each
//!   check go on, a step at least each [`MOST_ASIDE`], so its verdicts come, its steps a short
//!   share of that time.
//! - **What it costs.** A check's verdicts come later while the device's work goes on: until then
//!   the device takes the forms a device without them takes (each check's module docs). Counted:
//!   `WorkCounts::device_checks` (`steps_aside`, `aside_nanos`).
//! - **The checks' work is theirs.** The makings, releases and submissions of a thread whose every
//!   Vulkan call is its check's own (the submission checks' and the queue-switch check's, which
//!   reach the device's functions directly: [`Counted::Apart`]) are counted among the checks'
//!   (`WorkCounts::device_checks`), never the program's, so a program or a case that meets a
//!   device first does not carry them. The in-place check's are the device's
//!   ([`Counted::AsTheDevice`]): it makes its buffers, textures and command buffers through the
//!   program's paths, and a pass of the scheduler its waits take on its thread submits and
//!   releases the program's batches beside its own (`freed_memory.rs`'s in-place check test: the
//!   program's fills, submitted there, were never counted among `queue_submissions`).

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long the device's allocations and frees on other threads must have been quiet before a
/// check's step goes on: the gaps between a burst's makings are shorter, so a check's step does
/// not start inside a burst.
pub(crate) const QUIET: Duration = Duration::from_millis(20);

/// The longest a step waits before it goes on whatever the device does.
pub(crate) const MOST_ASIDE: Duration = Duration::from_millis(100);

/// The longest one sleep of a waiting step lasts before it looks again.
const SLICE: Duration = Duration::from_millis(2);

thread_local! {
    /// This thread is one of the device's checks' threads, whose own allocations and frees count
    /// for nothing here.
    static CHECKS: Cell<bool> = const { Cell::new(false) };
    /// Its makings, releases and submissions are counted as the checks' ([`Counted::Apart`]).
    static APART: Cell<bool> = const { Cell::new(false) };
}

/// Whose counts a check's thread's makings, releases and submissions are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Counted {
    /// The checks' (`WorkCounts::device_checks`): every Vulkan call of the thread is its check's
    /// own.
    Apart,
    /// The device's: the thread makes its check's objects through the program's paths, which may
    /// carry the program's work on it too.
    AsTheDevice,
}

/// Mark this thread as a check's, for good: it is a thread of its own, its work counted as
/// `counted` says.
pub(crate) fn mark_a_check_s_thread(counted: Counted) {
    CHECKS.with(|checks| checks.set(true));
    APART.with(|apart| apart.set(counted == Counted::Apart));
}

/// Whether this thread is a check's ([`mark_a_check_s_thread`]).
fn on_a_check_s_thread() -> bool {
    CHECKS.with(Cell::get)
}

/// Whether this thread's makings, releases and submissions are counted as the checks'
/// ([`Counted::Apart`]).
pub(crate) fn counted_apart() -> bool {
    APART.with(Cell::get)
}

/// One device's allocations and frees in progress on threads not its checks', and when the last
/// ended.
pub(crate) struct Activity {
    in_flight: AtomicU32,
    /// When the last ended, in nanoseconds past [`Activity::epoch`], plus one; 0 where none did.
    last: AtomicU64,
    epoch: Instant,
}

impl Activity {
    pub(crate) fn new() -> Activity {
        Activity {
            in_flight: AtomicU32::new(0),
            last: AtomicU64::new(0),
            epoch: Instant::now(),
        }
    }

    /// An allocation or free begins on this thread: whether it counts (not on a check's thread),
    /// in which case [`Activity::end`] follows it.
    pub(crate) fn begin(&self) -> bool {
        if on_a_check_s_thread() {
            return false;
        }
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// An allocation or free that [`Activity::begin`] counted ended.
    pub(crate) fn end(&self) {
        let nanos = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX - 1);
        self.last.fetch_max(nanos + 1, Ordering::AcqRel);
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// How long until a step may go on: zero where none is in progress and none ended within
    /// [`QUIET`].
    fn left(&self) -> Duration {
        if self.in_flight.load(Ordering::Acquire) > 0 {
            return SLICE;
        }
        let last = self.last.load(Ordering::Acquire);
        if last == 0 {
            return Duration::ZERO;
        }
        let since = self
            .epoch
            .elapsed()
            .saturating_sub(Duration::from_nanos(last - 1));
        QUIET.saturating_sub(since).min(SLICE)
    }

    /// Wait, on a check's thread, until a step may go on ([`Activity::left`]), `stop` says to
    /// go on now (the check stops, or a call waits for its verdicts) or it waited
    /// [`MOST_ASIDE`]: the time it waited, `None` where it went on at once.
    pub(crate) fn step_aside(&self, stop: impl Fn() -> bool) -> Option<Duration> {
        let began = Instant::now();
        let mut waited = false;
        loop {
            let left = self.left();
            let past = began.elapsed();
            if left.is_zero() || stop() || past >= MOST_ASIDE {
                return waited.then_some(past);
            }
            waited = true;
            std::thread::sleep(left.min(MOST_ASIDE - past));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_waits_while_an_allocation_is_in_progress_and_for_the_quiet_after_it() {
        let activity = Activity::new();
        assert_eq!(activity.left(), Duration::ZERO, "nothing yet");
        assert!(activity.begin());
        assert_eq!(activity.left(), SLICE, "in progress");
        activity.end();
        assert!(activity.left() > Duration::ZERO, "just ended");
        let waited = activity.step_aside(|| false).expect("it waited");
        assert!(waited >= QUIET - SLICE, "waited {waited:?}");
        assert_eq!(activity.left(), Duration::ZERO, "quiet since");
    }

    /// An allocation that never ends (or allocations that never pause) hold a step back for
    /// [`MOST_ASIDE`] alone.
    #[test]
    fn a_step_goes_on_after_the_longest_wait() {
        let activity = Activity::new();
        assert!(activity.begin());
        let waited = activity.step_aside(|| false).expect("it waited");
        assert!(waited >= MOST_ASIDE, "waited {waited:?}");
        assert!(waited < MOST_ASIDE * 3, "waited {waited:?}");
        activity.end();
    }

    #[test]
    fn a_step_goes_on_at_once_where_it_is_told_to() {
        let activity = Activity::new();
        assert!(activity.begin());
        assert_eq!(activity.step_aside(|| true), None);
        activity.end();
        assert_eq!(activity.step_aside(|| true), None, "told to go on");
    }

    /// The test's own thread is marked, for the rest of its life: each test has a thread of its
    /// own, and threads start only through the exit guard (`tests::exit_guard`).
    #[test]
    fn a_check_s_own_allocations_count_for_nothing() {
        let activity = Activity::new();
        mark_a_check_s_thread(Counted::Apart);
        assert!(!activity.begin());
        assert_eq!(activity.left(), Duration::ZERO);
    }
}
