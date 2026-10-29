//! A driver's stall in one of every so many signalling submissions to a `VkQueue`, taken while
//! the device's scheduler is idle.
//!
//! **The driver.** NVIDIA's proprietary driver spends 2 to 4 ms (1.97 to 4.06 ms measured) in one
//! `vkQueueSubmit` of every 1024 that signal a fence or a semaphore on a `VkQueue`, whatever the
//! call submits (615.71.09, `driverVersion` 0x99d1c240, on an RTX 5080 Laptop GPU under Linux;
//! 8192 calls a form, timed one by one): one command buffer and a timeline signal, no command
//! buffer and a timeline signal, a fence alone, `vkQueueSubmit2`, and four signalling submit infos
//! in one call each stall once in 1024 calls, 100 us apart or back to back; a call that signals
//! nothing never stalls. The count is the `VkQueue`'s: two queues of one device, two devices of
//! one process and two processes each stall once in 1024 of their own calls; `vkQueueWaitIdle`
//! counts as one. The other calls take 1.6 us (p50), though a few elsewhere take a millisecond or
//! more (a device's 62nd submission 10 ms, its 162nd 1.2 to 1.4 ms). Vulkan states no such cost.
//! Each batch the runtime submits signals its queue's timeline, so one command buffer in 1024
//! waited there, and every one committed behind it with it: `host_ab present` at 120 Hz (25
//! command buffers a frame) held a submission 2.3 to 5.1 ms every 341.6 ms, in 30 of 1200 frames.
//!
//! **Taken idle** ([`Period`], [`IdleStalls`]), where the driver has the cost fact
//! (`DriverFact::SignallingSubmitsStallEvery1024`, the translator's `device/driver_facts.rs`):
//!
//! - **Learnt per queue.** Each signalling call the runtime makes under a queue's lock and times
//!   (a batch's, a fence's with no work, the device's own submission, an empty one of the plan's)
//!   is counted; one that took [`STALL_NANOS`] or more is taken as the driver's stall, and the
//!   count starts again from it, unless the place is confirmed and the call is not where the
//!   count puts the next stall (within [`MARGIN`] calls before it): there it is another of the
//!   driver's costs. The place is confirmed once a stall came where the count put it, so a long
//!   call elsewhere before then (a device's first submissions) gives way to the next stall, and
//!   one after leaves the place. A use of the queue the runtime cannot count (the program's, a
//!   sparse binding, a wait for its idle) forgets the count until the next stall, as does a due
//!   stall that did not come: a release without the stall submits no empty submission after its
//!   first miss, and none before a stall was seen.
//! - **Taken while idle.** Once the scheduler's thread has had nothing to submit for
//!   [`IDLE_NANOS`] and a queue's next stall is due within its next burst of work (the most
//!   signalling calls it made between two such idles over its last [`BURSTS`], plus [`MARGIN`]),
//!   at most [`MOST_EMPTY`] calls away, the thread submits the calls left as empty submissions,
//!   each signalling a timeline semaphore of the queue's own: the last takes the stall while
//!   nothing waits for the queue. It stops before each one where a commit or a waker came
//!   meanwhile, so work waits behind one stall at most.
//! - **Counted** (`WorkCounts`): the empty submissions (`idle_submits`), the stalls they took
//!   (`submit_stalls_idle`), the stalls that fell on work (`submit_stalls_on_work`), and the work
//!   that came while one taken idle ran (`submits_behind_idle_stalls`, with the longest wait,
//!   `wait_behind_idle_stall_longest_nanos`): a submission that waited for the queue's lock
//!   behind it, its wait, and a commit that reached the scheduler meanwhile, at most the stall's.
//!
//! An empty submission holds the queue's lock alone, which no completion turn's wait and no
//! handler takes.

use super::functions::Functions;
use super::{lock, QueueSlot};
use crate::exit_guard::{self, Call as Guarded};
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The signalling calls to a `VkQueue` from one stall of the driver's to the next.
pub(crate) const PERIOD: u32 = 1024;

/// A signalling call that took this long or longer, in nanoseconds, is the driver's stall where
/// the count puts it, or where the queue's place is not confirmed ([`Period::called`]): the
/// stalls measured took 1.97 ms or more (3 of 15 under 2 ms in the device test on the RTX 5080),
/// and other signalling calls up to 1.4 ms, a device's 62nd 10 ms.
pub(crate) const STALL_NANOS: u64 = 1_000_000;

/// How long the scheduler's thread has had nothing to submit before it takes a stall due, in
/// nanoseconds: it begins a gap between bursts of work (a frame's), not one between two commits
/// of a burst, which come microseconds apart.
pub(crate) const IDLE_NANOS: u64 = 1_000_000;

/// The bursts of a queue's work the plan weighs: the next is taken to be no larger than the
/// largest of them.
pub(crate) const BURSTS: usize = 8;

/// The calls a stall may come early by, beyond the largest burst, and still be taken idle: in
/// `host_ab present` on the RTX 5080 (2400 frames of 25 command buffers, 120 Hz), 5 of 58
/// stalls came 24 calls before the 1024th, at the last command buffer of a frame, where the
/// plan's empty calls would have gone after it.
pub(crate) const MARGIN: u32 = 32;

/// The most empty submissions taken at one idle: a stall further off waits for a later idle, or
/// falls on work.
pub(crate) const MOST_EMPTY: u32 = 128;

/// A wait for the queue's lock, in nanoseconds, behind a stall taken idle that counts as one
/// (`WorkCounts::submits_behind_idle_stalls`): shorter, it waited for no stall.
pub(crate) const BEHIND_NANOS: u64 = 10_000;

/// The signalling calls a use of a queue outside a batch makes (`DeviceShared::use_queue`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Uses {
    /// One `vkQueueSubmit` that signals a fence or a semaphore: counted.
    OneSignalling,
    /// Any calls (the program's, a sparse binding's): the queue's place is forgotten.
    Unknown,
}

/// Who made a signalling call to the queue.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Caller {
    /// A submission of work, or the device's own.
    Work,
    /// An empty submission of the plan's, taken idle.
    Empty,
}

/// A queue's place in its driver's period, as counted, and its bursts of work (the module docs).
/// Kept under the queue's lock (`QueueSlot`).
#[derive(Default)]
pub(crate) struct Period {
    /// The signalling calls since the last stall seen, each counted: `None` before one is seen,
    /// once a use was not counted, or once a stall due did not come.
    since: Option<u32>,
    /// The place `since` counts from is confirmed: that stall came where the count before it put
    /// the stall. A long call elsewhere then is another of the driver's costs, and leaves it.
    confirmed: bool,
    /// The signalling calls of work in the burst under way.
    burst: u32,
    /// The last bursts' calls, a ring.
    bursts: [u32; BURSTS],
    at: usize,
    /// The host time the last call of work ended, in nanoseconds.
    last_end: u64,
    /// The last call was an empty one that stalled: one that took the lock meanwhile waited
    /// behind it.
    idle_stall: bool,
}

impl Period {
    /// A use of the queue the runtime cannot count: no stall is due until one is seen.
    pub(crate) fn forget(&mut self) {
        self.since = None;
        self.confirmed = false;
        self.idle_stall = false;
    }

    /// The call that took the queue's lock now waited `waited` nanoseconds for it: `Some(waited)`
    /// where it waited behind a stall taken idle ([`BEHIND_NANOS`]).
    pub(crate) fn entered(&mut self, waited: u64) -> Option<u64> {
        let behind = (self.idle_stall && waited >= BEHIND_NANOS).then_some(waited);
        self.idle_stall = false;
        behind
    }

    /// A signalling call of `caller`'s that took `nanos` and ended at host time `end`, in
    /// nanoseconds: whether it was the driver's stall, a call of [`STALL_NANOS`] or more where
    /// the count puts the next stall (the [`PERIOD`]th call since the last, or up to [`MARGIN`]
    /// calls before it), which confirms the place, or anywhere while the place is not confirmed.
    /// A call of work [`IDLE_NANOS`] or more after the one before begins a burst.
    pub(crate) fn called(&mut self, caller: Caller, nanos: u64, end: u64) -> bool {
        let expected = (self.since).is_some_and(|since| since + 1 + MARGIN >= PERIOD);
        let stalled = nanos >= STALL_NANOS && (expected || !self.confirmed);
        if stalled {
            self.confirmed = expected;
        }
        if caller == Caller::Work {
            let gap = end.saturating_sub(nanos).saturating_sub(self.last_end);
            if gap >= IDLE_NANOS {
                self.end_burst();
            }
            self.burst = self.burst.saturating_add(1);
            self.last_end = end;
        }
        self.idle_stall = stalled && caller == Caller::Empty;
        self.since = match (stalled, self.since) {
            (true, _) => Some(0),
            // The stall due did not come: the count is wrong, or the driver stalls no more.
            (false, Some(since)) if since + 1 >= PERIOD => {
                self.confirmed = false;
                None
            }
            (false, since) => since.map(|since| since + 1),
        };
        stalled
    }

    fn end_burst(&mut self) {
        if self.burst > 0 {
            self.bursts[self.at] = self.burst;
            self.at = (self.at + 1) % BURSTS;
            self.burst = 0;
        }
    }

    /// Whether a stall may be due at the next idle: a cheap look, which changes nothing.
    pub(crate) fn may_be_due(&self) -> bool {
        self.since.is_some_and(|since| PERIOD - since <= MOST_EMPTY)
    }

    /// The queue is idle: the burst under way ended, and the empty calls that take the next stall
    /// now, the calls left up to and including the stall's, where it is due within the next
    /// burst (the largest of the last [`BURSTS`], plus [`MARGIN`]) and [`MOST_EMPTY`] calls at
    /// most; else 0.
    pub(crate) fn due_idle(&mut self) -> u32 {
        self.end_burst();
        let Some(since) = self.since else {
            return 0;
        };
        let left = PERIOD - since;
        let next = self.bursts.iter().copied().max().unwrap_or(0);
        match left <= next.saturating_add(MARGIN).min(MOST_EMPTY) {
            true => left,
            false => 0,
        }
    }
}

/// `count` timeline semaphores at 0 of `raw`'s, one for each queue's empty submissions; `None`
/// where the device made not all of them (it releases those it made), and then takes no stall
/// idle.
pub(super) fn semaphores(raw: &ash::Device, count: usize) -> Option<Vec<vk::Semaphore>> {
    let mut made = Vec::with_capacity(count);
    for _ in 0..count {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the device is live and enables `timelineSemaphore`.
        match unsafe { raw.create_semaphore(&info, None) } {
            Ok(semaphore) => made.push(semaphore),
            Err(_) => {
                for semaphore in made {
                    // SAFETY: made just now, used by nothing.
                    unsafe { crate::release::object(raw, semaphore) };
                }
                return None;
            }
        }
    }
    Some(made)
}

/// What the scheduler's thread takes the stalls idle with: the device's queues, a timeline
/// semaphore of each queue's own for the empty submissions to signal, and their counts.
pub(crate) struct IdleStalls {
    functions: Functions,
    queues: Arc<[QueueSlot]>,
    /// By queue: the semaphore, and the value it was last signalled to.
    signals: Mutex<Vec<(vk::Semaphore, u64)>>,
    counts: StallCounts,
}

/// The counts of the stalls taken idle (the module docs).
#[derive(Default)]
struct StallCounts {
    empty: AtomicU64,
    idle: AtomicU64,
    behind: AtomicU64,
    behind_longest: AtomicU64,
}

/// What [`IdleStalls::counts`] reads: the empty submissions made, the stalls they took, and the
/// commits that came while one ran, with the longest of those stalls in nanoseconds.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct IdleCounts {
    pub(crate) empty: u64,
    pub(crate) idle: u64,
    pub(crate) behind: u64,
    pub(crate) behind_longest: u64,
}

impl IdleStalls {
    /// The plan of the device of `functions`, whose queues are `queues`, each with its own
    /// semaphore of `semaphores`, timeline semaphores at 0, which the device releases after its
    /// scheduler stopped ([`release`](Self::release)).
    pub(super) fn new(
        functions: Functions,
        queues: Arc<[QueueSlot]>,
        semaphores: Vec<vk::Semaphore>,
    ) -> IdleStalls {
        IdleStalls {
            functions,
            queues,
            signals: Mutex::new(semaphores.into_iter().map(|s| (s, 0)).collect()),
            counts: StallCounts::default(),
        }
    }

    /// Whether a stall may be due on any queue at the next idle ([`Period::may_be_due`]), read
    /// without the queues' locks.
    pub(crate) fn may_be_due(&self) -> bool {
        self.queues.iter().any(QueueSlot::stall_may_be_due)
    }

    /// The scheduler's thread has been idle [`IDLE_NANOS`]: on each queue whose next stall is due
    /// ([`Period::due_idle`]), submit the calls left as empty submissions, one at a time under the
    /// queue's lock, stopping before each one where `arrived` (a commit or a waker came) and after
    /// the one that stalls.
    pub(crate) fn take(&self, arrived: impl Fn() -> bool) {
        for (queue, slot) in self.queues.iter().enumerate() {
            let due = slot.with_period(Period::due_idle);
            for _ in 0..due {
                if arrived() {
                    return;
                }
                let Some(stalled) = self.submit_empty(queue, slot) else {
                    return;
                };
                if let Some(nanos) = stalled {
                    self.counts.idle.fetch_add(1, Ordering::Relaxed);
                    if arrived() {
                        self.counts.behind.fetch_add(1, Ordering::Relaxed);
                        self.counts
                            .behind_longest
                            .fetch_max(nanos, Ordering::Relaxed);
                    }
                    break;
                }
            }
        }
    }

    /// One empty submission to `slot`, the device's queue `queue`, signalling its semaphore's
    /// next value: `Some(Some(nanos))` where it stalled, `Some(None)` where not, `None` where it
    /// failed (a lost device) or the process is exiting.
    fn submit_empty(&self, queue: usize, slot: &QueueSlot) -> Option<Option<u64>> {
        let _busy = exit_guard::enter()?;
        let (semaphore, value) = {
            let mut signals = lock(&self.signals);
            let (semaphore, value) = signals.get_mut(queue)?;
            *value += 1;
            (*semaphore, *value)
        };
        let semaphores = [semaphore];
        let values = [value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::SubmitInfo::default()
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        let raw = self.functions.get();
        let ((submitted, stalled), _) = slot.with_counted(Caller::Empty, |vk_queue, period| {
            let start = crate::gpu_time::host_nanos();
            // SAFETY: a submission of no work signalling the queue's own semaphore of the plan's
            // above every value it was signalled to (they rise under the plan's lock, by one
            // thread); the queue is used under its lock, inside a section.
            let submitted = unsafe { raw.queue_submit(vk_queue, &[info], vk::Fence::null()) };
            let end = crate::gpu_time::host_nanos();
            let nanos = end.saturating_sub(start);
            let stalled = submitted.is_ok() && period.called(Caller::Empty, nanos, end);
            (submitted, stalled.then_some(nanos))
        });
        exit_guard::called(Guarded::QueueSubmit);
        self.counts.empty.fetch_add(1, Ordering::Relaxed);
        submitted.ok().map(|()| stalled)
    }

    /// The counts so far (`WorkCounts::idle_submits` and kin).
    pub(crate) fn counts(&self) -> IdleCounts {
        let counts = &self.counts;
        IdleCounts {
            empty: counts.empty.load(Ordering::Relaxed),
            idle: counts.idle.load(Ordering::Relaxed),
            behind: counts.behind.load(Ordering::Relaxed),
            behind_longest: counts.behind_longest.load(Ordering::Relaxed),
        }
    }

    /// Release the semaphores, the device being torn down: its scheduler stopped and its queues
    /// are idle.
    pub(crate) fn release(&self, raw: &ash::Device) {
        for (semaphore, _) in lock(&self.signals).drain(..) {
            // SAFETY: no submission signalling it is pending (the queues are idle), and nothing
            // submits again (the scheduler stopped).
            unsafe { crate::release::object(raw, semaphore) };
        }
    }
}
