//! The waits of command buffers behind another command queue's work on their shared `VkQueue`
//! (`WorkCounts::command_buffers_behind_other_queues` and kin): counted always, at a cost of O(1)
//! per batch and per completion the host sees.
//!
//! - **Why.** A `VkQueue` runs its submissions in order, so where command queues share one (one
//!   graphics queue on RADV, ANV and lavapipe, `Device::new_command_queue_unshared`'s included), a
//!   command buffer waits there for every other queue's work submitted before it, however long.
//! - **What counts.** A command buffer's first batch, as the scheduler gives it its value of its
//!   `VkQueue`'s timeline (submitted, or held in a group for one submission with later batches:
//!   `group.rs`), where the latest batch of another command queue's before it on that `VkQueue`
//!   is not yet seen complete. That batch took the lower value, so it went to the `VkQueue`, or
//!   into the group, before this command buffer's first batch did, and the `VkQueue` runs it
//!   first. A command buffer whose first batch the scheduler held back (for an event's value, a
//!   binary semaphore, a check) can be passed there by another queue's command buffer committed
//!   after it; such a wait counts too. A command buffer's later batches are not counted again.
//! - **How long.** From its first batch's submission (the host time the scheduler called
//!   `vkQueueSubmit`, or held the batch in a group) to the host's observation of that other batch's
//!   completion (`QueueSlot::completed`): an upper bound of the wait on the GPU, since the
//!   observation comes after the GPU's end (a completion waiter's wake, a busy host). It holds the
//!   other batch's whole run where that one had not started yet.
//! - **How.** Each `VkQueue` keeps its last batch's value and command queue, and the value of its
//!   latest batch of another command queue than that one: the latest batch of a queue other than
//!   a new batch's own is the last where the last is another queue's, else that kept one. Where a
//!   batch follows its own queue's last, that value is the one the batch before it found, and where
//!   it follows another queue's, it is that last one, above every earlier value: the values waited
//!   for never fall, so the waits being timed are kept in that order, and a completion takes those
//!   whose value it reached from the front. No command queue of the device is looked at. Only
//!   command buffers' batches are seen: the runtime's own submissions (staging, checks, a step's
//!   run again) neither count nor are waited behind.
//! - **What it waited behind** (unshared queues' waits over 16 ms alone,
//!   `WorkCounts::unshared_waits_over_16ms_batches_ahead` and kin): the batches in flight ahead of
//!   it as its first batch took its value (the values past the one the host saw complete), and the
//!   longest run among those it waited for: a run is the host's view of how long the oldest batch
//!   in flight was so, from the completion seen before it (or its submission, where the host saw
//!   every earlier batch complete) to the completion that ended it, so one batch's GPU work and
//!   the host's lateness to see it end, or several batches seen complete at once. A wait one run
//!   alone holds over 16 ms waits behind one long batch; one of many short runs, behind many
//!   batches. The longest run of each wait being timed is kept as a sliding maximum over the runs
//!   seen since the oldest one began: O(1) a completion, amortized, and no more runs kept than
//!   the batches that wait was behind.

use crate::device::lock;
use crate::work_counts::Counters;
use std::collections::VecDeque;
use std::sync::Mutex;

/// One `VkQueue`'s part (`QueueSlot::waits`): a leaf lock, taken under the scheduler's at a
/// batch's value and on a completion turn as the host sees its timeline move.
#[derive(Default)]
pub(crate) struct QueueWaits(Mutex<Ahead>);

/// What [`QueueWaits`] keeps.
#[derive(Default)]
struct Ahead {
    /// Its last batch's value and command queue (`QueueGen::id`); none before the first.
    last: Option<(u64, u64)>,
    /// The value of its latest batch of a command queue other than the last's; 0 for none.
    other: u64,
    /// The waits being timed, by the value each waits for, which rises.
    timed: VecDeque<Timed>,
    /// The host time its oldest batch in flight became so: the last completion it saw, or the
    /// submission of a batch taken while the host saw every earlier one complete.
    head_from: u64,
    /// The runs seen while a wait is timed, numbered from 1, their nanoseconds falling from the
    /// front: each the longest of those from it on (the module docs).
    runs: VecDeque<(u64, u64)>,
    /// The number of the last run kept.
    run: u64,
    /// Its batches in flight, where an unshared queue holds it: what other queues' batches the
    /// scheduler holds back are held against (`unshared_hold.rs`).
    flight: crate::unshared_hold::Flight,
}

/// A command buffer's wait behind another command queue's batch, being timed.
#[derive(Clone, Copy)]
struct Timed {
    /// The value of the batch it waits for.
    through: u64,
    /// The host time its first batch was submitted (`gpu_time::host_nanos`).
    since: u64,
    /// Its command queue was made with `Device::new_command_queue_unshared`.
    unshared: bool,
    /// The batches in flight ahead of it as it took its value.
    ahead: u64,
    /// The number of the last run kept before it began: the runs it waited for come after.
    after_run: u64,
}

/// A wait behind another command queue's batch that ended (`Counters::waited_behind_other_queue`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Waited {
    /// From its first batch's submission to the host's observation of the batch it waited for.
    pub(crate) nanos: u64,
    /// The batches in flight ahead of it as it took its value.
    pub(crate) batches: u64,
    /// The longest run among those it waited for (the module docs).
    pub(crate) longest: u64,
}

/// A command buffer's first batch, as [`QueueWaits::taken`] times it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct First {
    /// The host time it was submitted, or held in a group (`gpu_time::host_nanos`).
    pub(crate) since: u64,
    /// Its command queue was made with `Device::new_command_queue_unshared`.
    pub(crate) unshared: bool,
}

impl QueueWaits {
    /// A batch of command queue `queue` took `value` of the `VkQueue`'s timeline, under the
    /// scheduler's lock, so in the order of the values. Where it is its command buffer's `first`
    /// and the latest batch of another queue's before it is not seen complete (`completed`, the
    /// value the host saw the timeline reach, read under this lock: a completion that published a
    /// higher value before this lock was taken is seen here, one after takes the wait), the
    /// command buffer waits behind it: counted, and timed to that batch's completion.
    /// It enters the `VkQueue`'s flight as `flies` says (`unshared_hold.rs`).
    pub(crate) fn taken(
        &self,
        (value, queue, flies): (u64, u64, crate::unshared_hold::Flies<'_>),
        first: Option<First>,
        completed: impl FnOnce() -> u64,
        counts: &Counters,
    ) {
        let mut ahead = lock(&self.0);
        ahead.flight.taken(value, flies);
        let through = match ahead.last {
            Some((last, by)) if by != queue => {
                ahead.other = last;
                last
            }
            _ => ahead.other,
        };
        let before = ahead.last.map_or(0, |(last, _)| last);
        ahead.last = Some((value, queue));
        let completed = completed();
        // Taken while the host saw every earlier batch complete: its run begins here.
        if completed >= before {
            ahead.head_from = first.map_or_else(crate::gpu_time::host_nanos, |first| first.since);
        }
        let Some(First { since, unshared }) = first else {
            return;
        };
        if through > completed {
            counts.behind_other_queue(unshared);
            let after_run = ahead.run;
            ahead.timed.push_back(Timed {
                through,
                since,
                unshared,
                ahead: value.saturating_sub(completed + 1),
                after_run,
            });
        }
    }

    /// The host saw the timeline reach `value`, published before this call: each wait for a batch
    /// at or below it ends now, and each batch at or below it leaves the flight, its run seen
    /// (`unshared_hold.rs`).
    pub(crate) fn completed(&self, value: u64, counts: &Counters) {
        let mut ahead = lock(&self.0);
        let now = crate::gpu_time::host_nanos();
        let length = now.saturating_sub(ahead.head_from);
        ahead.head_from = now;
        ahead.flight.completed(value, length);
        if !ahead.timed.is_empty() {
            ahead.run += 1;
            let run = ahead.run;
            while ahead.runs.back().is_some_and(|&(_, nanos)| nanos <= length) {
                ahead.runs.pop_back();
            }
            ahead.runs.push_back((run, length));
        }
        while let Some(timed) = ahead.timed.front() {
            if timed.through > value {
                break;
            }
            let Timed {
                since,
                unshared,
                ahead: batches,
                after_run,
                ..
            } = *timed;
            while ahead.runs.front().is_some_and(|&(run, _)| run <= after_run) {
                ahead.runs.pop_front();
            }
            let longest = ahead.runs.front().map_or(0, |&(_, nanos)| nanos);
            let waited = Waited {
                nanos: now.saturating_sub(since),
                batches,
                longest,
            };
            counts.waited_behind_other_queue(waited, unshared);
            ahead.timed.pop_front();
        }
        if ahead.timed.is_empty() {
            ahead.runs.clear();
        }
    }

    /// Whether another command queue's batch may go to this `VkQueue` now, held against its
    /// flight (`unshared_hold.rs`).
    pub(crate) fn hold_ahead(&self) -> crate::unshared_hold::Room {
        let ahead = lock(&self.0);
        let ran = crate::gpu_time::host_nanos().saturating_sub(ahead.head_from);
        ahead.flight.ahead(ran)
    }
}
