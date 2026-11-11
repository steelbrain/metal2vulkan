//! Other command queues' work held back from a `VkQueue` an unshared queue shares with them
//! (`UnsharedIsolationPolicy::Bounded`, the default): only about [`BUDGET`] of their expected GPU
//! work is submitted ahead of the unshared queue's next command buffer.
//!
//! - **Why.** A `VkQueue` runs its submissions in order. On a device whose graphics family has
//!   one queue (RADV, ANV, lavapipe), every command queue submits to it,
//!   `Device::new_command_queue_unshared`'s (a compositor's) included, so a compositor's light
//!   command buffer waits for every batch submitted before it there. On a Radeon 780M (RADV,
//!   Mesa 26.2.6), a compositor committing one draw every 16 ms beside an application keeping
//!   three 30 ms command buffers in flight waited 76 ms at the median and 112 ms at worst, every
//!   one of its frames over 16 ms, with 7 batches in flight ahead of it on average. The kernel
//!   does not reorder within a `VkQueue`, and another `VkDevice`'s work (its own kernel context on
//!   i915) interleaves only between submissions, never inside one, whatever its priority.
//! - **Where.** A batch of a command queue not made unshared, on a `VkQueue` that an unshared
//!   queue holds (`SlotShare::unshared`), on a device whose policy is `Bounded`. Where an unshared
//!   queue has a `VkQueue` of its own, no other command queue's batch goes there, and nothing is
//!   held.
//! - **The bound.** Such a batch, otherwise ready, is submitted only while less than [`BUDGET`] of
//!   expected GPU work of other queues' batches is left in flight on its `VkQueue` ([`Flight`]):
//!   the expected runs of the batches submitted there and not seen complete, less what the oldest
//!   one has run since it became the oldest, up to its own expected run. So an unshared queue's
//!   command buffer goes in behind about one batch of theirs, and the next batch goes in before
//!   the one ahead of it ends, so the GPU is not left idle between them. A held batch keeps its
//!   place in the scheduler's order (`scheduler/order.rs`): a later commit of another queue whose
//!   footprint runs in commit order with it waits behind it, as behind any held batch.
//! - **Expected runs** ([`Expected`]). Each command queue's: the host's view of how long its
//!   recent batches ran as their `VkQueue`'s oldest in flight (`queue_wait.rs`'s runs), raised at
//!   once by a longer one and halved toward a shorter one, so a queue whose batches got shorter
//!   takes its new length within a few batches, while one long batch counts at once. A queue with
//!   none seen yet is expected to run [`BUDGET`]. A run several batches complete in shares among
//!   them.
//! - **When it goes.** A pass that holds a batch so leaves the scheduler's thread waiting for its
//!   `VkQueue`'s next completion (a gate, `scheduler/mod.rs`'s `Awaits`), and, where the oldest
//!   batch's expected run still covers it, for the time the work left ahead falls under the
//!   budget. A committing or waiting thread's own pass leaves such a batch to the scheduler's
//!   thread, which it wakes.
//! - **What it costs.** A batch longer than its queue's recent ones still holds the unshared
//!   queue's command buffer for its whole length: one command buffer's batch is one submission,
//!   which the GPU runs to its end. Every batch on such a `VkQueue` costs a place in its
//!   [`Flight`] and a lock of its own at the submission and the completion. Counted:
//!   `WorkCounts::heads_held_for_unshared_isolation` (each look that held a head so).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The expected GPU work of other queues' batches, in nanoseconds, below which another of their
/// batches is submitted to a `VkQueue` an unshared queue holds: 2 ms, the length past which a
/// command buffer marks its queue long where queues share `VkQueue`s (`queue_share::LONG_HEAD`).
pub(crate) const BUDGET: u64 = 2_000_000;

/// A command queue's expected run of one batch (the module docs), in nanoseconds; 0 before the
/// first is seen.
#[derive(Debug, Default)]
pub(crate) struct Expected(AtomicU64);

impl Expected {
    /// The run its next batch is expected to take: [`BUDGET`] where none was seen.
    pub(crate) fn nanos(&self) -> u64 {
        match self.0.load(Ordering::Relaxed) {
            0 => BUDGET,
            nanos => nanos,
        }
    }

    /// One of its batches ran `run` nanoseconds: taken at once where longer, halved toward it
    /// where shorter.
    pub(crate) fn observed(&self, run: u64) {
        let run = run.max(1);
        let _ = (self.0).try_update(Ordering::Relaxed, Ordering::Relaxed, |was| {
            Some(match was {
                0 => run,
                was if run >= was => run,
                was => was - (was - run) / 2,
            })
        });
    }
}

/// How a batch taken on a `VkQueue` enters its [`Flight`].
pub(crate) enum Flies<'a> {
    /// Not at all: no unshared queue holds the `VkQueue`, or the device's policy is `Off`.
    Not,
    /// As an unshared queue's: in flight, with no expected run of other queues' work.
    Unshared,
    /// As another queue's, with its queue's expected run.
    Other(&'a Arc<Expected>),
}

/// One batch in a [`Flight`].
#[derive(Debug)]
struct InFlight {
    /// Its value of its `VkQueue`'s timeline.
    value: u64,
    /// Its expected run, counted in [`Flight::expected`]; 0 for an unshared queue's.
    expected: u64,
    /// Its queue's expected run, which its run updates; none for an unshared queue's.
    of: Option<Arc<Expected>>,
}

/// The batches taken on one `VkQueue` an unshared queue holds and not seen complete
/// (`queue_wait::Ahead::flight`), under that `VkQueue`'s waits' lock.
#[derive(Debug, Default)]
pub(crate) struct Flight {
    /// In the order of their values, which rises.
    batches: VecDeque<InFlight>,
    /// The expected runs of the other queues' among them, summed.
    expected: u64,
}

/// Whether a batch may go to a `VkQueue` now ([`Flight::ahead`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Room {
    /// Less than [`BUDGET`] of other queues' expected work is left in flight.
    Go,
    /// [`BUDGET`] or more: held until the `VkQueue`'s next completion, or until `after`
    /// nanoseconds from now, where the oldest batch's expected run brings the work left under the
    /// budget then.
    Hold { after: Option<u64> },
}

impl Flight {
    /// A batch took `value`, entering as `flies` says.
    pub(crate) fn taken(&mut self, value: u64, flies: Flies<'_>) {
        let (expected, of) = match flies {
            Flies::Not => return,
            Flies::Unshared => (0, None),
            Flies::Other(of) => (of.nanos(), Some(Arc::clone(of))),
        };
        self.expected += expected;
        self.batches.push_back(InFlight {
            value,
            expected,
            of,
        });
    }

    /// The host saw the timeline reach `value`, `run` nanoseconds after the oldest batch in flight
    /// became so: each batch at or below `value` leaves, its queue's expected run updated with its
    /// share of `run`.
    pub(crate) fn completed(&mut self, value: u64, run: u64) {
        let done = self
            .batches
            .iter()
            .take_while(|batch| batch.value <= value)
            .count();
        if done == 0 {
            return;
        }
        let share = run / done as u64;
        for batch in self.batches.drain(..done) {
            self.expected -= batch.expected;
            if let Some(of) = batch.of {
                of.observed(share);
            }
        }
    }

    /// Whether another queue's batch may go now, the oldest batch in flight having run `ran`
    /// nanoseconds since it became so (the module docs).
    pub(crate) fn ahead(&self, ran: u64) -> Room {
        let front = self.batches.front().map_or(0, |batch| batch.expected);
        let credit = ran.min(front);
        let left = self.expected - credit;
        if left < BUDGET {
            return Room::Go;
        }
        let over = left - BUDGET + 1;
        let after = (front - credit >= over).then_some(over);
        Room::Hold { after }
    }

    /// The batches in flight, of every queue.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.batches.len()
    }
}

impl crate::Device {
    /// Whether other queues' batches are held back from `VkQueue` `index` (the module docs): an
    /// unshared queue holds it, and the device's policy is `Bounded`.
    pub(crate) fn holds_back_for_unshared(&self, index: usize) -> bool {
        self.holds_unshared(index)
            && self.compile_device().policy().unshared_isolation
                == metal2vulkan::device::UnsharedIsolationPolicy::Bounded
    }

    /// How a batch of a command queue (`unshared` where made so) taken on `VkQueue` `index` enters
    /// its flight, `expected` its queue's expected run.
    pub(crate) fn flies<'a>(
        &self,
        index: usize,
        unshared: bool,
        expected: &'a Arc<Expected>,
    ) -> Flies<'a> {
        match (self.holds_back_for_unshared(index), unshared) {
            (false, _) => Flies::Not,
            (true, true) => Flies::Unshared,
            (true, false) => Flies::Other(expected),
        }
    }
}
