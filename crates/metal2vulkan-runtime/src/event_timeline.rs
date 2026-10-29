//! What one event's timeline semaphore means in Metal's values, as a pure function of the signals
//! the scheduler submitted, the host's sets, and what the semaphore reached.
//!
//! **Metal's values.** An event's value only rises: a signal, from the GPU or the host, of a value
//! not above the current one changes nothing, and two signals land as their maximum whatever order
//! they execute in (measured on the M3 and the M4). A Vulkan timeline semaphore needs every signal
//! above the value before it, so the semaphore does not carry Metal's values:
//! - **Signal indices.** Each GPU signal the scheduler keeps raises the semaphore by one, and the
//!   timeline maps that index to its Metal value. A signal is kept only when its value is above
//!   every value submitted or set before it, in the scheduler's submission order, so kept values
//!   rise with their indices; a signal not above them is dropped, as Metal's maximum would make it
//!   a no-op. A kept signal that follows one still pending on another `VkQueue` waits for that one
//!   ([`Signal::after`]), so the semaphore's signals execute in index order; on one `VkQueue`,
//!   Vulkan's signal operation order already does.
//! - **The host's floor.** A host set never touches the semaphore: it takes effect at once as a
//!   floor the value never drops below. A GPU wait for a value the floor reached needs no Vulkan
//!   wait at all, so no batch ever waits for a host signal on a `VkQueue`.
//! - **What the scheduler knows.** The value every submitted signal and every host set reaches
//!   ([`EventTimeline::known`]). A wait for a known value is satisfied by work already submitted,
//!   or by the host: the scheduler submits only such waits.
//!
//! **A wait the host may satisfy is never handed to Vulkan** ([`EventTimeline::settable`]). An
//! `MTLSharedEvent`'s value is one thing the host and the device both see, so `setSignaledValue:`
//! releases an `encodeWaitForEvent:` as soon as the value is reached, whatever the device is
//! running. A Vulkan wait a batch was submitted with cannot be released by the host: the semaphore
//! value it waits for is the signal index of a GPU signal submitted before it, and signalling that
//! index from the host would make that signal's own `vkQueueSubmit` illegal, since a timeline
//! semaphore signal operation's "value must have a value greater than the current value of the
//! semaphore when the semaphore signal operation is executed"
//! (`VUID-vkQueueSubmit-pSignalSemaphores-03242`). So a wait for a value the host may set is held
//! until the value is reached, by the GPU signal or by the set, and goes with no Vulkan wait at
//! all: [`EventTimeline::wait`] answers `None` while it is not reached. Only an `MTLEvent`, whose
//! value no host call sets, chains on the `VkQueue`.
//!
//! The held batch goes once the value is reached: a host set wakes the scheduler, and so does a
//! completion turn that observes the GPU signal (`event.rs`).
//!
//! **A signal behind a merge is seen at its completion turn alone** ([`EventTimeline::keep`]).
//! A batch submitted while the device keeps a copied host buffer may write the copy, whose bytes
//! reach the caller's pages only at the batch's completion turn (`copied.rs`); Metal's writes
//! before a signal are in the pages once anyone sees its value. So no read of the counter observes
//! such a signal, nor any after it (a read of the counter is a raw read of the GPU's progress),
//! and no wait is handed its index, the scheduler's own host wait included: the completion turn,
//! which merges first, observes it, and wakes every host wait and held batch for it. A plain
//! event's wait for such a signal is held as a shared event's is. The signal itself stays the
//! batch's GPU signal: no submission is added.

use crate::command_buffer::completion::seen;
use std::collections::VecDeque;

/// One kept GPU signal the semaphore has not been seen to reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending<Q> {
    index: u64,
    value: u64,
    /// The `VkQueue` of the batch that signals it.
    queue: Q,
    /// Seen at its batch's completion turn alone, after the copied host buffers merged (the
    /// module docs).
    behind_merge: bool,
}

/// How a batch waits for a Metal value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WaitFor {
    /// The value was reached (on the semaphore, or by the host's floor): no Vulkan wait.
    Reached,
    /// A wait for the semaphore to reach this index, a submitted GPU signal's: a batch's on its
    /// `VkQueue` ([`EventTimeline::wait`]), or the host's ([`EventTimeline::landing`]).
    Index(u64),
}

/// What a GPU signal of a value becomes when its batch is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Signal {
    /// Not above what is known: Metal's maximum makes it a no-op.
    Dropped,
    /// Kept: the batch signals the semaphore to `index`, after it reached `after` when that is a
    /// signal pending on another `VkQueue`.
    Kept { index: u64, after: Option<u64> },
}

/// One event's map from its semaphore's signal indices to Metal's values.
#[derive(Debug)]
pub(super) struct EventTimeline<Q> {
    /// The highest index the semaphore was seen to reach.
    reached: u64,
    /// The Metal value of that index (0 before any).
    reached_value: u64,
    /// The kept GPU signals above `reached`, in index order (and so in value order).
    pending: VecDeque<Pending<Q>>,
    /// The highest value the host set.
    floor: u64,
    /// Whether the host may set the value: a shared event's (`MTLSharedEvent`), whose set must
    /// release a wait at once, which no Vulkan wait can do (the module docs).
    settable: bool,
}

impl<Q> EventTimeline<Q> {
    /// An event's timeline; `settable` for a shared event (`MTLSharedEvent`), whose value the host
    /// may set, and `false` for a plain one (`MTLEvent`).
    pub(super) fn new(settable: bool) -> EventTimeline<Q> {
        EventTimeline {
            reached: 0,
            reached_value: 0,
            pending: VecDeque::new(),
            floor: 0,
            settable,
        }
    }

    /// Whether the host may set the value.
    #[cfg(test)]
    pub(super) fn settable(&self) -> bool {
        self.settable
    }
}

impl<Q: Copy + Eq> EventTimeline<Q> {
    /// The event's value now: the semaphore's last seen index's value, or the host's floor.
    pub(super) fn value(&self) -> u64 {
        self.reached_value.max(self.floor)
    }

    /// The value every submitted signal and host set reaches: what a wait may be submitted for.
    pub(super) fn known(&self) -> u64 {
        self.pending
            .back()
            .map_or(self.value(), |last| last.value.max(self.value()))
    }

    /// The semaphore reached `index`. Whether the value rose.
    pub(super) fn observe(&mut self, index: u64) -> bool {
        let before = self.value();
        while let Some(&front) = self.pending.front() {
            if front.index > index {
                break;
            }
            self.pending.pop_front();
            self.reached = front.index;
            self.reached_value = front.value;
        }
        self.value() > before
    }

    /// The semaphore's counter read `read`: observed only up to the index a host wait returned
    /// for (`reaches`, the answer of `command_buffer/completion.rs`'s `seen`), as a read alone
    /// orders no host operation after the signal; a read past the index last seen is waited for
    /// first. Whether the value rose.
    pub(super) fn observe_read<E>(
        &mut self,
        read: u64,
        reaches: impl FnOnce(u64) -> Result<bool, E>,
    ) -> Result<bool, E> {
        // Never past a signal behind a merge, which its completion turn observes (the module
        // docs).
        let read = match self.pending.iter().find(|pending| pending.behind_merge) {
            Some(pending) => read.min(pending.index - 1),
            None => read,
        };
        let index = seen(self.reached, None, read, reaches)?;
        Ok(self.observe(index))
    }

    /// The host set `value`. Whether the value rose.
    pub(super) fn set(&mut self, value: u64) -> bool {
        let before = self.value();
        self.floor = self.floor.max(value);
        self.value() > before
    }

    /// How a batch submitted now waits for `value`; `None` while the batch must be held: no
    /// submitted signal and no host set reaches it yet, or the host may set it and it is not
    /// reached, which no Vulkan wait would answer to (the module docs).
    pub(super) fn wait(&self, value: u64) -> Option<WaitFor> {
        match self.landing(value) {
            Some(WaitFor::Index(_)) if self.settable => None,
            landing => landing,
        }
    }

    /// How the host waits for `value` to land: reached, or the index of the submitted GPU signal
    /// that reaches it, whether or not the host may set it, so a wait [`EventTimeline::wait`]
    /// holds on the host is one too; `None` while no submitted signal and no host set reaches it.
    /// The host waits for the semaphore's index (the scheduler's thread, `scheduler/mod.rs`), and
    /// a set meanwhile wakes it as any host signal does: no batch is handed that index.
    /// A value a signal behind a merge reaches, or one after it, is `None` too: only its
    /// completion turn observes it (the module docs).
    pub(super) fn landing(&self, value: u64) -> Option<WaitFor> {
        if value <= self.value() {
            return Some(WaitFor::Reached);
        }
        let mut behind_merge = false;
        for pending in &self.pending {
            behind_merge |= pending.behind_merge;
            if pending.value >= value {
                return (!behind_merge).then_some(WaitFor::Index(pending.index));
            }
        }
        None
    }

    /// Whether a wait for `value` is held on the host where a Vulkan wait for a submitted GPU
    /// signal's index would have served: the host may set the value, it is not reached, and a
    /// signal submitted before the waiting batch reaches it. Exactly the waits that chained on
    /// the `VkQueue` before, which cost a hold and a `vkQueueSubmit` now
    /// (`WorkCounts::shared_event_waits_held`).
    pub(super) fn held_for_the_host(&self, value: u64) -> bool {
        self.settable
            && value > self.value()
            && (self.pending.iter()).any(|pending| pending.value >= value)
    }

    /// What a GPU signal of `value` by a batch submitted now on `queue` becomes. Nothing changes
    /// until [`EventTimeline::keep`].
    pub(super) fn signal(&self, value: u64, queue: Q) -> Signal {
        if value <= self.known() {
            return Signal::Dropped;
        }
        match self.pending.back() {
            Some(last) => Signal::Kept {
                index: last.index + 1,
                after: (last.queue != queue).then_some(last.index),
            },
            None => Signal::Kept {
                index: self.reached + 1,
                after: None,
            },
        }
    }

    /// The batch of a kept signal (`index` from [`EventTimeline::signal`]) of `value` on `queue`
    /// was submitted; `behind_merge` where the device kept a copied host buffer then (the module
    /// docs).
    pub(super) fn keep(&mut self, index: u64, value: u64, queue: Q, behind_merge: bool) {
        self.pending.push_back(Pending {
            index,
            value,
            queue,
            behind_merge,
        });
    }

    /// The kept signals the semaphore has not been seen to reach.
    #[cfg(test)]
    pub(super) fn pending(&self) -> usize {
        self.pending.len()
    }
}
