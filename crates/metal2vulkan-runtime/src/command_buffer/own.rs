//! Waits from inside the runtime's handler threads: what a command buffer wait answers when it is
//! called from a scheduled handler, a completed handler, or a deallocator that runs as a command
//! buffer releases what it retained.
//!
//! Those run in their queue's serial context ([`contexts`](crate::contexts)): a queue's
//! transitions to `Scheduled` and to `Completed`, with their handlers, run one at a time in the
//! queue's order. A wait there for a transition still to come in the same queue would wait for
//! itself.
//!
//! Metal's answers (measured on an M4 with macOS 27.0.1, with and without API validation):
//! - a deallocator run as Metal releases a buffer returns from its wait for the completion of the
//!   command buffer that held it; the runtime returns too;
//! - a completed handler returns from its wait for its command buffer's scheduling; so does the
//!   runtime (that transition ran before);
//! - a handler's wait for its own command buffer's transition in progress, and for any later
//!   transition of its queue (a command buffer committed after its own, before or inside the
//!   handler), never returns, even when that command buffer's work already ran. The runtime does
//!   not copy the hang: it refuses those waits at once (`Error::WaitWouldDeadlock` of
//!   `DeadlockCause::OwnQueue`, or `TimedOut` for the bounded wait);
//! - a handler's wait for a command buffer of another queue returns, whether that command buffer
//!   was committed before the handler or inside it; so does the runtime's. It blocks the handler's
//!   thread, counted ([`Waits`]), and is refused only where it would deadlock: when it would
//!   close a cycle of such waits, or leave its device's pool with no thread that is not blocked
//!   so (`contexts::block`).
//!
//! A submission handler (a runtime extension, Metal has none) runs on the thread that took its
//! command buffer's last batch for its queue, and that command buffer's transition to `Scheduled`,
//! with every later transition of its queue, waits for it. On the scheduler thread, the device's
//! later submissions wait for it too. So any wait there for what has not happened (a command
//! buffer's scheduling or completion, a slot of a full queue no uncommitted command buffer holds)
//! is refused at once (`DeadlockCause::SubmissionHandler`); one for what happened returns.
//!
//! A completion-seen handler (a runtime extension, Metal has none) runs as its command buffer's
//! transition to `Completed` runs: on the thread that saw the command buffer complete (a queue's
//! waiter, the scheduler's thread, or the committing thread for one that needs no GPU work) where
//! that transition runs there, else on its queue's handler thread. That thread's turn finishes
//! every queue's command buffers, so a wait there for what has not happened would wait for itself,
//! and its waits are refused as a submission handler's are, wherever it runs
//! (`DeadlockCause::CompletionSeenHandler`); one for what happened returns.

use crate::error::DeadlockCause;
use std::cell::Cell;

/// A step of a command buffer's transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Its scheduled handlers.
    ScheduledHandlers,
    /// Its completed handlers.
    CompletedHandlers,
    /// The release of what it retained (a no-copy buffer's deallocator may run here).
    Release,
    /// Its submission handlers, on the thread that took its last batch (the scheduler's,
    /// or a committing one, which may be a handler thread): its transition to `Scheduled`, and
    /// so every later transition of its queue, waits for them.
    SubmissionHandlers,
    /// Its completion-seen handlers, as its transition to `Completed` or `Error` runs: on the
    /// thread that saw it complete, or its queue's handler thread.
    CompletionSeenHandlers,
}

impl Step {
    /// The cause a wait for what has not happened is refused with from this step: a submission
    /// handler's or a completion-seen handler's, which the device's other work waits for. `None`
    /// for the steps of a handler thread, whose waits answer by their queues.
    fn refuses(self) -> Option<DeadlockCause> {
        match self {
            Step::SubmissionHandlers => Some(DeadlockCause::SubmissionHandler),
            Step::CompletionSeenHandlers => Some(DeadlockCause::CompletionSeenHandler),
            Step::ScheduledHandlers | Step::CompletedHandlers | Step::Release => None,
        }
    }
}

/// A logical queue's serial context: its device (by the shared state's address) and the queue's
/// identity there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Context {
    pub(crate) device: usize,
    pub(crate) queue: u64,
}

/// What a handler thread runs: a step of a command buffer (by its shared state's address) in its
/// queue's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Marker {
    pub(crate) context: Context,
    pub(crate) buffer: usize,
    pub(crate) step: Step,
}

/// The command buffer a wait is for, in its queue's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub(crate) context: Context,
    pub(crate) buffer: usize,
}

thread_local! {
    static RUNNING: Cell<Option<Marker>> = const { Cell::new(None) };
}

/// This thread runs a marked step until the value is dropped; the marker before it comes back
/// then.
pub(crate) struct Running {
    previous: Option<Marker>,
}

impl Running {
    /// Mark this thread as running `marker`'s step.
    pub(crate) fn enter(marker: Marker) -> Running {
        let previous = RUNNING.get();
        RUNNING.set(Some(marker));
        Running { previous }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        RUNNING.set(self.previous);
    }
}

/// The step this thread runs, if any.
pub(crate) fn running() -> Option<Marker> {
    RUNNING.get()
}

/// What a wait answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Answer {
    /// Wait as usual: this is not a handler thread, or what it waits for happened.
    Wait,
    /// Return at once, as Metal does.
    Return,
    /// Refuse, for its cause: what it waits for comes later in this thread's own context
    /// (`DeadlockCause::OwnQueue`).
    Refuse(DeadlockCause),
    /// Wait for another queue's context, from this one: counted while it blocks, and refused
    /// where it would deadlock.
    Block(Context),
}

/// What `wait_until_completed` on `target` answers on a thread running `marker`; `finished`
/// says whether its waiters may already return.
pub(crate) fn completed_wait(marker: Option<Marker>, target: Target, finished: bool) -> Answer {
    let Some(marker) = marker else {
        return Answer::Wait;
    };
    if finished {
        Answer::Wait
    } else if let Some(cause) = marker.step.refuses() {
        Answer::Refuse(cause)
    } else if marker.buffer == target.buffer && marker.step == Step::Release {
        Answer::Return
    } else if marker.context == target.context {
        Answer::Refuse(DeadlockCause::OwnQueue)
    } else {
        Answer::Block(marker.context)
    }
}

/// What `wait_until_scheduled` on `target` answers on a thread running `marker`; `scheduled`
/// says whether its transition to `Scheduled` ran.
pub(crate) fn scheduled_wait(marker: Option<Marker>, target: Target, scheduled: bool) -> Answer {
    let Some(marker) = marker else {
        return Answer::Wait;
    };
    if marker.buffer == target.buffer && marker.step == Step::ScheduledHandlers {
        // That transition is in progress: Metal's never returns there.
        Answer::Refuse(DeadlockCause::OwnQueue)
    } else if scheduled {
        Answer::Wait
    } else if let Some(cause) = marker.step.refuses() {
        Answer::Refuse(cause)
    } else if marker.context == target.context {
        Answer::Refuse(DeadlockCause::OwnQueue)
    } else {
        Answer::Block(marker.context)
    }
}

/// What a new command buffer's wait for a slot of the full queue `queue` answers on a thread
/// running `marker` (`CommandQueue::command_buffer`); `open` says whether a command buffer not
/// committed holds one of the queue's slots, which the program may give back by dropping it.
///
/// A committed command buffer gives its slot back only in its queue's context, after its
/// completed handlers. In that context, then, no committed command buffer can give one back:
/// Metal's make never returns there (a completed handler's make on its own full
/// queue). The runtime refuses it, unless an uncommitted command buffer may still be dropped.
pub(crate) fn slot_wait(marker: Option<Marker>, queue: Context, open: bool) -> Answer {
    let Some(marker) = marker else {
        return Answer::Wait;
    };
    if let Some(cause) = marker.step.refuses() {
        return if open {
            Answer::Wait
        } else {
            Answer::Refuse(cause)
        };
    }
    if marker.context != queue {
        Answer::Block(marker.context)
    } else if open {
        Answer::Wait
    } else {
        Answer::Refuse(DeadlockCause::OwnQueue)
    }
}

/// The handler threads' waits for other contexts, as edges from the waiting context to the
/// awaited one. A context runs one item at a time, so it has at most one edge out.
#[derive(Debug)]
pub(crate) struct Waits {
    edges: Vec<(Context, Context)>,
}

impl Waits {
    /// No wait.
    pub(crate) const fn new() -> Waits {
        Waits { edges: Vec::new() }
    }

    /// Whether a wait of `from` for `to` would close a cycle: `to` waits, through the contexts
    /// it waits for, for `from`.
    pub(crate) fn closes_cycle(&self, from: Context, to: Context) -> bool {
        let mut at = to;
        // Each step follows a distinct edge, unless the edges hold a cycle already (they never
        // do: no wait that closes one is added).
        for _ in 0..=self.edges.len() {
            if at == from {
                return true;
            }
            match self.edges.iter().find(|(waiting, _)| *waiting == at) {
                Some(&(_, next)) => at = next,
                None => return false,
            }
        }
        true
    }

    /// `from` waits for `to`.
    pub(crate) fn insert(&mut self, from: Context, to: Context) {
        self.edges.push((from, to));
    }

    /// `from` no longer waits for `to`.
    pub(crate) fn remove(&mut self, from: Context, to: Context) {
        if let Some(at) = self.edges.iter().position(|&edge| edge == (from, to)) {
            self.edges.swap_remove(at);
        }
    }
}
