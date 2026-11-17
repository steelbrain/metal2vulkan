//! A command buffer completes after the command buffer of another queue whose GPU signal its work
//! waited for (`completion.rs`, "After what it waited for"): its transition to `Completed` waits,
//! its place in its own queue's context kept, until that command buffer's transition has its
//! status final and ran its completion-seen handlers, which never block, and reaches its completed
//! handlers, which may. Those and its own then run on two handler threads, the signaller's
//! dispatched first, as Metal runs two queues' handlers on two threads. Which starts first is not
//! guaranteed, as on Metal, which on the M3 Max (macOS 27.0.1) started the waiter's first in 0.1-6%
//! of rounds with plain events.
//!
//! - **Who follows whom.** A command buffer whose last batch keeps GPU signals carries a
//!   [`Follow`], which the events hold beside those signals until its transition reaches its
//!   completed handlers (`EventTimeline::signallers`). A batch of another queue's command buffer
//!   that waits for one of those signals takes it as it is submitted, so its command buffer knows
//!   whom it completes after.
//! - **Only behind what blocks nothing.** Metal runs each queue's transitions in their own
//!   context: while one of a queue's handlers blocks, another queue's transitions go on
//!   (`contexts/mod.rs`). So a transition waits for the signaller's only where the signaller's
//!   transition was posted to its context and nothing there before it may block
//!   ([`Follow::leads`]): no item before it, nor what a thread runs there, runs a handler of the
//!   program's but completion-seen handlers, which never block, or waits for one
//!   (`Run::may_block`). An earlier command buffer's transition with no scheduled, submission or
//!   completed handler left ends without the program, however long the host keeps the thread
//!   that runs it off the CPU, so the waiting transition waits behind it: a program whose command
//!   buffer waited on the GPU for another queue's never sees it complete while that one reads
//!   short of `Completed` only because a host thread was slow. Its release of what it retained
//!   may still run a deallocator of the program's (`program_objects.rs`), which the waiting
//!   transition waits behind too. An earlier command buffer of the signaller's queue whose waiters
//!   returned runs nothing there any more: a context is done with an item before the item's hold
//!   of its command buffer goes, which lets those waiters return (`contexts/mod.rs`). The
//!   signaller's transition runs no code of the program's that may block before its completed
//!   handlers, so the wait ends once a thread takes it and runs its completion-seen handlers, and
//!   it holds no thread meanwhile: a gate in the waiting transition's context
//!   (`contexts/lines.rs`). Behind an item that may block, the waiting transition goes on, as
//!   Metal's does. The completion turns take the signaller's batch before the waiting one's
//!   (`completion.rs`), so the signaller's transition is posted by the time the waiting one's is.
//!   Whether it leads is one look at its context's first item and first item that may block
//!   (`Lines::leads`), however many items the context holds.
//! - **Nothing of the signaller is held.** A [`Follow`] names its command buffer's transition by
//!   its place in its context alone: the waiter that completes the follower never holds the
//!   signaller's last reference, so never the device's.

use super::super::contexts::Contexts;
use super::super::device::lock;
use super::super::event::EventShared;
use super::Item;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The command buffer whose last batch's GPU signals another queue's command buffers may wait for,
/// as they complete after it (the module docs).
#[derive(Debug)]
pub(crate) struct Follow {
    /// Its queue's identity: its context's.
    queue: u64,
    /// The gates of the transitions posted behind its own, each a context and a gate's token,
    /// until its own reaches its completed handlers; `None` from then on.
    gates: Mutex<Option<Vec<(u64, u64)>>>,
    /// The place its transition to `Completed` or `Error` took in its context as it was posted
    /// ([`Follow::posted`], `Contexts::post`); 0 before then, and where it ran without a post.
    at: AtomicU64,
}

/// What a command buffer's transition to `Completed` or `Error` lets go as it reaches its completed
/// handlers:
/// its [`Follow`], and its last batch's kept GPU signals, by event and index, from which no
/// command buffer that waits for one completes after it any more (`EventShared::released`).
pub(super) struct Release {
    pub(super) follow: Arc<Follow>,
    pub(super) signals: Vec<(Arc<EventShared>, u64)>,
}

impl Follow {
    /// The follow of a command buffer of the queue `queue`.
    pub(super) fn new(queue: u64) -> Follow {
        Follow {
            queue,
            gates: Mutex::new(Some(Vec::new())),
            at: AtomicU64::new(0),
        }
    }

    /// Its queue's identity.
    pub(super) fn queue(&self) -> u64 {
        self.queue
    }

    /// Its transition to `Completed` or `Error` was posted to its context, at the place `at`.
    pub(super) fn posted(&self, at: u64) {
        self.at.store(at, Ordering::Release);
    }

    /// Whether its transition to `Completed` or `Error` has not reached its completed handlers and
    /// leads its context: it was posted there, and nothing there before it may block.
    pub(super) fn leads(&self, contexts: &Contexts<Item>) -> bool {
        let gates = lock(&self.gates);
        gates.is_some() && self.leads_its_context(contexts)
    }

    /// Whether its transition leads its context, where it has not reached its completed handlers,
    /// so is not done there.
    fn leads_its_context(&self, contexts: &Contexts<Item>) -> bool {
        let at = self.at.load(Ordering::Acquire);
        at > 0 && contexts.leads(self.queue, at)
    }

    /// Close `gate` of a transition of `queue`'s context until its own transition reaches its
    /// completed handlers, where it has not and leads its context: whether it closed it.
    pub(super) fn follow(&self, contexts: &Contexts<Item>, (queue, gate): (u64, u64)) -> bool {
        let mut gates = lock(&self.gates);
        let Some(gates) = gates.as_mut() else {
            return false;
        };
        if !self.leads_its_context(contexts) || !contexts.close(gate) {
            return false;
        }
        gates.push((queue, gate));
        true
    }

    /// Its own transition reached its completed handlers: open the gates closed behind it.
    pub(super) fn release(&self, contexts: &Contexts<Item>) {
        let gates = lock(&self.gates).take();
        for (queue, gate) in gates.into_iter().flatten() {
            contexts.open(queue, gate);
        }
    }
}
