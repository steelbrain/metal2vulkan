//! Trails: a runtime thread's hold of what it held for a command buffer's work, past the moment
//! the work could go on without it.
//!
//! Some of the runtime's threads make what a committed command buffer or a render state's creation
//! waits for, say so, and only then let go of what they held to make it:
//! - a pool piece settles a pipeline's making and then gives back its variant, or a capture's
//!   translation and then its device (`pipeline/dispatch_making.rs`, `pipeline/render.rs`);
//! - the reaper lets go of what a released state's piece gave back (`pipeline/reaper.rs`);
//! - the twin-decode thread says a decode is done and then drops its job's image
//!   (`texture/twin_decode.rs`).
//!
//! What they hold reaches the device. Between the signal and the let-go the command buffer can be
//! submitted, run and complete, and its waiter return; were the program to drop its last handles
//! then, the device would go on that thread rather than on the program's, after the drop returned
//! (`command_buffer/gate.rs`). A loaded host makes that window last: the signal wakes another
//! thread, which may take the signalling thread's CPU.
//!
//! - **Open before the signal, closed after the last let-go** ([`Trails::open`], [`Trail`]). Each
//!   such thread opens a trail of its device's before it signals, and drops the trail after the
//!   last thing it held for the work. A trail handed on with what it held (to the reaper) closes
//!   after that is let go of.
//! - **Numbered** in the order opened. A command buffer takes [`Trails::mark`] as it completes
//!   (`CommandBufferShared::complete`): every trail of the work it waited for is numbered below.
//! - **Waited for by a command buffer's waiters and a render state's creation**
//!   ([`Trails::wait_below`]): `wait_until_completed` and its timed form, once the command
//!   buffer's gate opened, wait until no trail numbered below its mark is open, within the timed
//!   form's timeout; a render state's creation, once the makings it handed the pool settled,
//!   until no trail opened before that is open, within its wait's bound
//!   (`RenderVariant::make_at_creation`). Nothing else waits: the completion, its status, its
//!   handlers, its gate and its queue's later transitions are as they were. An open trail is the time between a signal and a few drops, so a wait meets
//!   one only where that thread lost its CPU in between.
//! - **No deadlock.** A trail holds nothing of the device and takes no lock but this registry's;
//!   the thread that holds it waits for nothing a waiter holds (it settles, gives back, drops), and
//!   a waiter holds no lock of the runtime's. A trail's let-go is never the device's last handle
//!   while a waiter waits: the waiter holds the command buffer, which holds the device.
//! - **Exit.** Once the process is exiting, a wait answers as the gate's does (`exit_guard`): at
//!   once, or within [`exit_guard::WAKE`] where it was blocked.

use crate::device::lock;
use crate::exit_guard;
use std::collections::BTreeSet;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// One device's trails.
#[derive(Default)]
pub(crate) struct Trails {
    open: Mutex<Open>,
    /// Notified as a trail closes.
    closed: Condvar,
}

#[derive(Default)]
struct Open {
    /// The number the next trail takes.
    next: u64,
    /// The trails open, by number.
    numbers: BTreeSet<u64>,
    /// The waits that found a trail open below their mark, counted once each, for the tests'
    /// handshake with a wait that has gone on to block (`Trails::blocked_waits`).
    #[cfg(test)]
    blocked: usize,
}

/// An open trail: closed when dropped. It holds nothing of the device.
pub(crate) struct Trail {
    trails: Arc<Trails>,
    number: u64,
}

/// Why a wait for trails ended before they closed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Ended {
    /// The deadline passed.
    TimedOut,
    /// The process is exiting.
    Exiting,
}

impl Trails {
    /// Open a trail, before the signal it follows.
    pub(crate) fn open(self: &Arc<Self>) -> Trail {
        let mut open = lock(&self.open);
        let number = open.next;
        open.next += 1;
        open.numbers.insert(number);
        Trail {
            trails: Arc::clone(self),
            number,
        }
    }

    /// The number the next trail takes: every trail open now, or opened before, is numbered below.
    pub(crate) fn mark(&self) -> u64 {
        lock(&self.open).next
    }

    /// Wait until no trail numbered below `mark` is open, at most until `deadline` where one is
    /// given; `Err` at the deadline, and once the process is exiting where a trail was open.
    /// `Ok(true)` where it waited at all: a trail below `mark` was open when it looked
    /// (`WorkCounts::trail_waits`).
    pub(crate) fn wait_below(&self, mark: u64, deadline: Option<Instant>) -> Result<bool, Ended> {
        let mut open = lock(&self.open);
        let mut waited = false;
        #[cfg(test)]
        let mut counted = false;
        while open.numbers.first().is_some_and(|&first| first < mark) {
            waited = true;
            #[cfg(test)]
            if !counted {
                counted = true;
                open.blocked += 1;
                #[cfg(any(m2v_gpu, m2v_cpu_icd))]
                blocked::count();
            }
            if exit_guard::closed() {
                return Err(Ended::Exiting);
            }
            let slice = match deadline {
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(Ended::TimedOut);
                    }
                    left.min(exit_guard::WAKE)
                }
                None => exit_guard::WAKE,
            };
            open = self
                .closed
                .wait_timeout(open, slice)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        Ok(waited)
    }

    /// The trails open now.
    #[cfg(test)]
    pub(crate) fn open_now(&self) -> usize {
        lock(&self.open).numbers.len()
    }

    /// How many waits found a trail open below their mark, counted once each as they go on to
    /// block on `closed`. It rises while the wait holds this registry's lock, which it gives up
    /// only to block: a test that sees it rise, and then closes a trail, closes it under a wait
    /// that has already blocked.
    #[cfg(test)]
    pub(crate) fn blocked_waits(&self) -> usize {
        lock(&self.open).blocked
    }
}

impl Drop for Trail {
    fn drop(&mut self) {
        lock(&self.trails.open).numbers.remove(&self.number);
        self.trails.closed.notify_all();
    }
}

/// The runtime's tests' count of waits that found a trail open below their mark (`wait_below`):
/// a device test releases a held trail once a wait met it (`tests/teardown_threads.rs`, compiled
/// under the validation crate's GPU safety cfgs alone).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod blocked {
    use crate::device::lock;
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    static BLOCKED: Mutex<usize> = Mutex::new(0);
    static CHANGED: Condvar = Condvar::new();

    /// A wait found a trail open below its mark.
    pub(super) fn count() {
        *lock(&BLOCKED) += 1;
        CHANGED.notify_all();
    }

    /// Wait until `waits` waits found a trail open, at most `wait`: whether they did.
    pub(crate) fn reached(waits: usize, wait: Duration) -> bool {
        let blocked = lock(&BLOCKED);
        let (blocked, _) = CHANGED
            .wait_timeout_while(blocked, wait, |blocked| *blocked < waits)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *blocked >= waits
    }
}
