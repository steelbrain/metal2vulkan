//! When a command buffer's waiters return: once it finished (its completed handlers ran) and the
//! runtime's threads let go of every reference they took to it.
//!
//! - **Apart from the command buffer.** The [`Gate`] is not part of what it guards, so a thread
//!   lets go of its reference first and leaves the gate after: the gate never opens while a
//!   runtime thread still holds the command buffer, or anything it held for it.
//! - **Why.** A program that waits for its last command buffer and then drops its last handles
//!   destroys the device on its own thread, before the drop returns. Were a runtime thread to
//!   hold the last reference instead, the device would go on that thread, perhaps while the
//!   program exits, which drivers do not survive (RADV ended such a process with `SIGSEGV`).
//! - **And what fed its work** (`trails.rs`). A thread that made what the command buffer waited
//!   for, and holds what it made it with past its signal, holds the device without holding the
//!   command buffer: a pool piece between its making's settle and its give-back (a dispatch's or
//!   a draw's pipeline), the reaper with what such a piece gave back to a
//!   released state, the twin-decode thread between a decode's end and its job's drop. Each opens
//!   a trail before its signal and closes it after its last let-go; the command buffer takes the
//!   device's trail mark as it completes, and `wait_until_completed` and its timed form, once the
//!   gate opened, wait for the trails below it, so the program's drop right after the wait is the
//!   device's last (`tests/teardown_threads.rs` holds a dispatch's piece between the two). The
//!   gate itself, the status and the completed handlers do not wait for trails. Metal's
//!   `waitUntilCompleted` returns after the completed handlers ended too (the M3 Max: 10 µs at the
//!   median, 23 µs at p99, a handler of 20 ms included).
//! - **What it does not cover.** A program that learns of the completion otherwise (from a
//!   handler, an event the command buffer signals, or its status) may drop its last handles while
//!   the runtime's thread still holds the command buffer; the device then goes on that thread.
//!   An exit meanwhile waits for that teardown, or skips it once begun (`exit_guard`).
//! - **Notified only where it opened.** A notify is a system call, and one made while a [`Hold`]
//!   is alive wakes every waiting thread to a gate it must find shut and sleep on again. The
//!   thread that finishes a command buffer holds one over the whole transition
//!   (`Transition::run`), so the finish itself never opens the gate: it sets the flag and leaves
//!   the notify to that hold's going, one wake of the program's thread where there were two.
//!   Counted: `WorkCounts::gate_waits`, `WorkCounts::gate_waits_slept` and
//!   `WorkCounts::gate_wakes_found_shut`, which `tests::gate` holds to none.
//! - **Exit.** Once the process is exiting, the threads that would open a gate end, so a wait
//!   answers at once, and one already blocked within `exit_guard::WAKE`.

use crate::device::lock;
use crate::error::{DeadlockCause, TimedOut};
use crate::exit_guard;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// A wait that cannot end: the process is exiting.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Exiting;

/// What one wait for a gate did, for the counters its caller keeps
/// (`WorkCounts::gate_waits_slept` and `WorkCounts::gate_wakes_found_shut`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Waited {
    /// The gate was shut when the wait arrived, so it slept at least once.
    pub(crate) slept: bool,
    /// Notifies it was woken by that found the gate still shut, and slept again: work the thread
    /// that opens the gate did between its notify and the gate opening. A sleep that ran out its
    /// slice is not one ([`exit_guard::WAKE`] is 50 ms, far past any completion), so each of these
    /// is a notify made while the gate could not yet open.
    pub(crate) woke_shut: u64,
}

impl From<Exiting> for DeadlockCause {
    fn from(Exiting: Exiting) -> DeadlockCause {
        DeadlockCause::Exiting
    }
}

impl From<Exiting> for crate::error::Error {
    fn from(exiting: Exiting) -> crate::error::Error {
        DeadlockCause::from(exiting).into()
    }
}

/// One command buffer's gate.
#[derive(Debug, Default)]
pub(crate) struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    /// The command buffer finished: its completed handlers ran.
    finished: bool,
    /// The [`Hold`]s alive.
    held: usize,
    /// The threads blocked until it opens ([`Gate::awaited`]).
    waiting: usize,
}

impl GateState {
    fn open(&self) -> bool {
        self.finished && self.held == 0
    }
}

impl Gate {
    /// The command buffer finished; the gate opens once no [`Hold`] is left. Whether it notified
    /// the waiters, which it does only where the gate opened here: a notify is a system call, and
    /// one made while a [`Hold`] is still alive wakes every waiting thread to a gate it must find
    /// shut and sleep on again. The thread that finishes a command buffer holds one over the whole
    /// transition (`Transition::run`), so that notify was always such a wake, and the
    /// [`leave`](Self::leave) of its hold notifies for it a moment later.
    pub(crate) fn finish(&self) -> bool {
        let opened = {
            let mut state = lock(&self.state);
            state.finished = true;
            state.open()
        };
        if opened {
            self.changed.notify_all();
        }
        opened
    }

    /// Whether the waiters may return.
    pub(crate) fn is_open(&self) -> bool {
        lock(&self.state).open()
    }

    /// Whether a thread is blocked until the gate opens, in [`wait`](Self::wait) or
    /// [`wait_until`](Self::wait_until): what the thread that finishes the command buffer does
    /// before its hold goes delays that thread's return.
    pub(crate) fn awaited(&self) -> bool {
        lock(&self.state).waiting > 0
    }

    /// Block until the gate opens, answering what the wait did ([`Waited`]). `Err` once the
    /// process is exiting and it is not open: the runtime's threads that would open it end
    /// (`exit_guard`).
    pub(crate) fn wait(&self) -> Result<Waited, Exiting> {
        self.wait_unless(exit_guard::closed)
    }

    /// [`Gate::wait`], with `exiting` saying whether the process is exiting (the exit guard's
    /// [`closed`](exit_guard::closed), or a test's own).
    pub(crate) fn wait_unless(&self, exiting: impl Fn() -> bool) -> Result<Waited, Exiting> {
        let mut state = lock(&self.state);
        state.waiting += 1;
        let mut did = Waited::default();
        // Whether the sleep before this look ended at a notify rather than at its slice.
        let mut notified = false;
        let waited = loop {
            if state.open() {
                break Ok(did);
            }
            did.woke_shut += u64::from(notified);
            if exiting() {
                break Err(Exiting);
            }
            did.slept = true;
            let (woke, how) = self
                .changed
                .wait_timeout(state, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = woke;
            notified = !how.timed_out();
        };
        state.waiting -= 1;
        waited
    }

    /// Block until the gate opens, or until `deadline`, answering what the wait did ([`Waited`]);
    /// at once once the process is exiting, as [`Gate::wait`].
    pub(crate) fn wait_until(&self, deadline: Instant) -> Result<Waited, TimedOut> {
        let mut state = lock(&self.state);
        state.waiting += 1;
        let mut did = Waited::default();
        // Whether the sleep before this look ended at a notify rather than at its slice.
        let mut notified = false;
        let waited = loop {
            if state.open() {
                break Ok(did);
            }
            did.woke_shut += u64::from(notified);
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || exit_guard::closed() {
                break Err(TimedOut);
            }
            did.slept = true;
            let (woke, how) = self
                .changed
                .wait_timeout(state, left.min(exit_guard::WAKE))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = woke;
            notified = !how.timed_out();
        };
        state.waiting -= 1;
        waited
    }

    fn enter(&self) {
        lock(&self.state).held += 1;
    }

    fn leave(&self) {
        let mut state = lock(&self.state);
        state.held -= 1;
        let open = state.open();
        drop(state);
        if open {
            self.changed.notify_all();
        }
    }
}

/// A reference a runtime thread holds (the scheduler's, a completion turn's, a queue
/// context's transition), counted by a [`Gate`]. Its fields drop in declaration order: the
/// reference first, then the count, so the gate cannot open while the reference is alive.
pub(crate) struct Hold<T> {
    value: Arc<T>,
    _count: Count,
}

/// A [`Hold`]'s place in its gate's count: the gate opens only once it went.
pub(crate) struct Count(Arc<Gate>);

impl Drop for Count {
    fn drop(&mut self) {
        self.0.leave();
    }
}

impl<T> Hold<T> {
    /// A new reference to `value`, counted by `gate`.
    pub(crate) fn new(value: &Arc<T>, gate: &Arc<Gate>) -> Hold<T> {
        gate.enter();
        Hold {
            value: Arc::clone(value),
            _count: Count(Arc::clone(gate)),
        }
    }

    /// Whether `self` and `other` hold the same value.
    pub(crate) fn same(&self, other: &Hold<T>) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }

    /// Let go of the reference here unless it is the value's last, answering the hold's place in
    /// its gate's count, which leaves the gate as the caller lets it go. `Err` where it is the
    /// last: the value, still counted by the gate, for the caller to let go of on another thread
    /// ([`Last`]). Whether it is the last is decided as it goes (`Arc::into_inner`), so another
    /// holder's drop meanwhile never leaves the value's drop here.
    pub(crate) fn let_go_short_of_last(self) -> Result<Count, Box<Last<T>>> {
        let Hold { value, _count } = self;
        match Arc::into_inner(value) {
            Some(value) => Err(Box::new(Last { value, _count })),
            None => Ok(_count),
        }
    }
}

/// A [`Hold`]'s value where the hold was its last reference ([`Hold::let_go_short_of_last`]),
/// still counted by its gate: dropped, it drops the value, then leaves the gate.
pub(crate) struct Last<T> {
    value: T,
    _count: Count,
}

impl<T> Deref for Last<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> Deref for Hold<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Arc<T> {
        &self.value
    }
}
