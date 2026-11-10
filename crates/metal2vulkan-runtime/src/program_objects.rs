//! The program's objects a release drops or runs on a runtime thread (a no-copy buffer's or
//! heap's deallocator and keep-alive, a range token, a dma-buf source:
//! `exit_guard::program_release`), counted while the runtime holds them, so that a queue waiter
//! runs a command buffer's transition to `Completed` itself only while it holds none
//! (`CommandBufferShared::finish_here`): that transition's release then reaches none of the
//! program's code, which no thread that sees command buffers complete may run.
//!
//! - **Counted while held.** The runtime holds each in a [`ProgramObject`], from where it takes it to where
//!   it drops or runs it, and a `ProgramObject` counts while it lives.
//! - **None held meets such a run.** A waiter counts its run in ([`Unheld::enter`]) before it
//!   reads the count, and a release the program gives for objects a pending command buffer may
//!   already hold (a rebind's new pages, `rebind.rs`; every no-copy buffer's and heap's release
//!   takes the same path) waits, once counted, until no such run is under way ([`settle`]). Both
//!   counts are sequentially consistent: in their one order, either the waiter reads the new
//!   object counted and leaves the transition to its queue's handler thread, or the release
//!   waits for the waiter's run to end. A range token or a dma-buf source is held as its new
//!   resource is made, which no pending command buffer holds yet.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The program's objects the runtime holds.
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The runs under way that found none held ([`Unheld`]).
static UNHELD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The runs under way on this thread that found none held.
    static HERE: Cell<usize> = const { Cell::new(0) };
}

/// One of the program's objects the runtime holds, counted while it lives.
pub(crate) struct ProgramObject<T> {
    value: T,
    _counted: Counted,
}

/// One count of [`HELD`], taken back when dropped.
struct Counted;

impl Drop for Counted {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<T> ProgramObject<T> {
    /// Hold `value`, counted from now on.
    pub(crate) fn new(value: T) -> ProgramObject<T> {
        HELD.fetch_add(1, Ordering::SeqCst);
        ProgramObject {
            value,
            _counted: Counted,
        }
    }

    /// The value, to call or clone where it stays held.
    #[cfg(unix)]
    pub(crate) fn get(&self) -> &T {
        &self.value
    }

    /// Give the value to `f`, which runs or drops it: counted until `f` returned.
    pub(crate) fn run<R>(self, f: impl FnOnce(T) -> R) -> R {
        let ProgramObject { value, _counted } = self;
        f(value)
    }
}

/// A run under way that found none of the program's objects held: none is held for an object a
/// pending command buffer holds until it ends ([`settle`]).
pub(crate) struct Unheld(());

impl Unheld {
    /// Count a run in where none of the program's objects is held; `None` where one is.
    pub(crate) fn enter() -> Option<Unheld> {
        UNHELD.fetch_add(1, Ordering::SeqCst);
        if HELD.load(Ordering::SeqCst) != 0 {
            UNHELD.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        HERE.with(|here| here.set(here.get() + 1));
        Some(Unheld(()))
    }
}

impl Drop for Unheld {
    fn drop(&mut self) {
        HERE.with(|here| here.set(here.get() - 1));
        UNHELD.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Wait until no run counted in by [`Unheld::enter`] is under way, once the objects a release
/// gives are held ([`ProgramObject`]) and before anything holds them: a run that read none held ends first. A
/// run on this thread (the program's code it reached, a message sink) is not waited for.
pub(crate) fn settle() {
    if HERE.with(Cell::get) != 0 {
        return;
    }
    while UNHELD.load(Ordering::SeqCst) != 0 {
        std::thread::yield_now();
    }
}
