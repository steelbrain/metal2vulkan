//! What a render state's pool pieces hold: kept by the state, so a pool thread is never the last
//! holder of the device.
//!
//! A draw's making handed to the device's pool (`pool.rs`) needs its variant, its pass and its
//! making's ticket: each holds the device. Were the piece to hold them, the program could let go of the command buffer, the
//! state and the device while the piece waits or runs, and the piece's release would then tear
//! the device down on the pool's thread, perhaps while the process exits: after destructors the
//! exit hook could not precede freed what that teardown calls into, RADV and ANV ended such
//! processes with `SIGSEGV` in the validation layer's `vkDestroySemaphore`.
//!
//! - **Kept by the state.** [`PieceHolds::keep`], on the thread that hands the piece over, keeps
//!   what the piece needs here, in the state; the piece carries a [`Keeping`], which holds
//!   nothing of the device.
//! - **Taken while it runs.** [`Keeping::start`], as the piece starts, takes what was kept for
//!   it with a [`Lease`]; once the state let go ([`PieceHolds::close`]) it takes nothing, and the
//!   piece stops: its making fails, as a piece dropped unrun does.
//! - **Given back, never let go of on a pool thread.** The piece gives back what remains of it
//!   ([`Lease::give_back`]). What pieces gave back is let go of on the thread of the next
//!   hand-over, where the state, which holds the device, is alive, or with the state, on the
//!   releasing thread.
//! - **A release never waits** (the coordinator's rule: Metal's release never blocks, and a
//!   making takes 20 to 50 s on NVIDIA). The state's release lets go of what was kept and given
//!   back at once, and starts no thread; a piece running then gives back to the device's reaper
//!   (`reaper.rs`), which the piece starts unless it runs, and its thread lets go of it. Where
//!   the host cannot start that thread, the piece lets go of it itself. A piece that panics drops
//!   what it took on its thread.
//! - **With its trail.** A piece that settled its making before it gives back gives back with the
//!   trail it opened before the settle (`trails.rs`): it closes once the state keeps what was
//!   given back, or once the reaper (or the piece, without one) let go of it.

use super::reaper::Reaper;
use crate::device::lock;
use crate::trails::Trail;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What a state keeps for its pool pieces ([`PieceHolds::keep`]), by piece.
pub(crate) struct PieceHolds<T> {
    state: Mutex<Holds<T>>,
}

struct Holds<T> {
    /// The state let go: no piece takes what was kept any more.
    closed: bool,
    /// Where the pieces running as the state let go give back: the device's reaper, which the
    /// first of them starts; `None` before.
    reaper: Option<Arc<Reaper>>,
    /// The leases not ended.
    running: usize,
    next: u64,
    /// What was kept for each piece not started, by its number.
    waiting: HashMap<u64, T>,
    /// What pieces that ran gave back.
    given_back: Vec<T>,
}

/// A piece's claim on what its state keeps for it ([`Keeping::start`]): holds nothing of it.
pub(crate) struct Keeping<T> {
    holds: Arc<PieceHolds<T>>,
    number: u64,
}

/// A running piece's lease, through which it gives back what it took.
pub(crate) struct Lease<T> {
    holds: Arc<PieceHolds<T>>,
}

impl<T: Send + 'static> PieceHolds<T> {
    pub(crate) fn new() -> Arc<PieceHolds<T>> {
        Arc::new(PieceHolds {
            state: Mutex::new(Holds {
                closed: false,
                reaper: None,
                running: 0,
                next: 0,
                waiting: HashMap::new(),
                given_back: Vec::new(),
            }),
        })
    }

    /// Keep `held` for a piece about to be handed to the pool, and let go, on this thread, of
    /// what pieces that ran gave back.
    pub(crate) fn keep(self: &Arc<Self>, held: T) -> Keeping<T> {
        let number = {
            let mut state = lock(&self.state);
            let number = state.next;
            state.next += 1;
            state.waiting.insert(number, held);
            state.given_back.clear();
            number
        };
        Keeping {
            holds: Arc::clone(self),
            number,
        }
    }

    /// The state's release, returning at once: no piece takes what was kept any more; a piece
    /// running gives back to `reaper`, which it starts ([`Lease::give_back`]), never this
    /// release; everything kept is let go of on this thread.
    pub(crate) fn close(&self, reaper: &Arc<Reaper>) {
        let mut state = lock(&self.state);
        state.closed = true;
        if state.running > 0 {
            state.reaper = Some(Arc::clone(reaper));
        }
        state.waiting.clear();
        state.given_back.clear();
    }
}

impl<T> Keeping<T> {
    /// As the piece starts: what was kept for it, with its lease; `None` once the state let go,
    /// and the piece stops.
    pub(crate) fn start(self) -> Option<(Lease<T>, T)> {
        let mut state = lock(&self.holds.state);
        if state.closed {
            return None;
        }
        let held = state.waiting.remove(&self.number)?;
        state.running += 1;
        drop(state);
        Some((Lease { holds: self.holds }, held))
    }
}

/// What a piece of a released state hands the reaper: what it gave back, then its trail, which
/// closes once that went (a struct's fields drop in the order they are declared).
struct Reaped<T> {
    _held: T,
    _trail: Option<Trail>,
}

impl<T: Send + 'static> Lease<T> {
    /// Give back what remains of what the piece took, and end the lease: kept by the state until
    /// a hand-over's thread or the state's release lets go of it, or, once the state let go,
    /// handed to the reaper, whose thread it starts unless it runs. `trail`, the piece's where it
    /// settled its making before, closes once the state keeps it, or once it was let go of.
    pub(crate) fn give_back(self, held: T, trail: Option<Trail>) {
        let mut state = lock(&self.holds.state);
        if !state.closed {
            state.given_back.push(held);
            drop(state);
            drop(trail);
            return;
        }
        // The reaper's thread starts here, outside the lock, which the release takes: a start
        // can wait (`reaper.rs`).
        let reaper = state.reaper.clone();
        drop(state);
        match reaper.and_then(|reaper| reaper.start()) {
            Some(reaps) => reaps.reap(Box::new(Reaped {
                _held: held,
                _trail: trail,
            })),
            // No reaper thread: let go of it here, then of the trail.
            None => {
                drop(held);
                drop(trail);
            }
        }
    }
}

impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        lock(&self.holds.state).running -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::reaper::Reaper;
    use super::PieceHolds;
    use crate::exit_guard;
    use crate::trails::Trails;
    use std::sync::{mpsc, Arc};
    use std::thread::{self, Builder};
    use std::time::{Duration, Instant};

    /// The longest a test waits for another thread.
    const WAIT: Duration = Duration::from_secs(60);

    /// A value that sends the name of the thread it was dropped on.
    struct Noted(mpsc::Sender<String>);

    impl Drop for Noted {
        fn drop(&mut self) {
            let _ = self.0.send(here());
        }
    }

    /// This thread's name.
    fn here() -> String {
        thread::current().name().unwrap_or("unnamed").to_owned()
    }

    /// What a piece took and gave back goes on the next hand-over's thread, never on the
    /// piece's, and the piece takes what was kept for it.
    #[test]
    fn what_a_piece_gave_back_goes_on_the_next_hand_over_s_thread() {
        let (noted, dropped) = mpsc::channel();
        let reaper = Arc::new(Reaper::default());
        let holds = PieceHolds::new();
        let keeping = holds.keep(Noted(noted.clone()));
        exit_guard::spawn(Builder::new(), move || {
            let (lease, held) = keeping.start().expect("kept");
            lease.give_back(held, None);
        })
        .unwrap()
        .join()
        .unwrap();
        assert!(dropped.try_recv().is_err(), "the piece let go of it");
        drop(holds.keep(Noted(noted)));
        assert_eq!(dropped.try_recv().unwrap(), here());
        holds.close(&reaper);
        assert_eq!(
            dropped.try_recv().unwrap(),
            here(),
            "the state's release let go of what a piece never took"
        );
    }

    /// Once the state let go, a piece takes nothing (it stops), and what was kept for it went
    /// with the state, on the releasing thread.
    #[test]
    fn a_piece_of_a_released_state_takes_nothing() {
        let (noted, dropped) = mpsc::channel();
        let reaper = Arc::new(Reaper::default());
        let holds = PieceHolds::new();
        let keeping = holds.keep(Noted(noted));
        holds.close(&reaper);
        assert_eq!(dropped.try_recv().unwrap(), here());
        assert!(
            exit_guard::spawn(Builder::new(), move || keeping.start().is_none())
                .unwrap()
                .join()
                .unwrap()
        );
    }

    /// The state's release returns while a piece runs, and starts no thread (a start can wait,
    /// `reaper.rs`); what the piece gives back as it ends goes on the reaper's thread, which the
    /// piece started, never on the piece's.
    #[test]
    fn the_state_s_release_returns_while_a_piece_runs_and_the_reaper_lets_go_of_it() {
        let (noted, dropped) = mpsc::channel();
        let reaper = Arc::new(Reaper::default());
        let holds = PieceHolds::new();
        let keeping = holds.keep(Noted(noted));
        let (started, running) = mpsc::channel();
        let (end, ending) = mpsc::channel::<()>();
        let piece = exit_guard::spawn(Builder::new().name("piece".into()), move || {
            let (lease, held) = keeping.start().expect("kept");
            started.send(()).unwrap();
            ending.recv_timeout(WAIT).unwrap();
            lease.give_back(held, None);
        })
        .unwrap();
        running.recv_timeout(WAIT).unwrap();
        holds.close(&reaper);
        assert!(
            dropped.try_recv().is_err(),
            "the release let go of the running piece's"
        );
        assert!(!reaper.runs(), "the release started the reaper's thread");
        end.send(()).unwrap();
        piece.join().unwrap();
        assert!(reaper.runs(), "the piece did not start the reaper's thread");
        assert_eq!(dropped.recv_timeout(WAIT).unwrap(), "metal2vulkan-reaper");
        reaper.stop();
    }

    /// A piece of a state released while it ran gives back with its trail: the reaper lets go of
    /// what it gave back while the trail is open, then the trail closes, and a wait at a mark
    /// taken after the trail opened returns.
    #[test]
    fn a_trail_given_back_to_the_reaper_closes_after_what_it_held() {
        /// Sends how many trails are open as it is dropped.
        struct Counted(Arc<Trails>, mpsc::Sender<usize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                let _ = self.1.send(self.0.open_now());
            }
        }
        let trails = Arc::new(Trails::default());
        let (noted, dropped) = mpsc::channel();
        let reaper = Arc::new(Reaper::default());
        let holds = PieceHolds::new();
        let keeping = holds.keep(Counted(Arc::clone(&trails), noted));
        let (started, running) = mpsc::channel();
        let (end, ending) = mpsc::channel::<()>();
        let piece = {
            let trails = Arc::clone(&trails);
            exit_guard::spawn(Builder::new().name("piece".into()), move || {
                let (lease, held) = keeping.start().expect("kept");
                started.send(()).unwrap();
                ending.recv_timeout(WAIT).unwrap();
                let trail = trails.open();
                lease.give_back(held, Some(trail));
            })
            .unwrap()
        };
        running.recv_timeout(WAIT).unwrap();
        holds.close(&reaper);
        end.send(()).unwrap();
        piece.join().unwrap();
        assert_eq!(
            dropped.recv_timeout(WAIT).unwrap(),
            1,
            "the reaper let go of what was given back with its trail open"
        );
        assert!(trails
            .wait_below(trails.mark(), Some(Instant::now() + WAIT))
            .is_ok());
        assert_eq!(trails.open_now(), 0);
        reaper.stop();
    }

    /// What the reaper lets go of may hold the last share of the reaper itself, as the device's
    /// last holder does: its teardown stops the reaper on the reaper's own thread, which ends
    /// without joining itself.
    #[test]
    fn the_reaper_stopped_on_its_own_thread_ends() {
        /// Stops the reaper it holds as it goes, then sends the thread's name.
        struct Last(Arc<Reaper>, mpsc::Sender<String>);
        impl Drop for Last {
            fn drop(&mut self) {
                self.0.stop();
                let _ = self.1.send(here());
            }
        }
        let (sender, stopped) = mpsc::channel();
        let reaper = Arc::new(Reaper::default());
        let reaps = reaper.start().expect("a thread");
        reaps.reap(Box::new(Last(Arc::clone(&reaper), sender)));
        drop((reaper, reaps));
        assert_eq!(stopped.recv_timeout(WAIT).unwrap(), "metal2vulkan-reaper");
    }
}
