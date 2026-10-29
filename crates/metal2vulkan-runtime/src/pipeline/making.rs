//! A bounded memo of objects the driver makes on demand, made outside the memo's lock: a variant's
//! `VkPipeline`s by graphics key or region specialization, and an emulated mesh pipeline's compute
//! pipelines.
//!
//! - **One making per key.** The first caller of a key the memo lacks makes it, with no lock held;
//!   a caller of the same key meanwhile waits for that making and takes what it made. Callers of
//!   other keys look up and make theirs at the same time, so pipelines that several threads need
//!   are made concurrently, each by the driver's own call (MoltenVK and Mesa make one per calling
//!   thread). The making is the driver's call alone: its result, and what a caller sees of it,
//!   are what one caller making it alone would have.
//! - **Made elsewhere**. A caller may also only begin a key ([`Making::begin`]): it gets the object
//!   made, or the making in flight to look at later ([`Pending`]), or the making itself to hand to
//!   another thread ([`Ticket`]), which settles it with what it made.
//! - **A failed making is not kept.** Its caller gets the error, as before; a caller that waited
//!   for it makes the object itself, as it would have after the failed one (each draw makes its
//!   own attempt where none was made), and so does every later caller. A [`Pending`] look at it
//!   sees the error. A making that panics, or whose ticket is dropped unsettled, leaves the memo
//!   as a failure does.
//! - **Bounded** as the memo it replaces ([`Lru`]): at most its count of entries, made or being
//!   made, the least recently used going first. A making whose entry went meanwhile still hands
//!   its object to the callers waiting for it, and keeps it.
//! - **Waited for first** (ranked). A making handed to the device's pool ([`Pending::queued_at`])
//!   that a committed command buffer's batch waits for ([`Pending::hasten`]), or that a caller
//!   waits for ([`Pending::wait`], or a caller of its key in [`Making::get_or_make`]: a program's
//!   thread, or a host step), goes ahead of the pool's pieces that nothing waits for, at that rank
//!   (`budget::Urgency`, `pool.rs`).

use super::budget::{Asked, Urgency};
use super::memo::{At, Lru};
use super::pool::Place;
use crate::device::lock;
use crate::error::{internal, Error};
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// A memo of objects made on demand, each made once at a time (the module docs).
pub(crate) struct Making<K, V> {
    memo: Arc<Mutex<Lru<K, Entry<V>>>>,
}

/// An entry: the object made, or the turn of its making.
enum Entry<V> {
    Made(Arc<V>),
    Making(Arc<Turn<V>>),
}

impl<V> Clone for Entry<V> {
    fn clone(&self) -> Entry<V> {
        match self {
            Entry::Made(made) => Entry::Made(Arc::clone(made)),
            Entry::Making(turn) => Entry::Making(Arc::clone(turn)),
        }
    }
}

/// What a making's callers are told once it finished: the object, or why none was made.
type Outcome<V> = Result<Arc<V>, Error>;

/// What runs once a making finished, given its outcome (`Pending::when_finished`).
pub(crate) type Waker<V> = Box<dyn FnOnce(&Result<Arc<V>, Error>) + Send>;

/// One making in flight, which the callers of its key wait for.
struct Turn<V> {
    state: Mutex<TurnState<V>>,
    done: Condvar,
    /// Where it waits in the device's pool, once handed there.
    place: Mutex<Option<Place>>,
    /// The most urgent rank asked of it so far: one asked before it was handed to the pool is
    /// applied as it is handed, and a thread making it speculatively reads it
    /// (`budget::speculative`).
    asked: Arc<Asked>,
}

/// A making's state: in flight, with what runs once it finished (`Pending::when_finished`), or
/// finished.
enum TurnState<V> {
    Making(Vec<Waker<V>>),
    Finished(Outcome<V>),
}

impl<V> Turn<V> {
    fn new() -> Turn<V> {
        Turn {
            state: Mutex::new(TurnState::Making(Vec::new())),
            done: Condvar::new(),
            place: Mutex::new(None),
            asked: Arc::new(Asked::new()),
        }
    }

    /// Run it before the pool's pieces of a lower rank than `urgency`, where it waits there, or
    /// once it is handed there ([`queued_at`](Self::queued_at)).
    fn hasten(&self, urgency: Urgency) {
        // Asked before the place is read: `queued_at` sets the place before it reads what was
        // asked, so one of the two applies it.
        self.asked.raise(urgency);
        let place = lock(&self.place).clone();
        // The pool's lock and the budget's, taken with this one released.
        if let Some(place) = place {
            place.hasten(urgency);
        }
    }

    /// It waits at `place` in the device's pool: set once, by the one caller that handed it
    /// there, and hastened at once as asked before.
    fn queued_at(&self, place: Place) {
        {
            let mut set = lock(&self.place);
            if set.is_some() {
                return;
            }
            *set = Some(place.clone());
        }
        if let Some(urgency) = self.asked.get() {
            place.hasten(urgency);
        }
    }

    /// Wait until the making finished, made or failed, hastened meanwhile as this thread's own
    /// making would rank ([`waiter`]).
    fn wait(&self) -> Outcome<V> {
        // A turn's holder waiting for another making could, with every turn so held, stop all
        // of them (`budget.rs`'s module docs).
        debug_assert!(
            !super::budget::holds_turn(),
            "a making waited for another while holding a budget turn"
        );
        self.hasten(waiter());
        let mut state = lock(&self.state);
        loop {
            if let TurnState::Finished(outcome) = &*state {
                return outcome.clone();
            }
            state = self
                .done
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// [`wait`](Self::wait), but no later than `deadline`: `None` where the making had not
    /// finished by then.
    fn wait_until(&self, deadline: Instant) -> Option<Outcome<V>> {
        debug_assert!(
            !super::budget::holds_turn(),
            "a making waited for another while holding a budget turn"
        );
        self.hasten(waiter());
        let mut state = lock(&self.state);
        loop {
            if let TurnState::Finished(outcome) = &*state {
                return Some(outcome.clone());
            }
            let left = deadline.checked_duration_since(Instant::now())?;
            state = self
                .done
                .wait_timeout(state, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// Its outcome, once it finished.
    fn outcome(&self) -> Option<Outcome<V>> {
        match &*lock(&self.state) {
            TurnState::Finished(outcome) => Some(outcome.clone()),
            TurnState::Making(_) => None,
        }
    }

    /// Finish it with `outcome`: the callers waiting go, and its wakers run, given it, with no
    /// lock held.
    fn finish(&self, outcome: Outcome<V>) {
        let before = std::mem::replace(
            &mut *lock(&self.state),
            TurnState::Finished(outcome.clone()),
        );
        self.done.notify_all();
        if let TurnState::Making(wakers) = before {
            for waker in wakers {
                waker(&outcome);
            }
        }
    }
}

/// The rank a making this thread waits for rises to: a host step's in a host step, a warm
/// pass's on its threads (which asks nothing ahead of the program's makings, `warm/pass.rs`),
/// else a program's thread's.
fn waiter() -> Urgency {
    if crate::exit_guard::in_host_step() {
        Urgency::HostStep
    } else if crate::warm::warming() {
        Urgency::Warm
    } else {
        Urgency::Program
    }
}

impl<K: Clone + Eq + Hash, V> Making<K, V> {
    /// An empty memo of at most `max_entries` entries.
    pub(crate) fn counted(max_entries: usize) -> Making<K, V> {
        Making {
            memo: Arc::new(Mutex::new(Lru::counted(max_entries))),
        }
    }

    /// The object of `key`: from the memo, from the making in flight for it, or made now by
    /// `make` with no lock held, and kept.
    pub(crate) fn get_or_make(
        &self,
        key: &K,
        make: impl FnOnce() -> Result<V, Error>,
    ) -> Result<Arc<V>, Error> {
        loop {
            match self.begin(key) {
                Begun::Made(made) => return Ok(made),
                // Look again once it failed: the key is gone, and this caller makes it.
                Begun::Pending(pending) => match pending.turn.wait() {
                    Ok(made) => return Ok(made),
                    Err(_) => continue,
                },
                Begun::Yours(ticket) => return ticket.settle(make()),
            }
        }
    }

    /// Keep `made` under `key`, made before: a released variant's `VkPipeline` taken back
    /// (`kept.rs`).
    pub(crate) fn seed(&self, key: K, made: V) {
        lock(&self.memo).insert(key, Entry::Made(Arc::new(made)), 1);
    }

    /// Begin `key` (the module docs): the object made, the making in flight for it, or this
    /// caller's to make, which another thread may make and settle.
    pub(crate) fn begin(&self, key: &K) -> Begun<K, V> {
        self.begin_at(key).0
    }

    /// [`begin`](Self::begin), with where the entry of an object found made lies, to find it
    /// again by ([`made_at`](Self::made_at)).
    pub(crate) fn begin_at(&self, key: &K) -> (Begun<K, V>, Option<At>) {
        let mut memo = lock(&self.memo);
        let found = memo.get_with_at(key).map(|(at, entry)| (at, entry.clone()));
        match found {
            Some((at, Entry::Made(made))) => (Begun::Made(made), Some(at)),
            Some((_, Entry::Making(turn))) => (Begun::Pending(Pending { turn }), None),
            None => {
                let turn = Arc::new(Turn::new());
                memo.insert(key.clone(), Entry::Making(Arc::clone(&turn)), 1);
                let ticket = Ticket {
                    memo: Arc::clone(&self.memo),
                    key: key.clone(),
                    turn,
                    settled: false,
                };
                (Begun::Yours(ticket), None)
            }
        }
    }

    /// The object of `key` where it is made, as [`begin`](Self::begin) finds it, beginning
    /// nothing where it is not.
    pub(crate) fn made(&self, key: &K) -> Option<Arc<V>> {
        match lock(&self.memo).get(key)? {
            Entry::Made(made) => Some(Arc::clone(made)),
            Entry::Making(_) => None,
        }
    }

    /// The object of `key` where it is made, and where its entry lies, as
    /// [`begin_at`](Self::begin_at) finds it and with the same use of the entry, beginning nothing
    /// where it is not.
    pub(crate) fn made_with_at(&self, key: &K) -> Option<(Arc<V>, At)> {
        match lock(&self.memo).get_with_at(key)? {
            (at, Entry::Made(made)) => Some((Arc::clone(made), at)),
            (_, Entry::Making(_)) => None,
        }
    }

    /// The object made at `at` ([`begin_at`](Self::begin_at)), as [`begin`](Self::begin) of its
    /// key finds it and with the same use of the entry, while the entry stays: an entry stays
    /// what it was kept as, so one found made is made.
    pub(crate) fn made_at(&self, at: At) -> Option<Arc<V>> {
        match lock(&self.memo).get_at(at)? {
            Entry::Made(made) => Some(Arc::clone(made)),
            Entry::Making(_) => None,
        }
    }
}

/// What [`Making::begin`] found.
pub(crate) enum Begun<K: Clone + Eq + Hash, V> {
    /// The object, made.
    Made(Arc<V>),
    /// Another caller's making in flight.
    Pending(Pending<V>),
    /// The making is this caller's.
    Yours(Ticket<K, V>),
}

/// A making in flight, to look at later: once it finished, its object or its error.
pub(crate) struct Pending<V> {
    turn: Arc<Turn<V>>,
}

impl<V> Clone for Pending<V> {
    fn clone(&self) -> Pending<V> {
        Pending {
            turn: Arc::clone(&self.turn),
        }
    }
}

impl<V> Pending<V> {
    /// Its outcome, once it finished; `None` while it is being made.
    pub(crate) fn outcome(&self) -> Option<Result<Arc<V>, Error>> {
        self.turn.outcome()
    }

    /// Its outcome, waiting until it finished, hastened meanwhile (the module docs).
    pub(crate) fn wait(&self) -> Result<Arc<V>, Error> {
        self.turn.wait()
    }

    /// [`wait`](Self::wait), but no later than `deadline`: `None` where it had not finished by
    /// then (a command buffer's bound for the pipelines it draws with, `render/pending.rs`).
    pub(crate) fn wait_until(&self, deadline: Instant) -> Option<Result<Arc<V>, Error>> {
        self.turn.wait_until(deadline)
    }

    /// It waits at `place` in the device's pool.
    pub(crate) fn queued_at(&self, place: Place) {
        self.turn.queued_at(place);
    }

    /// Run it before the pool's pieces of a lower rank than `urgency`, where it waits there: a
    /// committed command buffer's submission waits for it (`budget::Urgency::Committed`), or a
    /// queue's first held batch's (`budget::Urgency::Head`).
    pub(crate) fn hasten(&self, urgency: Urgency) {
        self.turn.hasten(urgency);
    }

    /// Run `waker` once it finished, given its outcome, with no lock held: now, where it
    /// finished already.
    pub(crate) fn when_finished(&self, waker: Waker<V>) {
        let outcome = {
            let mut state = lock(&self.turn.state);
            match &mut *state {
                TurnState::Making(wakers) => {
                    wakers.push(waker);
                    return;
                }
                TurnState::Finished(outcome) => outcome.clone(),
            }
        };
        waker(&outcome);
    }
}

/// A making that is its holder's: settled with what it made ([`Ticket::settle`]); dropped
/// unsettled, it fails, as a making that fails does (the module docs).
pub(crate) struct Ticket<K: Clone + Eq + Hash, V> {
    memo: Arc<Mutex<Lru<K, Entry<V>>>>,
    key: K,
    turn: Arc<Turn<V>>,
    settled: bool,
}

impl<K: Clone + Eq + Hash, V> Ticket<K, V> {
    /// The making, to look at later.
    pub(crate) fn pending(&self) -> Pending<V> {
        Pending {
            turn: Arc::clone(&self.turn),
        }
    }

    /// Settle the making with `made`: an object is kept and handed to the callers waiting; an
    /// error is not kept, and is theirs to see.
    pub(crate) fn settle(mut self, made: Result<V, Error>) -> Result<Arc<V>, Error> {
        self.settled = true;
        let outcome = made.map(Arc::new);
        self.finish(outcome.clone());
        outcome
    }

    fn finish(&self, outcome: Outcome<V>) {
        {
            let mut memo = lock(&self.memo);
            // The entry is this making's unless it was evicted meanwhile (and maybe made again by
            // another caller since): only this making's own entry is replaced or forgotten.
            let ours = matches!(
                memo.get(&self.key),
                Some(Entry::Making(turn)) if Arc::ptr_eq(turn, &self.turn)
            );
            match &outcome {
                Ok(made) => memo.insert(self.key.clone(), Entry::Made(Arc::clone(made)), 1),
                Err(_) if ours => memo.remove(&self.key),
                Err(_) => {}
            }
        }
        self.turn.finish(outcome);
    }
}

impl<K: Clone + Eq + Hash, V> Drop for Ticket<K, V> {
    fn drop(&mut self) {
        if !self.settled {
            self.finish(Err(internal(
                "a pipeline's making ended without its object",
            )));
        }
    }
}
