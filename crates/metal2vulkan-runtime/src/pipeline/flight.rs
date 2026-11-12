//! One making in the air per compile: twin makings, whose `VkPipeline`s differ only in what the
//! driver's compile of their functions does not read, take off one after the other.
//!
//! - **Twins.** Two makings are twins where their variants' content (`objects.rs`,
//!   `variant_content`: the layout and each stage's stage, entry point, flags, subgroup size,
//!   specialization layout and SPIR-V) and their specialization match ([`FlightKey`]): states
//!   that differ only in a color format, a blend or a vertex layout, say. On Apple's GPUs MoltenVK
//!   converts their SPIR-V to the same MSL and Metal's compiler service compiles it once, the
//!   second making finding the first's functions in Metal's own cache; made at once, each compiles
//!   them cold (the consumer's cold boot: a state's pipeline made right after its twin's took
//!   1.2 ms of MSL compile, where twins made at once each took the full compile).
//! - **Single flight.** A making that finds its twin in the air waits for it to land, then makes
//!   its own `VkPipeline` as before: each state still makes its own pipeline, driver record,
//!   creation log line and notices, and nothing of one making is handed to another. A failed or
//!   unwound making lands too, and its twin then makes its own. Only the order of the driver
//!   calls changes, which the program sees nowhere: Metal makes a state's pipeline within its
//!   `newRenderPipelineState` call whichever other state it compiles beside.
//! - **No turn held while waiting** (`budget.rs`). A making takes off ([`Flights::board`]) only
//!   once it holds its turn of the process's budget, just before its driver call, and lands when
//!   that returns ([`Leader`]); a twin waits for it holding no turn: a program's thread before it
//!   asks for one, a pool's piece by giving its thread's back for the wait
//!   (`budget::without_turn`). Every making in the air holds a turn, or waits for compile memory
//!   holding none (`compile_memory.rs`), whose holders wait only for a turn: so each wait ends.
//! - **Bounded.** The table holds one key per making in the air, at most the budget's turns.
//! - **Counted.** A making that waited for a twin counts in `WorkCounts::flights_followed`.

use super::budget::{self, Turn};
use crate::device::lock;
use metal2vulkan::cache::Hash256;
use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex};

/// What names a making's flight: the digest of its variant's content and its specialization
/// (the module docs).
pub(crate) type FlightKey = Hash256;

/// The device's makings in the air (`DeviceShared::flights`).
#[derive(Default)]
pub(crate) struct Flights {
    air: Mutex<BTreeSet<FlightKey>>,
    landed: Condvar,
    /// The makings that began to wait for a twin (the unit tests).
    #[cfg(test)]
    following: std::sync::atomic::AtomicUsize,
}

/// A making in the air: it lands when dropped (its driver call returned, failed or unwound), and
/// its twins waiting go on.
pub(crate) struct Leader<'a> {
    flights: &'a Flights,
    key: FlightKey,
}

impl Drop for Leader<'_> {
    fn drop(&mut self) {
        lock(&self.flights.air).remove(&self.key);
        self.flights.landed.notify_all();
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        held::landed(self.key);
    }
}

impl Flights {
    /// The turn for a making of `key` (`budget::program_turn`: none on a pool's thread, which holds
    /// its piece's) and its flight, taken off once no twin is in the air; and whether it waited
    /// for one. A twin that took off while this making waited for its turn is waited for too, the
    /// turn given back meanwhile.
    pub(crate) fn board(&self, key: FlightKey) -> (Option<Turn>, Leader<'_>, bool) {
        let mut followed = false;
        loop {
            followed |= self.follow(&key);
            let turn = budget::program_turn();
            if let Some(leader) = self.take_off(key) {
                return (turn, leader, followed);
            }
            drop(turn);
        }
    }

    /// Wait, holding no turn, while a making of `key` is in the air; whether it waited.
    fn follow(&self, key: &FlightKey) -> bool {
        if !lock(&self.air).contains(key) {
            return false;
        }
        #[cfg(test)]
        self.following
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        held::following();
        budget::without_turn(|| {
            let mut air = lock(&self.air);
            while air.contains(key) {
                air = self
                    .landed
                    .wait(air)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        });
        true
    }

    /// Take off as `key`'s making, unless a twin is in the air.
    fn take_off(&self, key: FlightKey) -> Option<Leader<'_>> {
        if !lock(&self.air).insert(key) {
            return None;
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        held::took_off(key);
        Some(Leader { flights: self, key })
    }
}

#[cfg(test)]
mod tests {
    use super::Flights;
    use crate::exit_guard::spawn_scoped;
    use crate::pipeline::budget::holds_turn;
    use std::sync::atomic::Ordering;
    use std::thread::Builder;
    use std::time::{Duration, Instant};

    /// Wait until `done` holds, at most a minute (a guard against a hang).
    fn until(done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A making boards with its turn and takes off; its twin's, boarding meanwhile, waits for it
    /// holding no turn, and takes off with its own turn once it landed. A making of another key
    /// takes off beside it.
    #[test]
    fn a_twin_boarding_waits_for_its_twin_to_land() {
        let flights = Flights::default();
        let (twin, other) = ([7u8; 32], [8u8; 32]);
        let (turn, leader, followed) = flights.board(twin);
        assert!(turn.is_some() && !followed);
        std::thread::scope(|scope| {
            let follower = spawn_scoped(scope, Builder::new(), || {
                let (turn, leader, followed) = flights.board(twin);
                assert!(turn.is_some() && holds_turn() && followed);
                drop(leader);
            })
            .unwrap();
            let beside = spawn_scoped(scope, Builder::new(), || {
                let (turn, _leader, followed) = flights.board(other);
                assert!(turn.is_some() && !followed);
            })
            .unwrap();
            beside.join().unwrap();
            until(|| flights.following.load(Ordering::SeqCst) == 1);
            assert!(
                !follower.is_finished(),
                "the twin waits while its twin is in the air"
            );
            drop(leader);
            drop(turn);
            follower.join().unwrap();
        });
        assert!(crate::device::lock(&flights.air).is_empty());
    }
}

/// The tests' hold on the makings in the air: a making that takes off waits while a hold lives,
/// in the air and holding its turn, so a twin finds it there; and each flight's take-off and
/// landing, noted in order.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod held {
    use super::FlightKey;
    use crate::device::lock;
    use std::sync::{Condvar, Mutex};

    /// Whether a hold lives, the makings that waited for a twin since it was taken, and the
    /// flights' events in order: a take-off (`true`) or a landing of each key.
    struct State {
        holding: bool,
        following: usize,
        events: Vec<(FlightKey, bool)>,
    }

    static STATE: Mutex<State> = Mutex::new(State {
        holding: false,
        following: 0,
        events: Vec::new(),
    });
    static CHANGED: Condvar = Condvar::new();

    /// The hold, released when dropped.
    pub(crate) struct Hold(());

    /// Hold every making that takes off from now until the hold is dropped.
    pub(crate) fn hold() -> Hold {
        let mut state = lock(&STATE);
        state.holding = true;
        state.following = 0;
        state.events.clear();
        Hold(())
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            lock(&STATE).holding = false;
            CHANGED.notify_all();
        }
    }

    /// The makings that waited for a twin since the hold was taken.
    pub(crate) fn followers() -> usize {
        lock(&STATE).following
    }

    /// The flights' take-offs (`true`) and landings since the hold was taken, in order.
    pub(crate) fn events() -> Vec<(FlightKey, bool)> {
        lock(&STATE).events.clone()
    }

    pub(super) fn following() {
        lock(&STATE).following += 1;
        CHANGED.notify_all();
    }

    pub(super) fn took_off(key: FlightKey) {
        let mut state = lock(&STATE);
        state.events.push((key, true));
        while state.holding {
            state = CHANGED
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    pub(super) fn landed(key: FlightKey) {
        lock(&STATE).events.push((key, false));
    }
}
