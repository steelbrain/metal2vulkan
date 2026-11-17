//! Command queues taking turns on a `VkQueue` they share (`queue_turns.rs`), the parts that need
//! no device: each queue's own work in a `VkQueue`'s flight.
//!
//! - **Its own work alone.** A queue's batch is held only by its own queue's batches in flight,
//!   each at its queue's expected run: another queue's burst ahead of it holds it not at all, so a
//!   queue that submits after a burst goes at once.
//! - **Two budgets.** A queue alone keeps less than 16 ms of its own work in flight (the heads past
//!   2 ms let go as idle ones); once another queue took a batch there within 50 ms, less than
//!   2 ms; 50 ms after that queue's last, 16 ms again. An unshared queue's batches hold no turn of
//!   their own, and count as another queue's.
//! - **Priced at the run now.** A run longer than a queue's recent ones prices every batch of it
//!   still in flight at once, where the unshared bound prices each at its submission's run.
//! - **Unknown length.** A queue with no run seen keeps one batch in flight until it completes,
//!   and holds no other queue back.
//! - **Shares of a run.** A light batch seen complete together with another queue's long one keeps
//!   a light run (`unshared_hold::Shares`), so a compositor's plain queue is never held behind its
//!   own draw stuck behind an application's batch.
//! - **The oldest's run.** What the oldest batch in flight ran counts against its own queue's work
//!   alone, and only while no completion past it was published; the hold waits for the oldest
//!   batch's completion, and, where that is its own queue's, for the time its run brings the work
//!   left under the budget.

use crate::unshared_hold::{Expected, Flies, Flight, Room, ACTIVE, BUDGET, IDLE_BUDGET};
use std::sync::Arc;

const MS: u64 = 1_000_000;

/// A host time well past 0, which stands for none.
const T0: u64 = 1_000 * MS;

/// The burst's queue and another.
const A: u64 = 1;
const B: u64 = 2;
const UNSHARED: u64 = 3;

/// A queue whose batches were seen to run `nanos`.
fn queue(nanos: u64) -> Arc<Expected> {
    let expected = Arc::new(Expected::default());
    expected.observed(nanos);
    expected
}

/// Queue `id`'s turn in `flight` at host time `now`, nothing completed, the oldest batch having
/// run `ran`.
fn turn(flight: &Flight, (id, of): (u64, &Expected), ran: u64, now: u64) -> Room {
    flight.turn((id, of.nanos()), ran, 0, now)
}

/// A queue alone keeps less than 16 ms of its 4 ms batches in flight: the second to the fourth
/// go as idle ones, which 2 ms would hold, and the fifth is held until the first has run 1 ns of
/// its expected run, or completes.
#[test]
fn a_queue_alone_keeps_less_than_16_ms_of_its_own_work_in_flight() {
    let app = queue(4 * MS);
    let mut flight = Flight::default();
    assert_eq!(turn(&flight, (A, &app), 0, T0), Room::Go, "none in flight");
    for value in 1..=3 {
        flight.taken(value, Flies::Other(A, &app), T0);
        assert_eq!(
            turn(&flight, (A, &app), 0, T0),
            Room::Idle,
            "{value} in flight"
        );
    }
    flight.taken(4, Flies::Other(A, &app), T0);
    assert_eq!(
        turn(&flight, (A, &app), 0, T0),
        Room::Hold {
            after: Some(1),
            through: 1
        }
    );
    assert_eq!(
        turn(&flight, (A, &app), 1, T0),
        Room::Idle,
        "1 ns of it ran"
    );
}

/// Another queue's batch, taken there after a burst of A's, goes at once: A's work holds it not
/// at all. From then on A keeps less than 2 ms of its own in flight, until 50 ms after B's last.
#[test]
fn a_later_queue_goes_ahead_of_another_queue_s_burst() {
    let (app, probe) = (queue(4 * MS), queue(MS / 10));
    let mut flight = Flight::default();
    for value in 1..=4 {
        flight.taken(value, Flies::Other(A, &app), T0);
    }
    let later = T0 + 300 * MS;
    assert_eq!(
        turn(&flight, (B, &probe), 0, later),
        Room::Go,
        "the burst holds B not at all"
    );
    flight.taken(5, Flies::Other(B, &probe), later);
    // A's oldest ran 3 ms of its 4: 13 ms of A's left, 2 ms or more, which the 1 ms left of the
    // oldest's run does not bring under 2 ms: held until it completes.
    let a_held = Room::Hold {
        after: None,
        through: 1,
    };
    assert_eq!(turn(&flight, (A, &app), 3 * MS, later + 1), a_held);
    assert_eq!(
        turn(&flight, (B, &probe), 3 * MS, later + 1),
        Room::Go,
        "0.1 ms of B's own left: under 2 ms"
    );
    // B's last is 50 ms old: A's budget is 16 ms again, and 13 ms is under it.
    assert_eq!(turn(&flight, (A, &app), 3 * MS, later + ACTIVE), Room::Idle);
    assert_eq!(
        turn(&flight, (A, &app), 3 * MS, later + ACTIVE - 1),
        a_held,
        "49.99 ms after B's last"
    );
}

/// A batch of a queue whose last batch here is the latest (its own) finds the other queue's
/// latest one: B's batch, then A's, then A's turn sees B's as the other one's.
#[test]
fn the_latest_batch_of_another_queue_is_kept_behind_a_queue_s_own() {
    let app = queue(4 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(B, &app), T0);
    flight.taken(2, Flies::Other(A, &app), T0 + MS);
    flight.taken(3, Flies::Other(A, &app), T0 + 2 * MS);
    // A has 8 ms in flight: held under B's 2 ms budget, not under 16 ms.
    assert!(matches!(
        turn(&flight, (A, &app), 0, T0 + 3 * MS),
        Room::Hold { .. }
    ));
    assert_eq!(
        turn(&flight, (A, &app), 0, T0 + ACTIVE),
        Room::Idle,
        "B's batch went 50 ms ago"
    );
}

/// An unshared queue's batches hold no turn of their own, and bring the 2 ms budget as another
/// queue's do.
#[test]
fn an_unshared_queue_s_batches_hold_no_turn_and_count_as_another_queue_s() {
    let app = queue(4 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(A, &app), T0);
    assert_eq!(turn(&flight, (A, &app), 0, T0), Room::Idle);
    flight.taken(2, Flies::Unshared(UNSHARED), T0);
    assert!(matches!(turn(&flight, (A, &app), 0, T0), Room::Hold { .. }));
    assert_eq!(
        turn(&flight, (UNSHARED, &app), 0, T0),
        Room::Go,
        "its own batch counts nothing"
    );
}

/// A queue whose 0.5 ms batches went four at a time ran 8 ms in the first: the three left count
/// 8 ms each at once (24 ms: held under either budget), where the unshared bound still counts
/// the 0.5 ms each had at its submission.
#[test]
fn a_longer_run_prices_every_batch_of_its_queue_in_flight_at_once() {
    let app = queue(MS / 2);
    let mut flight = Flight::default();
    for value in 1..=4 {
        flight.taken(value, Flies::Other(A, &app), T0);
    }
    flight.completed(1, 8 * MS);
    assert_eq!(app.nanos(), Some(8 * MS));
    let held = Room::Hold {
        after: None,
        through: 2,
    };
    assert_eq!(turn(&flight, (A, &app), 0, T0), held, "24 ms left");
    assert_eq!(flight.own_ahead((A, app.nanos()), 0, 0, IDLE_BUDGET), held);
    assert_eq!(
        flight.ahead(0, 0, BUDGET),
        Room::Go,
        "the unshared bound: 1.5 ms at the runs they were taken at"
    );
}

/// A queue with no run seen keeps one batch in flight until it is seen complete, however long it
/// runs, and holds no other queue back.
#[test]
fn a_queue_with_no_run_seen_keeps_one_batch_in_flight() {
    let (new, other) = (Arc::new(Expected::default()), queue(4 * MS));
    let mut flight = Flight::default();
    assert_eq!(turn(&flight, (A, &new), 0, T0), Room::Go, "the first goes");
    flight.taken(1, Flies::Other(A, &new), T0);
    let first = Room::Hold {
        after: None,
        through: 1,
    };
    assert_eq!(turn(&flight, (A, &new), 0, T0), first);
    assert_eq!(
        turn(&flight, (A, &new), 500 * MS, T0),
        first,
        "however long"
    );
    assert_eq!(
        turn(&flight, (B, &other), 0, T0),
        Room::Go,
        "another queue's batch goes"
    );
    flight.completed(1, 3 * MS);
    assert_eq!(new.nanos(), Some(3 * MS));
    assert_eq!(turn(&flight, (A, &new), 0, T0), Room::Go);
}

/// The oldest batch's run counts against its own queue's work alone: A's work behind B's long
/// batch is held until B's completes, with no time to wait for; once A's is the oldest, its run
/// counts, and the hold waits for the time that brings A's work left under the budget.
#[test]
fn the_oldest_batch_s_run_counts_against_its_own_queue_s_work_alone() {
    let (app, long) = (queue(4 * MS), queue(30 * MS));
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(B, &long), T0);
    flight.taken(2, Flies::Other(A, &app), T0);
    let now = T0 + MS;
    // B's ran 20 ms: none of it counts for A, whose 4 ms waits behind it.
    let behind = Room::Hold {
        after: None,
        through: 1,
    };
    assert_eq!(turn(&flight, (A, &app), 20 * MS, now), behind);
    // B's completion was published, not yet taken out: A's is the oldest, none of its run counts
    // yet, and the hold waits for 2 ms more of it.
    let published = Room::Hold {
        after: Some(2 * MS + 1),
        through: 2,
    };
    assert_eq!(flight.turn((A, app.nanos()), 20 * MS, 1, now), published);
    flight.completed(1, 30 * MS);
    assert_eq!(turn(&flight, (A, &app), 0, now), published);
    assert_eq!(
        turn(&flight, (A, &app), 2 * MS + 1, now),
        Room::Go,
        "under 2 ms of A's left"
    );
}

/// A batch whose completion was published counts as left before it is taken out of the flight.
#[test]
fn a_published_completion_counts_its_batches_as_left() {
    let app = queue(4 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(A, &app), T0);
    flight.taken(2, Flies::Other(A, &app), T0);
    flight.taken(3, Flies::Other(B, &app), T0);
    assert!(matches!(
        flight.turn((A, app.nanos()), 0, 0, T0),
        Room::Hold { .. }
    ));
    assert_eq!(flight.turn((A, app.nanos()), 0, 2, T0), Room::Go);
    flight.completed(2, 8 * MS);
    assert_eq!(turn(&flight, (A, &app), 0, T0), Room::Go);
    assert_eq!(flight.len(), 1);
}

/// A compositor's light draw seen complete together with the application's 30 ms batch it waited
/// behind keeps its 0.1 ms, and the application its 30 ms: the compositor's next draw goes with
/// one in flight. An equal share, 15 ms each, held each draw until the one before it completed,
/// behind the application's next batch: a draw for each of the application's (RADV PHOENIX: a
/// draw committed every 16 ms fell 2 s behind).
#[test]
fn a_light_batch_seen_complete_with_another_queue_s_long_one_keeps_its_light_run() {
    let (app, compositor) = (queue(30 * MS), queue(MS / 10));
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(A, &app), T0);
    flight.taken(2, Flies::Other(B, &compositor), T0);
    flight.completed(2, 30 * MS + MS / 10);
    assert_eq!(app.nanos(), Some(30 * MS));
    assert_eq!(compositor.nanos(), Some(MS / 10));
    flight.taken(3, Flies::Other(A, &app), T0);
    flight.taken(4, Flies::Other(B, &compositor), T0);
    assert_eq!(
        turn(&flight, (B, &compositor), 0, T0),
        Room::Go,
        "0.1 ms of its own in flight"
    );
}

/// A batch of a queue with no run seen, seen complete with another queue's, takes the run beyond
/// the other's expected one, and 1 ns where the run is shorter than that; where every one is of
/// unknown length, they share the run equally.
#[test]
fn a_batch_of_unknown_length_takes_the_run_beyond_the_others_expected_ones() {
    let (app, new, newer) = (
        queue(30 * MS),
        Arc::new(Expected::default()),
        Arc::new(Expected::default()),
    );
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(A, &app), T0);
    flight.taken(2, Flies::Other(B, &new), T0);
    flight.completed(2, 30 * MS + MS / 2);
    assert_eq!(app.nanos(), Some(30 * MS));
    assert_eq!(new.nanos(), Some(MS / 2));
    flight.taken(3, Flies::Other(A, &app), T0);
    flight.taken(4, Flies::Other(B, &newer), T0);
    flight.completed(4, 20 * MS);
    assert_eq!(app.nanos(), Some(25 * MS), "halved toward the whole run");
    assert_eq!(newer.nanos(), Some(1), "none of it left");
    let (first, second) = (Arc::new(Expected::default()), Arc::new(Expected::default()));
    flight.taken(5, Flies::Other(A, &first), T0);
    flight.taken(6, Flies::Other(B, &second), T0);
    flight.completed(6, 8 * MS);
    assert_eq!(
        (first.nanos(), second.nanos()),
        (Some(4 * MS), Some(4 * MS))
    );
}
