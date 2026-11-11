//! Other queues' work held back from a `VkQueue` an unshared queue holds (`unshared_hold.rs`), the
//! parts that need no device: a queue's expected run and a `VkQueue`'s flight.
//!
//! - **Expected runs.** None seen is the budget; a longer run is taken at once, a shorter one
//!   halves the way to it, so a queue whose batches drop from 30 ms to 1 ms is let have two in
//!   flight again within five of them.
//! - **The flight.** Other queues' batches count their expected runs, an unshared queue's none;
//!   the oldest one's run so far counts against its own expected run alone; a batch may go while
//!   less than the budget is left, else it waits for the next completion, or for the time the
//!   oldest one's expected run brings the work left under the budget; a run several batches
//!   complete in is shared among them.

use crate::unshared_hold::{Expected, Flies, Flight, Room, BUDGET};
use std::sync::Arc;

const MS: u64 = 1_000_000;

/// A queue whose batches were seen to run `nanos`.
fn queue(nanos: u64) -> Arc<Expected> {
    let expected = Arc::new(Expected::default());
    expected.observed(nanos);
    expected
}

#[test]
fn a_queue_with_no_run_seen_is_expected_to_run_the_budget() {
    assert_eq!(Expected::default().nanos(), BUDGET);
}

#[test]
fn a_longer_run_is_taken_at_once_and_a_shorter_one_halves_the_way() {
    let expected = queue(4 * MS);
    assert_eq!(expected.nanos(), 4 * MS);
    expected.observed(30 * MS);
    assert_eq!(expected.nanos(), 30 * MS, "a longer run at once");
    expected.observed(10 * MS);
    assert_eq!(expected.nanos(), 20 * MS, "halfway to a shorter one");
    expected.observed(0);
    assert_eq!(
        expected.nanos(),
        10 * MS + 1,
        "a run of nothing counts as 1 ns, the half rounded toward the longer"
    );
}

/// A queue whose batches ran 30 ms, then 1 ms: within five of the short ones, its next batch may
/// go while one of them is in flight, so its throughput is back to the bound's, never one batch at
/// a time for long.
#[test]
fn a_queue_whose_batches_got_short_is_let_ahead_again_within_five_of_them() {
    let app = queue(30 * MS);
    let mut flight = Flight::default();
    let mut value = 0;
    let mut short = 0;
    loop {
        value += 1;
        flight.taken(value, Flies::Other(&app));
        // Another of its batches is ready as this one starts.
        if flight.ahead(0) == Room::Go {
            break;
        }
        flight.completed(value, MS);
        short += 1;
        assert!(
            short <= 5,
            "still one at a time after {short} short batches"
        );
    }
    assert!(short >= 1, "a 30 ms queue goes one at a time at first");
    assert!(app.nanos() < BUDGET, "{}", app.nanos());
}

#[test]
fn an_unshared_queue_s_batches_count_no_work_ahead() {
    let mut flight = Flight::default();
    for value in 1..=8 {
        flight.taken(value, Flies::Unshared);
    }
    assert_eq!(flight.ahead(0), Room::Go);
    flight.taken(9, Flies::Not);
    assert_eq!(flight.len(), 8, "a batch that flies not is not kept");
}

#[test]
fn a_batch_goes_while_less_than_the_budget_is_left_ahead() {
    let app = queue(30 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&app));
    // 30 ms left at its start: held until 28 ms of it ran, or its completion.
    assert_eq!(
        flight.ahead(0),
        Room::Hold {
            after: Some(28 * MS + 1)
        }
    );
    assert_eq!(
        flight.ahead(10 * MS),
        Room::Hold {
            after: Some(18 * MS + 1)
        }
    );
    assert_eq!(flight.ahead(28 * MS + 1), Room::Go);
    // The next one went: 30 ms more ahead, which the oldest's run left does not cover.
    flight.taken(2, Flies::Other(&app));
    assert_eq!(flight.ahead(29 * MS), Room::Hold { after: None });
    // An overrun counts the oldest's expected run alone.
    assert_eq!(flight.ahead(90 * MS), Room::Hold { after: None });
    flight.completed(1, 31 * MS);
    assert_eq!(
        app.nanos(),
        31 * MS,
        "its run raised the queue's expected one"
    );
    assert_eq!(flight.ahead(29 * MS + 1), Room::Go);
}

#[test]
fn a_run_several_batches_complete_in_is_shared_among_them() {
    let first = queue(MS);
    let second = queue(MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&first));
    flight.taken(2, Flies::Unshared);
    flight.taken(3, Flies::Other(&second));
    flight.taken(4, Flies::Other(&second));
    flight.completed(3, 9 * MS);
    assert_eq!(first.nanos(), 3 * MS);
    assert_eq!(second.nanos(), 3 * MS);
    assert_eq!(flight.len(), 1, "the fourth is still in flight");
    // Its own expected run was taken at its submission: 1 ms, under the budget.
    assert_eq!(flight.ahead(0), Room::Go);
    flight.completed(4, MS);
    assert_eq!(flight.len(), 0);
    assert_eq!(flight.ahead(0), Room::Go);
}
