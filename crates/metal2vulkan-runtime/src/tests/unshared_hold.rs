//! Other queues' work held back from a `VkQueue` an unshared queue holds (`unshared_hold.rs`), the
//! parts that need no device: a queue's expected run and a `VkQueue`'s flight.
//!
//! - **Expected runs.** None before one is seen; a longer run is taken at once, a shorter one
//!   halves the way to it, so a queue whose batches drop from 30 ms to 1 ms is let have two in
//!   flight again within five of them.
//! - **The flight.** Other queues' batches count their expected runs, an unshared queue's none;
//!   the oldest one's run so far counts against its own expected run alone; a batch may go while
//!   less than the budget is left and none of unknown length is in flight, else it waits for the
//!   oldest one's completion, or for the time the oldest one's expected run brings the work left
//!   under the budget; a batch whose completion was published counts as left before it leaves; a
//!   run several batches complete in is shared among the other queues' batches, an unshared
//!   queue's taking none.
//! - **Two budgets.** An unshared queue is active while its last commit is less than `ACTIVE`
//!   (50 ms) old, a commit stored after the time was read among them, and not once it is `ACTIVE`
//!   old or where none was made. While it is, the budget is 2 ms; while not, 16 ms: Geekbench 7's
//!   Super Resolution iteration on the Radeon 780M (four command buffers committed at once, at
//!   most 6.5 ms queued ahead of one) is never held, where the 2 ms budget holds it; and an
//!   application's three 30 ms command buffers committed at once hold the third back until the
//!   work left falls under 16 ms, so a compositor's first command buffer after its quiet stretch
//!   waits behind about 16 ms plus one batch, not all three.

use crate::unshared_hold::{active, Expected, Flies, Flight, Room, ACTIVE, BUDGET, IDLE_BUDGET};
use std::sync::Arc;

const MS: u64 = 1_000_000;

/// A queue whose batches were seen to run `nanos`.
fn queue(nanos: u64) -> Arc<Expected> {
    let expected = Arc::new(Expected::default());
    expected.observed(nanos);
    expected
}

#[test]
fn a_queue_with_no_run_seen_has_no_expected_run() {
    assert_eq!(Expected::default().nanos(), None);
}

/// An application that commits three command buffers at once before any run of its queue is
/// seen: the first goes, and the others wait for its completion, which gives its queue its run,
/// however long it ran.
#[test]
fn a_batch_of_unknown_length_holds_the_next_back_until_it_completes() {
    let app = Arc::new(Expected::default());
    let mut flight = Flight::default();
    assert_eq!(flight.ahead(0, 0, BUDGET), Room::Go, "the first goes");
    flight.taken(1, Flies::Other(&app));
    let first = Room::Hold {
        after: None,
        through: 1,
    };
    assert_eq!(flight.ahead(0, 0, BUDGET), first);
    assert_eq!(
        flight.ahead(500 * MS, 0, BUDGET),
        first,
        "however long it has run"
    );
    flight.completed(1, 30 * MS);
    assert_eq!(app.nanos(), Some(30 * MS));
    assert_eq!(flight.ahead(0, 0, BUDGET), Room::Go);
    flight.taken(2, Flies::Other(&app));
    assert_eq!(
        flight.ahead(0, 0, BUDGET),
        Room::Hold {
            after: Some(28 * MS + 1),
            through: 2
        }
    );
}

#[test]
fn a_longer_run_is_taken_at_once_and_a_shorter_one_halves_the_way() {
    let expected = queue(4 * MS);
    assert_eq!(expected.nanos(), Some(4 * MS));
    expected.observed(30 * MS);
    assert_eq!(expected.nanos(), Some(30 * MS), "a longer run at once");
    expected.observed(10 * MS);
    assert_eq!(expected.nanos(), Some(20 * MS), "halfway to a shorter one");
    expected.observed(0);
    assert_eq!(
        expected.nanos(),
        Some(10 * MS + 1),
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
        if flight.ahead(0, 0, BUDGET) == Room::Go {
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
    assert!(
        app.nanos().is_some_and(|nanos| nanos < BUDGET),
        "{:?}",
        app.nanos()
    );
}

#[test]
fn an_unshared_queue_s_batches_count_no_work_ahead() {
    let mut flight = Flight::default();
    for value in 1..=8 {
        flight.taken(value, Flies::Unshared);
    }
    assert_eq!(flight.ahead(0, 0, BUDGET), Room::Go);
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
        flight.ahead(0, 0, BUDGET),
        Room::Hold {
            after: Some(28 * MS + 1),
            through: 1
        }
    );
    assert_eq!(
        flight.ahead(10 * MS, 0, BUDGET),
        Room::Hold {
            after: Some(18 * MS + 1),
            through: 1
        }
    );
    assert_eq!(flight.ahead(28 * MS + 1, 0, BUDGET), Room::Go);
    // The next one went: 30 ms more ahead, which the oldest's run left does not cover.
    flight.taken(2, Flies::Other(&app));
    let both = Room::Hold {
        after: None,
        through: 1,
    };
    assert_eq!(flight.ahead(29 * MS, 0, BUDGET), both);
    // An overrun counts the oldest's expected run alone.
    assert_eq!(flight.ahead(90 * MS, 0, BUDGET), both);
    flight.completed(1, 31 * MS);
    assert_eq!(
        app.nanos(),
        Some(31 * MS),
        "its run raised the queue's expected one"
    );
    assert_eq!(flight.ahead(29 * MS + 1, 0, BUDGET), Room::Go);
}

#[test]
fn a_run_several_batches_complete_in_is_shared_among_the_other_queues_batches() {
    let first = queue(MS);
    let second = queue(MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&first));
    flight.taken(2, Flies::Unshared);
    flight.taken(3, Flies::Other(&second));
    flight.taken(4, Flies::Other(&second));
    flight.completed(3, 9 * MS);
    assert_eq!(
        first.nanos(),
        Some(4 * MS + MS / 2),
        "the unshared batch took no share"
    );
    assert_eq!(second.nanos(), Some(4 * MS + MS / 2));
    assert_eq!(flight.len(), 1, "the fourth is still in flight");
    // Its own expected run was taken at its submission: 1 ms, under the budget.
    assert_eq!(flight.ahead(0, 0, BUDGET), Room::Go);
    flight.completed(4, MS);
    assert_eq!(flight.len(), 0);
    assert_eq!(flight.ahead(0, 0, BUDGET), Room::Go);
}

/// An application's 30 ms batch and the compositor's light one behind it, seen complete together:
/// the application's queue keeps its 30 ms, so its next batch goes only near the end of the one
/// ahead, and a compositor's command buffer waits behind about one of them, not two.
#[test]
fn an_unshared_batch_seen_complete_with_another_queue_s_leaves_it_the_whole_run() {
    let app = queue(30 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&app));
    flight.taken(2, Flies::Unshared);
    flight.completed(2, 30 * MS);
    assert_eq!(app.nanos(), Some(30 * MS));
    flight.taken(3, Flies::Other(&app));
    assert_eq!(
        flight.ahead(14 * MS, 0, BUDGET),
        Room::Hold {
            after: Some(14 * MS + 1),
            through: 3
        },
        "half way through the next one, no other goes"
    );
    // An unshared batch seen complete alone updates nothing.
    flight.taken(4, Flies::Unshared);
    flight.completed(3, 30 * MS);
    flight.completed(4, MS / 10);
    assert_eq!(app.nanos(), Some(30 * MS));
}

/// A completion publishes its value before it takes its batches out of the flight
/// (`QueueSlot::completed`): the batches at or below it count as left already, so a held batch's
/// gate is never one the published value passed, and the oldest batch past them has no run yet.
#[test]
fn a_batch_whose_completion_was_published_counts_as_left_before_it_leaves() {
    let app = Arc::new(Expected::default());
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&app));
    assert_eq!(
        flight.ahead(0, 0, BUDGET),
        Room::Hold {
            after: None,
            through: 1
        }
    );
    assert_eq!(
        flight.ahead(0, 1, BUDGET),
        Room::Go,
        "published, not yet out"
    );
    let long = queue(30 * MS);
    let mut flight = Flight::default();
    flight.taken(1, Flies::Other(&long));
    flight.taken(2, Flies::Other(&long));
    assert_eq!(
        flight.ahead(20 * MS, 1, BUDGET),
        Room::Hold {
            after: Some(28 * MS + 1),
            through: 2
        },
        "the second's run counts from the first's completion, not the first's run"
    );
    assert_eq!(flight.ahead(0, 2, BUDGET), Room::Go);
    assert_eq!(flight.len(), 2, "neither left");
}

/// An unshared queue is active while its last commit is less than `ACTIVE` old, not after, nor
/// before its first commit.
#[test]
fn an_unshared_queue_is_active_for_the_window_after_its_commit() {
    let now = 1_000 * MS;
    assert!(active(now, now), "committed just now");
    assert!(active(now - ACTIVE + 1, now), "within the window");
    assert!(
        active(now + MS, now),
        "a commit stored after the time was read"
    );
    assert!(!active(now - ACTIVE, now), "the window's end");
    assert!(!active(now - 10 * ACTIVE, now), "long quiet");
    assert!(!active(0, now), "never committed");
    assert_eq!(ACTIVE, 50 * MS);
}

/// Geekbench 7's Super Resolution on the Radeon 780M with no hold (reims's perfscout4, gbo-780m):
/// each iteration commits four command buffers at once, of 1.7, 1.7, 2.5 and 4.2 ms on the GPU,
/// at most 6.48 ms queued ahead of one; its queue's expected run is its longest's, 4.2 ms, raised
/// at once by each iteration's last. Beside a compositor quiet for longer than `ACTIVE`, none of
/// them waits (the last goes with 12.6 ms costed ahead, under 16 ms); within `ACTIVE` of its
/// commit, the 2 ms budget holds the second until the first nearly ends.
#[test]
fn super_resolution_s_iterations_are_never_held_beside_a_quiet_compositor() {
    let sr = queue(4_200_000);
    let now = 1_000 * MS;
    for (last, quiet) in [(0, true), (now - ACTIVE, true), (now - MS, false)] {
        let mut flight = Flight::default();
        flight.taken(1, Flies::Unshared);
        for value in 2..=5 {
            let room = flight.room(0, 1, active(last, now));
            match (quiet, value) {
                (_, 2) => assert_eq!(room, Room::Go, "the first goes"),
                (true, _) => assert_eq!(room, Room::Idle, "batch {value} goes, quiet"),
                (false, _) => {
                    assert!(
                        matches!(room, Room::Hold { .. }),
                        "batch {value} held, active"
                    );
                    break;
                }
            }
            flight.taken(value, Flies::Other(&sr));
        }
    }
}

/// An application commits three 30 ms command buffers at once beside a compositor quiet for
/// longer than `ACTIVE`: the second goes once the first has under 16 ms left (where 2 ms would
/// hold it), the third once the second has, so a compositor's first command buffer after its
/// quiet stretch waits behind at most 16 ms of work plus one batch, not all three. Its commit
/// makes the budget 2 ms again: the third then goes once the second has under 2 ms left.
#[test]
fn a_quiet_compositor_s_first_command_buffer_waits_behind_the_idle_budget_and_one_batch() {
    let long = queue(30 * MS);
    let now = 1_000 * MS;
    let quiet = active(now - ACTIVE, now);
    let mut flight = Flight::default();
    assert_eq!(flight.room(0, 0, quiet), Room::Go, "the first goes");
    flight.taken(1, Flies::Other(&long));
    assert_eq!(
        flight.room(0, 0, quiet),
        Room::Hold {
            after: Some(30 * MS - IDLE_BUDGET + 1),
            through: 1
        },
        "the second waits until the first has under 16 ms left"
    );
    assert_eq!(
        flight.room(30 * MS - IDLE_BUDGET, 0, quiet),
        Room::Hold {
            after: Some(1),
            through: 1
        },
        "16 ms left"
    );
    assert_eq!(
        flight.room(30 * MS - IDLE_BUDGET + 1, 0, quiet),
        Room::Idle,
        "under 16 ms left: it goes, where 2 ms would hold it"
    );
    flight.taken(2, Flies::Other(&long));
    assert_eq!(
        flight.room(30 * MS - IDLE_BUDGET + 1, 0, quiet),
        Room::Hold {
            after: None,
            through: 1
        },
        "the third waits for the first's completion: over 16 ms is left until then"
    );
    assert_eq!(
        flight.room(0, 1, quiet),
        Room::Hold {
            after: Some(30 * MS - IDLE_BUDGET + 1),
            through: 2
        },
        "then until the second has under 16 ms left"
    );
    assert_eq!(
        flight.room(0, 1, active(now, now)),
        Room::Hold {
            after: Some(30 * MS - BUDGET + 1),
            through: 2
        },
        "the compositor active again: until the second has under 2 ms left"
    );
}
