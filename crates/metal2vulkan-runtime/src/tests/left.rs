//! `scheduler/left.rs`: the scheduler thread's way through the work its passes leave it. A commit
//! left to the thread during any round goes before its next piece of work, also one left during
//! the round that took an earlier one in; the work runs in the order the rounds found it, a
//! round's checks before its steps; and the exit ends it.

use super::inbox::{commit_elsewhere, Passes};
use crate::device::lock;
use crate::scheduler::inbox::Inbox;
use crate::scheduler::left::{work_through, Kind, Left};

/// What happened, in order: a commit taken in by a pass, a piece of work run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    TakenIn(u32),
    Ran(Kind, u32),
}

use Event::{Ran, TakenIn};
use Kind::{Check, Step};

type Log = Inbox<Vec<Event>, u32>;

/// Run [`work_through`] over `inbox`, each round a pass of the thread's that takes the commits
/// in, `finds` adding what the round of its number leaves and answering whether another thread
/// commits while its pass is under way (left to it); each pass answers it has work first while
/// work is left, as the thread's do. Whether a piece ran.
fn work(inbox: &Log, mut finds: impl FnMut(u32, &mut Left<u32>) -> bool) -> Option<bool> {
    let (mut rounds, mut commits) = (0, 0);
    work_through(
        inbox,
        &mut Left::default(),
        |left| {
            rounds += 1;
            inbox.pass(|log, taken| {
                log.extend(taken.drain(..).map(TakenIn));
                if finds(rounds, left) {
                    commits += 1;
                    commit_elsewhere(inbox, commits);
                }
                left.is_empty()
            });
            true
        },
        |kind, work| {
            lock(inbox.state()).push(Ran(kind, work));
            true
        },
    )
}

/// The first round leaves two checks, and a commit is left to it; so is another to the round
/// that takes that one in: the thread takes both in before it runs a check, not the second after
/// the first check.
#[test]
fn a_commit_left_to_the_thread_during_any_round_goes_before_its_next_piece_of_work() {
    let inbox = Log::new(Vec::new());
    let ran = work(&inbox, |round, left| {
        if round == 1 {
            left.push(Check, 1);
            left.push(Check, 2);
        }
        round <= 2
    });
    assert_eq!(ran, Some(true));
    assert_eq!(
        *lock(inbox.state()),
        [TakenIn(1), TakenIn(2), Ran(Check, 1), Ran(Check, 2)]
    );
    assert!(!inbox.holds_commits());
}

/// The first round finds steps and checks, a commit left to it; the round that takes it in finds
/// a check and a step. The first round's work runs first, its checks before its steps, then the
/// second's: a step waits for no check found after it.
#[test]
fn work_runs_in_the_order_the_rounds_found_it_a_round_s_checks_before_its_steps() {
    let inbox = Log::new(Vec::new());
    let ran = work(&inbox, |round, left| {
        match round {
            1 => {
                left.push(Step, 1);
                left.push(Check, 2);
                left.push(Step, 3);
                left.push(Check, 4);
            }
            2 => {
                left.push(Step, 5);
                left.push(Check, 6);
            }
            _ => {}
        }
        round == 1
    });
    assert_eq!(ran, Some(true));
    assert_eq!(
        *lock(inbox.state()),
        [
            TakenIn(1),
            Ran(Check, 2),
            Ran(Check, 4),
            Ran(Step, 1),
            Ran(Step, 3),
            Ran(Check, 6),
            Ran(Step, 5),
        ]
    );
}

/// A round that leaves nothing ends it with nothing run; a round or a piece of work that finds the
/// process exiting ends it at once.
#[test]
fn nothing_left_or_the_exit_ends_it() {
    let inbox = Log::new(Vec::new());
    assert_eq!(work(&inbox, |_, _| false), Some(false));
    assert_eq!(
        work_through(&inbox, &mut Left::<u32>::default(), |_| false, |_, _| true),
        None
    );
    let mut ran = Vec::new();
    let exiting = work_through(
        &inbox,
        &mut Left::default(),
        |left| {
            left.push(Check, 1);
            left.push(Check, 2);
            true
        },
        |kind, work| {
            ran.push((kind, work));
            false
        },
    );
    assert_eq!(exiting, None);
    assert_eq!(ran, [(Check, 1)]);
}
