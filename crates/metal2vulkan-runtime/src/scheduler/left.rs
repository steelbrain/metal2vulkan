//! What the scheduler's passes leave its thread to do without the lock (a batch's submission
//! checks; a batch's host step, which it hands to the steps' thread, `steps.rs`), and the thread's
//! way through it once woken ([`work_through`]).
//!
//! - **Commits first.** Before each piece of work, the thread makes rounds for as long as the
//!   inbox holds commits ([`Inbox::holds_commits`]): a commit left to the thread while a round's
//!   pass held the lock (`inbox.rs`) is taken in by the next round, never after a piece of work,
//!   which runs the program's code (a check) or hands a step over. Those rounds end at the first
//!   that leaves the inbox without one; while commits keep coming during each of the thread's
//!   passes, its work waits for them.
//! - **In the order found.** The work runs in the order the thread's rounds found it, a round's
//!   checks before its steps: a piece waits for the work found before it, never for work a later
//!   round found, which the rounds between pieces add.

use super::inbox::Inbox;
use std::collections::VecDeque;

/// A kind of work the thread does without its lock, in the order a round's finds run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    /// A batch's submission checks (`command_buffer/checks.rs`).
    Check,
    /// A batch's host step, handed to the steps' thread (`steps.rs`).
    Step,
}

/// The work the thread's passes left it, oldest first, each piece with its kind.
pub(crate) struct Left<W> {
    work: VecDeque<(Kind, W)>,
    /// How many of `work`, from the front, rounds before the current one found.
    ordered: usize,
}

impl<W> Default for Left<W> {
    fn default() -> Left<W> {
        Left {
            work: VecDeque::new(),
            ordered: 0,
        }
    }
}

impl<W> Left<W> {
    pub(crate) fn is_empty(&self) -> bool {
        self.work.is_empty()
    }

    /// Leave `work` of `kind`, found by the current round.
    pub(crate) fn push(&mut self, kind: Kind, work: W) {
        self.work.push_back((kind, work));
    }

    /// Whether a piece of `kind` that `same` picks is left.
    pub(crate) fn holds(&self, kind: Kind, same: impl Fn(&W) -> bool) -> bool {
        self.work
            .iter()
            .any(|(left, work)| *left == kind && same(work))
    }

    /// The current round ended: its finds go after the earlier rounds', its checks before its
    /// steps, each kind in the order found.
    fn end_round(&mut self) {
        let found = &mut self.work.make_contiguous()[self.ordered..];
        found.sort_by_key(|(kind, _)| *kind);
        self.ordered = self.work.len();
    }

    /// The oldest piece, taken out.
    fn pop(&mut self) -> Option<(Kind, W)> {
        let work = self.work.pop_front()?;
        self.ordered = self.ordered.saturating_sub(1);
        Some(work)
    }
}

/// The thread's way through its work once woken: a `round` (a pass of its own, looking at the inbox
/// again while it leaves the thread no work, and what it lets go of, `scheduler/mod.rs`), which adds
/// what it leaves to `left`, the thread's own, empty, which it keeps with its room from one waking
/// to the next; then, while work is left, rounds for as long as `inbox` holds commits, and the
/// oldest piece, which `run` runs (the module docs). `round` and `run` answer false once the process
/// is exiting, and this then answers `None`, what is left staying in `left`; else whether a piece
/// ran, `left` emptied.
pub(crate) fn work_through<S, C, W>(
    inbox: &Inbox<S, C>,
    left: &mut Left<W>,
    mut round: impl FnMut(&mut Left<W>) -> bool,
    mut run: impl FnMut(Kind, W) -> bool,
) -> Option<bool> {
    debug_assert!(
        left.is_empty(),
        "the thread's work left from an earlier waking"
    );
    if !round(left) {
        return None;
    }
    left.end_round();
    let mut ran = false;
    while !left.is_empty() {
        // A commit left to this thread meanwhile goes before the next piece of work, as does one
        // left to it during the round that took that one in.
        while inbox.holds_commits() {
            if !round(left) {
                return None;
            }
            left.end_round();
        }
        let Some((kind, work)) = left.pop() else {
            break;
        };
        if !run(kind, work) {
            return None;
        }
        ran = true;
    }
    Some(ran)
}
