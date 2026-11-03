//! The scheduler's lock with an inbox, as a pure structure: a commit leaves its batches in the
//! inbox and returns at once, as Metal's `commit` does, and the scheduler's thread makes the
//! passes, but for the one a thread about to block makes for itself ([`Inbox::pass_once`]).
//!
//! A pass takes the lock, takes in what was committed, and submits what is ready, one
//! `vkQueueSubmit` after another (about 80 us each on ANV with i915, whose execbuffer lists every
//! memory object of the device), a few batches a pass (`scheduler/mod.rs`). No commit waits for
//! one.
//!
//! - **Posted, never passed** ([`Inbox::post`]). A commit puts its commit in the inbox and
//!   returns: the scheduler's thread takes it in at its next pass. The inbox keeps commit order,
//!   and every pass takes it in before anything else: so the order sees commits in commit order,
//!   and the batches go to the `VkQueue`s, and are planned, in the order a pass picks them.
//! - **Nothing left behind.** The passer is counted while it makes a pass (`passers`). Once done
//!   and counted out, it looks at the inbox again after a sequentially consistent fence, and
//!   makes another pass for what it holds (a thread about to block wakes the scheduler's thread
//!   for it instead: [`Inbox::pass_once`]). A commit puts its own in before the same fence and its
//!   look at the count: whichever of the two fences comes first, the later look sees the other's
//!   write, so a commit is either seen by the thread's look or finds no passer. One that finds no
//!   passer marks the state under the lock, which then only a short hold has (a wake, a decode
//!   count, the thread's look for work), for its caller to wake the thread, which takes it in.
//! - **The scheduler's thread looks again until its passes leave it work** ([`Inbox::pass`]).
//!   Its passes may leave it work to do without the lock first (a submission check, a host step),
//!   which the batches behind it wait for: it stops looking again at the first pass that leaves
//!   some, and before each piece of that work, passes again for as long as the inbox holds
//!   commits ([`Inbox::holds_commits`], `scheduler/left.rs`), and looks before it sleeps. A commit
//!   left to it, while it holds the lock for a pass, so waits for that pass, the submission
//!   handlers the thread runs right after it, and the pass that takes it in; never for a wake or
//!   the thread's work.
//! - **Who runs the after-work.** The thread lets go, outside the lock, of what its passes
//!   submitted (the holds, the submission handlers) in submission order.
//! - **A thread about to block passes once** ([`Inbox::pass_once`]). A thread that is about to
//!   sleep for work it committed itself passes for it instead of waiting for a wake, where the
//!   inbox holds commits and no passer is counted: it is counted as the passer by a swap from
//!   none, so two such threads never both pass, and it looks again for nothing once counted out
//!   (it is going to sleep). What was left to it meanwhile it hands to the scheduler's thread
//!   instead, by the same fence and look its own `look_again` would make, and a wake
//!   ([`Inbox::wake`]). The scheduler's thread is never gated: its [`pass`](Inbox::pass) counts
//!   in whatever is counted, and waits for the lock while such a pass holds it, as a short hold
//!   makes it wait.
//! - **A committing thread may pass once for its own commit** ([`Inbox::post_quietly`],
//!   [`Inbox::pass_once_then`]). `CommandBuffer::commit_and_submit` posts its commit with no mark
//!   and no wake, then passes once as a thread about to block does, where no passer is counted;
//!   where one is, that passer takes the commit in (the thread's by its look again, a thread's
//!   that passes once by the wake its [`passed_over`](Inbox::passed_over) reads). It stays
//!   counted until the work its pass left it (the submission handlers, the holds, a head's
//!   submission checks and the passes after them, [`Inbox::pass_again`], which take nothing in)
//!   ran, so a commit made meanwhile is handed to it and from it to the thread as above.
//! - **A wake is handed off too** ([`Inbox::nudge`]). What a pass reads may change without a
//!   commit: an event's value rises, a batch completes (a check waits for its signal to land).
//!   Where the thread is counted, the waker leaves it a nudge and returns, as a commit does: the
//!   thread looks again for it once done, by the same fences, and each pass takes it in first.
//!   A completion turn so never waits for a pass's submissions before it finishes the batches
//!   it saw complete, whose completed handlers wait for it. A pass takes the nudge in by a
//!   sequentially consistent swap, which reads the waker's store where it takes that nudge, and
//!   so sees what the waker changed before it.

use crate::device::lock;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, TryLockError};

/// The commits the inbox holds from its making without growing: more than a frame commits (the
/// consumer's guest commits about 45 command buffers a frame, the `frame_costs` lanes 41 to 108
/// and wait for each frame's last), so a commit's push allocates only past a backlog that deep.
pub(crate) const ROOM: usize = 128;

/// A state `S` that passes work on under its lock, and the commits `C` waiting to be taken in.
pub(crate) struct Inbox<S, C> {
    state: Mutex<S>,
    /// Commits not yet taken in, in commit order. It starts with room for [`ROOM`], keeps its room,
    /// and the thread making passes grows it ahead of the commits ([`take`](Self::take)).
    commits: Mutex<Vec<C>>,
    /// The commits a pass takes in. Filled and drained by the thread making the pass, under the
    /// state's lock, so it keeps its room from pass to pass and a pass allocates no list of them
    /// in a steady state.
    taking: Mutex<Vec<C>>,
    /// `commits` holds one (read without its lock; written under it).
    posted: AtomicBool,
    /// A waker asked for a pass since the last one took the inbox in ([`Inbox::nudge`]).
    nudged: AtomicBool,
    /// The threads making a pass.
    passers: AtomicUsize,
}

impl<S, C> Inbox<S, C> {
    pub(crate) fn new(state: S) -> Inbox<S, C> {
        Inbox {
            state: Mutex::new(state),
            commits: Mutex::new(Vec::with_capacity(ROOM)),
            taking: Mutex::new(Vec::new()),
            posted: AtomicBool::new(false),
            nudged: AtomicBool::new(false),
            passers: AtomicUsize::new(0),
        }
    }

    /// The state's lock, for what makes no pass: a read, a mark for the thread, a wait on it.
    pub(crate) fn state(&self) -> &Mutex<S> {
        &self.state
    }

    /// Whether a commit waits in the inbox, not yet taken in.
    pub(crate) fn holds_commits(&self) -> bool {
        self.posted.load(Ordering::SeqCst)
    }

    /// Whether a waker asked for a pass that no pass took in yet.
    pub(crate) fn nudged(&self) -> bool {
        self.nudged.load(Ordering::SeqCst)
    }

    /// Ask for a pass: what a pass reads changed. Where a passer is counted, it looks again once
    /// done and passes for this, as for a commit, and this answers true. Else false: the caller
    /// wakes the thread that makes passes itself, under the lock, which only a short hold or a
    /// passer counted since (whose pass, made after this, reads the change) may have.
    pub(crate) fn nudge(&self) -> bool {
        self.nudged.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.passing()
    }

    /// Whether a passer is counted.
    fn passing(&self) -> bool {
        self.passers.load(Ordering::SeqCst) > 0
    }

    /// One pass for a thread about to block (the module docs), which makes no second one: where
    /// the inbox holds a commit and no passer is counted, it becomes the passer by a swap from
    /// none, `pass` takes in what the inbox holds under the state's lock, and it counts out.
    /// Whether it passed; false where it holds no commit, or where a passer was counted, whose
    /// pass takes the commits in.
    ///
    /// It never looks again: the caller is about to sleep, so what was posted or nudged during
    /// its pass goes to the thread that makes passes instead, which the caller
    /// [`wake`](Self::wake)s after the sequentially consistent fence this leaves behind
    /// ([`passed_over`](Self::passed_over) answers what is left).
    pub(crate) fn pass_once(&self, pass: impl FnOnce(&mut S, &mut Vec<C>)) -> bool {
        if !self.holds_commits() {
            return false;
        }
        if (self.passers)
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        let mut state = lock(&self.state);
        let mut taking = lock(&self.taking);
        self.take(&mut taking);
        pass(&mut state, &mut taking);
        debug_assert!(taking.is_empty(), "a pass takes in every commit");
        drop(taking);
        drop(state);
        self.passers.fetch_sub(1, Ordering::SeqCst);
        true
    }

    /// [`pass_once`](Self::pass_once), with `after` run on what `pass` answered once the state's
    /// lock is free and before the passer counts out: the work the pass left the caller (its
    /// submission handlers, the holds it let go of, a check and the passes after it,
    /// [`pass_again`](Self::pass_again)), during which a commit made finds the passer counted and
    /// is left to it. What `after` answered, where it passed.
    pub(crate) fn pass_once_then<R, T>(
        &self,
        pass: impl FnOnce(&mut S, &mut Vec<C>) -> R,
        after: impl FnOnce(R) -> T,
    ) -> Option<T> {
        if !self.holds_commits() {
            return None;
        }
        if (self.passers)
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }
        let passed = {
            let mut state = lock(&self.state);
            let mut taking = lock(&self.taking);
            self.take(&mut taking);
            let passed = pass(&mut state, &mut taking);
            debug_assert!(taking.is_empty(), "a pass takes in every commit");
            passed
        };
        let answered = after(passed);
        self.passers.fetch_sub(1, Ordering::SeqCst);
        Some(answered)
    }

    /// Another pass for a caller of [`pass_once_then`](Self::pass_once_then), from its `after`,
    /// still counted as the passer: `pass` under the state's lock over what the state holds,
    /// taking nothing in. What was posted or nudged since stays for the thread that makes passes,
    /// which the caller wakes once counted out ([`passed_over`](Self::passed_over)); every commit
    /// posted comes after every one the state holds, so the state's order stays commit order.
    pub(crate) fn pass_again<R>(&self, pass: impl FnOnce(&mut S) -> R) -> R {
        pass(&mut lock(&self.state))
    }

    /// Whether a commit or a nudge was left to the thread that makes passes, read after
    /// [`pass_once`](Self::pass_once) counted out: the same fence and look its
    /// [`look_again`](Self::look_again) would make, for a caller that sleeps instead of passing
    /// again. One posted or nudged during that pass put its write in before its own fence and
    /// its look at the count, which found a passer: whichever fence comes first, the later look
    /// sees the other's write, so it is seen here or its own caller found no passer and marked
    /// the state itself.
    pub(crate) fn passed_over(&self) -> bool {
        fence(Ordering::SeqCst);
        self.posted.load(Ordering::SeqCst) || self.nudged.load(Ordering::SeqCst)
    }

    /// Wake for a pass: [`nudge`](Self::nudge), and where that finds no passer, `mark` the state
    /// under the lock ([`mark`](Self::mark)). Whether it marked the state, for the caller to wake
    /// the thread that makes passes: the waker never waits for a pass's submissions.
    pub(crate) fn wake(&self, mark: impl FnOnce(&mut S)) -> bool {
        if self.nudge() {
            return false;
        }
        self.mark(mark)
    }

    /// Post `commit`, after every commit posted before it, for the caller's own pass to take in
    /// ([`pass_once_then`](Self::pass_once_then)), with no mark and no wake: where a passer is
    /// counted then, its pass takes the commit in, or the thread it wakes once counted out
    /// ([`passed_over`](Self::passed_over)). The same write and fence as [`post`](Self::post).
    pub(crate) fn post_quietly(&self, commit: C) {
        {
            let mut commits = lock(&self.commits);
            commits.push(commit);
            self.posted.store(true, Ordering::SeqCst);
        }
        fence(Ordering::SeqCst);
    }

    /// Post `commit`, after every commit posted before it, for the thread that makes passes to
    /// take in; where no passer is counted, `mark` the state under the lock ([`mark`](Self::mark)).
    /// Whether it marked the state, for the caller to wake the thread: a commit never makes a
    /// pass, nor waits for one (the module docs).
    pub(crate) fn post(&self, commit: C, mark: impl FnOnce(&mut S)) -> bool {
        {
            let mut commits = lock(&self.commits);
            commits.push(commit);
            self.posted.store(true, Ordering::SeqCst);
        }
        fence(Ordering::SeqCst);
        if self.passing() {
            return false;
        }
        self.mark(mark)
    }

    /// After a write and a sequentially consistent fence that found no passer counted: `mark` the
    /// state under the lock, which then only a short hold or a passer counted since (whose pass,
    /// made after the write, reads it) may have; true where it marked it.
    fn mark(&self, mark: impl FnOnce(&mut S)) -> bool {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            // A pass counted since: it takes the write in.
            Err(TryLockError::WouldBlock) if self.passing() => return false,
            Err(TryLockError::WouldBlock) => lock(&self.state),
        };
        mark(&mut state);
        true
    }

    /// A pass by the scheduler's thread, never gated (the module docs), `pass` taking in what the inbox
    /// holds and answering whether to look again; then look at the inbox again
    /// ([`look_again`](Self::look_again)) while each pass answers so. Where one answers no, the
    /// inbox may hold commits left to this thread, which it looks for itself
    /// ([`holds_commits`](Self::holds_commits)) and passes for before it sleeps (the module docs).
    /// Each pass takes every commit out of the list it is given.
    pub(crate) fn pass(&self, mut pass: impl FnMut(&mut S, &mut Vec<C>) -> bool) {
        self.passers.fetch_add(1, Ordering::SeqCst);
        let mut state = lock(&self.state);
        let mut taking = lock(&self.taking);
        self.take(&mut taking);
        let again = pass(&mut state, &mut taking);
        debug_assert!(taking.is_empty(), "a pass takes in every commit");
        drop(taking);
        drop(state);
        self.passers.fetch_sub(1, Ordering::SeqCst);
        if again {
            self.look_again(&mut pass);
        }
    }

    /// After the passer is counted out: while the inbox holds a commit or a nudge, take the lock
    /// and pass for it (waiting only for a holder that makes no pass), until a pass answers not
    /// to look again.
    fn look_again(&self, pass: &mut impl FnMut(&mut S, &mut Vec<C>) -> bool) {
        loop {
            fence(Ordering::SeqCst);
            let asked = self.posted.load(Ordering::SeqCst) || self.nudged.load(Ordering::SeqCst);
            if !asked {
                return;
            }
            self.passers.fetch_add(1, Ordering::SeqCst);
            let mut state = lock(&self.state);
            let mut taking = lock(&self.taking);
            self.take(&mut taking);
            let again = pass(&mut state, &mut taking);
            debug_assert!(taking.is_empty(), "a pass takes in every commit");
            drop(taking);
            drop(state);
            self.passers.fetch_sub(1, Ordering::SeqCst);
            if !again {
                return;
            }
        }
    }

    /// The commits in the inbox, in commit order, taken out onto `taking`, and its nudge with
    /// them; called under the state's lock, before the pass reads anything. The nudge is taken by
    /// a swap: where it reads a waker's store, it synchronizes with it, so the pass reads what the
    /// waker changed before its nudge; where it reads none, any later store of one stays for a
    /// later look.
    ///
    /// The inbox is left room for twice the commits taken in, made here, on the thread making
    /// passes: a commit's push then grows it, an allocation on the committing thread, only where
    /// more commits wait than [`ROOM`] and twice as many as a pass ever took in.
    fn take(&self, taking: &mut Vec<C>) {
        self.nudged.swap(false, Ordering::SeqCst);
        if !self.posted.load(Ordering::SeqCst) {
            return;
        }
        let mut commits = lock(&self.commits);
        self.posted.store(false, Ordering::SeqCst);
        taking.append(&mut commits);
        commits.reserve(2 * taking.len());
    }

    /// How many commits the inbox holds without growing.
    #[cfg(test)]
    pub(crate) fn room(&self) -> usize {
        lock(&self.commits).capacity()
    }
}
