//! The process's budget of pipeline makings in flight (the owner's HITCH rules 1-3): every
//! `vkCreate*Pipelines` call the runtime makes, on whatever thread, holds one of the budget's turns
//! ([`turns_for`] the host by default) while it runs, so the driver's compiler (on Apple's GPUs
//! Metal's compiler service, which every device of the process shares) never has more than that
//! many of the process's compiles at once.
//!
//! - **Every making counts.** A pool's thread takes a turn before it takes its next piece and
//!   holds it through the piece ([`pool_turn`]); a making on any other thread (a program's thread
//!   making a draw's, a dispatch's or a creation's own pipeline, the pool's caller where it starts
//!   no thread) takes one around its driver call ([`program_turn`], from
//!   `ShaderCache::make_pipelines`). A thread holding a turn takes no second one.
//! - **Ranks, then the order asked** (the consumer's cold-boot storm). Each asker has a rank: its
//!   [`Urgency`], then a ticket from one process-wide counter, drawn when the making was asked for
//!   (a pool's piece where it was handed to the pool). A free turn goes to the asker of the lowest
//!   rank of those that may take it:
//!   1. a making in a command buffer's host step, which holds its device's scheduler (every
//!      command buffer's submission) while it waits ([`Urgency::HostStep`]);
//!   2. a making the first batch held on a command queue waits for: every batch committed after
//!      it there waits behind it, and the program's thread, once the queue's command buffers are
//!      all held, in `commandBuffer` ([`Urgency::Head`]). The scheduler raises a batch's makings
//!      to it once the batch is first in its queue's line (`scheduler/mod.rs`, `hasten_heads`):
//!      Metal makes a pipeline state at its creation, so no submission of Metal's waits for a
//!      making, and the one that holds a queue goes before the rest;
//!   3. a making a committed command buffer's batch waits for: the GPU's queue, and the
//!      program's watchdog on it, wait behind it ([`Urgency::Committed`]);
//!   4. a making a program's thread waits for: its own draw's, dispatch's or helper's, a state's
//!      creation's predicted key on the creating thread, a caller blocked on a making in flight
//!      ([`Urgency::Program`]);
//!   5. a making nothing waits for yet: a draw's piece before its command buffer commits
//!      ([`Urgency::Ahead`]). These take idle turns, but hold at most the budget's `ahead`
//!      turns (by default all but one) at once, so a waited making that comes
//!      finds a turn free: the turns past the `ahead` ones are only for makings a thread, a
//!      commit or a host step waits for, which take them whatever waits ahead of them;
//!   6. a warm pass's making (`warm/pass.rs`), which nothing in the process asked for: its own
//!      thread's and the pieces it hands the pool ([`Urgency::Warm`]). These take the `ahead`
//!      turns alone, after every other making, and never age.
//!
//!   A piece's rank rises where something comes to wait for it (`Place::hasten`; a wait or a commit before the piece was handed to the pool is applied
//!   as it is handed), never falls. Within a rank the earlier ticket goes first. No turn is
//!   taken back (no preemption).
//! - **Aging** (the consumer's reviews). A making that has waited [`AGE`] is ordered as a committed
//!   command buffer's (rank 3) by its ticket, so a steady stream of other makings cannot starve a
//!   draw's piece of the ahead turns; a warm pass's making never ages, so the program's makings
//!   always go before it. Aging moves a making nothing waits for up the order only: it
//!   still takes only an ahead turn, so it never takes the turn a waited making finds free, and a
//!   host step's making, or a queue's first held batch's, goes before it always.
//! - **Nothing seen is decided here.** The budget orders driver calls in time and decides nothing
//!   else: which pipelines are made, their keys, memo entries, creation log numbers, notices and
//!   everything the program sees are decided by the callers as before; a turn only delays a call.
//! - **No making waits on another while it holds a turn.** A piece makes its pipeline and returns,
//!   and a program's thread holds its turn only around the driver call: so the turns always come
//!   back, and no set of turn holders can wait on each other. A wait for a making in flight
//!   (`making.rs`, `Turn::wait`: a memo's `get_or_make` or `Pending::wait`) asserts in builds
//!   with debug assertions that its thread holds no turn ([`holds_turn`]). The makings that a
//!   making's own path reaches (a draw's mesh pipelines, `Making::get_or_make` on the draw's
//!   path) are asked for on the draw's thread before any turn, never inside a piece's. A piece
//!   that waits for its twin's making in the air (`flight.rs`), and a caller that waits for
//!   another's making of a pipeline library it links (`libraries.rs`, `Library::made`), gives
//!   its thread's turn back for the wait and takes one again after it at the rank it was given
//!   at ([`without_turn`]).
//! - **A pool's turn is its next piece's.** A pool's thread is given its turn together with the
//!   piece the turn was ranked by, taken from the pool under the budget's lock ([`pool_turn`]), so
//!   no piece runs on a turn another piece's rank won; and a thread asking counts as busy, so a
//!   piece handed meanwhile starts a thread of its own (`pool::serve`), up to the pool's threads.
//! - **A helper's turn before its lock.** A helper pipeline made on its first use takes its turn
//!   before its helper's lock ([`helper_turn`]), so no thread holds that lock waiting for a turn,
//!   and the helper's other uses, made already, never wait behind the budget.
//! - **Not counted:** the device's priming (`device/prime.rs`), which makes three pipelines of
//!   trivial shaders on the creating thread before the device reaches the program, takes no turn:
//!   a device's creation never waits behind the process's compiles.
//! - **Measured, not guessed**. [`set_making_budget_for_measurement`] sets the turns and the ahead
//!   turns once, before the process's first device, for the storm benchmark (`corpus-draw-ab
//!   storm`); no environment variable reaches the budget.

use super::pool::{PoolQueue, Work};
use crate::device::lock;
use crate::exit_guard;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

/// The most pipeline makings in flight in a process on a host of `threads` hardware threads
/// (`available_parallelism`), by default. The makings nothing waits for hold every turn but one
/// at most (the module docs), which a waited making that comes then finds free (one turn ahead
/// alone compiled a creation's keys one at a time with three turns idle).
///
/// - **On Apple's platforms, four** ([`APPLE_TURNS`]): MoltenVK's compiles go to Metal's compiler
///   service, which every device of the process shares and which schedules its own work. On the
///   M3 Max, eight distinct compiles of a large fragment at once all finished in 1.5 times one's
///   time; on the consumer's Mac under a guest's cold boot (E26, C15, C14 at
///   7486754bb) a large fragment's compiles beside their siblings took 28-31 s each where one took
///   5-8 s.
/// - **Elsewhere, twelve from sixteen threads up ([`WIDE_HOST`]), else half the host's threads,
///   never fewer than four**: the driver compiles each pipeline on the thread that asks. On the RTX 3090's 16-thread host, nine makings of the
///   consumer's Xgc fragment inside one creation took 157 s one at a time, 98 s two, 63 s four
///   and 44.5 s eight at a time (each making 17, 20, 22.5 and 26 s), for 45% more CPU time in all
///   and about 0.4 GB of the driver's memory per compile in flight. On the RTX 5080 laptop's 24
///   threads, ten such makings took 41 s eight at a time and 27-28 s twelve, sixteen or twenty-four
///   at a time (all ten at once), for 13% more CPU time and about 0.5 GB per compile in flight. On
///   the RTX 3090's 16 threads, the same ten took 56-59 s eight at a time and 37-40 s twelve,
///   sixteen or twenty-four at a time, for 17% more CPU time. Hosts below sixteen threads are not
///   measured above half their threads, and keep it. Never fewer than the four every host had
///   before.
pub(crate) fn turns_for(threads: usize) -> usize {
    if cfg!(target_vendor = "apple") {
        APPLE_TURNS
    } else if threads >= WIDE_HOST {
        MOST_TURNS
    } else {
        (threads / 2).clamp(APPLE_TURNS, MOST_TURNS)
    }
}

/// The turns on Apple's platforms, and the fewest anywhere ([`turns_for`]).
pub(crate) const APPLE_TURNS: usize = 4;

/// The most turns any host gets ([`turns_for`]): twelve, measured on a 24-thread host and a
/// 16-thread one.
pub(crate) const MOST_TURNS: usize = 12;

/// The fewest hardware threads of a host that gets [`MOST_TURNS`]: sixteen, the smallest host
/// measured there.
pub(crate) const WIDE_HOST: usize = 16;

/// How long a making waits before it is ordered as a committed command buffer's, still taking
/// only an ahead turn where nothing waits for it (the module docs).
pub(crate) const AGE: Duration = Duration::from_secs(4);

/// The process's budget, which every device's makings share: made at its first use, from
/// [`set_making_budget_for_measurement`]'s numbers where it was called first.
static PROCESS: OnceLock<Arc<Budget>> = OnceLock::new();

/// The counter tickets are drawn from (a pool's pieces' numbers too): one order for the process.
static TICKETS: AtomicU64 = AtomicU64::new(0);

/// The turn a thread holds: its budget, whether it is an ahead turn, and the rank it was given at
/// (its asker's urgency and ticket, unaged), at which a lent turn is taken again
/// ([`without_turn`]).
struct Held {
    budget: Arc<Budget>,
    ahead: bool,
    rank: Rank,
}

thread_local! {
    /// The turn this thread holds now, if any.
    static HELD: RefCell<Option<Held>> = const { RefCell::new(None) };
    /// How long this thread's making waited for its turn, or its piece in its pool, since
    /// [`take_waited`] last read it (the creation log, `creation_log.rs`).
    static WAITED: Cell<Duration> = const { Cell::new(Duration::ZERO) };
}

/// How long this thread's making waited before its driver call since this was last called: for
/// a turn on a program's thread ([`program_turn`]), in its pool and for its turn on a pool's
/// thread (`pool::serve`, [`waited_in_pool`]). Read and reset.
pub(crate) fn take_waited() -> Duration {
    WAITED.with(|waited| waited.replace(Duration::ZERO))
}

/// The piece this pool's thread runs next waited `waited` since it was handed to the pool.
pub(crate) fn waited_in_pool(waited: Duration) {
    WAITED.with(|cell| cell.set(waited));
}

/// Whether this thread holds a turn now: a making's waits for another making assert it does
/// not (`making.rs`, the module docs' "no making waits on another while it holds a turn").
pub(crate) fn holds_turn() -> bool {
    HELD.with(|held| held.borrow().is_some())
}

/// A new ticket: later than every ticket drawn before it.
pub(crate) fn ticket() -> u64 {
    TICKETS.fetch_add(1, Ordering::Relaxed)
}

/// The process's budget: [`turns_for`] the host, all but one of them ahead.
pub(crate) fn process() -> &'static Arc<Budget> {
    PROCESS.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
        let turns = turns_for(threads);
        Arc::new(Budget::new(turns, turns - 1))
    })
}

/// [`set_making_budget_for_measurement`] came too late: the process's budget was in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MakingBudgetStarted;

impl std::fmt::Display for MakingBudgetStarted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the process's pipeline making budget is in use already")
    }
}

impl std::error::Error for MakingBudgetStarted {}

/// Set the process's budget of pipeline makings in flight, for a measurement of the runtime's
/// compile concurrency: at most `turns` `vkCreate*Pipelines` calls at once in the process, at
/// most `ahead` of them makings nothing waits for yet (each at least one). Not part of the API;
/// Metal has no counterpart (its compiler service schedules its own compiles).
///
/// Call it once, before the process's first device: it fails, changing nothing, once the
/// budget is in use (a device was made, or a pipeline was) or was set before.
pub fn set_making_budget_for_measurement(
    turns: usize,
    ahead: usize,
) -> Result<(), MakingBudgetStarted> {
    let turns = turns.max(1);
    PROCESS
        .set(Arc::new(Budget::new(turns, ahead.clamp(1, turns))))
        .map_err(|_| MakingBudgetStarted)
}

/// A budget of turns (the module docs): the process's ([`process`]), or a test's own.
pub(crate) struct Budget {
    state: Mutex<State>,
    freed: Condvar,
    turns: usize,
    /// The turns makings nothing waits for may hold at once ([`Urgency::Ahead`]).
    ahead: usize,
    /// How long a making waits before it ranks as a committed one ([`AGE`]).
    age: Duration,
}

struct State {
    /// The turns held now, and of them those held by makings nothing waits for.
    held: usize,
    held_ahead: usize,
    /// Those waiting for a turn, each by an id of its own.
    askers: Vec<(u64, Asker)>,
    /// The id of the next asker.
    next: u64,
}

/// One waiting for a turn.
enum Asker {
    /// A making on a thread that asks for itself, at this rank, asked for with this ticket,
    /// waiting since then.
    Thread(Urgency, u64, Instant),
    /// A pool's thread about to take the pool's next piece: ranked as that piece is.
    Pool(Weak<PoolQueue>),
}

/// How soon a making is needed (the module docs): earlier goes first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Urgency {
    /// A command buffer's host step makes it, holding its device's scheduler.
    HostStep,
    /// The first batch held on a command queue waits for it: every batch after it there waits
    /// behind it.
    Head,
    /// A committed command buffer's batch waits for it.
    Committed,
    /// A program's thread waits for it.
    Program,
    /// Nothing waits for it yet: a draw's piece before its commit.
    Ahead,
    /// A warm pass makes it (`warm/pass.rs`): nothing in the process asked for it. It takes an
    /// ahead turn after every other making, and never ages ([`aged`]).
    Warm,
}

impl Urgency {
    /// Its code in an [`Asked`].
    fn code(self) -> u8 {
        match self {
            Urgency::HostStep => 0,
            Urgency::Head => 1,
            Urgency::Committed => 2,
            Urgency::Program => 3,
            Urgency::Ahead => 4,
            Urgency::Warm => 5,
        }
    }

    /// The urgency of `code`; `None` for [`Asked::NONE`].
    fn from_code(code: u8) -> Option<Urgency> {
        match code {
            0 => Some(Urgency::HostStep),
            1 => Some(Urgency::Head),
            2 => Some(Urgency::Committed),
            3 => Some(Urgency::Program),
            4 => Some(Urgency::Ahead),
            5 => Some(Urgency::Warm),
            _ => None,
        }
    }
}

/// What has been asked of one making so far (its memo turn's, `making.rs`): the most urgent rank
/// something that came to wait for it asked. A pool's piece rises through its place, which reads
/// this as the piece is handed there.
pub(crate) struct Asked {
    /// The rank asked, as [`Urgency::code`]; [`Asked::NONE`] where nothing asked yet.
    urgency: AtomicU8,
}

impl Asked {
    const NONE: u8 = u8::MAX;

    /// Nothing asked yet.
    pub(crate) fn new() -> Asked {
        Asked {
            urgency: AtomicU8::new(Asked::NONE),
        }
    }

    /// Something came to wait for the making, as `urgency` says: never a fall.
    pub(crate) fn raise(&self, urgency: Urgency) {
        self.urgency.fetch_min(urgency.code(), Ordering::SeqCst);
    }

    /// The most urgent rank asked so far.
    pub(crate) fn get(&self) -> Option<Urgency> {
        Urgency::from_code(self.urgency.load(Ordering::SeqCst))
    }
}

/// An asker's rank: its urgency (aged, [`aged`]), then its ticket. Lower goes first.
pub(crate) type Rank = (Urgency, u64);

/// What an asker may take: its rank, and whether nothing waits for its making (so it takes an
/// ahead turn), aged or not.
pub(crate) type Candidate = (Rank, bool);

impl Asker {
    /// What it may take now, where an ahead turn is free or not: `None` where it is a pool's
    /// thread whose pool has nothing left (it leaves); `Some(None)` where it may take nothing
    /// now (its making is one nothing waits for, or all its pool's are, and no ahead turn is
    /// free).
    fn rank(&self, age: Duration, ahead_free: bool) -> Option<Option<Candidate>> {
        match self {
            Asker::Thread(urgency, ticket, since) => {
                let ahead = *urgency >= Urgency::Ahead;
                Some(
                    (ahead_free || !ahead).then(|| ((aged(*urgency, *since, age), *ticket), ahead)),
                )
            }
            Asker::Pool(queue) => queue.upgrade()?.next_rank(ahead_free),
        }
    }
}

/// The order of a making of `urgency` waiting since `since`: a committed one's once it waited
/// `age`, where it ranked lower, but a warm pass's, which keeps its rank (the module docs). Which
/// turns it may take follow its own `urgency`, never this.
pub(crate) fn aged(urgency: Urgency, since: Instant, age: Duration) -> Urgency {
    let ages = urgency > Urgency::Committed && urgency != Urgency::Warm;
    match ages && since.elapsed() >= age {
        true => Urgency::Committed,
        false => urgency,
    }
}

/// A turn, given back when dropped.
pub(crate) struct Turn {
    budget: Arc<Budget>,
    ahead: bool,
}

impl Drop for Turn {
    fn drop(&mut self) {
        HELD.with(|held| held.borrow_mut().take());
        self.budget.give_back(self.ahead);
    }
}

impl Budget {
    /// A budget of `turns` turns, at most `ahead` of them held by makings nothing waits for.
    pub(crate) fn new(turns: usize, ahead: usize) -> Budget {
        Budget::aging(turns, ahead, AGE)
    }

    /// [`Budget::new`]'s, its makings aging after `age`.
    pub(crate) fn aging(turns: usize, ahead: usize, age: Duration) -> Budget {
        Budget {
            state: Mutex::new(State {
                held: 0,
                held_ahead: 0,
                askers: Vec::new(),
                next: 0,
            }),
            freed: Condvar::new(),
            turns,
            ahead,
            age,
        }
    }

    /// Its turns: the most makings in flight, and the most threads a pool within it runs
    /// (`pool.rs`).
    pub(crate) fn turns(&self) -> usize {
        self.turns
    }

    /// How long a making waits before it ranks as a committed one.
    pub(crate) fn age(&self) -> Duration {
        self.age
    }

    /// A turn given back, an ahead turn where `ahead`: the askers rank themselves again.
    fn give_back(&self, ahead: bool) {
        let mut state = lock(&self.state);
        state.held -= 1;
        if ahead {
            state.held_ahead -= 1;
        }
        drop(state);
        self.freed.notify_all();
    }

    /// Wake the askers to rank themselves again: a pool's piece rose in rank (hastened).
    pub(crate) fn reranked(&self) {
        drop(lock(&self.state));
        self.freed.notify_all();
    }

    /// Wait for a turn as `asker`, with the piece it was ranked by for a pool's thread; `None`
    /// where it is a pool's thread whose pool has nothing left to take.
    fn take(self: &Arc<Self>, asker: Asker) -> Option<(Turn, Option<Work>)> {
        let (ahead, piece) = self.take_held(asker)?;
        let turn = Turn {
            budget: Arc::clone(self),
            ahead,
        };
        Some((turn, piece))
    }

    /// [`take`](Self::take)'s turn, counted held and noted as this thread's ([`Held`]) but
    /// given back by no [`Turn`] of its own: whether it is an ahead turn, and the piece.
    fn take_held(self: &Arc<Self>, asker: Asker) -> Option<(bool, Option<Work>)> {
        let mut state = lock(&self.state);
        let id = state.next;
        state.next += 1;
        state.askers.push((id, asker));
        loop {
            // A making nothing waits for, aged or not, takes only an ahead turn: the others are
            // for makings something waits for (the consumer's ruling).
            let ahead_free = state.held_ahead < self.ahead;
            let ranks: Vec<(u64, Option<Option<Candidate>>)> = state
                .askers
                .iter()
                .map(|(id, asker)| (*id, asker.rank(self.age, ahead_free)))
                .collect();
            let Some(mine) = ranks
                .iter()
                .find(|(at, _)| *at == id)
                .and_then(|(_, rank)| *rank)
            else {
                state.askers.retain(|(at, _)| *at != id);
                drop(state);
                // Another asker may now be first.
                self.freed.notify_all();
                return None;
            };
            // The first of those that may take a turn now: one that may not (nothing waits for
            // it, no ahead turn free) holds back no one behind it.
            let first = ranks
                .iter()
                .filter_map(|(at, rank)| rank.flatten().map(|(rank, _)| (rank, *at)))
                .min()
                .is_some_and(|(_, at)| at == id);
            if let (true, true, Some((_, ahead))) = (first, state.held < self.turns, mine) {
                let at = state
                    .askers
                    .iter()
                    .position(|(at, _)| *at == id)
                    .expect("an asker waits until it leaves");
                let (_, asker) = state.askers.remove(at);
                // A pool's thread takes the piece its rank was, under this lock (taken before the
                // pool's, never under it), so no other piece runs on the turn: one it may take
                // as `ahead_free` says, whose own `ahead` the turn is counted by.
                let (piece, rank, ahead) = match asker {
                    Asker::Thread(urgency, ticket, _) => (None, (urgency, ticket), ahead),
                    Asker::Pool(queue) => {
                        match queue.upgrade().and_then(|q| q.next_piece(ahead_free)) {
                            Some((work, (rank, ahead))) => (Some(work), rank, ahead),
                            None => {
                                drop(state);
                                self.freed.notify_all();
                                return None;
                            }
                        }
                    }
                };
                state.held += 1;
                if ahead {
                    state.held_ahead += 1;
                }
                drop(state);
                let budget = Arc::clone(self);
                HELD.with(|held| {
                    *held.borrow_mut() = Some(Held {
                        budget,
                        ahead,
                        rank,
                    })
                });
                // The next asker may have a turn too.
                self.freed.notify_all();
                return Some((ahead, piece));
            }
            // Woken by a turn given back, a new asker or a rise in rank; and at least four times
            // an age, so a making that aged meanwhile is ranked again.
            state = self
                .freed
                .wait_timeout(state, self.age / 4)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// The turns held now and the askers waiting (the tests).
    #[cfg(test)]
    pub(crate) fn census(&self) -> (usize, usize) {
        let state = lock(&self.state);
        (state.held, state.askers.len())
    }
}

/// A turn for a making this thread waits for, asked for now (the module docs): first in a host
/// step; `None` where this thread
/// holds one already (a pool's piece, a helper's making).
pub(crate) fn program_turn() -> Option<Turn> {
    if holds_turn() {
        return None;
    }
    thread_turn_in(process(), ticket())
}

/// A turn of `budget` for this thread's making with `ticket`, at the thread's rank now: a host
/// step's, a warm pass's, else the program's.
fn thread_turn_in(budget: &Arc<Budget>, ticket: u64) -> Option<Turn> {
    let urgency = if exit_guard::in_host_step() {
        Urgency::HostStep
    } else if crate::warm::warming() {
        Urgency::Warm
    } else {
        Urgency::Program
    };
    let asked = Instant::now();
    let (turn, _) = budget.take(Asker::Thread(urgency, ticket, asked))?;
    let waited = asked.elapsed();
    WAITED.with(|cell| cell.set(cell.get() + waited));
    Some(turn)
}

/// A turn for a thread of `queue`'s pool, from `budget`, with the pool's piece it was ranked by,
/// taken from the pool; `None` where the pool has nothing left to take.
pub(crate) fn pool_turn(budget: &Arc<Budget>, queue: &Arc<PoolQueue>) -> Option<(Turn, Work)> {
    match budget.take(Asker::Pool(Arc::downgrade(queue)))? {
        (turn, Some(work)) => Some((turn, work)),
        (_, None) => None,
    }
}

/// A turn for a helper pipeline's making on its first use, taken before the helper's lock
/// `objects` is held for the making (the module docs): `None` where `made` finds the helper made
/// already, or this thread holds a turn. The caller then locks and makes as before, holding the
/// turn, so its `make_pipelines` takes no second one; one made meanwhile by another thread it
/// finds under the lock.
pub(crate) fn helper_turn<M>(
    objects: &Mutex<M>,
    made: impl FnOnce(&mut M) -> bool,
) -> Option<Turn> {
    if holds_turn() || made(&mut lock(objects)) {
        return None;
    }
    program_turn()
}

/// Run `wait` holding no turn (the module docs' "no making waits on another while it holds a
/// turn"): where this thread holds one (a pool's piece), it is given back for the wait and taken
/// again after it, by an unwinding wait too, as an asker of the rank it was given at, so it is an
/// ahead turn again where it was one; the holder of the thread's [`Turn`] gives that back at its
/// end as before. Its wait counts in the making's wait ([`take_waited`]).
pub(crate) fn without_turn<T>(wait: impl FnOnce() -> T) -> T {
    let Some(held) = HELD.with(|held| held.borrow_mut().take()) else {
        return wait();
    };
    held.budget.give_back(held.ahead);
    let again = Again(held);
    let out = wait();
    drop(again);
    out
}

/// A turn lent by [`without_turn`], taken again when dropped.
struct Again(Held);

impl Drop for Again {
    fn drop(&mut self) {
        let (urgency, ticket) = self.0.rank;
        let asked = Instant::now();
        // An asker of the turn's own rank: one nothing waits for takes an ahead turn, as the
        // turn lent was (its urgency `Ahead`), and any other a turn of the rest.
        let _ = self
            .0
            .budget
            .take_held(Asker::Thread(urgency, ticket, asked));
        let waited = asked.elapsed();
        WAITED.with(|cell| cell.set(cell.get() + waited));
    }
}

/// A turn for this thread's making with `ticket`, from `budget` (the tests).
#[cfg(test)]
pub(crate) fn waited_turn(budget: &Arc<Budget>, ticket: u64) -> Turn {
    thread_turn_in(budget, ticket).expect("a thread's asker always gets a turn")
}

#[cfg(test)]
mod tests {
    use super::{
        helper_turn, holds_turn, program_turn, ticket, turns_for, waited_turn, without_turn,
        Budget, Urgency, APPLE_TURNS, MOST_TURNS,
    };
    use crate::pipeline::pool::Pool;
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::{Duration, Instant};

    /// The turns of the budgets these tests make: four, all but one ahead, as Apple's platforms
    /// have them.
    const TURNS: usize = APPLE_TURNS;
    const AHEAD: usize = TURNS - 1;

    /// A host's turns: four on Apple's platforms; elsewhere twelve from sixteen threads up, and
    /// half its threads below, never fewer than four.
    #[test]
    fn a_host_s_turns_follow_its_threads() {
        let turns: Vec<usize> = [0, 1, 4, 8, 10, 12, 14, 15, 16, 17, 24, 25, 64]
            .into_iter()
            .map(turns_for)
            .collect();
        if cfg!(target_vendor = "apple") {
            assert_eq!(turns, vec![APPLE_TURNS; 13]);
        } else {
            assert_eq!(
                turns,
                vec![4, 4, 4, 4, 5, 6, 7, 7, 12, 12, 12, 12, MOST_TURNS]
            );
        }
    }

    /// Wait until `done` holds, at most a minute (a guard against a hang).
    fn until(done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A piece that notes `name` in `order` when it runs.
    fn noting(
        order: &Arc<Mutex<Vec<&'static str>>>,
        name: &'static str,
    ) -> crate::pipeline::pool::Work {
        let order = Arc::clone(order);
        Box::new(move || order.lock().unwrap().push(name))
    }

    /// Ask for a turn of `budget` on a thread of its own, as a program's thread (in a host step
    /// where `step`), and note `name` in `order` once given it.
    fn ask(
        budget: &Arc<Budget>,
        order: &Arc<Mutex<Vec<&'static str>>>,
        (name, step): (&'static str, bool),
    ) -> std::thread::JoinHandle<()> {
        let (budget, order) = (Arc::clone(budget), Arc::clone(order));
        let asked = ticket();
        crate::exit_guard::spawn(std::thread::Builder::new(), move || {
            let take = || waited_turn(&budget, asked);
            let _turn = match step {
                true => crate::exit_guard::host_step(take),
                false => take(),
            };
            order.lock().unwrap().push(name);
        })
        .unwrap()
    }

    /// A pool's piece that waits holding no turn (for its twin's making in the air, `flight.rs`)
    /// gives its thread's back for the wait: a making asked for meanwhile takes it, though the
    /// budget has that one turn. The piece takes one again after its wait, at its rank, and every
    /// turn is given back once it ends.
    #[test]
    fn a_piece_waiting_without_its_turn_lends_it() {
        let budget = Arc::new(Budget::new(1, 1));
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        let (lent, waiting) = mpsc::channel();
        let (go, wait) = mpsc::channel::<()>();
        let noted = Arc::clone(&order);
        pool.run(Box::new(move || {
            assert!(holds_turn());
            without_turn(|| {
                assert!(!holds_turn());
                lent.send(()).unwrap();
                wait.recv().unwrap();
            });
            assert!(holds_turn());
            noted.lock().unwrap().push("piece");
        }));
        waiting.recv().unwrap();
        until(|| budget.census() == (0, 0));
        let waiter = ask(&budget, &order, ("waited", false));
        waiter.join().unwrap();
        go.send(()).unwrap();
        until(|| order.lock().unwrap().len() == 2);
        assert_eq!(*order.lock().unwrap(), ["waited", "piece"]);
        until(|| budget.census() == (0, 0));
    }

    /// A free turn goes to the making something waits for before one nothing waits for, though
    /// that one was asked for first; then the pool's piece runs.
    #[test]
    fn a_free_turn_goes_to_a_waited_making_before_one_nothing_waits_for() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        pool.run(noting(&order, "pool"));
        let waiter = ask(&budget, &order, ("waited", false));
        until(|| budget.census() == (1, 2));
        drop(held);
        waiter.join().unwrap();
        until(|| order.lock().unwrap().len() == 2);
        assert_eq!(*order.lock().unwrap(), ["waited", "pool"]);
        until(|| budget.census() == (0, 0));
    }

    /// Two devices' pools share one budget: a piece a committed command buffer waits for in one
    /// goes before a program's making asked for earlier and before every piece of the other pool
    /// nothing waits for, however many; those then run in their order.
    #[test]
    fn a_committed_piece_goes_first_whichever_pool_it_waits_in() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let (one, other) = (
            Pool::within(Arc::clone(&budget)),
            Pool::within(Arc::clone(&budget)),
        );
        let order = Arc::new(Mutex::new(Vec::new()));
        let waiter = ask(&budget, &order, ("waited", false));
        for name in ["a1", "a2", "a3"] {
            one.run(noting(&order, name));
        }
        let b1 = other.run(noting(&order, "b1"));
        // The waiter, the first pool's three threads and the other's one ask.
        until(|| budget.census() == (1, 5));
        // Hastened by a commit while its thread asks: it ranks before the program's making.
        b1.hasten(Urgency::Committed);
        drop(held);
        waiter.join().unwrap();
        until(|| order.lock().unwrap().len() == 5);
        assert_eq!(*order.lock().unwrap(), ["b1", "waited", "a1", "a2", "a3"]);
        until(|| budget.census() == (0, 0));
    }

    /// A thread holding a turn takes no second one: a pool's piece makes inside its thread's.
    #[test]
    fn a_thread_holding_a_turn_takes_no_second() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        assert!(program_turn().is_none());
        drop(held);
        assert_eq!(budget.census(), (0, 0));
    }

    /// Makings nothing waits for hold at most the ahead turns: of three turns with one ahead,
    /// two pieces of a pool hold one and wait for none of the others, which makings something
    /// waits for then take at once.
    #[test]
    fn makings_nothing_waits_for_hold_at_most_the_ahead_turns() {
        let budget = Arc::new(Budget::new(3, 1));
        let pool = Pool::within(Arc::clone(&budget));
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let (started, ran) = mpsc::channel();
        for name in ["a1", "a2"] {
            let (gate, started) = (Arc::clone(&gate), started.clone());
            pool.run(Box::new(move || {
                started.send(name).unwrap();
                gate.lock().unwrap().recv().unwrap();
            }));
        }
        assert_eq!(ran.recv().unwrap(), "a1");
        // The second piece's thread asks while the first holds the one ahead turn.
        until(|| budget.census() == (1, 1));
        let waited: Vec<_> = (0..2).map(|_| waited_turn(&budget, ticket())).collect();
        assert_eq!(budget.census(), (3, 1), "taken at once");
        drop(waited);
        release.send(()).unwrap();
        assert_eq!(ran.recv().unwrap(), "a2");
        release.send(()).unwrap();
        until(|| budget.census() == (0, 0));
    }

    /// The ranks go first, whatever the order asked (none aged): a host step's making, then a
    /// queue's first held batch's, then a committed command buffer's, then a program's, then one
    /// nothing waits for, each asked for after the one it goes before.
    #[test]
    fn the_ranks_go_host_step_head_committed_program_ahead() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let order = Arc::new(Mutex::new(Vec::new()));
        // A draw's piece before its commit, on a pool of its own.
        let ahead = Pool::within(Arc::clone(&budget));
        ahead.run(noting(&order, "ahead"));
        until(|| budget.census() == (1, 1));
        let program = ask(&budget, &order, ("program", false));
        until(|| budget.census() == (1, 2));
        let pool = Pool::within(Arc::clone(&budget));
        pool.run(noting(&order, "committed"))
            .hasten(Urgency::Committed);
        until(|| budget.census() == (1, 3));
        let head = Pool::within(Arc::clone(&budget));
        head.run(noting(&order, "head")).hasten(Urgency::Head);
        until(|| budget.census() == (1, 4));
        let step = ask(&budget, &order, ("host step", true));
        until(|| budget.census() == (1, 5));
        drop(held);
        for thread in [program, step] {
            thread.join().unwrap();
        }
        until(|| order.lock().unwrap().len() == 5);
        assert_eq!(
            *order.lock().unwrap(),
            ["host step", "head", "committed", "program", "ahead"]
        );
        until(|| budget.census() == (0, 0));
    }

    /// A warm pass's makings go after every other making, an aged one included, by their
    /// tickets, and never age themselves (`warm/pass.rs`): a warm piece handed first and a warm
    /// thread's making asked for next both wait out a piece nothing waits for, which ages past a
    /// program's making asked for last, and that program's making, though they waited past the
    /// age too. A warm piece runs as part of its pass.
    #[test]
    fn a_warm_pass_s_makings_go_last_and_never_age() {
        let budget = Arc::new(Budget::aging(1, 1, Duration::from_millis(20)));
        let held = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        let pass = Arc::new(crate::warm::Pass::default());
        let noted = Arc::clone(&order);
        crate::warm::within(&pass, || {
            pool.run(Box::new(move || {
                assert!(
                    crate::warm::warming(),
                    "a warm piece runs as part of its pass"
                );
                noted.lock().unwrap().push("warm piece");
            }))
        });
        assert!(!crate::warm::warming(), "the pass ends with its call");
        until(|| budget.census() == (1, 1));
        let warm_thread = {
            let (budget, order) = (Arc::clone(&budget), Arc::clone(&order));
            let pass = Arc::clone(&pass);
            let asked = ticket();
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                let _turn = crate::warm::within(&pass, || waited_turn(&budget, asked));
                order.lock().unwrap().push("warm thread");
            })
            .unwrap()
        };
        until(|| budget.census() == (1, 2));
        pool.run(noting(&order, "ahead"));
        until(|| budget.census() == (1, 3));
        let program = ask(&budget, &order, ("program", false));
        until(|| budget.census() == (1, 4));
        // Past the age for every one of them: only the piece nothing waits for ages.
        std::thread::sleep(Duration::from_millis(100));
        drop(held);
        for thread in [program, warm_thread] {
            thread.join().unwrap();
        }
        until(|| order.lock().unwrap().len() == 4);
        assert_eq!(
            *order.lock().unwrap(),
            ["ahead", "program", "warm piece", "warm thread"]
        );
        until(|| budget.census() == (0, 0));
    }

    /// A queue's first held batch's making goes first to a freed turn, though a piece nothing
    /// waits for, aged (ordered as a committed one's), and a committed batch's piece behind
    /// another queue's head were handed before it: Metal holds no submission for a making, so the
    /// making a queue waits behind goes before the makings of batches that could not go yet
    /// anyway. Those then go in their order: the aged one by its earlier ticket.
    ///
    /// One turn is free at a time, so the order noted is the order the turns were given: the head's
    /// piece holds its turn until the end, and the committed piece may take only the turn the aged
    /// one gives back once it noted itself. The aging is waited for as the pool ranks it (the aged
    /// piece first among those waiting), not slept for.
    #[test]
    fn a_queue_s_first_held_batch_s_making_goes_before_aged_and_earlier_committed_ones() {
        let budget = Arc::new(Budget::aging(2, 1, Duration::from_millis(50)));
        let first = waited_turn(&budget, ticket());
        let second = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        let aged = pool.run(noting(&order, "aged"));
        pool.run(noting(&order, "committed"))
            .hasten(Urgency::Committed);
        let (release, gate) = mpsc::channel::<()>();
        let head = {
            let order = Arc::clone(&order);
            pool.run(Box::new(move || {
                order.lock().unwrap().push("head");
                gate.recv().unwrap();
            }))
        };
        until(|| budget.census() == (2, 3));
        // The piece nothing waits for ages: ranked as a committed one, its earlier ticket puts it
        // before the committed piece.
        until(|| aged.is_first());
        // The scheduler finds the head's batch first in its queue's line.
        head.hasten(Urgency::Head);
        drop(first);
        until(|| !order.lock().unwrap().is_empty());
        drop(second);
        until(|| order.lock().unwrap().len() == 3);
        assert_eq!(*order.lock().unwrap(), ["head", "aged", "committed"]);
        release.send(()).unwrap();
        until(|| budget.census() == (0, 0));
    }

    /// Makings nothing waits for take idle turns, all but one by default: of the default budget's
    /// turns, the pool's pieces nothing waits for hold [`AHEAD`] at once and the next waits,
    /// while a program's making takes the last turn at once.
    #[test]
    fn makings_nothing_waits_for_use_every_idle_turn_but_one() {
        let budget = Arc::new(Budget::new(TURNS, AHEAD));
        let pool = Pool::within(Arc::clone(&budget));
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let (started, ran) = mpsc::channel();
        for at in 0..=AHEAD {
            let (gate, started) = (Arc::clone(&gate), started.clone());
            pool.run(Box::new(move || {
                started.send(at).unwrap();
                gate.lock().unwrap().recv().unwrap();
            }));
        }
        for _ in 0..AHEAD {
            ran.recv().unwrap();
        }
        until(|| budget.census() == (AHEAD, 1));
        let program = waited_turn(&budget, ticket());
        assert_eq!(budget.census(), (TURNS, 1), "the last turn, taken at once");
        drop(program);
        for _ in 0..=AHEAD {
            release.send(()).unwrap();
        }
        ran.recv().unwrap();
        until(|| budget.census() == (0, 0));
    }

    /// A making that waited its age is ordered as a committed one, on an ahead turn: a pool's
    /// piece nothing waits for, handed before a program's making is asked for, takes the ahead
    /// turn freed first once it aged, and the program's making the next.
    #[test]
    fn a_making_that_waited_its_age_goes_first_to_an_ahead_turn() {
        let budget = Arc::new(Budget::aging(2, 1, Duration::from_millis(50)));
        let first = waited_turn(&budget, ticket());
        let second = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        pool.run(noting(&order, "aged"));
        until(|| budget.census() == (2, 1));
        std::thread::sleep(Duration::from_millis(150));
        let program = ask(&budget, &order, ("program", false));
        until(|| budget.census() == (2, 2));
        drop(first);
        // The aged piece takes the freed turn; the program's making may take it next at once.
        until(|| !order.lock().unwrap().is_empty());
        drop(second);
        program.join().unwrap();
        until(|| order.lock().unwrap().len() == 2);
        assert_eq!(*order.lock().unwrap(), ["aged", "program"]);
        until(|| budget.census() == (0, 0));
    }

    /// Aged makings nothing waits for hold the ahead turns and no more: the last turn stays free
    /// for a making something waits for, which takes it at once though an aged one asked first.
    #[test]
    fn aged_makings_nothing_waits_for_leave_the_last_turn_to_a_waited_one() {
        let budget = Arc::new(Budget::aging(TURNS, AHEAD, Duration::from_millis(50)));
        let pool = Pool::within(Arc::clone(&budget));
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let (started, ran) = mpsc::channel();
        for at in 0..=AHEAD {
            let (gate, started) = (Arc::clone(&gate), started.clone());
            pool.run(Box::new(move || {
                started.send(at).unwrap();
                gate.lock().unwrap().recv().unwrap();
            }));
        }
        for _ in 0..AHEAD {
            ran.recv().unwrap();
        }
        until(|| budget.census() == (AHEAD, 1));
        // Every piece waiting has aged a few times over, and is ranked again meanwhile.
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(
            budget.census(),
            (AHEAD, 1),
            "the aged piece took no turn past the ahead ones"
        );
        let program = waited_turn(&budget, ticket());
        assert_eq!(budget.census(), (TURNS, 1), "the last turn, taken at once");
        drop(program);
        for _ in 0..=AHEAD {
            release.send(()).unwrap();
        }
        ran.recv().unwrap();
        until(|| budget.census() == (0, 0));
    }

    /// A making hastened before it is handed to the pool (a draw's wait or a commit between
    /// its memo entry's beginning and the pool's place) runs at the rank asked once handed: a
    /// committed one's before a program's making and a piece nothing waits for, handed and
    /// asked for before it.
    #[test]
    fn a_hasten_before_the_piece_is_handed_applies_as_it_is_handed() {
        use crate::pipeline::making::{Begun, Making};
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let order = Arc::new(Mutex::new(Vec::new()));
        pool.run(noting(&order, "ahead"));
        until(|| budget.census() == (1, 1));
        let program = ask(&budget, &order, ("program", false));
        until(|| budget.census() == (1, 2));
        let memo: Making<u32, u32> = Making::counted(4);
        let Begun::Yours(making) = memo.begin(&1) else {
            panic!("the first caller's making");
        };
        let pending = making.pending();
        // A commit's wait, before the making has a place.
        pending.hasten(Urgency::Committed);
        let made = {
            let order = Arc::clone(&order);
            Box::new(move || {
                order.lock().unwrap().push("made");
                drop(making.settle(Ok(1)));
            })
        };
        pending.queued_at(pool.run(made));
        until(|| budget.census() == (1, 3));
        drop(held);
        program.join().unwrap();
        until(|| order.lock().unwrap().len() == 3);
        assert_eq!(*order.lock().unwrap(), ["made", "program", "ahead"]);
        assert_eq!(*pending.wait().unwrap(), 1);
        until(|| budget.census() == (0, 0));
    }

    /// A piece of a joined set handed to the pool rises in rank where something waits for it: hastened through the place the join gave out as a commit's, it goes before a
    /// program's making and the set's other pieces, which then run in order, and the join
    /// returns after all of them.
    #[test]
    fn a_hasten_on_a_joined_piece_raises_it() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let order = Arc::new(Mutex::new(Vec::new()));
        let program = ask(&budget, &order, ("program", false));
        until(|| budget.census() == (1, 1));
        let (placed, places) = mpsc::channel();
        let joining = {
            let (budget, order) = (Arc::clone(&budget), Arc::clone(&order));
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                let pool = Pool::within(budget);
                let pieces: Vec<crate::pipeline::pool::Joined<()>> = ["k0", "k1", "k2"]
                    .into_iter()
                    .map(|name| {
                        let order = Arc::clone(&order);
                        let piece: crate::pipeline::pool::Joined<()> =
                            Box::new(move || order.lock().unwrap().push(name));
                        piece
                    })
                    .collect();
                pool.join_all(pieces, false, |index, place| {
                    placed.send((index, place.clone())).unwrap();
                });
            })
            .unwrap()
        };
        let places: Vec<_> = (0..3).map(|_| places.recv().unwrap()).collect();
        // The program's making and the set's three threads ask.
        until(|| budget.census() == (1, 4));
        let (_, kept) = places.iter().find(|(index, _)| *index == 2).unwrap();
        kept.hasten(Urgency::Committed);
        drop(held);
        program.join().unwrap();
        joining.join().unwrap();
        assert_eq!(*order.lock().unwrap(), ["k2", "program", "k0", "k1"]);
        until(|| budget.census() == (0, 0));
    }

    /// A pool's thread asking for a turn is busy: a piece handed meanwhile starts a thread of its
    /// own, which asks too, rather than waiting behind the asking thread.
    #[test]
    fn a_thread_asking_for_a_turn_is_busy() {
        let budget = Arc::new(Budget::new(1, 1));
        let held = waited_turn(&budget, ticket());
        let pool = Pool::within(Arc::clone(&budget));
        let (sent, got) = mpsc::channel();
        for at in 0..2 {
            let sent = sent.clone();
            pool.run(Box::new(move || sent.send(at).unwrap()));
            // Each piece's thread asks before the next piece comes.
            until(|| budget.census() == (1, at + 1));
        }
        drop(held);
        assert_eq!([got.recv().unwrap(), got.recv().unwrap()], [0, 1]);
        until(|| budget.census() == (0, 0));
    }

    /// A helper made already takes no turn; one to make takes its turn before its lock, so its
    /// lock is free while it waits, and its making inside takes no second turn.
    #[test]
    fn a_helper_takes_its_turn_before_its_lock_and_only_to_make() {
        let objects = Mutex::new(None::<u32>);
        let turn = helper_turn(&objects, |made| made.is_some());
        assert!(turn.is_some(), "a turn to make it");
        let mut made = objects.try_lock().expect("its lock is free");
        assert!(program_turn().is_none(), "its making takes no second turn");
        *made = Some(7);
        drop(made);
        drop(turn);
        assert!(helper_turn(&objects, |made| made.is_some()).is_none());
    }
}
