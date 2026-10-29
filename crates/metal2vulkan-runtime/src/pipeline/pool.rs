//! The device's pool of threads that make `VkPipeline`s ahead of the commands that record them: a
//! draw whose pipeline is not made yet hands its making here and goes on, and its command buffer's
//! batch is submitted once it is made.
//!
//! - **Bounded.** At most as many threads as its budget has turns (`Budget::turns`, sized to the
//!   host: `budget::turns_for`), and never fewer than four, the threads every pool had before
//!   ([`PoolQueue::makers`]), each started as work arrives while none is free, and ended once it
//!   found none for [`IDLE`]. The work waiting is bounded by the memos that hand it here: each
//!   piece is one key's making (`making.rs`), and each memo holds a bounded count of keys.
//! - **In order.** Work runs in the order it arrived, each piece on one thread, so makings of
//!   distinct keys run side by side, at most that many of them.
//! - **Waited for first** (ranked). A piece something waits for ([`Place::hasten`], with how: a
//!   committed command buffer's batch held for its pipeline, or a caller blocked on its making,
//!   `making.rs`) goes ahead of every piece still waiting that nothing waits for: the pieces go by
//!   their rank (`budget::Urgency`), then by their tickets, the order they were handed here.
//! - **The exit guard.** Each piece runs inside a section (`exit_guard`), which registers the
//!   hook again after the process's first pipeline made in one. Once the process is exiting, a
//!   piece not started is dropped unrun: it fails its making (`making::Ticket`), as Metal's
//!   pending work stays pending. While a piece's driver call makes a pipeline
//!   (`exit_guard::driver_making`), the exit waits for it, on macOS up to
//!   `exit_guard::MAKING_BOUND`, not the guard's two seconds: the program need not wait for the
//!   command buffer whose making it is before it exits, and a making outlasts those. On Linux the
//!   exit waits as for any section, and then ends the process at once (`exit_guard`).
//! - **No thread.** Where the host cannot start a thread and none runs, the piece runs on the
//!   caller's thread, as before the pool.
//! - **At the user-initiated class** (a concurrency change). On macOS each thread runs at the
//!   user-initiated quality of service, or the user-interactive class it started at, whoever
//!   started it: the driver's compile takes the class of the thread asking (MoltenVK's dispatch
//!   queue).
//! - **Joined sets** (the owner's HITCH rules 1 and 3). [`Pool::join_all`] runs pieces at once for
//!   a caller that waits for all of them (a state's creation making its keys): the first on the
//!   caller's thread (a piece too, [`in_piece`]), each other here, as a piece nothing waits for
//!   until something does (a creation's kept keys), and returns only once every one ran, or was
//!   dropped unrun by the exit guard: a full join, each piece counted in it before any is handed
//!   (so a panic while handing them cannot strand the wait). A piece's panic in the pool unwinds
//!   its thread (a thread starts in its place where pieces wait), and the caller panics in turn
//!   after the join. Its outcomes come back in the caller's order; whatever the program or host
//!   sees is decided by the caller before or after the join, never in a piece.
//! - **Within the process's budget**. A thread takes a turn of the process's budget of makings in
//!   flight before it takes its next piece, and holds it through the piece (`budget.rs`): the next
//!   piece, the one of the lowest rank here, is ranked by its urgency and the ticket it was handed
//!   here with against every other making of the process, and the budget hands the thread that
//!   piece with the turn. A thread asking is busy, so the pool starts another for a piece handed
//!   meanwhile, up to its threads.

use super::budget::{self, Budget, Candidate, Urgency};
use crate::device::lock;
use crate::exit_guard::{self, Call};
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

/// The fewest threads a pool may run ([`PoolQueue::makers`]): the four every pool ran previously
/// sized the process's budget to the host.
const FEWEST_MAKERS: usize = 4;

/// How long a thread waits for work before it ends.
const IDLE: Duration = Duration::from_secs(5);

/// One piece of work: a making, run on one of the pool's threads.
pub(crate) type Work = Box<dyn FnOnce() + Send>;

/// One piece of a joined set ([`Pool::join_all`]), giving its outcome.
pub(crate) type Joined<T> = Box<dyn FnOnce() -> T + Send>;

/// A joined set's outcomes, by piece, and the count of its pieces not finished.
struct Join<T> {
    state: Mutex<(Vec<Option<T>>, usize)>,
    finished: Condvar,
    /// Whether a piece run in the pool panicked: the caller panics in turn after the full join.
    /// The pool's thread lets the piece's panic unwind (its panic hook reports it there, and the
    /// thread ends), as every runtime thread does: only `exit_guard::program` catches a panic,
    /// and a piece is the runtime's code, not the program's, so it does not run as the
    /// program's (which would lift its section).
    panicked: AtomicBool,
}

impl<T> Join<T> {
    /// Piece `index` finished, with `outcome` (`None`: dropped unrun).
    fn finish(&self, index: usize, outcome: Option<T>) {
        let mut state = lock(&self.state);
        if let Some(slot) = state.0.get_mut(index) {
            *slot = outcome;
        }
        state.1 -= 1;
        self.finished.notify_all();
    }

    /// Wait until every piece finished: the full join.
    fn wait(&self) -> std::sync::MutexGuard<'_, (Vec<Option<T>>, usize)> {
        let mut state = lock(&self.state);
        while state.1 > 0 {
            state = self
                .finished
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
    }
}

/// A piece's hold on its set: it finishes the piece with its outcome, or, dropped unrun (the
/// exit guard drops a piece not started once the process is exiting) or by a panic, with none.
struct Piece<T> {
    join: Arc<Join<T>>,
    index: usize,
    done: bool,
}

/// A joined piece handed to the pool: its work, then its place in the join, in this order, so a
/// piece dropped unrun drops what its work owns before it finishes: Rust drops a struct's fields in
/// the order they are declared, which
/// `tests::a_piece_dropped_unrun_drops_its_work_before_it_finishes` pins. A piece that runs
/// consumes its work before it finishes.
struct Handed<T> {
    work: Joined<T>,
    piece: Piece<T>,
}

impl<T> Piece<T> {
    fn finish(mut self, outcome: T) {
        self.done = true;
        self.join.finish(self.index, Some(outcome));
    }
}

impl<T> Drop for Piece<T> {
    fn drop(&mut self) {
        if !self.done {
            // Dropped as its work's panic unwinds: the caller panics after the full join.
            if std::thread::panicking() {
                self.join.panicked.store(true, Ordering::Relaxed);
            }
            self.join.finish(self.index, None);
        }
    }
}

/// The caller's side of a joined set: dropped, by a panic of the caller's own piece too, it
/// waits for the rest, so no piece outlives the call.
struct Waiter<'a, T>(&'a Join<T>);

impl<T> Drop for Waiter<'_, T> {
    fn drop(&mut self) {
        drop(self.0.wait());
    }
}

/// The device's pool (the module docs).
pub(crate) struct Pool {
    shared: Arc<PoolQueue>,
}

/// A pool's work and threads, which its places and its budget's askers look at.
pub(crate) struct PoolQueue {
    state: Mutex<State>,
    wake: Condvar,
    /// The budget its threads take their turns from: the process's, or a test's.
    budget: Arc<Budget>,
}

struct State {
    /// The work waiting, in the order handed here, each by its number (its ticket of the
    /// process's order, `budget::ticket`) and its rank ([`Place::hasten`] raises it).
    work: VecDeque<Queued>,
    /// The threads alive, and those of them waiting for work.
    threads: usize,
    free: usize,
}

/// A piece waiting: its number, its rank, its work.
struct Queued {
    number: u64,
    urgency: Urgency,
    /// When it was handed here: it ages from then (`budget::aged`).
    handed: Instant,
    work: Work,
}

impl PoolQueue {
    /// The most threads the pool runs: one per turn of its budget, which holds that many makings
    /// at once, and never fewer than [`FEWEST_MAKERS`]. A thread asking for a turn is busy, so a
    /// piece handed meanwhile starts a thread of its own, which asks too, even where other
    /// devices' makings hold the turns (`budget::tests::a_thread_asking_for_a_turn_is_busy`).
    fn makers(&self) -> usize {
        self.budget.turns().max(FEWEST_MAKERS)
    }

    /// The piece a thread would take next (`budget.rs`): its rank (its urgency, then its ticket)
    /// and whether nothing waits for it; one something waits for only where `ahead_free` is
    /// not (no ahead turn free). `None` where none waits here; `Some(None)` where every piece
    /// here is one nothing waits for and no ahead turn is free.
    pub(crate) fn next_rank(&self, ahead_free: bool) -> Option<Option<Candidate>> {
        lock(&self.state).next_rank(self.budget.age(), ahead_free)
    }

    /// The piece [`next_rank`](Self::next_rank) ranks, taken, with its rank unaged and whether
    /// nothing waits for it: for the budget, which hands it to the thread it gives the turn
    /// (`budget.rs`).
    pub(crate) fn next_piece(&self, ahead_free: bool) -> Option<(Work, Candidate)> {
        lock(&self.state).next_piece(self.budget.age(), ahead_free)
    }
}

impl State {
    /// The next piece to run, of those it may be (every piece where `ahead_free`, else those
    /// something waits for): the lowest rank (`budget::Urgency`, then ticket), and whether
    /// nothing waits for it. Each piece ranks as aged (`budget::aged`): a committed one's once
    /// it waited `age`; aged, a piece nothing waits for still takes only an ahead turn.
    fn next_rank(&self, age: Duration, ahead_free: bool) -> Option<Option<Candidate>> {
        if self.work.is_empty() {
            return None;
        }
        let rank = self
            .work
            .iter()
            .filter(|q| ahead_free || q.urgency < Urgency::Ahead)
            .map(|q| {
                let rank = (budget::aged(q.urgency, q.handed, age), q.number);
                (rank, q.urgency >= Urgency::Ahead)
            })
            .min();
        Some(rank)
    }

    /// The next piece to run, taken, as [`next_rank`](Self::next_rank) ranks it, with its rank
    /// unaged (its urgency, its number) and whether nothing waits for it.
    fn next_piece(&mut self, age: Duration, ahead_free: bool) -> Option<(Work, Candidate)> {
        let ((_, number), ahead) = self.next_rank(age, ahead_free)??;
        let at = self.work.iter().position(|q| q.number == number)?;
        let queued = self.work.remove(at)?;
        Some((queued.work, ((queued.urgency, number), ahead)))
    }
}

/// Where a piece handed to the pool waits ([`Pool::run`]), to hasten it.
#[derive(Clone)]
pub(crate) struct Place {
    shared: Weak<PoolQueue>,
    number: u64,
}

impl Place {
    /// Something waits for the piece, as `urgency` says (the module docs): it runs before every
    /// piece waiting of a lower rank. No change where it started, or ranks as high already; a
    /// rank never falls.
    pub(crate) fn hasten(&self, urgency: Urgency) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let mut state = lock(&shared.state);
        let Some(piece) = state.work.iter_mut().find(|q| q.number == self.number) else {
            return;
        };
        if piece.urgency <= urgency {
            return;
        }
        piece.urgency = urgency;
        drop(state);
        // Its pool's next piece may now rank as waited for: the budget's askers look again.
        shared.budget.reranked();
    }

    /// Whether the piece waits first in the pool, the next to run (the tests).
    #[cfg(test)]
    pub(crate) fn is_first(&self) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return false;
        };
        let state = lock(&shared.state);
        state
            .next_rank(shared.budget.age(), true)
            .flatten()
            .is_some_and(|((_, n), _)| n == self.number)
    }
}

impl Default for Pool {
    /// A pool within the process's budget.
    fn default() -> Pool {
        Pool::within(Arc::clone(budget::process()))
    }
}

impl Pool {
    /// A pool whose threads take their turns from `budget`.
    pub(crate) fn within(budget: Arc<Budget>) -> Pool {
        Pool {
            shared: Arc::new(PoolQueue {
                state: Mutex::new(State {
                    work: VecDeque::new(),
                    threads: 0,
                    free: 0,
                }),
                wake: Condvar::new(),
                budget,
            }),
        }
    }

    /// Run `work` on one of the pool's threads, a new one where none is free and fewer than
    /// [`PoolQueue::makers`] run; on this thread where the host starts none and none runs. Nothing waits for
    /// it yet ([`Urgency::Ahead`]; a warm pass's piece ranks [`Urgency::Warm`] and runs as part
    /// of its pass, `warm::within`); where it waits, to hasten it.
    pub(crate) fn run(&self, work: Work) -> Place {
        let number = budget::ticket();
        // Its wait here and for its turn, for the creation log: noted as it starts.
        let handed = Instant::now();
        let pass = crate::warm::pass();
        let urgency = match pass {
            Some(_) => Urgency::Warm,
            None => Urgency::Ahead,
        };
        let work: Work = Box::new(move || {
            budget::waited_in_pool(handed.elapsed());
            match pass {
                Some(pass) => crate::warm::within(&pass, work),
                None => work(),
            }
        });
        let mut state = lock(&self.shared.state);
        state.work.push_back(Queued {
            number,
            urgency,
            handed,
            work,
        });
        let place = Place {
            shared: Arc::downgrade(&self.shared),
            number,
        };
        if state.free >= state.work.len() || state.threads >= self.shared.makers() {
            drop(state);
            self.shared.wake.notify_one();
            return place;
        }
        state.threads += 1;
        state.free += 1;
        drop(state);
        let shared = Arc::clone(&self.shared);
        let started = exit_guard::spawn(
            std::thread::Builder::new().name("metal2vulkan-pipelines".into()),
            move || serve(&shared),
        );
        if started.is_ok() {
            return place;
        }
        let mut state = lock(&self.shared.state);
        state.threads -= 1;
        state.free -= 1;
        if state.threads > 0 {
            return place;
        }
        // No thread: the work waiting runs here, in its order.
        let age = self.shared.budget.age();
        let waiting: Vec<Work> = std::iter::from_fn(|| state.next_piece(age, true))
            .map(|(work, _)| work)
            .collect();
        drop(state);
        for work in waiting {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held::wait();
            run_one(work);
        }
        place
    }

    /// Run `pieces` at once and wait for all of them (the module docs): the first on this thread
    /// where `first_here`, each other on the pool, ranked as nothing waits for it until something
    /// does, each one's place given to `handed` with its index as it is handed there, so a wait for
    /// it can hasten it; their outcomes in order, `None` for a piece the exit guard dropped unrun.
    /// Returns only once every piece finished; where a piece run in the pool panicked, this panics
    /// then, after the join.
    pub(crate) fn join_all<T: Send + 'static>(
        &self,
        pieces: Vec<Joined<T>>,
        first_here: bool,
        mut handed: impl FnMut(usize, &Place),
    ) -> Vec<Option<T>> {
        let count = pieces.len();
        let join = Arc::new(Join {
            state: Mutex::new(((0..count).map(|_| None).collect(), count)),
            finished: Condvar::new(),
            panicked: AtomicBool::new(false),
        });
        let waiter = Waiter(&join);
        // Every piece takes its place in the join before anything here can panic: one never
        // handed, a panic of this loop included, drops and finishes with none, so the waiter's
        // join returns.
        let mut pieces: std::collections::VecDeque<Handed<T>> = pieces
            .into_iter()
            .enumerate()
            .map(|(index, work)| Handed {
                work,
                piece: Piece {
                    join: Arc::clone(&join),
                    index,
                    done: false,
                },
            })
            .collect();
        let first = match first_here {
            true => pieces.pop_front(),
            false => None,
        };
        for one in pieces {
            let index = one.piece.index;
            // Dropped unrun (the process exiting), the work and what it owns go before the join
            // learns the piece finished: the fields drop in order.
            let place = self.run(Box::new(move || {
                let Handed { work, piece } = one;
                // A panic unwinds this thread and drops the piece, which marks the join: the
                // caller panics too, after the full join (`Piece`'s drop).
                piece.finish(work());
            }));
            // Not hastened: the caller waits for the set, but no draw has asked for these keys,
            // so they rank as makings nothing waits for, until a draw or a commit hastens one
            // through its place.
            handed(index, &place);
        }
        if let Some(Handed { work, piece }) = first {
            // The caller's own piece runs beside the pool's, so it is a piece as theirs are: the
            // HITCH rule 1 debug assertions hold in it too. Nothing it waited before is its own
            // (the creation log's wait).
            let outcome = as_piece(|| {
                budget::waited_in_pool(Duration::ZERO);
                work()
            });
            piece.finish(outcome);
        }
        let mut outcomes = Vec::with_capacity(count);
        outcomes.append(&mut join.wait().0);
        drop(waiter);
        // The join's lock taken in `wait` orders the pieces' marks before this read.
        assert!(
            !join.panicked.load(Ordering::Relaxed),
            "a piece of a joined set panicked on the device's pipeline pool (reported above)"
        );
        outcomes
    }

    /// The urgency the piece waiting first here ranks at, aged as the budget orders it, where any
    /// waits (the device tests, `tests/pipeline_wait_order.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn first_urgency(&self) -> Option<Urgency> {
        let state = lock(&self.shared.state);
        let first = state.next_rank(self.shared.budget.age(), true).flatten();
        first.map(|((urgency, _), _)| urgency)
    }

    /// The pieces it holds as asked, waiting or running on its threads
    /// (`WorkCounts::pipeline_pool_busy`).
    pub(crate) fn busy(&self) -> usize {
        let state = lock(&self.shared.state);
        state.work.len() + state.threads.saturating_sub(state.free)
    }

    /// The threads alive and those of them free (the pool's tests).
    #[cfg(test)]
    fn census(&self) -> (usize, usize) {
        let state = lock(&self.shared.state);
        (state.threads, state.free)
    }
}

/// A thread's loop: run the work in order, each piece inside a turn of the pool's budget, and end
/// once none came for [`IDLE`].
fn serve(shared: &Arc<PoolQueue>) {
    let _alive = Alive(shared);
    // The pool's makings compile at the user-initiated class, whoever started the thread.
    exit_guard::raise_to_user_initiated();
    let mut state = lock(&shared.state);
    loop {
        if !state.work.is_empty() {
            // Asking, the thread is busy: a piece handed meanwhile starts a thread of its own.
            state.free -= 1;
            drop(state);
            // A test's hold on the pool holds the thread before it asks, holding no turn.
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held::wait();
            // The budget's lock is never taken under the pool's: the turn is asked for, and
            // given back, with the pool's lock released; the budget takes the piece its rank was
            // from the pool under its own (`budget::pool_turn`).
            if let Some((turn, work)) = budget::pool_turn(&shared.budget, shared) {
                run_one(work);
                drop(turn);
            }
            state = lock(&shared.state);
            state.free += 1;
            continue;
        }
        let (next, waited) = shared
            .wake
            .wait_timeout(state, IDLE)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next;
        if waited.timed_out() && state.work.is_empty() {
            state.free -= 1;
            break;
        }
    }
}

/// A thread of the pool, counted until it ends, by a panic in its work too.
struct Alive<'a>(&'a Arc<PoolQueue>);

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.0.state);
        state.threads -= 1;
        // Ended with pieces waiting that no free thread will take (a piece's panic unwound it,
        // or work came as it idled out): another thread in its place, as `Pool::run` starts
        // one, so every set's pieces still run and every join returns. Where none can start,
        // the next piece handed starts one, or runs on its caller.
        if state.free < state.work.len() && state.threads < self.0.makers() {
            state.threads += 1;
            state.free += 1;
            drop(state);
            let shared = Arc::clone(self.0);
            let started = exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-pipelines".into()),
                move || serve(&shared),
            );
            if started.is_err() {
                let mut state = lock(&self.0.state);
                state.threads -= 1;
                state.free -= 1;
            }
        }
    }
}

thread_local! {
    /// The pieces this thread is running now ([`in_piece`]).
    static PIECES: Cell<usize> = const { Cell::new(0) };
}

/// Whether this thread is running one of the pool's pieces now, on a thread of the pool or (with
/// no thread) the caller's: the debug assertions that nothing the program sees, nor the order of
/// anything shared, is decided in one (the owner's HITCH rule 1; `degradation.rs`, `kept.rs`).
pub(crate) fn in_piece() -> bool {
    PIECES.with(Cell::get) > 0
}

/// Run `work` inside a section of the exit guard, or drop it unrun once the process is exiting.
fn run_one(work: Work) {
    let Some(_busy) = exit_guard::enter() else {
        return;
    };
    as_piece(work);
    exit_guard::called(Call::CreateGraphicsPipelines);
}

/// Run `work` as one of the pool's pieces on this thread ([`in_piece`]), counted down again
/// however it ends: a pool thread's piece, and a joined set's first piece on its caller.
fn as_piece<R>(work: impl FnOnce() -> R) -> R {
    /// This thread's count of pieces running, down again however the piece ends.
    struct InPiece;
    impl Drop for InPiece {
        fn drop(&mut self) {
            PIECES.with(|pieces| pieces.set(pieces.get() - 1));
        }
    }
    PIECES.with(|pieces| pieces.set(pieces.get() + 1));
    let _in_piece = InPiece;
    work()
}

/// The runtime's tests' hold on the pool: while a [`Hold`](held::Hold) lives, no piece starts,
/// so every pipeline a draw hands here stays being made (`tests/pipeline_wait_order.rs`, a
/// device test, compiled under the validation crate's GPU safety cfgs alone). The holds are
/// counted: tests that run at once in one process each keep the pool held until their own is
/// dropped, whatever the others drop.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod held {
    use crate::device::lock;
    use std::sync::{Condvar, Mutex};

    /// The holds that live.
    static HELD: Mutex<usize> = Mutex::new(0);
    static RELEASED: Condvar = Condvar::new();

    /// One hold, given back when dropped.
    pub(crate) struct Hold(());

    /// Hold every piece the pool starts from now until this hold and every other is dropped.
    pub(crate) fn hold() -> Hold {
        *lock(&HELD) += 1;
        Hold(())
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            let mut held = lock(&HELD);
            *held -= 1;
            if *held == 0 {
                RELEASED.notify_all();
            }
        }
    }

    /// Whether a hold lives.
    pub(crate) fn holding() -> bool {
        *lock(&HELD) > 0
    }

    /// Wait while a hold lives.
    pub(super) fn wait() {
        let mut held = lock(&HELD);
        while *held > 0 {
            held = RELEASED
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// A point in a piece that pieces stop at while a hold of it lives, counting those that reached
    /// it.
    struct Stop {
        /// Whether a hold lives, and how many pieces reached the point since it was taken.
        state: Mutex<(bool, usize)>,
        changed: Condvar,
    }

    impl Stop {
        const fn new() -> Stop {
            Stop {
                state: Mutex::new((false, 0)),
                changed: Condvar::new(),
            }
        }

        /// Hold every piece that reaches the point from now until the hold is dropped.
        fn hold(&'static self) -> Holding {
            *lock(&self.state) = (true, 0);
            Holding(self)
        }

        /// A piece reached the point: wait while a hold lives.
        fn reach(&self) {
            let mut state = lock(&self.state);
            if !state.0 {
                return;
            }
            state.1 += 1;
            self.changed.notify_all();
            while state.0 {
                state = self
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }

    /// A hold on a [`Stop`]: released when dropped.
    pub(crate) struct Holding(&'static Stop);

    impl Holding {
        /// Wait until `pieces` pieces reached the hold, at most `wait`: whether they did.
        pub(crate) fn reached(&self, pieces: usize, wait: std::time::Duration) -> bool {
            let state = lock(&self.0.state);
            let (state, _) = self
                .0
                .changed
                .wait_timeout_while(state, wait, |state| state.1 < pieces)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.1 >= pieces
        }
    }

    impl Drop for Holding {
        fn drop(&mut self) {
            lock(&self.0.state).0 = false;
            self.0.changed.notify_all();
        }
    }

    /// A draw's making that took what its state kept for it (`render.rs`).
    static MID_MAKING: Stop = Stop::new();

    /// A dispatch's making that its piece settled, before the piece gives back what it took
    /// (`dispatch_making.rs`): inside the piece's trail (`trails.rs`).
    static SETTLED: Stop = Stop::new();

    /// Hold every draw's making from now, once it took what its state kept for it, until the
    /// hold is dropped: a making running, which its state's release cannot stop.
    pub(crate) fn mid_making_hold() -> Holding {
        MID_MAKING.hold()
    }

    /// A draw's making that took what its state kept for it: wait while a
    /// [`mid_making_hold`] lives.
    pub(crate) fn mid_making() {
        MID_MAKING.reach();
    }

    /// Hold every dispatch's piece from now, once it settled its making and before it gives back
    /// what it took, until the hold is dropped: the batches that waited for the making may run
    /// and complete meanwhile.
    pub(crate) fn settled_hold() -> Holding {
        SETTLED.hold()
    }

    /// A dispatch's piece that settled its making: wait while a [`settled_hold`] lives.
    pub(crate) fn settled() {
        SETTLED.reach();
    }
}

#[cfg(test)]
mod tests {
    use super::{in_piece, Handed, Join, Joined, Piece, Place, Pool};
    use crate::device::lock;
    use crate::pipeline::budget::{Budget, Urgency};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    /// How long a piece waits for the others before the test calls them serial: a guard against
    /// a hang, never reached when they overlap.
    const GUARD: Duration = Duration::from_secs(60);

    /// The turns of the budgets these tests make, so the threads of a pool within one.
    const MAKERS: usize = 4;

    /// A pool within a budget of its own, whose turns let [`MAKERS`] pieces nothing waits for run
    /// at once, whatever the other tests make in the process's.
    fn roomy() -> Pool {
        Pool::within(Arc::new(Budget::new(MAKERS, MAKERS)))
    }

    /// [`MAKERS`] pieces handed to the pool at once run together, each on a thread of its own,
    /// and no more threads than that start for more pieces.
    #[test]
    fn makers_pieces_run_together_on_at_most_makers_threads() {
        let pool = roomy();
        let meeting = Arc::new((Mutex::new(0usize), Condvar::new()));
        let (sent, got) = mpsc::channel();
        for _ in 0..MAKERS {
            let (meeting, sent) = (Arc::clone(&meeting), sent.clone());
            pool.run(Box::new(move || {
                let (count, arrived) = &*meeting;
                let mut count = count.lock().unwrap();
                *count += 1;
                arrived.notify_all();
                let (count, waited) = arrived
                    .wait_timeout_while(count, GUARD, |count| *count < MAKERS)
                    .unwrap();
                drop(count);
                sent.send(!waited.timed_out()).unwrap();
            }));
        }
        let together: Vec<bool> = (0..MAKERS).map(|_| got.recv().unwrap()).collect();
        assert_eq!(together, vec![true; MAKERS], "every piece met the others");
        // More pieces than threads wait their turn on the same threads.
        let (sent, got) = mpsc::channel();
        for at in 0..3 * MAKERS {
            let sent = sent.clone();
            pool.run(Box::new(move || sent.send(at).unwrap()));
        }
        let mut ran: Vec<usize> = (0..3 * MAKERS).map(|_| got.recv().unwrap()).collect();
        ran.sort_unstable();
        assert_eq!(ran, (0..3 * MAKERS).collect::<Vec<_>>());
        let (threads, _) = pool.census();
        assert!(threads <= MAKERS, "{threads} threads");
    }

    /// With every thread busy and [`MAKERS`] pieces waiting, a piece a commit waits for runs
    /// first, one a program's thread waits for next, and the rest in the order they arrived;
    /// hastening a piece to a lower rank changes nothing.
    #[test]
    fn a_waited_for_piece_runs_before_the_pieces_queued_ahead_of_it() {
        let pool = roomy();
        // Pieces that hold every thread, one let go per message.
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let (started, busy) = mpsc::channel();
        for _ in 0..MAKERS {
            let (gate, started) = (Arc::clone(&gate), started.clone());
            pool.run(Box::new(move || {
                started.send(()).unwrap();
                gate.lock().unwrap().recv_timeout(GUARD).unwrap();
            }));
        }
        for _ in 0..MAKERS {
            busy.recv_timeout(GUARD).unwrap();
        }
        let (ran, order) = mpsc::channel();
        let places: Vec<Place> = (0..MAKERS)
            .map(|at| {
                let ran = ran.clone();
                pool.run(Box::new(move || ran.send(at).unwrap()))
            })
            .collect();
        places[3].hasten(Urgency::Committed);
        places[2].hasten(Urgency::Program);
        places[3].hasten(Urgency::Program);
        // One thread let go runs every piece waiting, in the pool's order.
        release.send(()).unwrap();
        let first: Vec<usize> = (0..MAKERS)
            .map(|_| order.recv_timeout(GUARD).unwrap())
            .collect();
        let rest = (0..MAKERS).filter(|at| ![3, 2].contains(at));
        assert_eq!(first, [3, 2].into_iter().chain(rest).collect::<Vec<_>>());
        for _ in 1..MAKERS {
            release.send(()).unwrap();
        }
    }

    /// A joined set runs its pieces together, the first on the caller's thread and the others on
    /// the pool's, each in a piece (the caller's too), returns
    /// their outcomes in order, and returns only once
    /// every piece finished: the slowest included, and none is left waiting in the pool.
    #[test]
    fn a_joined_set_runs_together_and_returns_after_every_piece() {
        let pool = roomy();
        let meeting = Arc::new((Mutex::new(0usize), Condvar::new()));
        let finished = Arc::new(Mutex::new(0usize));
        let pieces: Vec<Joined<(usize, bool, bool)>> = (0..MAKERS)
            .map(|at| {
                let (meeting, finished) = (Arc::clone(&meeting), Arc::clone(&finished));
                let piece: Joined<(usize, bool, bool)> = Box::new(move || {
                    let (count, arrived) = &*meeting;
                    let mut count = count.lock().unwrap();
                    *count += 1;
                    arrived.notify_all();
                    let (count, waited) = arrived
                        .wait_timeout_while(count, GUARD, |count| *count < MAKERS)
                        .unwrap();
                    drop(count);
                    // The last piece is the slowest.
                    if at == MAKERS - 1 {
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    *finished.lock().unwrap() += 1;
                    (at, !waited.timed_out(), in_piece())
                });
                piece
            })
            .collect();
        let outcomes = pool.join_all(pieces, true, |_, _| {});
        assert_eq!(
            *finished.lock().unwrap(),
            MAKERS,
            "every piece finished first"
        );
        let outcomes: Vec<(usize, bool, bool)> = outcomes.into_iter().map(Option::unwrap).collect();
        for (at, &(index, together, piece)) in outcomes.iter().enumerate() {
            assert_eq!(index, at, "in order");
            assert!(together, "piece {at} met the others");
            assert!(piece, "piece {at} in a piece, the caller's too");
        }
        assert!(!in_piece(), "the caller is not in a piece after the join");
        assert!(pool
            .join_all(Vec::<Joined<()>>::new(), true, |_, _| {})
            .is_empty());
    }

    /// A joined piece's panic in the pool is the caller's: the join still waits for every other
    /// piece, then the caller panics in turn; the next piece handed runs in a piece.
    #[test]
    fn a_joined_piece_s_panic_is_resumed_on_the_caller_after_the_join() {
        let pool = roomy();
        let finished = Arc::new(Mutex::new(0usize));
        let pieces: Vec<Joined<usize>> = (0..3)
            .map(|at| {
                let finished = Arc::clone(&finished);
                let piece: Joined<usize> = Box::new(move || {
                    if at == 1 {
                        panic!("piece 1 failed");
                    }
                    std::thread::sleep(Duration::from_millis(50));
                    *finished.lock().unwrap() += 1;
                    at
                });
                piece
            })
            .collect();
        // On a thread of its own, whose join gives the caller's panic.
        let pool = Arc::new(pool);
        let joining = {
            let pool = Arc::clone(&pool);
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                pool.join_all(pieces, true, |_, _| {})
            })
            .unwrap()
        };
        let payload = joining.join().expect_err("the caller panicked in turn");
        assert!(
            payload
                .downcast_ref::<&str>()
                .is_some_and(|message| message.contains("joined set panicked")),
            "the caller's panic names the joined set's"
        );
        assert_eq!(
            *finished.lock().unwrap(),
            2,
            "every other piece finished first"
        );
        assert!(!in_piece());
        let (sent, got) = mpsc::channel();
        pool.run(Box::new(move || sent.send(in_piece()).unwrap()));
        assert!(
            got.recv_timeout(GUARD).unwrap(),
            "the next piece runs in a piece"
        );
    }

    /// A piece dropped unrun (as the exit guard drops one) drops its work, and what the work
    /// owns, while its place in the join is still unfinished; then it finishes with none.
    #[test]
    fn a_piece_dropped_unrun_drops_its_work_before_it_finishes() {
        /// Notes, as it drops, how many of its join's pieces are unfinished.
        struct Seen(Arc<Join<()>>, Arc<Mutex<Option<usize>>>);
        impl Drop for Seen {
            fn drop(&mut self) {
                *self.1.lock().unwrap() = Some(lock(&self.0.state).1);
            }
        }
        let join = Arc::new(Join {
            state: Mutex::new((vec![None], 1)),
            finished: Condvar::new(),
            panicked: AtomicBool::new(false),
        });
        let seen = Arc::new(Mutex::new(None));
        let owned = Seen(Arc::clone(&join), Arc::clone(&seen));
        let handed = Handed {
            work: Box::new(move || {
                let _owned = &owned;
            }),
            piece: Piece {
                join: Arc::clone(&join),
                index: 0,
                done: false,
            },
        };
        drop(handed);
        assert_eq!(
            *seen.lock().unwrap(),
            Some(1),
            "the work went while its piece was unfinished"
        );
        assert_eq!(lock(&join.state).1, 0, "then the piece finished, with none");
        assert!(
            !join.panicked.load(Ordering::Relaxed),
            "and no panic marked"
        );
    }

    /// Every thread of the pool ends on a panicking piece while pieces of the same set and of a
    /// second set wait: threads start in their places, so both joins return, the panicked set's
    /// caller panicking in turn and the other set's with every outcome.
    #[test]
    fn pieces_waiting_as_every_thread_dies_on_a_panic_still_run() {
        let pool = Arc::new(roomy());
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let (started, running) = mpsc::channel();
        // The first set: MAKERS pieces that hold every thread until released, then panic, and
        // one more that waits for a thread.
        let mut first: Vec<Joined<usize>> = (0..MAKERS)
            .map(|at| {
                let (gate, started) = (Arc::clone(&gate), started.clone());
                let piece: Joined<usize> = Box::new(move || {
                    started.send(()).unwrap();
                    gate.lock().unwrap().recv_timeout(GUARD).unwrap();
                    panic!("piece {at} failed");
                });
                piece
            })
            .collect();
        first.push(Box::new(|| MAKERS));
        let first = {
            let pool = Arc::clone(&pool);
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                pool.join_all(first, false, |_, _| {})
            })
            .unwrap()
        };
        for _ in 0..MAKERS {
            running.recv_timeout(GUARD).unwrap();
        }
        // The second set, every piece waiting for a thread.
        let (handed, all_handed) = mpsc::channel();
        let second: Vec<Joined<usize>> = (0..3)
            .map(|at| {
                let piece: Joined<usize> = Box::new(move || at);
                piece
            })
            .collect();
        let second = {
            let pool = Arc::clone(&pool);
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                pool.join_all(second, false, |_, _| handed.send(()).unwrap())
            })
            .unwrap()
        };
        for _ in 0..3 {
            all_handed.recv_timeout(GUARD).unwrap();
        }
        assert_eq!(
            pool.census(),
            (MAKERS, 0),
            "every thread busy, the pieces waiting"
        );
        for _ in 0..MAKERS {
            release.send(()).unwrap();
        }
        assert!(
            first.join().is_err(),
            "the panicked set's caller panicked in turn"
        );
        assert_eq!(
            second.join().unwrap(),
            vec![Some(0), Some(1), Some(2)],
            "the other set ran every piece"
        );
    }

    /// What a joined piece owns (a creation's piece: its variant, its key, its key's pass) lives
    /// while its work runs and is dropped before the join returns, wherever the piece ran: the
    /// caller's own piece and the pool's alike.
    #[test]
    fn a_joined_piece_s_captures_live_through_its_work_and_go_before_the_join_returns() {
        let pool = Pool::default();
        let owned = Arc::new(());
        let pieces: Vec<Joined<usize>> = (0..MAKERS + 1)
            .map(|_| {
                let owned = Arc::clone(&owned);
                let piece: Joined<usize> = Box::new(move || {
                    std::thread::sleep(Duration::from_millis(20));
                    Arc::strong_count(&owned)
                });
                piece
            })
            .collect();
        let seen: Vec<usize> = pool
            .join_all(pieces, false, |_, _| {})
            .into_iter()
            .map(Option::unwrap)
            .collect();
        assert!(
            seen.iter().all(|&holders| holders >= 2),
            "each piece's capture alive through its work: {seen:?}"
        );
        assert_eq!(
            Arc::strong_count(&owned),
            1,
            "every piece's captures dropped before the join returned"
        );
    }

    /// A piece runs on a thread at the user-initiated quality of service, whatever the class of
    /// the thread that handed it and so started the pool's thread: exactly user-initiated from a
    /// lower class, at least that from the user-interactive class.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_piece_runs_at_the_user_initiated_class_whoever_started_its_thread() {
        use libc::qos_class_t as Q;
        fn class() -> u32 {
            let mut class = Q::QOS_CLASS_UNSPECIFIED;
            let mut relative = 0;
            // SAFETY: the calling thread's own handle, and two out-parameters that live for the
            // call.
            let read = unsafe {
                libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
            };
            assert_eq!(read, 0, "pthread_get_qos_class_np");
            class as u32
        }
        for start in [
            Q::QOS_CLASS_BACKGROUND,
            Q::QOS_CLASS_UTILITY,
            Q::QOS_CLASS_DEFAULT,
            Q::QOS_CLASS_USER_INITIATED,
            Q::QOS_CLASS_USER_INTERACTIVE,
        ] {
            // A thread started as the runtime starts its own, standing for the program's: its
            // class is set after the start.
            let ran = crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                // SAFETY: sets the calling thread's own class; no pointer.
                assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
                let pool = roomy();
                let (sent, got) = mpsc::channel();
                pool.run(Box::new(move || sent.send(class()).unwrap()));
                got.recv_timeout(GUARD).unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
            if start as u32 == Q::QOS_CLASS_USER_INTERACTIVE as u32 {
                assert!(
                    [Q::QOS_CLASS_USER_INITIATED, Q::QOS_CLASS_USER_INTERACTIVE]
                        .iter()
                        .any(|&at| at as u32 == ran),
                    "from {start:?}: {ran:#x}"
                );
            } else {
                assert_eq!(ran, Q::QOS_CLASS_USER_INITIATED as u32, "from {start:?}");
            }
        }
    }
}
