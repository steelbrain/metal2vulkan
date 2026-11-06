//! The commits that wait for a render encoder's end the device's lowering thread has not lowered
//! (`RenderLoweringPolicy::Worker`, `render/deferral.rs`), and those made behind them.
//!
//! - **Left at the call.** A `commit` whose command buffer holds such an end makes Metal's checks
//!   at the call, of the command buffer as the program sees it (its last encoder ended), and
//!   returns (`CommandBuffer::queue_commit`): its status reads `Committed` from then on, and a
//!   second commit, a handler added and anything encoded are refused. The commit's own work waits
//!   here, and runs once the end is lowered, on the thread that lowered it
//!   (`CommandBuffer::commit_queued`). A `commit_and_submit` left so has that thread make its
//!   pass too, as the call would have (`Scheduler::submit_at_commit`): the lowering thread then
//!   submits what is ready and runs the submission checks and handlers of what it submits, so the
//!   commit needs no wake of the scheduler's thread. A render encoder opened there (by such a
//!   check or handler) lowers its calls at their calls (`render/deferral.rs`), as one that handed
//!   them to that thread would wait for itself.
//! - **In commit order.** While a commit waits here, every later commit of the device's command
//!   buffers waits behind it, ready at once where its command buffer holds no end in flight. The
//!   device's scheduler so takes them in commit order across all its queues, as it takes commits
//!   made inside their calls: no binary semaphore wait or event of a later one passes an earlier
//!   one.
//! - **One runner.** The thread that makes the first commit waiting here ready runs it, and every
//!   one ready behind it, in order; a commit made ready meanwhile is left to that runner. The
//!   lowering thread runs each inside a section of the exit guard, as the program's call
//!   (`exit_guard::lowering_program_calls`), and lets its handle go off that thread
//!   (`CommandBuffer::let_go_off_thread`); once the process exits it runs none, and what waits
//!   here stays, as the scheduler's held work does.
//! - **Waits drive them.** A wait for a command buffer whose commit waits here lowers, on the
//!   waiting thread, the ends its commit and those before it wait for, which the thread has not
//!   (`CommandBuffer::settle_commit`).
//! - **Settled before a rebind.** A rebind or detach of host pages, and a read of the work
//!   counts, lower every end and then wait until no commit waits here
//!   ([`QueuedCommits::settle`]), running them on their own thread where no thread runs them:
//!   they then meet every command buffer committed before them as committed, as inside the
//!   commits' calls. Not on the thread that runs them (a message sink that reads the counts),
//!   which would wait for itself.

use crate::command_buffer::{CommandBuffer, Waiting};
use crate::device::lock;
use crate::exit_guard;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

thread_local! {
    /// This thread runs a device's commits that wait (the module docs).
    static RUNNING: Cell<bool> = const { Cell::new(false) };
}

/// A device's commits that wait (the module docs), in commit order.
pub(crate) struct QueuedCommits {
    queued: Mutex<Queued>,
    /// Signalled as the thread that runs them stops.
    stopped: Condvar,
}

struct Queued {
    commits: VecDeque<QueuedCommit>,
    /// A thread runs the ready commits at the head.
    running: bool,
}

struct QueuedCommit {
    /// Its command buffer, as the commits that wait hold it (`CommandBuffer::waiting`).
    command_buffer: Arc<dyn Waiting>,
    /// Its command buffer holds no end the lowering thread has not lowered.
    ready: bool,
}

impl Queued {
    /// Whether the caller is to run the commits now: the first is ready and no thread runs
    /// them; that thread is the caller from now on.
    fn take_run(&mut self) -> bool {
        let runs = !self.running && self.commits.front().is_some_and(|commit| commit.ready);
        self.running |= runs;
        runs
    }
}

/// The device's commits that wait, locked by a commit deciding whether it waits too.
pub(crate) struct QueuedLock<'a>(MutexGuard<'a, Queued>);

impl QueuedLock<'_> {
    /// No commit waits, and no thread runs one.
    pub(crate) fn idle(&self) -> bool {
        self.0.commits.is_empty() && !self.0.running
    }

    /// Leave `command_buffer`'s commit to wait, last, `ready` where its command buffer holds no
    /// end in flight: whether the caller is to run the commits now ([`QueuedCommits::run`]).
    pub(crate) fn push(mut self, command_buffer: CommandBuffer, ready: bool) -> bool {
        self.0.commits.push_back(QueuedCommit {
            command_buffer: command_buffer.waiting(),
            ready,
        });
        self.0.take_run()
    }
}

/// The runner's turn: a panic that unwinds it leaves the commits to the next thread that makes
/// one ready.
struct Turn<'a>(&'a QueuedCommits);

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        RUNNING.with(|running| running.set(false));
        if std::thread::panicking() {
            self.0.stop(&mut lock(&self.0.queued));
        }
    }
}

impl QueuedCommits {
    /// None waits.
    pub(crate) fn new() -> QueuedCommits {
        QueuedCommits {
            queued: Mutex::new(Queued {
                commits: VecDeque::new(),
                running: false,
            }),
            stopped: Condvar::new(),
        }
    }

    /// Locked: taken under a command buffer's state, never the other way round.
    pub(crate) fn lock(&self) -> QueuedLock<'_> {
        QueuedLock(lock(&self.queued))
    }

    /// The thread that runs them stops, under `queued`, the lock.
    fn stop(&self, queued: &mut Queued) {
        queued.running = false;
        self.stopped.notify_all();
    }

    /// `command_buffer`'s end was lowered: its commit, which waits, is ready. Whether the caller
    /// is to run the commits now ([`run`](Self::run)).
    pub(crate) fn ready(&self, command_buffer: &CommandBuffer) -> bool {
        let mut queued = lock(&self.queued);
        if let Some(commit) =
            (queued.commits.iter_mut()).find(|commit| command_buffer.is(&*commit.command_buffer))
        {
            commit.ready = true;
        }
        queued.take_run()
    }

    /// Run the ready commits at the head, in order, until none is left or the first is not
    /// ready: on the device's lowering thread (`by_thread`), each inside a section of the exit
    /// guard and as the program's call, or on the program's (the module docs).
    pub(crate) fn run(&self, by_thread: bool) {
        let _turn = Turn(self);
        RUNNING.with(|running| running.set(true));
        loop {
            let section = match by_thread {
                true => match exit_guard::enter() {
                    Some(section) => Some(section),
                    None => {
                        self.stop(&mut lock(&self.queued));
                        return;
                    }
                },
                false => None,
            };
            let next = {
                let mut queued = lock(&self.queued);
                let ready = queued.commits.front().is_some_and(|commit| commit.ready);
                let next = ready.then(|| queued.commits.pop_front()).flatten();
                if next.is_none() {
                    self.stop(&mut queued);
                }
                next
            };
            let Some(QueuedCommit { command_buffer, .. }) = next else {
                return;
            };
            let command_buffer = command_buffer.command_buffer();
            match by_thread {
                true => {
                    exit_guard::lowering_program_calls(|| command_buffer.commit_queued(true));
                    drop(section);
                    command_buffer.let_go_off_thread();
                }
                false => command_buffer.commit_queued(false),
            }
        }
    }

    /// The command buffers whose commits wait for an end, of those up to `command_buffer`'s, in
    /// order: none where its commit no longer waits.
    pub(crate) fn unready_through(&self, command_buffer: &CommandBuffer) -> Vec<CommandBuffer> {
        let queued = lock(&self.queued);
        let Some(at) =
            (queued.commits.iter()).position(|commit| command_buffer.is(&*commit.command_buffer))
        else {
            return Vec::new();
        };
        (queued.commits.iter().take(at + 1))
            .filter(|commit| !commit.ready)
            .map(|commit| Arc::clone(&commit.command_buffer).command_buffer())
            .collect()
    }

    /// Wait until no commit waits, the caller having drained every slot: run them on this thread
    /// where no thread runs them, else wait for the one that does, or for the lowering thread to
    /// make the first ready. Nothing on the thread that runs them, which would wait for itself,
    /// nor once the process exits (the module docs).
    pub(crate) fn settle(&self) {
        if RUNNING.with(Cell::get) {
            return;
        }
        let mut queued = lock(&self.queued);
        loop {
            if queued.commits.is_empty() && !queued.running {
                return;
            }
            if queued.take_run() {
                drop(queued);
                self.run(false);
                queued = lock(&self.queued);
                continue;
            }
            // The first waits for an end its thread lowers now, or one handed since the drain,
            // which that thread lowers next; once the process exits, none is.
            if exit_guard::closed() {
                return;
            }
            queued = (self.stopped.wait_timeout(queued, exit_guard::WAKE))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }
}
