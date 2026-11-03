//! The device's handler threads: each logical queue's serial context, run on a pool of threads, as
//! Metal runs its queues' handlers.
//!
//! - **What a context runs.** In its queue's order, each command buffer's transitions: to
//!   `Scheduled` with its scheduled handlers, and to `Completed` (or `Error`) with its completed
//!   handlers and the release of what it retained. The scheduler posts them ([`Contexts::post`]),
//!   and so does a completion turn, but for a transition that runs none of the program's code: the
//!   turn runs such a transition to `Completed` itself where the context runs on no thread and
//!   holds only such transitions, in order after them ([`Contexts::run_here`],
//!   `command_buffer/completion.rs`), and the scheduler such a transition to `Scheduled` where the
//!   context runs on no thread and holds nothing ([`Contexts::run_alone`],
//!   `CommandBufferShared::schedule`); neither runs a handler. (A transition to `Scheduled` waits
//!   for its command buffer's submission handlers, which the thread that took its last batch for
//!   its queue runs.) A release the scheduler's thread hands off (`Device::let_go_elsewhere`) runs
//!   in a context of its own, behind no queue's handlers.
//! - **Metal's order** (measured on an M4, macOS 27.0.1): while one of a
//!   queue's handlers blocks, the queue's later transitions wait, even of a command buffer whose
//!   work already ran; another queue's transitions and handlers go on, on another thread. Metal
//!   ran 256 queues' handlers blocked at once, on 256 threads, and reached no limit. On the M3
//!   Max under macOS 27.0.1 it ran 511 queues' handlers blocked at once in a wait for another
//!   queue's command buffer, and those waits returned once it completed; with 512 blocked so (the
//!   process's 512 dispatch threads), no handler ran after and the command buffer they waited for
//!   never completed: a hang.
//! - **Threads.** One thread starts with the device and stays. Another starts when a context is
//!   ready and no thread is free, up to [`THREADS`]; one that found nothing to run for [`IDLE`]
//!   ends. The cap is Metal's 512. Each has a stack of [`STACK`] bytes.
//! - **Waits that would deadlock** ([`block`]). A handler's wait for another queue's command
//!   buffer blocks its thread. It is refused when it would close a cycle of such waits (across
//!   every device) or leave the device's pool with no thread that is not blocked so, the 512th,
//!   where Metal's hangs; then at least one thread can always run the contexts the blocked waits
//!   wait for.
//! - **Stopping.** The device's drop stops the threads, which end once no item is left: a
//!   transition holds its command buffer, which holds the device, so none is left by then; a
//!   release may hold nothing of the device, and runs before its thread ends. A drop on one of the
//!   threads (its item held the last reference) does not wait for itself.
//!
//! - **Woken only where one sleeps.** A post wakes a thread only where a line is ready and a
//!   thread sleeps waiting for one (`State::parked`), read under the state's lock: a thread that
//!   sleeps later looks at the lines first. A post to a context a thread is running, or while
//!   every thread runs, so costs no system call (a notify is one, sleeper or not). Counted:
//!   `WorkCounts::wakes_handler_threads`, `WorkCounts::wakes_spared`.
//!
//! Locks: a pool's state, then its census; the waits' graph, then a census. No call made under
//! either takes the state again.

pub(crate) mod lines;

use super::command_buffer::own::{Context, Waits};
use super::device::lock;
use super::error::{DeadlockCause, Error, MemoryKind};
use super::exit_guard;
use lines::Lines;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// The most threads a device's pool runs at once: Metal's 512 dispatch threads, of which 511
/// handlers blocked at once in waits for other queues and their waits returned, where 512 hung
/// (the module docs).
pub(crate) const THREADS: usize = 512;

/// Each thread's stack, in bytes.
pub(crate) const STACK: usize = 512 * 1024;

/// How long a thread other than the first finds nothing to run before it ends.
const IDLE: Duration = Duration::from_secs(2);

/// An item of a queue's context: run once, on one of the device's handler threads, or on the
/// thread that runs the context itself ([`Contexts::run_here`]).
pub(crate) trait Run: Send + 'static {
    /// Run the item; dropping it lets go of what it holds.
    fn run(self);

    /// Whether the item runs none of the program's code (no handler, no notice, no release of
    /// the program's), so that a thread that may run none can run it: none by default.
    fn plain(&self) -> bool {
        false
    }

    /// Whether the item may run on a runtime thread that saw its command buffer complete: it
    /// runs none of the program's code but its completion-seen handlers
    /// (`CommandBuffer::add_completion_seen_handler`). Every plain item does.
    fn runs_where_seen(&self) -> bool {
        self.plain()
    }
}

/// One device's handler threads. A clone is another handle of the same pool, which holds nothing
/// of the device.
pub(crate) struct Contexts<I: Run> {
    shared: Arc<Shared<I>>,
}

impl<I: Run> Clone for Contexts<I> {
    fn clone(&self) -> Contexts<I> {
        Contexts {
            shared: Arc::clone(&self.shared),
        }
    }
}

struct Shared<I> {
    state: Mutex<State<I>>,
    /// Signaled when a line becomes ready, where a thread sleeps, or the pool stops.
    wake: Condvar,
    census: Arc<Census>,
    /// The notifies made, and those spared as no thread slept or no line was ready.
    woke: AtomicU64,
    spared: AtomicU64,
    /// The identity the next context takes ([`Contexts::fresh`]).
    next: AtomicU64,
}

struct State<I> {
    lines: Lines<I>,
    /// The threads that run no item: waiting for one, or starting.
    free: usize,
    /// The threads sleeping on the condition variable, waiting for a line.
    parked: usize,
    stopping: bool,
}

/// What a pool counts, shared with the waits its threads block in.
pub(crate) struct Census {
    count: Mutex<Count>,
    /// Signaled when a thread ends.
    gone: Condvar,
}

struct Count {
    /// The threads alive.
    threads: usize,
    /// The threads blocked in a wait for another queue's command buffer.
    blocked: usize,
    /// The most threads: [`THREADS`], or the count reached when the host refused another.
    limit: usize,
}

thread_local! {
    /// The census of the pool this thread belongs to, on a handler thread.
    static POOL: RefCell<Option<Arc<Census>>> = const { RefCell::new(None) };
}

/// The waits of every handler thread for another queue's context.
static WAITS: Mutex<Waits> = Mutex::new(Waits::new());

impl<I: Run> Contexts<I> {
    /// A pool of at most `limit` threads (the device's: [`THREADS`]), with its first thread;
    /// refused when the host cannot start it.
    pub(crate) fn start(limit: usize) -> Result<Contexts<I>, Error> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                lines: Lines::default(),
                free: 1,
                parked: 0,
                stopping: false,
            }),
            wake: Condvar::new(),
            woke: AtomicU64::new(0),
            spared: AtomicU64::new(0),
            next: AtomicU64::new(0),
            census: Arc::new(Census {
                count: Mutex::new(Count {
                    threads: 1,
                    blocked: 0,
                    limit,
                }),
                gone: Condvar::new(),
            }),
        });
        spawn(&shared, true).map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        Ok(Contexts { shared })
    }

    /// A context's identity, each call a new one: a logical queue's, or a release's of its own
    /// (`Device::let_go_elsewhere`).
    pub(crate) fn fresh(&self) -> u64 {
        self.shared.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Append `item` to the context of logical queue `queue`; start a thread when a context is
    /// ready that no free thread will take, below the cap.
    pub(crate) fn post(&self, queue: u64, item: I) {
        let mut state = lock(&self.shared.state);
        state.lines.push(queue, item);
        self.serve(state);
    }

    /// Run `item` in the context of logical queue `queue` on this thread, after the items that
    /// context holds, where no thread runs that context and `may_run` holds for those items and
    /// `item` ([`plain`](Run::plain), or, on a thread that may run completion-seen handlers,
    /// [`runs_where_seen`](Run::runs_where_seen)): the context runs on this thread meanwhile, so
    /// the items posted to it wait for `item`, then run on the pool's threads. Else give `item`
    /// back, changing nothing.
    pub(crate) fn run_here(&self, queue: u64, item: I, may_run: fn(&I) -> bool) -> Result<(), I> {
        if !may_run(&item) {
            return Err(item);
        }
        let Some(before) = lock(&self.shared.state).lines.take_plain(queue, may_run) else {
            return Err(item);
        };
        for earlier in before {
            earlier.run();
        }
        item.run();
        let mut state = lock(&self.shared.state);
        state.lines.done(queue);
        self.serve(state);
        Ok(())
    }

    /// Run `item` in the context of logical queue `queue` on this thread, where no thread runs
    /// that context, it holds no item, and `item` is [`plain`](Run::plain): the context runs on
    /// this thread meanwhile, so the items posted to it wait for `item`, then run on the pool's
    /// threads. Else give `item` back, changing nothing.
    pub(crate) fn run_alone(&self, queue: u64, item: I) -> Result<(), I> {
        if !item.plain() || !lock(&self.shared.state).lines.take_idle(queue) {
            return Err(item);
        }
        item.run();
        let mut state = lock(&self.shared.state);
        state.lines.done(queue);
        self.serve(state);
        Ok(())
    }

    /// Start a thread where a context is ready that no free thread will take, below the cap, and
    /// wake one where one sleeps and a context is ready: `state` was just changed.
    fn serve(&self, mut state: std::sync::MutexGuard<'_, State<I>>) {
        if state.lines.ready() > state.free {
            let start = {
                let mut count = lock(&self.shared.census.count);
                let start = count.threads < count.limit;
                if start {
                    count.threads += 1;
                }
                start
            };
            if start {
                if spawn(&self.shared, false).is_ok() {
                    state.free += 1;
                } else {
                    // The host's threads ran out: the pool keeps those it has, and so does the
                    // cap on blocked waits.
                    let mut count = lock(&self.shared.census.count);
                    count.threads -= 1;
                    count.limit = count.threads;
                }
            }
        }
        let wake = state.parked > 0 && state.lines.is_ready();
        drop(state);
        if wake {
            self.shared.woke.fetch_add(1, Ordering::Relaxed);
            self.shared.wake.notify_one();
        } else {
            self.shared.spared.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The notifies made to the threads, and those spared.
    pub(crate) fn wakes(&self) -> (u64, u64) {
        (
            self.shared.woke.load(Ordering::Relaxed),
            self.shared.spared.load(Ordering::Relaxed),
        )
    }

    /// Stop the threads and wait for them to end, but for this one when it is one of them: each
    /// runs the items left first. The device is being dropped, so no transition is left.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.wake.notify_all();
        let census = &self.shared.census;
        let own = POOL.with(|pool| {
            pool.borrow()
                .as_ref()
                .is_some_and(|pool| Arc::ptr_eq(pool, census))
        });
        let floor = usize::from(own);
        let mut count = lock(&census.count);
        while count.threads > floor {
            count = census
                .gone
                .wait(count)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The threads alive and the threads blocked in a wait.
    #[cfg(test)]
    pub(crate) fn census(&self) -> (usize, usize) {
        let count = lock(&self.shared.census.count);
        (count.threads, count.blocked)
    }
}

/// Start one of `shared`'s threads (counted already); `first` stays until the pool stops.
fn spawn<I: Run>(shared: &Arc<Shared<I>>, first: bool) -> std::io::Result<()> {
    let shared = Arc::clone(shared);
    exit_guard::spawn_user_interactive(
        std::thread::Builder::new()
            .name("metal2vulkan-handlers".into())
            .stack_size(STACK),
        move || run(&shared, first),
    )
    .map(drop)
}

/// A thread's loop: run the ready contexts' items, one at a time.
fn run<I: Run>(shared: &Shared<I>, first: bool) {
    POOL.with(|pool| *pool.borrow_mut() = Some(Arc::clone(&shared.census)));
    let mut state = lock(&shared.state);
    loop {
        if let Some((queue, item)) = state.lines.take() {
            state.free -= 1;
            drop(state);
            item.run();
            state = lock(&shared.state);
            state.lines.done(queue);
            state.free += 1;
            continue;
        }
        if state.stopping {
            break;
        }
        state.parked += 1;
        let (next, waited) = shared
            .wake
            .wait_timeout(state, IDLE)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next;
        state.parked -= 1;
        if waited.timed_out() && !first && !state.lines.is_ready() && !state.stopping {
            break;
        }
    }
    state.free -= 1;
    drop(state);
    POOL.with(|pool| *pool.borrow_mut() = None);
    let mut count = lock(&shared.census.count);
    count.threads -= 1;
    drop(count);
    shared.census.gone.notify_all();
}

/// Whether this thread is one of a pool's handler threads.
pub(crate) fn on_handler_thread() -> bool {
    POOL.with(|pool| pool.borrow().is_some())
}

/// A handler thread's wait for another queue's context, counted until dropped.
pub(crate) struct Blocked {
    census: Arc<Census>,
    from: Context,
    to: Context,
}

/// Count a wait of this handler thread, which runs `from`'s item, for `to`'s context. Refused,
/// with its cause: when it would close a cycle of such waits (`Cycle`), and when it would leave
/// this thread's pool with no thread that is not blocked so, or this is not a handler thread
/// (`HandlerThreadCap`).
pub(crate) fn block(from: Context, to: Context) -> Result<Blocked, DeadlockCause> {
    let census = POOL
        .with(|pool| pool.borrow().clone())
        .ok_or(DeadlockCause::HandlerThreadCap)?;
    let mut waits = lock(&WAITS);
    if waits.closes_cycle(from, to) {
        return Err(DeadlockCause::Cycle);
    }
    {
        let mut count = lock(&census.count);
        if count.blocked + 1 >= count.limit {
            return Err(DeadlockCause::HandlerThreadCap);
        }
        count.blocked += 1;
    }
    waits.insert(from, to);
    Ok(Blocked { census, from, to })
}

impl Drop for Blocked {
    fn drop(&mut self) {
        lock(&WAITS).remove(self.from, self.to);
        lock(&self.census.count).blocked -= 1;
    }
}
