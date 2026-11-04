//! The threads that record the batches' host steps (`scheduler/mod.rs`, "Host steps"), apart from
//! the scheduler's: a step records an execution's commands, which takes as long as the encoder's
//! calls of them would (milliseconds for thousands of draws, seconds for some), and makes what no
//! earlier settle made, while the scheduler's thread goes on submitting every other queue's ready
//! batches.
//!
//! - **A step a thread, up to [`THREADS`].** The scheduler's thread hands each step its passes
//!   find to these threads ([`post`]): a free one takes it, else one is started for it, below the
//!   cap, else it waits for the first to be free, in the order handed over. So one queue's long
//!   step holds back another queue's only where [`THREADS`] steps record at once. Each is recorded
//!   inside a section of the exit guard.
//! - **A serial point of its own stream.** Only a line's head is handed over, and the batch is
//!   marked as stepping from the hand-over until its step ended
//!   (`CommandBufferShared::begin_step`): it waits for its host step meanwhile, no pass hands it
//!   over again, and its line's later batches wait behind it, as does another queue's later
//!   batch that Metal orders after it (`order.rs`). So a logical queue's steps are recorded one
//!   at a time, in its order, and two recorded at once share nothing Metal orders.
//! - **Then it wakes the scheduler**, before it lets go of the command buffer, whose device the
//!   scheduler's thread serves: the next pass submits the batch in its order. Once the device
//!   stopped the scheduler (`Scheduler::stop`), it wakes nothing.
//! - **Started at the first step** a device's scheduler hands over, each at the user-interactive
//!   class (`exit_guard::spawn_user_interactive`), as the scheduler's own thread is; where the host
//!   cannot start the first, the scheduler's thread records the step itself, and where it cannot
//!   start a later one, the step waits for one that runs. The first ends once the scheduler
//!   stopped and no step is left, or once the process is exiting; each later one ends too once it
//!   found no step for [`IDLE`].
//! - Counted: `WorkCounts::host_steps_recorded` and `WorkCounts::host_step_nanos`.

use super::{SchedulerShared, Woken, Work};
use crate::device::lock;
use crate::exit_guard;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The most threads a device's steps record on at once (the module docs): as many logical queues'
/// steps go on together, each queue's one at a time.
const THREADS: usize = 8;

/// How long a thread other than the first finds no step before it ends.
const IDLE: Duration = Duration::from_secs(2);

/// The steps handed over and not yet taken, and the threads that record them.
#[derive(Default)]
pub(super) struct Steps {
    state: Mutex<StepsState>,
    posted: Condvar,
}

#[derive(Default)]
struct StepsState {
    work: VecDeque<Work>,
    /// The threads alive.
    threads: usize,
    /// Those of them waiting for a step.
    free: usize,
    stopping: bool,
}

impl Steps {
    /// The scheduler stopped: the threads end once no step is left.
    pub(super) fn stop(&self) {
        lock(&self.state).stopping = true;
        self.posted.notify_all();
    }
}

/// Hand `work`'s host step to the steps' threads, one started now where none is free for it,
/// below the cap (the module docs). Where the host cannot start the first, the step is recorded
/// here. False once the process is exiting, with nothing recorded.
pub(super) fn post(shared: &Arc<SchedulerShared>, work: Work) -> bool {
    let (buffer, index) = &work;
    buffer.begin_step(*index);
    let mut state = lock(&shared.steps.state);
    if state.free <= state.work.len() && state.threads < THREADS {
        let thread = Arc::clone(shared);
        let first = state.threads == 0;
        let started = exit_guard::spawn_user_interactive(
            std::thread::Builder::new().name("metal2vulkan-steps".into()),
            move || record_steps(&thread, first),
        );
        match started {
            Ok(_) => state.threads += 1,
            Err(_) if first => {
                drop(state);
                return record(shared, work);
            }
            // One that runs takes it.
            Err(_) => {}
        }
    }
    state.work.push_back(work);
    drop(state);
    shared.steps.posted.notify_one();
    true
}

/// A thread's loop (the module docs); `first` stays until the scheduler stops.
fn record_steps(shared: &SchedulerShared, first: bool) {
    loop {
        let work = {
            let mut state = lock(&shared.steps.state);
            let idle_until = Instant::now() + IDLE;
            loop {
                if let Some(work) = state.work.pop_front() {
                    break work;
                }
                let now = Instant::now();
                if state.stopping || (!first && now >= idle_until) {
                    state.threads -= 1;
                    return;
                }
                state.free += 1;
                let posted = &shared.steps.posted;
                state = match first {
                    true => posted.wait(state).unwrap_or_else(PoisonError::into_inner),
                    false => {
                        let waited = posted.wait_timeout(state, idle_until - now);
                        waited.unwrap_or_else(PoisonError::into_inner).0
                    }
                };
                state.free -= 1;
            }
        };
        if !record(shared, work) {
            lock(&shared.steps.state).threads -= 1;
            return;
        }
    }
}

/// Record `work`'s host step, a section of the exit guard, counted, and wake the scheduler's
/// thread, before the command buffer is let go of; false, with nothing recorded, once the process
/// is exiting.
fn record(shared: &SchedulerShared, work: Work) -> bool {
    let Some(busy) = exit_guard::enter() else {
        return false;
    };
    let (buffer, index) = &work;
    let started = Instant::now();
    buffer.host_step(*index);
    for &call in exit_guard::HOST_STEP {
        exit_guard::called(call);
    }
    let took = started.elapsed();
    buffer.device().shared().counts.host_step_recorded(took);
    buffer.end_step();
    drop(busy);
    let stopping = {
        let mut state = lock(shared.inbox.state());
        if !state.stopping {
            state.dirty = true;
        }
        state.stopping
    };
    if !stopping {
        shared.wake_thread(Woken::Host, shared.nudge.get());
    }
    // Out of the section: it may be the command buffer's last reference.
    drop(work);
    true
}
