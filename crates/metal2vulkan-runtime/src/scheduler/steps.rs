//! The thread that records the batches' host steps (`scheduler/mod.rs`, "Host steps"), apart from
//! the scheduler's: a step records an execution's commands, which takes as long as the encoder's
//! calls of them would (milliseconds for thousands of draws), and makes what no earlier settle
//! made, while the scheduler's thread goes on submitting every other queue's ready batches.
//!
//! - **Handed over in the order found.** The scheduler's thread hands each step its passes find to
//!   this thread ([`post`]), which records them one after another in that order, each inside a
//!   section of the exit guard: a host step stays a serial point of the command buffers' stream.
//!   The batch is marked as stepping from the hand-over until its step ended
//!   (`CommandBufferShared::begin_step`): it waits for its host step meanwhile, and no pass hands
//!   it over again.
//! - **Then it wakes the scheduler**, before it lets go of the command buffer, whose device the
//!   scheduler's thread serves: the next pass submits the batch in its order. Once the device
//!   stopped the scheduler (`Scheduler::stop`), it wakes nothing.
//! - **Started at the first step** a device's scheduler hands over, at the user-interactive class
//!   (`exit_guard::spawn_user_interactive`), as the scheduler's own thread is; where the host
//!   cannot start it, the scheduler's thread records the step itself. It ends
//!   once the scheduler stopped and no step is left, or once the process is exiting.
//! - Counted: `WorkCounts::host_steps_recorded` and `WorkCounts::host_step_nanos`.

use super::{SchedulerShared, Woken, Work};
use crate::device::lock;
use crate::exit_guard;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Instant;

/// The steps handed over and not yet recorded, and whether the thread runs.
#[derive(Default)]
pub(super) struct Steps {
    state: Mutex<StepsState>,
    posted: Condvar,
}

#[derive(Default)]
struct StepsState {
    work: VecDeque<Work>,
    started: bool,
    stopping: bool,
}

impl Steps {
    /// The scheduler stopped: the thread ends once no step is left.
    pub(super) fn stop(&self) {
        lock(&self.state).stopping = true;
        self.posted.notify_all();
    }
}

/// Hand `work`'s host step to the steps' thread, started now where it does not run (the module
/// docs). Where the host cannot start it, the step is recorded here. False once the process is
/// exiting, with nothing recorded.
pub(super) fn post(shared: &Arc<SchedulerShared>, work: Work) -> bool {
    let (buffer, index) = &work;
    buffer.begin_step(*index);
    let mut state = lock(&shared.steps.state);
    if !state.started {
        let thread = Arc::clone(shared);
        let started = exit_guard::spawn_user_interactive(
            std::thread::Builder::new().name("metal2vulkan-steps".into()),
            move || record_steps(&thread),
        );
        if started.is_err() {
            drop(state);
            return record(shared, work);
        }
        state.started = true;
    }
    state.work.push_back(work);
    drop(state);
    shared.steps.posted.notify_one();
    true
}

/// The thread's loop (the module docs).
fn record_steps(shared: &SchedulerShared) {
    loop {
        let work = {
            let mut state = lock(&shared.steps.state);
            loop {
                if let Some(work) = state.work.pop_front() {
                    break work;
                }
                if state.stopping {
                    return;
                }
                state = (shared.steps.posted)
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        if !record(shared, work) {
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
    {
        let mut state = lock(shared.inbox.state());
        if !state.stopping {
            let asleep = state.mark(shared.nudge.get());
            drop(state);
            shared.woken(Woken::Host, asleep);
        }
    }
    // Out of the section: it may be the command buffer's last reference.
    drop(work);
    true
}
