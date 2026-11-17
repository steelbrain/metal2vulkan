//! The reaper thread: it frees what the runtime's other threads give up, one object at a time, and
//! runs a teardown that starts on the completion thread. No driver free runs on the completion
//! thread, so no completion waits behind one, and a submission on another thread waits behind at
//! most one free.

use crate::counters::Gauge;
use crate::device::{lock, Core};
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

/// The reaper's queue.
pub(crate) struct Reaper {
    state: Mutex<State>,
    changed: Condvar,
}

struct State {
    jobs: VecDeque<Job>,
    /// The reaper ends once the queue is empty.
    stop: bool,
}

pub(crate) enum Job {
    Free(Free),
    /// The device's teardown, after its last handle dropped on the completion thread.
    Teardown(Teardown),
}

/// A teardown the reaper runs.
pub(crate) type Teardown = Box<dyn FnOnce(&Arc<Core>) + Send>;

/// An object whose last user is gone.
pub(crate) enum Free {
    /// An event's timeline semaphore; the events gauge falls with it.
    Semaphore(vk::Semaphore),
    /// A command pool past the idle ones the device keeps, or one whose reset failed.
    CommandPool(vk::CommandPool),
    /// A dropped queue's timestamp queries.
    QueryPool(vk::QueryPool),
    /// A pipeline of a dropped pipeline state; the pipelines gauge falls with it.
    Pipeline(vk::Pipeline),
    /// The shader module of a dropped pipeline state.
    ShaderModule(vk::ShaderModule),
}

impl Reaper {
    pub(crate) fn new() -> Reaper {
        Reaper {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                stop: false,
            }),
            changed: Condvar::new(),
        }
    }

    /// Queues a job. Once the reaper has ended, a free waits for the core's drop.
    pub(crate) fn give(&self, job: Job) {
        lock(&self.state).jobs.push_back(job);
        self.changed.notify_one();
    }

    /// Tells the reaper to end once its queue is empty.
    pub(crate) fn stop(&self) {
        lock(&self.state).stop = true;
        self.changed.notify_one();
    }

    /// At the core's drop, the reaper having ended: frees what is left. No teardown is left: one
    /// holds blocks, and so the core.
    pub(crate) fn drain(&self, core: &Core) {
        let jobs = std::mem::take(&mut lock(&self.state).jobs);
        for job in jobs {
            if let Job::Free(object) = job {
                free(core, object);
            }
        }
    }
}

/// Frees one object.
fn free(core: &Core, object: Free) {
    let device = &core.device;
    match object {
        Free::Semaphore(semaphore) => {
            // SAFETY: no pending work names it: its event's last user, which held it until its
            // work retired, is gone.
            unsafe { device.destroy_semaphore(semaphore, None) };
            core.counters.fall(Gauge::Events);
        }
        // SAFETY: no command buffer of the pool is pending: its work retired, or never ran.
        Free::CommandPool(pool) => unsafe { device.destroy_command_pool(pool, None) },
        Free::QueryPool(pool) => {
            lock(&core.query_pools).retain(|&live| live != pool);
            // SAFETY: every command buffer of the queue retired or never ran.
            unsafe { device.destroy_query_pool(pool, None) };
        }
        Free::Pipeline(pipeline) => {
            // SAFETY: every command buffer that used it retired: each kept its state until then.
            unsafe { device.destroy_pipeline(pipeline, None) };
            core.counters.fall(Gauge::Pipelines);
        }
        // SAFETY: its pipelines are made, and no more will be.
        Free::ShaderModule(module) => unsafe { device.destroy_shader_module(module, None) },
    }
}

/// The reaper thread, until it is stopped and its queue is empty.
pub(crate) fn run(core: Arc<Core>) {
    let reaper = &core.reaper;
    loop {
        let job = {
            let mut state = lock(&reaper.state);
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    break job;
                }
                if state.stop {
                    return;
                }
                state = reaper
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        match job {
            Job::Free(object) => free(&core, object),
            Job::Teardown(teardown) => teardown(&core),
        }
    }
}
