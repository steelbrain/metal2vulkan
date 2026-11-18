//! The reaper thread: it frees what the runtime's other threads give up, one object at a time, runs
//! a teardown that starts on the completion thread, lowers the pipeline states' emulated-SIMD-group
//! variants after their creation returned, and writes the pipeline data to the cache directory
//! when it is due. No driver free runs on the completion thread, so no completion waits behind
//! one, and a submission on another thread waits behind at most one free.

use crate::counters::Gauge;
use crate::device::{lock, Core};
use crate::pipeline::PipelineInner;
use crate::texture::Backing;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};

/// The reaper's queue.
pub(crate) struct Reaper {
    state: Mutex<State>,
    changed: Condvar,
}

struct State {
    jobs: VecDeque<Job>,
    /// The states whose emulated variant the reaper lowers, in order, after every job: no one waits
    /// for them, except for one an encoding call waits for, which goes first
    /// ([`Reaper::hurry`]).
    lowerings: VecDeque<Weak<PipelineInner>>,
    /// The reaper ends once the queue is empty, lowering nothing more.
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
    /// A dropped texture's views, image and memory: its place is freed, or its allocation of its
    /// own with the blocks gauge. The textures gauge falls with it.
    Texture {
        views: [Option<vk::ImageView>; 2],
        image: vk::Image,
        backing: Backing,
    },
    /// A device sampler no sampler state holds; the device samplers gauge falls with it.
    Sampler(vk::Sampler),
    /// A staging buffer of texture bytes, and its memory; the blocks gauge falls with it.
    Staging {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    },
}

impl Reaper {
    pub(crate) fn new() -> Reaper {
        Reaper {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                lowerings: VecDeque::new(),
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

    /// Queues the lowering of `state`'s emulated variant. Once the reaper has ended, the variant
    /// is not lowered.
    pub(crate) fn lower(&self, state: Weak<PipelineInner>) {
        let mut queue = lock(&self.state);
        if queue.stop {
            drop(queue);
            if let Some(state) = state.upgrade() {
                state.quiesce_emulated();
            }
            return;
        }
        queue.lowerings.push_back(state);
        drop(queue);
        self.changed.notify_one();
    }

    /// Puts `state`'s lowering first, for an encoding call that waits for it; nothing where the
    /// reaper took it already.
    pub(crate) fn hurry(&self, state: &Arc<PipelineInner>) {
        let mut queue = lock(&self.state);
        let lowerings = &mut queue.lowerings;
        if let Some(at) = lowerings
            .iter()
            .position(|queued| std::ptr::eq(queued.as_ptr(), Arc::as_ptr(state)))
        {
            if let Some(hurried) = lowerings.remove(at) {
                lowerings.push_front(hurried);
            }
        }
    }

    /// Wakes the reaper to time the pipeline data's next write.
    pub(crate) fn wake(&self) {
        // Taken so the wake cannot fall between the reaper's look at the caches and its wait.
        drop(lock(&self.state));
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
        Free::Texture {
            views,
            image,
            backing,
        } => {
            // SAFETY: no pending work uses them: every command buffer and upload that used the
            // texture held it until it retired, and its elements of set 1 are null.
            unsafe {
                for view in views.into_iter().flatten() {
                    device.destroy_image_view(view, None);
                }
                device.destroy_image(image, None);
            }
            match backing {
                Backing::Own(memory) => {
                    // SAFETY: the image bound to it is destroyed.
                    unsafe { device.free_memory(memory, None) };
                    core.counters.fall(Gauge::Blocks);
                }
                Backing::Placed(place) => core.places.release(core, place),
            }
            core.counters.fall(Gauge::Textures);
        }
        Free::Sampler(sampler) => {
            // SAFETY: no element of set 1 holds it, and every command buffer that used a state
            // of it held the state until it retired.
            unsafe { device.destroy_sampler(sampler, None) };
            core.counters.fall(Gauge::DeviceSamplers);
        }
        Free::Staging { buffer, memory } => {
            // SAFETY: the upload that copied through it retired, and no reader holds its mapping;
            // freeing the memory unmaps it.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            core.counters.fall(Gauge::Blocks);
        }
    }
}

/// What the reaper does next.
enum Next {
    Job(Job),
    /// Lower a state's emulated variant.
    Lower(Weak<PipelineInner>),
    /// Write the pipeline data.
    Save,
    /// End, after telling the states whose lowering is left that it will not run.
    End(VecDeque<Weak<PipelineInner>>),
}

/// The reaper thread, until it is stopped and its queue is empty. The device's end writes the
/// pipeline data a last time.
pub(crate) fn run(core: Arc<Core>) {
    let reaper = &core.reaper;
    loop {
        let next = {
            let mut state = lock(&reaper.state);
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    break Next::Job(job);
                }
                if state.stop {
                    break Next::End(std::mem::take(&mut state.lowerings));
                }
                if let Some(lowering) = state.lowerings.pop_front() {
                    break Next::Lower(lowering);
                }
                state = match core.caches.due_in() {
                    Some(wait) if wait.is_zero() => break Next::Save,
                    Some(wait) => {
                        reaper
                            .changed
                            .wait_timeout(state, wait)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0
                    }
                    None => reaper
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner),
                };
            }
        };
        match next {
            Next::Job(Job::Free(object)) => free(&core, object),
            Next::Job(Job::Teardown(teardown)) => teardown(&core),
            Next::Lower(lowering) => {
                if let Some(state) = lowering.upgrade() {
                    state.lower_emulated();
                }
            }
            Next::Save => core.caches.save(&core.device),
            Next::End(left) => {
                for state in left.iter().filter_map(Weak::upgrade) {
                    state.quiesce_emulated();
                }
                return;
            }
        }
    }
}
