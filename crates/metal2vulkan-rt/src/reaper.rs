//! The reaper thread: it makes the device's `VkPipelineCache` from its blob first, then frees
//! what the runtime's other threads give up, one object at a time, looks at the release of a
//! texture whose image an upload held past it (`crate::submit::Work::release`), runs a teardown
//! that starts in a handler or a listener, writes the lowerings the pipeline states hand it to
//! the store, and looks at the pipeline cache once creation is quiet, writing its blob or
//! replacing it past its bound (`crate::cache`). No driver free runs on the completion thread,
//! so no completion waits behind one. Device memory it does not free: a texture's allocation of
//! its own, a surface plane's import and a copy's staging buffer go to the block-maker thread,
//! whose form of every free a submission waits out at most one of (`crate::submit_calls`).

use crate::counters::Gauge;
use crate::device::{lock, Core};
use crate::memory::Loose;
use crate::submit::Held;
use crate::texture::{Backing, Image};
use crate::times::host_time;
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
    /// The reaper ends once the queue is empty.
    stop: bool,
}

pub(crate) enum Job {
    Free(Free),
    /// A released texture's image, of its bytes, which an upload held past the texture: the
    /// release is looked at under the work lock, which the texture's drop may run under
    /// ([`Work::release`](crate::submit::Work::release)). Nothing where the image is gone.
    Released(Weak<Image>, u64),
    /// The device's teardown, after its last handle dropped on the completion thread or the
    /// submit thread, in a handler or a listener.
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
    /// A pipeline of a dropped pipeline state; the pipelines gauge falls with it.
    Pipeline(vk::Pipeline),
    /// The shader module of a dropped pipeline state.
    ShaderModule(vk::ShaderModule),
    /// A dropped texture's views, image and memory: its place is freed, or its allocation of its
    /// own with the blocks gauge; a drawable's views alone, its image being the consumer's
    /// swapchain's. The textures gauge falls with it.
    Texture {
        views: [Option<vk::ImageView>; 2],
        image: vk::Image,
        backing: Backing,
    },
    /// A dropped texture view's image views; the textures gauge falls with them.
    Views([Option<vk::ImageView>; 2]),
    /// A dropped image's shadow (`crate::shadow`), its image and memory of `bytes`, freed as a
    /// texture's are; the shadow gauges fall with it.
    Shadow {
        image: vk::Image,
        backing: Backing,
        bytes: u64,
    },
    /// A dropped texture's view of its image's shadow.
    ShadowView(vk::ImageView),
    /// A dropped texture's attachment views; the attachment views gauge falls with each.
    AttachmentViews(Vec<vk::ImageView>),
    /// A device sampler no sampler state holds; the device samplers gauge falls with it.
    Sampler(vk::Sampler),
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

    /// Wakes the reaper: to see when to look at the pipeline cache, after the first creation since
    /// its last look, or to write a lowering handed to it (`cache::keep_translation`).
    pub(crate) fn poke(&self) {
        let _state = lock(&self.state);
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
            match job {
                Job::Free(object) => free(core, object),
                Job::Released(..) | Job::Teardown(_) => {}
            }
        }
    }
}

/// Gives up an image's memory: its place in a texture block, or its allocation of its own or its
/// import, which the block-maker thread frees ([`Loose`]).
///
/// # Safety
///
/// The image bound to it is destroyed.
unsafe fn free_backing(core: &Core, backing: Backing) {
    match backing {
        Backing::Own(memory, bytes, spilled) => {
            let loose = Loose::Texture {
                memory,
                bytes,
                spilled,
            };
            // SAFETY: the caller's contract: no work uses the memory, and its image is destroyed.
            unsafe { core.pool.give_back_loose(loose) };
        }
        Backing::Import(memory, bytes) => {
            // SAFETY: as above.
            unsafe {
                core.pool
                    .give_back_loose(Loose::SurfaceImport { memory, bytes })
            };
        }
        Backing::Placed(place) => core.places.release(core, place),
        Backing::InBlock | Backing::Drawable => {}
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
                // A drawable's image is the consumer's swapchain's, which destroys it.
                if !matches!(backing, Backing::Drawable) {
                    device.destroy_image(image, None);
                }
                // SAFETY: the image bound to it is destroyed.
                free_backing(core, backing);
            }
            core.counters.fall(Gauge::Textures);
        }
        Free::Views(views) => {
            // SAFETY: no pending work uses them: every command buffer that used the view held it
            // until it retired, and its elements of set 1 are null.
            unsafe {
                for view in views.into_iter().flatten() {
                    device.destroy_image_view(view, None);
                }
            }
            core.counters.fall(Gauge::Textures);
        }
        Free::Shadow {
            image,
            backing,
            bytes,
        } => {
            // SAFETY: no pending work uses it: every command buffer and upload that used it held
            // its image until it retired, and every view of it went before it (each texture's
            // goes before the image its texture holds).
            unsafe {
                device.destroy_image(image, None);
                free_backing(core, backing);
            }
            core.counters.fall(Gauge::Shadows);
            core.counters.fall_by(Gauge::ShadowBytes, bytes);
        }
        // SAFETY: no pending work uses it: every command buffer that read through it held its
        // texture until it retired, and its element of set 1 is null or another texture's.
        Free::ShadowView(view) => unsafe { device.destroy_image_view(view, None) },
        Free::AttachmentViews(views) => {
            // SAFETY: no pending work uses them: every command buffer whose pass rendered into
            // the texture held it until it retired.
            unsafe {
                for &view in &views {
                    device.destroy_image_view(view, None);
                }
            }
            core.counters
                .fall_by(Gauge::AttachmentViews, views.len() as u64);
        }
        Free::Sampler(sampler) => {
            // SAFETY: no element of set 1 holds it, and every command buffer that used a state
            // of it held the state until it retired.
            unsafe { device.destroy_sampler(sampler, None) };
            core.counters.fall(Gauge::DeviceSamplers);
        }
    }
}

/// What the reaper does next.
enum Next {
    Job(Job),
    /// Look at the pipeline cache, creation being quiet.
    Look,
    /// Write a lowering, refusal or record to the store.
    Write(String, Vec<u8>),
    /// End: the reaper was told to, and its queue is empty.
    End,
}

/// The reaper thread, until it is stopped and its queue is empty: the pipeline cache first.
pub(crate) fn run(core: Arc<Core>) {
    // SAFETY: the caches are the core's device's.
    unsafe { core.caches.seed(&core.device, &core.counters) };
    let reaper = &core.reaper;
    loop {
        let next = {
            let mut state = lock(&reaper.state);
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    break Next::Job(job);
                }
                // Before the end, so a teardown writes every lowering handed over.
                if let Some((file, bytes)) = core.caches.next_write() {
                    break Next::Write(file, bytes);
                }
                if state.stop {
                    break Next::End;
                }
                // A creation that ends after this read sets its mark and then pokes under the
                // lock, which this thread holds until it waits.
                state = match core.caches.look_in(host_time()) {
                    Some(due) if due.is_zero() => break Next::Look,
                    Some(due) => match reaper.changed.wait_timeout(state, due) {
                        Ok((state, _)) => state,
                        Err(poisoned) => poisoned.into_inner().0,
                    },
                    None => (reaper.changed.wait(state)).unwrap_or_else(PoisonError::into_inner),
                };
            }
        };
        match next {
            Next::Job(Job::Free(object)) => free(&core, object),
            Next::Job(Job::Released(image, bytes)) => {
                if let Some(image) = image.upgrade() {
                    // Where this is the image's last holder, it goes here, after the work lock.
                    let held = Held::Texture(image);
                    core.work.release(&core, &held, bytes);
                }
            }
            Next::Job(Job::Teardown(teardown)) => teardown(&core),
            // SAFETY: the caches are the core's device's.
            Next::Look => unsafe { core.caches.look(&core.device, &core.counters) },
            Next::Write(file, bytes) => core.caches.write_kept(&file, &bytes, &core.counters),
            Next::End => return,
        }
    }
}
