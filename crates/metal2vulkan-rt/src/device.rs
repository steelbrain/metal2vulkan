//! The device: one `VkDevice` for every command queue, every guest process and the compositor;
//! its queues, its memory blocks, its threads, quiesce and teardown.

use crate::complete::{self, Done};
use crate::counters::{Counters, Gauge};
use crate::error::Error;
use crate::extension::{self, DeviceExtension};
use crate::fill::Filler;
use crate::floor;
use crate::hazard::Ids;
use crate::instance::{Instance, PhysicalDevice};
use crate::memory::{self, Block, BlockId, BlockInfo, BlockMappings, MemoryClass, Pool, BLOCK_LEN};
use crate::pipeline::PUSH_BYTES;
use crate::reaper::{self, Job, Reaper};
use crate::record::Recorders;
use crate::ring::Ring;
use crate::submit::{self, Work};
use crate::times::Ticks;
use ash::vk;
use metal2vulkan::device::{
    ApiVersion, DeviceDescription, DevicePolicy, HandleDomain, PointerDomain,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// `MTLDevice`, over one `VkDevice` that every command queue, every guest process and the
/// compositor share.
///
/// A cheap handle: clones share one device, and every object made from it holds a clone, except
/// a committed command buffer. Blocks the consumer holds do not.
///
/// The runtime runs four threads: the block-maker, the submit thread, the completion thread,
/// which runs every handler and listener, and the reaper, which frees the semaphores and command
/// pools the others give up, one at a time, so no driver free runs on the completion thread. The
/// device is torn down when the last handle is dropped: the submit thread submits what is ready
/// and ends what it holds with [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun);
/// the completion thread retires the rest as the GPU completes it; the threads are joined; then
/// every block is freed, after [`BlockMappings::unmap`] for each the consumer was handed, and the
/// `VkDevice` is destroyed; every gauge reads zero only then. Where the last handle drops in a
/// handler or a listener, the reaper runs the teardown once the handler returns, and the consumer
/// waits for the gauges to read zero before it drops its instance or exits. No thread of the
/// runtime outlives the teardown, and nothing relies on a static destructor or an `atexit`
/// handler: the consumer drops every handle, or calls [`quiesce`](Device::quiesce), before the
/// process exits (a driver may unmap device memory inside `exit()`).
#[derive(Clone)]
pub struct Device {
    shared: Arc<Shared>,
}

/// What the device's handles share: the core, the blocks the consumer holds, and the runtime's
/// threads, which hold the core but no handle.
struct Shared {
    core: Arc<Core>,
    claims: Mutex<HashMap<BlockId, Arc<Block>>>,
    threads: Mutex<Threads>,
}

/// The runtime's threads, each until it is stopped.
#[derive(Default)]
struct Threads {
    blocks: Option<Runner>,
    submit: Option<Runner>,
    complete: Option<Runner>,
    reaper: Option<Runner>,
}

struct Runner {
    handle: JoinHandle<()>,
    ended: Arc<Ended>,
    role: Role,
}

/// What a runtime thread does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Not a runtime thread.
    None,
    Blocks,
    Submit,
    /// The completion thread: the only one that runs the consumer's code.
    Complete,
    Reaper,
}

/// Set as a runtime thread ends, so a stop can wait for it with a deadline.
#[derive(Default)]
struct Ended {
    flag: Mutex<bool>,
    changed: Condvar,
}

/// Marks its thread ended when dropped, after the thread's body, panics included.
struct Ending(Arc<Ended>);

impl Drop for Ending {
    fn drop(&mut self) {
        *lock(&self.0.flag) = true;
        self.0.changed.notify_all();
    }
}

thread_local! {
    /// The core a runtime thread serves, by address, and its role; 0 and `Role::None` on every
    /// other thread. Constant and without a destructor, so it reads the same while the thread's
    /// other thread-locals are torn down (in `exit()`).
    static SERVES: Cell<(usize, Role)> = const { Cell::new((0, Role::None)) };
}

/// The role of this thread for `core`.
fn role(core: &Arc<Core>) -> Role {
    let (serves, role) = SERVES.get();
    if serves == Arc::as_ptr(core) as usize {
        role
    } else {
        Role::None
    }
}

/// The device itself: what the runtime's threads and every block hold.
pub(crate) struct Core {
    instance: Instance,
    physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    family: u32,
    extensions: Vec<CString>,
    pub(crate) queues: Queues,
    pub(crate) memory_types: floor::MemoryTypes,
    unified_memory: bool,
    pub(crate) counters: Counters,
    /// What the device's timestamps count.
    pub(crate) ticks: Ticks,
    /// The longest block: the device's `maxMemoryAllocationSize` or `maxBufferSize`, whichever is
    /// less.
    max_block_len: u64,
    pub(crate) pool: Pool,
    pub(crate) filler: Filler,
    pub(crate) recorders: Recorders,
    pub(crate) work: Work,
    pub(crate) done: Done,
    pub(crate) reaper: Reaper,
    /// Every queue's timestamp pool not yet destroyed: one whose queue outlives the device is
    /// destroyed with it.
    pub(crate) query_pools: Mutex<Vec<vk::QueryPool>>,
    /// The one pipeline layout of every pipeline: the push window over every stage.
    pub(crate) layout: vk::PipelineLayout,
    /// The device's compute limits.
    pub(crate) limits: ComputeLimits,
    /// The parameter ring.
    pub(crate) ring: Ring,
    /// Resource ids, for the hazard sets.
    pub(crate) ids: Ids,
    /// The translator, for what device creation enables, under the identity domains.
    pub(crate) translator: metal2vulkan::device::Device,
    /// Quiesced or torn down: calls that need a thread are refused.
    closed: AtomicBool,
    /// [`RUNNING`], [`HALTING`], or the answer of the stop that ran.
    halt: AtomicU8,
}

/// No stop has begun.
const RUNNING: u8 = 0;
/// A stop is running.
const HALTING: u8 = 1;
/// A stop ended with the GPU idle and every thread joined.
const HALTED_IDLE: u8 = 2;
/// A stop ended at its deadline.
const HALTED_LATE: u8 = 3;

/// The device's queues, each behind the lock that Vulkan's external synchronization of a queue
/// needs. Where the family has a second queue, presenting has it to itself; where it has one, the
/// presenting queue is the runtime's own queue and its lock.
pub(crate) struct Queues {
    pub(crate) runtime: Arc<Mutex<vk::Queue>>,
    present: Arc<Mutex<vk::Queue>>,
}

/// No Metal counterpart: what [`Device::new`] takes besides the physical device.
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// The consumer's hook before the runtime frees a block it handed out. `None` where the
    /// consumer keeps no mapping of its own, so a block is freed at once.
    pub block_mappings: Option<Arc<dyn BlockMappings>>,
    /// Device extensions the consumer uses beyond the floor's, each enabled where the device
    /// offers it and the features the consumer needs ([`Device::extensions`]).
    pub extensions: Vec<DeviceExtension>,
}

/// Takes a lock. A panic while another thread held it leaves what it guards usable (a queue, the
/// pool's lists, the claims), so the lock is taken whatever that thread left behind.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Device {
    /// `MTLCreateSystemDefaultDevice`, for the physical device the consumer chose.
    ///
    /// Checks the floor ([`PhysicalDevice::check`]), then makes one `VkDevice` with the floor's
    /// features and extensions, each of the optional features and extensions the device has, and each
    /// of the consumer's extensions ([`DeviceOptions::extensions`]) the device offers with the
    /// features the consumer needs, and chooses its memory types on it. It takes one queue of the floor's queue
    /// family for the runtime and, where that family has a second queue, a queue for presenting;
    /// where it has one, presenting shares the runtime's queue and its lock. It starts the
    /// block-maker thread, which makes one block of each memory class ready, the submit thread and
    /// the completion thread.
    ///
    /// Err: [`Error::BelowFloor`] with every item the device lacks; [`Error::Vulkan`] where a
    /// Vulkan call fails; [`Error::Thread`]; [`Error::Translation`] where the translator takes no
    /// description of the device.
    pub fn new(physical: &PhysicalDevice, options: DeviceOptions) -> Result<Device, Error> {
        let instance = physical.instance().clone();
        let vk_instance = instance.vk();
        let raw = physical.raw();
        let capabilities = physical.check().map_err(Error::BelowFloor)?;
        // SAFETY: `raw` is the instance's, which was made at the floor's version.
        let translator = unsafe { translator(vk_instance, raw, &capabilities) }?;
        let family = capabilities.queue_family;
        // SAFETY: as above.
        let families = unsafe { vk_instance.get_physical_device_queue_family_properties(raw) };
        // The family's second queue, where it has one, is for presenting.
        let count = families[family as usize].queue_count.min(2);

        let priorities = [1.0_f32; 2];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities[..count as usize]);
        // Every structure of the floor's features goes in the chain: a structure added to
        // `floor::Features` fails to compile here until it is chained.
        let floor::Features {
            core,
            mut vulkan11,
            mut vulkan12,
            mut vulkan13,
            mut workgroup_layout,
            mut maximal_reconvergence,
            mut subgroup_uniform_control_flow,
        } = capabilities.features;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut workgroup_layout);
        // An optional extension's structure goes in only where the extension is enabled.
        let enabled = |name: &std::ffi::CStr| capabilities.extensions.contains(&name);
        if enabled(ash::khr::shader_maximal_reconvergence::NAME) {
            features = features.push_next(&mut maximal_reconvergence);
        }
        if enabled(ash::khr::shader_subgroup_uniform_control_flow::NAME) {
            features = features.push_next(&mut subgroup_uniform_control_flow);
        }
        // SAFETY: as for the floor check.
        let listed = unsafe { vk_instance.enumerate_device_extension_properties(raw) }
            .map_err(Error::vulkan("vkEnumerateDeviceExtensionProperties"))?;
        // SAFETY: as for the floor check.
        let (extensions, mut links) = unsafe {
            extension::choose(
                vk_instance,
                raw,
                &listed,
                &capabilities.extensions,
                &options.extensions,
            )
        };
        let pointers: Vec<*const c_char> = extensions.iter().map(|name| name.as_ptr()).collect();
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&pointers)
            .push_next(&mut features);
        // SAFETY: `info.p_next` heads the floor's chain, and the links live through the call.
        info.p_next = unsafe { extension::chain(info.p_next, &mut links) };
        // SAFETY: `info` and everything it points to live through the call; the floor checked
        // that the device offers every feature and extension of its own it enables, and
        // `extension::choose` that it offers each of the consumer's and its features.
        let device = unsafe { vk_instance.create_device(raw, &info, None) }
            .map_err(Error::vulkan("vkCreateDevice"))?;
        drop(links);
        let counters = Counters::new();
        // SAFETY: the device was made from `raw` with `capabilities.features`, and the family is
        // its queues' family.
        let made = unsafe { objects(&device, &capabilities, family, &counters) };
        let (memory_types, filler, done, layout) = match made {
            Ok(made) => made,
            Err(error) => {
                // SAFETY: nothing made from the device is left.
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };

        // SAFETY: the device was made with `count` queues of `family`.
        let runtime = Arc::new(Mutex::new(unsafe { device.get_device_queue(family, 0) }));
        let present = if count == 2 {
            // SAFETY: as above.
            Arc::new(Mutex::new(unsafe { device.get_device_queue(family, 1) }))
        } else {
            Arc::clone(&runtime)
        };
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan13);
        // SAFETY: as for the floor check; the device is Vulkan 1.3, which defines both structures.
        unsafe { vk_instance.get_physical_device_properties2(raw, &mut properties) };
        let unified_memory = matches!(
            properties.properties.device_type,
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
        );
        let ticks = Ticks::new(
            properties.properties.limits.timestamp_period,
            families[family as usize].timestamp_valid_bits,
        );
        let limits = &properties.properties.limits;
        let limits = ComputeLimits {
            max_group_count: limits.max_compute_work_group_count,
            max_group_size: limits.max_compute_work_group_size,
        };
        let max_block_len = vulkan11
            .max_memory_allocation_size
            .min(vulkan13.max_buffer_size);

        let core = Arc::new(Core {
            instance,
            physical: raw,
            device,
            family,
            extensions,
            queues: Queues { runtime, present },
            memory_types,
            unified_memory,
            counters,
            ticks,
            max_block_len,
            pool: Pool::new(options.block_mappings),
            filler,
            recorders: Recorders::new(family),
            work: Work::new(),
            done,
            reaper: Reaper::new(),
            query_pools: Mutex::new(Vec::new()),
            layout,
            limits,
            ring: Ring::new(),
            ids: Ids::new(),
            translator,
            closed: AtomicBool::new(false),
            halt: AtomicU8::new(RUNNING),
        });
        let device = Device {
            shared: Arc::new(Shared {
                core,
                claims: Mutex::new(HashMap::new()),
                threads: Mutex::new(Threads::default()),
            }),
        };
        // A thread that does not start drops the device, which stops those that did.
        let core = &device.shared.core;
        let reaper = spawn("m2v-reaper", core, Role::Reaper, reaper::run)?;
        lock(&device.shared.threads).reaper = Some(reaper);
        let blocks = spawn("m2v-blocks", core, Role::Blocks, memory::run)?;
        lock(&device.shared.threads).blocks = Some(blocks);
        let submit = spawn("m2v-submit", core, Role::Submit, submit::run)?;
        lock(&device.shared.threads).submit = Some(submit);
        let complete = spawn("m2v-complete", core, Role::Complete, complete::run)?;
        lock(&device.shared.threads).complete = Some(complete);
        Ok(device)
    }

    /// `hasUnifiedMemory`: true where Vulkan's device type is integrated or CPU, false for a
    /// discrete or virtual GPU. On a discrete GPU, `Shared` memory is system memory the GPU reads
    /// across its bus, as on a Mac with a discrete GPU.
    pub fn has_unified_memory(&self) -> bool {
        self.shared.core.unified_memory
    }

    /// No Metal counterpart: the device extensions enabled: the floor's first, then each of the
    /// consumer's ([`DeviceOptions::extensions`]) the device offers with its features.
    pub fn extensions(&self) -> &[CString] {
        &self.shared.core.extensions
    }

    /// No Metal counterpart: the memory types the device's buffers are allocated from, as the
    /// floor chose them on the device.
    pub fn memory_types(&self) -> floor::MemoryTypes {
        self.shared.core.memory_types
    }

    /// No Metal counterpart: the Vulkan handles, for the consumer's swapchain and presents. The
    /// runtime keeps owning them; the consumer destroys none of them, and destroys whatever it
    /// makes with them before the device is torn down.
    pub fn raw(&self) -> RawDevice<'_> {
        let core = &self.shared.core;
        RawDevice {
            instance: core.instance.vk(),
            physical_device: core.physical,
            device: &core.device,
            queue_family_index: core.family,
        }
    }

    /// No Metal counterpart: runs `f` with the presenting queue, holding its lock for the call.
    /// Where the family has one queue, that lock is the one every submission of the runtime takes,
    /// so `f` and the runtime never submit at once.
    ///
    /// The queue is externally synchronized only inside `f`, so the Vulkan calls `f` makes with it
    /// keep no handle past its return. `f` waits for no work of the runtime's, which may need the
    /// lock `f` holds; a present in `f` waits only on semaphores whose signal is already submitted
    /// (the command buffer that signals one is scheduled).
    pub fn with_present_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let queue = lock(&self.shared.core.queues.present);
        f(*queue)
    }

    /// No Metal counterpart: the device's metrics and gauges. The handle stays readable after the
    /// device is torn down.
    pub fn counters(&self) -> Counters {
        self.shared.core.counters.clone()
    }

    /// No Metal counterpart: takes a block of `class` memory at least `len` bytes long, for the
    /// consumer to place buffers in.
    ///
    /// For `len` up to [`BLOCK_LEN`] it is a pooled block. The block-maker thread makes pooled
    /// blocks ahead of need, off every caller's path, so this call waits only when no block of
    /// the class is ready. Past [`BLOCK_LEN`] it is a dedicated block, made in this call.
    ///
    /// Every block reads zero, as Metal's new buffers do, whether new or taken again: a Shared
    /// block is zeroed through its mapping, which also makes each page resident before the block
    /// is handed out, and a Private block by a GPU fill. A pooled block released and taken again
    /// keeps its number, address and mapping.
    ///
    /// Err: [`Limit::BlockLen`](crate::Limit::BlockLen) past the device's largest allocation or
    /// buffer; [`Limit::BlockIds`](crate::Limit::BlockIds); [`Error::Vulkan`] where the device is
    /// out of memory or lost; [`Error::Quiesced`].
    pub fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, Error> {
        let core = &self.shared.core;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let memory = if len <= BLOCK_LEN {
            core.pool.take(class)?
        } else {
            let len = memory::dedicated_len(len, core.max_block_len)?;
            memory::make(core, class, len, core.pool.number()?, false)?
        };
        let info = memory.info();
        core.pool.hand(memory.id);
        lock(&self.shared.claims).insert(memory.id, Arc::new(Block::new(memory, Arc::clone(core))));
        Ok(info)
    }

    /// No Metal counterpart: stops the runtime's threads and its GPU work, for a process about to
    /// exit with the device still alive. True once the GPU went idle and every thread ended
    /// within `timeout`; false where the timeout passed first, or the call ran in a handler or a
    /// listener (the completion thread cannot wait for itself).
    ///
    /// In order, each wait bounded by `timeout`: the submit thread submits what is ready, ends
    /// every command buffer it holds or has not yet submitted with
    /// [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun), and ends; the
    /// block-maker thread ends; the GPU completes everything submitted; the completion thread
    /// retires it, runs the handlers, and ends; then [`BlockMappings::unmap`] runs on this thread
    /// for every block still handed out. After it, no runtime thread touches a block, unless the
    /// call answered false.
    ///
    /// It frees nothing: every block's memory and mapping stays valid until the process ends, or
    /// until the device's last handle drops, which then frees it without the hook. Later calls
    /// that need a runtime thread return [`Error::Quiesced`]: taking a block, a command buffer,
    /// an enqueue or commit, a zeroed Private buffer, and a listener.
    ///
    /// The consumer calls it before `exit()`. It can also run in an `atexit` handler on whichever
    /// thread calls `exit()`: it joins a thread only once that thread has ended, reads no
    /// thread-local that `exit()` tears down, and takes no lock a consumer's thread can hold. Such
    /// a handler runs before the driver's own exit teardown only where it was registered after the
    /// driver loaded, after `Device::new`. A second call returns at once, with the first call's
    /// answer, or false while the first is still running.
    pub fn quiesce(&self, timeout: Duration) -> bool {
        let threads = self.shared.take_threads();
        halt(
            &self.shared.core,
            threads,
            Instant::now().checked_add(timeout),
            true,
        )
    }

    /// No Metal counterpart: gives a block back.
    ///
    /// It takes effect once no [`Buffer`](crate::Buffer) of the block remains and every command
    /// buffer that used it has retired. A pooled block is then zeroed and kept ready, with the same number, address and mapping, while its class has
    /// fewer than two idle; otherwise the block-maker thread frees it, after
    /// [`BlockMappings::unmap`]. A dedicated block is always freed that way.
    ///
    /// Err: [`Error::NoSuchBlock`] for a block the consumer does not hold.
    pub fn release_block(&self, block: BlockId) -> Result<(), Error> {
        let held = lock(&self.shared.claims)
            .remove(&block)
            .ok_or(Error::NoSuchBlock(block))?;
        drop(held);
        Ok(())
    }

    /// The core, for the objects made from the device.
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.shared.core
    }

    /// A block the consumer holds. Err: [`Error::NoSuchBlock`].
    pub(crate) fn block(&self, id: BlockId) -> Result<Arc<Block>, Error> {
        lock(&self.shared.claims)
            .get(&id)
            .cloned()
            .ok_or(Error::NoSuchBlock(id))
    }

    /// For the device tests: runs `f` with the runtime's queue under its lock.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn with_runtime_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        f(*lock(&self.shared.core.queues.runtime))
    }

    /// For the device tests: the runtime acts as at a loss of the device.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn lose_for_test(&self) {
        complete::lose(&self.shared.core);
    }
}

/// Starts a runtime thread that serves `core` in `role`.
fn spawn(name: &str, core: &Arc<Core>, role: Role, body: fn(Arc<Core>)) -> Result<Runner, Error> {
    let ended = Arc::new(Ended::default());
    let ending = Ending(Arc::clone(&ended));
    let core = Arc::clone(core);
    let handle = thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let _ending = ending;
            SERVES.set((Arc::as_ptr(&core) as usize, role));
            body(core);
        })
        .map_err(Error::Thread)?;
    Ok(Runner {
        handle,
        ended,
        role,
    })
}

impl Ended {
    /// Waits for the thread to end, up to `deadline`. True once it has.
    fn wait(&self, deadline: Option<Instant>) -> bool {
        let mut flag = lock(&self.flag);
        while !*flag {
            flag = match deadline {
                None => self
                    .changed
                    .wait(flag)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    self.changed
                        .wait_timeout(flag, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
        true
    }
}

impl Shared {
    /// The runtime's threads, for the stop that waits for them.
    fn take_threads(&self) -> Threads {
        let mut threads = lock(&self.threads);
        Threads {
            blocks: threads.blocks.take(),
            submit: threads.submit.take(),
            complete: threads.complete.take(),
            reaper: threads.reaper.take(),
        }
    }
}

/// Stops the runtime's threads and its GPU work: for [`Device::quiesce`] (`quiesce`, with its
/// deadline) and for teardown. Only the first call stops anything; a later one answers at once.
/// True once the GPU went idle and every thread ended by `deadline`.
fn halt(core: &Arc<Core>, threads: Threads, deadline: Option<Instant>, quiesce: bool) -> bool {
    match core
        .halt
        .compare_exchange(RUNNING, HALTING, Ordering::AcqRel, Ordering::Acquire)
    {
        Ok(_) => {}
        Err(halt) => return halt == HALTED_IDLE,
    }
    core.closed.store(true, Ordering::Release);
    let Threads {
        blocks,
        submit,
        complete,
        reaper,
    } = threads;
    // Submission stops: what is ready goes, and what is held ends.
    core.work.stop();
    let mut idle = finish(submit, core, deadline);
    core.pool.stop();
    idle &= finish(blocks, core, deadline);
    if quiesce {
        idle &= core.wait_progress(core.work.progress(), deadline);
    }
    core.done.stop(core, deadline);
    idle &= finish(complete, core, deadline) && core.done.drained();
    core.reaper.stop();
    idle &= finish(reaper, core, deadline);
    if quiesce {
        core.pool.unmap_all();
    }
    let answer = if idle { HALTED_IDLE } else { HALTED_LATE };
    core.halt.store(answer, Ordering::Release);
    idle
}

/// Waits for a thread to end, up to `deadline`, and joins it once it has; a thread past the
/// deadline is left to end alone. True once it ended. A thread is never waited for by itself: the
/// completion thread where a handler or listener quiesces, the reaper where it runs a teardown;
/// it ends once the call returns.
fn finish(runner: Option<Runner>, core: &Arc<Core>, deadline: Option<Instant>) -> bool {
    let Some(Runner {
        handle,
        ended,
        role: of,
    }) = runner
    else {
        return true;
    };
    if role(core) == of {
        return false;
    }
    let done = ended.wait(deadline);
    if done {
        // A thread that panicked has ended all the same.
        let _ = handle.join();
    }
    done
}

/// The memory types the floor chooses on a new device, the zero fill's objects, and the
/// completion thread's semaphores.
///
/// # Safety
///
/// `device` was made with `capabilities.features`, and `family` is its queues' family.
/// The translator for `raw` with what device creation enables of `capabilities`, under the identity
/// domains: a buffer argument is the guest's address of the buffer, a push field, and a texture or
/// a sampler its resource ID. It is decided by the device's Vulkan queries alone
/// (`Device::from_queries`): no driver or GPU identity, and no fact measured on one, applies.
///
/// # Safety
///
/// `raw` is a physical device of `instance`, which was made at [`floor::VERSION`].
unsafe fn translator(
    instance: &ash::Instance,
    raw: vk::PhysicalDevice,
    capabilities: &floor::Capabilities,
) -> Result<metal2vulkan::device::Device, Error> {
    // SAFETY: the caller's contract.
    let described =
        unsafe { DeviceDescription::query(instance, raw, ApiVersion::from_packed(floor::VERSION)) };
    let description = described
        .and_then(|description| description.with_enabled(&floor::shader_enables(capabilities)))
        .map_err(Error::Translation)?;
    let mut policy = DevicePolicy::default();
    policy.pointer_domain = PointerDomain::Identity;
    policy.handle_domain = HandleDomain::Identity;
    metal2vulkan::device::Device::from_queries(description, policy).map_err(Error::Translation)
}

unsafe fn objects(
    device: &ash::Device,
    capabilities: &floor::Capabilities,
    family: u32,
    counters: &Counters,
) -> Result<(floor::MemoryTypes, Filler, Done, vk::PipelineLayout), Error> {
    // SAFETY: the caller's contract.
    let memory_types =
        unsafe { floor::memory_types(device, capabilities) }.map_err(Error::BelowFloor)?;
    // SAFETY: the caller's contract; the Shared type is host-visible and host-coherent, and the
    // floor chose it for buffers of every usage.
    let mut filler = unsafe { Filler::new(device, family, memory_types.shared, counters) }?;
    // SAFETY: the device is live.
    let made = unsafe { Done::new(device) }.and_then(|mut done| {
        let push = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: PUSH_BYTES,
        }];
        let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: the device is live, and every device holds 128 bytes of push constants.
        match unsafe { device.create_pipeline_layout(&info, None) } {
            Ok(layout) => Ok((done, layout)),
            Err(result) => {
                // SAFETY: the completion objects are unused.
                unsafe { done.destroy(device) };
                Err(Error::Vulkan {
                    call: "vkCreatePipelineLayout",
                    result,
                })
            }
        }
    });
    match made {
        Ok((done, layout)) => Ok((memory_types, filler, done, layout)),
        Err(error) => {
            // SAFETY: the fill's objects are unused.
            if unsafe { filler.destroy(device) } {
                counters.fall(Gauge::Blocks);
            }
            Err(error)
        }
    }
}

/// The device's compute limits a dispatch is checked against.
pub(crate) struct ComputeLimits {
    /// `maxComputeWorkGroupCount`.
    pub(crate) max_group_count: [u32; 3],
    /// `maxComputeWorkGroupSize`.
    pub(crate) max_group_size: [u32; 3],
}

impl Core {
    /// The memory type of a class's blocks.
    pub(crate) fn memory_type(&self, class: MemoryClass) -> u32 {
        match class {
            MemoryClass::Private => self.memory_types.private,
            MemoryClass::Shared => self.memory_types.shared,
        }
    }

    /// Quiesced or torn down.
    pub(crate) fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Waits for the device to go idle. A lost device does so at once.
    pub(crate) fn wait_idle(&self) {
        // `vkDeviceWaitIdle` needs every queue of the device externally synchronized.
        let _runtime = lock(&self.queues.runtime);
        let _present = (!Arc::ptr_eq(&self.queues.runtime, &self.queues.present))
            .then(|| lock(&self.queues.present));
        // SAFETY: every queue is held. A lost device fails the wait, and then nothing is pending.
        let _ = unsafe { self.device.device_wait_idle() };
    }
}

/// No Metal counterpart: a device's Vulkan handles ([`Device::raw`]).
#[non_exhaustive]
pub struct RawDevice<'a> {
    /// The instance's function table.
    pub instance: &'a ash::Instance,
    /// The physical device.
    pub physical_device: vk::PhysicalDevice,
    /// The device's function table.
    pub device: &'a ash::Device,
    /// The family of the runtime's queue and of the presenting queue.
    pub queue_family_index: u32,
}

impl Drop for Shared {
    fn drop(&mut self) {
        let threads = self.take_threads();
        // The blocks the consumer still holds go back to the pool once the threads have stopped,
        // for the core to free. No buffer of them is left: each holds a handle of the device.
        let claims = std::mem::take(
            self.claims
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        let teardown = move |core: &Arc<Core>| {
            halt(core, threads, None, false);
            drop(claims);
        };
        if role(&self.core) == Role::Complete {
            // A handler or listener dropped the last handle: the reaper tears down, once the
            // handler returns and the completion thread can end.
            self.core.reaper.give(Job::Teardown(Box::new(teardown)));
        } else {
            teardown(&self.core);
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        self.wait_idle();
        // The runtime's threads have ended: the core is dropped only once none holds it, and with
        // it every command buffer and upload, each of which held what it used.
        for memory in self.pool.remaining() {
            memory::free(self, memory);
        }
        self.recorders.destroy_all(&self.device);
        // The reaper has ended; what it was given since is freed here, and then the pools of the
        // queues still alive.
        self.reaper.drain(self);
        let pools = std::mem::take(
            self.query_pools
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for pool in pools {
            // SAFETY: the device is idle.
            unsafe { self.device.destroy_query_pool(pool, None) };
        }
        // SAFETY: the device is idle; nothing reads the ring.
        let ring_blocks = unsafe { self.ring.destroy(&self.device) };
        for _ in 0..ring_blocks {
            self.counters.fall(Gauge::Blocks);
        }
        // SAFETY: the device is idle, and nothing made from it is left but these objects.
        let zero_source = unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
            self.done.destroy(&self.device);
            let zero_source = self.filler.destroy(&self.device);
            self.device.destroy_device(None);
            zero_source
        };
        // The zero source's gauge falls last, once the `VkDevice` is destroyed: every gauge reads
        // zero only when the teardown has made its last Vulkan call, so a consumer whose last
        // handle dropped in a handler can wait for that before it exits.
        if zero_source {
            self.counters.fall(Gauge::Blocks);
        }
    }
}
