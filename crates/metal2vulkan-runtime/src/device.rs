//! The device: one `VkDevice` for every command queue, every guest process and the compositor;
//! its queues, its memory blocks, its threads, quiesce and teardown.

use crate::arrays::Arrays;
use crate::block::{Block, Claim};
use crate::cache::{self, Caches};
use crate::command::DeviceLoss;
use crate::complete::{self, Done};
use crate::counters::{Counters, Gauge, Site};
use crate::error::{Error, Refusal};
use crate::export::Exports;
use crate::extension::{self, ConsumerFeature, DeviceExtension};
use crate::fill::Filler;
use crate::floor;
use crate::hazard::{HeapIds, Ids};
use crate::import::Imports;
use crate::instance::{Instance, PhysicalDevice};
use crate::kernel_map::KernelMap;
use crate::memory::{self, BlockId, BlockMappings, Memory, MemoryClass, Pool};
use crate::pass::ResolveModes;
use crate::pipeline::{PUSH_BYTES, PUSH_STAGES};
use crate::places::Places;
use crate::reaper::{self, Job, Reaper};
use crate::record::Recorders;
use crate::ring::Ring;
use crate::sampler::{Constexpr, SamplerTable};
use crate::submit::{self, Work};
use crate::submit_calls::SubmitCalls;
use crate::times::{StampPools, Ticks};
use ash::vk;
use metal2vulkan::device::{
    ApiVersion, DeviceDescription, DevicePolicy, HandleDomain, PointerDomain,
};
use metal2vulkan::metal::{ArgumentBuffersTier, ReadWriteTextureTier};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// `MTLDevice`, over one `VkDevice` that every command queue, every guest process and the
/// compositor share.
///
/// A cheap handle: clones share one device, and every object made from it holds a clone, except
/// a committed command buffer. Blocks the consumer holds do not, and nothing the runtime keeps for
/// itself counts as a handle of the device: what a command buffer keeps of a buffer, a texture or
/// a sampler state it used is the object alone.
///
/// The runtime runs four threads: the block-maker, which makes the pool's allocations ahead of need
/// and the runtime's fixed-size internals (the zero source, the parameter ring's blocks and the
/// staging region) off every caller's thread, and frees device memory; the submit thread, which runs every scheduled
/// handler; the completion thread, which runs every completed handler and listener; and the
/// reaper, which frees the semaphores and command pools the others give up, one at a time, so no
/// driver free runs on the completion thread. The device is torn down when the last handle is
/// dropped: the submit thread submits what is ready and ends what it holds with
/// [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun); the completion thread
/// retires the rest as the GPU completes it; the threads are joined; then every allocation is
/// freed, after [`BlockMappings::unmap`] for each the consumer mapped, and the
/// `VkDevice` is destroyed; every gauge reads zero only then. A last handle dropped on the
/// consumer's thread tears the device down in that drop. Where the last handle drops in a handler
/// or a listener, the reaper runs the teardown once the handler returns, and the consumer calls
/// [`Instance::wait_until_devices_torn_down`](crate::Instance::wait_until_devices_torn_down)
/// before it drops its instance or exits. Nothing relies on a static destructor or an `atexit`
/// handler: the consumer drops every handle before the process exits, or, where a handle is still
/// out, calls [`quiesce`](Device::quiesce) and ends with `_exit`. `exit()` with a live device is
/// supported on no device.
#[derive(Clone)]
pub struct Device {
    shared: Arc<Shared>,
}

/// What the device's handles share: the core and the runtime's threads, which hold the core but
/// no handle.
struct Shared {
    core: Arc<Core>,
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
    /// The submit thread: it runs the consumer's scheduled handlers.
    Submit,
    /// The completion thread: it runs the consumer's completed handlers and listeners.
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

/// The devices of one instance not yet torn down, and the runtime threads that ran a teardown,
/// until a wait joins them ([`Instance::wait_until_devices_torn_down`]).
#[derive(Default)]
pub(crate) struct Teardowns {
    state: Mutex<Torn>,
    changed: Condvar,
}

#[derive(Default)]
struct Torn {
    /// The devices whose core has not dropped its last field.
    live: usize,
    /// The runtime threads that ran a teardown and are not yet joined: those still ending, as
    /// each hand joins the ones that have ended.
    threads: Vec<Runner>,
}

/// A device's count among its instance's [`Teardowns`]: the core's last field, which drops once
/// every other field has, the `VkDevice` destroyed among them.
pub(crate) struct Live(Arc<Teardowns>);

impl Drop for Live {
    fn drop(&mut self) {
        lock(&self.0.state).live -= 1;
        self.0.changed.notify_all();
    }
}

impl Teardowns {
    /// Counts a new device of the instance until its core's last field drops.
    pub(crate) fn live(self: &Arc<Self>) -> Live {
        lock(&self.state).live += 1;
        Live(Arc::clone(self))
    }

    /// Keeps `runner`, the runtime thread that ran a teardown and ends once it returns, for a
    /// wait to join, and joins those kept earlier that have ended.
    fn hand(&self, runner: Runner) {
        let ended: Vec<Runner> = {
            let mut torn = lock(&self.state);
            let (ended, running) =
                (torn.threads.drain(..)).partition(|kept: &Runner| kept.handle.is_finished());
            torn.threads = running;
            torn.threads.push(runner);
            ended
        };
        for kept in ended {
            // A thread that panicked has ended all the same.
            let _ = kept.handle.join();
        }
    }

    /// Waits, up to `deadline`, for every device counted to drop its core's last field, then
    /// for each thread handed to end, and joins it: true once all have. False at once on a
    /// thread of the runtime, which would wait for itself.
    pub(crate) fn wait(&self, deadline: Option<Instant>) -> bool {
        if SERVES.get().1 != Role::None {
            return false;
        }
        let mut torn = lock(&self.state);
        while torn.live > 0 {
            torn = match deadline {
                None => (self.changed.wait(torn)).unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    (self.changed.wait_timeout(torn, left))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
        // No device is left to hand a thread: each one handed has ended its teardown, and ends.
        while let Some(runner) = torn.threads.pop() {
            if !runner.ended.wait(deadline) {
                torn.threads.push(runner);
                return false;
            }
            // A thread that panicked has ended all the same.
            let _ = runner.handle.join();
        }
        true
    }
}

thread_local! {
    /// The core a runtime thread serves, by address, and its role; 0 and `Role::None` on every
    /// other thread. Constant and without a destructor, so it reads the same while the thread's
    /// other thread-locals are torn down (in `exit()`).
    static SERVES: Cell<(usize, Role)> = const { Cell::new((0, Role::None)) };
}

/// The role of this thread for `core`.
/// Whether this thread is `core`'s completion thread, which runs its completed handlers and its
/// listeners.
pub(crate) fn on_completion_thread(core: &Arc<Core>) -> bool {
    role(core) == Role::Complete
}

/// Whether this thread is one of `core`'s threads that run the consumer's code: the submit thread
/// (its scheduled handlers) or the completion thread (its completed handlers and listeners).
pub(crate) fn on_handler_thread(core: &Arc<Core>) -> bool {
    matches!(role(core), Role::Submit | Role::Complete)
}

fn role(core: &Arc<Core>) -> Role {
    let (serves, role) = SERVES.get();
    if serves == Arc::as_ptr(core) as usize {
        role
    } else {
        Role::None
    }
}

/// The device itself: what the runtime's threads, every block and every object a command buffer
/// keeps hold.
pub(crate) struct Core {
    pub(crate) instance: Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    /// The family of the runtime's queues, which ownership transfers name as the device's own.
    pub(crate) family: u32,
    extensions: Vec<CString>,
    /// The consumer's features enabled.
    consumer_features: Vec<ConsumerFeature>,
    pub(crate) queues: Queues,
    pub(crate) memory_types: floor::MemoryTypes,
    /// The device-local memory types, one bit each: a texture's image takes the first its
    /// requirements allow.
    pub(crate) device_local: u32,
    unified_memory: bool,
    pub(crate) counters: Counters,
    /// What the device's timestamps count.
    pub(crate) ticks: Ticks,
    /// `VK_EXT_calibrated_timestamps`, the floor's: the GPU's clock read with the host's.
    pub(crate) calibrated: ash::ext::calibrated_timestamps::Device,
    /// `VK_EXT_attachment_feedback_loop_dynamic_state`'s command, where device creation enabled
    /// the attachment feedback loop (`floor::Capabilities::feedback_loop`): every render pipeline
    /// takes the feedback loop's enable as dynamic state, which each draw sets.
    pub(crate) feedback: Option<ash::ext::attachment_feedback_loop_dynamic_state::Device>,
    /// A draw may read an attachment of its own pass through a texture
    /// (`floor::Capabilities::attachment_reads`): through the feedback loop where `feedback` is
    /// set, else through a copy of the image into its shadow (`crate::shadow`).
    pub(crate) attachment_reads: bool,
    /// Device creation enabled `linearColorAttachment` (`VK_NV_linear_color_attachment`, where
    /// the device lists it): an opaque handle's surface plane whose format's `LINEAR` tiling
    /// renders through `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV` alone lies in place
    /// (`crate::surface`), its image marked, and a pass renders it only where Vulkan's rule for
    /// such images holds ([`Gap::LinearPlanePass`](crate::Gap::LinearPlanePass)).
    pub(crate) linear_attachments: bool,
    /// The depth and stencil resolve modes a render pass may resolve by.
    pub(crate) resolve_modes: ResolveModes,
    /// The longest block: the device's `maxMemoryAllocationSize` or `maxBufferSize`, whichever is
    /// less.
    pub(crate) max_block_len: u64,
    /// The device's `maxMemoryAllocationCount`: the most allocations live at once
    /// ([`memory::allocate`]).
    pub(crate) max_allocations: u32,
    /// The device's `bufferImageGranularity`, which a heap texture's size and alignment are
    /// rounded to.
    pub(crate) granularity: u64,
    /// For the device tests: how many more Private allocations of the Private type fail as out
    /// of device memory, before the driver is called.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fail_private: std::sync::atomic::AtomicU32,
    /// For the device tests: nanoseconds added to the host time of each calibration read, as a
    /// calibration whose host clock is read that long after its GPU tick.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) calibration_lateness: std::sync::atomic::AtomicU64,
    /// For the device tests: the span of each command buffer's two timestamps the completion
    /// thread placed, in the order it placed them ([`Ticks::span`](crate::times::Ticks::span)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) tick_spans: Mutex<Vec<u64>>,
    /// For the device tests: where set, the making of the staging region meets the test at
    /// the barrier twice before it allocates, once parked and once released.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) park_region: Mutex<Option<Arc<std::sync::Barrier>>>,
    /// For the device tests: what the block-maker thread does with the internals asked of it
    /// ([`DeviceOptions::maker`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) maker_test: memory::MakerTest,
    pub(crate) pool: Pool,
    /// The runtime's `vkQueueSubmit2` calls inside the driver, which the block-maker thread waits
    /// out after each of its memory calls.
    pub(crate) submit_calls: Arc<SubmitCalls>,
    /// What the device imports (`Device::import_dma_buf` on Unix, [`Device::import_host_memory`]).
    pub(crate) imports: Imports,
    /// What the device exports ([`Device::new_exported_block`]).
    pub(crate) exports: Exports,
    /// The blocks the consumer holds handles of, by number ([`Block`]): each claim goes once its
    /// last handle drops, or at teardown.
    pub(crate) claims: Mutex<HashMap<BlockId, Weak<Claim>>>,
    /// The blocks textures are placed in.
    pub(crate) places: Places,
    /// The zero source, once the block-maker thread made it ([`crate::fill::zeros`]).
    pub(crate) filler: OnceLock<Filler>,
    pub(crate) recorders: Recorders,
    pub(crate) work: Work,
    pub(crate) done: Done,
    pub(crate) reaper: Reaper,
    /// The timestamp pools every command buffer takes its pair of queries from.
    pub(crate) stamps: StampPools,
    /// The one pipeline layout of every pipeline: the empty set 0, set 1, and the push window
    /// over every stage.
    pub(crate) layout: vk::PipelineLayout,
    /// Set 1: the texture and sampler arrays, and the grants of their indices.
    pub(crate) arrays: Arrays,
    /// The sampler states' device samplers, shared by create info.
    pub(crate) samplers: Mutex<SamplerTable>,
    /// The constexpr samplers the live pipeline states number, by index and state words.
    pub(crate) constexprs: Mutex<HashMap<(u32, [u64; 2]), Constexpr>>,
    /// [`Device::minimum_linear_texture_alignment`]'s answers, one per pixel format asked.
    pub(crate) linear_alignments: Mutex<HashMap<metal2vulkan::metal::PixelFormat, u64>>,
    /// The live textures over a buffer in an image of their own (`crate::over`): while there are
    /// any, the commands track what they write.
    pub(crate) copied_overs: AtomicUsize,
    /// The device's compute limits.
    pub(crate) limits: EncoderLimits,
    /// The parameter ring.
    pub(crate) ring: Ring,
    /// Resource ids, for the hazard sets.
    pub(crate) ids: Ids,
    /// The heaps' ids in the hazard sets, which their resources take.
    pub(crate) heap_ids: HeapIds,
    /// The translator, for what device creation enables, under the identity domains.
    pub(crate) translator: metal2vulkan::device::Device,
    /// The lowered kernels on disk, and the `VkPipelineCache` every pipeline is made through with
    /// its blob.
    pub(crate) caches: Caches,
    /// The compiled kernels by content, which every compute pipeline state of a content shares.
    pub(crate) kernels: KernelMap,
    /// The translator's decodes and specialized entries of the functions the device's pipeline
    /// states lowered, which every creation reads, so a module is decoded and a stage
    /// specialized once for the device rather than once a state. Bounded by the translator: the
    /// least recently used go ([`StageEntries`](metal2vulkan::pipeline::StageEntries)).
    pub(crate) entries: metal2vulkan::pipeline::StageEntries,
    /// Quiesced or torn down: calls that need a thread are refused.
    closed: AtomicBool,
    /// [`RUNNING`], [`HALTING`], or the answer of the stop that ran.
    halt: AtomicU8,
    /// Its count among its instance's devices not yet torn down, which only its drop reads: the
    /// last field, so it drops once every other field has.
    _live: Live,
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
    /// The consumer's hooks around each allocation of the pool: its map before any of its blocks
    /// goes out, and its unmap before the allocation is freed. `None` where the consumer keeps no
    /// mapping of its own.
    pub block_mappings: Option<Arc<dyn BlockMappings>>,
    /// Device extensions the consumer uses beyond the floor's, each enabled where the device
    /// offers it and the features the consumer needs ([`Device::extensions`]).
    pub extensions: Vec<DeviceExtension>,
    /// Vulkan features the consumer needs for its own raw Vulkan work, each enabled where the
    /// device reports it ([`Device::features`]).
    pub features: Vec<ConsumerFeature>,
    /// The directory the device keeps the compute and render functions it lowered, each render
    /// content's record of its lowerings and its pipeline cache in, made where it is missing, so a
    /// later device on it, in this process or another, takes them in place of lowering and
    /// compiling again ([`Metrics::translations_loaded`]). The lowerings take at most
    /// [`pipeline_cache_cap`](Self::pipeline_cache_cap) bytes, in two generations of half of it
    /// each: a file used again moves into the young one, and the old one is removed whole when
    /// the young one fills. Beside them, one file per pipeline cache header (a kind of device and
    /// driver version; at most four such files, the oldest by last write removed) holds the data
    /// of the `VkPipelineCache` every `VkPipeline` of the device is made through, at most 32 MiB:
    /// the device's cache takes it in off every guest thread, and it is
    /// replaced, never appended to, once pipeline creation has been quiet for 20 ms after the
    /// cache grew by more than an eighth, and at the device's drop where the cache changed. A
    /// file that does not check, or a failed read or write, is a miss and never fails a call.
    /// `None` keeps nothing on disk, so every boot lowers every function and compiles every
    /// pipeline cold: a consumer should pass a directory. Metal keeps its compiled shaders in a
    /// system cache the program does not name.
    ///
    /// [`Metrics::translations_loaded`]: crate::Metrics::translations_loaded
    pub pipeline_cache_dir: Option<PathBuf>,
    /// The most bytes the lowerings in [`pipeline_cache_dir`](Self::pipeline_cache_dir) take;
    /// `None` is 128 MiB. The pipeline cache's file, at most 32 MiB per kind of device, is beside
    /// them. Half of the cap is what the directory is sure to keep: a lowering stays for at least
    /// half the cap, and at most the cap, of lowerings written or used after its last use, and
    /// one larger than half is not kept. Devices on one directory, in one process or several,
    /// share its files and one count of their bytes, and should pass one cap: each starts a new
    /// generation at half its own.
    pub pipeline_cache_cap: Option<u64>,
    /// For the device tests: what the block-maker thread does with the internals asked of it,
    /// from its first work on.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) maker: memory::MakerTest,
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
    /// where it has one, presenting shares the runtime's queue and its lock. It makes one
    /// allocation of each memory class for blocks, then starts the block-maker thread, the submit
    /// thread and the completion thread. The block-maker thread's first work, after the call
    /// returns, is the runtime's fixed-size internals: the zero source (64 KiB), the parameter
    /// ring's first block (2 MiB) and the staging region (16 MiB). A call that needs one before it
    /// is made waits for that make, counted in
    /// [`Metrics::maker_waits`](crate::Metrics::maker_waits); where the make failed, the call
    /// makes it on its own thread. After that the thread makes a class's next allocation while
    /// the class holds no wholly free unit (16 MiB that no block holds).
    ///
    /// Err: [`Error::BelowFloor`] with every item the device lacks; [`Error::Vulkan`] where a
    /// Vulkan call fails; [`Error::Thread`]; [`Error::Translation`] where the translator takes no
    /// description of the device.
    pub fn new(physical: &PhysicalDevice, options: DeviceOptions) -> Result<Device, Error> {
        Device::open(physical, options, |_| {})
    }

    /// [`Device::new`], with the device's limits as `answer` leaves them: the tests' way to run a
    /// device at a smaller limit than its own.
    pub(crate) fn open(
        physical: &PhysicalDevice,
        options: DeviceOptions,
        answer: impl FnOnce(&mut vk::PhysicalDeviceLimits),
    ) -> Result<Device, Error> {
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
        // SAFETY: as for the floor check.
        let (enabled_features, consumer_features) =
            unsafe { consumer_features(vk_instance, raw, &capabilities, &options.features) };
        let floor::Features {
            core,
            mut vulkan11,
            mut vulkan12,
            mut vulkan13,
            mut maximal_reconvergence,
            mut subgroup_uniform_control_flow,
            mut quad_control,
            mut interlock,
            mut barycentric,
            mut robustness2,
            mut portability,
            mut feedback_loop,
            mut feedback_loop_dynamic,
            mut divisor,
            mut border_swizzle,
            mut linear_attachment,
            mut line_rasterization,
        } = enabled_features;
        // `sparseBinding` for imports over several dma-bufs (`Device::import_dma_bufs`), only where
        // the device's queries offer that import: it changes no other path, as only that import
        // makes a sparse buffer.
        // SAFETY: as for the floor check.
        let sparse = unsafe {
            crate::import::binds_dma_bufs(
                vk_instance,
                raw,
                &families[family as usize],
                &capabilities.extensions,
            )
        };
        let core = vk::PhysicalDeviceFeatures {
            sparse_binding: core.sparse_binding | vk::Bool32::from(sparse),
            ..core
        };
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut robustness2)
            .push_next(&mut line_rasterization);
        // An optional extension's structure goes in only where the extension is enabled.
        let enabled = |name: &std::ffi::CStr| capabilities.extensions.contains(&name);
        if enabled(ash::khr::shader_maximal_reconvergence::NAME) {
            features = features.push_next(&mut maximal_reconvergence);
        }
        if enabled(ash::khr::shader_subgroup_uniform_control_flow::NAME) {
            features = features.push_next(&mut subgroup_uniform_control_flow);
        }
        if enabled(ash::khr::shader_quad_control::NAME) {
            features = features.push_next(&mut quad_control);
        }
        if enabled(ash::ext::fragment_shader_interlock::NAME) {
            features = features.push_next(&mut interlock);
        }
        if enabled(ash::khr::fragment_shader_barycentric::NAME) {
            features = features.push_next(&mut barycentric);
        }
        if enabled(ash::khr::portability_subset::NAME) {
            features = features.push_next(&mut portability);
        }
        if capabilities.feedback_loop {
            features = features
                .push_next(&mut feedback_loop)
                .push_next(&mut feedback_loop_dynamic);
        }
        if enabled(ash::khr::vertex_attribute_divisor::NAME)
            || enabled(ash::ext::vertex_attribute_divisor::NAME)
        {
            features = features.push_next(&mut divisor);
        }
        if enabled(ash::ext::border_color_swizzle::NAME) {
            features = features.push_next(&mut border_swizzle);
        }
        if enabled(ash::nv::linear_color_attachment::NAME) {
            features = features.push_next(&mut linear_attachment);
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
        // The layouts its commands name, where a test records them (`tests::layouts`).
        // SAFETY: the device was made from `vk_instance` just now, and nothing has used it.
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let device = unsafe { crate::tests::layouts::recorded(vk_instance, device) };
        drop(links);
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        // SAFETY: as for the floor check; the device is Vulkan 1.3, which defines the three
        // structures.
        unsafe { vk_instance.get_physical_device_properties2(raw, &mut properties) };
        answer(&mut properties.properties.limits);
        let max_allocations = properties.properties.limits.max_memory_allocation_count;
        let granularity = properties.properties.limits.buffer_image_granularity;
        let counters = Counters::new();
        let caches = Caches::new(
            options.pipeline_cache_dir.as_deref(),
            (options.pipeline_cache_cap).unwrap_or(cache::DEFAULT_CAP),
            &counters,
        );
        // SAFETY: the device was made from `raw` with `capabilities.features`, and the family is
        // its queues' family.
        let made = unsafe { objects(&device, &capabilities, &counters) };
        let (memory_types, done, arrays, layout) = match made {
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
        let unified_memory = matches!(
            properties.properties.device_type,
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
        );
        let ticks = Ticks::new(
            properties.properties.limits.timestamp_period,
            families[family as usize].timestamp_valid_bits,
        );
        let calibrated = ash::ext::calibrated_timestamps::Device::new(vk_instance, &device);
        // SAFETY: the device was made from `raw` with `capabilities.extensions`.
        let imports = unsafe {
            Imports::new(
                vk_instance,
                raw,
                &device,
                &capabilities.extensions,
                &capabilities.memory,
            )
        };
        #[cfg(unix)]
        let imports = imports.with_sparse(
            &device,
            sparse.then_some(properties.properties.limits.sparse_address_space_size),
        );
        // SAFETY: as above.
        let exports = unsafe { Exports::new(vk_instance, raw, &device, &capabilities.extensions) };
        let feedback = (capabilities.feedback_loop).then(|| {
            ash::ext::attachment_feedback_loop_dynamic_state::Device::new(vk_instance, &device)
        });
        let limits = &properties.properties.limits;
        // The sampler arrays' two defaults count against the device's samplers too.
        let sampler_cap = limits.max_sampler_allocation_count.saturating_sub(2);
        let limits = EncoderLimits {
            max_group_count: limits.max_compute_work_group_count,
            max_shared_memory: limits.max_compute_shared_memory_size,
            max_viewport: limits.max_viewport_dimensions,
            viewport_bounds: limits.viewport_bounds_range,
        };
        let max_block_len = vulkan11
            .max_memory_allocation_size
            .min(vulkan13.max_buffer_size);
        let resolve_modes = ResolveModes {
            depth: vulkan12.supported_depth_resolve_modes,
            stencil: vulkan12.supported_stencil_resolve_modes,
            independent_none: vulkan12.independent_resolve_none == vk::TRUE,
            independent: vulkan12.independent_resolve == vk::TRUE,
        };
        let device_local = (0u32..)
            .zip(capabilities.memory.memory_types_as_slice())
            .filter(|(_, kind)| {
                kind.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .fold(0, |bits, (index, _)| bits | 1 << index);

        let work = Work::new();
        let alignment = memory::buffer_alignment(&device);
        let pool = Pool::new(
            options.block_mappings,
            counters.clone(),
            granularity,
            alignment,
        );
        let submit_calls = Arc::new(SubmitCalls::new(counters.clone()));
        let live = instance.teardowns().live();
        let core = Arc::new(Core {
            instance,
            physical: raw,
            device,
            family,
            extensions,
            consumer_features,
            queues: Queues { runtime, present },
            memory_types,
            device_local,
            unified_memory,
            counters,
            ticks,
            calibrated,
            feedback,
            attachment_reads: capabilities.attachment_reads,
            linear_attachments: (capabilities.features.linear_attachment).linear_color_attachment
                == vk::TRUE,
            resolve_modes,
            max_block_len,
            max_allocations,
            granularity,
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            fail_private: std::sync::atomic::AtomicU32::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            calibration_lateness: std::sync::atomic::AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            tick_spans: Mutex::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            park_region: Mutex::new(None),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            maker_test: options.maker,
            pool,
            submit_calls,
            imports,
            exports,
            claims: Mutex::new(HashMap::new()),
            places: Places::new(),
            filler: OnceLock::new(),
            recorders: Recorders::new(family),
            work,
            done,
            reaper: Reaper::new(),
            stamps: StampPools::new(),
            layout,
            arrays,
            samplers: Mutex::new(SamplerTable::new(sampler_cap)),
            constexprs: Mutex::new(HashMap::new()),
            linear_alignments: Mutex::new(HashMap::new()),
            copied_overs: AtomicUsize::new(0),
            limits,
            ring: Ring::new(),
            ids: Ids::new(),
            heap_ids: HeapIds::default(),
            translator,
            caches,
            kernels: KernelMap::new(),
            entries: metal2vulkan::pipeline::StageEntries::new(),
            closed: AtomicBool::new(false),
            halt: AtomicU8::new(RUNNING),
            _live: live,
        });
        let device = Device {
            shared: Arc::new(Shared {
                core,
                threads: Mutex::new(Threads::default()),
            }),
        };
        device.shared.core.counters.rise(Gauge::HandlesOut);
        // A thread that does not start drops the device, which stops those that did.
        let core = &device.shared.core;
        memory::stock(core);
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

    /// `readWriteTextureSupport`: `MTLReadWriteTextureTier2` on every device above the floor.
    /// Tier 2's 18 formats are storage images Vulkan requires of every device (the four-channel
    /// and the 32-bit ones) or the floor's `shaderStorageImageExtendedFormats` requires (the six 8-
    /// and 16-bit single-channel ones), read and written through `access::read_write`.
    pub fn read_write_texture_support(&self) -> ReadWriteTextureTier {
        ReadWriteTextureTier::Tier2
    }

    /// `argumentBuffersSupport`: `MTLArgumentBuffersTier2` on every device above the floor, which
    /// holds what tier 2 stands on: buffer device addresses (a program writes a buffer's
    /// `gpuAddress` into an argument buffer and indexes through it), and the set-1 arrays'
    /// runtime descriptor arrays, partial binding and update-after-bind of sampled and storage
    /// images (a program writes a texture's `gpuResourceID`, of a writable texture too).
    pub fn argument_buffers_support(&self) -> ArgumentBuffersTier {
        ArgumentBuffersTier::Tier2
    }

    /// `hasUnifiedMemory`: true where Vulkan's device type is integrated or CPU, false for a
    /// discrete or virtual GPU. On a discrete GPU, `Shared` memory is system memory the GPU reads
    /// across its bus, as on a Mac with a discrete GPU.
    pub fn has_unified_memory(&self) -> bool {
        self.shared.core.unified_memory
    }

    /// `supportsTextureSampleCount:`: whether `count` is a sample count the device renders into a
    /// colour and a depth attachment (its `framebufferColorSampleCounts` and
    /// `framebufferDepthSampleCounts`, of which Vulkan requires 1 and 4) that an Apple GPU takes:
    /// 1, 2 or 4 (Apple9's "Maximum sample count in render passes with MSAA" is 4). A count Vulkan
    /// has no bit for (0, 3, above 64) is not. A texture's own format may take fewer
    /// ([`Limit::TextureSamples`](crate::Limit::TextureSamples)).
    pub fn supports_texture_sample_count(&self, count: u32) -> bool {
        let counts = self
            .shared
            .core
            .translator
            .metal_properties()
            .raster_sample_counts;
        count.is_power_of_two() && count <= 64 && counts.as_raw() & count != 0
    }

    /// No Metal counterpart: the device extensions enabled: the floor's first, then each of the
    /// consumer's ([`DeviceOptions::extensions`]) the device offers with its features.
    pub fn extensions(&self) -> &[CString] {
        &self.shared.core.extensions
    }

    /// No Metal counterpart: the consumer's features enabled, each of
    /// [`DeviceOptions::features`] the device reports, in the consumer's order.
    pub fn features(&self) -> &[ConsumerFeature] {
        &self.shared.core.consumer_features
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
    /// lock `f` holds; a present in `f` waits only on semaphores whose signal is already submitted:
    /// the command buffer that presents the image
    /// ([`CommandBuffer::present_drawable`](crate::CommandBuffer::present_drawable)) is
    /// scheduled ([`CommandBuffer::wait_until_scheduled`](crate::CommandBuffer::wait_until_scheduled)
    /// has returned for it, or `f` runs in its scheduled handler).
    pub fn with_present_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let queue = lock(&self.shared.core.queues.present);
        f(*queue)
    }

    /// No Metal counterpart: the device's metrics and gauges. The handle stays readable after the
    /// device is torn down.
    pub fn counters(&self) -> Counters {
        self.shared.core.counters.clone()
    }

    /// What lost the device: the Vulkan call and result of its first loss, a call of the
    /// runtime's or one the consumer reported ([`Device::report_loss`]), None while it is not
    /// lost. Metal's counterpart is the code and description of `MTLCommandBuffer.error`
    /// (`MTLCommandBufferError`'s `OutOfMemory`, `PageFault`, `Timeout`, ...). The runtime ends
    /// every command buffer a loss reaches with
    /// [`CommandBufferError::DeviceLost`](crate::CommandBufferError::DeviceLost), whatever the
    /// cause; this names the call and result behind it ([`DeviceLoss`]). It is set before any
    /// command buffer ends `DeviceLost`, and a later loss never changes it.
    pub fn loss(&self) -> Option<DeviceLoss> {
        self.shared.core.work.lock().loss()
    }

    /// No Metal counterpart: the consumer reports a loss that its own Vulkan call on this device
    /// returned, as a [`DeviceLoss`] of that call's name and result: an acquire, a present or
    /// another swapchain call, such as `vkQueuePresentKHR` returning `ERROR_DEVICE_LOST`. Such a
    /// call can meet the loss before any call of the runtime's, which otherwise learns of it only
    /// at its own next submission or wait. The runtime ends the device's work by the path of its
    /// own loss: the submit thread ends what it holds, the completion thread ends what is in
    /// flight once the device is idle, host waits for pending signals end false, and every
    /// command buffer the loss reaches, each later commit included, ends
    /// [`CommandBufferError::DeviceLost`](crate::CommandBufferError::DeviceLost), as at a loss
    /// the runtime meets itself. [`Device::loss`] then answers `loss`, unless the device was
    /// already lost: the first loss is kept, and a later report changes nothing. The runtime
    /// takes `loss` as given and reads nothing of its result, so a report of a result that is no
    /// error, `SUCCESS` included, ends the device's work all the same. As at any loss, the device
    /// is not recovered.
    pub fn report_loss(&self, loss: DeviceLoss) {
        complete::lose(&self.shared.core, loss);
    }

    /// No Metal counterpart: takes a block of `class` memory of `len` bytes rounded up to 64 KiB
    /// (to the device's `bufferImageGranularity` where that is larger), for the consumer to place
    /// buffers in: a handle ([`Block`]) whose last drop gives it back. A block is a range of one
    /// of the device's allocations, whose one `VkBuffer` spans it
    /// ([`BlockInfo::allocation`](crate::BlockInfo::allocation),
    /// [`BlockInfo::offset`](crate::BlockInfo::offset)).
    ///
    /// It is a free range of the class's allocations: the first that fits, in the order the
    /// allocations were made, with no Vulkan call. An allocation longer than
    /// [`MemoryClass::block_len`] (16 MiB), a long one, which a longer take made for itself,
    /// serves only takes of its own length. Where no free range fits,
    /// it is the start of an allocation made in this call, of 16 MiB or the block's length where
    /// that is longer, whose rest later blocks share, which [`BlockMappings::map`] maps before the
    /// call returns. The block-maker thread makes a class's next allocation of 16 MiB off every
    /// caller's path while the class holds no wholly free unit, an allocation of 16 MiB that no
    /// block holds, so a take of up to 16 MiB lands in memory made ahead once it has made it; a
    /// small take that breaks the class's last whole unit sets off an allocation beside the
    /// consumer's next calls.
    ///
    /// A block's bytes are undefined, new or taken again: the runtime zeroes nothing ahead of
    /// need. They are the consumer's to zero where Metal owes zeros
    /// ([`Device::zero_block_range`] zeroes a range of a Private block ahead of the next commit).
    /// A buffer reads zero where
    /// [`Device::new_buffer`] zeroes its range, as `newBufferWithLength:options:` does in Metal;
    /// Metal is silent on the bytes of a placement heap's memory and of a texture placed there.
    ///
    /// Err: [`Limit::BlockLen`](crate::Limit::BlockLen) past the device's largest allocation or
    /// buffer; [`Limit::BlockIds`](crate::Limit::BlockIds);
    /// [`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations) where the block takes an
    /// allocation past the device's count; [`Limit::AllocationNotMapped`](crate::Limit::AllocationNotMapped)
    /// where the consumer did not map the allocation the take made; [`Error::Vulkan`] where the
    /// device is out of memory or lost; [`Error::Quiesced`].
    pub fn take_block(&self, class: MemoryClass, len: u64) -> Result<Block, Error> {
        let core = &self.shared.core;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let memory = core.take_memory(class, len, Site::take(class))?;
        let info = memory.info();
        let block = memory::Block::new(memory, Arc::clone(core));
        Ok(Block::claim(core, info, block))
    }

    /// No Metal counterpart: stops the runtime's threads and its GPU work, for a process that must
    /// end while a handle of the device is still out. True once the GPU went idle and every thread ended
    /// within `timeout`; false where the timeout passed first, or the call ran in a handler or a
    /// listener (the submit thread, which runs the scheduled handlers, and the completion thread,
    /// which runs the completed handlers and listeners, cannot wait for themselves).
    ///
    /// In order, each wait bounded by `timeout`: the submit thread submits what is ready, ends
    /// every command buffer it holds or has not yet submitted with
    /// [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun), and ends; the
    /// block-maker thread ends; the GPU completes everything submitted; the completion thread
    /// retires it, runs the completed handlers, and ends; then [`BlockMappings::unmap`] runs on
    /// this thread for every allocation still mapped. After it, no runtime thread touches a block,
    /// unless the call answered false. A thread past the timeout ends alone: the completion
    /// thread once the GPU completed what was submitted and it retired that, running its
    /// completed handlers, so what the work held goes then. Such a thread is not joined here; the device's last handle joins
    /// it, with no deadline, before it destroys anything the thread can reach.
    ///
    /// It frees nothing: every allocation's memory and mapping stays valid until the process ends,
    /// or until the device's last handle drops, which then frees it without the hook. Later calls
    /// that need a runtime thread return [`Error::Quiesced`]: taking a block, a command buffer,
    /// an enqueue or commit, a zeroed Private buffer, and a listener.
    ///
    /// The consumer then ends the process with `_exit`, whatever it answered: `exit()` with a live
    /// device is supported on no device (a driver's own exit teardown can fault after work ran;
    /// the crate's README, "Quiesce and exit"). It joins a thread only once that thread has ended
    /// and takes no lock a consumer's thread can hold. A second call returns at once, with the
    /// first call's answer, or false while the first is still running. After a true answer, the
    /// last handle dropped on the consumer's thread tears the device down in that drop. After a
    /// false answer, the consumer keeps the device and ends with `_exit`, or drops its last
    /// handle on its own thread, which joins with no deadline; a handle a handler still holds is
    /// then freed only by the process's end.
    pub fn quiesce(&self, timeout: Duration) -> bool {
        let threads = self.shared.take_threads();
        let deadline = Instant::now().checked_add(timeout);
        let (idle, left) = halt(&self.shared.core, threads, deadline, true);
        self.shared.keep_threads(left);
        idle
    }

    /// No Metal counterpart: gives a block back where `block` is its last handle: the drop of the
    /// handle, made explicit ([`Block`]).
    ///
    /// It takes effect once no [`Buffer`](crate::Buffer) of the block remains and every command
    /// buffer that used it has retired. Its bytes are then a free range of its allocation, which
    /// stays mapped, for a later [`take_block`](Device::take_block) to land in. An allocation none
    /// of whose blocks is held is idle, and one that holds a block is in use. The block-maker
    /// thread frees each allocation that has been idle for 250 ms, the idle age, after
    /// [`BlockMappings::unmap`] of it, one at a time and whatever the device's queues are doing; a
    /// take that lands in an idle allocation before then ends its idle time. Each class keeps a
    /// reserve that it does not free, its wholly free unit: the last idle 16 MiB allocation in the
    /// order the allocations were made. A long allocation, which a take longer than 16 MiB made
    /// for itself, is never the reserve.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's block, whose handle is dropped all the
    /// same.
    pub fn release_block(&self, block: Block) -> Result<(), Error> {
        if !block.is_of(&self.shared.core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        drop(block);
        Ok(())
    }

    /// The core, for the objects made from the device.
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.shared.core
    }

    /// Gives up this handle where it is not the last, which ends nothing; the last comes back,
    /// for the caller to drop once it holds no lock, which ends the device on its thread. A call
    /// that holds a handle while another thread may drop the consumer's last one gives its own up
    /// first, so the consumer's drop is the last and the teardown runs where the contract says.
    pub(crate) fn release(self) -> Option<Device> {
        Arc::into_inner(self.shared).map(|shared| Device {
            shared: Arc::new(shared),
        })
    }

    /// For the device tests: how many handles of the device there are.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn handles(&self) -> usize {
        Arc::strong_count(&self.shared)
    }

    /// A block the consumer holds a handle of. Err: [`Error::NoSuchBlock`].
    pub(crate) fn block(&self, id: BlockId) -> Result<Arc<memory::Block>, Error> {
        let claim = lock(&self.shared.core.claims)
            .get(&id)
            .and_then(Weak::upgrade);
        claim
            .and_then(|claim| claim.block())
            .ok_or(Error::NoSuchBlock(id))
    }

    /// For the device tests: runs `f` with the runtime's queue under its lock.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn with_runtime_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        f(*lock(&self.shared.core.queues.runtime))
    }
}

/// Starts a thread of the runtime's named `name`, which runs `body` holding a slot of the
/// device's [`Gauges::threads`](crate::Gauges::threads) until it ends: every thread the runtime
/// starts starts here. It returns once the thread holds its slot, so a reading of the gauge
/// counts every thread started before it.
pub(crate) fn start(
    name: &str,
    counters: &Counters,
    body: impl FnOnce() + Send + 'static,
) -> std::io::Result<thread::JoinHandle<()>> {
    let counters = counters.clone();
    // Nothing is sent: the thread drops `holding` once it holds its slot, or as it unwinds.
    let (holding, held) = mpsc::channel::<()>();
    let handle = thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let _slot = counters.thread_started();
            drop(holding);
            body();
        })?;
    let _ = held.recv();
    Ok(handle)
}

/// Starts a runtime thread that serves `core` in `role`.
fn spawn(name: &str, core: &Arc<Core>, role: Role, body: fn(Arc<Core>)) -> Result<Runner, Error> {
    let ended = Arc::new(Ended::default());
    let ending = Ending(Arc::clone(&ended));
    let core = Arc::clone(core);
    let counters = core.counters.clone();
    let handle = start(name, &counters, move || {
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

    /// The threads a stop left running, kept for the teardown to join.
    fn keep_threads(&self, left: Threads) {
        let mut guard = lock(&self.threads);
        let threads = &mut *guard;
        let Threads {
            blocks,
            submit,
            complete,
            reaper,
        } = left;
        for (slot, runner) in [
            (&mut threads.blocks, blocks),
            (&mut threads.submit, submit),
            (&mut threads.complete, complete),
            (&mut threads.reaper, reaper),
        ] {
            if runner.is_some() {
                *slot = runner;
            }
        }
    }
}

/// Stops the runtime's threads and its GPU work, and runs the consumer's unmap hook for every
/// allocation still mapped, on this thread: for [`Device::quiesce`] (`quiesce`, with its deadline)
/// and for teardown, with none. Only the first call stops anything; a later one answers at once
/// and hands `threads` back. Each wait is bounded by `deadline`: a thread that has not ended by
/// then is left running, not joined, and handed back, with this thread's own, for the teardown to
/// join ([`join`]). True once the GPU went idle and every thread ended by `deadline`.
fn halt(
    core: &Arc<Core>,
    threads: Threads,
    deadline: Option<Instant>,
    quiesce: bool,
) -> (bool, Threads) {
    match core
        .halt
        .compare_exchange(RUNNING, HALTING, Ordering::AcqRel, Ordering::Acquire)
    {
        Ok(_) => {}
        Err(halt) => return (halt == HALTED_IDLE, threads),
    }
    core.closed.store(true, Ordering::Release);
    let Threads {
        blocks,
        submit,
        complete,
        reaper,
    } = threads;
    let mut left = Threads::default();
    // Where a scheduled handler quiesces, on the submit thread, the thread's own submission and
    // the completion thread wait for the handler to return: neither is waited for.
    let on_submit = role(core) == Role::Submit;
    // Submission stops: what is ready goes, and what is held ends.
    core.work.stop();
    let submitted = finish(submit, core, deadline, &mut left.submit);
    let mut idle = submitted;
    core.pool.stop();
    idle &= finish(blocks, core, deadline, &mut left.blocks);
    if quiesce && !on_submit {
        idle &= core.wait_progress(core.work.progress(), deadline);
    }
    // The submit thread stops the completion thread as it ends; here only where it has ended, or
    // never ran. One past the deadline stops it when it ends alone.
    if submitted {
        core.done.stop(core);
    }
    match on_submit {
        false => idle &= finish(complete, core, deadline, &mut left.complete),
        true => left.complete = complete,
    }
    core.reaper.stop();
    idle &= finish(reaper, core, deadline, &mut left.reaper);
    // The consumer's mappings go here, on the thread that stops the device, whichever thread
    // later drops the core's last reference and frees the memory.
    core.pool.unmap_all();
    let answer = if idle { HALTED_IDLE } else { HALTED_LATE };
    core.halt.store(answer, Ordering::Release);
    (idle, left)
}

/// Waits for a thread to end, up to `deadline`, and joins it once it has: true. A thread past the
/// deadline is left running, not joined, its handle put in `left` for the teardown to join; so is
/// this thread's own, which is never waited for by itself (the completion thread where a
/// completed handler or a listener quiesces, the submit thread where a scheduled handler does,
/// the reaper where it runs a teardown) and ends once the call returns.
fn finish(
    runner: Option<Runner>,
    core: &Arc<Core>,
    deadline: Option<Instant>,
    left: &mut Option<Runner>,
) -> bool {
    let Some(runner) = runner else {
        return true;
    };
    if role(core) == runner.role || !runner.ended.wait(deadline) {
        *left = Some(runner);
        return false;
    }
    // A thread that panicked has ended all the same.
    let _ = runner.handle.join();
    true
}

/// Joins each thread a stop left running ([`halt`]), with no deadline, before the teardown
/// destroys anything such a thread can reach (the device, the instance, a queue): after a late
/// quiesce, the threads still in the driver or ending alone. This thread's own handle goes to the
/// instance, for [`Instance::wait_until_devices_torn_down`] to join: it ends once the teardown
/// returns.
fn join(core: &Arc<Core>, threads: Threads) {
    let Threads {
        blocks,
        submit,
        complete,
        reaper,
    } = threads;
    for runner in [submit, blocks, complete, reaper] {
        let mut own = None;
        finish(runner, core, None, &mut own);
        if let Some(own) = own {
            core.instance.teardowns().hand(own);
        }
    }
}

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

/// The features device creation enables: the floor's of `capabilities`, and each of `asked` the
/// device reports, which are returned too.
///
/// # Safety
///
/// `raw` is a physical device of `instance`, of Vulkan 1.3.
unsafe fn consumer_features(
    instance: &ash::Instance,
    raw: vk::PhysicalDevice,
    capabilities: &floor::Capabilities,
    asked: &[ConsumerFeature],
) -> (floor::Features, Vec<ConsumerFeature>) {
    let mut enabled = capabilities.features;
    if asked.is_empty() {
        return (enabled, Vec::new());
    }
    let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::<'static>::default();
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::<'static>::default();
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::<'static>::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan11)
        .push_next(&mut vulkan12)
        .push_next(&mut vulkan13);
    // SAFETY: the caller's contract; the structures are Vulkan 1.3's and live through the call.
    unsafe { instance.get_physical_device_features2(raw, &mut features) };
    let core = features.features;
    vulkan11.p_next = std::ptr::null_mut();
    vulkan12.p_next = std::ptr::null_mut();
    vulkan13.p_next = std::ptr::null_mut();
    let mut reported = floor::Features {
        core,
        vulkan11,
        vulkan12,
        vulkan13,
        ..floor::Features::default()
    };
    let mut granted = Vec::new();
    for &feature in asked {
        let field = floor::consumer_feature(feature).field;
        if *field(&mut reported) == vk::TRUE && !granted.contains(&feature) {
            *field(&mut enabled) = vk::TRUE;
            granted.push(feature);
        }
    }
    (enabled, granted)
}

/// The memory types the floor chooses on a new device, the completion thread's semaphores, set 1's
/// arrays and the pipeline layout.
///
/// # Safety
///
/// `device` was made with `capabilities.features`.
unsafe fn objects(
    device: &ash::Device,
    capabilities: &floor::Capabilities,
    counters: &Counters,
) -> Result<(floor::MemoryTypes, Done, Arrays, vk::PipelineLayout), Error> {
    // SAFETY: the caller's contract.
    let memory_types =
        unsafe { floor::memory_types(device, capabilities) }.map_err(Error::BelowFloor)?;
    // SAFETY: the device is live.
    let made = unsafe { Done::new(device) }.and_then(|mut done| {
        // SAFETY: the device is live, with the floor's features.
        let arrays = match unsafe { Arrays::new(device, counters, capabilities.attachment_reads) } {
            Ok(arrays) => arrays,
            Err(error) => {
                // SAFETY: the completion objects are unused.
                unsafe { done.destroy(device) };
                return Err(error);
            }
        };
        let push = [vk::PushConstantRange {
            stage_flags: PUSH_STAGES,
            offset: 0,
            size: PUSH_BYTES,
        }];
        let sets = arrays.set_layouts();
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&sets)
            .push_constant_ranges(&push);
        // SAFETY: the device is live, and the floor holds `PUSH_BYTES` of push constants and set
        // 1's update-after-bind descriptors in one stage.
        match unsafe { device.create_pipeline_layout(&info, None) } {
            Ok(layout) => Ok((done, arrays, layout)),
            Err(result) => {
                // SAFETY: the completion objects and set 1 are unused.
                unsafe {
                    arrays.destroy(device, counters);
                    done.destroy(device);
                }
                Err(Error::Vulkan {
                    call: "vkCreatePipelineLayout",
                    result,
                })
            }
        }
    });
    let (done, arrays, layout) = made?;
    Ok((memory_types, done, arrays, layout))
}

/// The device's limits a dispatch or a render encoder's state is checked against.
pub(crate) struct EncoderLimits {
    /// `maxComputeWorkGroupCount`.
    pub(crate) max_group_count: [u32; 3],
    /// `maxComputeSharedMemorySize`.
    pub(crate) max_shared_memory: u32,
    /// `maxViewportDimensions`.
    pub(crate) max_viewport: [u32; 2],
    /// `viewportBoundsRange`.
    pub(crate) viewport_bounds: [f32; 2],
}

impl Core {
    /// The gauges of the device's stage entries ([`Gauges::stage_entries`] and
    /// [`Gauges::stage_entry_bytes`]), read after a creation moved them.
    ///
    /// [`Gauges::stage_entries`]: crate::Gauges::stage_entries
    /// [`Gauges::stage_entry_bytes`]: crate::Gauges::stage_entry_bytes
    pub(crate) fn note_entries(&self) {
        let (entries, decodes, bytes) = self.entries.held();
        self.counters.set(Gauge::StageEntries, entries + decodes);
        self.counters.set(Gauge::StageEntryBytes, bytes);
    }

    /// The memory type of a class's blocks.
    /// What the device makes of an image of `info`'s format, type, tiling, usage and flags;
    /// `None` where it makes none.
    pub(crate) fn image_format_properties(
        &self,
        info: &vk::PhysicalDeviceImageFormatInfo2<'_>,
    ) -> Option<vk::ImageFormatProperties> {
        let mut properties = vk::ImageFormatProperties2::default();
        // SAFETY: the physical device is the instance's, which is Vulkan 1.3.
        unsafe {
            self.instance
                .vk()
                .get_physical_device_image_format_properties2(self.physical, info, &mut properties)
        }
        .ok()
        .map(|()| properties.image_format_properties)
    }

    /// The device's features for `format`: its linear tiling's, then its optimal tiling's.
    pub(crate) fn format_features(
        &self,
        format: vk::Format,
    ) -> (vk::FormatFeatureFlags2, vk::FormatFeatureFlags2) {
        let mut three = vk::FormatProperties3::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut three);
        // SAFETY: the physical device is the instance's, which is Vulkan 1.3.
        unsafe {
            (self.instance.vk()).get_physical_device_format_properties2(
                self.physical,
                format,
                &mut properties,
            )
        };
        (three.linear_tiling_features, three.optimal_tiling_features)
    }

    pub(crate) fn memory_type(&self, class: MemoryClass) -> u32 {
        match class {
            MemoryClass::Private => self.memory_types.private,
            MemoryClass::Shared => self.memory_types.shared,
        }
    }

    /// Whether an allocation of `class` in its own type is to fail as out of device memory: the
    /// device tests' stand-in for a full device, which only they set.
    pub(crate) fn fails_private(&self, class: MemoryClass) -> bool {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if class == MemoryClass::Private {
            let fail = |left: u32| left.checked_sub(1);
            let ordering = Ordering::Relaxed;
            return self
                .fail_private
                .try_update(ordering, ordering, fail)
                .is_ok();
        }
        let _ = class;
        false
    }

    /// Whether Vulkan's device type is integrated or CPU ([`Device::has_unified_memory`]): then a
    /// Managed buffer is its Shared range, with no device copy; else every Managed buffer keeps
    /// one.
    pub(crate) fn unified_memory(&self) -> bool {
        self.unified_memory
    }

    /// A block of `class` for `len` bytes ([`Pool::take`]).
    ///
    /// Err: what [`Pool::take`] returns; [`Limit::BlockLen`](crate::Limit::BlockLen) past the
    /// device's longest allocation or buffer.
    pub(crate) fn take_memory(
        &self,
        class: MemoryClass,
        len: u64,
        site: Site,
    ) -> Result<Memory, Error> {
        self.pool.take(self, class, len, self.max_block_len, site)
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
        // The last handle is gone, and the teardown starts.
        self.core.counters.fall(Gauge::HandlesOut);
        let threads = self.take_threads();
        let teardown = move |core: &Arc<Core>| {
            // Every thread is joined before anything it can reach goes: those this stop waits
            // for, and those a late quiesce left running.
            let (_, left) = halt(core, threads, None, false);
            join(core, left);
            // The blocks the consumer still holds handles of go back to the pool once the threads
            // have stopped and the consumer's unmap hook has returned for each allocation
            // (`halt`), for the core to free; a handle dropped later finds its claim empty. No buffer
            // of them is left: the consumer's each hold a handle of the device, and the command
            // buffers that kept one have retired.
            let claims: Vec<Weak<Claim>> = lock(&core.claims).drain().map(|(_, c)| c).collect();
            let blocks: Vec<Arc<memory::Block>> = (claims.iter())
                .filter_map(Weak::upgrade)
                .filter_map(|claim| claim.take())
                .collect();
            drop(blocks);
            // The map's kernels hold the core: a kernel no state holds goes now, its modules and
            // pipelines freed with the core.
            core.kernels.clear(&core.counters);
        };
        if matches!(role(&self.core), Role::Complete | Role::Submit) {
            // A handler or listener dropped the last handle: the reaper tears down, once the
            // handler returns and its thread can end.
            self.core.reaper.give(Job::Teardown(Box::new(teardown)));
        } else {
            teardown(&self.core);
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        self.wait_idle();
        // The stage entries go with the core.
        self.counters.set(Gauge::StageEntries, 0);
        self.counters.set(Gauge::StageEntryBytes, 0);
        // The runtime's threads have ended: the core is dropped only once none holds it, and with
        // it every command buffer and upload, each of which held what it used. The teardown ran
        // the consumer's unmap hook, so freeing runs none.
        self.recorders.destroy_all(&self.device);
        // The reaper has ended; what it was given since is freed here (a texture's place among
        // it, which may give a texture block back to the pool), and then the pools of the queues
        // still alive.
        self.reaper.drain(self);
        // SAFETY: the caches are the device's, and no creation runs: every handle is gone.
        unsafe { self.caches.close(&self.device, &self.counters) };
        // The store's removal thread, the one thread no stop ends, is waited for here, so every
        // gauge, `Gauges::threads` among them, reads zero once the core is gone.
        self.caches.join_removal();
        if let Some(region) = self.work.take_region() {
            self.counters
                .fall_by(Gauge::StagingBytes, crate::staging::REGION_LEN);
            // SAFETY: every upload that copied through it retired, and no call reads it.
            unsafe { region.free(self) };
        }
        memory::free_all(self);
        #[cfg(unix)]
        self.imports.destroy(&self.device);
        self.stamps.destroy(&self.device, &self.counters);
        // SAFETY: the device is idle; nothing reads the ring.
        let ring_blocks = unsafe { self.ring.destroy(&self.device) };
        let filler = self.filler.take();
        // SAFETY: the device is idle, and nothing made from it is left but these objects.
        let zero_source = unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
            self.arrays.destroy(&self.device, &self.counters);
            self.done.destroy(&self.device);
            let zero_source = filler.is_some_and(|mut filler| filler.destroy(&self.device));
            self.device.destroy_device(None);
            zero_source
        };
        // The ring's and the zero source's gauges fall last, once the `VkDevice` is destroyed:
        // where the block-maker thread made them, every gauge reads zero only when the teardown
        // has made its last Vulkan call, so a consumer whose last handle dropped in a handler can
        // wait for that before it exits.
        for _ in 0..ring_blocks + usize::from(zero_source) {
            self.counters.fall(Gauge::Blocks);
        }
    }
}
