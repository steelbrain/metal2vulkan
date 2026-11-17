//! The device: one `VkDevice` for every command queue, every guest process and the compositor;
//! its queues, its memory blocks and its teardown.

use crate::counters::Counters;
use crate::error::Error;
use crate::fill::Filler;
use crate::floor;
use crate::instance::{Instance, PhysicalDevice};
use crate::memory::{self, Block, BlockId, BlockInfo, BlockMappings, MemoryClass, Pool, BLOCK_LEN};
use ash::vk;
use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};

/// `MTLDevice`, over one `VkDevice` that every command queue, every guest process and the
/// compositor share.
///
/// A cheap handle: clones share one device, and every object made from it holds a clone. Blocks
/// the consumer holds do not. The device is torn down when the last handle is dropped: the
/// block-maker thread is joined and the GPU goes idle; then, on the dropping thread, every block
/// is freed, after [`BlockMappings::unmap`] for each the consumer was handed, and the `VkDevice`
/// is destroyed, with every gauge at zero. No thread of the runtime outlives it,
/// and nothing relies on a static destructor or an `atexit` handler, so the consumer drops every
/// handle before the process exits (a driver may unmap device memory inside `exit()`).
#[derive(Clone)]
pub struct Device {
    shared: Arc<Shared>,
}

/// What the device's handles share: the core, the blocks the consumer holds, and the runtime's
/// threads, which hold the core but no handle.
struct Shared {
    core: Arc<Core>,
    claims: Mutex<HashMap<BlockId, Arc<Block>>>,
    threads: Vec<JoinHandle<()>>,
}

/// The device itself: what the runtime's threads and every block hold.
pub(crate) struct Core {
    instance: Instance,
    physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    family: u32,
    pub(crate) queues: Queues,
    memory_types: floor::MemoryTypes,
    unified_memory: bool,
    pub(crate) counters: Counters,
    /// The longest block: the device's `maxMemoryAllocationSize` or `maxBufferSize`, whichever is
    /// less.
    max_block_len: u64,
    pub(crate) pool: Pool,
    pub(crate) filler: Filler,
}

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
}

/// Takes a lock. A panic while another thread held it leaves what it guards usable (a queue, the
/// pool's lists, the claims), so the lock is taken whatever that thread left behind.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Device {
    /// `MTLCreateSystemDefaultDevice`, for the physical device the consumer chose.
    ///
    /// Checks the floor ([`floor::check`]), then makes one `VkDevice` with the floor's features
    /// and extensions and each of the four query-decided features the device has, and chooses its
    /// memory types on it ([`floor::memory_types`]). It takes one queue of the floor's queue
    /// family for the runtime and, where that family has a second queue, a queue for presenting;
    /// where it has one, presenting shares the runtime's queue and its lock. It starts the
    /// block-maker thread, which makes one block of each memory class ready.
    ///
    /// Err: [`Error::BelowFloor`] with every item the device lacks; [`Error::Vulkan`] where a
    /// Vulkan call fails; [`Error::Thread`].
    pub fn new(physical: &PhysicalDevice, options: DeviceOptions) -> Result<Device, Error> {
        let instance = physical.instance().clone();
        let vk_instance = instance.vk();
        let raw = physical.raw();
        // SAFETY: `raw` is one of `instance`'s physical devices, and the instance was created at
        // the floor's version.
        let capabilities = unsafe { floor::check(vk_instance, raw) }.map_err(Error::BelowFloor)?;
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
        } = capabilities.features;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut workgroup_layout);
        let extensions: Vec<*const c_char> = capabilities
            .extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&extensions)
            .push_next(&mut features);
        // SAFETY: `info` and everything it points to live through the call; the floor checked
        // that the device offers every feature and extension it enables.
        let device = unsafe { vk_instance.create_device(raw, &info, None) }
            .map_err(Error::vulkan("vkCreateDevice"))?;
        let counters = Counters::new();
        // SAFETY: the device was made from `raw` with `capabilities.features`, and the family is
        // its queues' family.
        let memory = unsafe { memory_types_and_filler(&device, &capabilities, family, &counters) };
        let (memory_types, filler) = match memory {
            Ok(memory) => memory,
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
        let max_block_len = vulkan11
            .max_memory_allocation_size
            .min(vulkan13.max_buffer_size);

        let core = Arc::new(Core {
            instance,
            physical: raw,
            device,
            family,
            queues: Queues { runtime, present },
            memory_types,
            unified_memory,
            counters,
            max_block_len,
            pool: Pool::new(options.block_mappings),
            filler,
        });
        let maker = Arc::clone(&core);
        let blocks = thread::Builder::new()
            .name("m2v-blocks".to_string())
            .spawn(move || memory::run(maker))
            .map_err(Error::Thread)?;
        Ok(Device {
            shared: Arc::new(Shared {
                core,
                claims: Mutex::new(HashMap::new()),
                threads: vec![blocks],
            }),
        })
    }

    /// `hasUnifiedMemory`: true where Vulkan's device type is integrated or CPU, false for a
    /// discrete or virtual GPU. On a discrete GPU, `Shared` memory is system memory the GPU reads
    /// across its bus, as on a Mac with a discrete GPU.
    pub fn has_unified_memory(&self) -> bool {
        self.shared.core.unified_memory
    }

    /// No Metal counterpart: the memory types the device's buffers are allocated from, as the
    /// floor chose them on the device ([`floor::memory_types`]).
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
    /// out of memory or lost.
    pub fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, Error> {
        let core = &self.shared.core;
        let memory = if len <= BLOCK_LEN {
            core.pool.take(class)?
        } else {
            let len = memory::dedicated_len(len, core.max_block_len)?;
            memory::make(core, class, len, core.pool.number()?, false)?
        };
        let info = memory.info();
        lock(&self.shared.claims).insert(memory.id, Arc::new(Block::new(memory, Arc::clone(core))));
        Ok(info)
    }

    /// No Metal counterpart: gives a block back.
    ///
    /// It takes effect once no [`Buffer`](crate::Buffer) of the block remains. A pooled block is
    /// then zeroed and kept ready, with the same number, address and mapping, while its class has
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
}

/// The memory types the floor chooses on a new device, and the zero fill's objects.
///
/// # Safety
///
/// `device` was made with `capabilities.features`, and `family` is its queues' family.
unsafe fn memory_types_and_filler(
    device: &ash::Device,
    capabilities: &floor::Capabilities,
    family: u32,
    counters: &Counters,
) -> Result<(floor::MemoryTypes, Filler), Error> {
    // SAFETY: the caller's contract.
    let memory_types =
        unsafe { floor::memory_types(device, capabilities) }.map_err(Error::BelowFloor)?;
    // SAFETY: the caller's contract; the Shared type is host-visible and host-coherent, and the
    // floor chose it for buffers of every usage.
    let filler = unsafe { Filler::new(device, family, memory_types.shared, counters) }?;
    Ok((memory_types, filler))
}

impl Core {
    /// The memory type of a class's blocks.
    pub(crate) fn memory_type(&self, class: MemoryClass) -> u32 {
        match class {
            MemoryClass::Private => self.memory_types.private,
            MemoryClass::Shared => self.memory_types.shared,
        }
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
        self.core.pool.stop();
        for thread in self.threads.drain(..) {
            // A thread that panicked has stopped all the same.
            let _ = thread.join();
        }
        // The blocks the consumer still holds go back to the pool, for the core to free. No
        // buffer of them is left: each holds a handle of the device.
        self.claims
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        {
            // `vkDeviceWaitIdle` needs every queue of the device externally synchronized.
            let _runtime = lock(&self.queues.runtime);
            let _present = (!Arc::ptr_eq(&self.queues.runtime, &self.queues.present))
                .then(|| lock(&self.queues.present));
            // SAFETY: every queue is held. A lost device fails the wait; teardown goes on.
            let _ = unsafe { self.device.device_wait_idle() };
        }
        // The runtime's threads are joined: the core is dropped only once none holds it.
        for (memory, handed) in self.pool.remaining() {
            memory::free(self, memory, handed);
        }
        // SAFETY: the device is idle, and nothing made from it is left but the fill's objects.
        unsafe {
            self.filler.destroy(&self.device, &self.counters);
            self.device.destroy_device(None);
        }
    }
}
