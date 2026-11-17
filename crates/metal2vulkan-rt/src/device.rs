//! The device: one `VkDevice` for every command queue, every guest process and the compositor;
//! its queues; and its teardown.

use crate::counters::Counters;
use crate::error::Error;
use crate::floor;
use crate::instance::{Instance, PhysicalDevice};
use ash::vk;
use std::ffi::c_char;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// `MTLDevice`, over one `VkDevice` that every command queue, every guest process and the
/// compositor share.
///
/// A cheap handle: clones share one device, and every object made from it holds a clone. The
/// device is torn down when the last of them is dropped: it waits until the GPU is idle, then
/// destroys the `VkDevice`. No thread of the runtime outlives it, and nothing relies on a static
/// destructor or an `atexit` handler, so the consumer drops every handle before the process exits
/// (a driver may unmap device memory inside `exit()`).
#[derive(Clone)]
pub struct Device {
    shared: Arc<Shared>,
}

struct Shared {
    instance: Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    family: u32,
    queues: Queues,
    memory_types: floor::MemoryTypes,
    unified_memory: bool,
    counters: Counters,
}

/// The device's queues, each behind the lock that Vulkan's external synchronization of a queue
/// needs. Where the family has a second queue, presenting has it to itself; where it has one, the
/// presenting queue is the runtime's own queue and its lock.
struct Queues {
    runtime: Arc<Mutex<vk::Queue>>,
    present: Arc<Mutex<vk::Queue>>,
}

/// The queue family the runtime takes and how many of its queues: the first family that does both
/// graphics and compute, with two queues where it has them (the second for presenting) and one
/// where it has one. `None` where no family does both.
pub(crate) fn queue_plan(families: &[vk::QueueFamilyProperties]) -> Option<(u32, u32)> {
    let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    let index = families
        .iter()
        .position(|family| family.queue_flags.contains(wanted) && family.queue_count > 0)?;
    let family = u32::try_from(index).ok()?;
    Some((family, families[index].queue_count.min(2)))
}

/// Takes a queue's lock. A panic while another thread held it leaves the queue itself usable, so
/// the lock is taken whatever that thread left behind.
fn lock(queue: &Mutex<vk::Queue>) -> MutexGuard<'_, vk::Queue> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Device {
    /// `MTLCreateSystemDefaultDevice`, for the physical device the consumer chose.
    ///
    /// Checks the floor ([`floor::check`]), then makes one `VkDevice` with the floor's features
    /// and extensions and each of the four query-decided features the device has, and chooses its
    /// memory types on it ([`floor::memory_types`]). It takes one graphics-and-compute queue for
    /// the runtime and, where that family has a second queue, a queue for presenting; where it has
    /// one, presenting shares the runtime's queue and its lock.
    ///
    /// Err: [`Error::BelowFloor`] with every item the device lacks;
    /// [`Error::NoGraphicsComputeQueue`]; [`Error::Vulkan`] where `vkCreateDevice` fails.
    pub fn new(physical: &PhysicalDevice) -> Result<Device, Error> {
        let instance = physical.instance().clone();
        let vk_instance = instance.vk();
        let raw = physical.raw();
        // SAFETY: `raw` is one of `instance`'s physical devices, and the instance was created at
        // the floor's version.
        let capabilities = unsafe { floor::check(vk_instance, raw) }.map_err(Error::BelowFloor)?;
        // SAFETY: as above.
        let families = unsafe { vk_instance.get_physical_device_queue_family_properties(raw) };
        let (family, count) = queue_plan(&families).ok_or(Error::NoGraphicsComputeQueue)?;

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
        // SAFETY: the device was made from `raw` with `capabilities.features`.
        let memory_types = match unsafe { floor::memory_types(&device, &capabilities) } {
            Ok(memory_types) => memory_types,
            Err(below) => {
                // SAFETY: nothing has been made from the device.
                unsafe { device.destroy_device(None) };
                return Err(Error::BelowFloor(below));
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
        // SAFETY: as for the floor check.
        let device_type = unsafe { vk_instance.get_physical_device_properties(raw) }.device_type;
        let unified_memory = matches!(
            device_type,
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
        );
        Ok(Device {
            shared: Arc::new(Shared {
                instance,
                physical: raw,
                device,
                family,
                queues: Queues { runtime, present },
                memory_types,
                unified_memory,
                counters: Counters::new(),
            }),
        })
    }

    /// `hasUnifiedMemory`: true where Vulkan's device type is integrated or CPU, false for a
    /// discrete or virtual GPU. On a discrete GPU, `Shared` memory is system memory the GPU reads
    /// across its bus, as on a Mac with a discrete GPU.
    pub fn has_unified_memory(&self) -> bool {
        self.shared.unified_memory
    }

    /// No Metal counterpart: the memory types the device's buffers are allocated from, as the
    /// floor chose them on the device ([`floor::memory_types`]).
    pub fn memory_types(&self) -> floor::MemoryTypes {
        self.shared.memory_types
    }

    /// No Metal counterpart: the Vulkan handles, for the consumer's swapchain and presents. The
    /// runtime keeps owning them; the consumer destroys none of them, and destroys whatever it
    /// makes with them before the device is torn down.
    pub fn raw(&self) -> RawDevice<'_> {
        RawDevice {
            instance: self.shared.instance.vk(),
            physical_device: self.shared.physical,
            device: &self.shared.device,
            queue_family_index: self.shared.family,
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
        let queue = lock(&self.shared.queues.present);
        f(*queue)
    }

    /// No Metal counterpart: the device's metrics and gauges. The handle stays readable after the
    /// device is torn down.
    pub fn counters(&self) -> Counters {
        self.shared.counters.clone()
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
        // `vkDeviceWaitIdle` needs every queue of the device externally synchronized.
        let _runtime = lock(&self.queues.runtime);
        let _present = (!Arc::ptr_eq(&self.queues.runtime, &self.queues.present))
            .then(|| lock(&self.queues.present));
        // SAFETY: every queue is held, and no object made from the device is alive: each holds a
        // clone of it. A lost device fails the wait; it is destroyed all the same.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
