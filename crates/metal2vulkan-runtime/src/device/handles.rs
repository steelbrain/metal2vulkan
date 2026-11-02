//! The raw Vulkan objects of a runtime device (`Device::raw`).

use super::Device;
use ash::vk;
use std::ffi::CStr;

/// No Metal counterpart: the Vulkan objects behind a runtime device, with the function tables the
/// runtime loaded for them: what the caller needs to read the device, or to make its own objects
/// on it (a surface's swapchain, for example). The runtime keeps owning all of them; the
/// references live as long as the borrow of the device.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct DeviceHandles<'a> {
    /// The loader entry the instance was made through: its `vkGetInstanceProcAddr`.
    pub entry: &'a ash::Entry,
    /// The instance, with its instance-level commands.
    pub instance: &'a ash::Instance,
    /// The `VkApplicationInfo::apiVersion` the instance was created with.
    pub instance_api_version: metal2vulkan::vk::ApiVersion,
    /// The physical device the device was made on.
    pub physical_device: vk::PhysicalDevice,
    /// The device, with its device-level commands as the runtime loaded them.
    pub device: &'a ash::Device,
    /// The device's primary queue family: the first of [`DeviceHandles::queue_families`], and,
    /// on a device of one family, the family every `VkQueue` of the device's queues belongs to.
    pub queue_family: u32,
    /// Every queue family the device's queues belong to, the primary first, each once, the
    /// families of its [`reserved_queues`](Self::reserved_queues) among them. A device
    /// the runtime creates where every queue family is equivalent (the same flags, queue count,
    /// timestamp bits and transfer granularity, as MoltenVK's) takes queues of all of them, so
    /// its queues run side by side; its buffers and images are then concurrent among all these
    /// families, and a buffer or image the caller wraps must be too. An adopted device has its
    /// one family. [`CommandQueue::with_vk_queue`](super::super::CommandQueue::with_vk_queue)
    /// names the family of each queue's own `VkQueue`.
    pub queue_families: &'a [u32],
    /// The extra queues the device made for the caller
    /// ([`DeviceOptions::extra_queues`](super::DeviceOptions::extra_queues)), in the options'
    /// order, leaving out each the device had no queue for: none for an adopted device. The
    /// runtime never submits to them.
    pub extra_queues: &'a [ExtraQueue],
    /// The queues of its own families the device reserved for the caller
    /// ([`DeviceOptions::reserved_queues`](super::DeviceOptions::reserved_queues)), as many as
    /// it had to spare, at most as many as asked: none for an adopted device. The runtime never
    /// submits to them.
    pub reserved_queues: &'a [ReservedQueue],
}

/// No Metal counterpart: one of the queues of its own queue families a created device reserved
/// for the caller out of the round robin its command queues share
/// ([`DeviceOptions::reserved_queues`](super::DeviceOptions::reserved_queues)), which
/// [`DeviceHandles::reserved_queues`] returns. Metal has no queue a program owns apart from its
/// command queues: a `CommandQueue` here may share its `VkQueue` with another command queue
/// (`WorkCounts::queues_sharing_a_vk_queue`), and the runtime submits to that `VkQueue` itself,
/// while no runtime work ever goes to this one. No command queue takes it, presenting or not,
/// and no internal or staging copy goes there: the caller is its only submitter, and
/// synchronizes its submissions to it externally, as Vulkan requires. Its family is one of
/// [`DeviceHandles::queue_families`], which the device's buffers and images are shared among, so
/// a submission to it may use them with no ownership transfer, under their handles' contracts
/// ([`Buffer::vk_buffer`](crate::Buffer::vk_buffer),
/// [`Texture::vk_image`](crate::Texture::vk_image)): while the object lives, ordered against the
/// runtime's work on it through semaphores or completion, and an image the runtime made left in
/// `VK_IMAGE_LAYOUT_GENERAL`. It lives as long as the device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservedQueue {
    /// The queue's family, one of [`DeviceHandles::queue_families`].
    pub family: u32,
    /// The queue.
    pub queue: vk::Queue,
}

/// No Metal counterpart: one of the queues a created device made for the caller beside its own
/// ([`DeviceOptions::extra_queues`](super::DeviceOptions::extra_queues)), which
/// [`DeviceHandles::extra_queues`] returns. The runtime never submits to it: the caller is its
/// only submitter, and synchronizes its submissions to it externally, as Vulkan requires. It
/// lives as long as the device. Where its family is not one of
/// [`DeviceHandles::queue_families`], the device's own buffers and images are not shared with
/// it: a submission to it uses only the caller's objects made to be shared with its family.
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ExtraQueue {
    /// The flags the options asked for, which the family's contain.
    pub flags: vk::QueueFlags,
    /// The queue's family.
    pub family: u32,
    /// The queue.
    pub queue: vk::Queue,
}

impl std::fmt::Debug for ExtraQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtraQueue")
            .field("flags", &QueueFlagsText(self.flags))
            .field("family", &self.family)
            .field("queue", &self.queue)
            .finish()
    }
}

/// `VkQueueFlags` as text: its bits' names joined by `|`, any bit without a name in hex (`ash`'s
/// flags print only with its `debug` feature, which the runtime leaves off).
pub(crate) struct QueueFlagsText(pub(crate) vk::QueueFlags);

impl std::fmt::Debug for QueueFlagsText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const NAMES: [(vk::QueueFlags, &str); 8] = [
            (vk::QueueFlags::GRAPHICS, "GRAPHICS"),
            (vk::QueueFlags::COMPUTE, "COMPUTE"),
            (vk::QueueFlags::TRANSFER, "TRANSFER"),
            (vk::QueueFlags::SPARSE_BINDING, "SPARSE_BINDING"),
            (vk::QueueFlags::PROTECTED, "PROTECTED"),
            (vk::QueueFlags::VIDEO_DECODE_KHR, "VIDEO_DECODE_KHR"),
            (vk::QueueFlags::VIDEO_ENCODE_KHR, "VIDEO_ENCODE_KHR"),
            (vk::QueueFlags::OPTICAL_FLOW_NV, "OPTICAL_FLOW_NV"),
        ];
        let mut parts: Vec<String> = (NAMES.iter())
            .filter(|(bit, _)| self.0.contains(*bit))
            .map(|(_, name)| (*name).to_owned())
            .collect();
        let named = NAMES
            .iter()
            .fold(vk::QueueFlags::empty(), |all, (bit, _)| all | *bit);
        let rest = self.0.as_raw() & !named.as_raw();
        if rest != 0 || parts.is_empty() {
            parts.push(format!("{rest:#x}"));
        }
        f.write_str(&parts.join(" | "))
    }
}

impl std::fmt::Debug for DeviceHandles<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceHandles")
            .field("instance", &self.instance.handle())
            .field("instance_api_version", &self.instance_api_version)
            .field("physical_device", &self.physical_device)
            .field("device", &self.device.handle())
            .field("queue_family", &self.queue_family)
            .field("queue_families", &self.queue_families)
            .field("extra_queues", &self.extra_queues)
            .field("reserved_queues", &self.reserved_queues)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// The device's raw Vulkan objects: the consumer's own for an adopted device, the runtime's
    /// for one it created.
    ///
    /// # Safety
    ///
    /// The objects stay their owner's: the caller destroys none of them, uses them only while
    /// the device lives, and submits to the queues the runtime submits to only through
    /// [`CommandQueue::with_vk_queue`](super::super::CommandQueue::with_vk_queue) (the extra and
    /// reserved queues are the caller's alone). On MoltenVK, a
    /// device with a debug-utils messenger may find its first word overwritten: the calls the
    /// Vulkan loader serves for the device itself (`vkAllocateCommandBuffers`, `vkGetDeviceQueue`,
    /// `vkGetDeviceProcAddr`, `vkDestroyDevice`, `vkCreateSwapchainKHR`:
    /// [`Device::with_loader_word`] lists them) then fault, even through
    /// [`DeviceHandles::device`], so the caller loads its commands through
    /// [`Device::device_proc_addr`] and makes those calls inside [`Device::with_loader_word`].
    pub unsafe fn raw(&self) -> DeviceHandles<'_> {
        let shared = self.shared();
        let instance = &shared.instance;
        DeviceHandles {
            entry: &instance.entry,
            instance: &instance.raw,
            instance_api_version: instance.api_version,
            physical_device: shared.physical,
            device: shared.vk(),
            queue_family: shared.queue_family,
            queue_families: &shared.queue_families,
            extra_queues: &shared.extra_queues,
            reserved_queues: &shared.reserved_queues,
        }
    }
}

impl Device {
    /// No Metal counterpart: `vkGetDeviceProcAddr` of the device command `name`, with the
    /// device's loader word put back first ([`Device::with_loader_word`]). The caller's own
    /// command tables (`VK_KHR_swapchain`'s, `VK_EXT_metal_objects`', or any other extension the
    /// options enabled) load here at any time, one name at a time, as ash's table loaders ask:
    /// `ash::khr::swapchain::DeviceFn::load(|name| device.device_proc_addr(name)...)`. Through
    /// [`DeviceHandles::instance`] the same load faults on MoltenVK once a message named the
    /// device. `None` is Vulkan's null: a command of an extension the device did not enable, or
    /// no device command of that name.
    ///
    /// Which commands a loaded pointer reaches is the loader's: most are the driver's own (on
    /// MoltenVK `vkAcquireNextImageKHR`, `vkQueuePresentKHR`, `vkGetSwapchainImagesKHR`,
    /// `vkDestroySwapchainKHR`, `vkExportMetalObjectsEXT`), which read no word, but some are the
    /// loader's own and read it at every call; the caller makes those inside
    /// [`Device::with_loader_word`], which lists them.
    pub fn device_proc_addr(&self, name: &CStr) -> vk::PFN_vkVoidFunction {
        let shared = self.shared();
        let handle = shared.vk().handle();
        self.with_loader_word(|| {
            // SAFETY: the device is live while `self` is, made from this instance, and its first
            // word is the loader's again; `name` is NUL-terminated.
            unsafe {
                shared
                    .instance
                    .raw
                    .get_device_proc_addr(handle, name.as_ptr())
            }
        })
    }

    /// No Metal counterpart: `call`'s answer, with the device's first word put back first, as the
    /// Vulkan loader wrote it. MoltenVK 1.4.1 writes `ICD_LOADER_MAGIC` over it whenever it names
    /// the device in a message to a debug-utils messenger. A device the runtime creates has one
    /// wherever the loader lists `VK_EXT_debug_utils` (the relay's, `device/relay.rs`), an adopted
    /// one where its consumer registered one. The loader's own entry points for the device read the
    /// word as their dispatch table:
    ///
    /// - those `vkGetDeviceProcAddr` hands out for the device itself, which fault on the magic:
    ///   `vkGetDeviceProcAddr`, `vkGetDeviceQueue`, `vkGetDeviceQueue2`,
    ///   `vkAllocateCommandBuffers`, `vkDestroyDevice`;
    /// - its terminators, which find the device by comparing first words, not handles:
    ///   `vkCreateSwapchainKHR`, `vkCreateSharedSwapchainsKHR`,
    ///   `vkGetDeviceGroupSurfacePresentModesKHR`, `vkGetDeviceGroupSurfacePresentModes2EXT`, and
    ///   `VK_EXT_debug_utils`' and `VK_EXT_debug_marker`'s object naming and tagging. A device
    ///   whose word holds the magic matches the first device in the loader's lists whose word
    ///   holds it too, and the call takes that device's driver entry and unwraps the surface for
    ///   that device's instance (loader 1.4.357: `loader_get_icd_and_device`,
    ///   `wsi_unwrap_icd_surface`): right with one such device in the process, wrong with two;
    /// - with a layer between the loader and the driver (the validation layer), every command:
    ///   the layer finds its state by the word.
    ///
    /// The caller makes such calls inside `call`. The loader's entry points for a queue or a
    /// command buffer read that object's own word: a queue's is put back by
    /// [`CommandQueue::with_vk_queue`](super::super::CommandQueue::with_vk_queue). A driver
    /// thread that writes the word again between the restore and the loader's read (a message
    /// about the device from MoltenVK's completion handlers) is not guarded against. On every
    /// other driver the word never changes, and the restore writes nothing.
    pub fn with_loader_word<R>(&self, call: impl FnOnce() -> R) -> R {
        // SAFETY: the device is live while `self` is.
        unsafe { self.shared().loader_word.restore() };
        call()
    }
}
