//! Exports: a block of device memory the consumer shares with another API through a handle
//! ([`Device::new_exported_block`]), such as a video decoder that writes decoded frames into it.
//! Metal has no counterpart, as it has none of presenting: the consumer's own path.
//!
//! The form is one: an allocation of its own, dedicated to its one buffer
//! (`VkMemoryDedicatedAllocateInfo`), never a place in a pooled block. Vulkan lets a driver
//! require a dedicated allocation for an export (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`),
//! so this is the form every driver takes, and what exportable memory costs a submission falls on
//! these allocations alone. A handle type is offered where the device lists its extensions and its
//! buffers export it (`vkGetPhysicalDeviceExternalBufferProperties`), and refused elsewhere.

use crate::block::Block;
use crate::counters::{Metric, Part};
use crate::device::Device;
use crate::error::{Error, Limit, Refusal};
use crate::floor;
use crate::memory::{self, Own};
use crate::submit::Held;
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

/// No Metal counterpart: the handle an export gives ([`Device::new_exported_block`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExportHandle {
    /// An opaque fd (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`), on a Unix host: another
    /// API of the same device and driver imports it, the NVIDIA decoder's route among them.
    OpaqueFd,
    /// A Linux dma-buf fd (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`), on a Unix host:
    /// another driver imports it, VA-API's decoders among them.
    DmaBuf,
    /// An NT handle (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`), on a Windows host.
    OpaqueWin32,
}

impl ExportHandle {
    pub(crate) const ALL: [ExportHandle; 3] = [
        ExportHandle::OpaqueFd,
        ExportHandle::DmaBuf,
        ExportHandle::OpaqueWin32,
    ];

    pub(crate) fn vk(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            ExportHandle::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            ExportHandle::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            ExportHandle::OpaqueWin32 => vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32,
        }
    }

    /// The device extensions that give it.
    fn extensions(self) -> &'static [&'static CStr] {
        match self {
            ExportHandle::OpaqueFd => &[ash::khr::external_memory_fd::NAME],
            ExportHandle::DmaBuf => &[
                ash::khr::external_memory_fd::NAME,
                ash::ext::external_memory_dma_buf::NAME,
            ],
            ExportHandle::OpaqueWin32 => &[ash::khr::external_memory_win32::NAME],
        }
    }

    /// Whether the other side of its handle is another driver, whose ownership transfers name
    /// the foreign queue family (`VK_QUEUE_FAMILY_FOREIGN_EXT`): a dma-buf's. An opaque handle is
    /// imported by the same device and driver, which `VK_QUEUE_FAMILY_EXTERNAL` names.
    pub(crate) fn foreign(self) -> bool {
        self == ExportHandle::DmaBuf
    }

    /// The queue family an ownership transfer names as the other side of its handle.
    pub(crate) fn family(self) -> u32 {
        match self.foreign() {
            true => vk::QUEUE_FAMILY_FOREIGN_EXT,
            false => vk::QUEUE_FAMILY_EXTERNAL,
        }
    }
}

/// No Metal counterpart: an exported block's memory as another API imports it
/// ([`Device::new_exported_block`]).
#[non_exhaustive]
#[derive(Debug)]
pub struct Export {
    /// A new handle of the memory, the consumer's own to close: an fd on a Unix host.
    #[cfg(unix)]
    pub handle: std::os::fd::OwnedFd,
    /// A new handle of the memory, the consumer's own to close: an NT handle on a Windows host.
    #[cfg(windows)]
    pub handle: std::os::windows::io::OwnedHandle,
    /// The allocation's size, at least the block's length, which an import of the handle names
    /// (`VkMemoryAllocateInfo::allocationSize`).
    pub size: u64,
    /// The index of the memory type it is in, the device's Private type, which a Vulkan import of
    /// an opaque handle names again.
    pub memory_type: u32,
}

/// What the device exports, read once as it is made.
pub(crate) struct Exports {
    /// Whether the device exports each handle, by its place in [`ExportHandle::ALL`].
    offered: [bool; 3],
    /// Whether `VK_EXT_queue_family_foreign` is enabled, which a dma-buf's export needs.
    pub(crate) foreign: bool,
    /// Whether `VK_EXT_image_drm_format_modifier` is enabled, which a dma-buf's surface in place
    /// needs ([`Device::new_exported_surface`]).
    pub(crate) modifiers: bool,
    /// `VK_KHR_external_memory_fd`, where it is enabled.
    #[cfg(unix)]
    fd: Option<ash::khr::external_memory_fd::Device>,
    /// `VK_KHR_external_memory_win32`, where it is enabled.
    #[cfg(windows)]
    win32: Option<ash::khr::external_memory_win32::Device>,
}

impl Exports {
    /// What `device`, made from `physical` with `extensions` enabled, exports: each handle this
    /// host has whose extensions are enabled and that the device's buffers of the floor's usage
    /// export.
    ///
    /// # Safety
    ///
    /// `physical` is `instance`'s, of Vulkan 1.3, and `device` was made from it.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        extensions: &[&CStr],
    ) -> Exports {
        let enabled = |name: &CStr| extensions.contains(&name);
        let exportable = |handle: ExportHandle| {
            let info = vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(floor::BUFFER_USAGE)
                .handle_type(handle.vk());
            let mut properties = vk::ExternalBufferProperties::default();
            // SAFETY: the caller's.
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    physical,
                    &info,
                    &mut properties,
                )
            };
            (properties.external_memory_properties)
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        };
        let host = |handle: ExportHandle| match handle {
            ExportHandle::OpaqueFd | ExportHandle::DmaBuf => cfg!(unix),
            ExportHandle::OpaqueWin32 => cfg!(windows),
        };
        let offered = ExportHandle::ALL.map(|handle| {
            host(handle)
                && handle.extensions().iter().all(|&name| enabled(name))
                && exportable(handle)
        });
        Exports {
            offered,
            foreign: enabled(ash::ext::queue_family_foreign::NAME),
            modifiers: enabled(ash::ext::image_drm_format_modifier::NAME),
            #[cfg(unix)]
            fd: enabled(ash::khr::external_memory_fd::NAME)
                .then(|| ash::khr::external_memory_fd::Device::new(instance, device)),
            #[cfg(windows)]
            win32: enabled(ash::khr::external_memory_win32::NAME)
                .then(|| ash::khr::external_memory_win32::Device::new(instance, device)),
        }
    }

    /// Whether the device exports `handle`.
    pub(crate) fn offers(&self, handle: ExportHandle) -> bool {
        let at = ExportHandle::ALL.iter().position(|&h| h == handle);
        at.is_some_and(|at| self.offered[at])
    }
}

impl Device {
    /// No Metal counterpart: a block of Private memory of `len` bytes rounded up to 64 KiB, as a
    /// take's are ([`Device::take_block`]), for the consumer to place buffers in, whose memory it
    /// shares with another API through `handle`'s kind of handle: a video decoder that writes
    /// frames into it, for one. The block is one allocation of its own, dedicated to the one
    /// buffer that spans it, never a place in a pooled block, and exportable as `handle` alone.
    /// Its bytes are undefined, as a taken block's are.
    ///
    /// It is offered where the device lists the handle's extensions (`VK_KHR_external_memory_fd`
    /// for an opaque fd; with it `VK_EXT_external_memory_dma_buf` for a dma-buf;
    /// `VK_KHR_external_memory_win32` for an NT handle), which device creation enables where
    /// listed, and its buffers export the handle (`vkGetPhysicalDeviceExternalBufferProperties`).
    /// A dma-buf also needs `VK_EXT_queue_family_foreign`: the other side of a dma-buf is another
    /// driver, which an ownership transfer names as the foreign queue family
    /// (`VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-09099`). The memory is in the device's
    /// Private type.
    ///
    /// With the block comes the [`Export`]: a new handle of the memory (`vkGetMemoryFdKHR`,
    /// `vkGetMemoryWin32HandleKHR`), the consumer's to close, and what an import of it names: the
    /// allocation's size and its memory type. The block lives until its last handle and its last
    /// buffer, command buffer or upload are gone; the block-maker thread then frees its memory.
    /// What another API reads or writes through an import of it after that is undefined, as
    /// Vulkan leaves it; an import may outlive the block unused. Each export is one
    /// `VkDeviceMemory` allocation and counts against the device's `maxMemoryAllocationCount`;
    /// [`Gauges::exports`](crate::Gauges::exports) counts those live.
    ///
    /// No image is made in its memory, which is dedicated to its buffer
    /// (`VUID-vkBindImageMemory-memory-10926`): a texture over a buffer in it is copied, as over
    /// host memory's import ([`Buffer::new_texture`](crate::Buffer::new_texture)), and it is no
    /// heap's memory ([`Device::new_heap_texture`]). The block's handle states its device address
    /// ([`BlockInfo::gpu_address`](crate::BlockInfo::gpu_address)), so the make counts in
    /// [`Metrics::exported_blocks_addressed`](crate::Metrics::exported_blocks_addressed), as
    /// [`Buffer::gpu_address`](crate::Buffer::gpu_address) does.
    ///
    /// Err: [`Limit::NoExport`] where the device does not export the handle;
    /// [`Limit::ExportType`] for a dma-buf on a device without the foreign queue family, or where
    /// the buffer's memory requirements do not list the Private type;
    /// [`Limit::BlockLen`] past the device's longest allocation or
    /// buffer; [`Limit::MemoryAllocations`];
    /// [`Limit::BlockIds`]; [`Error::Vulkan`] where the buffer, the
    /// allocation, its binding or the handle fails; [`Error::Quiesced`].
    pub fn new_exported_block(
        &self,
        handle: ExportHandle,
        len: u64,
    ) -> Result<(Block, Export), Error> {
        let (block, export) = self.exported_block(handle, len)?;
        block.held()?.1.address_out();
        Ok((block, export))
    }

    /// An exported block, as [`Device::new_exported_block`] makes it, whose address no call has
    /// handed out yet: an exported surface's in the copy form, whose handle the surface keeps.
    pub(crate) fn exported_block(
        &self,
        handle: ExportHandle,
        len: u64,
    ) -> Result<(Block, Export), Error> {
        let started = crate::host_time();
        let core = self.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        if !core.exports.offers(handle) {
            return Err(Error::Limit(Limit::NoExport { handle }));
        }
        if handle.foreign() && !core.exports.foreign {
            return Err(Error::Limit(Limit::ExportType {
                handle,
                listed: None,
            }));
        }
        // SAFETY: the device's buffers export `handle`, its extensions enabled.
        let (memory, size) = unsafe { memory::export(core, len, handle) }?;
        let info = memory.info();
        let raw = memory.device_memory();
        let claiming = crate::host_time();
        let block = Block::claim(core, info, memory::Block::new(memory, Arc::clone(core)));
        let exporting = crate::host_time();
        (core.counters).timed(Part::ExportClaim, size, exporting - claiming, false);
        // A failure here gives the block back with its memory as `block` drops.
        let handle = core.exports.handle(raw, handle)?;
        let exported = crate::host_time();
        (core.counters).timed(Part::ExportHandle, size, exported - exporting, false);
        let export = Export {
            handle,
            size,
            memory_type: core.memory_type(crate::MemoryClass::Private),
        };
        (core.counters).timed(Part::ExportMake, size, exported - started, true);
        Ok((block, export))
    }
}

impl Block {
    /// No Metal counterpart: the other API's accesses of this exported block
    /// ([`Device::new_exported_block`]) are done, and the runtime's commands may now read and
    /// write it. The consumer calls it once the other API's work that accessed the block (work
    /// that wrote it, or that read it after a release) has completed (its own wait on the
    /// decoder, for one), and before it commits the command buffers that use the block.
    ///
    /// The call records the block's acquire from the other API's queue family: the whole
    /// block, from `VK_QUEUE_FAMILY_EXTERNAL` for an opaque handle, or from
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf, to the runtime's, made visible to every
    /// later command. It goes in the upload, as [`Device::zero_block_range`]'s fill does:
    /// submitted ahead of every command buffer committed after this call, holding the block until
    /// it completes. The call never waits for the GPU. One call is one buffer barrier,
    /// counted in [`Metrics::external_acquires`](crate::Metrics::external_acquires).
    ///
    /// The contract has three parts: between this call and the other API's next write, the other
    /// API does not access the block, unless a release
    /// ([`release_to_external`](crate::BlitCommandEncoder::release_to_external)) comes between,
    /// from whose completion it may read the whole block, the runtime's writes and its own
    /// earlier ones; when it writes again, it reads nothing of the block but what it has itself
    /// written since and what a release since handed it; and after each acquire, a byte the other
    /// API has not written since the previous acquire is undefined to the runtime, unless a
    /// release came between the two: a release and the acquire after it transfer ownership, and
    /// Vulkan leaves contents undefined only where "no ownership transfer is defined", so the
    /// bytes the other API did not write keep what the release handed it. The acquire is not
    /// followed by a release of its own: without one, Vulkan leaves the block's contents undefined
    /// to the other API's reads ("the contents of that resource are undefined for any read
    /// accesses performed by the second queue family").
    ///
    /// Err: [`Refusal::NotExported`] for a block that is not exported;
    /// [`Refusal::InScheduledHandler`] in a scheduled handler; [`Error::Quiesced`]; and
    /// [`Error::Vulkan`] where the upload's command buffer cannot be made.
    pub fn external_writes_done(&self) -> Result<(), Error> {
        crate::command::not_in_scheduled_handler()?;
        let (core, block) = self.held()?;
        let Some(Own::Export { handle, .. }) = block.memory.own else {
            return Err(Error::Refused(Refusal::NotExported { block: self.id }));
        };
        let (buffer, _) = block.memory.range(0);
        let acquire = [vk::BufferMemoryBarrier2::default()
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_queue_family_index(handle.family())
            .dst_queue_family_index(core.family)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        let held = [Held::Block(Arc::clone(&block))];
        core.work.upload(&core, None, held, |device, command| {
            let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&acquire);
            // SAFETY: the upload's command buffer is recording, under the work lock; the buffer
            // spans the block's allocation, which the upload holds, and a dma-buf's foreign
            // family is enabled (`Device::new_exported_block`).
            unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
        })?;
        // The other API's writes change the bytes of the textures over the block
        // (`crate::over::ExportedOvers`).
        core.work
            .exported_changed(block.memory.id, (0, block.memory.len));
        core.counters.count(Metric::Barriers);
        core.counters.count(Metric::ExternalAcquires);
        Ok(())
    }
}

impl Exports {
    /// `VK_KHR_external_memory_fd`, where it is enabled.
    #[cfg(unix)]
    pub(crate) fn fds(&self) -> Option<&ash::khr::external_memory_fd::Device> {
        self.fd.as_ref()
    }

    /// A new fd of `memory`, exported as `handle`.
    #[cfg(unix)]
    pub(crate) fn handle(
        &self,
        memory: vk::DeviceMemory,
        handle: ExportHandle,
    ) -> Result<std::os::fd::OwnedFd, Error> {
        use std::os::fd::FromRawFd;
        let fds = (self.fd.as_ref()).ok_or(Error::Limit(Limit::NoExport { handle }))?;
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(handle.vk());
        // SAFETY: the extension is enabled; the memory is live and exportable as `handle`.
        let fd = unsafe { fds.get_memory_fd(&info) }.map_err(Error::vulkan("vkGetMemoryFdKHR"))?;
        // SAFETY: the call made the fd, which no one else owns.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
    }

    /// `VK_KHR_external_memory_win32`, where it is enabled.
    #[cfg(windows)]
    pub(crate) fn win32(&self) -> Option<&ash::khr::external_memory_win32::Device> {
        self.win32.as_ref()
    }

    /// A new NT handle of `memory`, exported as `handle`.
    #[cfg(windows)]
    pub(crate) fn handle(
        &self,
        memory: vk::DeviceMemory,
        handle: ExportHandle,
    ) -> Result<std::os::windows::io::OwnedHandle, Error> {
        use std::os::windows::io::{FromRawHandle, OwnedHandle, RawHandle};
        let handles = (self.win32.as_ref()).ok_or(Error::Limit(Limit::NoExport { handle }))?;
        let info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(handle.vk());
        // SAFETY: the extension is enabled; the memory is live and exportable as `handle`.
        let raw = unsafe { handles.get_memory_win32_handle(&info) }
            .map_err(Error::vulkan("vkGetMemoryWin32HandleKHR"))?;
        // SAFETY: the call made the handle, which no one else owns.
        Ok(unsafe { OwnedHandle::from_raw_handle(raw as RawHandle) })
    }
}
