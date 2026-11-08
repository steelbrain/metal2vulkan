//! Buffers whose memory another API imports (a runtime extension: Metal has no counterpart).
//!
//! A consumer whose other GPU work writes into a runtime buffer directly (a video decoder's
//! pictures through CUDA's `cuImportExternalMemory`, a second `VkDevice`) asks for the buffer at
//! its making ([`Device::new_exportable_buffer`]) and takes its memory as an opaque file
//! descriptor ([`Buffer::export_memory_fd`], `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`). No
//! other buffer pays for it: only such a buffer lies in an allocation of its own, made
//! exportable; every other keeps the device's pooled blocks (`memory.rs`).
//!
//! - **What the device must have.** `VK_KHR_external_memory_fd`, which the runtime enables
//!   wherever the device lists it (an adopted device: wherever its consumer enabled it), and an
//!   answer of `vkGetPhysicalDeviceExternalBufferProperties` for a buffer of the runtime's usage
//!   that the handle type is exportable. Both are asked once, as the device is made; a device
//!   without them refuses the making (`NotImplemented(RuntimeGap::ExportableMemory)`).
//! - **The allocation.** The `VkBuffer` names the handle type
//!   (`VkExternalMemoryBufferCreateInfo`) and has the device-address usage of its own; its memory
//!   (`VkExportMemoryAllocateInfo`) is a block of its own, of the memory type its storage mode
//!   takes as any buffer's does, and its dedicated allocation
//!   (`VkMemoryDedicatedAllocateInfo`) where the device exports such memory only so
//!   (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`) or the driver prefers it
//!   (`VkMemoryDedicatedRequirements`). It is never parked for reuse by another buffer.
//! - **The descriptor.** Each call of [`Buffer::export_memory_fd`] is a new one
//!   (`vkGetMemoryFdKHR`), the caller's to close; with it the allocation's size, the buffer's
//!   offset and length in it, whether it is a dedicated allocation (CUDA's
//!   `cudaExternalMemoryDedicated`), and its memory type, which an import into another
//!   `VkDevice` of the same physical device names again.
//! - **Lifetime.** The buffer must outlive every import of its memory: its release frees the
//!   allocation, and what another API reads or writes through its import after that is undefined.
//! - **Device loss.** On a device loss the memory's contents are lost for every import of it too.
//!   The device recovers from the loss with a new `VkDevice` (`device/line.rs`): the buffer, made
//!   before, is refused (`NotImplemented(MadeBeforeRecovery)`), and the device makes new buffers
//!   in new allocations, each exported anew, with a new descriptor; the old imports are to be let
//!   go of. A device the consumer made answers `DeviceLost`.

use super::buffer::{buffer_class, usage, BufferParts, Modes, Owner};
use super::device::lock;
use super::error::{nil, vulkan, Error, Problem, RuntimeGap};
use super::memory::{allocate_exported, Exported};
use super::values::ResourceOptions;
use super::{Buffer, Device};
use crate::address::AddressPlacement;
use crate::release;
use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};

/// The handle type the buffers export their memory as.
const HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

/// How the device exports a buffer's memory as an opaque file descriptor, asked once as the
/// device is made.
pub(super) struct OpaqueFdExport {
    functions: ash::khr::external_memory_fd::Device,
    /// The device exports a buffer's memory as one only from the buffer's dedicated allocation.
    dedicated_only: bool,
}

impl OpaqueFdExport {
    /// The device's export of a buffer's memory, of a buffer of `usage`, as an opaque file
    /// descriptor: where it enables `VK_KHR_external_memory_fd` (`functions`, loaded then) and
    /// `vkGetPhysicalDeviceExternalBufferProperties` answers the handle type exportable from a
    /// buffer's memory. `None` where it lacks either.
    ///
    /// # Safety
    ///
    /// `physical` is a physical device of `instance`, made at Vulkan 1.1 or later.
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        functions: Option<ash::khr::external_memory_fd::Device>,
        usage: vk::BufferUsageFlags,
    ) -> Option<OpaqueFdExport> {
        let functions = functions?;
        let query = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(HANDLE);
        let mut properties = vk::ExternalBufferProperties::default();
        // SAFETY: the caller's contract.
        unsafe {
            instance.get_physical_device_external_buffer_properties(
                physical,
                &query,
                &mut properties,
            )
        };
        let memory = properties.external_memory_properties;
        let features = memory.external_memory_features;
        let exports = features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
            && memory.compatible_handle_types.contains(HANDLE);
        exports.then_some(OpaqueFdExport {
            functions,
            dedicated_only: features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
        })
    }
}

/// No Metal counterpart: a buffer's memory, exported as an opaque file descriptor
/// (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`, [`Buffer::export_memory_fd`]), and what an
/// import of it names: CUDA's `cuImportExternalMemory` takes the descriptor, the allocation's
/// size and whether it is dedicated, and `cuExternalMemoryGetMappedBuffer` the buffer's offset
/// and length in it; a `VkDevice`'s `VkImportMemoryFdInfoKHR` takes the descriptor with the
/// allocation's size and memory type.
#[non_exhaustive]
#[derive(Debug)]
pub struct ExportedMemory {
    /// A new descriptor of the memory, the caller's: an import that succeeds takes it (Vulkan's
    /// and CUDA's), else the caller closes it, by dropping it.
    pub fd: OwnedFd,
    /// The bytes of the allocation, as `vkAllocateMemory` made it.
    pub allocation_size: u64,
    /// The buffer's first byte in the allocation.
    pub offset: u64,
    /// The buffer's length, as made.
    pub length: u64,
    /// The allocation is the buffer's dedicated allocation (`VkMemoryDedicatedAllocateInfo`): an
    /// import names it so (CUDA's `cudaExternalMemoryDedicated`, Vulkan's dedicated import).
    pub dedicated: bool,
    /// The allocation's memory type, as the device that made it numbers its types.
    pub memory_type_index: u32,
}

impl Device {
    /// No Metal counterpart (a runtime extension): [`Device::new_buffer`] of `length` bytes with
    /// `options`, all zero, whose memory another API imports ([`Buffer::export_memory_fd`]): a
    /// buffer of its own allocation, made exportable as an opaque file descriptor, where any other
    /// buffer lies in the device's pooled blocks (`exported_memory.rs`). It is a buffer as any
    /// other in every call; a `Managed` one on a device without unified memory exports the GPU's
    /// copy, which the host's reaches as Metal's managed copies do (`didModifyRange:`,
    /// `synchronizeResource:`).
    ///
    /// The buffer must outlive every import of its memory. On a device loss the memory's contents
    /// are lost, for every import too, and a device made again makes new allocations, each
    /// exported anew.
    ///
    /// Refused as [`Device::new_buffer`] refuses it, and `NotImplemented`
    /// ([`RuntimeGap::ExportableMemory`]) on a device without `VK_KHR_external_memory_fd` or
    /// that exports no buffer's memory as an opaque file descriptor.
    pub fn new_exportable_buffer(
        &self,
        length: u64,
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        self.current().new_exportable_buffer_now(length, options)
    }

    /// [`Device::new_exportable_buffer`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_exportable_buffer_now(
        &self,
        length: u64,
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_exportable_buffer";
        self.keep_room();
        let options = self.own_memory_options(CALL, options)?;
        let modes = Modes::of(options)?;
        let class = buffer_class(CALL, modes)?;
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(CALL, Problem::BufferLength { length }));
        }
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let shared = self.shared();
        let Some(export) = shared.opaque_fd.as_ref() else {
            return Err(Error::NotImplemented(RuntimeGap::ExportableMemory));
        };
        let raw_device = shared.vk();
        // Padded to 16 bytes, as any buffer's, so a device fill clears every byte too.
        let size = length.next_multiple_of(16);
        let usage = usage(shared.device_address);
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .push_next(&mut external),
        );
        // SAFETY: the device is live and `info` and its chain valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        shared.counts.buffer_made(usage);
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::BufferMemoryRequirementsInfo2::default().buffer(raw);
        // SAFETY: `raw` is a live buffer of this device, at Vulkan 1.1 or later.
        unsafe { raw_device.get_buffer_memory_requirements2(&query, &mut requirements) };
        let requirements = requirements.memory_requirements;
        let exported = Exported {
            handle: HANDLE,
            dedicated: export.dedicated_only
                || dedicated.prefers_dedicated_allocation == vk::TRUE
                || dedicated.requires_dedicated_allocation == vk::TRUE,
        };
        let placed = allocate_exported(
            &shared.memory,
            raw_device,
            &requirements,
            class,
            (raw, exported),
        );
        let placement = match placed {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        // SAFETY: the placement meets the buffer's requirements, is its own allocation (its
        // dedicated one where `exported.dedicated`), and is bound nowhere else.
        let bound = unsafe {
            crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
        };
        let address = (bound.map_err(vulkan))
            .and_then(|()| lock(&shared.addresses).place(AddressPlacement::Any, length));
        let (address, reserved) = match address {
            Ok(address) => address,
            Err(error) => {
                lock(&shared.memory).free(raw_device, &placement);
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let buffer = self.buffer_of(BufferParts {
            raw,
            memory: placement,
            offset: 0,
            base: 0,
            length,
            reach: length,
            modes,
            address,
            reserved,
            owner: Owner::Runtime,
            release: None,
            of_heap: false,
            drops_writes: false,
            mirror: None,
            again: None,
        });
        // New memory holds no value until written: zero, as Metal's new buffer reads. Dropping
        // `buffer` on failure releases it.
        buffer.clear_first()?;
        self.degrade_system_memory(&buffer, class, || {
            format!("exportable buffer of {length} bytes")
        });
        Ok(self.visible(buffer))
    }
}

impl Buffer {
    /// No Metal counterpart (a runtime extension): the buffer's memory as a new opaque file
    /// descriptor (`vkGetMemoryFdKHR`, `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`), with what an
    /// import of it names ([`ExportedMemory`]). Each call answers a new descriptor, the caller's
    /// to close; the allocation, its size and the buffer's offset in it stay the buffer's for
    /// its life.
    ///
    /// The buffer must outlive every import of its memory: its release frees the allocation. On a
    /// device loss the memory's contents are lost for every import too, and the call answers as
    /// the module docs say.
    ///
    /// Refused: a buffer not made by [`Device::new_exportable_buffer`] (`Invalid`,
    /// `Problem::NotExportable`), one made before its device's recovery
    /// (`NotImplemented(MadeBeforeRecovery)`) or on a lost device (`DeviceLost`), and the driver's
    /// failure.
    pub fn export_memory_fd(&self) -> Result<ExportedMemory, Error> {
        const CALL: &str = "export_memory_fd";
        let device = self.device();
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let shared = device.shared();
        let found = self.own_placement(|placement, offset| {
            let exported = lock(&shared.memory).exported(placement)?;
            let at = placement.offset + offset;
            Some((exported, placement.memory, at, placement.type_index()))
        });
        let Some(((exported, allocation_size), memory, offset, memory_type_index)) =
            found.flatten()
        else {
            return Err(super::error::invalid(CALL, Problem::NotExportable));
        };
        let Some(export) = shared.opaque_fd.as_ref() else {
            return Err(super::error::invalid(CALL, Problem::NotExportable));
        };
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(exported.handle);
        // SAFETY: the memory is the buffer's, live while the buffer is, allocated exportable as
        // the handle type (`allocate_exported`); the functions are the device's.
        let fd = unsafe { export.functions.get_memory_fd(&info) }.map_err(vulkan)?;
        Ok(ExportedMemory {
            // SAFETY: a new descriptor Vulkan handed over, which the caller owns from here on.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            allocation_size,
            offset,
            length: self.length(),
            dedicated: exported.dedicated,
            memory_type_index,
        })
    }
}
