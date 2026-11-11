//! Buffers whose memory another API imports (a runtime extension: Metal has no counterpart).
//!
//! A consumer whose other GPU work writes into a runtime buffer directly (a video decoder's
//! pictures through CUDA's `cuImportExternalMemory` or VA-API's `DRM_PRIME_2` surfaces, a second
//! `VkDevice`) asks for the buffer at its making, of a handle type ([`MemoryExport`]):
//! [`Device::new_exportable_buffer`] for an opaque file descriptor, or a descriptor's
//! `exported_as` ([`BufferDescriptor::exported_as`](crate::BufferDescriptor::exported_as)) for
//! either, with the descriptor's `aligned_byte` and `padding_unobserved`. It takes the memory as
//! that handle: [`Buffer::export_memory_fd`] (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`) or
//! [`Buffer::export_memory_dma_buf`] (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`). No other
//! buffer pays for it: such buffers lie in blocks of their own kind, one per handle type,
//! allocated exportable, and every other keeps the device's other pooled blocks (`memory.rs`).
//!
//! - **What the device must have.** For an opaque file descriptor, `VK_KHR_external_memory_fd`,
//!   which the runtime enables wherever the device lists it (an adopted device: wherever its
//!   consumer enabled it). For a dma-buf, `VK_EXT_external_memory_dma_buf` with it and
//!   `VK_EXT_queue_family_foreign`, which the external write's acquire names, enabled wherever
//!   the device lists them (`create.rs`'s `dma_buf_names`, `foreign_family_name`). For either, an
//!   answer of `vkGetPhysicalDeviceExternalBufferProperties` for a buffer of the runtime's usage
//!   that the handle type is exportable. All are asked once, as the device is made; a device
//!   without them refuses the making (`NotImplemented(RuntimeGap::ExportableMemory)`, or
//!   `NotImplemented(RuntimeGap::DmaBufExport)`).
//! - **The allocation.** The `VkBuffer` names the handle type
//!   (`VkExternalMemoryBufferCreateInfo`) and has the device-address usage of its own; its memory
//!   is of the memory type its storage mode takes as any buffer's does. Where the device exports
//!   such memory only from a dedicated allocation (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`)
//!   or the driver requires one (`VkMemoryDedicatedRequirements::requiresDedicatedAllocation`), it
//!   is the buffer's dedicated allocation (`VkMemoryDedicatedAllocateInfo`,
//!   `VkExportMemoryAllocateInfo`), freed with it. A driver's preference for one
//!   (`prefersDedicatedAllocation`) alone does not make one: no image may be bound to a dedicated
//!   allocation (`VUID-vkBindImageMemory-memory-10926`), and a texture over the buffer is an image
//!   in its memory (below), so a dedicated allocation would leave every such texture a shadow.
//!   Else it lies in a block that exportable buffers of its handle type share, allocated
//!   exportable (`memory.rs`'s `Holds::Exports`, `Holds::DmaBufs`), at a multiple of 256 bytes as
//!   any buffer of the runtime's memory (`buffer.rs`'s `placed`), as other buffers lie in pooled
//!   blocks: a block holds a 4K NV12 picture's 12.4 MB five times, and one that empties is kept
//!   for the next such buffers (`memory.rs`). On an RTX 5080 Laptop GPU a `Private` exportable
//!   buffer of such a picture, an allocation of its own, took 0.25 ms to make in one run and 39.9
//!   ms in another, where a plain one took 2.5 us and 2.9 us (medians of 8); NVIDIA's
//!   `vkAllocateMemory` holds every submission until it returns. The buffer is never parked for
//!   reuse by another buffer. With `aligned_byte` its bytes start that byte's shortfall from the
//!   next multiple of 256 into its `VkBuffer`, as a plain buffer's do (`buffer.rs`'s
//!   `placed_new_buffer`).
//! - **The descriptor.** Each export call answers a new one (`vkGetMemoryFdKHR`), the caller's to
//!   close; with it the allocation's size, the buffer's offset and length in it, whether it is a
//!   dedicated allocation (CUDA's `cudaExternalMemoryDedicated`), and its memory type, which an
//!   import into another `VkDevice` of the same physical device names again. The export settles
//!   the zeros (or bytes) the buffer was made with first, alone (`Buffer::settle_first_work`): the
//!   other API may write the memory at once.
//! - **The external writes.** Vulkan leaves undefined what another API wrote to the memory until
//!   the device acquires it from the queue family reserved for that API: for a dma-buf,
//!   `VK_QUEUE_FAMILY_FOREIGN_EXT` (a media codec or another driver's queue); for an opaque file
//!   descriptor, `VK_QUEUE_FAMILY_EXTERNAL` (another instance of the same driver and device, and
//!   CUDA). Once the caller waited for the other API's writes (`vaSyncSurface`, a CUDA stream's
//!   synchronize), [`Buffer::external_write_done`] notes that acquire, of the whole `VkBuffer`,
//!   for the device's next submission to record before any of its commands
//!   (`texture/first_move.rs`'s `FirstFill::acquired`), so it waits for nothing; a host-visible
//!   buffer, whose host may read its contents at once, has it made then and waits for it alone. A
//!   texture over the buffer is a linear image of its memory in `GENERAL` (or a shadow of its
//!   bytes, which copies them), a host-accessible alias that reads what the buffer holds, made
//!   for the memory's handle type, which an image bound in exported memory names
//!   (`VUID-vkBindImageMemory-memory-02728`), so made `UNDEFINED`, and taken where the device's
//!   check of that exported memory passed (`in_place.rs`, "The runtime's exported memory"). Over
//!   a dma-buf where no image binds in the memory itself (RADV's linear images for dma-bufs, its
//!   `DRM_FORMAT_MOD_LINEAR` ones that require a dedicated allocation, or the buffer's dedicated
//!   allocation, to which no image may be bound, `VUID-vkBindImageMemory-memory-10926`), it is a
//!   `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated import of the memory's own dma-buf
//!   ([`Device::dma_buf_of`], `dma_buf.rs`'s `bind_exported_dma_buf_image`), whose first move
//!   acquires it from `VK_QUEUE_FAMILY_FOREIGN_EXT`; else a shadow. The call lets go of every shadow's standing over the
//!   bytes (`shadow_standing.rs`), so a shadow's next
//!   copy in, in the next command buffer that uses it, moves what the other API wrote. A write the
//!   caller does not announce so is not the device's to read: a shadow whose image a command
//!   buffer's end left standing keeps reading what the buffer held then. Nothing
//!   is released to the other API: it writes the memory as its own, and what the runtime wrote
//!   there before is not its to read (Vulkan's implicit acquire makes it undefined).
//! - **Lifetime.** The buffer must outlive every import of its memory: its release frees its
//!   dedicated allocation, or gives its bytes of a shared block to the next exportable buffer,
//!   and what another API reads or writes through its import after that is undefined. An import
//!   of a shared block reaches the whole block: the other API is to read and write the buffer's
//!   `offset` and `length` alone. The other API writes the memory only while no command of the
//!   runtime's that uses the buffer is in flight.
//! - **Device loss.** On a device loss the memory's contents are lost for every import of it too.
//!   The device recovers from the loss with a new `VkDevice` (`device/line.rs`): the buffer, made
//!   before, is refused (`NotImplemented(MadeBeforeRecovery)`), and the device makes new buffers
//!   in new allocations, each exported anew, with a new descriptor; the old imports are to be let
//!   go of. A device the consumer made answers `DeviceLost`.

use super::address::Place;
use super::buffer::LINEAR_IMAGE_ALIGNMENT;
use super::buffer::{buffer_class, placed, usage, BufferParts, Modes, Naming, Owner};
use super::device::lock;
use super::error::{invalid, nil, vulkan, Error, Problem};
use super::memory::{allocate_exported, Exported, DMA_BUF_HANDLE, EXPORT_HANDLE};
use super::values::ResourceOptions;
use super::{Buffer, Device};
use crate::address::AddressPlacement;
use crate::release;
use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};

use super::buffer_descriptor::MemoryExport;

impl MemoryExport {
    /// The handle type its buffers export their memory as.
    fn handle(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            MemoryExport::OpaqueFd => EXPORT_HANDLE,
            MemoryExport::DmaBuf => DMA_BUF_HANDLE,
        }
    }

    /// The queue family reserved for external memory that the other API's writes are acquired
    /// from ([`Buffer::external_write_done`]).
    fn writer_family(self) -> u32 {
        match self {
            MemoryExport::OpaqueFd => vk::QUEUE_FAMILY_EXTERNAL,
            MemoryExport::DmaBuf => vk::QUEUE_FAMILY_FOREIGN_EXT,
        }
    }

    /// The handle type a block was allocated exportable as, where it is one of these.
    fn of_handle(handle: vk::ExternalMemoryHandleTypeFlags) -> Option<MemoryExport> {
        match handle {
            EXPORT_HANDLE => Some(MemoryExport::OpaqueFd),
            DMA_BUF_HANDLE => Some(MemoryExport::DmaBuf),
            _ => None,
        }
    }
}

/// How the device exports a buffer's memory as a file descriptor of one handle type, asked once
/// as the device is made.
pub(super) struct FdExport {
    functions: ash::khr::external_memory_fd::Device,
    /// The device exports a buffer's memory as one only from the buffer's dedicated allocation.
    dedicated_only: bool,
}

impl FdExport {
    /// The device's export of a buffer's memory, of a buffer of `usage`, as `export`'s handle
    /// type: where it enables what that handle type needs (`functions`, loaded then; `None`
    /// where it lacks an extension) and `vkGetPhysicalDeviceExternalBufferProperties` answers the
    /// handle type exportable from a buffer's memory. `None` where it lacks either.
    ///
    /// # Safety
    ///
    /// `physical` is a physical device of `instance`, made at Vulkan 1.1 or later.
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        functions: Option<ash::khr::external_memory_fd::Device>,
        (usage, export): (vk::BufferUsageFlags, MemoryExport),
    ) -> Option<FdExport> {
        let functions = functions?;
        let handle = export.handle();
        let query = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(handle);
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
            && memory.compatible_handle_types.contains(handle);
        exports.then_some(FdExport {
            functions,
            dedicated_only: features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
        })
    }

    /// Whether the device exports such memory from a block that buffers share, not only from a
    /// buffer's dedicated allocation.
    pub(super) fn pooled(&self) -> bool {
        !self.dedicated_only
    }
}

/// No Metal counterpart: a buffer's memory, exported as a file descriptor of its handle type
/// ([`Buffer::export_memory_fd`], an opaque one; [`Buffer::export_memory_dma_buf`], a dma-buf),
/// and what an import of it names: CUDA's `cuImportExternalMemory` takes the descriptor, the
/// allocation's size and whether it is dedicated, and `cuExternalMemoryGetMappedBuffer` the
/// buffer's offset and length in it; a `VkDevice`'s `VkImportMemoryFdInfoKHR` takes the
/// descriptor with the allocation's size and memory type; VA-API's `DRM_PRIME_2` surface takes a
/// dma-buf with its size and each plane's offset in it, the buffer's `offset` plus the plane's.
#[non_exhaustive]
#[derive(Debug)]
pub struct ExportedMemory {
    /// A new descriptor of the memory, the caller's: an import that succeeds takes it (Vulkan's
    /// and CUDA's), else the caller closes it, by dropping it.
    pub fd: OwnedFd,
    /// The bytes of the allocation, as `vkAllocateMemory` made it: the buffer's own where it is
    /// `dedicated`, else those of a block other exportable buffers may lie in too.
    pub allocation_size: u64,
    /// The buffer's first byte in the allocation: where its `VkBuffer` is bound (0 where it is
    /// `dedicated`, else wherever the block placed it, at a multiple of 256 bytes), plus, with
    /// `BufferDescriptor::aligned_byte`, that byte's shortfall from the next multiple of 256, so
    /// that the byte lies on one.
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
    /// buffer in memory allocated exportable as an opaque file descriptor, its dedicated
    /// allocation where the device needs one, else a block such buffers share, where
    /// any other buffer lies in the device's other pooled blocks (`exported_memory.rs`). It is a
    /// buffer as any other in every call; a `Managed` one on a device without unified memory
    /// exports the GPU's copy, which the host's reaches as Metal's managed copies do
    /// (`didModifyRange:`, `synchronizeResource:`). A descriptor's `exported_as`
    /// ([`Device::new_buffer_with_descriptor`]) makes such a buffer too, of either handle type,
    /// with the descriptor's other fields.
    ///
    /// The buffer must outlive every import of its memory. On a device loss the memory's contents
    /// are lost, for every import too, and a device made again makes new allocations, each
    /// exported anew.
    ///
    /// Refused as [`Device::new_buffer`] refuses it, and `NotImplemented`
    /// ([`RuntimeGap::ExportableMemory`](crate::RuntimeGap::ExportableMemory)) on a device
    /// without `VK_KHR_external_memory_fd` or that exports no buffer's memory as an opaque file
    /// descriptor.
    pub fn new_exportable_buffer(
        &self,
        length: u64,
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        self.at_serving(|device| device.new_exportable_buffer_now(length, options))
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
        let place = (AddressPlacement::Any.into(), None);
        let buffer =
            self.exportable_buffer(CALL, length, options, place, MemoryExport::OpaqueFd)?;
        Ok(self.visible(buffer))
    }

    /// A buffer of `length` bytes, all zero, with `options` (the caller's, checked as the runtime's
    /// own memory takes them), at `place` in the address space, its byte `aligned` (where given)
    /// on a multiple of [`LINEAR_IMAGE_ALIGNMENT`] of its memory, whose memory is exportable as
    /// `export`'s handle type (the module docs), for `call`: refused as [`Device::new_buffer`]
    /// refuses it, and `NotImplemented` on a device that does not export such memory
    /// (`RuntimeGap::ExportableMemory`, `RuntimeGap::DmaBufExport`). Announced as a new buffer in
    /// system memory is; not yet among the program's resources (`Device::visible`).
    pub(super) fn exportable_buffer(
        &self,
        call: &'static str,
        length: u64,
        options: ResourceOptions,
        (place, aligned): (Place, Option<u64>),
        export: MemoryExport,
    ) -> Result<Buffer, Error> {
        let modes = Modes::of(options)?;
        let class = buffer_class(call, modes)?;
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(call, Problem::BufferLength { length }));
        }
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let shared = self.shared();
        let exports = match export {
            MemoryExport::OpaqueFd => shared.opaque_fd.as_ref(),
            MemoryExport::DmaBuf => shared.dma_buf_export.as_ref(),
        };
        let Some(exports) = exports else {
            return Err(Error::NotImplemented(export.gap()));
        };
        let raw_device = shared.vk();
        let handle = export.handle();
        // The bytes before the buffer's own in its `VkBuffer`, as a plain buffer's
        // (`placed_new_buffer`).
        let lead = aligned.map_or(0, |byte| {
            (LINEAR_IMAGE_ALIGNMENT - byte % LINEAR_IMAGE_ALIGNMENT) % LINEAR_IMAGE_ALIGNMENT
        });
        // Padded to 16 bytes, as any buffer's, so a device fill clears every byte too.
        let size = length
            .checked_add(lead)
            .and_then(|size| size.checked_next_multiple_of(16))
            .ok_or_else(|| nil(call, Problem::BufferLength { length }))?;
        let usage = usage(shared.device_address);
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
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
        // Its own allocation only where the device needs one for the export: a preference alone
        // keeps it in a shared block, where a texture over it may be an image in its memory (the
        // module docs).
        let exported = Exported {
            handle,
            dedicated: exports.dedicated_only
                || dedicated.requires_dedicated_allocation == vk::TRUE,
        };
        // A dedicated allocation is as long as the buffer requires, at offset 0; a shared block
        // places it at a multiple of 256 bytes, as every buffer of the runtime's memory.
        let requirements = match exported.dedicated {
            true => requirements,
            false => placed(requirements),
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
            .and_then(|()| self.address_space_for(place).place(place, length));
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
            offset: lead,
            base: lead,
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
            naming: Naming::New,
        });
        // New memory holds no value until written: zero, as Metal's new buffer reads. Dropping
        // `buffer` on failure releases it.
        buffer.clear_first()?;
        self.degrade_system_memory(&buffer, class, || {
            format!("exportable buffer of {length} bytes")
        });
        Ok(buffer)
    }

    /// The queue family an acquire of an exportable buffer's memory names as its target: the
    /// device's one family for buffers, or `VK_QUEUE_FAMILY_IGNORED` where its buffers are
    /// concurrent among several (`sharing.rs`), as Vulkan has an acquire of such a buffer name it.
    fn external_acquirer(&self) -> u32 {
        match &self.shared().buffer_families[..] {
            [family] => *family,
            _ => vk::QUEUE_FAMILY_IGNORED,
        }
    }
}

impl Device {
    /// A new dma-buf of `memory`, the runtime's own allocated exportable as one
    /// (`vkGetMemoryFdKHR`, as [`Buffer::export_memory_dma_buf`] makes one), the caller's to close
    /// or hand to an import: what a texture's image over a buffer in that memory imports
    /// (`dma_buf.rs`'s `bind_exported_dma_buf_image`). Why not where the device exports no
    /// dma-buf or the driver refuses one; a lost device is `DeviceLost`.
    pub(super) fn dma_buf_of(
        &self,
        memory: vk::DeviceMemory,
    ) -> Result<Result<OwnedFd, String>, Error> {
        let Some(exports) = self.shared().dma_buf_export.as_ref() else {
            return Ok(Err("the device exports no dma-buf".to_owned()));
        };
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(DMA_BUF_HANDLE);
        // SAFETY: the memory is a live allocation of this device's, exportable as a dma-buf
        // (`allocate_exported`); the functions are the device's.
        match unsafe { exports.functions.get_memory_fd(&info) } {
            // SAFETY: a new descriptor Vulkan handed over, the caller's from here on.
            Ok(fd) => Ok(Ok(unsafe { OwnedFd::from_raw_fd(fd) })),
            Err(vk::Result::ERROR_DEVICE_LOST) => Err(Error::DeviceLost),
            Err(error) => Ok(Err(format!(
                "the driver exports no dma-buf of the buffer's memory ({error:?})"
            ))),
        }
    }
}

impl Buffer {
    /// No Metal counterpart (a runtime extension): the buffer's memory as a new opaque file
    /// descriptor (`vkGetMemoryFdKHR`, `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`), with what an
    /// import of it names ([`ExportedMemory`]). Each call answers a new descriptor, the caller's
    /// to close; the allocation, its size and the buffer's offset in it stay the buffer's for
    /// its life.
    ///
    /// The buffer must outlive every import of its memory: its release frees its dedicated
    /// allocation, or gives its bytes of a shared block to the next exportable buffer. On a device
    /// loss the memory's contents are lost for every import too, and the call answers as the
    /// module docs say. What the other API writes there is the device's to read once the caller
    /// says so ([`Buffer::external_write_done`]).
    ///
    /// Refused: a buffer not made exportable as an opaque file descriptor
    /// ([`Device::new_exportable_buffer`], or `MemoryExport::OpaqueFd`; `Invalid`,
    /// `Problem::NotExportable`), one made before its device's recovery
    /// (`NotImplemented(MadeBeforeRecovery)`) or on a lost device (`DeviceLost`), and the driver's
    /// failure.
    pub fn export_memory_fd(&self) -> Result<ExportedMemory, Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        self.export_memory("export_memory_fd", MemoryExport::OpaqueFd)
    }

    /// No Metal counterpart (a runtime extension): the buffer's memory as a new dma-buf
    /// (`vkGetMemoryFdKHR`, `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`), with what an import
    /// of it names ([`ExportedMemory`]): VA-API's `DRM_PRIME_2` surface takes the descriptor, the
    /// allocation's size and each plane's offset, the buffer's `offset` plus the plane's, at
    /// `DRM_FORMAT_MOD_LINEAR`. Each call answers a new descriptor, the caller's to close; the
    /// allocation, its size and the buffer's offset in it stay the buffer's for its life.
    ///
    /// The buffer must outlive every import of its memory, as for [`Buffer::export_memory_fd`].
    /// What the other API writes there is the device's to read once the caller says so
    /// ([`Buffer::external_write_done`]).
    ///
    /// Refused: a buffer not made exportable as a dma-buf (`MemoryExport::DmaBuf`; `Invalid`,
    /// `Problem::NotExportable`), one made before its device's recovery
    /// (`NotImplemented(MadeBeforeRecovery)`) or on a lost device (`DeviceLost`), and the driver's
    /// failure.
    pub fn export_memory_dma_buf(&self) -> Result<ExportedMemory, Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        self.export_memory("export_memory_dma_buf", MemoryExport::DmaBuf)
    }

    /// The buffer's memory as a new descriptor of `export`'s handle type, for `call`.
    fn export_memory(
        &self,
        call: &'static str,
        export: MemoryExport,
    ) -> Result<ExportedMemory, Error> {
        let device = self.device();
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let shared = device.shared();
        let Some((exported, allocation_size, (memory, offset, memory_type_index))) = self
            .exported()
            .filter(|(exported, ..)| exported.handle == export.handle())
        else {
            return Err(invalid(call, Problem::NotExportable));
        };
        let exports = match export {
            MemoryExport::OpaqueFd => shared.opaque_fd.as_ref(),
            MemoryExport::DmaBuf => shared.dma_buf_export.as_ref(),
        };
        let Some(exports) = exports else {
            return Err(invalid(call, Problem::NotExportable));
        };
        // The zeros it waits for as it was made go first, alone (`Buffer::settle_first_work`):
        // the other API may write the memory at once.
        self.settle_first_work(self.at(0).0)?;
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(exported.handle);
        // SAFETY: the memory is the buffer's, live while the buffer is, allocated exportable as
        // the handle type (`allocate_exported`); the functions are the device's.
        let fd = unsafe { exports.functions.get_memory_fd(&info) }.map_err(vulkan)?;
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

    /// What the buffer's memory was allocated exportable as, with the allocation's size, its
    /// memory, the buffer's first byte's offset in it and its memory type; `None` for a buffer
    /// not made exportable.
    pub(super) fn exported(&self) -> Option<(Exported, u64, (vk::DeviceMemory, u64, u32))> {
        let shared = self.device().shared();
        let found = self.own_placement(|placement, offset| {
            let (exported, size) = lock(&shared.memory).exported(placement)?;
            let at = placement.offset + offset;
            Some((
                exported,
                size,
                (placement.memory, at, placement.type_index()),
            ))
        });
        found.flatten()
    }

    /// No Metal counterpart (a runtime extension): the other API that imported the buffer's
    /// memory ([`Buffer::export_memory_fd`], [`Buffer::export_memory_dma_buf`]) wrote it, and the
    /// caller waited for those writes to complete (VA-API's `vaSyncSurface`, a CUDA stream's
    /// synchronize). The device acquires the memory from the queue family Vulkan reserves for
    /// that API (`VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf, `VK_QUEUE_FAMILY_EXTERNAL` for an
    /// opaque file descriptor), which Vulkan needs before the device reads what another API wrote:
    /// its next submission, on any queue, records the acquire before any of its commands, so the
    /// call waits for no GPU work. A host-visible buffer's acquire is submitted at once and waited
    /// for, alone, so that its contents read the writes when the call returns. Textures over the
    /// buffer read what it holds then: a texture's shadow over its bytes (a texture whose rows
    /// no image of the memory holds) copies them in again at its next use in a command buffer
    /// committed after the call.
    ///
    /// Call it once after each of the other API's writes, while no command of the runtime's that
    /// uses the buffer is in flight: the other API writes the memory as its own (the module docs).
    ///
    /// Refused: a buffer not made exportable (`Invalid`, `Problem::NotExportable`), one made
    /// before its device's recovery (`NotImplemented(MadeBeforeRecovery)`) or on a lost device
    /// (`DeviceLost`), and the driver's failure.
    pub fn external_write_done(&self) -> Result<(), Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        const CALL: &str = "external_write_done";
        let device = self.device();
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let Some(export) =
            (self.exported()).and_then(|(exported, ..)| MemoryExport::of_handle(exported.handle))
        else {
            return Err(invalid(CALL, Problem::NotExportable));
        };
        let raw = self.at(0).0;
        // Its own fills go first: the acquire's barrier leads a submission's fills.
        self.settle_first_work(raw)?;
        self.note_acquire((export.writer_family(), device.external_acquirer()));
        // A shadow over the bytes whose image a command buffer's end left standing holds what
        // the buffer held then: its next copy in moves the other API's writes
        // (`shadow_standing.rs`).
        device.shared().standings.written_apart(self.identity().key);
        if self.contents().is_some() {
            self.settle_first_work(raw)?;
        }
        Ok(())
    }
}
