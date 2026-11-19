//! Imports: blocks over memory the consumer brings, the memory of Metal's no-copy buffers
//! (`newBufferWithBytesNoCopy:length:options:deallocator:`). A dma-buf by its fd
//! ([`Device::import_dma_buf`]) or host memory by its address ([`Device::import_host_memory`]),
//! each into the memory type its own rule takes from the handle's query, and each a block the
//! consumer places buffers in as it places them in a block it took.
//!
//! The two rules differ. A dma-buf is imported only into the first type its fd's query offers,
//! and only where that type is host-visible and host-coherent: on NVIDIA's Linux driver
//! (620.70.06) one failed dma-buf import into a later type broke every later allocation of that
//! type, the Shared type among them, so the rule attempts nothing it is not sure of. Host memory
//! is imported into the first type the pointer's query offers that is host-visible and
//! host-coherent.

use crate::block::Block;
use crate::device::{Core, Device};
use crate::error::{Error, Limit};
use crate::floor;
use crate::memory::{self, ImportFailed, Mapping, Release};
use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

/// No Metal counterpart: the handle an import takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImportHandle {
    /// A Linux dma-buf by its fd ([`Device::import_dma_buf`]).
    DmaBuf,
    /// Host memory by its address ([`Device::import_host_memory`]).
    HostMemory,
}

impl ImportHandle {
    fn vk(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            ImportHandle::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            ImportHandle::HostMemory => vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
        }
    }
}

/// What the device imports, read once as it is made.
pub(crate) struct Imports {
    /// `VK_EXT_external_memory_host` and the device's `minImportedHostPointerAlignment`, where
    /// the extension is enabled and the device's buffers import host memory.
    pub(crate) host: Option<(ash::ext::external_memory_host::Device, u64)>,
    /// `VK_KHR_external_memory_fd`, where it and `VK_EXT_external_memory_dma_buf` are enabled and
    /// the device's buffers import a dma-buf.
    #[cfg(unix)]
    pub(crate) dma_buf: Option<ash::khr::external_memory_fd::Device>,
    /// The host-visible and host-coherent memory types, one bit each.
    pub(crate) coherent: u32,
}

impl Imports {
    /// What `device`, made from `physical` with `extensions` enabled, imports.
    ///
    /// # Safety
    ///
    /// `physical` is `instance`'s, of Vulkan 1.3, and `device` was made from it.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        extensions: &[&std::ffi::CStr],
        memory: &vk::PhysicalDeviceMemoryProperties,
    ) -> Imports {
        let enabled = |name: &std::ffi::CStr| extensions.contains(&name);
        let importable = |handle: ImportHandle| {
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
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        };
        let host = (enabled(ash::ext::external_memory_host::NAME)
            && importable(ImportHandle::HostMemory))
        .then(|| {
            let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
            // SAFETY: the caller's; the extension is enabled, so its structure is defined.
            unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
            let alignment = host.min_imported_host_pointer_alignment;
            (
                ash::ext::external_memory_host::Device::new(instance, device),
                alignment,
            )
        });
        #[cfg(unix)]
        let dma_buf = (enabled(ash::khr::external_memory_fd::NAME)
            && enabled(ash::ext::external_memory_dma_buf::NAME)
            && importable(ImportHandle::DmaBuf))
        .then(|| ash::khr::external_memory_fd::Device::new(instance, device));
        let coherent = (0u32..)
            .zip(memory.memory_types_as_slice())
            .filter(|(_, kind)| kind.property_flags.contains(COHERENT))
            .fold(0, |bits, (index, _)| bits | 1 << index);
        Imports {
            host,
            #[cfg(unix)]
            dma_buf,
            coherent,
        }
    }
}

/// The flags both rules ask of an import's memory type.
const COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

/// The dma-buf rule: the first type the fd's query `offered`, where the buffer's requirements
/// `listed` it and it is one of the `coherent` types; none otherwise, whatever else is offered.
#[cfg(unix)]
pub(crate) fn dma_buf_type(offered: u32, listed: u32, coherent: u32) -> Option<u32> {
    let first = offered.trailing_zeros();
    (first < u32::BITS && (listed & coherent) & (1 << first) != 0).then_some(first)
}

/// The host-memory rule: the first type the pointer's query `offered` that the buffer's
/// requirements `listed` and that is one of the `coherent` types.
pub(crate) fn host_type(offered: u32, listed: u32, coherent: u32) -> Option<u32> {
    let bits = offered & listed & coherent;
    (bits != 0).then(|| bits.trailing_zeros())
}

impl Device {
    /// `newBufferWithBytesNoCopy:length:options:deallocator:`'s memory, from a Linux dma-buf: a
    /// block over the first `len` bytes of the dma-buf `fd`, for the consumer to place buffers in
    /// ([`Device::new_buffer`]) as in a block it took. Shared, as Metal's no-copy buffers are:
    /// the host reaches the bytes through the consumer's own mappings of the pages, so the
    /// block's `contents` is `None`, and its zero mark is 0.
    ///
    /// It imports only into the first memory type the fd's query (`vkGetMemoryFdPropertiesKHR`)
    /// offers, and only where that type is host-visible and host-coherent and the buffer's memory
    /// requirements list it, and returns the block with the index of that type. Every refusal
    /// comes before any allocation is attempted, with nothing imported.
    ///
    /// The call takes `fd`, the consumer's own fd of the dma-buf (a dup of it, where the consumer
    /// keeps one). After `Ok` the driver owns it. After a refusal (`Limit`, [`Error::Quiesced`])
    /// or a failed query the runtime closes it: no driver was given it. After a failed import
    /// (`vkAllocateMemory`'s [`Error::Vulkan`]) the runtime closes it only where it still names the
    /// same file: Vulkan leaves it the caller's, but a driver may close it itself, and the number
    /// may name another file since. After a failure past the import (binding its buffer), the
    /// import is freed, and the fd with it.
    ///
    /// The block lives until its last handle and its last buffer, command buffer or upload are
    /// gone. The block-maker thread then frees its memory, which ends the import, and runs
    /// `release`, Metal's deallocator, after that: on the block-maker thread, or at the device's
    /// teardown on the thread that ends it. `release` makes no call into the runtime and drops no
    /// runtime handle. On an `Err` it is dropped without running.
    /// [`BlockMappings::unmap`](crate::BlockMappings::unmap) never runs for an imported block. Each
    /// import is one `VkDeviceMemory` allocation and counts against the device's
    /// `maxMemoryAllocationCount`.
    ///
    /// Metal takes any page-aligned memory; here a device that offers the dma-buf only in a type
    /// the rule does not take refuses it, and the consumer chooses what comes next.
    ///
    /// Err: [`Limit::NoImport`](crate::Limit::NoImport) where the device does not import a dma-buf;
    /// [`Limit::ImportType`](crate::Limit::ImportType) where the rule takes no type;
    /// [`Limit::ImportAlignment`](crate::Limit::ImportAlignment) for a length of 0 or one below
    /// the buffer's memory requirements; [`Limit::BlockLen`](crate::Limit::BlockLen) past the
    /// device's longest allocation or buffer;
    /// [`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations);
    /// [`Limit::BlockIds`](crate::Limit::BlockIds); [`Error::Vulkan`] where the query or the
    /// import fails (the fd is no dma-buf of `len` bytes or more, or the device is out of memory
    /// or lost); [`Error::Quiesced`].
    #[cfg(unix)]
    pub fn import_dma_buf(
        &self,
        fd: std::os::fd::OwnedFd,
        len: u64,
        release: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(Block, u32), Error> {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
        let core = self.core();
        let handle = ImportHandle::DmaBuf;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        check_len(core, handle, len)?;
        let Some(fds) = &core.imports.dma_buf else {
            return Err(Error::Limit(Limit::NoImport { handle }));
        };
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: the extensions are enabled; `fd` is open, and the query takes no ownership.
        unsafe { fds.get_memory_fd_properties(handle.vk(), fd.as_raw_fd(), &mut properties) }
            .map_err(Error::vulkan("vkGetMemoryFdPropertiesKHR"))?;
        let needs = memory::import_requirements(core, len, handle.vk());
        let offered = properties.memory_type_bits;
        let index = dma_buf_type(offered, needs.memory_type_bits, core.imports.coherent).ok_or(
            Error::Limit(Limit::ImportType {
                handle,
                offered,
                listed: needs.memory_type_bits,
            }),
        )?;
        check_size(handle, len, &needs)?;
        let identity = identity(fd.as_raw_fd());
        let raw = fd.into_raw_fd();
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle.vk())
            .fd(raw);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(index)
            .push_next(&mut import)
            .push_next(&mut flags);
        // SAFETY: an import of the dma-buf into a type its query offered and its buffers can use.
        match unsafe { memory::import(core, &info, handle.vk(), None) } {
            Ok(memory) => Ok((claim(core, memory, release), index)),
            Err(ImportFailed::Before(error)) => {
                // Nothing was imported, so the fd is still ours: closed, where it is still the
                // dma-buf.
                if identity.is_some() && identity == self::identity(raw) {
                    // SAFETY: `raw` is open and names the dma-buf the call was given, which no
                    // one else owns.
                    drop(unsafe { OwnedFd::from_raw_fd(raw) });
                }
                Err(error)
            }
            Err(ImportFailed::After(error)) => Err(error),
        }
    }

    /// `newBufferWithBytesNoCopy:length:options:deallocator:`'s memory, from host memory: a block
    /// over the `len` bytes at `pointer`, for the consumer to place buffers in
    /// ([`Device::new_buffer`]) as in a block it took. Shared, as Metal's no-copy buffers are:
    /// its `contents` is `pointer`, and its zero mark is 0.
    ///
    /// It imports into the first memory type the pointer's query
    /// (`vkGetMemoryHostPointerPropertiesEXT`) offers that is host-visible and host-coherent and
    /// that the buffer's memory requirements list, and returns the block with the index of that
    /// type. Every refusal comes before any allocation is attempted, with nothing imported. A
    /// driver may refuse memory of some kinds at the import itself (RADV, Mesa 26.2, refused the
    /// pages of a shared mapping of a memfd with `VK_ERROR_INVALID_EXTERNAL_HANDLE`, after its
    /// query offered a type), which returns its [`Error::Vulkan`] with nothing imported.
    ///
    /// The block lives, and `release` runs, as for [`Device::import_dma_buf`]: `release`,
    /// Metal's deallocator, runs once the memory is freed, and from then the pages are the
    /// consumer's alone. Each import is one `VkDeviceMemory` allocation and counts against the
    /// device's `maxMemoryAllocationCount`.
    ///
    /// Err: [`Limit::NoImport`](crate::Limit::NoImport) where the device does not import host
    /// memory; [`Limit::ImportType`](crate::Limit::ImportType) where the rule takes no type;
    /// [`Limit::ImportAlignment`](crate::Limit::ImportAlignment) for a length of 0, or an address
    /// or length that is not a multiple of the device's `minImportedHostPointerAlignment` (or
    /// below the buffer's memory requirements); [`Limit::BlockLen`](crate::Limit::BlockLen) past
    /// the device's longest allocation or buffer;
    /// [`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations);
    /// [`Limit::BlockIds`](crate::Limit::BlockIds); [`Error::Vulkan`] where the query or the
    /// import fails; [`Error::Quiesced`].
    ///
    /// # Safety
    ///
    /// The `len` bytes at `pointer` are host memory mapped readable and writable, and stay
    /// mapped, and not mapped anew, until `release` runs; where the call returns `Err`, they are
    /// the consumer's again at once.
    pub unsafe fn import_host_memory(
        &self,
        pointer: NonNull<u8>,
        len: u64,
        release: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(Block, u32), Error> {
        let core = self.core();
        let handle = ImportHandle::HostMemory;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        check_len(core, handle, len)?;
        let Some((host, alignment)) = &core.imports.host else {
            return Err(Error::Limit(Limit::NoImport { handle }));
        };
        let alignment = *alignment;
        let address = pointer.addr().get() as u64;
        if !address.is_multiple_of(alignment) || !len.is_multiple_of(alignment) {
            return Err(Error::Limit(Limit::ImportAlignment {
                handle,
                len,
                alignment,
            }));
        }
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the extension is enabled; the pointer is host memory (the caller's).
        unsafe {
            (host.fp().get_memory_host_pointer_properties_ext)(
                host.device(),
                handle.vk(),
                pointer.as_ptr().cast_const().cast(),
                &mut properties,
            )
        }
        .result()
        .map_err(Error::vulkan("vkGetMemoryHostPointerPropertiesEXT"))?;
        let needs = memory::import_requirements(core, len, handle.vk());
        let offered = properties.memory_type_bits;
        let index = host_type(offered, needs.memory_type_bits, core.imports.coherent).ok_or(
            Error::Limit(Limit::ImportType {
                handle,
                offered,
                listed: needs.memory_type_bits,
            }),
        )?;
        check_size(handle, len, &needs)?;
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle.vk())
            .host_pointer(pointer.as_ptr().cast());
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(index)
            .push_next(&mut import)
            .push_next(&mut flags);
        // SAFETY: an import of the caller's memory, at the device's alignment, into a type its
        // query offered and its buffers can use.
        let memory =
            unsafe { memory::import(core, &info, handle.vk(), Some(Mapping::new(pointer))) }
                .map_err(ImportFailed::error)?;
        Ok((claim(core, memory, release), index))
    }
}

/// A length the device holds in one block: Err for 0 and past its longest.
fn check_len(core: &Core, handle: ImportHandle, len: u64) -> Result<(), Error> {
    if len == 0 {
        return Err(Error::Limit(Limit::ImportAlignment {
            handle,
            len,
            alignment: 1,
        }));
    }
    if len > core.max_block_len {
        return Err(Error::Limit(Limit::BlockLen {
            len,
            max: core.max_block_len,
        }));
    }
    Ok(())
}

/// The import's length holds the buffer's memory requirements.
fn check_size(handle: ImportHandle, len: u64, needs: &vk::MemoryRequirements) -> Result<(), Error> {
    if needs.size > len {
        return Err(Error::Limit(Limit::ImportAlignment {
            handle,
            len,
            alignment: needs.alignment,
        }));
    }
    Ok(())
}

/// The consumer's handle of an imported block.
fn claim(core: &Arc<Core>, memory: memory::Memory, release: Option<Release>) -> Block {
    let info = memory.info();
    let block = memory::Block::imported(memory, Arc::clone(core), release);
    Block::claim(core, info, block)
}

/// The file `fd` names, by its device and inode; `None` where it names none.
#[cfg(unix)]
fn identity(fd: std::os::fd::RawFd) -> Option<(libc::dev_t, libc::ino_t)> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `stat` is written by the call where it succeeds, and read only then.
    match unsafe { libc::fstat(fd, stat.as_mut_ptr()) } {
        0 => {
            // SAFETY: the call succeeded.
            let stat = unsafe { stat.assume_init() };
            Some((stat.st_dev, stat.st_ino))
        }
        _ => None,
    }
}
