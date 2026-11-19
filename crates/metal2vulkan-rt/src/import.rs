//! Imports: blocks over memory the consumer brings, the memory of Metal's no-copy buffers
//! (`newBufferWithBytesNoCopy:length:options:deallocator:`). A dma-buf by its fd
//! (`Device::import_dma_buf`, on Unix), several dma-bufs behind one buffer
//! (`Device::import_dma_bufs`, on Unix) or host memory by its address
//! ([`Device::import_host_memory`]), each into the memory type its own rule takes from the
//! handle's query, and each a block the consumer places buffers in as it places them in a block
//! it took.
//!
//! The two rules differ. A dma-buf is imported only into the first type its fd's query offers,
//! and only where that type is host-visible and host-coherent: on NVIDIA's Linux driver
//! (620.70.06) one failed dma-buf import into a later type broke every later allocation of that
//! type, the Shared type among them, so the rule attempts nothing it is not sure of. Host memory
//! is imported into the first type the pointer's query offers that is host-visible and
//! host-coherent.

use crate::block::Block;
#[cfg(unix)]
use crate::counters::Metric;
#[cfg(unix)]
use crate::device::lock;
use crate::device::{Core, Device};
use crate::error::{Error, Limit};
use crate::floor;
use crate::memory::{self, ImportFailed, Mapping, Own, Release};
use ash::vk;
use std::ptr::NonNull;
#[cfg(unix)]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
#[cfg(unix)]
use std::sync::OnceLock;

/// No Metal counterpart: the handle an import takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImportHandle {
    /// A Linux dma-buf by its fd (`Device::import_dma_buf`, on Unix).
    DmaBuf,
    /// Several Linux dma-bufs by their fds, behind one buffer ([`Device::import_dma_bufs`]).
    DmaBufs,
    /// Host memory by its address ([`Device::import_host_memory`]).
    HostMemory,
}

impl ImportHandle {
    fn vk(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            ImportHandle::DmaBuf | ImportHandle::DmaBufs => {
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
            }
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
    /// The import over several dma-bufs' sparse binding, where the device was made with it
    /// ([`binds_dma_bufs`]) and its granule holds ([`Sparse::new`]).
    #[cfg(unix)]
    pub(crate) sparse: Option<Sparse>,
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
            #[cfg(unix)]
            sparse: None,
            coherent,
        }
    }

    /// The imports with the sparse binding of an import over several dma-bufs, where the device
    /// was made with it (`space`, its `sparseAddressSpaceSize`, is `Some` where
    /// [`binds_dma_bufs`] held) and its granule holds ([`Sparse::new`]).
    #[cfg(unix)]
    pub(crate) fn with_sparse(self, device: &ash::Device, space: Option<u64>) -> Imports {
        let sparse = (space.filter(|_| self.dma_buf.is_some()))
            .and_then(|space| Sparse::new(device, self.coherent, space));
        Imports { sparse, ..self }
    }

    /// At the device's teardown, once it is idle and every import is freed: the binds' timeline.
    #[cfg(unix)]
    pub(crate) fn destroy(&self, device: &ash::Device) {
        if let Some(&timeline) = self
            .sparse
            .as_ref()
            .and_then(|sparse| sparse.timeline.get())
        {
            // SAFETY: the device is idle, and no wait or signal names the timeline.
            unsafe { device.destroy_semaphore(timeline, None) };
        }
    }
}

/// Whether device creation enables `sparseBinding` for imports over several dma-bufs
/// ([`Device::import_dma_bufs`]): the device reports the feature, the runtime's queue `family`
/// binds sparse memory, `VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf` are in
/// `extensions`, and a sparse buffer of the floor's usage imports a dma-buf with no dedicated
/// allocation (`vkGetPhysicalDeviceExternalBufferProperties`, as the Vulkan rules for a sparse
/// bind of imported memory name the handle in the buffer, `VUID-VkSparseMemoryBind-memory-02731`).
/// Elsewhere the feature stays as the floor and the consumer leave it, and the import refuses.
/// The feature changes nothing else: only a buffer made with `VK_BUFFER_CREATE_SPARSE_BINDING_BIT`
/// binds sparse, and only this import makes one.
///
/// # Safety
///
/// `physical` is `instance`'s, of Vulkan 1.3.
pub(crate) unsafe fn binds_dma_bufs(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: &vk::QueueFamilyProperties,
    extensions: &[&std::ffi::CStr],
) -> bool {
    let enabled = |name: &std::ffi::CStr| extensions.contains(&name);
    if !enabled(ash::khr::external_memory_fd::NAME)
        || !enabled(ash::ext::external_memory_dma_buf::NAME)
        || !family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
    {
        return false;
    }
    // SAFETY: the caller's.
    let features = unsafe { instance.get_physical_device_features(physical) };
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .flags(vk::BufferCreateFlags::SPARSE_BINDING)
        .usage(floor::BUFFER_USAGE)
        .handle_type(ImportHandle::DmaBufs.vk());
    let mut properties = vk::ExternalBufferProperties::default();
    // SAFETY: the caller's.
    unsafe {
        instance.get_physical_device_external_buffer_properties(physical, &info, &mut properties)
    };
    let offered = properties
        .external_memory_properties
        .external_memory_features;
    features.sparse_binding == vk::TRUE
        && offered.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        && !offered.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
}

/// The sparse binding of imports over several dma-bufs ([`Device::import_dma_bufs`]).
///
/// One `vkQueueBindSparse` on the runtime's queue binds each such buffer to its imports, under
/// the queue's lock, and signals the next value of the binds' timeline; nothing waits for it
/// there. Vulkan orders no sparse bind against command buffers, "even within a single queue",
/// and a semaphore wait's second scope is every command of its batch "and all commands that occur
/// later in submission order", so the next submission waits for that value once ([`Sparse::owed`]),
/// and every later one is ordered after it. The block's free waits for the value on the
/// block-maker thread before the buffer goes, for a block no command buffer used.
#[cfg(unix)]
pub(crate) struct Sparse {
    /// What a chunk's length is a multiple of ([`Device::dma_buf_granule`]).
    pub(crate) granule: u64,
    /// The device's `sparseAddressSpaceSize`.
    space: u64,
    /// The bytes the live sparse buffers' memory requirements name, against `space`
    /// (`VUID-vkCreateBuffer-flags-00911`).
    live: AtomicU64,
    /// The binds' timeline, made at the first bind.
    timeline: OnceLock<vk::Semaphore>,
    /// The value the last bind signalled. Read and written under the runtime queue's lock.
    last: AtomicU64,
    /// The value of a bind no submission has waited for yet, 0 where none. Written under the
    /// runtime queue's lock, by a bind, and taken under it by the next submission.
    owed: AtomicU64,
}

#[cfg(unix)]
impl Sparse {
    /// The sparse binding on a device made with it, where its granule holds: a sparse buffer of
    /// the floor's usage for dma-bufs takes every host-visible and host-coherent type a plain one
    /// takes, so wherever the dma-buf rule takes a type for one import of an fd it takes the same
    /// type for a chunk of this one. Its granule is that buffer's alignment, in which Vulkan binds
    /// a sparse buffer's memory (`VUID-VkSparseMemoryBind-resourceOffset-09491`), rounded up to
    /// the host's page. `None` elsewhere.
    fn new(device: &ash::Device, coherent: u32, space: u64) -> Option<Sparse> {
        // SAFETY: `sysconf` reads a constant of the host.
        let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
        let handle = ImportHandle::DmaBufs.vk();
        let plain = requirements(device, crate::memory::GRANULE, handle, false);
        let sparse = requirements(device, crate::memory::GRANULE, handle, true);
        let granule = (sparse.alignment.max(1).checked_next_multiple_of(page))
            .filter(|granule| granule.is_multiple_of(sparse.alignment.max(1)))?;
        (plain.memory_type_bits & coherent & !sparse.memory_type_bits == 0).then(|| Sparse {
            granule,
            space,
            live: AtomicU64::new(0),
            timeline: OnceLock::new(),
            last: AtomicU64::new(0),
            owed: AtomicU64::new(0),
        })
    }

    /// Takes `len` bytes of the sparse address space for a buffer whose memory requirements
    /// name them. Err: [`Limit::SparseAddressSpace`].
    fn reserve(&self, len: u64) -> Result<(), Error> {
        (self.live)
            .fetch_update(Relaxed, Relaxed, |live| {
                live.checked_add(len).filter(|&total| total <= self.space)
            })
            .map(drop)
            .map_err(|live| {
                Error::Limit(Limit::SparseAddressSpace {
                    len,
                    live,
                    max: self.space,
                })
            })
    }

    /// Gives back `len` bytes of the sparse address space, once their buffer is destroyed.
    pub(crate) fn unreserve(&self, len: u64) {
        self.live.fetch_sub(len, Relaxed);
    }

    /// Binds `binds` into `buffer` by one `vkQueueBindSparse` on the runtime's queue, under its
    /// lock, signalling the timeline's next value, which it returns and which the next submission
    /// waits for. Counted in [`Metrics::sparse_binds`](crate::Metrics::sparse_binds).
    ///
    /// Err: [`Error::Vulkan`] where the timeline cannot be made or the bind fails.
    ///
    /// # Safety
    ///
    /// `buffer` is a sparse buffer of `core`'s device that nothing uses, and each bind lies in it
    /// and in its memory, at its alignment, in a type it takes.
    unsafe fn bind(
        &self,
        core: &Core,
        buffer: vk::Buffer,
        binds: &[vk::SparseMemoryBind],
    ) -> Result<u64, Error> {
        let queue = lock(&core.queues.runtime);
        let timeline = match self.timeline.get() {
            Some(&timeline) => timeline,
            None => {
                let made = crate::event::timeline(&core.device)?;
                // Under the queue's lock, so no other bind set it first.
                *self.timeline.get_or_init(|| made)
            }
        };
        let value = self.last.load(Relaxed) + 1;
        let buffer_binds = [vk::SparseBufferMemoryBindInfo::default()
            .buffer(buffer)
            .binds(binds)];
        let signals = [timeline];
        let values = [value];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::BindSparseInfo::default()
            .buffer_binds(&buffer_binds)
            .signal_semaphores(&signals)
            .push_next(&mut timeline_info);
        core.counters.count(Metric::SparseBinds);
        // SAFETY: the caller's for the buffer and the binds; the queue's family binds sparse
        // memory ([`binds_dma_bufs`]) and the queue is held; the value is above every value the
        // timeline was given, all of them signalled under this lock.
        unsafe {
            core.device
                .queue_bind_sparse(*queue, &[info], vk::Fence::null())
        }
        .map_err(Error::vulkan("vkQueueBindSparse"))?;
        self.last.store(value, Relaxed);
        self.owed.store(value, Relaxed);
        drop(queue);
        Ok(value)
    }

    /// The timeline and the value of a bind no submission has waited for yet, which the calling
    /// submission now waits for: taken, so only the next one waits. Called under the runtime
    /// queue's lock by the submission that is about to be made.
    pub(crate) fn owed(&self) -> Option<(vk::Semaphore, u64)> {
        let value = self.owed.load(Relaxed);
        if value == 0 {
            return None;
        }
        self.owed.store(0, Relaxed);
        self.timeline.get().map(|&timeline| (timeline, value))
    }

    /// Waits on the host until the bind that signals `value` completed: before its buffer is
    /// destroyed (`VUID-vkDestroyBuffer-buffer-00922`). At once where a submission waited for it
    /// and retired, as each that used the buffer did; a lost device ends the wait.
    pub(crate) fn wait(&self, device: &ash::Device, value: u64) {
        let Some(&timeline) = self.timeline.get() else {
            return;
        };
        let semaphores = [timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the timeline is live, and the bind that signals `value` was submitted. A lost
        // device fails the wait, and then nothing is pending.
        let _ = unsafe { device.wait_semaphores(&info, u64::MAX) };
    }
}

/// The memory requirements of a buffer of `len` bytes and the floor's usage for `handle`, made
/// with `VK_BUFFER_CREATE_SPARSE_BINDING_BIT` where `sparse` holds.
#[cfg(unix)]
fn requirements(
    device: &ash::Device,
    len: u64,
    handle: vk::ExternalMemoryHandleTypeFlags,
    sparse: bool,
) -> vk::MemoryRequirements {
    let flags = match sparse {
        true => vk::BufferCreateFlags::SPARSE_BINDING,
        false => vk::BufferCreateFlags::empty(),
    };
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
    let buffer = vk::BufferCreateInfo::default()
        .flags(flags)
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let of = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut needs = vk::MemoryRequirements2::default();
    // SAFETY: the device is live and the create info valid: its buffers import `handle`, and a
    // sparse one is asked only of a device made with `sparseBinding` ([`binds_dma_bufs`]).
    unsafe { device.get_device_buffer_memory_requirements(&of, &mut needs) };
    needs.memory_requirements
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
        match unsafe { memory::import(core, &info, handle.vk(), None, Own::Import) } {
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

    /// No Metal counterpart: what [`Device::import_dma_bufs`] cuts at. Each dma-buf's length,
    /// and so its offset in the block, is a multiple of it: the alignment in which the device binds
    /// a sparse buffer's memory, rounded up to the host's page (64 KiB on RADV of Mesa 26.2.6, 4
    /// KiB on lavapipe). `None` where the call refuses every import
    /// ([`Limit::NoImport`](crate::Limit::NoImport)): the device does not bind dma-bufs into a
    /// sparse buffer (Mesa's Intel driver, 26.2.6, imports no dma-buf for one), or its sparse
    /// buffer does not take every type a plain one takes.
    #[cfg(unix)]
    pub fn dma_buf_granule(&self) -> Option<u64> {
        (self.core().imports.sparse.as_ref()).map(|sparse| sparse.granule)
    }

    /// `newBufferWithBytesNoCopy:length:options:deallocator:`'s memory, from several Linux
    /// dma-bufs: one block over the dma-bufs of `chunks`, each fd with its length, in order, each
    /// starting where the one before ends, for the consumer to place buffers in
    /// ([`Device::new_buffer`]) as in a block it took. Metal's memory is one range; a consumer
    /// whose pages take more than one dma-buf (a udmabuf holds a bounded number of a memfd's
    /// ranges) cuts them at [`Device::dma_buf_granule`], the last padded, and imports them here.
    /// As for [`Device::import_dma_buf`], the block is Shared, its `contents` is `None` and its
    /// zero mark is 0; its `gpu_address`, and every buffer of it, run across the dma-bufs as over
    /// one range: the block's `VkBuffer` is one sparse buffer
    /// (`VK_BUFFER_CREATE_SPARSE_BINDING_BIT`) that each dma-buf's import is bound into at its
    /// offset.
    ///
    /// Each fd's import takes the dma-buf rule's type ([`Device::import_dma_buf`]) of the types the
    /// sparse buffer's memory requirements list, and every fd the same one, which the call returns
    /// with the block. Every refusal comes before any allocation is attempted, with nothing
    /// imported. The call takes owned fds, as [`Device::import_dma_buf`] does, and returns a block,
    /// so a consumer that imports one dma-buf or several places its buffers the same way.
    ///
    /// One `vkQueueBindSparse` on the runtime's queue binds the imports, and the call does not
    /// wait for it: the runtime's next `vkQueueSubmit2` waits for it, once, and so does the block's
    /// free (on the block-maker thread) where nothing used the block. So the caller waits as long as
    /// one import's `vkAllocateMemory` takes for each dma-buf, and one queue call. Counted in
    /// [`Metrics::sparse_binds`](crate::Metrics::sparse_binds).
    ///
    /// The fds: after `Ok` the driver owns each. After a refusal or a failed query the runtime
    /// closes them all: no driver was given any. After a failed import the imports made are freed,
    /// and their fds with them; the failing fd is closed only where it still names the same file
    /// (as for [`Device::import_dma_buf`]), and the rest are closed.
    ///
    /// The block lives until its last handle and its last buffer, command buffer or upload are
    /// gone. The block-maker thread then destroys its buffer, frees each import, which ends it, and
    /// runs `release`, Metal's deallocator, after the last: as for [`Device::import_dma_buf`].
    /// Each dma-buf's import is one `VkDeviceMemory` allocation: it counts against the device's
    /// `maxMemoryAllocationCount`, and in [`Gauges::imports`](crate::Gauges::imports) and
    /// [`Gauges::blocks`](crate::Gauges::blocks) while it lives.
    ///
    /// Err: [`Limit::NoImport`](crate::Limit::NoImport) where [`Device::dma_buf_granule`] is
    /// `None`; [`Limit::ImportAlignment`](crate::Limit::ImportAlignment) for no fds, a length of
    /// 0 or one that is no multiple of the granule (or of the sparse buffer's alignment);
    /// [`Limit::BlockLen`](crate::Limit::BlockLen) past the device's longest allocation or buffer;
    /// [`Limit::SparseAddressSpace`](crate::Limit::SparseAddressSpace);
    /// [`Limit::ImportType`](crate::Limit::ImportType) where the rule takes no type for an fd, or
    /// another type than the first fd's (then `listed` names that type alone);
    /// [`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations);
    /// [`Limit::BlockIds`](crate::Limit::BlockIds); [`Error::Vulkan`] where a query, the buffer,
    /// an import or the bind fails (an fd is no dma-buf of its length or more, or the device is out
    /// of memory or lost); [`Error::Quiesced`].
    #[cfg(unix)]
    pub fn import_dma_bufs(
        &self,
        chunks: Vec<(std::os::fd::OwnedFd, u64)>,
        release: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(Block, u32), Error> {
        use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
        let core = self.core();
        let handle = ImportHandle::DmaBufs;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let (Some(fds), Some(sparse)) = (&core.imports.dma_buf, &core.imports.sparse) else {
            return Err(Error::Limit(Limit::NoImport { handle }));
        };
        let granule = sparse.granule;
        let misaligned = |len: u64, alignment: u64| {
            Error::Limit(Limit::ImportAlignment {
                handle,
                len,
                alignment,
            })
        };
        if chunks.is_empty() {
            return Err(misaligned(0, granule));
        }
        let mut len = 0u64;
        for &(_, chunk) in &chunks {
            if chunk == 0 || !chunk.is_multiple_of(granule) {
                return Err(misaligned(chunk, granule));
            }
            len = len.saturating_add(chunk);
        }
        check_len(core, handle, len)?;
        // SAFETY: the device was made with `sparseBinding` for such buffers (`sparse` is `Some`),
        // and `len` is at most its `maxBufferSize` (`check_len`).
        let (buffer, needs) = unsafe { memory::sparse_buffer(core, len) }?;
        let destroy = || {
            // SAFETY: nothing uses or binds the buffer.
            unsafe { core.device.destroy_buffer(buffer, None) }
        };
        if !granule.is_multiple_of(needs.alignment.max(1)) {
            destroy();
            return Err(misaligned(len, needs.alignment));
        }
        let mut index = None;
        for (fd, _) in &chunks {
            let mut properties = vk::MemoryFdPropertiesKHR::default();
            // SAFETY: the extensions are enabled; `fd` is open, and the query takes no ownership.
            let queried = unsafe {
                fds.get_memory_fd_properties(handle.vk(), fd.as_raw_fd(), &mut properties)
            };
            if let Err(result) = queried {
                destroy();
                return Err(Error::Vulkan {
                    call: "vkGetMemoryFdPropertiesKHR",
                    result,
                });
            }
            let offered = properties.memory_type_bits;
            let listed = index.map_or(needs.memory_type_bits, |first: u32| {
                needs.memory_type_bits & (1 << first)
            });
            match dma_buf_type(offered, listed, core.imports.coherent) {
                Some(taken) => index = Some(taken),
                None => {
                    destroy();
                    return Err(Error::Limit(Limit::ImportType {
                        handle,
                        offered,
                        listed,
                    }));
                }
            }
        }
        let Some(index) = index else {
            destroy();
            return Err(misaligned(0, granule));
        };
        if let Err(error) = sparse.reserve(needs.size) {
            destroy();
            return Err(error);
        }
        let undo = |made: &[(vk::DeviceMemory, u64)]| {
            for &(memory, _) in made {
                // SAFETY: nothing is bound to the import yet, and nothing uses it.
                unsafe { memory::free_chunk(core, memory) };
            }
            sparse.unreserve(needs.size);
            destroy();
        };
        let numbers = match core.pool.own_numbers() {
            Ok(numbers) => numbers,
            Err(error) => {
                undo(&[]);
                return Err(error);
            }
        };
        let mut made = Vec::with_capacity(chunks.len());
        let mut rest = chunks.into_iter();
        for (fd, chunk) in rest.by_ref() {
            let identity = identity(fd.as_raw_fd());
            let raw = fd.into_raw_fd();
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle.vk())
                .fd(raw);
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(chunk)
                .memory_type_index(index)
                .push_next(&mut import)
                .push_next(&mut flags);
            // SAFETY: an import of the dma-buf into a type its query offered and the sparse
            // buffer takes.
            match unsafe { memory::import_chunk(core, &info) } {
                Ok(memory) => made.push((memory, chunk)),
                Err(error) => {
                    // Nothing was imported of this fd, so it is still ours: closed, where it is
                    // still the dma-buf; the fds after it close as `rest` drops.
                    if identity.is_some() && identity == self::identity(raw) {
                        // SAFETY: `raw` is open and names the dma-buf the call was given, which
                        // no one else owns.
                        drop(unsafe { OwnedFd::from_raw_fd(raw) });
                    }
                    undo(&made);
                    return Err(error);
                }
            }
        }
        let mut offset = 0;
        let binds: Vec<vk::SparseMemoryBind> = (made.iter())
            .map(|&(memory, size)| {
                let bind = vk::SparseMemoryBind::default()
                    .resource_offset(offset)
                    .size(size)
                    .memory(memory);
                offset += size;
                bind
            })
            .collect();
        // SAFETY: the buffer is new and unused; each bind lies in it at a multiple of its
        // alignment, from the first byte of an import of at least its size, of a type the buffer
        // takes, imported for the handle the buffer names (`VUID-VkSparseMemoryBind-memory-02731`).
        let bind = match unsafe { sparse.bind(core, buffer, &binds) } {
            Ok(bind) => bind,
            Err(error) => {
                undo(&made);
                return Err(error);
            }
        };
        let parts = memory::SparseParts {
            buffer,
            len,
            chunks: made,
            index,
            bind,
            space: needs.size,
            numbers,
        };
        let block = memory::sparse_block(core, parts, release);
        let info = block.memory.info();
        Ok((Block::claim(core, info, block), index))
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
    /// The pages are pinnable host memory: anonymous memory, or a mapping of a file (a memfd's
    /// among them). Pages the consumer maps through a dma-buf's fd go through
    /// `Device::import_dma_buf` (Unix) instead. On NVIDIA's Linux driver (620.70.06, an RTX
    /// 3090), a pointer into a mapping of a udmabuf is refused with
    /// `VK_ERROR_OUT_OF_DEVICE_MEMORY` in each type its query offers, and the process's later
    /// allocations fail too: a device-local one after either refusal, and a host-cached one after
    /// the refusal in the host-cached type.
    ///
    /// The block lives, and `release` runs, as for `Device::import_dma_buf` (Unix): `release`,
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
        let memory = unsafe {
            memory::import(
                core,
                &info,
                handle.vk(),
                Some(Mapping::new(pointer)),
                Own::Import,
            )
        }
        .map_err(ImportFailed::error)?;
        Ok((claim(core, memory, release), index))
    }
}

/// Makes a dma-buf surface's buffer ([`crate::Surface::buffer`]): one more import of the
/// surface's dma-buf, of its `len` bytes, through a duplicate of `fd`, the runtime's own fd of it,
/// into memory type `index`, the one the planes' imports took, with device addresses, bound to a
/// buffer at its first byte. Private, never mapped; counted in the surface gauges. Vulkan lets one
/// payload be imported many times ("multiple times into a given Vulkan instance. In all cases,
/// each import operation must create a distinct `VkDeviceMemory` object"). The import attempts no
/// other type, as the dma-buf rule attempts nothing it is not sure of: where the fd's query or
/// the buffer's memory requirements do not take `index`, or the requirements ask more than `len`,
/// it refuses before any allocation. `fd` stays the runtime's whatever the answer.
///
/// Err: [`Limit::NoImport`] where the device's buffers import no dma-buf; [`Limit::ExportType`]
/// where the type or the length is refused; [`Error::Fd`]; [`Limit::MemoryAllocations`];
/// [`Limit::BlockIds`]; [`Error::Vulkan`] where the query or the import fails.
#[cfg(unix)]
pub(crate) fn surface_import(
    core: &Arc<Core>,
    fd: &std::os::fd::OwnedFd,
    len: u64,
    index: u32,
) -> Result<memory::Block, Error> {
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
    let handle = ImportHandle::DmaBuf;
    let Some(fds) = &core.imports.dma_buf else {
        return Err(Error::Limit(Limit::NoImport { handle }));
    };
    let mut properties = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: the extensions are enabled; `fd` is open, and the query takes no ownership.
    unsafe { fds.get_memory_fd_properties(handle.vk(), fd.as_raw_fd(), &mut properties) }
        .map_err(Error::vulkan("vkGetMemoryFdPropertiesKHR"))?;
    let needs = memory::import_requirements(core, len, handle.vk());
    let takes = properties.memory_type_bits & needs.memory_type_bits;
    if takes & (1 << index) == 0 || needs.size > len {
        return Err(Error::Limit(Limit::ExportType {
            handle: crate::ExportHandle::DmaBuf,
            listed: Some(takes),
        }));
    }
    let raw = fd.try_clone().map_err(Error::Fd)?.into_raw_fd();
    let identity = identity(raw);
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
    let own = Own::Surface { import: true };
    // SAFETY: an import of the dma-buf into a type its query offered, its buffers list and its
    // planes' imports took, of at least the buffer's size.
    match unsafe { memory::import(core, &info, handle.vk(), None, own) } {
        Ok(memory) => Ok(memory::Block::new(memory, Arc::clone(core))),
        Err(ImportFailed::Before(error)) => {
            // Nothing was imported, so the duplicate is still ours: closed, where it is still
            // the dma-buf.
            if identity.is_some() && identity == self::identity(raw) {
                // SAFETY: `raw` is open and names the dma-buf, and no one else owns it.
                drop(unsafe { OwnedFd::from_raw_fd(raw) });
            }
            Err(error)
        }
        Err(ImportFailed::After(error)) => Err(error),
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
