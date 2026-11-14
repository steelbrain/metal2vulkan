//! A command buffer's completion as a sync file (`command_buffer/sync_file.rs`), imported into a
//! second `VkDevice`, made by the test on another vendor's physical device where the instance
//! lists one (the GPU itself on a venue with one): the command buffer fills host pages through a
//! no-copy buffer, and the second device's submission that waits for the sync file copies the
//! pages and finds every byte filled (`SYNC-FILE:` lines name both devices, and whether the GPU
//! still ran when the descriptor was handed out). Its refusals beside it; a device the runtime
//! refuses the sync file on (`RuntimeGap::SyncFileExport`) must lack the extension or the export.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::{Error, Invalid, Problem, RuntimeGap, SyncFileProblem};
use crate::{CommandBuffer, CommandBufferStatus, CommandQueue, HostMemory, ResourceOptions};
use ash::vk;
use std::alloc::Layout;
use std::ffi::CStr;
use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};
use std::ptr::NonNull;

/// The bytes the command buffer fills: enough that its GPU still runs as the export returns.
const LENGTH: u64 = 256 << 20;

/// The byte it fills them with.
const BYTE: u8 = 0x5a;

/// Zeroed host pages of [`LENGTH`] bytes, aligned to 64 KiB (past any device's host pointer
/// alignment), freed when dropped: made before every buffer over them, so dropped after.
struct Pages(NonNull<u8>, Layout);

impl Pages {
    fn new() -> Pages {
        let layout = Layout::from_size_align(LENGTH as usize, 64 << 10).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        Pages(NonNull::new(pointer).unwrap(), layout)
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
    }
}

/// Whether `physical` of `instance` lists `extension`.
fn lists(instance: &ash::Instance, physical: vk::PhysicalDevice, extension: &CStr) -> bool {
    // SAFETY: the physical device is the instance's.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
    (listed.iter()).any(|listed| listed.extension_name_as_c_str() == Ok(extension))
}

/// Whether `physical` of `instance` lists `VK_KHR_external_semaphore_fd` and answers `feature`
/// for a binary semaphore's sync file.
fn sync_fd(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    feature: vk::ExternalSemaphoreFeatureFlags,
) -> bool {
    let query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: as above, at Vulkan 1.2.
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            physical,
            &query,
            &mut properties,
        )
    };
    lists(instance, physical, ash::khr::external_semaphore_fd::NAME)
        && properties.external_semaphore_features.contains(feature)
}

/// A physical device of `instance` that imports a binary semaphore's sync file and host pages by
/// their pointer: another vendor's than `ours` where one does (a discrete one first), else
/// another device, else `ours`.
fn importer(instance: &ash::Instance, ours: vk::PhysicalDevice) -> Option<vk::PhysicalDevice> {
    // SAFETY: the instance is live.
    let all = unsafe { instance.enumerate_physical_devices() }.unwrap();
    // SAFETY: the physical devices are the instance's.
    let properties = |physical| unsafe { instance.get_physical_device_properties(physical) };
    let vendor = properties(ours).vendor_id;
    let able = |physical: &vk::PhysicalDevice| {
        sync_fd(
            instance,
            *physical,
            vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
        ) && lists(instance, *physical, ash::ext::external_memory_host::NAME)
    };
    let rank = |physical: &vk::PhysicalDevice| {
        let theirs = properties(*physical);
        let other_vendor = theirs.vendor_id != vendor;
        let discrete = theirs.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let other = *physical != ours;
        (!other_vendor, !discrete, !other)
    };
    all.into_iter().filter(able).min_by_key(rank)
}

/// The test's own device on `physical`: its queue, the family's index, and its functions for host
/// pages and sync files.
struct Second {
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    host: ash::ext::external_memory_host::Device,
    semaphores: ash::khr::external_semaphore_fd::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
}

impl Second {
    fn new(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Second {
        // SAFETY: the physical device is the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = (families.iter())
            .position(|family| {
                (family.queue_flags).intersects(
                    vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                )
            })
            .unwrap() as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = [
            ash::khr::external_semaphore_fd::NAME.as_ptr(),
            ash::ext::external_memory_host::NAME.as_ptr(),
        ];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions);
        // SAFETY: the physical device is the instance's and lists both extensions (`importer`).
        let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        // SAFETY: the family's first queue, which the device was made with.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: as above.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        Second {
            host: ash::ext::external_memory_host::Device::new(instance, &device),
            semaphores: ash::khr::external_semaphore_fd::Device::new(instance, &device),
            device,
            queue,
            family,
            memory,
        }
    }

    /// The index of a memory type among `bits` with `flags`.
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        (0..self.memory.memory_type_count)
            .find(|&index| {
                bits & (1 << index) != 0
                    && (self.memory.memory_types[index as usize].property_flags).contains(flags)
            })
            .unwrap()
    }

    /// A buffer of `length` bytes bound to `memory`'s import of `pages`, or, with `pages` `None`,
    /// to new host-visible, coherent memory; its usage `usage`.
    ///
    /// # Safety
    ///
    /// `pages` are `length` bytes aligned to the device's host pointer alignment, which outlive
    /// the buffer and its memory.
    unsafe fn buffer(
        &self,
        pages: Option<*mut u8>,
        length: u64,
        usage: vk::BufferUsageFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let mut info = vk::BufferCreateInfo::default().size(length).usage(usage);
        if pages.is_some() {
            info = info.push_next(&mut external);
        }
        // SAFETY: the device is live.
        let buffer = unsafe { self.device.create_buffer(&info, None) }.unwrap();
        // SAFETY: as above.
        let needs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory = match pages {
            Some(pages) => {
                let mut found = vk::MemoryHostPointerPropertiesEXT::default();
                // SAFETY: the caller's pages.
                unsafe {
                    (self.host.fp().get_memory_host_pointer_properties_ext)(
                        self.device.handle(),
                        handle,
                        pages.cast_const().cast::<std::ffi::c_void>(),
                        &mut found,
                    )
                }
                .result()
                .unwrap();
                let index = self.memory_type(
                    found.memory_type_bits & needs.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                );
                let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                    .handle_type(handle)
                    .host_pointer(pages.cast::<std::ffi::c_void>());
                let allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(length)
                    .memory_type_index(index)
                    .push_next(&mut import);
                // SAFETY: the caller's pages, of a type the device answered for them.
                unsafe { self.device.allocate_memory(&allocate, None) }.unwrap()
            }
            None => {
                let index = self.memory_type(
                    needs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                let allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(index);
                // SAFETY: as above.
                unsafe { self.device.allocate_memory(&allocate, None) }.unwrap()
            }
        };
        // SAFETY: a new buffer and memory of its size.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        (buffer, memory)
    }

    /// Copy `pages`' `length` bytes after a wait for `file` (the descriptor -1 for `None`) and
    /// answer them as this device's copy read them.
    ///
    /// # Safety
    ///
    /// As [`buffer`](Self::buffer)'s.
    unsafe fn copy_after(&self, file: Option<OwnedFd>, pages: *mut u8, length: u64) -> Vec<u8> {
        let device = &self.device;
        // SAFETY: the caller's.
        let (source, imported) =
            unsafe { self.buffer(Some(pages), length, vk::BufferUsageFlags::TRANSFER_SRC) };
        // SAFETY: new memory of its own.
        let (target, memory) =
            unsafe { self.buffer(None, length, vk::BufferUsageFlags::TRANSFER_DST) };
        // SAFETY: the device is live.
        let semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let import = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(file.map_or(-1, IntoRawFd::into_raw_fd));
        // SAFETY: a sync file of a binary semaphore (or -1, one already signaled), imported
        // temporarily as the handle type requires; a successful import owns the descriptor.
        unsafe { self.semaphores.import_semaphore_fd(&import) }.unwrap();
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(self.family);
        // SAFETY: as above.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1);
        // SAFETY: as above.
        let raw = unsafe { device.allocate_command_buffers(&allocate) }.unwrap()[0];
        let region = vk::BufferCopy::default().size(length);
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: a new command buffer, recorded once; the buffers are live.
        unsafe {
            device
                .begin_command_buffer(raw, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_copy_buffer(raw, source, target, &[region]);
            device.cmd_pipeline_barrier(
                raw,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[to_host],
                &[],
                &[],
            );
            device.end_command_buffer(raw).unwrap();
        }
        let waits = [semaphore];
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let buffers = [raw];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers);
        // SAFETY: as above.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        // SAFETY: the semaphore's imported payload has its signal submitted (the runtime's
        // export); the queue is used on this thread alone.
        unsafe { device.queue_submit(self.queue, &[submit], fence) }.unwrap();
        // SAFETY: as above.
        unsafe { device.wait_for_fences(&[fence], true, WAIT.as_nanos() as u64) }.unwrap();
        // SAFETY: host-visible, coherent memory the completed copy wrote, mapped once.
        let bytes = unsafe {
            let mapped = device
                .map_memory(memory, 0, length, vk::MemoryMapFlags::empty())
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), length as usize).to_vec()
        };
        // SAFETY: nothing pending uses them: the fence was signaled.
        unsafe {
            device.unmap_memory(memory);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            device.destroy_semaphore(semaphore, None);
            device.destroy_buffer(source, None);
            device.free_memory(imported, None);
            device.destroy_buffer(target, None);
            device.free_memory(memory, None);
        }
        bytes
    }
}

impl Drop for Second {
    fn drop(&mut self) {
        // SAFETY: every object of the device was destroyed; nothing is pending.
        unsafe { self.device.destroy_device(None) };
    }
}

/// Whether `file` is signaled now (`poll(2)` of a sync file).
fn signaled(file: &OwnedFd) -> bool {
    let mut poll = libc::pollfd {
        fd: file.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live descriptor, polled without waiting.
    let ready = unsafe { libc::poll(&mut poll, 1, 0) };
    ready > 0
}

/// The invalid call's problem.
fn problem(result: Result<impl std::fmt::Debug, Error>) -> Problem {
    match result {
        Err(Error::Invalid(Invalid { problem, .. })) => problem,
        other => panic!("not an invalid call: {other:?}"),
    }
}

/// A command buffer of `queue` with nothing encoded that asked for a sync file.
fn asking(queue: &CommandQueue) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.add_completion_sync_file().unwrap();
    command_buffer
}

#[test]
fn a_command_buffer_s_completion_crosses_to_another_gpu_as_a_sync_file() {
    let scratch = Scratch::new("sync-file");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    let (instance, ours) = (handles.instance, handles.physical_device);
    let queue = device.new_command_queue().unwrap();
    let probe = queue.command_buffer().unwrap();
    match probe.add_completion_sync_file() {
        Ok(()) => {}
        Err(Error::NotImplemented(RuntimeGap::SyncFileExport)) => {
            assert!(
                !sync_fd(
                    instance,
                    ours,
                    vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
                ),
                "{name} exports a binary semaphore as a sync file, and the runtime refused it"
            );
            eprintln!("SYNC-FILE on {name}: refused, the device exports no sync file");
            return;
        }
        Err(error) => panic!("{name}: {error:?}"),
    }
    drop(probe);

    // The refusals: no sync file asked for, asked after the commit, an export before the commit
    // and a second export.
    let plain = queue.command_buffer().unwrap();
    plain.commit().unwrap();
    assert_eq!(
        problem(plain.export_completion_sync_file()),
        Problem::NotExportable
    );
    assert_eq!(
        problem(plain.add_completion_sync_file()),
        Problem::SyncFile(SyncFileProblem::Committed)
    );
    // The plain one, taken with no submission, is counted before the counts are read below.
    plain.wait_until_completed_for(WAIT).unwrap();
    let empty = asking(&queue);
    assert_eq!(
        problem(empty.export_completion_sync_file()),
        Problem::SyncFile(SyncFileProblem::NotCommitted)
    );
    // One with no GPU work is submitted for its signal, and its sync file signals (or is
    // answered signaled, its completion first).
    let before = device.work_counts();
    empty.commit().unwrap();
    let file = empty.export_completion_sync_file().unwrap();
    assert_eq!(
        problem(empty.export_completion_sync_file()),
        Problem::SyncFile(SyncFileProblem::Exported)
    );
    empty.wait_until_completed_for(WAIT).unwrap();
    if let Some(file) = &file {
        assert!(
            signaled(file),
            "{name}: the empty command buffer's sync file"
        );
    }
    let after = device.work_counts();
    assert_eq!(
        after.command_buffers_unsubmitted, before.command_buffers_unsubmitted,
        "{name}: the empty command buffer was taken with no submission"
    );
    // An export after the completion answers a sync file already signaled.
    let finished = asking(&queue);
    finished.commit().unwrap();
    finished.wait_until_completed_for(WAIT).unwrap();
    assert!(finished.export_completion_sync_file().unwrap().is_none());

    // The fill and the second device's copy after its wait.
    let Some(physical) = importer(instance, ours) else {
        eprintln!("SYNC-FILE on {name}: no device of the instance imports a sync file and pages");
        return;
    };
    let second = Second::new(instance, physical);
    // SAFETY: the physical device is the instance's.
    let theirs = unsafe { instance.get_physical_device_properties(physical) };
    let theirs = theirs.device_name_as_c_str().unwrap().to_string_lossy();
    let pages = Pages::new();
    // SAFETY: the pages outlive the buffer, which only this command buffer uses.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(pages.0),
            LENGTH,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("SYNC-FILE on {name}: the device makes no buffer over host pages: {gap:?}");
            return;
        }
        Err(error) => panic!("{name}: new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let backing = buffer.host_backing();
    let filling = asking(&queue);
    let mut blit = filling.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..LENGTH, BYTE).unwrap();
    blit.end_encoding().unwrap();
    let before = device.work_counts();
    filling.commit_and_submit().unwrap();
    let file = filling.export_completion_sync_file().unwrap();
    let handed = match &file {
        Some(file) if signaled(file) => "a descriptor, signaled",
        Some(_) => "a descriptor, not signaled yet: the GPU still ran",
        None => "none (-1, signaled)",
    };
    let status = filling.status();
    // SAFETY: the pages outlive the second device's import, which the copy frees.
    let bytes = unsafe { second.copy_after(file, pages.0.as_ptr(), LENGTH) };
    let unfilled = bytes.iter().filter(|&&byte| byte != BYTE).count();
    eprintln!(
        "SYNC-FILE on {name}: filled {LENGTH} bytes of host pages ({backing:?}); {theirs}'s copy \
         after its wait read {unfilled} bytes unfilled; handed out: {handed} (status then \
         {status:?})"
    );
    assert_eq!(
        unfilled, 0,
        "{name} -> {theirs}: bytes the copy read unfilled"
    );
    filling.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(filling.error(), None);
    // Still running after the export, it was running as the export looked: the export took the
    // submitted signal's payload, or waited for the host copies.
    let after = device.work_counts();
    if !matches!(
        status,
        CommandBufferStatus::Completed | CommandBufferStatus::Error
    ) {
        assert_eq!(
            after.sync_files_exported + after.sync_files_after_host_copies,
            before.sync_files_exported + before.sync_files_after_host_copies + 1,
            "{name}: the fill's export"
        );
    }
    drop(buffer);
}
