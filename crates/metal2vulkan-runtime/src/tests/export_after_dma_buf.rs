//! A `Private` exportable buffer made after a dedicated dma-buf import was made and freed: which
//! memory type it takes, against one made on a device that imported no dma-buf.
//!
//! The import is a no-copy buffer over the mapping of one udmabuf of 1,024 scattered pages of the
//! test's own `memfd` (`UDMABUF_CREATE_LIST`), its source answering that dma-buf: pages the kernel
//! cannot pin (`VM_PFNMAP`), whose pointer the runtime keeps from NVIDIA's driver
//! (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`) and imports as the dma-buf, which
//! NVIDIA's driver takes dedicated to its buffer. The buffer is freed at once. Then a `Private`
//! buffer of 12,441,600 bytes, exportable as an opaque file descriptor and as a dma-buf, is made
//! and exported: each one's memory type, its flags, whether it is dedicated, and whether the
//! device counted an overflow past device-local memory (`WorkCounts::device_memory_overflows`).
//! Each heap's `VK_EXT_memory_budget` budget and usage is read before the import, after its free
//! and after the exports (the runtime reads neither: a placement overflows only where the
//! driver's `vkAllocateMemory` answers out of memory for every device-local type).
//!
//! It prints its findings (`EXPORT-AFTER-DMA-BUF`) and asserts that each export lies in a
//! device-local memory type with the control's flags, no overflow counted. Where `/dev/udmabuf`
//! cannot be opened, or the device exports no such memory, it says so and passes.
//!
//! The driver's behaviour behind it, as evidence (`EXPORT-AFTER-DMA-BUF-RAW`, printed, not
//! asserted: a driver may change it): NVIDIA's 620.70.06 on an RTX 3090, once it refused an
//! import of the udmabuf into a system-memory type the host maps (its type 4,
//! `ERROR_OUT_OF_DEVICE_MEMORY`), answers `ERROR_OUT_OF_DEVICE_MEMORY` to every exportable
//! allocation of its device-local type, for the device's life, where its successful import into
//! type 0 leaves them made. So the runtime's imports try the system-memory types the host maps
//! last (`memory::dma_buf_candidates`), and `DMA-BUF-IMPORT-TYPES` prints where that order puts
//! each class's import on the device under test, against the class's order alone.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_residency::{device, udmabuf_list, udmabuf_openable, Scratch};
use crate::{
    BufferContents, BufferDescriptor, Device, DmaBufChunk, DmaBufRequest, Error, HostMemory,
    MemoryExport, ResourceOptions,
};
use ash::vk;
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;

const TEST: &str = "EXPORT-AFTER-DMA-BUF";
/// The consumer's exportable buffer's bytes.
const LENGTH: u64 = 12_441_600;
/// The pages of the probe's one udmabuf, and of the file they are drawn from.
const PAGES: u64 = 1024;
const FILE_PAGES: u64 = 4 * PAGES;

/// Each heap's size, `VK_EXT_memory_budget` budget and usage, in MiB; empty where the device does
/// not list the extension.
fn budgets(device: &Device) -> Vec<(u64, u64, u64)> {
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let extensions = unsafe {
        handles
            .instance
            .enumerate_device_extension_properties(handles.physical_device)
    }
    .unwrap_or_default();
    let listed = extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
    if !listed {
        return Vec::new();
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: the physical device is the instance's, at Vulkan 1.1 or later, and lists the
    // extension whose structure is chained.
    unsafe {
        handles
            .instance
            .get_physical_device_memory_properties2(handles.physical_device, &mut properties)
    };
    let count = properties.memory_properties.memory_heap_count as usize;
    let heaps = properties.memory_properties.memory_heaps;
    (0..count.min(vk::MAX_MEMORY_HEAPS))
        .map(|heap| {
            (
                heaps[heap].size >> 20,
                budget.heap_budget[heap] >> 20,
                budget.heap_usage[heap] >> 20,
            )
        })
        .collect()
}

/// The property flags of `device`'s memory type `index`, as bits (`VkMemoryPropertyFlagBits`).
fn type_flags(device: &Device, index: u32) -> u32 {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let memory = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    memory.memory_types[index as usize].property_flags.as_raw()
}

/// What a `Private` buffer of [`LENGTH`] bytes exportable as `export` lies in: its memory type and
/// flags, whether dedicated, and the overflows the device counted for it; `None` where the device
/// exports no such memory.
fn exported(device: &Device, export: MemoryExport) -> Option<(u32, u32, bool, u64)> {
    let overflows = device.work_counts().device_memory_overflows;
    let descriptor =
        BufferDescriptor::new(LENGTH, ResourceOptions::private(), BufferContents::Zero)
            .with_exported_as(export);
    let buffer = match device.new_buffer_with_descriptor(descriptor) {
        Ok(buffer) => buffer,
        Err(Error::NotImplemented(gap)) => {
            eprintln!("{TEST} on {}: {export:?}: {gap:?}", device.name());
            return None;
        }
        Err(error) => panic!("{export:?}: {error:?}"),
    };
    let memory = match export {
        MemoryExport::OpaqueFd => buffer.export_memory_fd(),
        MemoryExport::DmaBuf => buffer.export_memory_dma_buf(),
    }
    .unwrap();
    let flags = type_flags(device, memory.memory_type_index);
    let overflowed = device.work_counts().device_memory_overflows - overflows;
    let found = (
        memory.memory_type_index,
        flags,
        memory.dedicated,
        overflowed,
    );
    drop((memory, buffer));
    Some(found)
}

/// The probe: a no-copy buffer over the mapping of one udmabuf of [`PAGES`] scattered pages of a
/// `memfd` of [`FILE_PAGES`], its source answering that dma-buf, made and freed at once. What the
/// device counted of dma-buf imports, dedicated and live, after its making.
fn probe(device: &Device, page: u64) -> (u64, u64) {
    // SAFETY: a name and flags the call takes.
    let raw = unsafe {
        libc::memfd_create(
            c"m2v-export-probe".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    assert!(raw >= 0);
    // SAFETY: the descriptor was just made, and is the test's alone.
    let memfd = unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(raw) };
    // SAFETY: the test's descriptor.
    assert_eq!(
        unsafe { libc::ftruncate(memfd.as_raw_fd(), (FILE_PAGES * page) as libc::off_t) },
        0
    );
    // udmabuf takes pages only of a file that cannot shrink.
    // SAFETY: as above.
    assert_eq!(
        unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
        0
    );
    // Every fourth page, in an order of its own: scattered, none twice.
    let runs: Vec<(u64, u64)> = (0..PAGES)
        .map(|index| ((index * 2_654_435_761 % PAGES) * 4 * page, page))
        .collect();
    let dma_buf = Arc::new(udmabuf_list(&memfd, &runs).unwrap());
    let length = PAGES * page;
    // SAFETY: a shared mapping of the dma-buf the test made, unmapped below once the buffer went.
    let view = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            dma_buf.as_raw_fd(),
            0,
        )
    };
    assert_ne!(view, libc::MAP_FAILED);
    let base = view as u64;
    let source_buf = Arc::clone(&dma_buf);
    let source = Arc::new(move |request: DmaBufRequest| {
        let start = request.start.as_ptr() as u64;
        if start != base || request.length != length {
            return Err(std::io::Error::other("pages other than the view's"));
        }
        Ok(vec![DmaBufChunk {
            fd: source_buf.try_clone()?,
            length,
        }])
    });
    let before = device.work_counts();
    // SAFETY: the view stays mapped until the buffer went (dropped below, before the unmap), and
    // the source's dma-buf holds the view's pages, in order.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(NonNull::new(view.cast()).unwrap(), source),
            length,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let made = device.work_counts();
    drop(buffer);
    // SAFETY: the mapping made above, which no buffer lies over any more.
    unsafe { libc::munmap(view, length as usize) };
    (
        made.dma_buf_buffer_imports_dedicated - before.dma_buf_buffer_imports_dedicated,
        made.dma_buf_memories_live - before.dma_buf_memories_live,
    )
}

/// The device's memory type of `DEVICE_LOCAL` alone, where it has one.
fn device_local_type(device: &Device) -> Option<u32> {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let memory = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    (0..memory.memory_type_count).find(|&at| {
        memory.memory_types[at as usize].property_flags == vk::MemoryPropertyFlags::DEVICE_LOCAL
    })
}

/// The driver's own answer to an allocation of `size` bytes of memory type `type_index` with each
/// chain the runtime's blocks pass: plain, with the device-address flag, exportable as an opaque
/// file descriptor or a dma-buf, and both. Each allocation made is freed at once. The answers as
/// `VkResult` codes (0 where it was made).
fn raw_answers(device: &Device, type_index: u32, size: u64) -> Vec<(&'static str, i32)> {
    use vk::ExternalMemoryHandleTypeFlags as H;
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    let chains: [(&str, bool, Option<H>); 5] = [
        ("plain", false, None),
        ("address", true, None),
        ("opaque-fd", false, Some(H::OPAQUE_FD)),
        ("dma-buf", false, Some(H::DMA_BUF_EXT)),
        ("address+opaque-fd", true, Some(H::OPAQUE_FD)),
    ];
    chains
        .into_iter()
        .map(|(name, address, export)| {
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut exportable =
                vk::ExportMemoryAllocateInfo::default().handle_types(export.unwrap_or_default());
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(type_index);
            if address {
                info = info.push_next(&mut flags);
            }
            if export.is_some() {
                info = info.push_next(&mut exportable);
            }
            // SAFETY: a valid allocation of the device's type, freed at once where made.
            let answer = unsafe { handles.device.allocate_memory(&info, None) };
            match answer {
                Ok(memory) => {
                    // SAFETY: just made, used by nothing.
                    unsafe { handles.device.free_memory(memory, None) };
                    (name, 0)
                }
                Err(result) => (name, result.as_raw()),
            }
        })
        .collect()
}

/// The test's own import of `dma_buf`'s `size` bytes as memory type `type_index`, dedicated to
/// nothing, freed at once where made: the driver's answer, as a `VkResult` code. The driver is
/// handed a duplicate, closed here where the driver left it open after a failure.
fn raw_import(device: &Device, dma_buf: &OwnedFd, size: u64, type_index: u32) -> i32 {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    // SAFETY: a descriptor the test holds.
    let handed = unsafe { libc::fcntl(dma_buf.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
    assert!(handed >= 0);
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(handed);
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(type_index)
        .push_next(&mut import)
        .push_next(&mut flags);
    // SAFETY: a valid import of a dma-buf the test made; the memory is freed at once.
    match unsafe { handles.device.allocate_memory(&info, None) } {
        Ok(memory) => {
            // SAFETY: just made, used by nothing.
            unsafe { handles.device.free_memory(memory, None) };
            0
        }
        Err(result) => {
            // SAFETY: the test's duplicate, which no other thread's descriptor takes meanwhile
            // (the test runs alone in its process).
            if unsafe { libc::fcntl(handed, libc::F_GETFD) } >= 0 {
                // SAFETY: as above.
                unsafe { libc::close(handed) };
            }
            result.as_raw()
        }
    }
}

/// The memory types `dma_buf` may be imported as, as bits (`vkGetMemoryFdPropertiesKHR`).
fn dma_buf_types(device: &Device, dma_buf: &OwnedFd) -> Result<u32, vk::Result> {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    let loader = ash::khr::external_memory_fd::Device::new(handles.instance, handles.device);
    let mut properties = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: a dma-buf the test holds, read and not taken.
    unsafe {
        loader.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            dma_buf.as_raw_fd(),
            &mut properties,
        )
    }
    .map(|()| properties.memory_type_bits)
}

/// A udmabuf of [`PAGES`] scattered pages of a sealed `memfd` of [`FILE_PAGES`], as the probe's.
fn scattered_udmabuf(page: u64) -> OwnedFd {
    // SAFETY: a name and flags the call takes.
    let raw = unsafe {
        libc::memfd_create(
            c"m2v-export-raw".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    assert!(raw >= 0);
    // SAFETY: the descriptor was just made, and is the test's alone.
    let memfd = unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(raw) };
    // SAFETY: the test's descriptor.
    assert_eq!(
        unsafe { libc::ftruncate(memfd.as_raw_fd(), (FILE_PAGES * page) as libc::off_t) },
        0
    );
    // SAFETY: as above.
    assert_eq!(
        unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) },
        0
    );
    let runs: Vec<(u64, u64)> = (0..PAGES)
        .map(|index| ((index * 2_654_435_761 % PAGES) * 4 * page, page))
        .collect();
    udmabuf_list(&memfd, &runs).unwrap()
}

/// What a device answers after each kind of dma-buf import, each on a device of its own: none
/// (the control); the test's own import of a scattered udmabuf as each type the driver lists for
/// it; the runtime's no-copy buffer over the udmabuf's mapping (the probe), as it imports it, and
/// with its import dedicated to nothing refused (a test's stand-in), so it is dedicated to the
/// buffer. After each: the driver's answers to device-local allocations of a pooled block's bytes
/// and of the exportable buffer's, of each chain, and where the runtime places the exportable
/// buffers. It prints them (`EXPORT-AFTER-DMA-BUF-RAW`) and asserts nothing: it is the evidence
/// of a driver's behaviour, which a driver may change. Measured on NVIDIA's 620.70.06 on an RTX
/// 3090: after a refused import as type 4 (host-visible) every exportable chain answered
/// `ERROR_OUT_OF_DEVICE_MEMORY` (-2) at both sizes, the plain and device-address ones made; after
/// a successful import as type 0, every chain was made.
#[test]
fn the_driver_s_answers_after_each_kind_of_dma_buf_import() {
    const RAW: &str = "EXPORT-AFTER-DMA-BUF-RAW";
    if !udmabuf_openable(RAW) {
        return;
    }
    let scratch = Scratch::new();
    let after = |name: &str, device: &Device| {
        let Some(local) = device_local_type(device) else {
            eprintln!("{RAW} {name}: no type of DEVICE_LOCAL alone");
            return;
        };
        let block = crate::device::lock(&device.shared().memory).pooled_block_bytes_of(local);
        let at_block = raw_answers(device, local, block);
        let at_length = raw_answers(device, local, LENGTH);
        let fd = exported(device, MemoryExport::OpaqueFd);
        let dma_buf = exported(device, MemoryExport::DmaBuf);
        eprintln!(
            "{RAW} {name}: type {local}: block {block} bytes {at_block:?}; {LENGTH} bytes \
             {at_length:?}; exports opaque fd {fd:?}, dma-buf {dma_buf:?}"
        );
    };
    let control = device(&scratch.0);
    let page = control
        .shared()
        .host_import
        .as_ref()
        .map_or(4096, |import| import.page);
    after("control", &control);
    drop(control);

    // The test's own imports, one type at a time, each on a device of its own.
    let probe_device = device(&scratch.0);
    let udmabuf = scattered_udmabuf(page);
    let types = dma_buf_types(&probe_device, &udmabuf).unwrap();
    drop(probe_device);
    eprintln!("{RAW}: the udmabuf's types {types:#b}");
    for type_index in (0..32).filter(|at| types & (1 << at) != 0) {
        let device = device(&scratch.0);
        let answer = raw_import(&device, &udmabuf, PAGES * page, type_index);
        after(
            &format!("raw import as type {type_index}: {answer}"),
            &device,
        );
    }

    // The runtime's import, as it makes it.
    let device_ = device(&scratch.0);
    let (dedicated, live) = probe(&device_, page);
    after(
        &format!("runtime import (dedicated {dedicated}, live {live})"),
        &device_,
    );
    drop(device_);

    // The runtime's import with its form dedicated to nothing refused.
    let device_ = device(&scratch.0);
    crate::device::lock(&device_.shared().memory).undedicated_import_refusal =
        Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    let (dedicated, live) = probe(&device_, page);
    after(
        &format!("runtime import, undedicated refused (dedicated {dedicated}, live {live})"),
        &device_,
    );
}

/// After the runtime's import of a dma-buf (the probe's), a `Private` buffer exportable as an
/// opaque file descriptor and as a dma-buf lies in a device-local type, as on a device that
/// imported none, with no overflow counted.
#[test]
fn a_private_exportable_buffer_after_a_dma_buf_import_lies_in_device_local_memory() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    // The control: a device that imported no dma-buf.
    let control = device(&scratch.0);
    let name = control.name().to_owned();
    let page = control
        .shared()
        .host_import
        .as_ref()
        .map_or(4096, |import| import.page);
    let control_budgets = budgets(&control);
    let control_fd = exported(&control, MemoryExport::OpaqueFd);
    let control_dma_buf = exported(&control, MemoryExport::DmaBuf);
    eprintln!(
        "{TEST} on {name}: control: heaps (MiB: size, budget, usage) {control_budgets:?}; opaque \
         fd {control_fd:?}; dma-buf {control_dma_buf:?}"
    );
    drop(control);
    let device = device(&scratch.0);
    let before = budgets(&device);
    let (dedicated, live) = probe(&device, page);
    let after_free = budgets(&device);
    let fd = exported(&device, MemoryExport::OpaqueFd);
    let dma_buf = exported(&device, MemoryExport::DmaBuf);
    let after = budgets(&device);
    eprintln!(
        "{TEST} on {name}: the probe's import: dedicated {dedicated}, live at its making \
         {live}; heaps before {before:?}, after its free {after_free:?}, after the exports \
         {after:?}; opaque fd {fd:?}; dma-buf {dma_buf:?}"
    );
    let local = vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw();
    for (export, found, control) in [
        ("opaque fd", fd, control_fd),
        ("dma-buf", dma_buf, control_dma_buf),
    ] {
        assert_eq!(found.is_some(), control.is_some(), "{export}: exported");
        let (Some(found), Some(control)) = (found, control) else {
            continue;
        };
        assert_eq!(found.1, control.1, "{export}: the control's type flags");
        assert_ne!(found.1 & local, 0, "{export}: a device-local type");
        assert_eq!(found.3, 0, "{export}: no overflow");
    }
}

/// A dma-buf the device exports of `bytes` bytes of its device-local memory (a type of
/// `DEVICE_LOCAL` alone, else the first with it), with the memory, which the caller frees; `None`
/// where the driver refuses either.
fn device_local_dma_buf(device: &Device, bytes: u64) -> Option<(OwnedFd, vk::DeviceMemory)> {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let memory = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    let local = device_local_type(device).or_else(|| {
        (0..memory.memory_type_count).find(|&at| {
            (memory.memory_types[at as usize].property_flags)
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
    })?;
    let mut exportable = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(bytes)
        .memory_type_index(local)
        .push_next(&mut exportable)
        .push_next(&mut flags);
    // SAFETY: a valid allocation of the device's type, freed by the caller.
    let made = unsafe { handles.device.allocate_memory(&info, None) }.ok()?;
    let loader = ash::khr::external_memory_fd::Device::new(handles.instance, handles.device);
    let info = vk::MemoryGetFdInfoKHR::default()
        .memory(made)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    // SAFETY: memory just made, exportable as a dma-buf.
    match unsafe { loader.get_memory_fd(&info) } {
        Ok(fd) => {
            // SAFETY: the driver hands the descriptor over.
            let fd = unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) };
            Some((fd, made))
        }
        Err(_) => {
            // SAFETY: just made, used by nothing.
            unsafe { handles.device.free_memory(made, None) };
            None
        }
    }
}

/// The memory types a buffer of `bytes` bytes, importable from a dma-buf, takes, as bits.
fn dma_buf_buffer_types(device: &Device, bytes: u64) -> u32 {
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::BufferCreateInfo::default()
        .size(bytes)
        .usage(
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .push_next(&mut external);
    // SAFETY: a valid buffer, destroyed at once.
    let buffer = unsafe { handles.device.create_buffer(&info, None) }.unwrap();
    // SAFETY: the buffer just made.
    let bits = unsafe { handles.device.get_buffer_memory_requirements(buffer) }.memory_type_bits;
    // SAFETY: just made, bound to nothing.
    unsafe { handles.device.destroy_buffer(buffer, None) };
    bits
}

/// Where the runtime's order of a dma-buf import's types (the system-memory types the host maps
/// last, `memory::dma_buf_candidates`) puts a buffer's import of each class, against the class's order
/// alone (`memory::dma_buf_class_order`, the order before): for a scattered udmabuf and for a
/// dma-buf of the device's own device-local memory, the types the driver lists and a buffer
/// takes, both orders, the type the runtime's import took (`memory::import_dma_buf`, dedicated
/// to nothing) and the first type of the class's order the driver imports the dma-buf as (the
/// test's own imports, made after the runtime's, so that a refusal among them reaches none of
/// the runtime's), each with its flags. It prints them (`DMA-BUF-IMPORT-TYPES`), and asserts that
/// where both orders are the same the runtime's import took the type the class's order took.
#[test]
fn the_type_each_class_s_dma_buf_import_takes_against_the_class_s_order() {
    const TYPES: &str = "DMA-BUF-IMPORT-TYPES";
    use crate::memory::{
        dma_buf_candidates, dma_buf_class_order, import_dma_buf, DmaBufFor, MemoryClass,
    };
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let page = device
        .shared()
        .host_import
        .as_ref()
        .map_or(4096, |import| import.page);
    let bytes = PAGES * page;
    // SAFETY: as in `budgets`.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let properties = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    let unified = device.properties().has_unified_memory;
    let flags = |index: u32| format!("{:#x}", types[index as usize].property_flags.as_raw());
    let mut sources = Vec::new();
    // The device's own dma-buf first: no import was refused yet when it is made.
    let own = device_local_dma_buf(&device, bytes);
    match &own {
        Some((fd, _)) => sources.push(("device-local", fd.try_clone().unwrap())),
        None => eprintln!(
            "{TYPES} on {}: no dma-buf of device-local memory",
            device.name()
        ),
    }
    match udmabuf_openable(TYPES) {
        true => sources.push(("udmabuf", scattered_udmabuf(page))),
        false => eprintln!("{TYPES} on {}: no udmabuf", device.name()),
    }
    let buffer_types = dma_buf_buffer_types(&device, bytes);
    let classes = [
        MemoryClass::Shared,
        MemoryClass::WriteCombined,
        MemoryClass::Private,
        MemoryClass::Managed,
    ];
    // Per source and class: the allowed types, both orders and the runtime's import's type.
    let mut found = Vec::new();
    for (source, fd) in &sources {
        let listed = match dma_buf_types(&device, fd) {
            Ok(listed) => listed,
            Err(error) => {
                eprintln!(
                    "{TYPES} on {}: {source}: types refused, {error:?}",
                    device.name()
                );
                continue;
            }
        };
        let allowed = listed & buffer_types;
        for class in classes {
            let before = dma_buf_class_order(types, allowed, class, unified);
            let after = dma_buf_candidates(types, allowed, class, unified);
            let handed = fd.try_clone().unwrap();
            let imported = import_dma_buf(
                &device.shared().memory,
                device.shared().vk(),
                handed,
                bytes,
                (allowed, class),
                DmaBufFor::Shared,
            );
            let took = imported.map(|(memory, type_index, _)| {
                // SAFETY: the import just made, used by nothing.
                unsafe { handles.device.free_memory(memory, None) };
                type_index
            });
            found.push((*source, class, allowed, before, after, took));
        }
    }
    // The test's own imports in the class's order, after every one of the runtime's.
    for (source, class, allowed, before, after, took) in found {
        let fd = &sources.iter().find(|(name, _)| *name == source).unwrap().1;
        let took_before = (before.iter())
            .copied()
            .find(|&index| raw_import(&device, fd, bytes, index) == 0);
        eprintln!(
            "{TYPES} on {}: {source}, {class:?}: allowed {allowed:#b}; class order {before:?}, \
             runtime order {after:?}; the runtime's import took {:?}, the class's order takes \
             {:?}",
            device.name(),
            took.map(|index| (index, flags(index))),
            took_before.map(|index| (index, flags(index))),
        );
        if before == after {
            assert_eq!(took.ok(), took_before, "{source}, {class:?}");
        }
    }
    drop(sources);
    if let Some((fd, memory)) = own {
        drop(fd);
        // SAFETY: the test's memory, which no import holds any more.
        unsafe { handles.device.free_memory(memory, None) };
    }
}
