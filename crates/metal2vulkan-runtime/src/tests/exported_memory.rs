//! An exportable buffer's memory (`exported_memory.rs`) imported into a second `VkDevice` of the
//! same physical device, as another API imports it: the second device reads what the runtime's
//! host and GPU wrote, and the runtime's GPU reads what the second device wrote. A device the
//! runtime refuses the buffer on (`RuntimeGap::ExportableMemory`) must lack the extension or the
//! export. It prints what the device exported and what a making cost beside a plain buffer's
//! (`EXPORTED-MEMORY:` lines).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::{Error, Problem, RuntimeGap};
use crate::{Buffer, Device, ResourceOptions};
use ash::vk;
use std::os::fd::IntoRawFd;
use std::time::{Duration, Instant};

/// The buffer's length: a 3840 x 2160 NV12 picture's bytes, the consumer's decoded frame.
const LENGTH: u64 = 3840 * 2160 * 3 / 2;

/// A byte unique to `index` below 2^24.
fn pattern(index: u64, seed: u32) -> u8 {
    let mut x = (index as u32).wrapping_mul(2654435761) ^ seed;
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x ^ (x >> 24)) as u8
}

/// The `Shared` buffer's bytes.
fn bytes(buffer: &Buffer) -> &[u8] {
    let at = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of its length, whose commands all completed.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), buffer.length() as usize) }
}

/// Whether the physical device under `device` lists `VK_KHR_external_memory_fd` and exports a
/// buffer of `usage`'s memory as an opaque file descriptor.
fn exports(device: &Device, usage: vk::BufferUsageFlags) -> bool {
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let listed = unsafe {
        handles
            .instance
            .enumerate_device_extension_properties(handles.physical_device)
    }
    .unwrap();
    let named = listed.iter().any(|extension| {
        extension.extension_name_as_c_str() == Ok(ash::khr::external_memory_fd::NAME)
    });
    let query = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut properties = vk::ExternalBufferProperties::default();
    // SAFETY: as above, at Vulkan 1.2.
    unsafe {
        handles
            .instance
            .get_physical_device_external_buffer_properties(
                handles.physical_device,
                &query,
                &mut properties,
            )
    };
    let features = properties
        .external_memory_properties
        .external_memory_features;
    named && features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
}

/// The median of `times`.
fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

#[test]
fn an_exportable_buffer_s_memory_crosses_to_a_second_device() {
    let scratch = Scratch::new("exported-memory");
    let device = device(&scratch.0);
    let device_address = device.shared().device_address;
    let usage = crate::buffer::usage(device_address);
    let name = device.name().to_owned();
    let buffer = match device.new_exportable_buffer(LENGTH, ResourceOptions::shared()) {
        Ok(buffer) => buffer,
        Err(Error::NotImplemented(RuntimeGap::ExportableMemory)) => {
            assert!(
                !exports(&device, usage),
                "{name} exports a buffer's memory as an opaque fd, and the runtime refused it"
            );
            eprintln!("EXPORTED-MEMORY on {name}: refused, the device exports no opaque fd");
            return;
        }
        Err(error) => panic!("{name}: {error:?}"),
    };
    // A plain buffer's memory is not the caller's to export.
    let plain = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    assert!(matches!(
        plain.export_memory_fd(),
        Err(Error::Invalid(crate::error::Invalid {
            problem: Problem::NotExportable,
            ..
        }))
    ));
    // Made zero, as any buffer.
    assert!(bytes(&buffer).iter().all(|&byte| byte == 0));
    // The host writes the first half, the GPU fills the second's first 4096 bytes.
    let half = LENGTH / 2;
    let host: Vec<u8> = (0..half).map(|index| pattern(index, 7)).collect();
    // SAFETY: a `Shared` buffer no command uses now.
    unsafe {
        std::ptr::copy_nonoverlapping(
            host.as_ptr(),
            buffer.contents().unwrap().as_ptr(),
            half as usize,
        )
    };
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, half..half + 4096, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);

    let exported = buffer.export_memory_fd().unwrap();
    eprintln!(
        "EXPORTED-MEMORY on {name}: {} bytes at {} of an allocation of {}, dedicated {}, type {}",
        exported.length,
        exported.offset,
        exported.allocation_size,
        exported.dedicated,
        exported.memory_type_index
    );
    assert_eq!(exported.length, LENGTH);
    assert!(exported.offset + exported.length <= exported.allocation_size);
    // Each call a new descriptor.
    let again = buffer.export_memory_fd().unwrap();
    assert_ne!(
        std::os::fd::AsRawFd::as_raw_fd(&again.fd),
        std::os::fd::AsRawFd::as_raw_fd(&exported.fd)
    );
    drop(again);

    // SAFETY: the handles are used while the device lives; the second device is the test's own
    // and destroyed below.
    let handles = unsafe { device.raw() };
    let (instance, physical) = (handles.instance, handles.physical_device);
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(handles.queue_family)
        .queue_priorities(&priorities)];
    let extensions = [ash::khr::external_memory_fd::NAME.as_ptr()];
    let mut features12 =
        vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(device_address);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .push_next(&mut features12);
    // SAFETY: the physical device is the instance's; the extension is listed (the export runs).
    let second = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    // The buffer the import binds, made as the runtime made the exported one: a dedicated import
    // names a buffer made with the same parameters.
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let buffer_info = device.shared().sharing().buffer(
        vk::BufferCreateInfo::default()
            .size(LENGTH.next_multiple_of(16))
            .usage(usage)
            .push_next(&mut external),
    );
    // SAFETY: the second device is live and `buffer_info` valid for it.
    let imported_buffer = unsafe { second.create_buffer(&buffer_info, None) }.unwrap();
    let allocation_size = exported.allocation_size;
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
        .fd(exported.fd.into_raw_fd());
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(imported_buffer);
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(allocation_size)
        .memory_type_index(exported.memory_type_index)
        .push_next(&mut import);
    if exported.dedicated {
        allocate = allocate.push_next(&mut dedicated);
    }
    if device_address {
        allocate = allocate.push_next(&mut flags);
    }
    // SAFETY: the descriptor is the export's of the same physical device, its size and type the
    // allocation's; a successful import owns it.
    let memory = unsafe { second.allocate_memory(&allocate, None) }.unwrap();
    // SAFETY: the buffer and the memory are the second device's, the memory as long as the buffer
    // requires (the allocation the exported buffer is bound to at its offset).
    unsafe { second.bind_buffer_memory(imported_buffer, memory, exported.offset) }.unwrap();
    // SAFETY: the memory is host-visible (a `Shared` buffer's type) and mapped whole once.
    let mapped =
        unsafe { second.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .unwrap()
            .cast::<u8>();
    // SAFETY: the mapping covers the allocation; the runtime's commands completed.
    let seen = unsafe {
        std::slice::from_raw_parts(mapped.add(exported.offset as usize), LENGTH as usize)
    };
    assert!(seen[..half as usize] == host[..], "{name}: the host's half");
    assert!(
        seen[half as usize..(half + 4096) as usize]
            .iter()
            .all(|&byte| byte == 0x5a),
        "{name}: the GPU's fill"
    );
    // The second device writes the last 4096 bytes; the runtime's GPU copies them out.
    let theirs: Vec<u8> = (0..4096).map(|index| pattern(index, 11)).collect();
    // SAFETY: as above; nothing reads them now. The memory is host-coherent (a `Shared`
    // buffer's type), so the runtime's next submission sees them.
    unsafe {
        std::ptr::copy_nonoverlapping(
            theirs.as_ptr(),
            mapped.add((exported.offset + LENGTH - 4096) as usize),
            4096,
        )
    };
    let out = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&buffer, LENGTH - 4096, &out, 0, 4096)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert!(
        bytes(&out) == &theirs[..],
        "{name}: the second device's bytes"
    );
    // SAFETY: the second device's objects, unused now; the buffer outlives its memory's import.
    unsafe {
        second.unmap_memory(memory);
        second.destroy_buffer(imported_buffer, None);
        second.free_memory(memory, None);
        second.destroy_device(None);
    }
    drop(buffer);

    // What a making costs beside a plain buffer's, of the picture's length (medians of 8).
    let time = |make: &dyn Fn() -> Buffer| {
        let times = (0..8)
            .map(|_| {
                let start = Instant::now();
                let made = make();
                let took = start.elapsed();
                drop(made);
                took
            })
            .collect();
        median(times)
    };
    for options in [ResourceOptions::shared(), ResourceOptions::private()] {
        let exportable = time(&|| device.new_exportable_buffer(LENGTH, options).unwrap());
        let plain = time(&|| device.new_buffer(LENGTH, options).unwrap());
        eprintln!(
            "EXPORTED-MEMORY-COST on {name}: a {options:?} buffer of {LENGTH} bytes made in \
             {exportable:?} exportable, {plain:?} plain (medians of 8)"
        );
    }
}
