//! An exportable buffer's memory (`exported_memory.rs`) imported into a second `VkDevice` of the
//! same physical device, as another API imports it: the second device reads what the runtime's
//! host and GPU wrote, and the runtime's GPU reads what the second device wrote. A texture over a
//! `Private` exportable buffer whose rows no image holds reads each write of the second device
//! that `Buffer::external_write_done` announced, its shadow's standing let go by each. A device the
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
use metal2vulkan::metal::{PixelFormat, TextureUsage};
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

/// A second `VkDevice` of the runtime's physical device that writes an exportable buffer's memory
/// through an import of its opaque descriptor, as a video decoder writes its pictures: each write
/// a fill on its own queue, waited for, releasing the memory to `VK_QUEUE_FAMILY_EXTERNAL`.
struct Writer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pool: vk::CommandPool,
    queue: vk::Queue,
    family: u32,
    length: u64,
}

impl Writer {
    /// The writer of `exported`'s `length` bytes, the memory of a buffer of `usage` of `device`.
    fn import(
        device: &Device,
        exported: crate::ExportedMemory,
        usage: vk::BufferUsageFlags,
    ) -> Writer {
        let device_address = device.shared().device_address;
        let length = exported.length;
        // SAFETY: the handles are used while the device lives; the second device is the test's own
        // and destroyed as the writer drops.
        let handles = unsafe { device.raw() };
        let (instance, physical, family) = (
            handles.instance,
            handles.physical_device,
            handles.queue_family,
        );
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = [ash::khr::external_memory_fd::NAME.as_ptr()];
        let mut features12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(device_address);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features12);
        // SAFETY: the physical device is the instance's; the extension is listed (the export ran).
        let second = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = device.shared().sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(length.next_multiple_of(16))
                .usage(usage)
                .push_next(&mut external),
        );
        // SAFETY: the second device is live and `buffer_info` valid for it.
        let buffer = unsafe { second.create_buffer(&buffer_info, None) }.unwrap();
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(exported.fd.into_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(exported.allocation_size)
            .memory_type_index(exported.memory_type_index)
            .push_next(&mut import);
        if exported.dedicated {
            allocate = allocate.push_next(&mut dedicated);
        }
        if device_address {
            allocate = allocate.push_next(&mut flags);
        }
        // SAFETY: the descriptor is the export's of the same physical device, its size and type
        // the allocation's; a successful import owns it.
        let memory = unsafe { second.allocate_memory(&allocate, None) }.unwrap();
        // SAFETY: the buffer and the memory are the second device's, the memory as long as the
        // buffer requires (the allocation the exported buffer is bound to at its offset).
        unsafe { second.bind_buffer_memory(buffer, memory, exported.offset) }.unwrap();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the second device is live.
        let pool = unsafe { second.create_command_pool(&pool_info, None) }.unwrap();
        // SAFETY: the family's queue 0, which the device was made with.
        let queue = unsafe { second.get_device_queue(family, 0) };
        Writer {
            device: second,
            buffer,
            memory,
            pool,
            queue,
            family,
            length,
        }
    }

    /// Fill every byte of the buffer with `value`, and wait for it.
    fn fill(&self, value: u8) {
        let device = &self.device;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is the device's; the command buffer is recorded, submitted, waited for
        // and freed here.
        unsafe {
            let command_buffer = device.allocate_command_buffers(&allocate).unwrap()[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(command_buffer, &begin).unwrap();
            let word = u32::from_ne_bytes([value; 4]);
            device.cmd_fill_buffer(command_buffer, self.buffer, 0, self.length, word);
            // The writes released to the external family the runtime's device acquires them from
            // (`Buffer::external_write_done`).
            let release = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(self.family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .buffer(self.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[release],
                &[],
            );
            device.end_command_buffer(command_buffer).unwrap();
            let buffers = [command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(self.queue).unwrap();
            device.free_command_buffers(self.pool, &buffers);
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // SAFETY: the second device's objects, idle and unused now; the exported buffer outlives
        // its memory's import.
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
        }
    }
}

/// A texture over an exportable `Private` buffer whose rows no image holds reads it through a
/// shadow, whose image stands across command buffers (`shadow_standing.rs`): another API's write
/// of the bytes, which `Buffer::external_write_done` announces, lets go of the standing, so the
/// texture's next read copies in what that write left, write after write; with no write between
/// two reads, the second copies nothing in. The writes are a second `VkDevice`'s fills through an
/// import of the buffer's opaque descriptor, as a decoder's; each read is a kernel's
/// (`shadow_standing_device`'s `Tools`).
#[test]
fn a_shadow_over_an_exported_buffer_reads_each_announced_write() {
    use super::shadow_standing_device::{inverted, Tools, HEIGHT, LENGTH, PITCH, WIDTH};
    let scratch = Scratch::new("exported-standing");
    let device = device(&scratch.0);
    let usage = crate::buffer::usage(device.shared().device_address);
    let name = device.name().to_owned();
    let buffer = match device.new_exportable_buffer(LENGTH, ResourceOptions::private()) {
        Ok(buffer) => buffer,
        Err(Error::NotImplemented(RuntimeGap::ExportableMemory)) => {
            assert!(
                !exports(&device, usage),
                "{name} exports a buffer's memory as an opaque fd, and the runtime refused it"
            );
            eprintln!("EXPORTED-STANDING on {name}: refused, the device exports no opaque fd");
            return;
        }
        Err(error) => panic!("{name}: {error:?}"),
    };
    let tools = Tools::new(&device);
    let mut descriptor =
        crate::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let plane = crate::IoSurfacePlane::new(0, PITCH, true);
    let texture = buffer.new_texture_with_iosurface_plane(&descriptor, plane);
    let texture = texture.unwrap();
    assert!(
        texture.shadow().is_some(),
        "{name}: padded rows make a shadow"
    );
    let writer = Writer::import(&device, buffer.export_memory_fd().unwrap(), usage);
    for value in [0x2a, 0x5c, 0x93] {
        writer.fill(value);
        buffer.external_write_done().unwrap();
        let (texels, first) = tools.texels(&texture);
        assert!(
            inverted(&texels, value),
            "{name}: the read after the write of {value:#x} took another's bytes"
        );
        assert_eq!(first.skipped, 0, "{name}: the write let go of the standing");
        let (texels, again) = tools.texels(&texture);
        assert!(inverted(&texels, value), "{name}: the read again");
        assert_eq!(
            again.skipped, 1,
            "{name}: with no write between, the image stood"
        );
    }
    eprintln!("EXPORTED-STANDING on {name}: each announced write reached the shadow");
}
