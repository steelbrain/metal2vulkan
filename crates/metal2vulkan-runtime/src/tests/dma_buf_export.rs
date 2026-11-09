//! A buffer whose memory is exported as a dma-buf (`BufferDescriptor::exported_as`,
//! `MemoryExport::DmaBuf`, `exported_memory.rs`), as a consumer's device cache of a decoded frame
//! that a video decoder writes through its import:
//!
//! - **The export.** A `Private` and a `Shared` frame laying byte `0x20` on a multiple of 256
//!   (`aligned_byte`), its padding unobserved, export a dma-buf whose `offset` puts that byte on a
//!   multiple of 256 of the allocation.
//! - **The external write.** A second `VkDevice` of the same physical device imports the dma-buf
//!   (`VkImportMemoryFdInfoKHR`), writes the frame's two planes through the import and releases
//!   the buffer to `VK_QUEUE_FAMILY_FOREIGN_EXT`, then waits for that (as a decoder's
//!   `vaSyncSurface`). `Buffer::external_write_done` notes the runtime's acquire: for the
//!   `Private` frame it waits in the first-work queue for the next submission
//!   (`FirstMoves::fills_wait`), which takes it; the `Shared` frame's is made at once.
//! - **The read.** The planes' textures over the buffer (`R8Unorm` 1024x64 at `0x20`, `RG8Unorm`
//!   512x32 after it, rows 1024 bytes apart) are each the linear image aliasing the buffer's
//!   memory (no `shadowed_textures`), made for its dma-buf handle type once the device's check of
//!   that exported memory answered and passed, wherever the same texture at offset 0 of a plain
//!   buffer is one; where the check did not pass for a texture that reads (a device that makes no
//!   image over a dma-buf: the RX 550 under RADV), each is a shadow. A blit of each reads the
//!   second device's bytes.
//! - **Refusals.** A device without the export refuses the making
//!   (`NotImplemented(RuntimeGap::DmaBufExport)`) only where it lacks an extension or the export;
//!   `NoCopy` contents with `exported_as` are `Invalid` (`ExportedNoCopy`); a buffer not made
//!   exportable as a dma-buf refuses `export_memory_dma_buf` and one made exportable as neither
//!   handle refuses `external_write_done` (`NotExportable`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`DMA-BUF-EXPORT`): what the
//! device exported and each plane's path.

use super::blit_holds::{device, Host, Scratch, WAIT};
use crate::error::{Error, Invalid, Problem, RuntimeGap};
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, IoSurfacePlane, MemoryExport,
    Origin, ResourceOptions, Size, Texture, TextureDescriptor, TexturePlacement,
    TextureShadowReason,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::os::fd::IntoRawFd;

/// The byte the frames lay on a multiple of 256: CoreVideo's first plane's.
const K: u64 = 0x20;

/// The bytes before byte 0 of the frame in its `VkBuffer`: `K`'s shortfall from 256.
const LEAD: u64 = 256 - K;

/// A plane of the frame: its format, size, bytes a row and offset in the frame.
#[derive(Clone, Copy, Debug)]
struct Plane {
    format: PixelFormat,
    size: [u64; 2],
    row: u64,
    at: u64,
}

/// The frame's luma and chroma planes, rows a multiple of 256 bytes apart.
const PLANES: [Plane; 2] = [
    Plane {
        format: PixelFormat::R8Unorm,
        size: [1024, 64],
        row: 1024,
        at: K,
    },
    Plane {
        format: PixelFormat::RG8Unorm,
        size: [512, 32],
        row: 1024,
        at: K + 1024 * 64,
    },
];

/// The frame's bytes.
const LENGTH: u64 = K + 1024 * 64 * 3 / 2;

/// The byte the second device writes at `offset` of the frame.
fn decoded_byte(offset: u64) -> u8 {
    (offset * 13 + offset / 1031 + 3) as u8
}

/// Whether the physical device under `device` lists what a dma-buf export needs and exports a
/// buffer of `usage`'s memory as a dma-buf.
fn exports_dma_bufs(device: &Device, usage: vk::BufferUsageFlags) -> bool {
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let listed = unsafe {
        handles
            .instance
            .enumerate_device_extension_properties(handles.physical_device)
    }
    .unwrap();
    let lists = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    let named = lists(ash::khr::external_memory_fd::NAME)
        && lists(ash::ext::external_memory_dma_buf::NAME)
        && lists(ash::ext::queue_family_foreign::NAME);
    let query = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
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

/// A frame of [`LENGTH`] zero bytes of `options`, laying byte `K` on a multiple of 256, its
/// padding unobserved, its memory exportable as `export` where given.
fn frame(
    device: &Device,
    options: ResourceOptions,
    export: Option<MemoryExport>,
) -> Result<Buffer, Error> {
    let descriptor = BufferDescriptor::new(LENGTH, options, BufferContents::Zero)
        .with_aligned_byte(K)
        .with_padding_unobserved();
    let descriptor = match export {
        Some(export) => descriptor.with_exported_as(export),
        None => descriptor,
    };
    device.new_buffer_with_descriptor(descriptor)
}

/// `plane`'s sampled texture over `buffer` at `at`, and whether it is the linear image aliasing
/// the buffer's memory: no shadow counted as it is made.
fn plane_texture(device: &Device, buffer: &Buffer, (plane, at): (Plane, u64)) -> (Texture, bool) {
    let before = device.work_counts().shadowed_textures;
    let texture = buffer
        .new_texture_with_iosurface_plane(&plane_descriptor(buffer, plane), plane_at(plane, at))
        .unwrap();
    let aliases = device.work_counts().shadowed_textures == before;
    (texture, aliases)
}

/// The descriptor of `plane`'s texture over `buffer`, which only reads.
fn plane_descriptor(buffer: &Buffer, plane: Plane) -> TextureDescriptor {
    let [w, h] = plane.size;
    let mut descriptor = TextureDescriptor::texture_2d(plane.format, w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    descriptor
}

/// `plane` at byte `at` of its buffer, as an IOSurface plane.
fn plane_at(plane: Plane, at: u64) -> IoSurfacePlane {
    IoSurfacePlane::new(at, plane.row, true)
}

/// Where `plane`'s texture over `buffer` at byte `at` lies if made now.
fn plane_placement(buffer: &Buffer, (plane, at): (Plane, u64)) -> TexturePlacement {
    let descriptor = plane_descriptor(buffer, plane);
    (buffer.texture_placement_with_iosurface_plane(&descriptor, plane_at(plane, at))).unwrap()
}

/// The texels of `plane`'s `texture` as a blit copies them into a buffer, rows one after another.
fn blitted(device: &Device, texture: &Texture, plane: Plane) -> Vec<u8> {
    let [w, h] = plane.size;
    let bytes = plane.row * h;
    let read = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &read,
        0,
        plane.row,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    // SAFETY: the read buffer's mapped bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// A second `VkDevice` of the physical device under `device`, with the dma-buf import's
/// extensions, which writes the frame through its import of `exported` and releases it to the
/// foreign queue family, waited for: a decoder's write and its `vaSyncSurface`.
fn write_through_import(device: &Device, exported: crate::ExportedMemory) {
    let device_address = device.shared().device_address;
    // SAFETY: the handles are used while the device lives; the second device is this
    // function's own and destroyed before it returns.
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
    let extensions = [
        ash::khr::external_memory_fd::NAME.as_ptr(),
        ash::ext::external_memory_dma_buf::NAME.as_ptr(),
        ash::ext::queue_family_foreign::NAME.as_ptr(),
    ];
    let mut features12 =
        vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(device_address);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .push_next(&mut features12);
    // SAFETY: the physical device is the instance's and lists the extensions (the export ran).
    let second = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    let fds = ash::khr::external_memory_fd::Device::new(instance, &second);
    let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
    // SAFETY: the descriptor is a dma-buf the runtime exported, of this physical device.
    unsafe {
        fds.get_memory_fd_properties(
            handle,
            std::os::fd::AsRawFd::as_raw_fd(&exported.fd),
            &mut fd_properties,
        )
    }
    .unwrap();
    assert_ne!(
        fd_properties.memory_type_bits & 1 << exported.memory_type_index,
        0,
        "the dma-buf imports as the memory type it was exported from"
    );
    // The buffer the import binds, of the exported one's size and usage, from the frame's
    // `VkBuffer`'s first byte, `LEAD` before the frame's: exclusive to the second device's one
    // family, whose release to the foreign family Vulkan takes without synchronization2.
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
    let buffer_info = vk::BufferCreateInfo::default()
        .size((LENGTH + LEAD).next_multiple_of(16))
        .usage(crate::buffer::usage(device_address))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: the second device is live and `buffer_info` valid for it.
    let imported = unsafe { second.create_buffer(&buffer_info, None) }.unwrap();
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle)
        .fd(exported.fd.into_raw_fd());
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(imported);
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
    // SAFETY: the descriptor is the export's, its size and type the allocation's; a successful
    // import owns it.
    let memory = unsafe { second.allocate_memory(&allocate, None) }.unwrap();
    // SAFETY: the buffer and the memory are the second device's; the frame's `VkBuffer` lies
    // there, `LEAD` bytes before the frame's first.
    unsafe { second.bind_buffer_memory(imported, memory, exported.offset - LEAD) }.unwrap();

    // The planes' bytes, in a host-visible buffer of the second device's own.
    let decoded: Vec<u8> = (0..LENGTH).map(decoded_byte).collect();
    let staging_info = vk::BufferCreateInfo::default()
        .size(LENGTH)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    // SAFETY: as above.
    let staging = unsafe { second.create_buffer(&staging_info, None) }.unwrap();
    // SAFETY: as above.
    let requirements = unsafe { second.get_buffer_memory_requirements(staging) };
    // SAFETY: the physical device is the instance's.
    let types = unsafe { instance.get_physical_device_memory_properties(physical) };
    let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = (0..types.memory_type_count)
        .find(|&index| {
            requirements.memory_type_bits & 1 << index != 0
                && types.memory_types[index as usize]
                    .property_flags
                    .contains(visible)
        })
        .unwrap();
    let staging_allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    // SAFETY: as above.
    let staging_memory = unsafe { second.allocate_memory(&staging_allocate, None) }.unwrap();
    // SAFETY: as above; host-visible memory of the staging buffer's size, mapped whole once.
    unsafe {
        second
            .bind_buffer_memory(staging, staging_memory, 0)
            .unwrap();
        let mapped = second
            .map_memory(
                staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap()
            .cast::<u8>();
        std::ptr::copy_nonoverlapping(decoded.as_ptr(), mapped, decoded.len());
    }

    // The planes' copies through the import, then the release to the foreign family.
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
    // SAFETY: as above.
    let pool = unsafe { second.create_command_pool(&pool_info, None) }.unwrap();
    let command_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: as above.
    let command_buffer = unsafe { second.allocate_command_buffers(&command_info) }.unwrap()[0];
    let copies: Vec<vk::BufferCopy> = PLANES
        .iter()
        .map(|plane| vk::BufferCopy {
            src_offset: plane.at,
            dst_offset: LEAD + plane.at,
            size: plane.row * plane.size[1],
        })
        .collect();
    let release = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty())
        .src_queue_family_index(family)
        .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
        .buffer(imported)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    // SAFETY: the command buffer is the second device's, recorded once; the copies lie within
    // both buffers; the release names the queue's family and the foreign one, which the device
    // enables.
    unsafe {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        second.begin_command_buffer(command_buffer, &begin).unwrap();
        second.cmd_copy_buffer(command_buffer, staging, imported, &copies);
        second.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[release],
            &[],
        );
        second.end_command_buffer(command_buffer).unwrap();
        let queue = second.get_device_queue(family, 0);
        let buffers = [command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        let fence = second
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        second.queue_submit(queue, &[submit], fence).unwrap();
        second
            .wait_for_fences(&[fence], true, WAIT.as_nanos() as u64)
            .unwrap();
        second.destroy_fence(fence, None);
        second.destroy_command_pool(pool, None);
        second.destroy_buffer(staging, None);
        second.free_memory(staging_memory, None);
        // The frame outlives its memory's import.
        second.destroy_buffer(imported, None);
        second.free_memory(memory, None);
        second.destroy_device(None);
    }
}

#[test]
fn a_frame_a_second_device_writes_through_its_dma_buf_reads_in_the_runtime_s_textures() {
    let scratch = Scratch::new("dma-buf-export");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let usage = crate::buffer::usage(device.shared().device_address);
    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
        let buffer = match frame(&device, options, Some(MemoryExport::DmaBuf)) {
            Ok(buffer) => buffer,
            Err(Error::NotImplemented(RuntimeGap::DmaBufExport)) => {
                assert!(
                    !exports_dma_bufs(&device, usage),
                    "{name} exports a buffer's memory as a dma-buf, and the runtime refused it"
                );
                eprintln!("DMA-BUF-EXPORT on {name}: refused, the device exports no dma-buf");
                return;
            }
            Err(error) => panic!("{name}: {error:?}"),
        };
        let exported = buffer.export_memory_dma_buf().unwrap();
        eprintln!(
            "DMA-BUF-EXPORT on {name}: {options:?}: {} bytes at {:#x} of an allocation of {}, \
             dedicated {}, type {}",
            exported.length,
            exported.offset,
            exported.allocation_size,
            exported.dedicated,
            exported.memory_type_index
        );
        assert_eq!(exported.length, LENGTH);
        assert_eq!(
            (exported.offset + K) % 256,
            0,
            "{options:?}: byte {K:#x} on a multiple of 256 of the allocation"
        );
        assert!(exported.offset + exported.length <= exported.allocation_size);
        // An opaque descriptor is not this buffer's export.
        assert!(matches!(
            buffer.export_memory_fd(),
            Err(Error::Invalid(Invalid {
                problem: Problem::NotExportable,
                ..
            }))
        ));
        write_through_import(&device, exported);
        let raw = buffer.at(0).0;
        let firsts = &device.shared().textures.first_moves;
        buffer.external_write_done().unwrap();
        let waits = firsts.fills_wait(raw);
        assert_eq!(
            waits,
            buffer.contents().is_none(),
            "{options:?}: the acquire waits for the next submission unless the host reads the \
             contents"
        );
        if let Some(contents) = buffer.contents() {
            // SAFETY: the buffer's mapped bytes, which no command uses now.
            let seen = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LENGTH as usize) };
            for plane in PLANES {
                let range = plane.at as usize..(plane.at + plane.row * plane.size[1]) as usize;
                let want: Vec<u8> = (range.start as u64..range.end as u64)
                    .map(decoded_byte)
                    .collect();
                assert!(
                    seen[range] == want[..],
                    "{options:?} {plane:?}: the contents"
                );
            }
        }
        let plain = frame(&device, options, None).unwrap();
        // An image over exported memory starts `UNDEFINED`, as one that names a handle type does:
        // the device's check of that memory decides it (`in_place.rs`, "The runtime's exported
        // memory"), so its verdicts are known first.
        assert!(
            device.settle_in_place_checks(WAIT),
            "{name}: the check answered"
        );
        for plane in PLANES {
            // Once the check answered: as the plain frame's where its verdict for the memory
            // passed, else the shadow its verdict decided.
            let placement = plane_placement(&buffer, (plane, plane.at));
            let (texture, aliases) = plane_texture(&device, &buffer, (plane, plane.at));
            let (_, plain_aliases) = plane_texture(&device, &plain, (plane, plane.at));
            eprintln!(
                "DMA-BUF-EXPORT on {name}: {options:?} {plane:?}: aliases its buffer {aliases} \
                 (a plain frame's {plain_aliases}), placed {placement:?}"
            );
            match placement {
                TexturePlacement::Shadow(
                    TextureShadowReason::Unchecked { .. } | TextureShadowReason::CheckFailed { .. },
                ) => assert!(
                    !aliases,
                    "{options:?} {plane:?}: a shadow where the check did not pass"
                ),
                _ => assert_eq!(aliases, plain_aliases, "{options:?} {plane:?}: no shadow"),
            }
            let want: Vec<u8> = (plane.at..plane.at + plane.row * plane.size[1])
                .map(decoded_byte)
                .collect();
            let got = blitted(&device, &texture, plane);
            assert!(
                got == want,
                "{options:?} {plane:?}: the second device's texels"
            );
        }
        assert!(
            !firsts.fills_wait(raw),
            "{options:?}: a submission took the acquire"
        );
    }
}

#[test]
fn a_dma_buf_export_is_refused_where_the_device_lacks_it_and_for_other_buffers() {
    let scratch = Scratch::new("dma-buf-export-refusals");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let usage = crate::buffer::usage(device.shared().device_address);
    let made = frame(
        &device,
        ResourceOptions::private(),
        Some(MemoryExport::DmaBuf),
    );
    match &made {
        Ok(_) => assert!(
            exports_dma_bufs(&device, usage),
            "{name}: made without the export"
        ),
        Err(Error::NotImplemented(RuntimeGap::DmaBufExport)) => {
            assert!(
                !exports_dma_bufs(&device, usage),
                "{name}: refused with the export"
            )
        }
        Err(error) => panic!("{name}: {error:?}"),
    }
    eprintln!("DMA-BUF-EXPORT on {name}: made {}", made.is_ok());
    let not_exportable = |result: Result<(), Error>| {
        matches!(
            result,
            Err(Error::Invalid(Invalid {
                problem: Problem::NotExportable,
                ..
            }))
        )
    };
    // A plain buffer exports nothing and acquires nothing.
    let plain = frame(&device, ResourceOptions::private(), None).unwrap();
    assert!(not_exportable(plain.export_memory_dma_buf().map(drop)));
    assert!(not_exportable(plain.external_write_done()));
    // An opaque descriptor's buffer is not a dma-buf's.
    if let Ok(opaque) = frame(
        &device,
        ResourceOptions::private(),
        Some(MemoryExport::OpaqueFd),
    ) {
        assert!(not_exportable(opaque.export_memory_dma_buf().map(drop)));
        assert!(opaque.export_memory_fd().is_ok());
    }
    // Over the caller's memory, refused before any import.
    let host = Host::new();
    // SAFETY: the memory outlives the call, which refuses it before any import.
    let memory = unsafe { host.memory() };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: None,
    };
    let descriptor = BufferDescriptor::new(1 << 16, ResourceOptions::shared(), contents)
        .with_exported_as(MemoryExport::DmaBuf);
    let made = device.new_buffer_with_descriptor(descriptor);
    assert!(
        matches!(
            made,
            Err(Error::Invalid(Invalid {
                problem: Problem::ExportedNoCopy,
                ..
            }))
        ),
        "the export over the caller's memory: {:?}",
        made.err()
    );
}
