//! An exportable buffer's allocation and a texture over it (`exported_memory.rs`, `in_place.rs`'s
//! "The runtime's exported memory"), of each handle type the device exports. An image may not be
//! bound to a buffer's dedicated allocation (`VUID-vkBindImageMemory-memory-10926`), so a buffer
//! the device lets lie in a shared exportable block must lie there, where a texture over it is an
//! image in its memory: the buffer takes its own dedicated allocation only where the device
//! exports such memory only so (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`) or requires it
//! (`VkMemoryDedicatedRequirements::requiresDedicatedAllocation`), never where the driver only
//! prefers it (`prefersDedicatedAllocation`).
//!
//! For each handle type, the device's answers are asked apart from the runtime and printed
//! (`EXPORT-DEDICATION`): of a buffer of the runtime's usage, whether the handle type is
//! exportable, only so, and preferred or required dedicated; of a linear [`SIDE`] squared
//! `R8_UNORM` image of a sampled texture's usage made for the handle type, the same. Then an
//! exportable `Shared` buffer of [`SIDE`] squared bytes is made and exported: dedicated exactly
//! where the buffer's answers require it. A texture over it, `R8Unorm` rows [`SIDE`] bytes apart,
//! is asked for once the device's in-place check answered: where the buffer's memory is shared and
//! the image's answers neither require a dedicated allocation nor export only so, an image in place
//! (`TexturePlacement::NotHostPages`), no shadow counted, that reads what the buffer holds. Every
//! placement, the shadows counted and the check's notes of the class are printed.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::{Error, RuntimeGap};
use crate::{
    BlitOption, BufferContents, BufferDescriptor, Device, MemoryExport, Origin, ResourceOptions,
    Size, Texture, TextureDescriptor, TexturePlacement,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The side of the texture, and the bytes of each of its rows.
const SIDE: u64 = 1024;

/// What a device answers of exporting memory of one handle type from a resource.
#[derive(Debug)]
struct Dedication {
    /// The handle type is exportable from such a resource's memory.
    exportable: bool,
    /// Only from its dedicated allocation (`DEDICATED_ONLY`).
    only: bool,
    /// The driver prefers a dedicated allocation for the resource.
    prefers: bool,
    /// The driver requires one.
    requires: bool,
}

impl Dedication {
    /// Whether the resource must take a dedicated allocation of its own.
    fn needed(&self) -> bool {
        self.only || self.requires
    }
}

/// The handle type of `export`.
fn handle(export: MemoryExport) -> vk::ExternalMemoryHandleTypeFlags {
    match export {
        MemoryExport::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
        MemoryExport::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
    }
}

/// What `device` answers of a buffer of the runtime's usage, `SIDE` squared bytes, whose memory
/// it exports as `handle`: asked, and the buffer made for its requirements and destroyed.
fn buffer_dedication(device: &Device, handle: vk::ExternalMemoryHandleTypeFlags) -> Dedication {
    let shared = device.shared();
    let usage = crate::buffer::usage(shared.device_address);
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    let query = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(handle);
    let mut properties = vk::ExternalBufferProperties::default();
    // SAFETY: the physical device is the instance's, at Vulkan 1.1 or later.
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
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
    let info = shared.sharing().buffer(
        vk::BufferCreateInfo::default()
            .size(SIDE * SIDE)
            .usage(usage)
            .push_next(&mut external),
    );
    // SAFETY: the device is live and `info` and its chain valid.
    let raw = unsafe { crate::creation::buffer(handles.device, &info) }.unwrap();
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    let asked = vk::BufferMemoryRequirementsInfo2::default().buffer(raw);
    // SAFETY: `raw` is a live buffer of this device, destroyed right after with no memory bound.
    unsafe {
        handles
            .device
            .get_buffer_memory_requirements2(&asked, &mut requirements);
        crate::release::object(handles.device, raw);
    }
    Dedication {
        exportable: features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
        only: features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
        prefers: dedicated.prefers_dedicated_allocation == vk::TRUE,
        requires: dedicated.requires_dedicated_allocation == vk::TRUE,
    }
}

/// What `device` answers of a linear `SIDE` squared `R8_UNORM` image, sampled and copied to and
/// from, whose memory it exports as `handle`: asked, and the image made for its requirements and
/// destroyed where the format query takes it. Not exportable where the query refuses it.
fn image_dedication(device: &Device, handle: vk::ExternalMemoryHandleTypeFlags) -> Dedication {
    let shared = device.shared();
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let (format, tiling) = (vk::Format::R8_UNORM, vk::ImageTiling::LINEAR);
    let mut external_info =
        vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
    let query = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(usage)
        .push_next(&mut external_info);
    let mut external = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external);
    // SAFETY: the physical device is the instance's, at Vulkan 1.1 or later.
    let taken = unsafe {
        handles
            .instance
            .get_physical_device_image_format_properties2(
                handles.physical_device,
                &query,
                &mut properties,
            )
    };
    let features = external.external_memory_properties.external_memory_features;
    let none = Dedication {
        exportable: false,
        only: false,
        prefers: false,
        requires: false,
    };
    if taken.is_err() || !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
        return none;
    }
    let mut made_for = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let info = shared.sharing().image(
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: SIDE as u32,
                height: SIDE as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut made_for),
    );
    // SAFETY: the device is live and `info` and its chain valid; the format query took it.
    let raw = unsafe { crate::creation::image(handles.device, &info) }.unwrap();
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    let asked = vk::ImageMemoryRequirementsInfo2::default().image(raw);
    // SAFETY: `raw` is a live image of this device, destroyed right after with no memory bound.
    unsafe {
        handles
            .device
            .get_image_memory_requirements2(&asked, &mut requirements);
        crate::release::object(handles.device, raw);
    }
    Dedication {
        exportable: true,
        only: features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
        prefers: dedicated.prefers_dedicated_allocation == vk::TRUE,
        requires: dedicated.requires_dedicated_allocation == vk::TRUE,
    }
}

/// The byte the buffer holds at `offset`.
fn byte(offset: u64) -> u8 {
    (offset * 13 + offset / 1021 + 3) as u8
}

/// `texture`'s texels as a blit copies them into a new `Shared` buffer, rows [`SIDE`] bytes apart.
fn read(device: &Device, texture: &Texture) -> Vec<u8> {
    let bytes = SIDE * SIDE;
    let read = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &read,
        0,
        SIDE,
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

#[test]
fn an_exportable_buffer_is_dedicated_only_where_required_and_its_texture_lies_in_its_memory() {
    let scratch = Scratch::new("export-dedication");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    for export in [MemoryExport::OpaqueFd, MemoryExport::DmaBuf] {
        let descriptor =
            BufferDescriptor::new(SIDE * SIDE, ResourceOptions::shared(), BufferContents::Zero)
                .with_exported_as(export);
        let buffer = match device.new_buffer_with_descriptor(descriptor) {
            Ok(buffer) => buffer,
            Err(Error::NotImplemented(
                gap @ (RuntimeGap::ExportableMemory | RuntimeGap::DmaBufExport),
            )) => {
                eprintln!("EXPORT-DEDICATION on {name}: {export:?} refused: {gap:?}");
                continue;
            }
            Err(error) => panic!("{name} {export:?}: {error:?}"),
        };
        // Asked of a device that exports the handle type, which enabled its extensions.
        let buffer_asked = buffer_dedication(&device, handle(export));
        let image_asked = image_dedication(&device, handle(export));
        eprintln!(
            "EXPORT-DEDICATION on {name}: {export:?}: buffer {buffer_asked:?}; linear R8 image \
             {image_asked:?}"
        );
        let exported = match export {
            MemoryExport::OpaqueFd => buffer.export_memory_fd(),
            MemoryExport::DmaBuf => buffer.export_memory_dma_buf(),
        }
        .unwrap();
        let bytes: Vec<u8> = (0..SIDE * SIDE).map(byte).collect();
        // SAFETY: the buffer's mapped bytes, which no command uses now.
        unsafe {
            let contents = buffer.contents().unwrap().as_ptr();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents, bytes.len());
        }
        let mut texture_descriptor =
            TextureDescriptor::texture_2d(PixelFormat::R8Unorm, SIDE, SIDE, false);
        texture_descriptor.usage = TextureUsage::SHADER_READ;
        texture_descriptor.resource_options = ResourceOptions::shared();
        // The first query starts the device's check, which answers in seconds.
        let asked = buffer
            .texture_placement(&texture_descriptor, 0, SIDE)
            .unwrap();
        assert!(device.settle_in_place_checks(WAIT), "the check answered");
        let placement = buffer
            .texture_placement(&texture_descriptor, 0, SIDE)
            .unwrap();
        let shadows = device.work_counts().shadowed_textures;
        let texture = buffer.new_texture(&texture_descriptor, 0, SIDE).unwrap();
        let shadowed = device.work_counts().shadowed_textures - shadows;
        eprintln!(
            "EXPORT-DEDICATION on {name}: {export:?}: an allocation of {} bytes, dedicated {}, \
             type {}; a texture over it: {asked:?} before the check, {placement:?} after, \
             shadows made {shadowed}",
            exported.allocation_size, exported.dedicated, exported.memory_type_index
        );
        let class = match export {
            MemoryExport::OpaqueFd => "ExportedFd ",
            MemoryExport::DmaBuf => "ExportedDmaBuf ",
        };
        let report = device.in_place_check_report();
        for line in
            (report.iter().skip(crate::in_place::VERDICTS)).filter(|line| line.starts_with(class))
        {
            eprintln!("EXPORT-DEDICATION on {name}: the check: {line}");
        }
        assert!(read(&device, &texture) == bytes, "{export:?}: the texels");
        assert_eq!(
            exported.dedicated,
            buffer_asked.needed(),
            "{name} {export:?}: a dedicated allocation exactly where the device requires one"
        );
        if !exported.dedicated && image_asked.exportable && !image_asked.needed() {
            assert_eq!(
                placement,
                TexturePlacement::NotHostPages,
                "{name} {export:?}: the texture lies in the buffer's shared memory"
            );
            assert_eq!(shadowed, 0, "{name} {export:?}: no shadow");
        }
    }
}
