//! Textures over the runtime's exported memory (`exported_memory.rs`), shaped as a consumer's
//! cache of a decoded 3840x2160 NV12 picture: one exportable buffer, its luma plane (`R8Unorm`
//! 3840x2160) 32 bytes in and its chroma plane (`RG8Unorm` 1920x1080) right after it, rows 3840
//! bytes apart with no bytes between them, the buffer laying byte 32 on a multiple of 256 of its
//! memory (`aligned_byte`), so each plane lies at a multiple of 256 there.
//!
//! - **The image names the memory's handle type.** Each plane's texture is an image bound in the
//!   buffer's exported memory, which must name a handle type the memory is exportable as
//!   (`VUID-vkBindImageMemory-memory-02728`); the validation layer, enabled on the device, reports
//!   nothing ([`Watched::layer_reported_nothing`]): without the handle type it reports 02728 at the
//!   bind.
//! - **A shadow first, then in place.** Made before the device's check of that exported memory
//!   answered, each texture is a shadow (`TexturePlacement::Pending`), whose blit reads the
//!   buffer's bytes; once the check answered (`Device::settle_in_place_checks`) and passed, the
//!   next commit moves both in place (`WorkCounts::pooled_textures_moved_in_place`), and their
//!   blits read the same bytes through the image in the buffer's memory. A texture made after is
//!   in place at once (`TexturePlacement::NotHostPages`, no shadow counted). Where the check did
//!   not pass for that memory (a device that makes no image over a dma-buf: the RX 550 under
//!   RADV), each texture stays the shadow its verdict decided (`TextureShadowReason::Unchecked`
//!   or `CheckFailed`), and reads the same bytes.
//! - **Each handle type, each storage.** Of an opaque file descriptor and of a dma-buf where the
//!   device exports one, `Private` (its bytes written by a blit, as a decoder writes a cache the CPU
//!   never maps) and `Shared` (written through its contents). Where the device exports such memory
//!   only as the buffer's dedicated allocation, to which no image may be bound
//!   (`VUID-vkBindImageMemory-memory-10926`), each texture is a shadow that reads the same bytes,
//!   its placement refused (`InPlaceRefusal::Placement`).
//!
//! - **A dma-buf's block, its images naming an opaque file descriptor.** Where the device's
//!   buffer queries (`vkGetPhysicalDeviceExternalBufferProperties`) answer a buffer's memory
//!   exportable as a dma-buf and as an opaque file descriptor together, each compatible with the
//!   other and the opaque one not only as a dedicated allocation, the block of dma-bufs is
//!   exportable as both, and elsewhere as a dma-buf alone (the queries asked here, apart from the
//!   runtime's). Where the device's image format query then takes a dma-buf's handle type for no
//!   image of a plane's format in a shared block, linear or `DRM_FORMAT_MOD_LINEAR` (RADV: none
//!   linear, and the other only in a dedicated allocation), the plane's image names the opaque
//!   one: in place where the device's check of that memory passed, which made the same image
//!   (`WorkCounts::exported_dma_buf_images_named_opaque_fd`), or a shadow of a byte the check
//!   found changed, never one it made no image for. The descriptor the buffer exports is a dma-buf
//!   of its memory either way.
//!
//! Printed (`EXPORTED-TEXTURES`, `EXPORTED-DMA-BUF`): what the device exported, and for each plane
//! its image's alignment, row pitch and size as `vkGetImageMemoryRequirements` and
//! `vkGetImageSubresourceLayout` answer them, and its offset in the memory.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::layer_claim::Watched;
use super::pending_rebind::device;
use crate::error::{Error, RuntimeGap};
use crate::memory::BoundMemory;
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, InPlaceRefusal, MemoryExport,
    Origin, ResourceOptions, Size, Texture, TextureDescriptor, TexturePlacement,
    TextureShadowReason,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The byte the luma plane starts at: a header's 32 bytes in, as the consumer's cache lays it.
const LUMA_AT: u64 = 32;

/// The bytes of a row of each plane.
const ROW: u64 = 3840;

/// The picture's luma rows.
const HEIGHT: u64 = 2160;

/// The byte the chroma plane starts at, right after the luma plane.
const CHROMA_AT: u64 = LUMA_AT + ROW * HEIGHT;

/// The buffer's bytes: the header and both planes.
const LENGTH: u64 = CHROMA_AT + ROW * HEIGHT / 2;

/// A plane of the picture: its format, size and offset in the buffer.
#[derive(Clone, Copy, Debug)]
struct Plane {
    format: PixelFormat,
    size: [u64; 2],
    at: u64,
}

/// The picture's luma and chroma planes.
const PLANES: [Plane; 2] = [
    Plane {
        format: PixelFormat::R8Unorm,
        size: [3840, HEIGHT],
        at: LUMA_AT,
    },
    Plane {
        format: PixelFormat::RG8Unorm,
        size: [1920, HEIGHT / 2],
        at: CHROMA_AT,
    },
];

/// The byte the picture holds at `offset`.
fn byte(offset: u64) -> u8 {
    (offset * 7 + offset / 4093 + 1) as u8
}

/// The bytes of `plane`, rows one after another.
fn want(plane: Plane) -> Vec<u8> {
    (plane.at..plane.at + ROW * plane.size[1])
        .map(byte)
        .collect()
}

/// The picture: an exportable buffer of `options` whose memory is exported as `export`, its byte
/// [`LUMA_AT`] on a multiple of 256 of that memory.
fn picture(
    device: &Device,
    options: ResourceOptions,
    export: MemoryExport,
) -> Result<Buffer, Error> {
    let descriptor = BufferDescriptor::new(LENGTH, options, BufferContents::Zero)
        .with_aligned_byte(LUMA_AT)
        .with_exported_as(export);
    device.new_buffer_with_descriptor(descriptor)
}

/// A blit of `bytes` from `source` into `destination`, waited for.
fn copy(device: &Device, source: &Buffer, destination: &Buffer, bytes: u64) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_buffer(source, 0, destination, 0, bytes)).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Write the picture's bytes: through its contents where it has them, else by a blit from a
/// `Shared` buffer.
fn fill(device: &Device, picture: &Buffer) {
    let bytes: Vec<u8> = (0..LENGTH).map(byte).collect();
    match picture.contents() {
        Some(contents) => {
            // SAFETY: the buffer's mapped bytes, which no command uses now.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
        }
        None => {
            let staging = device
                .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
                .unwrap();
            copy(device, &staging, picture, LENGTH);
        }
    }
}

/// `plane`'s sampled texture's descriptor over a buffer of `options`.
fn descriptor(plane: Plane, options: ResourceOptions) -> TextureDescriptor {
    let [w, h] = plane.size;
    let mut descriptor = TextureDescriptor::texture_2d(plane.format, w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    descriptor
}

/// The texels of `plane`'s `texture` as a blit copies them into a buffer, rows [`ROW`] bytes
/// apart.
fn read(device: &Device, texture: &Texture, plane: Plane) -> Vec<u8> {
    let [w, h] = plane.size;
    let bytes = ROW * h;
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
        ROW,
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

/// What the device answers of `texture`'s image, an image bound in `picture`'s memory at
/// `plane`'s texels: its alignment, its first subresource's row pitch and offset, its size, and
/// the texels' offset in the memory.
fn image_facts(device: &Device, picture: &Buffer, texture: &Texture, plane: Plane) -> [u64; 5] {
    // SAFETY: the image is the texture's, used only to ask of it while the texture lives.
    let image = unsafe { texture.vk_image() }.expect("an image in place");
    let raw = device.shared().vk();
    let subresource = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: a live linear image of the device.
    let (requirements, layout) = unsafe {
        (
            raw.get_image_memory_requirements(image),
            raw.get_image_subresource_layout(image, subresource),
        )
    };
    let at = picture.memory().1 + plane.at;
    [
        requirements.alignment,
        layout.row_pitch,
        layout.offset,
        requirements.size,
        at,
    ]
}

#[test]
fn an_nv12_picture_s_planes_over_exported_memory_go_in_place_naming_its_handle_type() {
    let watched = Watched::default();
    let scratch = Scratch::new("exported-textures");
    let device = device(&watched, &scratch.0);
    let name = device.name().to_owned();
    // Whether no texture over exported memory was made yet: the device's check has not answered.
    let mut first = true;
    for export in [MemoryExport::OpaqueFd, MemoryExport::DmaBuf] {
        for options in [ResourceOptions::private(), ResourceOptions::shared()] {
            let picture = match picture(&device, options, export) {
                Ok(picture) => picture,
                Err(Error::NotImplemented(
                    gap @ (RuntimeGap::ExportableMemory | RuntimeGap::DmaBufExport),
                )) => {
                    eprintln!("EXPORTED-TEXTURES on {name}: {export:?} refused: {gap:?}");
                    break;
                }
                Err(error) => panic!("{name} {export:?} {options:?}: {error:?}"),
            };
            let exported = match export {
                MemoryExport::OpaqueFd => picture.export_memory_fd(),
                MemoryExport::DmaBuf => picture.export_memory_dma_buf(),
            }
            .unwrap();
            eprintln!(
                "EXPORTED-TEXTURES on {name}: {export:?} {options:?}: {} bytes at {:#x} of an \
                 allocation of {}, dedicated {}, type {}",
                exported.length,
                exported.offset,
                exported.allocation_size,
                exported.dedicated,
                exported.memory_type_index
            );
            assert_eq!(
                (exported.offset + LUMA_AT) % 256,
                0,
                "{export:?} {options:?}: the luma plane on a multiple of 256 of the memory"
            );
            fill(&device, &picture);
            let made = |plane: Plane| {
                let descriptor = descriptor(plane, options);
                let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
                let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
                (texture, placement)
            };
            if exported.dedicated {
                for plane in PLANES {
                    let (texture, placement) = made(plane);
                    let refused = TextureShadowReason::Refused(InPlaceRefusal::Placement);
                    assert_eq!(
                        placement,
                        TexturePlacement::Shadow(refused),
                        "{export:?} {options:?} {plane:?}: no image in a dedicated allocation"
                    );
                    assert!(texture.shadow().is_some());
                    assert!(read(&device, &texture, plane) == want(plane));
                }
                eprintln!(
                    "EXPORTED-TEXTURES on {name}: {export:?} {options:?}: a dedicated \
                     allocation, each plane a shadow"
                );
                continue;
            }
            let textures = PLANES.map(made);
            for (plane, (texture, placement)) in PLANES.iter().zip(&textures) {
                if first {
                    // The first texture's placement query starts the device's check, which takes
                    // seconds: the textures are made before it answers.
                    assert_eq!(
                        *placement,
                        TexturePlacement::Pending,
                        "{export:?} {options:?} {plane:?}: made before the check answered"
                    );
                    assert!(texture.shadow().is_some(), "{plane:?}: a shadow first");
                }
                assert!(
                    read(&device, texture, *plane) == want(*plane),
                    "{export:?} {options:?} {plane:?}: the texels, read as made"
                );
            }
            // Once the check answered: in place where its verdict for the memory passed, else a
            // shadow its verdict decided.
            let answer = |plane: Plane| {
                let descriptor = descriptor(plane, options);
                let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
                assert!(
                    matches!(
                        placement,
                        TexturePlacement::NotHostPages
                            | TexturePlacement::Shadow(
                                TextureShadowReason::Unchecked { .. }
                                    | TextureShadowReason::CheckFailed { .. }
                            )
                    ),
                    "{export:?} {options:?} {plane:?}: in place, or a shadow the device's check \
                     of that memory decided: {placement:?}"
                );
                placement
            };
            if first {
                let counts = device.work_counts();
                assert!(device.settle_in_place_checks(WAIT), "the check answered");
                let passed = PLANES
                    .iter()
                    .filter(|&&plane| answer(plane) == TexturePlacement::NotHostPages)
                    .count() as u64;
                // The next commit moves the shadows in place where the check passed.
                let plane = PLANES[0];
                assert!(read(&device, &textures[0].0, plane) == want(plane));
                let moved = device.work_counts().pooled_textures_moved_in_place
                    - counts.pooled_textures_moved_in_place;
                assert_eq!(
                    moved, passed,
                    "{export:?} {options:?}: the planes the check passed for moved in place"
                );
                first = false;
            }
            for (plane, (texture, _)) in PLANES.iter().zip(&textures) {
                let placement = answer(*plane);
                if placement != TexturePlacement::NotHostPages {
                    assert!(
                        texture.shadow().is_some(),
                        "{export:?} {options:?} {plane:?}: a shadow"
                    );
                    assert!(
                        read(&device, texture, *plane) == want(*plane),
                        "{export:?} {options:?} {plane:?}: the texels, read through the shadow"
                    );
                    eprintln!(
                        "EXPORTED-TEXTURES on {name}: {export:?} {options:?} {:?} {}x{}: a \
                         shadow, the device's check of that memory did not pass: {placement:?}",
                        plane.format, plane.size[0], plane.size[1]
                    );
                    // What the check found of that memory: its notes on the class.
                    let class = match export {
                        MemoryExport::OpaqueFd => "ExportedFd ",
                        MemoryExport::DmaBuf => "ExportedDmaBuf ",
                    };
                    let report = device.in_place_check_report();
                    for line in (report.iter().skip(crate::in_place::VERDICTS))
                        .filter(|line| line.starts_with(class))
                    {
                        eprintln!("EXPORTED-TEXTURES on {name}: the check: {line}");
                    }
                    continue;
                }
                assert!(
                    texture.shadow().is_none(),
                    "{export:?} {options:?} {plane:?}: the image in place"
                );
                assert!(
                    read(&device, texture, *plane) == want(*plane),
                    "{export:?} {options:?} {plane:?}: the texels, read in place"
                );
                let [alignment, row_pitch, offset, size, at] =
                    image_facts(&device, &picture, texture, *plane);
                eprintln!(
                    "EXPORTED-TEXTURES on {name}: {export:?} {options:?} {:?} {}x{}: in place, \
                     at {at} of the memory, alignment {alignment}, row pitch {row_pitch}, \
                     offset {offset}, size {size}",
                    plane.format, plane.size[0], plane.size[1]
                );
                assert_eq!(at % alignment, 0);
                assert_eq!((row_pitch, offset), (ROW, 0));
            }
            // Made after the check answered: in place at once where it passed, else a shadow.
            let shadows = device.work_counts().shadowed_textures;
            let mut kept = 0;
            for plane in PLANES {
                let expected = answer(plane);
                let (texture, placement) = made(plane);
                assert_eq!(placement, expected, "{plane:?}");
                if placement != TexturePlacement::NotHostPages {
                    kept += 1;
                }
                assert!(read(&device, &texture, plane) == want(plane));
            }
            assert_eq!(
                device.work_counts().shadowed_textures,
                shadows + kept,
                "{export:?} {options:?}: a shadow made after the check only where it did not pass"
            );
        }
    }
    watched.layer_reported_nothing(
        "textures over exported memory name its handle type (VUID-vkBindImageMemory-memory-02728)",
    );
}

/// The dma-buf and opaque file descriptor handle types together.
const BOTH: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::from_raw(
    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT.as_raw()
        | vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD.as_raw(),
);

/// Whether `device`'s buffer queries (`vkGetPhysicalDeviceExternalBufferProperties`, of the usage
/// its buffers are made with) answer a buffer's memory exportable as a dma-buf and as an opaque
/// file descriptor together: each exportable and compatible with the other, the opaque one not
/// only as a dedicated allocation.
fn exports_both(device: &Device) -> bool {
    let shared = device.shared();
    let usage = crate::buffer::usage(shared.device_address);
    let query = |handle| {
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(handle);
        let mut properties = vk::ExternalBufferProperties::default();
        // SAFETY: the device's physical device, of its instance at Vulkan 1.2 or later.
        unsafe {
            shared
                .instance
                .raw
                .get_physical_device_external_buffer_properties(
                    shared.physical,
                    &info,
                    &mut properties,
                )
        };
        properties.external_memory_properties
    };
    let dma_buf = query(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let opaque = query(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let exportable = vk::ExternalMemoryFeatureFlags::EXPORTABLE;
    (dma_buf.external_memory_features).contains(exportable)
        && dma_buf.compatible_handle_types.contains(BOTH)
        && (opaque.external_memory_features).contains(exportable)
        && !(opaque.external_memory_features)
            .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        && opaque.compatible_handle_types.contains(BOTH)
}

/// Whether `device` takes a dma-buf's handle type for an image of `plane`'s format and extent in a
/// shared block, with the narrowest usage a sampled texture over a buffer has (sampled, both
/// transfers): linear, or `DRM_FORMAT_MOD_LINEAR` where the device makes such images, shared among
/// its queue families as the runtime's are. Its image format query reports the type exportable
/// and not only as a dedicated allocation, and such an image, made, does not require one
/// (`VkMemoryDedicatedRequirements`: RADV requires it of every shareable image not of linear
/// tiling).
fn dma_buf_images_shared(device: &Device, plane: Plane) -> bool {
    let shared = device.shared();
    let raw_device = shared.vk();
    let format = vk_format(plane);
    let dma_buf = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let families = &shared.queue_families;
    let modifiers = [crate::dma_buf::DRM_FORMAT_MOD_LINEAR];
    let takes = |tiling: vk::ImageTiling| {
        let modifier_tiling = tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT;
        let mut external =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(dma_buf);
        let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(crate::dma_buf::DRM_FORMAT_MOD_LINEAR);
        modifier = match families.len() > 1 {
            true => modifier
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
            false => modifier.sharing_mode(vk::SharingMode::EXCLUSIVE),
        };
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(tiling)
            .usage(usage)
            .push_next(&mut external);
        if modifier_tiling {
            info = info.push_next(&mut modifier);
        }
        let mut answer = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default().push_next(&mut answer);
        // SAFETY: the device's physical device, of its instance at Vulkan 1.2 or later, which
        // enables `VK_EXT_image_drm_format_modifier` where that tiling is asked; `info` and
        // `properties` and their chains live for the call.
        let queried = unsafe {
            (shared.instance.raw).get_physical_device_image_format_properties2(
                shared.physical,
                &info,
                &mut properties,
            )
        };
        let features = answer.external_memory_properties.external_memory_features;
        if queried.is_err()
            || !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return false;
        }
        let mut handles = vk::ExternalMemoryImageCreateInfo::default().handle_types(dma_buf);
        let mut listed =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
        let [width, height] = plane.size;
        let mut image = shared
            .sharing()
            .image(
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: width as u32,
                        height: height as u32,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(tiling)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
            )
            .push_next(&mut handles);
        if modifier_tiling {
            image = image.push_next(&mut listed);
        }
        // SAFETY: the device is live; `image` and its chain are within what its format query
        // answered for such an image.
        let Ok(raw) = (unsafe { crate::creation::image(raw_device, &image) }) else {
            return false;
        };
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::ImageMemoryRequirementsInfo2::default().image(raw);
        // SAFETY: `raw` is a live image of the device; destroyed right after, with no memory
        // bound and no command using it.
        unsafe {
            raw_device.get_image_memory_requirements2(&query, &mut requirements);
            crate::release::object(raw_device, raw);
        }
        dedicated.requires_dedicated_allocation == vk::FALSE
    };
    takes(vk::ImageTiling::LINEAR)
        || (device.makes_modifier_images() && takes(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT))
}

/// `plane`'s format as Vulkan names it.
fn vk_format(plane: Plane) -> vk::Format {
    match plane.format {
        PixelFormat::R8Unorm => vk::Format::R8_UNORM,
        PixelFormat::RG8Unorm => vk::Format::R8G8_UNORM,
        format => panic!("no plane of {format:?}"),
    }
}

/// Whether `fd` is a dma-buf: its `fdinfo` names the exporter of one (`exp_name`), as the kernel
/// writes for a dma-buf alone.
#[cfg(target_os = "linux")]
fn is_dma_buf(fd: &std::os::fd::OwnedFd) -> bool {
    use std::os::fd::AsRawFd;
    let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.as_raw_fd())).unwrap();
    info.lines().any(|line| line.starts_with("exp_name:"))
}

#[test]
fn a_dma_buf_block_s_planes_name_an_opaque_fd_where_its_images_take_no_dma_buf_in_a_shared_block() {
    let watched = Watched::default();
    let scratch = Scratch::new("exported-dma-buf-opaque");
    let device = device(&watched, &scratch.0);
    let name = device.name().to_owned();
    let both = exports_both(&device);
    let options = ResourceOptions::shared();
    let picture = match picture(&device, options, MemoryExport::DmaBuf) {
        Ok(picture) => picture,
        Err(Error::NotImplemented(
            gap @ (RuntimeGap::ExportableMemory | RuntimeGap::DmaBufExport),
        )) => {
            eprintln!("EXPORTED-DMA-BUF on {name}: refused: {gap:?}");
            return;
        }
        Err(error) => panic!("{name}: {error:?}"),
    };
    let exported = picture.export_memory_dma_buf().unwrap();
    #[cfg(target_os = "linux")]
    assert!(
        is_dma_buf(&exported.fd),
        "the buffer's export is a dma-buf, whatever else its block is exportable as"
    );
    let exports = match picture.bound_memory() {
        BoundMemory::External(exports) => exports,
        BoundMemory::NoImage(no) => {
            assert!(exported.dedicated, "{no:?}");
            eprintln!("EXPORTED-DMA-BUF on {name}: a dedicated allocation, which takes no image");
            return;
        }
        BoundMemory::Own => panic!("{name}: the buffer's memory is not exported"),
    };
    let dma_buf = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    assert_eq!(
        exports.as_raw(),
        if both { BOTH } else { dma_buf }.as_raw(),
        "{name}: the block is exportable as the device's buffer queries answer (both {both})"
    );
    fill(&device, &picture);
    // The first placement query starts the device's check of that memory: its answer first.
    let first = descriptor(PLANES[0], options);
    let _ = (picture.texture_placement(&first, PLANES[0].at, ROW)).unwrap();
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let counts = device.work_counts();
    let (mut in_place, mut opaque) = (0, 0);
    for plane in PLANES {
        let shared_dma_buf = dma_buf_images_shared(&device, plane);
        let descriptor = descriptor(plane, options);
        let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
        let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
        assert!(
            read(&device, &texture, plane) == want(plane),
            "{name} {plane:?}: the texels"
        );
        let report = |placement: &TexturePlacement| {
            eprintln!(
                "EXPORTED-DMA-BUF on {name}: {:?} {}x{}: block exports {:#x}, a dma-buf image \
                 in a shared block {shared_dma_buf}: {placement:?}",
                plane.format,
                plane.size[0],
                plane.size[1],
                exports.as_raw()
            );
        };
        report(&placement);
        if both && !shared_dma_buf {
            // The image names an opaque file descriptor, which the check made too: in place, or a
            // shadow of a byte the check found changed.
            let changed = matches!(
                placement,
                TexturePlacement::Shadow(TextureShadowReason::CheckFailed { .. })
            );
            if placement != TexturePlacement::NotHostPages {
                for line in (device.in_place_check_report().iter())
                    .skip(crate::in_place::VERDICTS)
                    .filter(|line| line.starts_with("ExportedDmaBuf "))
                {
                    eprintln!("EXPORTED-DMA-BUF on {name}: the check: {line}");
                }
            }
            assert!(
                placement == TexturePlacement::NotHostPages || changed,
                "{name} {plane:?}: an image naming an opaque file descriptor, in place or a \
                 shadow of a changed byte: {placement:?}"
            );
            if placement == TexturePlacement::NotHostPages {
                opaque += 1;
            }
        }
        if placement == TexturePlacement::NotHostPages {
            in_place += 1;
            assert!(texture.shadow().is_none(), "{name} {plane:?}: in place");
            let [alignment, row_pitch, offset, size, at] =
                image_facts(&device, &picture, &texture, plane);
            eprintln!(
                "EXPORTED-DMA-BUF on {name}: {:?}: in place, at {at} of the memory, alignment \
                 {alignment}, row pitch {row_pitch}, offset {offset}, size {size}",
                plane.format
            );
            assert_eq!(at % alignment, 0);
            assert_eq!((row_pitch, offset), (ROW, 0));
        } else {
            assert!(texture.shadow().is_some(), "{name} {plane:?}: a shadow");
        }
    }
    let named = device.work_counts().exported_dma_buf_images_named_opaque_fd
        - counts.exported_dma_buf_images_named_opaque_fd;
    eprintln!(
        "EXPORTED-DMA-BUF on {name}: {in_place} planes in place, {named} naming an opaque file \
         descriptor"
    );
    // Each plane the dma-buf type takes no image of names the opaque one; one it takes may too,
    // where its query of the texture's own usage refuses what the narrowest's took.
    assert!(
        (opaque..=in_place).contains(&named),
        "{name}: {opaque} planes name an opaque file descriptor at least, {in_place} at most"
    );
    if !both {
        assert_eq!(named, 0, "{name}: a block exportable as a dma-buf alone");
    }
    watched.layer_reported_nothing(
        "images over a dma-buf block name a handle type it is exportable as \
         (VUID-vkBindImageMemory-memory-02728)",
    );
}
