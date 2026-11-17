//! Textures over the runtime's exported memory (`exported_memory.rs`), shaped as a consumer's
//! cache of a decoded 3840x2160 NV12 picture: one exportable buffer, its luma plane (`R8Unorm`
//! 3840x2160) 32 bytes in and its chroma plane (`RG8Unorm` 1920x1080) right after it, rows 3840
//! bytes apart with no bytes between them, the buffer laying byte 32 on a multiple of 256 of its
//! memory (`aligned_byte`), so each plane lies at a multiple of 256 there.
//!
//! - **The image names the memory's handle type.** Each plane's texture is an image bound in the
//!   buffer's exported memory, which must name the memory's handle type
//!   (`VUID-vkBindImageMemory-memory-02728`); the validation layer, enabled on the device, reports
//!   nothing ([`Watched::layer_reported_nothing`]): without the handle type it reports 02728 at the
//!   bind.
//! - **A shadow first, then in place.** Made before the device's check of that exported memory
//!   answered, each texture is a shadow (`TexturePlacement::Pending`), whose blit reads the
//!   buffer's bytes; once the check answered (`Device::settle_in_place_checks`) and passed, the
//!   next commit moves both in place (`WorkCounts::pooled_textures_moved_in_place`), and their
//!   blits read the same bytes through the image in the buffer's memory. A texture made after is
//!   in place at once (`TexturePlacement::NotHostPages`, no shadow counted). Where the check did
//!   not pass for that memory (the RX 550 under RADV makes no image over the runtime's memory
//!   exported as a dma-buf: none in the memory, and no `DRM_FORMAT_MOD_LINEAR` image to import it
//!   into), each texture stays the shadow its verdict decided (`TextureShadowReason::Unchecked`
//!   or `CheckFailed`), and reads the same bytes.
//! - **A dma-buf's own import.** Where the device takes no image of a plane in the memory exported
//!   as a dma-buf (its format query takes the dma-buf handle type for no such linear image, and
//!   its `DRM_FORMAT_MOD_LINEAR` image requires a dedicated allocation: RADV on the 780M, asked
//!   here apart from the runtime), the plane's image is a `DRM_FORMAT_MOD_LINEAR` one at the
//!   texels' offset, bound to a dedicated import of that memory's own dma-buf: in place where the
//!   check passed, which made the same image, counted (`WorkCounts::exported_dma_buf_images`),
//!   never a shadow the check made no image for where the device makes such images; the import
//!   goes with the texture (`WorkCounts::dma_buf_image_memories_live`), and the descriptor the
//!   buffer exports is a dma-buf of its memory as before.
//! - **Each handle type, each storage.** Of an opaque file descriptor and of a dma-buf where the
//!   device exports one, `Private` (its bytes written by a blit, as a decoder writes a cache the CPU
//!   never maps) and `Shared` (written through its contents). Where the device exports an opaque
//!   file descriptor only as the buffer's dedicated allocation, to which no image may be bound
//!   (`VUID-vkBindImageMemory-memory-10926`), each texture is a shadow that reads the same bytes,
//!   its placement refused (`InPlaceRefusal::Placement`); a dma-buf's dedicated allocation takes
//!   the import of its own dma-buf (below) where the device makes one.
//! - **The runtime's own buffers' blocks.** The same picture in a buffer the program made with no
//!   export, keeping its padding. One made before the device's in-place check answered lies in a
//!   plain block. Once it answered, where the device allocates its buffers' blocks exportable and
//!   the check passed them (`memory.rs`'s "Buffers in exportable blocks", `in_place.rs`'s
//!   `blocks_take`), its memory is such a block (`WorkCounts::fd_buffer_blocks_made`), which
//!   the runtime never exports (`export_memory_fd` refuses it, `NotExportable`), and each plane is
//!   in place, its image the device's linear image naming the block's handle type, an opaque file
//!   descriptor, the layer reporting nothing (`VUID-vkBindImageMemory-memory-02728`,
//!   `VUID-VkExportMemoryAllocateInfo-handleTypes-09860`); elsewhere its memory is a plain block,
//!   each plane in place or the shadow the check decided. Each reads the same bytes. One that
//!   declares its padding unobserved, as the consumer's caches do, lies in a block of its own kind
//!   where the device allocates those blocks exportable as dma-bufs as well (`memory.rs`'s
//!   `Holds::FdDmaBufs`), in which no image is bound: each plane there is in place, a
//!   `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated import of the block's dma-buf
//!   (`WorkCounts::exported_dma_buf_images`), no shadow counted and no notice, and reads the same
//!   bytes; elsewhere it lies in a plain block. A second device of
//!   the driver, its shader cache the first's, reads the verdicts from the check's record as it is
//!   made, and lays its first buffer where they say, none waiting for the check.
//!
//! Printed (`EXPORTED-TEXTURES`): what the device exported, and for each plane its image's
//! alignment, row pitch and size as `vkGetImageMemoryRequirements` and `vkGetImageSubresourceLayout`
//! answer them, and its offset in the memory.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::layer_claim::Watched;
use super::pending_rebind::device;
use crate::error::{Error, Invalid, Problem, RuntimeGap};
use crate::memory::BoundMemory;
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, InPlaceRefusal, MemoryExport,
    Origin, ResourceOptions, Size, Texture, TextureDescriptor, TexturePlacement,
    TextureShadowReason,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
/// `plane`'s texels, or where `imported` a `DRM_FORMAT_MOD_LINEAR` one bound to a dedicated import
/// of that memory's own dma-buf, its plane at the texels' offset: its alignment, its first
/// subresource's (or memory plane's) row pitch and offset, its size, and the texels' offset in the
/// memory.
fn image_facts(
    device: &Device,
    (picture, texture): (&Buffer, &Texture),
    plane: Plane,
    imported: bool,
) -> [u64; 5] {
    // SAFETY: the image is the texture's, used only to ask of it while the texture lives.
    let image = unsafe { texture.vk_image() }.expect("an image in place");
    let raw = device.shared().vk();
    // A DRM format modifier's image is asked of its memory plane
    // (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
    let aspect_mask = match imported {
        true => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        false => vk::ImageAspectFlags::COLOR,
    };
    let subresource = vk::ImageSubresource {
        aspect_mask,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: a live image of the device, of linear or DRM format modifier tiling as `imported`
    // says.
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
            if exported.dedicated && export == MemoryExport::OpaqueFd {
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
                // A dma-buf's dedicated allocation where the device makes no image to import its
                // dma-buf into stays the shadow its placement decided.
                let refused = TextureShadowReason::Refused(InPlaceRefusal::Placement);
                assert!(
                    matches!(
                        placement,
                        TexturePlacement::NotHostPages
                            | TexturePlacement::Shadow(
                                TextureShadowReason::Unchecked { .. }
                                    | TextureShadowReason::CheckFailed { .. }
                            )
                    ) || (exported.dedicated && placement == TexturePlacement::Shadow(refused)),
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
                // Over a dma-buf, an image the device takes nowhere in the memory is the import's.
                let imported = export == MemoryExport::DmaBuf
                    && (exported.dedicated || !dma_buf_images_shared(&device, *plane));
                let [alignment, row_pitch, offset, size, at] =
                    image_facts(&device, (&picture, texture), *plane, imported);
                eprintln!(
                    "EXPORTED-TEXTURES on {name}: {export:?} {options:?} {:?} {}x{}: in place, \
                     at {at} of the memory, alignment {alignment}, row pitch {row_pitch}, \
                     offset {offset}, size {size}, through the memory's own dma-buf {imported}",
                    plane.format, plane.size[0], plane.size[1]
                );
                match imported {
                    // The plane lies at the texels' offset of the import, bound at its start.
                    true => assert_eq!((row_pitch, offset), (ROW, at)),
                    false => {
                        assert_eq!(at % alignment, 0);
                        assert_eq!((row_pitch, offset), (ROW, 0));
                    }
                }
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

/// Whether `device` takes a dma-buf's handle type for an image of `plane`'s format and extent in
/// the memory itself, with the narrowest usage a sampled texture over a buffer has (sampled, both
/// transfers): linear, or `DRM_FORMAT_MOD_LINEAR` where the device makes such images, shared among
/// its queue families as the runtime's are. Its image format query reports the type exportable
/// and not only as a dedicated allocation, and such an image, made, does not require one
/// (`VkMemoryDedicatedRequirements`: RADV requires it of every shareable image not of linear
/// tiling). Asked here apart from the runtime.
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
fn a_dma_buf_export_s_planes_go_in_place_through_its_own_dma_buf_where_no_image_binds_in_it() {
    let watched = Watched::default();
    let scratch = Scratch::new("exported-dma-buf-import");
    let device = device(&watched, &scratch.0);
    let name = device.name().to_owned();
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
        "{name}: the buffer's export is a dma-buf"
    );
    fill(&device, &picture);
    // The first placement query starts the device's check of that memory: its answer first.
    let first = descriptor(PLANES[0], options);
    let _ = (picture.texture_placement(&first, PLANES[0].at, ROW)).unwrap();
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let imports_before = device.work_counts().dma_buf_image_memories_live;
    let made_before = device.work_counts().exported_dma_buf_images;
    let makes = device.makes_modifier_images();
    let mut textures = Vec::new();
    let mut imported = 0;
    for plane in PLANES {
        let in_memory = !exported.dedicated && dma_buf_images_shared(&device, plane);
        let descriptor = descriptor(plane, options);
        let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
        let counted = device.work_counts().exported_dma_buf_images;
        let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
        let this = device.work_counts().exported_dma_buf_images - counted;
        assert!(
            read(&device, &texture, plane) == want(plane),
            "{name} {plane:?}: the texels"
        );
        eprintln!(
            "EXPORTED-DMA-BUF on {name}: {:?} {}x{}: dedicated {}, an image in the memory {}, \
             DRM_FORMAT_MOD_LINEAR images {makes}: {placement:?}, through the memory's own \
             dma-buf {this}",
            plane.format, plane.size[0], plane.size[1], exported.dedicated, in_memory
        );
        if placement != TexturePlacement::NotHostPages {
            for line in (device.in_place_check_report().iter())
                .skip(crate::in_place::VERDICTS)
                .filter(|line| line.starts_with("ExportedDmaBuf "))
            {
                eprintln!("EXPORTED-DMA-BUF on {name}: the check: {line}");
            }
            assert!(texture.shadow().is_some(), "{name} {plane:?}: a shadow");
        }
        if !in_memory && makes {
            // The import's image, which the check made too: in place, or a shadow of a byte the
            // check found changed.
            let changed = matches!(
                placement,
                TexturePlacement::Shadow(TextureShadowReason::CheckFailed { .. })
            );
            assert!(
                placement == TexturePlacement::NotHostPages || changed,
                "{name} {plane:?}: an image through the memory's own dma-buf, in place or a \
                 shadow of a changed byte: {placement:?}"
            );
        }
        if placement == TexturePlacement::NotHostPages {
            assert!(texture.shadow().is_none(), "{name} {plane:?}: in place");
            assert_eq!(
                this,
                u64::from(!in_memory),
                "{name} {plane:?}: counted as its route"
            );
            let [alignment, row_pitch, offset, size, at] =
                image_facts(&device, (&picture, &texture), plane, !in_memory);
            eprintln!(
                "EXPORTED-DMA-BUF on {name}: {:?}: in place, at {at} of the memory, alignment \
                 {alignment}, row pitch {row_pitch}, offset {offset}, size {size}",
                plane.format
            );
            match in_memory {
                true => assert_eq!((at % alignment, row_pitch, offset), (0, ROW, 0)),
                false => assert_eq!((row_pitch, offset), (ROW, at)),
            }
            imported += this;
        } else {
            assert_eq!(this, 0, "{name} {plane:?}: a shadow makes no import");
        }
        textures.push(texture);
    }
    let made = device.work_counts().exported_dma_buf_images - made_before;
    let live = device.work_counts().dma_buf_image_memories_live - imports_before;
    eprintln!(
        "EXPORTED-DMA-BUF on {name}: {made} images through the memory's own dma-buf, {live} \
         imports live"
    );
    assert_eq!(
        (made, live),
        (imported, imported),
        "{name}: one import per such image"
    );
    // The textures' teardown leaves no import behind: each goes with its image.
    drop(textures);
    let began = std::time::Instant::now();
    while device.work_counts().dma_buf_image_memories_live != imports_before {
        assert!(
            began.elapsed() < WAIT,
            "{name}: the imports outlive their textures: {} live, {imports_before} before",
            device.work_counts().dma_buf_image_memories_live
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    watched.layer_reported_nothing(
        "images over a dma-buf export, in its memory or through its own dma-buf",
    );
}

#[test]
fn an_nv12_picture_s_planes_over_a_buffer_of_the_runtime_s_own_name_its_block_s_handle_type() {
    let watched = Watched::default();
    let scratch = Scratch::new("own-buffer-textures");
    let device = device(&watched, &scratch.0);
    let name = device.name().to_owned();
    let types = crate::device::lock(&device.shared().memory).fd_buffer_types;
    eprintln!("EXPORTED-TEXTURES on {name}: the blocks of the runtime's own buffers: {types:?}");
    // Its planes' rows lie one after another: the same images whatever its padding.
    let descriptor_of = |options| {
        BufferDescriptor::new(LENGTH, options, BufferContents::Zero).with_aligned_byte(LUMA_AT)
    };
    // Made before the device's check answered: a plain block, whatever the check answers.
    let early =
        (device.new_buffer_with_descriptor(descriptor_of(ResourceOptions::shared()))).unwrap();
    let verdicts_known = device.settle_in_place_checks(Duration::ZERO);
    // A `Shared` buffer lies in a block of the runtime's own pages where the device takes them
    // (`memory.rs`'s `Holds::PageBuffers`), whatever the check of exportable blocks answers.
    let host_pages = BoundMemory::External(crate::memory::HOST_PAGES_HANDLE);
    if !verdicts_known && !early.in_page_block() {
        assert!(
            early.bound_memory() == BoundMemory::Own,
            "made while the check was pending"
        );
    }
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let verdict = |class, access, rows| device.in_place_verdict(class, access, rows);
    let take = crate::in_place::blocks_take(crate::in_place::PagesClass::FdBlocks, verdict);
    let exportable = types.is_ok() && take == crate::in_place::BlocksTake::Take;
    eprintln!(
        "EXPORTED-TEXTURES on {name}: the check of exportable blocks: {take:?} (known at once \
         {verdicts_known}); awaited {}, refused {}",
        device.work_counts().fd_blocks_awaited,
        device.work_counts().fd_blocks_refused
    );
    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
        let picture = device
            .new_buffer_with_descriptor(descriptor_of(options))
            .unwrap();
        let (bound, counts) = (picture.bound_memory(), device.work_counts());
        let paged = picture.in_page_block();
        match exportable {
            _ if paged => assert!(bound == host_pages, "{options:?}: {bound:?}"),
            true => {
                assert!(bound == BoundMemory::Blocks, "{options:?}: {bound:?}");
                assert!(
                    counts.fd_buffer_blocks_made > 0,
                    "{options:?}: such blocks made"
                );
                assert!(
                    matches!(
                        picture.export_memory_fd(),
                        Err(Error::Invalid(Invalid {
                            problem: Problem::NotExportable,
                            ..
                        }))
                    ),
                    "{options:?}: the runtime exports no such buffer"
                );
            }
            false => {
                assert!(
                    bound == BoundMemory::Own,
                    "{options:?}: a plain block, {bound:?}"
                );
                if types.is_err() {
                    let made = (counts.fd_buffer_blocks_made, counts.fd_buffer_blocks_plain);
                    assert_eq!(made, (0, 0), "{options:?}: no block of such buffers");
                    let asked = (counts.fd_blocks_awaited, counts.fd_blocks_refused);
                    assert_eq!(asked, (0, 0), "{options:?}: no check of such blocks asked");
                }
            }
        }
        fill(&device, &picture);
        for plane in PLANES {
            let descriptor = descriptor(plane, options);
            let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
            let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
            // Over a block of the runtime's own pages, as over fresh pages: in place where the
            // check passed for them (`texture/route.rs`).
            let in_place = match paged {
                true => TexturePlacement::InPlace,
                false => TexturePlacement::NotHostPages,
            };
            assert!(
                placement == in_place
                    || matches!(
                        placement,
                        TexturePlacement::Shadow(
                            TextureShadowReason::Unchecked { .. }
                                | TextureShadowReason::CheckFailed { .. }
                        )
                    ),
                "{options:?} {plane:?}: in place, or a shadow the device's check decided: \
                 {placement:?}"
            );
            // Where the buffers lie in such blocks, every plane is in place as over a plain
            // block's buffer: the device's linear image made for the blocks' handle type.
            if exportable && !paged {
                assert_eq!(
                    placement,
                    TexturePlacement::NotHostPages,
                    "{options:?} {plane:?}"
                );
            }
            assert_eq!(texture.shadow().is_none(), placement == in_place);
            assert!(
                read(&device, &texture, plane) == want(plane),
                "{options:?} {plane:?}: the texels, read as made"
            );
            eprintln!(
                "EXPORTED-TEXTURES on {name}: a buffer of the runtime's own, {options:?} {:?} \
                 {}x{}: {placement:?}",
                plane.format, plane.size[0], plane.size[1]
            );
        }
    }
    // A picture that declares its padding unobserved lies in a block of its own kind exportable as
    // a dma-buf too where the device takes such blocks (`memory.rs`'s `Memory::unobserved_buffers`),
    // else in a plain block; one that keeps its padding lies where the others do.
    let co_exported = crate::device::lock(&device.shared().memory).unobserved_buffers(exportable)
        == crate::memory::Holds::FdDmaBufs;
    let before = device.work_counts().object_churn;
    let unobserved = (device.new_buffer_with_descriptor(
        descriptor_of(ResourceOptions::shared()).with_padding_unobserved(),
    ))
    .unwrap();
    let kept =
        (device.new_buffer_with_descriptor(descriptor_of(ResourceOptions::shared()))).unwrap();
    let after = device.work_counts().object_churn;
    let (unobserved_at, kept_at) = (unobserved.bound_memory(), kept.bound_memory());
    eprintln!(
        "EXPORTED-TEXTURES on {name}: a picture that declares its padding unobserved lies in \
         {unobserved_at:?} (blocks exportable as dma-bufs too {co_exported}), one that keeps it \
         in {kept_at:?}; buffers parked before them {}, taken by them {}",
        before.buffers_parked,
        after.buffers_recycled - before.buffers_recycled
    );
    match exportable && co_exported {
        true => assert!(
            unobserved_at == BoundMemory::DmaBufBlocks,
            "{unobserved_at:?}"
        ),
        false => assert!(unobserved_at == BoundMemory::Own, "{unobserved_at:?}"),
    }
    match exportable {
        _ if kept.in_page_block() => assert!(kept_at == host_pages, "{kept_at:?}"),
        true => assert!(kept_at == BoundMemory::Blocks, "{kept_at:?}"),
        false => assert!(kept_at == BoundMemory::Own, "{kept_at:?}"),
    }
    // Over the picture in such a block, each plane in place through a dedicated import of the
    // block's dma-buf, reading the bytes written.
    if unobserved_at == BoundMemory::DmaBufBlocks {
        fill(&device, &unobserved);
        for plane in PLANES {
            let descriptor = descriptor(plane, ResourceOptions::shared());
            let before = device.work_counts();
            let texture = (unobserved.new_texture(&descriptor, plane.at, ROW)).unwrap();
            let after = device.work_counts();
            eprintln!(
                "EXPORTED-TEXTURES on {name}: a picture that declares its padding unobserved, \
                 {:?} {}x{}: images over a dedicated import of its block's dma-buf {}, shadows {}",
                plane.format,
                plane.size[0],
                plane.size[1],
                after.exported_dma_buf_images - before.exported_dma_buf_images,
                after.shadowed_textures - before.shadowed_textures
            );
            assert!(texture.shadow().is_none(), "{plane:?}: in place");
            assert_eq!(
                after.exported_dma_buf_images - before.exported_dma_buf_images,
                1,
                "{plane:?}: its image over the block's dma-buf"
            );
            assert_eq!(
                refusals(&after),
                refusals(&before),
                "{plane:?}: no refusal counted"
            );
            assert!(
                read(&device, &texture, plane) == want(plane),
                "{plane:?}: the texels, read through the image"
            );
        }
    }
    watched.layer_reported_nothing(
        "buffers and textures in blocks allocated exportable name their handle type, which the \
         image format query takes for each image there, and no image is bound in a block \
         exportable as a dma-buf too (VUID-vkBindBufferMemory-memory-02726, \
         VUID-vkBindImageMemory-memory-02728, VUID-VkExportMemoryAllocateInfo-handleTypes-09860)",
    );
}

/// A device of a driver whose in-place check ran before reads its verdicts from the check's record
/// as it is made: its first buffer lies in a block allocated exportable where its check passed
/// such blocks, none waiting for the check (`WorkCounts::fd_blocks_awaited`), and its planes are
/// in place as the first device's were.
#[test]
fn a_driver_s_second_device_lays_its_first_buffer_where_the_record_says() {
    let watched = Watched::default();
    let scratch = Scratch::new("own-buffer-record");
    let take = {
        let first = device(&watched, &scratch.0);
        assert!(
            first.settle_in_place_checks(WAIT),
            "the first device's check answered"
        );
        let verdict = |class, access, rows| first.in_place_verdict(class, access, rows);
        let take = crate::in_place::blocks_take(crate::in_place::PagesClass::FdBlocks, verdict);
        let types = crate::device::lock(&first.shared().memory).fd_buffer_types;
        types.is_ok() && take == crate::in_place::BlocksTake::Take
    };
    let second = device(&watched, &scratch.0);
    let name = second.name().to_owned();
    assert!(
        second.settle_in_place_checks(Duration::ZERO),
        "known from the record at once"
    );
    let options = ResourceOptions::private();
    let picture = (second.new_buffer_with_descriptor(
        BufferDescriptor::new(LENGTH, options, BufferContents::Zero).with_aligned_byte(LUMA_AT),
    ))
    .unwrap();
    let counts = second.work_counts();
    eprintln!(
        "EXPORTED-TEXTURES on {name}: the second device's first buffer: {:?}; exportable blocks \
         made {}, awaited {}, refused {}",
        picture.bound_memory(),
        counts.fd_buffer_blocks_made,
        counts.fd_blocks_awaited,
        counts.fd_blocks_refused
    );
    assert_eq!(
        counts.fd_blocks_awaited, 0,
        "no buffer waited for the check"
    );
    match take {
        true => assert!(picture.bound_memory() == BoundMemory::Blocks),
        false => assert!(picture.bound_memory() == BoundMemory::Own),
    }
    fill(&second, &picture);
    for plane in PLANES {
        let descriptor = descriptor(plane, options);
        let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
        let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
        if take {
            assert_eq!(placement, TexturePlacement::NotHostPages, "{plane:?}");
        }
        assert!(
            read(&second, &texture, plane) == want(plane),
            "{plane:?}: the texels"
        );
    }
    watched.layer_reported_nothing("a second device's buffers and textures");
}

/// No texture over a buffer of the program's loses its place to the blocks allocated exportable
/// (`memory.rs`'s "Buffers in exportable blocks"): for every pixel format a buffer backs
/// (`linear_alignment::backed_formats`), at each width of `linear_alignment::WIDTHS`, rows the
/// width's bytes rounded up to 16 (Metal's rule), to the format's linear alignment answer and to
/// 256 apart, 256 bytes more than that and twice it, three rows at offset 0 and at that answer and
/// four at offset 0, of `ShaderRead`, `ShaderRead | ShaderWrite` and `ShaderRead | RenderTarget`
/// usage, a texture over a `Shared` buffer the program makes (`new_buffer_with_descriptor`) lies
/// in place wherever one over a plain block's buffer of the same layout and padding does
/// (`Buffer::texture_placement`): over a buffer that keeps its padding, which lies in such a block
/// where the device lays its buffers there (`in_place.rs`'s `blocks_take`, which
/// `texture/fd_blocks_cover.rs` gates), and over one that declares it unobserved, which lies in a
/// block of its own kind exportable as a dma-buf too where the device takes such blocks, in which
/// no image is bound, else in a plain block (`buffer.rs`'s `NewBufferOf::Unobserved`, `memory.rs`'s
/// `Holds::FdDmaBufs`). A texture that loses its place fails. For each, the losses a buffer
/// of the same padding in such a block would see are printed, and those the images alone make,
/// asked as the device's check asks them (its verdicts aside), with the notice of the first few
/// such textures.
///
/// Where the declared buffer lies in a block exportable as a dma-buf too, the first texture that
/// takes a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch over a plain block's buffer (where any
/// does), and the first that takes the device's linear image there, are made over it too: each
/// image is bound to a dedicated import of the block's dma-buf (`texture.rs`'s
/// `exported_dma_buf_alias`, `WorkCounts::exported_dma_buf_images`), no shadow, no refusal counted
/// and no notice, and a blit reads the texels the CPU wrote at the pitch through it. The validation
/// layer, enabled on the device, reports nothing (`VUID-VkExportMemoryAllocateInfo-handleTypes-09860`,
/// `VUID-vkBindBufferMemory-memory-02726`, `VUID-vkBindImageMemory-memory-02728`).
///
/// Printed (`EXPORTED-TEXTURES`): the verdicts' and the tiling's answers and how long the cover
/// took, the textures tried, where each buffer lies, and each kind of loss.
#[test]
fn every_texture_in_place_over_a_plain_block_is_in_place_over_an_exportable_one() {
    use crate::memory::Holds;
    use crate::texture::linear_alignment::{backed_formats, WIDTHS};
    let watched = Watched::default();
    let scratch = Scratch::new("exportable-placements");
    // Each shadow's notice printed while `noticed` is set: the first few lost by the images. Every
    // notice counted.
    let noticed = Arc::new(AtomicBool::new(false));
    let notices = Arc::new(AtomicU64::new(0));
    let device = {
        let noticed = Arc::clone(&noticed);
        let notices = Arc::clone(&notices);
        let print = move |message: &crate::DebugMessage| {
            if message.degradation.is_some() {
                notices.fetch_add(1, Ordering::Relaxed);
            }
            if message.degradation.is_some() && noticed.load(Ordering::Relaxed) {
                eprintln!("EXPORTED-TEXTURES notice: {}", message.text);
            }
        };
        let options = crate::DeviceOptions {
            shader_cache: crate::ShaderCacheOptions {
                directory: Some(scratch.0.clone()),
                ..crate::ShaderCacheOptions::default()
            },
            layers: watched.layers(),
            messages: Some(watched.sink(print)),
            ..super::pending_rebind::options()
        };
        Device::system_default_with(&options).unwrap()
    };
    let name = device.name().to_owned();
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    let types = crate::device::lock(&device.shared().memory).fd_buffer_types;
    if let Err(why) = types {
        eprintln!(
            "EXPORTED-TEXTURES on {name}: no block of the runtime's buffers exportable: {why}"
        );
        return;
    }
    let verdict = |class, access, rows| device.in_place_verdict(class, access, rows);
    let take = crate::in_place::blocks_take(crate::in_place::PagesClass::FdBlocks, verdict);
    let asked = std::time::Instant::now();
    let covered = device.fd_blocks_cover();
    let asked = asked.elapsed();
    eprintln!(
        "EXPORTED-TEXTURES on {name}: the verdicts of exportable blocks: {take:?}; such a block \
         holds every image a plain one does: {covered:?} (asked in {asked:?})"
    );
    let blocks = take == crate::in_place::BlocksTake::Take && covered.is_ok();
    // The buffers that declare their padding unobserved lie in blocks of their own exportable as
    // dma-bufs too where the device takes them (`memory.rs`'s `Memory::unobserved_buffers`).
    let co_exported =
        crate::device::lock(&device.shared().memory).unobserved_buffers(blocks) == Holds::FdDmaBufs;
    eprintln!(
        "EXPORTED-TEXTURES on {name}: blocks exportable as dma-bufs too for the buffers that \
         declare their padding unobserved: {co_exported}"
    );
    let usages = [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    ];
    let mut layouts = Vec::new();
    for pf in backed_formats() {
        // A format of no whole texel per pixel (a packed 4:2:2 one) is left out.
        let Some(texel) = crate::buffer_texture::texel_bytes(pf) else {
            continue;
        };
        let align = device.linear_texture_alignment(pf);
        for width in WIDTHS {
            let packed = width * texel;
            let mut rows = vec![
                packed.next_multiple_of(16),
                packed.next_multiple_of(align),
                packed.next_multiple_of(256),
                packed.next_multiple_of(256) + 256,
                packed.next_multiple_of(256) * 2,
            ];
            rows.sort_unstable();
            rows.dedup();
            for row in rows {
                for (offset, height) in [(0, 3), (align, 3), (0, 4)] {
                    for usage in usages {
                        let mut descriptor =
                            TextureDescriptor::texture_2d(pf, width, height, false);
                        descriptor.usage = usage;
                        descriptor.resource_options = ResourceOptions::shared();
                        layouts.push((descriptor, offset, row));
                    }
                }
            }
        }
    }
    // Over buffers that keep the bytes between rows, and over buffers that declare them
    // unobserved (`BufferDescriptor::padding_unobserved`), as a consumer's caches do.
    for unobserved in [false, true] {
        let padding = match unobserved {
            true => "padding unobserved",
            false => "padding kept",
        };
        let length = 1 << 20;
        let plain = (device.pooled_check_buffer(length, Holds::Buffers)).unwrap();
        let exportable = (device.pooled_check_buffer(length, Holds::FdBuffers)).unwrap();
        assert!(exportable.bound_memory() == BoundMemory::Blocks);
        if unobserved {
            plain.declare_padding_unobserved();
            exportable.declare_padding_unobserved();
        }
        let descriptor =
            BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero);
        let descriptor = match unobserved {
            true => descriptor.with_padding_unobserved(),
            false => descriptor,
        };
        let program = device.new_buffer_with_descriptor(descriptor).unwrap();
        let lies = program.bound_memory();
        eprintln!("EXPORTED-TEXTURES on {name}, {padding}: the program's buffer lies in {lies:?}");
        let expected = match (blocks, unobserved) {
            (true, false) => BoundMemory::Blocks,
            (true, true) if co_exported => BoundMemory::DmaBufBlocks,
            (true, true) | (false, _) => BoundMemory::Own,
        };
        assert!(lies == expected, "{padding}: {lies:?}");
        let in_place = |placement: &Result<TexturePlacement, Error>| {
            matches!(placement, Ok(TexturePlacement::NotHostPages))
        };
        let plain_at: Vec<bool> = (layouts.iter())
            .map(|(d, offset, row)| in_place(&plain.texture_placement(d, *offset, *row)))
            .collect();
        // Why the first few textures of rows past the device's own pitch are shadows over a plain
        // block's buffer, their notices printed.
        noticed.store(true, Ordering::Relaxed);
        let past = (layouts.iter().enumerate())
            .filter(|&(at, (d, _, row))| {
                let texel = crate::buffer_texture::texel_bytes(d.pixel_format).unwrap_or(1);
                !plain_at[at] && *row > (d.width * texel).next_multiple_of(256)
            })
            .take(3);
        for (_, (d, offset, row)) in past {
            drop(plain.new_texture(d, *offset, *row));
        }
        noticed.store(false, Ordering::Relaxed);
        let lost = |buffer: &Buffer| -> Vec<usize> {
            (layouts.iter().enumerate())
                .filter(|&(at, (d, offset, row))| {
                    plain_at[at] && !in_place(&buffer.texture_placement(d, *offset, *row))
                })
                .map(|(at, _)| at)
                .collect()
        };
        let product = lost(&program);
        let over_blocks = lost(&exportable);
        // Of the textures in place over a plain block's buffer, those whose image there is a
        // DRM_FORMAT_MOD_LINEAR one at Metal's pitch (`WorkCounts::pooled_pitched_images`),
        // which an exportable block takes none of.
        let mut pitched = Vec::new();
        for (at, (d, offset, row)) in layouts.iter().enumerate() {
            if !plain_at[at] {
                continue;
            }
            let before = device.work_counts().pooled_pitched_images;
            drop(plain.new_texture(d, *offset, *row));
            if device.work_counts().pooled_pitched_images > before {
                pitched.push(at);
            }
        }
        // As the device's check asks: the images alone, whatever its verdicts.
        let images = std::thread::scope(|scope| {
            let thread = scope.spawn(|| {
                crate::in_place::mark_checking();
                let lost = lost(&exportable);
                noticed.store(true, Ordering::Relaxed);
                for &at in lost.iter().take(3) {
                    let (d, offset, row) = &layouts[at];
                    drop(exportable.new_texture(d, *offset, *row));
                }
                noticed.store(false, Ordering::Relaxed);
                lost
            });
            thread.join().unwrap()
        });
        let in_place_plain = plain_at.iter().filter(|&&at| at).count();
        eprintln!(
            "EXPORTED-TEXTURES on {name}, {padding}: {} textures tried, {in_place_plain} in place \
             over a plain block's buffer; over the program's buffer, {} of those lose their \
             place; over an exportable block's, {}, {} by the images alone",
            layouts.len(),
            product.len(),
            over_blocks.len(),
            images.len()
        );
        eprintln!(
            "EXPORTED-TEXTURES on {name}, {padding}: of those in place over a plain block's \
             buffer, {} take a DRM_FORMAT_MOD_LINEAR image at Metal's pitch, the rest the \
             device's linear image",
            pitched.len()
        );
        for &at in pitched.iter().take(8) {
            let (d, offset, row) = &layouts[at];
            eprintln!(
                "EXPORTED-TEXTURES on {name}, {padding}: pitched over a plain block: {:?} \
                 {}x{} at {offset}, rows {row} apart, {:?}",
                d.pixel_format, d.width, d.height, d.usage
            );
        }
        let mut kinds = std::collections::BTreeMap::new();
        for &at in &images {
            let (d, offset, row) = &layouts[at];
            let placement = exportable.texture_placement(d, *offset, *row);
            let key = format!("{:?} {:?}", d.pixel_format, placement);
            *kinds.entry(key).or_insert(0) += 1;
        }
        for (kind, count) in kinds {
            eprintln!(
                "EXPORTED-TEXTURES on {name}, {padding}: lost by the images: {count} of {kind}"
            );
        }
        assert!(
            product.is_empty(),
            "{padding}: textures that lose their place: {:?}",
            (product.iter().take(8))
                .map(|&at| &layouts[at])
                .map(|(d, offset, row)| (d.pixel_format, d.width, d.height, offset, row, d.usage))
                .collect::<Vec<_>>()
        );
        // Over the declared buffer in a block exportable as a dma-buf too, a texture whose image
        // over a plain block's is a DRM_FORMAT_MOD_LINEAR one at Metal's pitch, and one whose image
        // there is the device's linear image: each bound to a dedicated import of the block's
        // dma-buf, and reading what the CPU wrote.
        if lies == BoundMemory::DmaBufBlocks {
            let linear = (0..layouts.len()).find(|at| plain_at[*at] && !pitched.contains(at));
            for (kind, at) in [("pitched", pitched.first().copied()), ("linear", linear)] {
                match at {
                    Some(at) => over_a_dma_buf_block(&device, &program, &layouts[at], &notices),
                    None => eprintln!(
                        "EXPORTED-TEXTURES on {name}: no texture took a {kind} image over a plain \
                         block's buffer, so none is made over the block"
                    ),
                }
            }
        }
    }
    watched.layer_reported_nothing(
        "buffers in blocks exportable as opaque file descriptors, linear images bound there naming \
         that handle type, buffers in blocks exportable as dma-bufs too, in which no image is \
         bound, and DRM_FORMAT_MOD_LINEAR images over a dedicated import of such a block's dma-buf \
         (VUID-VkExportMemoryAllocateInfo-handleTypes-09860, VUID-vkBindBufferMemory-memory-02726, \
         VUID-vkBindImageMemory-memory-02728)",
    );
}

/// A plane of the picture's luma, 64 of its rows: each texture of the parking test reads it.
#[cfg(target_os = "linux")]
const PARK_PLANE: Plane = Plane {
    format: PixelFormat::R8Unorm,
    size: [ROW, 64],
    at: LUMA_AT,
};

/// The device's counts once `until` holds of them, within [`WAIT`].
#[cfg(target_os = "linux")]
fn counts_until(device: &Device, until: impl Fn(&crate::WorkCounts) -> bool) -> crate::WorkCounts {
    let began = std::time::Instant::now();
    loop {
        let counts = device.work_counts();
        if until(&counts) {
            return counts;
        }
        assert!(began.elapsed() < WAIT, "{counts:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Textures over a picture that declares its padding unobserved, in a block of the runtime's own
/// buffers exportable as a dma-buf too (`memory.rs`'s `Holds::FdDmaBufs`), as a consumer's
/// surface caches lie: each in place through a dedicated import of the block's dma-buf, whose
/// image parks under the buffer's pages as the texture goes (`dma_buf/parked.rs`, "Over the
/// runtime's exported memory").
///
/// - **Taken.** The first texture's image, made, read and dropped, parks; the CPU writes other
///   bytes; a second texture of the same plan at the same texels takes it: one image made for the
///   two (`WorkCounts::exported_dma_buf_images`), one taken (`exported_dma_buf_images_taken`),
///   and its blit reads the new bytes. A third, made while the second lives, finds none parked
///   and is made anew, reading the same bytes.
/// - **Not parked** where the texture's `VkImage` was handed out (`Texture::vk_image`): its image
///   goes with its import.
/// - **Released** untaken: kept by a residency pass within its window where held imports cost
///   nothing, released by the residency thread's next pass where they cost every submission
///   (`exported_dma_buf_images_parked_released`), each destruction in a section of the exit
///   guard, without which a test build ends the process there; and, parked as the buffer goes,
///   with the buffer's pages, which leaves the device's dma-buf image imports as they were
///   before.
///
/// The import tax is forced to cost nothing while a texture is to take a parked image, so that no
/// pass of the residency thread releases it first. A negative control, run by hand: with the take
/// disabled, the second texture is made anew and the test fails at its counts. Where the picture
/// lies in no such block, or its texture is a shadow, it says so and passes without running. On
/// Linux, whose dma-bufs these are.
#[test]
#[cfg(target_os = "linux")]
fn a_texture_over_a_dma_buf_block_takes_the_image_the_last_one_parked() {
    use crate::submit_check::{Check, Verdict};
    let watched = Watched::default();
    let scratch = Scratch::new("exported-parked-images");
    let device = device(&watched, &scratch.0);
    let name = device.name().to_owned();
    assert!(device.settle_in_place_checks(WAIT), "the check answered");
    assert!(device.settle_submit_checks(WAIT), "the submission checks");
    let picture = (device.new_buffer_with_descriptor(
        BufferDescriptor::new(LENGTH, ResourceOptions::shared(), BufferContents::Zero)
            .with_aligned_byte(LUMA_AT)
            .with_padding_unobserved(),
    ))
    .unwrap();
    if picture.bound_memory() != BoundMemory::DmaBufBlocks {
        eprintln!(
            "EXPORTED-PARKED on {name}: the picture lies in {:?}, no block exportable as a \
             dma-buf too: not run",
            picture.bound_memory()
        );
        return;
    }
    let plane = PARK_PLANE;
    let descriptor = descriptor(plane, ResourceOptions::shared());
    fill(&device, &picture);
    let checks = &device.shared().submit_checks;
    let measured = checks.verdict(Check::ImportTax);
    checks.force(Check::ImportTax, Verdict::Fails);
    let before = device.work_counts();
    let first = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
    let made = device.work_counts();
    if first.shadow().is_some() {
        eprintln!("EXPORTED-PARKED on {name}: the texture is a shadow: not run");
        checks.force(Check::ImportTax, measured);
        return;
    }
    assert_eq!(
        made.exported_dma_buf_images - before.exported_dma_buf_images,
        1,
        "the first texture's image made"
    );
    assert!(
        read(&device, &first, plane) == want(plane),
        "the first texels"
    );
    drop(first);
    let parked = counts_until(&device, |counts| {
        counts.exported_dma_buf_images_parked > made.exported_dma_buf_images_parked
    });
    // The CPU's other bytes, which the taken image reads.
    let other: Vec<u8> = want(plane).iter().map(|byte| byte ^ 0x5a).collect();
    // SAFETY: the picture's mapped bytes, which no command uses now.
    unsafe {
        let at = picture.contents().unwrap().as_ptr().add(plane.at as usize);
        std::ptr::copy_nonoverlapping(other.as_ptr(), at, other.len());
    }
    let second = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
    let taken = device.work_counts();
    let third = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
    let anew = device.work_counts();
    let made_taken = |counts: &crate::WorkCounts| {
        (
            counts.exported_dma_buf_images - before.exported_dma_buf_images,
            counts.exported_dma_buf_images_taken - before.exported_dma_buf_images_taken,
        )
    };
    eprintln!(
        "EXPORTED-PARKED on {name}: made, taken: the first {:?}, the second {:?}, a third while \
         the second lives {:?}; parked {}",
        made_taken(&made),
        made_taken(&taken),
        made_taken(&anew),
        parked.exported_dma_buf_images_parked - before.exported_dma_buf_images_parked
    );
    assert_eq!(
        (made_taken(&taken), made_taken(&anew)),
        ((1, 1), (2, 1)),
        "one image made for the first two textures, the second's taken; the third made anew"
    );
    assert!(
        second.shadow().is_none() && third.shadow().is_none(),
        "in place"
    );
    assert!(
        read(&device, &second, plane) == other,
        "the taken image reads the CPU's later bytes"
    );
    assert!(read(&device, &third, plane) == other, "the third's texels");
    drop((second, third));
    let both = counts_until(&device, |counts| {
        counts.exported_dma_buf_images_parked >= parked.exported_dma_buf_images_parked + 2
    });
    // Released untaken by the residency's passes: kept within its window where held imports cost
    // nothing, released at the next pass where they cost every submission.
    let released = |counts: &crate::WorkCounts| {
        counts.exported_dma_buf_images_parked_released
            - both.exported_dma_buf_images_parked_released
    };
    device.shared().residency.sweep(false);
    let kept = device.work_counts();
    // The residency thread's own next pass, a runtime thread, whose Vulkan calls each lie in a
    // section of the exit guard (a test build ends the process at one that does not).
    checks.force(Check::ImportTax, Verdict::Holds);
    let swept = counts_until(&device, |counts| released(counts) >= 2);
    checks.force(Check::ImportTax, measured);
    assert_eq!(
        (released(&kept), released(&swept)),
        (0, 2),
        "kept by a pass within the window where held imports cost nothing, released by the \
         thread's next where they cost every submission"
    );
    // A handed-out image parks nowhere.
    let handed = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
    // SAFETY: the handle is only read here, never used.
    let raw = unsafe { handed.vk_image() };
    assert!(raw.is_some(), "the texture's image handed out");
    let out = device.work_counts();
    drop(handed);
    let gone = counts_until(&device, |counts| {
        counts.dma_buf_image_memories_live < out.dma_buf_image_memories_live
    });
    assert_eq!(
        gone.exported_dma_buf_images_parked, out.exported_dma_buf_images_parked,
        "a handed-out image parks nowhere"
    );
    // One parked as the picture goes is released with its pages.
    let last = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
    drop(last);
    let parked_last = counts_until(&device, |counts| {
        counts.exported_dma_buf_images_parked > gone.exported_dma_buf_images_parked
    });
    drop(picture);
    let freed = counts_until(&device, |counts| {
        counts.dma_buf_image_memories_live == before.dma_buf_image_memories_live
    });
    let total = |counts: &crate::WorkCounts| {
        (
            counts.exported_dma_buf_images_parked - before.exported_dma_buf_images_parked,
            counts.exported_dma_buf_images_taken - before.exported_dma_buf_images_taken,
            counts.exported_dma_buf_images_parked_released
                - before.exported_dma_buf_images_parked_released,
        )
    };
    eprintln!(
        "EXPORTED-PARKED on {name}: parked, taken, released: {:?} (the last parked {:?}); dma-buf \
         image imports live {} as before",
        total(&freed),
        total(&parked_last),
        freed.dma_buf_image_memories_live
    );
    let (parked, taken, released) = total(&freed);
    assert_eq!(
        parked,
        taken + released,
        "every parked image taken or released once the picture went"
    );
    watched.layer_reported_nothing(
        "DRM_FORMAT_MOD_LINEAR images over a dedicated import of a block's dma-buf, parked and \
         taken",
    );
}

/// The counts of textures made shadows because no image of theirs was made
/// (`WorkCounts::host_texture_shadows_refused_*`), summed.
fn refusals(counts: &crate::WorkCounts) -> u64 {
    counts.host_texture_shadows_refused_rows
        + counts.host_texture_shadows_refused_detached
        + counts.host_texture_shadows_refused_image
        + counts.host_texture_shadows_refused_pitch
        + counts.host_texture_shadows_refused_placement
        + counts.host_texture_shadows_refused_source
        + counts.host_texture_shadows_refused_dma_buf
}

/// The texture of `layout` (its descriptor, offset and row pitch) over `buffer`, a `Shared` buffer
/// that declares its padding unobserved in a block of the runtime's own buffers exportable as a
/// dma-buf too, in which no image is bound: an image bound to a dedicated import of the block's
/// dma-buf (`WorkCounts::exported_dma_buf_images`, `dma_buf_image_memories_live`), no shadow, no
/// refusal counted (`WorkCounts::host_texture_shadows_refused_*`) and no notice (`notices`, the
/// device's notices so far), through which a blit reads the bytes the CPU wrote in the texels' rows
/// at the pitch.
fn over_a_dma_buf_block(
    device: &Device,
    buffer: &Buffer,
    layout: &(TextureDescriptor, u64, u64),
    notices: &AtomicU64,
) {
    let (descriptor, offset, pitch) = layout;
    let name = device.name();
    let texel = crate::buffer_texture::texel_bytes(descriptor.pixel_format).unwrap();
    let (width, height) = (descriptor.width, descriptor.height);
    let row = width * texel;
    // SAFETY: the buffer's mapped bytes, which no command uses now.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            buffer.contents().unwrap().as_ptr(),
            buffer.length() as usize,
        )
    };
    for y in 0..height {
        for x in 0..row {
            bytes[(offset + y * pitch + x) as usize] = byte(y * 4099 + x);
        }
    }
    let (before, noticed) = (device.work_counts(), notices.load(Ordering::Relaxed));
    let texture = buffer.new_texture(descriptor, *offset, *pitch).unwrap();
    let (after, notice) = (
        device.work_counts(),
        notices.load(Ordering::Relaxed) - noticed,
    );
    eprintln!(
        "EXPORTED-TEXTURES on {name}: {:?} {width}x{height} at {offset}, rows {pitch} apart, {:?}, \
         over a block exportable as a dma-buf too: images over a dedicated import of a dma-buf \
         {}, imports live {}, shadows {}, refusals counted {}, notices {notice}",
        descriptor.pixel_format,
        descriptor.usage,
        after.exported_dma_buf_images - before.exported_dma_buf_images,
        after.dma_buf_image_memories_live,
        after.shadowed_textures - before.shadowed_textures,
        refusals(&after) - refusals(&before),
    );
    assert!(
        texture.shadow().is_none() && texture.aliases_buffer(),
        "in place over the block"
    );
    assert_eq!(refusals(&after), refusals(&before), "no refusal counted");
    assert_eq!(notice, 0, "no notice");
    assert_eq!(
        after.exported_dma_buf_images - before.exported_dma_buf_images,
        1,
        "its image bound to a dedicated import of the block's dma-buf"
    );
    let read = device
        .new_buffer(row * height, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &read,
        0,
        row,
        row * height,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    // SAFETY: the read buffer's mapped bytes, which no command uses now.
    let read = unsafe {
        std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), (row * height) as usize)
    };
    for y in 0..height {
        for x in 0..row {
            assert_eq!(
                read[(y * row + x) as usize],
                byte(y * 4099 + x),
                "byte {x} of row {y}, read through the image"
            );
        }
    }
}
