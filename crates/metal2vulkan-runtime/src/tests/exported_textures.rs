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
//!
//! - **The runtime's own buffers' blocks.** The same picture in a buffer the program made with no
//!   export, its rows' padding declared unobserved as the consumer's caches declare it: where the
//!   device's own buffers lie in blocks exportable as opaque file descriptors (`memory.rs`'s
//!   "Buffers in exportable blocks"), its memory is such a block (`WorkCounts::fd_buffer_blocks_made`),
//!   which the runtime never exports (`export_memory_fd` refuses it, `NotExportable`), and each
//!   plane's image names the block's handle type, the layer reporting nothing; once the device's
//!   check of that memory answered, each plane is in place or the shadow its verdict decided, and
//!   reads the same bytes. Elsewhere its memory is a plain block, as before.
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
    let types = crate::device::lock(&device.shared().memory).fd_buffer_types();
    eprintln!("EXPORTED-TEXTURES on {name}: the blocks of the runtime's own buffers: {types:?}");
    let mut first = true;
    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
        let descriptor = BufferDescriptor::new(LENGTH, options, BufferContents::Zero)
            .with_aligned_byte(LUMA_AT)
            .with_padding_unobserved();
        let picture = device.new_buffer_with_descriptor(descriptor).unwrap();
        let (bound, counts) = (picture.bound_memory(), device.work_counts());
        match types {
            Ok(_) => {
                let fd = BoundMemory::External(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
                assert!(
                    bound == fd,
                    "{options:?}: a block exportable as an opaque fd"
                );
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
            Err(_) => {
                assert!(bound == BoundMemory::Own, "{options:?}: a plain block");
                let made = (counts.fd_buffer_blocks_made, counts.fd_buffer_blocks_plain);
                assert_eq!(made, (0, 0), "{options:?}: no block of such buffers");
            }
        }
        fill(&device, &picture);
        for plane in PLANES {
            let texture =
                (picture.new_texture(&descriptor(plane, options), plane.at, ROW)).unwrap();
            assert!(
                read(&device, &texture, plane) == want(plane),
                "{options:?} {plane:?}: the texels, read as made"
            );
        }
        if first {
            assert!(device.settle_in_place_checks(WAIT), "the check answered");
            first = false;
        }
        for plane in PLANES {
            let descriptor = descriptor(plane, options);
            let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
            let texture = (picture.new_texture(&descriptor, plane.at, ROW)).unwrap();
            assert!(
                matches!(
                    placement,
                    TexturePlacement::NotHostPages
                        | TexturePlacement::Shadow(
                            TextureShadowReason::Unchecked { .. }
                                | TextureShadowReason::CheckFailed { .. }
                        )
                ),
                "{options:?} {plane:?}: in place, or a shadow the device's check decided: \
                 {placement:?}"
            );
            assert_eq!(
                texture.shadow().is_none(),
                placement == TexturePlacement::NotHostPages
            );
            assert!(
                read(&device, &texture, plane) == want(plane),
                "{options:?} {plane:?}: the texels, read after the check answered"
            );
            eprintln!(
                "EXPORTED-TEXTURES on {name}: a buffer of the runtime's own, {options:?} {:?} \
                 {}x{}: {placement:?}",
                plane.format, plane.size[0], plane.size[1]
            );
        }
    }
    watched.layer_reported_nothing(
        "buffers and textures in blocks exportable as opaque fds name the handle type \
         (VUID-vkBindBufferMemory-memory-02726, VUID-vkBindImageMemory-memory-02728)",
    );
}
