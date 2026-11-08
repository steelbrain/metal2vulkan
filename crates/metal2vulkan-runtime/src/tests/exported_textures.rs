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
//!   buffer's bytes; once the check answered (`Device::settle_in_place_checks`), the next commit
//!   moves both in place (`WorkCounts::pooled_textures_moved_in_place`), and their blits read the
//!   same bytes through the image in the buffer's memory. A texture made after is in place at once
//!   (`TexturePlacement::NotHostPages`, no shadow counted).
//! - **Each handle type, each storage.** Of an opaque file descriptor and of a dma-buf where the
//!   device exports one, `Private` (its bytes written by a blit, as a decoder writes a cache the CPU
//!   never maps) and `Shared` (written through its contents). Where the device exports such memory
//!   only as the buffer's dedicated allocation, to which no image may be bound
//!   (`VUID-vkBindImageMemory-memory-10926`), each texture is a shadow that reads the same bytes,
//!   its placement refused (`InPlaceRefusal::Placement`).
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
use crate::error::{Error, RuntimeGap};
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
            if first {
                let counts = device.work_counts();
                assert!(device.settle_in_place_checks(WAIT), "the check answered");
                // The next commit moves the shadows in place.
                let plane = PLANES[0];
                assert!(read(&device, &textures[0].0, plane) == want(plane));
                let moved = device.work_counts().pooled_textures_moved_in_place
                    - counts.pooled_textures_moved_in_place;
                assert_eq!(
                    moved, 2,
                    "{export:?} {options:?}: both planes moved in place"
                );
                first = false;
            }
            for (plane, (texture, _)) in PLANES.iter().zip(&textures) {
                let descriptor = descriptor(*plane, options);
                let placement = (picture.texture_placement(&descriptor, plane.at, ROW)).unwrap();
                assert_eq!(
                    placement,
                    TexturePlacement::NotHostPages,
                    "{export:?} {options:?} {plane:?}: in place once the check passed"
                );
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
            // Made after the check answered: in place at once.
            let shadows = device.work_counts().shadowed_textures;
            for plane in PLANES {
                let (texture, placement) = made(plane);
                assert_eq!(placement, TexturePlacement::NotHostPages, "{plane:?}");
                assert!(read(&device, &texture, plane) == want(plane));
            }
            assert_eq!(
                device.work_counts().shadowed_textures,
                shadows,
                "{export:?} {options:?}: no shadow made after the check"
            );
        }
    }
    watched.layer_reported_nothing(
        "textures over exported memory name its handle type (VUID-vkBindImageMemory-memory-02728)",
    );
}
