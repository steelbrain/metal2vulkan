//! A texture over dma-buf pages whose texels lie past their first page's first byte, on the
//! device the cfg selects made without `VK_EXT_external_memory_host`, so that its no-copy buffers
//! are imported as dma-bufs of their pages (`tests/no_pointer_import.rs`): a 32x32 `BGRA8Unorm`
//! texture, rows 256 bytes apart, at offset 0 of a no-copy buffer half a host page into a `memfd`'s
//! pages mapped shared, whose source makes `udmabuf`s of them (8192 bytes into a 16 KiB page, as a
//! guest's 4 KiB page lies in an arm64 host's; 2048 into a 4 KiB one).
//!
//! The image in place over such texels (`dma_buf.rs`) is a `DRM_FORMAT_MOD_LINEAR` image whose
//! explicit plane starts at the texels' offset, bound to a dedicated import of a dma-buf of their
//! pages; where the driver lays that plane out at another offset, an image whose plane starts at
//! offset 0, bound at the texels' offset of an import of the pages that is no dedicated
//! allocation (`WorkCounts::dma_buf_images_at_offset`), where the driver takes such a binding;
//! else the texture is a shadow. This test is also the probe of a driver: it prints, one line
//! each,
//!
//! - **`answer`**: what the driver answers of an explicit plane at several offsets
//!   (`vkGetImageSubresourceLayout`'s offset, rows and slices; the memory's size and alignment;
//!   whether the image requires or prefers a dedicated allocation), and the external memory
//!   features the image format query gives dma-bufs for such an image;
//! - **`route answered`**: the product's route, the driver's answers deciding: which image it made
//!   (or its shadow and why), and whether a GPU copy out of the texture reads its texels;
//! - **`route unchecked`**: the explicit plane at the texels' offset taken whatever the driver
//!   answers (`dma_buf::PlaneRoute::Unchecked`): what such an image reads, the texels, or the
//!   bytes from the page's first byte, as a driver that ignores the plane's offset reads;
//! - **`route at offset`**: the image of a plane at offset 0 bound at the texels' offset taken
//!   even where the driver answers the explicit offset (`dma_buf::PlaneRoute::AtOffset`), so the
//!   route runs on such a driver too: made, or why not, and what it reads.
//!
//! It checks that the product's texture reads its texels and the CPU's later bytes, whatever
//! route it took; that a driver that answers the explicit offset keeps the dedicated import (no
//! image counted at an offset); that the image bound at the offset reads its texels and the
//! CPU's later bytes wherever it is made, and is made wherever the driver takes such a binding
//! (it requires no dedicated allocation, its memory's alignment divides the offset, and the format
//! query takes dma-bufs for it in an allocation shared with others) and the product's texture
//! was in place. What the unchecked route reads is the driver's, printed alone.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, or the device lists no dma-buf route or makes no
//! `DRM_FORMAT_MOD_LINEAR` images, it says so and passes without running.

use super::host_residency::{counting, host_page, udmabuf_openable, Ram, Scratch};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::dma_buf::PlaneRoute;
use crate::{
    BlitOption, Buffer, CommandQueue, Degradation, Device, HostMemory, MessageSink, Origin,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TEST: &str = "PLANE-OFFSET";
/// The texture's width and height, and the bytes from a row's first texel to the next row's.
const SIDE: u64 = 32;
const ROW: u64 = 256;
/// The bytes of a row's texels.
const ROW_BYTES: u64 = 4 * SIDE;
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// What the driver answers of a `DRM_FORMAT_MOD_LINEAR` image of the texture's shape for
/// dma-bufs, its one plane at `offset` with rows [`ROW`] apart, made by the test itself (of
/// [`usage`]): the plane's layout, the memory's requirements and dedicated requirements; or the
/// driver's refusal.
struct Answer {
    offset: u64,
    made: Result<(vk::SubresourceLayout, vk::MemoryRequirements, bool, bool), vk::Result>,
}

impl Answer {
    /// Whether the driver lays the plane out at the offset asked, rows [`ROW`] apart.
    fn as_asked(&self) -> bool {
        (self.made.as_ref())
            .is_ok_and(|(layout, ..)| layout.offset == self.offset && layout.row_pitch == ROW)
    }

    /// Whether an image of this answer binds at `offset` of memory it shares with others, which
    /// ends at `end`: laid out as asked, no dedicated allocation required, its memory's alignment
    /// dividing `offset`, and its memory from `offset` within `end`.
    fn binds_at(&self, offset: u64, end: u64) -> bool {
        self.as_asked()
            && (self.made.as_ref()).is_ok_and(|(_, requirements, requires, _)| {
                !requires
                    && offset.is_multiple_of(requirements.alignment.max(1))
                    && offset + requirements.size <= end
            })
    }

    fn words(&self) -> String {
        match &self.made {
            Ok((layout, requirements, requires, prefers)) => format!(
                "a plane at offset {} is laid out at offset {}, rows {} and slices {} bytes \
                 apart, {} bytes; its memory {} bytes, aligned to {}, types {:#x}; a dedicated \
                 allocation required {requires}, preferred {prefers}",
                self.offset,
                layout.offset,
                layout.row_pitch,
                layout.array_pitch,
                layout.size,
                requirements.size,
                requirements.alignment,
                requirements.memory_type_bits,
            ),
            Err(error) => format!("a plane at offset {} is refused ({error:?})", self.offset),
        }
    }
}

/// The usage of the test's own images and its format query: as a texture's that only reads.
fn usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
}

/// [`Answer`] of the driver under `device` for a plane at `offset`.
fn answer(device: &Device, offset: u64) -> Answer {
    // SAFETY: the handles are used while the device lives, for an image the test destroys.
    let handles = unsafe { device.raw() };
    let planes = [vk::SubresourceLayout {
        offset,
        size: 0,
        row_pitch: ROW,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(crate::dma_buf::DRM_FORMAT_MOD_LINEAR)
        .plane_layouts(&planes);
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .extent(vk::Extent3D {
            width: SIDE as u32,
            height: SIDE as u32,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage())
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut explicit)
        .push_next(&mut external);
    // SAFETY: the device enables VK_EXT_image_drm_format_modifier and the dma-buf extensions
    // (the caller checked), and `info` and its chain live for the call.
    let made = unsafe { handles.device.create_image(&info, None) }.map(|image| {
        let plane = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            mip_level: 0,
            array_layer: 0,
        };
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let asked = vk::ImageMemoryRequirementsInfo2::default().image(image);
        // SAFETY: `image` is a live image of a DRM format modifier's tiling, whose one memory
        // plane is plane 0; it is destroyed after the queries, with no memory bound.
        let layout = unsafe {
            let layout = handles.device.get_image_subresource_layout(image, plane);
            (handles.device).get_image_memory_requirements2(&asked, &mut requirements);
            handles.device.destroy_image(image, None);
            layout
        };
        let requirements = requirements.memory_requirements;
        (
            layout,
            requirements,
            dedicated.requires_dedicated_allocation != vk::FALSE,
            dedicated.prefers_dedicated_allocation != vk::FALSE,
        )
    });
    Answer { offset, made }
}

/// The external memory features the image format query gives dma-bufs for the test's image (of
/// [`usage`]), or its refusal; and the driver's name and information.
fn dma_buf_features_and_driver(
    device: &Device,
) -> (Result<vk::ExternalMemoryFeatureFlags, vk::Result>, String) {
    // SAFETY: the handles are used while the device lives.
    let handles = unsafe { device.raw() };
    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(crate::dma_buf::DRM_FORMAT_MOD_LINEAR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let query = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(vk::Format::B8G8R8A8_UNORM)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage())
        .push_next(&mut modifier)
        .push_next(&mut external);
    let mut imported = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut imported);
    // SAFETY: the physical device is the instance's; the query and its chains live for the call.
    let queried = unsafe {
        handles
            .instance
            .get_physical_device_image_format_properties2(
                handles.physical_device,
                &query,
                &mut properties,
            )
    };
    let features = queried.map(|()| imported.external_memory_properties.external_memory_features);
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later.
    unsafe {
        handles
            .instance
            .get_physical_device_properties2(handles.physical_device, &mut properties)
    };
    let name = driver.driver_name_as_c_str().map_or_else(
        |_| "?".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let info = driver.driver_info_as_c_str().map_or_else(
        |_| "?".to_owned(),
        |info| info.to_string_lossy().into_owned(),
    );
    (features, format!("{name} {info}"))
}

/// The `height` rows of `row` bytes at `offset` of `bytes`, `pitch` apart, packed.
fn packed(bytes: &[u8], offset: u64, pitch: u64, row: u64, height: u64) -> Vec<u8> {
    (0..height)
        .flat_map(|r| {
            let at = (offset + r * pitch) as usize;
            &bytes[at..at + row as usize]
        })
        .copied()
        .collect()
}

/// The texture's texels, packed: copied out by a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = (queue.device())
        .new_buffer(ROW_BYTES * SIDE, ResourceOptions::shared())
        .unwrap();
    blit(queue, |encoder| {
        (encoder.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &out,
            0,
            ROW_BYTES,
            ROW_BYTES * SIDE,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, ROW_BYTES * SIDE)
}

/// What `read` is: the texels at `offset` of `bytes`, or the bytes from the first byte of the
/// page they lie in, as an image whose plane's offset the driver ignores reads.
fn what_it_reads(read: &[u8], bytes: &[u8], offset: u64, page: u64) -> &'static str {
    if read == packed(bytes, offset, ROW, ROW_BYTES, SIDE) {
        "its texels"
    } else if read == packed(bytes, offset - offset % page, ROW, ROW_BYTES, SIDE) {
        "the bytes from its page's first byte, not its texels"
    } else {
        "neither its texels nor the bytes from its page's first byte"
    }
}

/// Take `route` for `device`'s images over dma-buf pages from now on.
fn force(device: &Device, route: PlaneRoute) {
    let forced = &device.shared().plane_route_forced;
    forced.store(route as u8, std::sync::atomic::Ordering::Relaxed);
}

/// What one route made of the texture: in place or not, the dma-buf images and those bound at an
/// offset it counted, whether its first read and the read after the CPU's later write read the
/// texels, and the words of its line.
struct Made {
    in_place: bool,
    at_offset: u64,
    reads: bool,
    line: String,
}

/// Make the texture over a new no-copy buffer `offset` bytes into a page of a `memfd`, its bytes
/// counting from `seed`, by `route`; read it, write the CPU's later bytes and read it again.
fn make(device: &Device, queue: &CommandQueue, route: PlaneRoute, seed: usize) -> Made {
    let page = host_page() as u64;
    let offset = page / 2;
    let length = ROW * SIDE;
    let file = 2 * page + length.next_multiple_of(page);
    let ram = Ram::new(file as usize, 0, file as usize);
    let bytes = counting(file as usize, seed);
    ram.write_file(0, &bytes);
    let calls = Arc::new(AtomicUsize::new(0));
    // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
    let buffer: Buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(offset as usize), ram.source(&calls)),
            length,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    force(device, route);
    let placement = buffer.texture_placement(&descriptor, 0, ROW).unwrap();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let after = device.work_counts();
    force(device, PlaneRoute::Answered);
    // SAFETY: the handle is only compared, while the texture lives.
    let in_place = unsafe { texture.vk_image() }.is_some();
    let images = after.dma_buf_images - before.dma_buf_images;
    let at_offset = after.dma_buf_images_at_offset - before.dma_buf_images_at_offset;
    let first = texels_of(queue, &texture);
    let later = counting(length as usize, seed + 97);
    ram.write_file(offset as usize, &later);
    let mut later_bytes = bytes.clone();
    later_bytes[offset as usize..(offset + length) as usize].copy_from_slice(&later);
    let second = texels_of(queue, &texture);
    let reads = (what_it_reads(&first, &bytes, offset, page), {
        what_it_reads(&second, &later_bytes, offset, page)
    });
    let line = format!(
        "{} ({placement:?}; {images} dma-buf image(s), {at_offset} bound at an offset): reads {}, \
         then after the CPU's later write {}",
        if in_place { "in place" } else { "a shadow" },
        reads.0,
        reads.1,
    );
    drop(texture);
    drop(buffer);
    drop(ram);
    Made {
        in_place,
        at_offset,
        reads: reads == ("its texels", "its texels"),
        line,
    }
}

/// The probe, and the checks the module's docs list, at half a host page into the texels' page.
#[test]
fn a_texture_over_dma_buf_pages_past_a_page_s_first_byte_reads_its_texels_by_each_route() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let notices: Arc<Mutex<Vec<String>>> = Arc::default();
    let heard = Arc::clone(&notices);
    let sink = MessageSink::new(move |message| {
        let notice = message.degradation.as_ref();
        if notice.is_some_and(|notice| notice.degradation == Degradation::ShadowedTexture) {
            heard.lock().unwrap().push(message.text.clone());
        }
    });
    let device = device(&scratch.0, Some(sink));
    let page = host_page() as u64;
    let offset = page / 2;
    if !lists_dma_buf_route(&device) || !device.makes_modifier_images() {
        println!(
            "{TEST}: {} lists no dma-buf route or makes no DRM_FORMAT_MOD_LINEAR images: not run",
            device.name()
        );
        return;
    }
    let (features, driver) = dma_buf_features_and_driver(&device);
    let name = device.name();
    println!("{TEST} on {name} ({driver}), {page}-byte pages: texels {offset} bytes into a page");
    let dedicated_only = match features {
        Ok(features) => {
            println!(
                "{TEST} on {name}: answer: the format query gives dma-bufs features {:#x}",
                features.as_raw()
            );
            features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        }
        Err(error) => {
            println!("{TEST} on {name}: answer: the format query refuses dma-bufs ({error:?})");
            true
        }
    };
    let offsets = [0, 256, 4096, offset];
    let answers: Vec<Answer> = (offsets.iter())
        .filter(|&&at| at < page)
        .map(|&at| answer(&device, at))
        .collect();
    for answer in &answers {
        println!("{TEST} on {name}: answer: {}", answer.words());
    }
    let at_texels = answers.last().unwrap();
    let at_zero = &answers[0];
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    for line in device.in_place_check_report() {
        if line.starts_with("DmaBuf") {
            println!("{TEST} on {name}: the check: {line}");
        }
    }
    let queue = device.new_command_queue().unwrap();
    let routes = [
        ("answered", PlaneRoute::Answered),
        ("unchecked", PlaneRoute::Unchecked),
        ("at offset", PlaneRoute::AtOffset),
    ];
    let mut made = Vec::new();
    for (seed, (words, route)) in routes.into_iter().enumerate() {
        let route_made = make(&device, &queue, route, 13 + 31 * seed);
        println!("{TEST} on {name}: route {words}: {}", route_made.line);
        made.push(route_made);
    }
    for notice in notices.lock().unwrap().iter() {
        println!("{TEST} on {name}: notice: {notice}");
    }
    let [answered, _unchecked, at_offset] = &made[..] else {
        unreachable!()
    };
    // The product's texture reads its texels, in place or a shadow.
    assert!(
        answered.reads,
        "{TEST}: the product's route: {}",
        answered.line
    );
    // A driver that answers the explicit offset keeps the dedicated import.
    if at_texels.as_asked() {
        assert_eq!(answered.at_offset, 0, "{TEST}: {}", answered.line);
    }
    // The image bound at the offset reads its texels wherever it is made (and a shadow does).
    assert!(
        at_offset.reads,
        "{TEST}: the route at the offset: {}",
        at_offset.line
    );
    assert_eq!(
        at_offset.at_offset,
        u64::from(at_offset.in_place),
        "{TEST}: {}",
        at_offset.line
    );
    // It is made wherever the driver takes such a binding within the buffer's pages, and the
    // texture is in place at all.
    let end = (offset + ROW * SIDE).next_multiple_of(page);
    if answered.in_place && at_zero.binds_at(offset, end) && !dedicated_only {
        assert!(at_offset.in_place, "{TEST}: {}", at_offset.line);
    }
}
