//! Textures over a buffer's bytes that a device keeps as shadows for the layout of their rows,
//! and what the driver reports of that layout: the probe of the shadows a consumer's walks
//! counted on RADV, each case's placement (`Buffer::texture_placement`), what its making counted
//! (`WorkCounts::shadowed_textures`, `device_copy_textures`, `host_textures_in_place`,
//! `pooled_pitched_images`, `kept_padding_aliases`), and its texels read back by a blit, which
//! are the buffer's whatever the route.
//!
//! - [`rows_the_device_lays_no_linear_image_at`]: an `RGBA16Float` IOSurface plane of 18 x 28
//!   texels, rows 192 bytes apart (144 bytes of texels), over a `Shared` buffer of the device's
//!   own memory and over a no-copy buffer's host pages; `Managed` `RGBA8Unorm` textures of 38 x 8
//!   texels, rows 256 bytes apart (152 of texels), read alone and rendered, over a buffer of the
//!   device's memory and over host pages; and a `BGRA8Unorm` texture of 1024 x 1014 texels, rows
//!   14592 bytes apart, over a no-copy buffer that ends 13056 bytes into its last row (one page
//!   past it, for contrast). Printed beside them, the driver's answers: the row pitch of its
//!   linear image of the width (`vkGetImageSubresourceLayout`), and a `DRM_FORMAT_MOD_LINEAR`
//!   image whose explicit plane lies at the texels' pitch made or refused
//!   (`VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT`), and
//!   `minimumLinearTextureAlignmentForPixelFormat:`'s answer.
//! - [`dma_buf_rows_the_device_lays_no_plane_at`]: on a device made without
//!   `VK_EXT_external_memory_host`, whose no-copy buffers are imported as dma-bufs, `BGRA8Unorm`
//!   textures of 96 x 8 texels, rows 384 bytes apart, at offsets 0, 256 and 32 of the buffer, and
//!   an `R8Unorm` plane of 1920 x 4 texels at offset 32, rows 1920 bytes apart; with the driver's
//!   answer for a dma-buf `DRM_FORMAT_MOD_LINEAR` image at each offset and pitch.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Printed (`SHADOW-CAUSES`).

use super::host_residency::{counting, host_page, udmabuf_openable, Ram};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::lists_dma_buf_route;
use crate::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, HazardTrackingMode, HostMemory,
    IoSurfacePlane, Origin, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
    WorkCounts,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "SHADOW-CAUSES";

/// The longest the device's check of images in place may take.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// Page-aligned host memory of `len` bytes, freed when dropped: declared before the device that
/// uses it, so dropped after it.
struct Pages(NonNull<u8>, usize);

impl Pages {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, 64 << 10).unwrap()
    }

    /// `len` bytes counting from `seed`.
    fn new(len: usize, seed: usize) -> Pages {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc(Pages::layout(len)) };
        let pages = Pages(NonNull::new(pointer).unwrap(), len);
        let bytes = counting(len, seed);
        // SAFETY: the allocation's own bytes, which nothing else uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, len) };
        pages
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes; each case orders the CPU's and the GPU's accesses.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), self.1) }.to_vec()
    }

    /// A no-copy buffer of `length` bytes over the pages, of `options`.
    fn buffer(&self, device: &Device, length: u64, options: ResourceOptions) -> Option<Buffer> {
        // SAFETY: the pages outlive the buffer (each case drops it first), which alone uses them.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(HostMemory::new(self.0), length, options, None)
        };
        made.map_err(|error| println!("{TEST}: no no-copy buffer ({error:?})"))
            .ok()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Pages::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Pages::layout(self.1)) };
    }
}

/// `Managed` storage with the other modes at their defaults.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A buffer of the device's own memory of `options` holding `bytes`.
fn holding(device: &Device, bytes: &[u8], options: ResourceOptions) -> Buffer {
    let buffer = device.new_buffer(bytes.len() as u64, options).unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    if options.storage_mode().unwrap() == StorageMode::Managed {
        buffer.did_modify_range(0..bytes.len() as u64).unwrap();
    }
    buffer
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

/// The texture's texels, `row` bytes a row, packed: copied out by a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture, row: u64) -> Vec<u8> {
    let [width, height] = [texture.width(), texture.height()];
    let out = (queue.device())
        .new_buffer(row * height, ResourceOptions::shared())
        .unwrap();
    blit(queue, |encoder| {
        (encoder.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &out,
            0,
            row,
            row * height,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, row * height)
}

/// What a making counted, between `before` and `after`.
fn took(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "shadows {}, device copies {}, in place over host pages {}, pitched images {}, kept \
         padding aliases {}",
        after.shadowed_textures - before.shadowed_textures,
        after.device_copy_textures - before.device_copy_textures,
        after.host_textures_in_place - before.host_textures_in_place,
        after.pooled_pitched_images - before.pooled_pitched_images,
        after.kept_padding_aliases - before.kept_padding_aliases,
    )
}

/// One case: a texture of `descriptor` (of `texel` bytes a texel) over `buffer` (holding `bytes`
/// from its first byte) at `offset`, rows `pitch` apart, made as an IOSurface plane's where
/// `plane`; printed, and its texels checked to be the buffer's.
fn case(
    queue: &CommandQueue,
    what: &str,
    (buffer, bytes): (&Buffer, &[u8]),
    descriptor: &TextureDescriptor,
    (offset, pitch): (u64, u64),
    (plane, texel): (bool, u64),
) {
    let device = queue.device();
    let surface = IoSurfacePlane {
        offset,
        bytes_per_row: pitch,
        planar: false,
        client_direction: None,
        ycbcr: None,
    };
    let placement = match plane {
        true => buffer.texture_placement_with_iosurface_plane(descriptor, surface),
        false => buffer.texture_placement(descriptor, offset, pitch),
    };
    let before = device.work_counts();
    let texture = match plane {
        true => buffer.new_texture_with_iosurface_plane(descriptor, surface),
        false => buffer.new_texture(descriptor, offset, pitch),
    }
    .unwrap();
    let after = device.work_counts();
    println!(
        "{TEST} on {}: {what}: {:?} {}x{} {:?}, offset {offset}, rows {pitch}: {placement:?}; {}",
        device.name(),
        descriptor.pixel_format,
        descriptor.width,
        descriptor.height,
        descriptor.usage,
        took(&before, &after),
    );
    let row = descriptor.width * texel;
    assert_eq!(
        texels_of(queue, &texture, row),
        packed(bytes, offset, pitch, row, descriptor.height),
        "{TEST}: {what}: the texture's texels are the buffer's"
    );
}

/// What the driver reports of `format`'s images `extent` texels: the row pitch of its linear
/// image, and the `DRM_FORMAT_MOD_LINEAR` image whose one plane lies at `offset`, rows `pitch`
/// apart (for dma-bufs where `dma_buf`), laid out or refused.
fn answers(
    device: &Device,
    format: vk::Format,
    extent: [u32; 2],
    (offset, pitch): (u64, u64),
    dma_buf: bool,
) -> String {
    // SAFETY: the handles are used while the device lives, for images the test destroys.
    let handles = unsafe { device.raw() };
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let info = || {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let color = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    let plane_0 = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        ..color
    };
    let linear = info().tiling(vk::ImageTiling::LINEAR);
    // SAFETY: the device is live and `linear` valid; the image is destroyed after the query.
    let linear = unsafe { handles.device.create_image(&linear, None) }.map(|image| {
        // SAFETY: a live linear image of a color format, destroyed with no memory bound.
        unsafe {
            let layout = handles.device.get_image_subresource_layout(image, color);
            handles.device.destroy_image(image, None);
            layout.row_pitch
        }
    });
    let explicit = if device.makes_modifier_images() {
        let planes = [vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch: pitch,
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut modifier = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(crate::dma_buf::DRM_FORMAT_MOD_LINEAR)
            .plane_layouts(&planes);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut info = info()
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .push_next(&mut modifier);
        if dma_buf {
            info = info.push_next(&mut external);
        }
        // SAFETY: the device enables VK_EXT_image_drm_format_modifier (and, for a dma-buf's,
        // the dma-buf extensions, where the caller asks); the image is destroyed after the query.
        let made = unsafe { handles.device.create_image(&info, None) }.map(|image| {
            // SAFETY: a live image of a DRM format modifier's tiling, whose one memory plane is
            // plane 0, destroyed with no memory bound.
            unsafe {
                let layout = handles.device.get_image_subresource_layout(image, plane_0);
                handles.device.destroy_image(image, None);
                layout
            }
        });
        match made {
            Ok(layout) => format!(
                "made, its plane at offset {}, rows {} apart",
                layout.offset, layout.row_pitch
            ),
            Err(error) => format!("refused ({error:?})"),
        }
    } else {
        "not asked (no VK_EXT_image_drm_format_modifier)".to_owned()
    };
    format!(
        "{format:?} {}x{}: its linear image's rows {linear:?} apart; a DRM_FORMAT_MOD_LINEAR \
         image{} with its plane at offset {offset}, rows {pitch} apart: {explicit}",
        extent[0],
        extent[1],
        if dma_buf { " for dma-bufs" } else { "" },
    )
}

#[test]
fn rows_the_device_lays_no_linear_image_at() {
    let scratch = super::blit_holds::Scratch::new("shadow-causes");
    let device = super::blit_holds::device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let reads = TextureUsage::SHADER_READ;
    let renders = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    // An IOSurface plane of RGBA16Float, 18 x 28 texels, rows 192 bytes apart.
    let alignment = device
        .minimum_linear_texture_alignment_for_pixel_format(PixelFormat::RGBA16Float)
        .unwrap();
    println!("{TEST} on {name}: RGBA16Float's minimum linear texture alignment: {alignment}");
    let format = vk::Format::R16G16B16A16_SFLOAT;
    for pitch in [192, 256] {
        let answer = answers(&device, format, [18, 28], (0, pitch), false);
        println!("{TEST} on {name}: answer: {answer}");
    }
    let surface_bytes = counting(192 * 28, 5);
    for usage in [reads, renders] {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 18, 28, false);
        d.usage = usage;
        let own = holding(&device, &surface_bytes, ResourceOptions::shared());
        let at = (0, 192);
        case(
            &queue,
            "plane, own memory",
            (&own, &surface_bytes),
            &d,
            at,
            (true, 8),
        );
        let pages = Pages::new(64 << 10, 5);
        if let Some(host) = pages.buffer(&device, 64 << 10, ResourceOptions::shared()) {
            let bytes = pages.bytes();
            case(
                &queue,
                "plane, host pages",
                (&host, &bytes),
                &d,
                at,
                (true, 8),
            );
        }
    }
    // Managed RGBA8Unorm, 38 x 8 texels, rows 256 bytes apart.
    let managed_bytes = counting(256 * 8, 11);
    for usage in [reads, renders] {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 38, 8, false);
        d.usage = usage;
        let own = holding(&device, &managed_bytes, managed());
        let over = (&own, managed_bytes.as_slice());
        case(
            &queue,
            "Managed, own memory",
            over,
            &d,
            (0, 256),
            (false, 4),
        );
        let pages = Pages::new(64 << 10, 11);
        if let Some(host) = pages.buffer(&device, 64 << 10, managed()) {
            let bytes = pages.bytes();
            case(
                &queue,
                "Managed, host pages",
                (&host, &bytes),
                &d,
                (0, 256),
                (false, 4),
            );
        }
    }
    // BGRA8Unorm, 1024 x 1014 texels, rows 14592 bytes apart, over a buffer that ends 13056
    // bytes into its last row, and one a page longer.
    let short = 1013 * 14592 + 13056;
    let page = host_page() as u64;
    let pages = Pages::new(16 << 20, 3);
    let bytes = pages.bytes();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 1024, 1014, false);
    d.usage = reads;
    for (what, length) in [("short last row", short), ("a page past it", short + page)] {
        if let Some(host) = pages.buffer(&device, length, ResourceOptions::shared()) {
            case(&queue, what, (&host, &bytes), &d, (0, 14592), (false, 4));
        }
    }
}

#[test]
fn dma_buf_rows_the_device_lays_no_plane_at() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = super::host_residency::Scratch::new();
    let device = super::no_pointer_import::device(&scratch.0, None);
    let name = device.name().to_owned();
    if !lists_dma_buf_route(&device) || !device.makes_modifier_images() {
        println!("{TEST}: {name} lists no dma-buf route or makes no modifier images: not run");
        return;
    }
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    let queue = device.new_command_queue().unwrap();
    let cases: [(PixelFormat, vk::Format, [u64; 2], u64, u64, u64); 5] = [
        (
            PixelFormat::BGRA8Unorm,
            vk::Format::B8G8R8A8_UNORM,
            [96, 8],
            0,
            384,
            4,
        ),
        (
            PixelFormat::BGRA8Unorm,
            vk::Format::B8G8R8A8_UNORM,
            [96, 8],
            256,
            384,
            4,
        ),
        (
            PixelFormat::BGRA8Unorm,
            vk::Format::B8G8R8A8_UNORM,
            [96, 8],
            32,
            384,
            4,
        ),
        (
            PixelFormat::R8Unorm,
            vk::Format::R8_UNORM,
            [1920, 4],
            32,
            1920,
            1,
        ),
        (
            PixelFormat::BGRA8Unorm,
            vk::Format::B8G8R8A8_UNORM,
            [64, 8],
            0,
            256,
            4,
        ),
    ];
    let page = host_page();
    let length = 4 * page;
    for (pf, format, [width, height], offset, pitch, texel) in cases {
        let extent = [width as u32, height as u32];
        let answer = answers(&device, format, extent, (offset, pitch), true);
        println!("{TEST} on {name}: answer: {answer}");
        let ram = Ram::new(length, 0, length);
        let bytes = counting(length, 7);
        ram.write_file(0, &bytes);
        let calls = Arc::new(AtomicUsize::new(0));
        // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
        let buffer: Buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::with_dma_buf(ram.pointer(0), ram.source(&calls)),
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let mut d = TextureDescriptor::texture_2d(pf, width, height, false);
        d.usage = TextureUsage::SHADER_READ;
        let at = (offset, pitch);
        case(
            &queue,
            "dma-buf pages",
            (&buffer, &bytes),
            &d,
            at,
            (false, texel),
        );
        drop(buffer);
        drop(ram);
    }
}
