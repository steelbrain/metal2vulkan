//! What a heap texture that a shadow of its bytes holds to Metal's answer costs on the device the
//! cfg selects: report only (`HEAP-SHADOW` lines). Ignored by default: a measurement, not a check.
//!
//! The shapes are those a guest's compositor made in its heaps on an Iris Xe GPU under ANV (the
//! consumer's count of one macOS 26.7.2 boot): `RGBA16Float` 2D arrays of one slice from 1x1 to
//! 32x32, and 2D textures of `RGBA16Float` and `RGBA32Float` to 456x255. Each is made in an
//! automatic heap of `Private` storage, of `ShaderRead | RenderTarget` usage, under each
//! `HeapCompressionPolicy`, and printed:
//!
//! - its answer (`heap_texture_size_and_align`) and the notices its making announced, which name
//!   its route, what its images take of the heap and the answer (`texture/heap.rs`);
//! - the device's own images of its format and shape, made beside it through the queue's
//!   `VkDevice` (`CommandQueue::with_vk_queue`): optimal, optimal of a mutable format, and linear,
//!   each image's size and alignment;
//! - per frame of a command buffer that clears it in a render pass and blits it into a texture of
//!   its own (and of one that only blits it), the work counts that moved and the GPU time's
//!   median, beside the same frames over a texture of the same descriptor outside a heap.

use super::video_frames::moved_per_frame;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::device::HeapCompressionPolicy;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    CommandBuffer, CommandQueue, Device, HeapDescriptor, HeapType, LoadAction,
    RenderPassDescriptor, ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// The shapes: format, type, width and height.
const SHAPES: [(PixelFormat, TextureType, u64, u64); 9] = [
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 32, 32),
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 16, 16),
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 1, 1),
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 8, 8),
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 19, 19),
    (PixelFormat::RGBA16Float, TextureType::Type2D, 28, 18),
    (PixelFormat::RGBA16Float, TextureType::Type2D, 140, 130),
    (PixelFormat::RGBA32Float, TextureType::Type2D, 456, 255),
    (PixelFormat::RGBA16Float, TextureType::Type2DArray, 299, 299),
];

/// Frames timed, after [`WARM`] untimed.
const FRAMES: usize = 40;
const WARM: usize = 8;

fn descriptor(
    (format, texture_type, width, height): (PixelFormat, TextureType, u64, u64),
) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(format, width, height, false);
    d.texture_type = texture_type;
    d.array_length = 1;
    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    d
}

/// The Vulkan format of `format` (the shapes' two).
fn vk_format(format: PixelFormat) -> vk::Format {
    match format {
        PixelFormat::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
        PixelFormat::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,
        other => panic!("no Vulkan format named for {other:?}"),
    }
}

/// The size and alignment of the device's image of `shape` with `tiling` and `flags`, made and
/// destroyed through the queue's `VkDevice`, or why it was not made.
fn requirements(
    queue: &CommandQueue,
    (format, _, width, height): (PixelFormat, TextureType, u64, u64),
    tiling: vk::ImageTiling,
    flags: vk::ImageCreateFlags,
) -> String {
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format(format))
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
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: the image is made, queried and destroyed on the queue's own device, which outlives
    // the call, and nothing else sees it.
    unsafe {
        queue.with_vk_queue(|device, _, _| match device.create_image(&info, None) {
            Ok(image) => {
                let r = device.get_image_memory_requirements(image);
                device.destroy_image(image, None);
                format!("{} at {}", r.size, r.alignment)
            }
            Err(e) => format!("not made ({e:?})"),
        })
    }
}

/// The GPU time's median of [`FRAMES`] command buffers that `encode` fills, in microseconds, and
/// the work counts that moved per frame over them.
fn frames(device: &Device, queue: &CommandQueue, encode: &dyn Fn(&CommandBuffer)) -> String {
    let mut gpu = Vec::new();
    let mut before = device.work_counts();
    for frame in 0..WARM + FRAMES {
        if frame == WARM {
            before = device.work_counts();
        }
        let command_buffer = queue.command_buffer().unwrap();
        encode(&command_buffer);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        if frame >= WARM {
            let start = command_buffer.gpu_start_time().unwrap();
            let end = command_buffer.gpu_end_time().unwrap();
            gpu.push((end - start) * 1e6);
        }
    }
    let after = device.work_counts();
    gpu.sort_by(f64::total_cmp);
    format!(
        "{:.1} us; per frame: {}",
        gpu[gpu.len() / 2],
        moved_per_frame(&before, &after, FRAMES)
    )
}

/// A command buffer's frame over `texture`: a pass that clears it and stores, where `write`, then
/// a blit of it into `copy`.
fn frame(texture: &Texture, copy: &Texture, write: bool) -> impl Fn(&CommandBuffer) + '_ {
    let copy = copy.clone();
    move |command_buffer: &CommandBuffer| {
        if write {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(texture.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
        }
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture(texture, &copy).unwrap();
        blit.end_encoding().unwrap();
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn heap_texture_shadow_costs() {
    for policy in [
        HeapCompressionPolicy::Compressed,
        HeapCompressionPolicy::Uncompressed,
    ] {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.heap_compression = policy;
        })
        .unwrap();
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        for shape in SHAPES {
            let d = descriptor(shape);
            let answer = device.heap_texture_size_and_align(&d).unwrap();
            let heap = device
                .new_heap(&HeapDescriptor::new(
                    HeapType::Automatic,
                    4 * answer.size.max(answer.align),
                    ResourceOptions::private(),
                ))
                .unwrap();
            let before = device.work_counts();
            let in_heap = heap.new_texture(&d).unwrap();
            let made = moved_per_frame(&before, &device.work_counts(), 1);
            let notices: Vec<String> = messages
                .take_notices()
                .into_iter()
                .map(|n| n.text)
                .collect();
            let own = device.new_texture(&d).unwrap();
            let copy = device.new_texture(&d).unwrap();
            messages.take_notices();
            let (format, texture_type, width, height) = shape;
            let optimal = requirements(
                &queue,
                shape,
                vk::ImageTiling::OPTIMAL,
                vk::ImageCreateFlags::empty(),
            );
            let mutable = requirements(
                &queue,
                shape,
                vk::ImageTiling::OPTIMAL,
                vk::ImageCreateFlags::MUTABLE_FORMAT,
            );
            let linear = requirements(
                &queue,
                shape,
                vk::ImageTiling::LINEAR,
                vk::ImageCreateFlags::empty(),
            );
            eprintln!(
                "HEAP-SHADOW on {} ({policy:?}) {format:?} {texture_type:?} {width}x{height}: \
                 answer {} at {}; made: {made}; notices {notices:?}; images: optimal {optimal}, \
                 mutable {mutable}, linear {linear}",
                device.name(),
                answer.size,
                answer.align,
            );
            for (label, write) in [("clear+blit", true), ("blit", false)] {
                let heap_frames = frames(&device, &queue, &frame(&in_heap, &copy, write));
                let own_frames = frames(&device, &queue, &frame(&own, &copy, write));
                eprintln!(
                    "HEAP-SHADOW-FRAMES {format:?} {texture_type:?} {width}x{height} {label} \
                     ({policy:?}): in the heap {heap_frames}; outside {own_frames}"
                );
            }
        }
    }
}
