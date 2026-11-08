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
//!
//! Two more measurements take no GPU work, for a host where the GPU is busy: `HEAP-IMAGES` lines,
//! the memory requirements of the device's images of every shape of the count (optimal, optimal
//! of a mutable format, optimal of host-transfer usage, linear), made and destroyed on a
//! `VkDevice` of the test's own that records and submits nothing; and `HEAP-ANSWERS` lines,
//! Metal's answer and the device's for each shape, which ask a device no GPU work either.

use super::video_frames::moved_per_frame;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::{DeviceType, TEST_DEVICE};
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

/// Every shape of the consumer's count: the `RGBA16Float` 2D arrays of one slice, then the 2D
/// textures.
const COUNTED: [(PixelFormat, TextureType, u64, u64); 22] = {
    const A: (PixelFormat, TextureType) = (PixelFormat::RGBA16Float, TextureType::Type2DArray);
    const H: (PixelFormat, TextureType) = (PixelFormat::RGBA16Float, TextureType::Type2D);
    const F: (PixelFormat, TextureType) = (PixelFormat::RGBA32Float, TextureType::Type2D);
    [
        (A.0, A.1, 32, 32),
        (A.0, A.1, 16, 16),
        (A.0, A.1, 1, 1),
        (A.0, A.1, 8, 8),
        (A.0, A.1, 19, 19),
        (A.0, A.1, 4, 4),
        (A.0, A.1, 10, 10),
        (A.0, A.1, 5, 5),
        (A.0, A.1, 3, 3),
        (A.0, A.1, 6, 6),
        (A.0, A.1, 6, 5),
        (A.0, A.1, 20, 20),
        (A.0, A.1, 20, 19),
        (A.0, A.1, 299, 299),
        (H.0, H.1, 28, 18),
        (H.0, H.1, 77, 77),
        (H.0, H.1, 75, 69),
        (H.0, H.1, 67, 67),
        (H.0, H.1, 146, 79),
        (H.0, H.1, 140, 130),
        (H.0, H.1, 128, 67),
        (F.0, F.1, 456, 255),
    ]
};

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
fn frame(texture: &Texture, copy: &Texture, write: bool) -> impl Fn(&CommandBuffer) {
    let (texture, copy) = (texture.clone(), copy.clone());
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
        blit.copy_texture(&texture, &copy).unwrap();
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

/// A `VkDevice` of the test's own on the physical device the cfg selects, with host image copy
/// where the device has it, which records and submits nothing.
struct QueryDevice {
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    name: String,
    host_transfer: bool,
    _entry: ash::Entry,
}

impl QueryDevice {
    fn new() -> QueryDevice {
        // SAFETY: the loader is loaded once here and outlives the instance, which `Drop`
        // destroys before it.
        let entry = unsafe { ash::Entry::load() }.expect("a Vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: a valid create info with no layers or extensions.
        let instance = unsafe { entry.create_instance(&info, None) }.expect("an instance");
        let wanted = match TEST_DEVICE {
            DeviceType::Cpu => [vk::PhysicalDeviceType::CPU; 2],
            DeviceType::Gpu => [
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::DISCRETE_GPU,
            ],
        };
        // SAFETY: queries of the live instance.
        let (physical, properties) = unsafe {
            let all = instance.enumerate_physical_devices().unwrap();
            (all.into_iter())
                .map(|p| (p, instance.get_physical_device_properties(p)))
                .find(|(_, properties)| wanted.contains(&properties.device_type))
                .expect("a physical device of the cfg's type")
        };
        let name = properties
            .device_name_as_c_str()
            .map_or_else(|_| "?".to_owned(), |n| n.to_string_lossy().into_owned());
        // SAFETY: queries of the live physical device.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) };
        let named = |name: &std::ffi::CStr| {
            (extensions.iter().flatten()).any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut copy = vk::PhysicalDeviceHostImageCopyFeaturesEXT::default();
        if named(ash::ext::host_image_copy::NAME) {
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut copy);
            // SAFETY: a query of the live physical device; the chain lives for the call.
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
        }
        let host_transfer = copy.host_image_copy == vk::TRUE;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&priorities)];
        let names = [ash::ext::host_image_copy::NAME.as_ptr()];
        let mut enable =
            vk::PhysicalDeviceHostImageCopyFeaturesEXT::default().host_image_copy(true);
        let mut info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        if host_transfer {
            info = info.enabled_extension_names(&names).push_next(&mut enable);
        }
        // SAFETY: a valid create info for the live physical device: its first queue family, and
        // the host image copy extension and feature only where it has them.
        let device = unsafe { instance.create_device(physical, &info, None) }.expect("a device");
        QueryDevice {
            instance,
            device,
            physical,
            name,
            host_transfer,
            _entry: entry,
        }
    }

    /// The size and alignment of the device's image of `shape` with `tiling`, `flags` and the
    /// colour usage a heap's render target takes (and `extra`), or why none is made: the format's
    /// limits for it are asked first, so no image is made the device refuses.
    fn requirements(
        &self,
        (format, _, width, height): (PixelFormat, TextureType, u64, u64),
        tiling: vk::ImageTiling,
        flags: vk::ImageCreateFlags,
        extra: vk::ImageUsageFlags,
    ) -> String {
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | extra;
        let format = vk_format(format);
        // SAFETY: a query of the live physical device.
        let limits = unsafe {
            self.instance.get_physical_device_image_format_properties(
                self.physical,
                format,
                vk::ImageType::TYPE_2D,
                tiling,
                usage,
                flags,
            )
        };
        if let Err(e) = limits {
            return format!("not made ({e:?})");
        }
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
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
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: an image within the format's limits for it, made, queried and destroyed on the
        // live device, which nothing else uses.
        unsafe {
            match self.device.create_image(&info, None) {
                Ok(image) => {
                    let r = self.device.get_image_memory_requirements(image);
                    self.device.destroy_image(image, None);
                    format!("{} at {}", r.size, r.alignment)
                }
                Err(e) => format!("not made ({e:?})"),
            }
        }
    }
}

impl Drop for QueryDevice {
    fn drop(&mut self) {
        // SAFETY: the device made no object that outlives it, and the instance outlives it.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn heap_texture_image_requirements() {
    let query = QueryDevice::new();
    for shape in COUNTED {
        let (format, texture_type, width, height) = shape;
        let optimal = vk::ImageTiling::OPTIMAL;
        let none = vk::ImageUsageFlags::empty();
        let plain = vk::ImageCreateFlags::empty();
        let image = query.requirements(shape, optimal, plain, none);
        let mutable =
            query.requirements(shape, optimal, vk::ImageCreateFlags::MUTABLE_FORMAT, none);
        let transfer = match query.host_transfer {
            true => {
                let usage = vk::ImageUsageFlags::HOST_TRANSFER_EXT;
                query.requirements(shape, optimal, plain, usage)
            }
            false => "not made (no hostImageCopy)".to_owned(),
        };
        let linear = query.requirements(shape, vk::ImageTiling::LINEAR, plain, none);
        eprintln!(
            "HEAP-IMAGES on {} {format:?} {texture_type:?} {width}x{height}: optimal {image}, \
             mutable {mutable}, host-transfer {transfer}, linear {linear}",
            query.name
        );
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn heap_texture_answers() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    for shape in COUNTED {
        let d = descriptor(shape);
        let (format, texture_type, width, height) = shape;
        let metal = device.metal_heap_texture_size_and_align(&d).unwrap();
        let answer = device.heap_texture_size_and_align(&d).unwrap();
        let metal = metal.map_or_else(
            || "none".to_owned(),
            |m| format!("{} at {}", m.size, m.align),
        );
        eprintln!(
            "HEAP-ANSWERS {format:?} {texture_type:?} {width}x{height}: Metal {metal}; {} {} at {}",
            device.name(),
            answer.size,
            answer.align
        );
    }
}
