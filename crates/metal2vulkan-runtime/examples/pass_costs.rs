//! Not a consumer's example: what a render pass instance costs the driver's CPU, by how it is
//! recorded. It uses no runtime: a Vulkan 1.3 device of its own, made through the loader, on the
//! first discrete GPU (else integrated, virtual or CPU device) the loader lists.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example pass_costs -- \
//!     [--frames F] [--repeats R] [--draws D]
//! ```
//!
//! Each frame is [`PASSES`] render pass instances of `D` draws (1 by default), over one
//! `B8G8R8A8_UNORM` color attachment held in `GENERAL`, loaded and stored, submitted [`GROUP`] passes
//! to a `vkQueueSubmit`, two frames in flight; each draw binds the pipeline and draws a triangle.
//! The shapes:
//!
//! - **Command buffers**: a primary of its own for each pass (`per-pass`), as the runtime records
//!   an encoder's pass, or one primary for each submission's passes (`per-group`). A new primary
//!   starts with no dynamic state, so it sets the 17 dynamic states the runtime's draws record
//!   (`render.rs`'s `RECORDED` and the color write mask); `per-group-restated` sets them at each
//!   pass too.
//! - **The instance**: a `VkRenderPass` instance (`vkCmdBeginRenderPass`) over a `VkFramebuffer`
//!   made for the pass and destroyed once its frame completed (`made`), as the runtime makes one
//!   for each instance (`render_target::framebuffer`), or over one framebuffer kept (`kept`); or a
//!   dynamic rendering instance (`vkCmdBeginRendering`, `dynamic`), which takes no framebuffer.
//!
//! It prints a `PASS-COSTS` line for each shape: the calling thread's CPU time a pass (the
//! median of `R` runs of `F` frames, 5 of 200 by default, after a run of 20 not counted) to
//! record (its framebuffer's making included), to submit, and to reclaim (the framebuffers'
//! destruction and the command pool's reset once the frame completed), and their sum. Then the
//! differences between shapes that differ in one thing, a pass.

use ash::vk;

/// Render pass instances a frame: the walk's frames of the window server averaged about 150.
const PASSES: usize = 150;

/// Passes a `vkQueueSubmit`: the walk's submissions averaged about 6 primaries each.
const GROUP: usize = 6;

/// Frames in flight.
const SLOTS: usize = 2;

/// The color attachment's extent.
const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 1024,
    height: 1024,
};

/// The color attachment's format.
const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// The longest wait for a frame.
const TIMEOUT: u64 = 20_000_000_000;

/// The vertex shader: a small triangle from the vertex index, no input. Compiled from this GLSL
/// with glslang (`--target-env vulkan1.0`, debug names stripped):
///
/// ```glsl
/// #version 450
/// void main() {
///     vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
///     gl_Position = vec4(p * 0.25 - 0.5, 0.0, 1.0);
/// }
/// ```
#[rustfmt::skip]
const VERTEX_SPIRV: [u32; 236] = [
    0x07230203, 0x00010000, 0x0008000b, 0x0000002c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x0000001d, 0x00040047,
    0x0000000c, 0x0000000b, 0x0000002a, 0x00030047, 0x0000001b, 0x00000002, 0x00050048, 0x0000001b,
    0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000001b, 0x00000001, 0x0000000b, 0x00000001,
    0x00050048, 0x0000001b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000001b, 0x00000003,
    0x0000000b, 0x00000004, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020, 0x00000008,
    0x00000007, 0x00000007, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020, 0x0000000b,
    0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b, 0x0000000a,
    0x0000000e, 0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000002, 0x00040017, 0x00000017,
    0x00000006, 0x00000004, 0x00040015, 0x00000018, 0x00000020, 0x00000000, 0x0004002b, 0x00000018,
    0x00000019, 0x00000001, 0x0004001c, 0x0000001a, 0x00000006, 0x00000019, 0x0006001e, 0x0000001b,
    0x00000017, 0x00000006, 0x0000001a, 0x0000001a, 0x00040020, 0x0000001c, 0x00000003, 0x0000001b,
    0x0004003b, 0x0000001c, 0x0000001d, 0x00000003, 0x0004002b, 0x0000000a, 0x0000001e, 0x00000000,
    0x0004002b, 0x00000006, 0x00000020, 0x3e800000, 0x0004002b, 0x00000006, 0x00000022, 0x3f000000,
    0x0004002b, 0x00000006, 0x00000025, 0x00000000, 0x0004002b, 0x00000006, 0x00000026, 0x3f800000,
    0x00040020, 0x0000002a, 0x00000003, 0x00000017, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003d,
    0x0000000a, 0x0000000d, 0x0000000c, 0x000500c4, 0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e,
    0x000500c7, 0x0000000a, 0x00000011, 0x0000000f, 0x00000010, 0x0004006f, 0x00000006, 0x00000012,
    0x00000011, 0x0004003d, 0x0000000a, 0x00000013, 0x0000000c, 0x000500c7, 0x0000000a, 0x00000014,
    0x00000013, 0x00000010, 0x0004006f, 0x00000006, 0x00000015, 0x00000014, 0x00050050, 0x00000007,
    0x00000016, 0x00000012, 0x00000015, 0x0003003e, 0x00000009, 0x00000016, 0x0004003d, 0x00000007,
    0x0000001f, 0x00000009, 0x0005008e, 0x00000007, 0x00000021, 0x0000001f, 0x00000020, 0x00050050,
    0x00000007, 0x00000023, 0x00000022, 0x00000022, 0x00050083, 0x00000007, 0x00000024, 0x00000021,
    0x00000023, 0x00050051, 0x00000006, 0x00000027, 0x00000024, 0x00000000, 0x00050051, 0x00000006,
    0x00000028, 0x00000024, 0x00000001, 0x00070050, 0x00000017, 0x00000029, 0x00000027, 0x00000028,
    0x00000025, 0x00000026, 0x00050041, 0x0000002a, 0x0000002b, 0x0000001d, 0x0000001e, 0x0003003e,
    0x0000002b, 0x00000029, 0x000100fd, 0x00010038,
];

/// The fragment shader: one color. Compiled as [`VERTEX_SPIRV`] from:
///
/// ```glsl
/// #version 450
/// layout(location = 0) out vec4 color;
/// void main() {
///     color = vec4(0.25, 0.5, 0.75, 1.0);
/// }
/// ```
#[rustfmt::skip]
const FRAGMENT_SPIRV: [u32; 84] = [
    0x07230203, 0x00010000, 0x0008000b, 0x0000000f, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
    0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
    0x00000003, 0x0004002b, 0x00000006, 0x0000000a, 0x3e800000, 0x0004002b, 0x00000006, 0x0000000b,
    0x3f000000, 0x0004002b, 0x00000006, 0x0000000c, 0x3f400000, 0x0004002b, 0x00000006, 0x0000000d,
    0x3f800000, 0x0007002c, 0x00000007, 0x0000000e, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000d,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e,
    0x00000009, 0x0000000e, 0x000100fd, 0x00010038,
];

/// How a frame's passes take primaries.
#[derive(Clone, Copy, PartialEq)]
enum Buffers {
    /// A primary of its own for each pass, setting the dynamic states.
    PerPass,
    /// One primary for each submission's passes, setting the dynamic states once.
    PerGroup,
    /// One primary for each submission's passes, setting the dynamic states at each pass.
    PerGroupRestated,
}

/// How a pass begins its instance.
#[derive(Clone, Copy, PartialEq)]
enum Instance {
    /// `vkCmdBeginRenderPass` over a framebuffer made for the pass.
    Made,
    /// `vkCmdBeginRenderPass` over one framebuffer kept.
    Kept,
    /// `vkCmdBeginRendering`.
    Dynamic,
}

/// A way to record a frame.
#[derive(Clone, Copy)]
struct Shape {
    buffers: Buffers,
    instance: Instance,
}

impl Shape {
    fn name(self) -> String {
        let buffers = match self.buffers {
            Buffers::PerPass => "per-pass",
            Buffers::PerGroup => "per-group",
            Buffers::PerGroupRestated => "per-group-restated",
        };
        let instance = match self.instance {
            Instance::Made => "made",
            Instance::Kept => "kept",
            Instance::Dynamic => "dynamic",
        };
        format!("{buffers}/{instance}")
    }
}

/// The shapes, in the order they run.
const SHAPES: [Shape; 7] = [
    Shape {
        buffers: Buffers::PerPass,
        instance: Instance::Made,
    },
    Shape {
        buffers: Buffers::PerPass,
        instance: Instance::Kept,
    },
    Shape {
        buffers: Buffers::PerPass,
        instance: Instance::Dynamic,
    },
    Shape {
        buffers: Buffers::PerGroupRestated,
        instance: Instance::Made,
    },
    Shape {
        buffers: Buffers::PerGroup,
        instance: Instance::Made,
    },
    Shape {
        buffers: Buffers::PerGroup,
        instance: Instance::Kept,
    },
    Shape {
        buffers: Buffers::PerGroup,
        instance: Instance::Dynamic,
    },
];

/// The pairs of shapes that differ in one thing, and what that is.
const DIFFERENCES: [(usize, usize, &str); 6] = [
    (
        0,
        3,
        "a primary per pass (begin, end, submission), states set in both",
    ),
    (3, 4, "the 17 dynamic states set again"),
    (0, 1, "a framebuffer made and destroyed (primary per pass)"),
    (
        1,
        2,
        "vkCmdBeginRenderPass against vkCmdBeginRendering (primary per pass)",
    ),
    (
        0,
        2,
        "a render pass instance with its framebuffer against dynamic rendering",
    ),
    (
        0,
        6,
        "today's shape against a primary per group and dynamic rendering",
    ),
];

/// A frame's slot: its command pool and primaries, the framebuffers its passes made, and the fence
/// its last submission signals.
struct Slot {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    framebuffers: Vec<vk::Framebuffer>,
    fence: vk::Fence,
}

/// The calling thread's CPU time in nanoseconds.
fn thread_ns() -> u64 {
    let mut at = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `at` is a live `timespec` the call writes.
    let answer = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) };
    assert_eq!(answer, 0, "{}", std::io::Error::last_os_error());
    at.tv_sec as u64 * 1_000_000_000 + at.tv_nsec as u64
}

/// The CPU time of a run's phases, in nanoseconds a pass.
#[derive(Clone, Copy, Default)]
struct Phases {
    record: f64,
    submit: f64,
    reclaim: f64,
}

impl Phases {
    fn total(self) -> f64 {
        self.record + self.submit + self.reclaim
    }
}

/// The median of `values`.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The device, its attachment, render pass, pipelines and frame slots.
struct Bench {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    color_mask: Option<ash::ext::extended_dynamic_state3::Device>,
    queue: vk::Queue,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    render_pass: vk::RenderPass,
    kept: vk::Framebuffer,
    layout: vk::PipelineLayout,
    pass_pipeline: vk::Pipeline,
    dynamic_pipeline: vk::Pipeline,
    slots: Vec<Slot>,
    next: usize,
    draws: u32,
}

impl Bench {
    fn new(draws: u32) -> Bench {
        // SAFETY: loading the Vulkan loader runs its initialization; nothing else is assumed.
        let entry = unsafe { ash::Entry::load() }.expect("the Vulkan loader");
        // SAFETY: a global command of a loaded entry.
        let listed = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
        let portability = listed.iter().any(|extension| {
            extension.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME)
        });
        let extensions: Vec<*const std::ffi::c_char> = match portability {
            true => vec![ash::khr::portability_enumeration::NAME.as_ptr()],
            false => Vec::new(),
        };
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extensions)
            .flags(match portability {
                true => vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
                false => vk::InstanceCreateFlags::empty(),
            });
        // SAFETY: `info` and what it points at live for the call.
        let instance = unsafe { entry.create_instance(&info, None) }.unwrap();
        // SAFETY: the instance is live.
        let physicals = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let types = [
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::VIRTUAL_GPU,
            vk::PhysicalDeviceType::CPU,
        ];
        // SAFETY: each is the instance's.
        let properties = |physical: vk::PhysicalDevice| unsafe {
            instance.get_physical_device_properties(physical)
        };
        let physical = (types.iter())
            .find_map(|&wanted| {
                (physicals.iter().copied()).find(|&p| properties(p).device_type == wanted)
            })
            .expect("a device");
        let found = properties(physical);
        assert!(
            found.api_version >= vk::API_VERSION_1_3,
            "a Vulkan 1.3 device"
        );
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut wanted = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance's; Vulkan 1.1's command.
        unsafe { instance.get_physical_device_properties2(physical, &mut wanted) };
        let name = (found.device_name_as_c_str())
            .map_or("?".to_owned(), |name| name.to_string_lossy().into_owned());
        let driver_info = (driver.driver_info_as_c_str())
            .map_or("?".to_owned(), |info| info.to_string_lossy().into_owned());
        // SAFETY: the instance's.
        let device_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
        let has_state3 = device_extensions.iter().any(|extension| {
            extension.extension_name_as_c_str() == Ok(ash::ext::extended_dynamic_state3::NAME)
        });
        let mut state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut state3);
        // SAFETY: the instance's; Vulkan 1.1's command.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        let color_mask = has_state3 && state3.extended_dynamic_state3_color_write_mask == vk::TRUE;
        println!(
            "PASS-COSTS device {name} driver {driver_info} color write mask {color_mask} \
             passes {PASSES} group {GROUP} draws {draws}"
        );
        // SAFETY: the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = (families.iter())
            .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("a family of graphics work") as u32;
        let priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_color_write_mask(true);
        let names = [ash::ext::extended_dynamic_state3::NAME.as_ptr()];
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut v13);
        if color_mask {
            info = info.enabled_extension_names(&names).push_next(&mut state3);
        }
        // SAFETY: `info` and what it points at live for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        let color_mask =
            color_mask.then(|| ash::ext::extended_dynamic_state3::Device::new(&instance, &device));
        // SAFETY: the family's first queue was created.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: the instance's.
        let memory_types = unsafe { instance.get_physical_device_memory_properties(physical) };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(EXTENT.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `image_info` is valid for the device.
        let image = unsafe { device.create_image(&image_info, None) }.unwrap();
        // SAFETY: the image is the device's.
        let needs = unsafe { device.get_image_memory_requirements(image) };
        let types = &memory_types.memory_types[..memory_types.memory_type_count as usize];
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let type_index = (0..types.len())
            .find(|&index| {
                needs.memory_type_bits & (1 << index) != 0
                    && types[index].property_flags.contains(local)
            })
            .or_else(|| (0..types.len()).find(|&index| needs.memory_type_bits & (1 << index) != 0))
            .expect("a memory type for the image") as u32;
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(type_index);
        // SAFETY: a valid allocation of the device; the image is bound to it once.
        let memory = unsafe { device.allocate_memory(&allocate, None) }.unwrap();
        // SAFETY: as above.
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(FORMAT)
            .subresource_range(range);
        // SAFETY: the image is bound and live.
        let view = unsafe { device.create_image_view(&view_info, None) }.unwrap();
        let attachment = [vk::AttachmentDescription::default()
            .format(FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::GENERAL)
            .final_layout(vk::ImageLayout::GENERAL)];
        let reference = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::GENERAL)];
        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&reference)];
        let pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment)
            .subpasses(&subpass);
        // SAFETY: `pass_info` and what it points at live for the call.
        let render_pass = unsafe { device.create_render_pass(&pass_info, None) }.unwrap();
        let kept = framebuffer(&device, render_pass, view);
        // SAFETY: an empty layout.
        let layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        }
        .unwrap();
        let (pass_pipeline, dynamic_pipeline) =
            pipelines(&device, render_pass, layout, color_mask.is_some());
        let slots = (0..SLOTS)
            .map(|_| {
                let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
                // SAFETY: the family's.
                let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
                let allocate = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(PASSES as u32);
                // SAFETY: the pool is live.
                let buffers = unsafe { device.allocate_command_buffers(&allocate) }.unwrap();
                let fence_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                // SAFETY: a valid fence of the device.
                let fence = unsafe { device.create_fence(&fence_info, None) }.unwrap();
                Slot {
                    pool,
                    buffers,
                    framebuffers: Vec::with_capacity(PASSES),
                    fence,
                }
            })
            .collect();
        let bench = Bench {
            _entry: entry,
            instance,
            device,
            color_mask,
            queue,
            image,
            memory,
            view,
            render_pass,
            kept,
            layout,
            pass_pipeline,
            dynamic_pipeline,
            slots,
            next: 0,
            draws,
        };
        bench.to_general(range);
        bench
    }

    /// Move the attachment into `GENERAL`, once, and wait.
    fn to_general(&self, range: vk::ImageSubresourceRange) {
        let device = &self.device;
        let slot = &self.slots[0];
        let command = slot.buffers[0];
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range)];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let commands = [command];
        let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
        // SAFETY: the command buffer is idle; the barrier names the live image; the wait ends its
        // use before the pool is reset.
        unsafe {
            device.begin_command_buffer(command, &begin).unwrap();
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            device.end_command_buffer(command).unwrap();
            device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(self.queue).unwrap();
            (device.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())).unwrap();
        }
    }

    /// Set the 17 dynamic states on `command`, as a new primary needs.
    fn set_states(&self, command: vk::CommandBuffer) {
        let device = &self.device;
        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: EXTENT.width as f32,
            height: EXTENT.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: EXTENT,
        }];
        let both = vk::StencilFaceFlags::FRONT_AND_BACK;
        // SAFETY: `command` is recording, of a device of Vulkan 1.3, whose pipelines take each
        // state as dynamic.
        unsafe {
            device.cmd_set_viewport(command, 0, &viewport);
            device.cmd_set_scissor(command, 0, &scissor);
            device.cmd_set_depth_bias(command, 0.0, 0.0, 0.0);
            device.cmd_set_blend_constants(command, &[0.0; 4]);
            device.cmd_set_stencil_compare_mask(command, both, !0);
            device.cmd_set_stencil_write_mask(command, both, !0);
            device.cmd_set_stencil_reference(command, both, 0);
            device.cmd_set_cull_mode(command, vk::CullModeFlags::NONE);
            device.cmd_set_front_face(command, vk::FrontFace::COUNTER_CLOCKWISE);
            device.cmd_set_primitive_topology(command, vk::PrimitiveTopology::TRIANGLE_LIST);
            device.cmd_set_primitive_restart_enable(command, false);
            device.cmd_set_depth_test_enable(command, false);
            device.cmd_set_depth_write_enable(command, false);
            device.cmd_set_depth_compare_op(command, vk::CompareOp::ALWAYS);
            device.cmd_set_stencil_test_enable(command, false);
            device.cmd_set_stencil_op(
                command,
                both,
                vk::StencilOp::KEEP,
                vk::StencilOp::KEEP,
                vk::StencilOp::KEEP,
                vk::CompareOp::ALWAYS,
            );
            if let Some(color_mask) = &self.color_mask {
                color_mask.cmd_set_color_write_mask(command, 0, &[vk::ColorComponentFlags::RGBA]);
            }
        }
    }

    /// Record one pass into `command`: its instance, `draws` draws, its end; a framebuffer made
    /// for it goes into `made`.
    fn pass(
        &self,
        command: vk::CommandBuffer,
        instance: Instance,
        made: &mut Vec<vk::Framebuffer>,
    ) {
        let device = &self.device;
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: EXTENT,
        };
        // SAFETY: `command` is recording, outside any instance; the framebuffer, view and
        // pipelines are live until the frame completed.
        unsafe {
            let pipeline = match instance {
                Instance::Made | Instance::Kept => {
                    let target = match instance {
                        Instance::Made => {
                            let target = framebuffer(device, self.render_pass, self.view);
                            made.push(target);
                            target
                        }
                        _ => self.kept,
                    };
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(self.render_pass)
                        .framebuffer(target)
                        .render_area(area);
                    device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
                    self.pass_pipeline
                }
                Instance::Dynamic => {
                    let colors = [vk::RenderingAttachmentInfo::default()
                        .image_view(self.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)];
                    let begin = vk::RenderingInfo::default()
                        .render_area(area)
                        .layer_count(1)
                        .color_attachments(&colors);
                    device.cmd_begin_rendering(command, &begin);
                    self.dynamic_pipeline
                }
            };
            for _ in 0..self.draws {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_draw(command, 3, 1, 0, 0);
            }
            match instance {
                Instance::Dynamic => device.cmd_end_rendering(command),
                _ => device.cmd_end_render_pass(command),
            }
        }
    }

    /// Run `frames` frames of `shape`: the CPU time of each phase, a pass.
    fn run(&mut self, shape: Shape, frames: usize) -> Phases {
        let mut spent = [0u64; 3];
        for _ in 0..frames {
            let at = self.next;
            self.next = (self.next + 1) % SLOTS;
            let fence = self.slots[at].fence;
            // SAFETY: the slot's fence, signaled by its last frame's last submission.
            unsafe { self.device.wait_for_fences(&[fence], true, TIMEOUT) }.unwrap();
            let start = thread_ns();
            for target in self.slots[at].framebuffers.drain(..) {
                // SAFETY: the frame that used it completed.
                unsafe { self.device.destroy_framebuffer(target, None) };
            }
            // SAFETY: the slot's command buffers completed (the fence).
            unsafe {
                let pool = self.slots[at].pool;
                (self
                    .device
                    .reset_command_pool(pool, vk::CommandPoolResetFlags::empty()))
                .unwrap();
                self.device.reset_fences(&[fence]).unwrap();
            }
            let reclaimed = thread_ns();
            let (submissions, made) = self.record(at, shape);
            let recorded = thread_ns();
            for (index, commands) in submissions.iter().enumerate() {
                let submit = [vk::SubmitInfo::default().command_buffers(commands)];
                let signal = match index + 1 == submissions.len() {
                    true => fence,
                    false => vk::Fence::null(),
                };
                // SAFETY: each command buffer is executable and submitted once.
                unsafe { self.device.queue_submit(self.queue, &submit, signal) }.unwrap();
            }
            let submitted = thread_ns();
            self.slots[at].framebuffers = made;
            spent[0] += recorded - reclaimed;
            spent[1] += submitted - recorded;
            spent[2] += reclaimed - start;
        }
        let passes = (frames * PASSES) as f64;
        Phases {
            record: spent[0] as f64 / passes,
            submit: spent[1] as f64 / passes,
            reclaim: spent[2] as f64 / passes,
        }
    }

    /// Record a frame of `shape` into slot `at`'s primaries: each submission's command buffers,
    /// and the framebuffers its passes made.
    fn record(
        &self,
        at: usize,
        shape: Shape,
    ) -> (Vec<Vec<vk::CommandBuffer>>, Vec<vk::Framebuffer>) {
        let mut made = Vec::with_capacity(PASSES);
        let device = &self.device;
        let buffers = &self.slots[at].buffers;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let mut submissions = Vec::with_capacity(PASSES / GROUP + 1);
        for group in 0..PASSES.div_ceil(GROUP) {
            let passes = group * GROUP..((group + 1) * GROUP).min(PASSES);
            let mut commands = Vec::with_capacity(GROUP);
            // SAFETY: each primary is reset (its pool) and begun once a frame.
            unsafe {
                match shape.buffers {
                    Buffers::PerPass => {
                        for pass in passes {
                            let command = buffers[pass];
                            device.begin_command_buffer(command, &begin).unwrap();
                            self.set_states(command);
                            self.pass(command, shape.instance, &mut made);
                            device.end_command_buffer(command).unwrap();
                            commands.push(command);
                        }
                    }
                    Buffers::PerGroup | Buffers::PerGroupRestated => {
                        let command = buffers[group];
                        device.begin_command_buffer(command, &begin).unwrap();
                        for pass in passes.clone() {
                            if pass == passes.start || shape.buffers == Buffers::PerGroupRestated {
                                self.set_states(command);
                            }
                            self.pass(command, shape.instance, &mut made);
                        }
                        device.end_command_buffer(command).unwrap();
                        commands.push(command);
                    }
                }
            }
            submissions.push(commands);
        }
        (submissions, made)
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        let device = &self.device;
        // SAFETY: the queue goes idle first; then nothing uses any of these.
        unsafe {
            device.queue_wait_idle(self.queue).unwrap();
            for slot in &self.slots {
                for &target in &slot.framebuffers {
                    device.destroy_framebuffer(target, None);
                }
                device.destroy_fence(slot.fence, None);
                device.destroy_command_pool(slot.pool, None);
            }
            device.destroy_pipeline(self.pass_pipeline, None);
            device.destroy_pipeline(self.dynamic_pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_framebuffer(self.kept, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
            device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A framebuffer of `render_pass` over `view`.
fn framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    view: vk::ImageView,
) -> vk::Framebuffer {
    let views = [view];
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&views)
        .width(EXTENT.width)
        .height(EXTENT.height)
        .layers(1);
    // SAFETY: `info` and the view are live for the call; the view is of `EXTENT`.
    unsafe { device.create_framebuffer(&info, None) }.unwrap()
}

/// The two pipelines: of `render_pass`'s subpass, and of dynamic rendering into [`FORMAT`]; every
/// state the runtime's draws record is dynamic in both (the color write mask where `color_mask`).
fn pipelines(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    color_mask: bool,
) -> (vk::Pipeline, vk::Pipeline) {
    let module = |code: &[u32]| {
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        // SAFETY: valid SPIR-V of Vulkan 1.0.
        unsafe { device.create_shader_module(&info, None) }.unwrap()
    };
    let (vertex, fragment) = (module(&VERTEX_SPIRV), module(&FRAGMENT_SPIRV));
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default();
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let mut states = vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::DEPTH_BIAS,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::CULL_MODE,
        vk::DynamicState::FRONT_FACE,
        vk::DynamicState::PRIMITIVE_TOPOLOGY,
        vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::DEPTH_COMPARE_OP,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_OP,
    ];
    if color_mask {
        states.push(vk::DynamicState::COLOR_WRITE_MASK_EXT);
    }
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
    let base = || {
        vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blending)
            .dynamic_state(&dynamic)
            .layout(layout)
    };
    let formats = [FORMAT];
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let infos = [
        base().render_pass(render_pass).subpass(0),
        base().push_next(&mut rendering),
    ];
    // SAFETY: `infos` and what they point at live for the call; the device has dynamic rendering
    // on and, where `color_mask`, the color write mask's dynamic state.
    let made = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None) }
        .map_err(|(_, error)| error)
        .unwrap();
    // SAFETY: the pipelines no longer need the modules.
    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }
    (made[0], made[1])
}

/// The value after `flag` on the command line, parsed, else `default`.
fn option(arguments: &[String], flag: &str, default: usize) -> usize {
    match arguments.iter().position(|argument| argument == flag) {
        Some(at) => (arguments.get(at + 1))
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("{flag} takes a count")),
        None => default,
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let frames = option(&arguments, "--frames", 200);
    let repeats = option(&arguments, "--repeats", 5).max(1);
    let draws = option(&arguments, "--draws", 1) as u32;
    let mut bench = Bench::new(draws);
    for shape in SHAPES {
        bench.run(shape, 20);
    }
    let mut runs: Vec<Vec<Phases>> = vec![Vec::new(); SHAPES.len()];
    for _ in 0..repeats {
        for (index, shape) in SHAPES.iter().enumerate() {
            runs[index].push(bench.run(*shape, frames));
        }
    }
    let medians: Vec<Phases> = (runs.iter())
        .map(|phases| Phases {
            record: median(phases.iter().map(|p| p.record).collect()),
            submit: median(phases.iter().map(|p| p.submit).collect()),
            reclaim: median(phases.iter().map(|p| p.reclaim).collect()),
        })
        .collect();
    for (shape, phases) in SHAPES.iter().zip(&medians) {
        println!(
            "PASS-COSTS {:<26} record_ns {:8.0} submit_ns {:7.0} reclaim_ns {:7.0} total_ns {:8.0}",
            shape.name(),
            phases.record,
            phases.submit,
            phases.reclaim,
            phases.total(),
        );
    }
    for (from, to, what) in DIFFERENCES {
        let (a, b) = (medians[from], medians[to]);
        println!(
            "PASS-COSTS difference {} - {}: {:7.0} ns a pass ({what})",
            SHAPES[from].name(),
            SHAPES[to].name(),
            a.total() - b.total(),
        );
    }
}
