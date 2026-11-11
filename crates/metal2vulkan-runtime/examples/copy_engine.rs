//! Not a consumer's example: what a copy between host memory and device memory costs on a queue
//! of the device's graphics family, where the runtime records its shadows' copies and its first
//! work, and on a queue of a family of transfers alone (a copy engine, the family the runtime's
//! staging queue takes: `src/device/staging.rs`), alone and beside a compute load on another
//! queue of the graphics family. It uses no runtime: a Vulkan device of its own, made through the
//! loader, on the first discrete GPU (else integrated, virtual or CPU device) the loader lists.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example copy_engine
//! ```
//!
//! The device takes the graphics family's first two queues (copies on the first, the load on the
//! second) and the first queue of the first family of transfers alone whose copies take any texel
//! (`copy`), else of the first other family of compute work (`compute`), as the runtime's staging
//! queue takes them. It prints `COPY-ENGINE` lines:
//!
//! - **Copies**: host-visible memory the device does not hold (the host's pages, a shadow's or a
//!   `Managed` buffer's host side) to and from device-local memory: into and out of an optimal
//!   `R8G8B8A8_UNORM` image [`WIDTH`] texels wide (`image`), and a buffer (`buffer`); each one
//!   submission waited for. GPU time from its timestamps, wall time from `vkQueueSubmit` to the
//!   wait's return, and the bytes a second of GPU time. First, 4 MiB round trip through the
//!   image on each queue and are checked.
//! - **Load**: a kernel of [`LOAD_WORDS`] invocations of integer arithmetic, sized to about
//!   [`LOAD_MS`] a dispatch, two in flight on the graphics family's second queue; each phase
//!   beside it prints the dispatches completed a second, against the load alone before it.
//! - **Handoff**: a copy into the device buffer and then a dispatch that reads it, on the graphics
//!   queue in one submission (`graphics`), or the copy on the other queue and the dispatch on the
//!   graphics queue waiting for its timeline semaphore (`copy`, `compute`): wall time from the
//!   first `vkQueueSubmit` to the dispatch's completion, what moving a copy the next commands wait
//!   for costs them.
//!
//! Each phase prints its span on the host's clock (`from`, `to`, Unix milliseconds), so a
//! sampler's lines (`nvidia-smi dmon`) can be placed in it.

use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The texels of a row of the image, and so a row's bytes.
const WIDTH: u32 = 8192;
const ROW: u64 = WIDTH as u64 * 4;
/// The most bytes a copy moves: the host buffer's, the device buffer's and the image's size.
const MOST: u64 = 256 << 20;
/// The copies timed beside the load and alone, a phase each, and how long a phase runs.
const LARGE: [u64; 2] = [64 << 20, 256 << 20];
const LARGE_SPAN: Duration = Duration::from_secs(4);
/// The copies timed alone, and how long a phase runs.
const SMALL: [u64; 4] = [64 << 10, 1 << 20, 4 << 20, 16 << 20];
const SMALL_SPAN: Duration = Duration::from_millis(1500);
/// The handoffs' copies, and how long a phase runs.
const HANDOFF: [u64; 5] = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20];
/// The fewest copies a phase times.
const LEAST: usize = 5;
/// The load's invocations, one word of its buffer each (32768 workgroups of 64: within the
/// 65535 every device dispatches along x), and the GPU time a dispatch aims at.
const LOAD_WORDS: u64 = 32768 * 64;
const LOAD_MS: f64 = 4.0;
/// How long one wait lasts before the test fails.
const TIMEOUT: u64 = 20_000_000_000;

/// The load kernel: each invocation of 64 a workgroup reads its word, runs `rounds` steps of a
/// linear congruential generator on it and stores it. Compiled from this GLSL with glslang
/// (`--target-env vulkan1.0`, debug names stripped):
///
/// ```glsl
/// #version 450
/// layout(local_size_x = 64) in;
/// layout(std430, set = 0, binding = 0) buffer Words { uint words[]; };
/// layout(push_constant) uniform Push { uint rounds; };
/// void main() {
///     uint i = gl_GlobalInvocationID.x;
///     uint v = words[i];
///     for (uint k = 0u; k < rounds; k++) {
///         v = v * 1664525u + 1013904223u;
///     }
///     words[i] = v;
/// }
/// ```
#[rustfmt::skip]
const LOAD_SPIRV: [u32; 309] = [
    0x07230203, 0x00010000, 0x0008000b, 0x00000038, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c,
    0x00040047, 0x00000011, 0x00000006, 0x00000004, 0x00030047, 0x00000012, 0x00000003, 0x00050048,
    0x00000012, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000014, 0x00000021, 0x00000000,
    0x00040047, 0x00000014, 0x00000022, 0x00000000, 0x00030047, 0x00000022, 0x00000002, 0x00050048,
    0x00000022, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000037, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009, 0x00000006,
    0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001,
    0x00000006, 0x0003001d, 0x00000011, 0x00000006, 0x0003001e, 0x00000012, 0x00000011, 0x00040020,
    0x00000013, 0x00000002, 0x00000012, 0x0004003b, 0x00000013, 0x00000014, 0x00000002, 0x00040015,
    0x00000015, 0x00000020, 0x00000001, 0x0004002b, 0x00000015, 0x00000016, 0x00000000, 0x00040020,
    0x00000018, 0x00000002, 0x00000006, 0x0003001e, 0x00000022, 0x00000006, 0x00040020, 0x00000023,
    0x00000009, 0x00000022, 0x0004003b, 0x00000023, 0x00000024, 0x00000009, 0x00040020, 0x00000025,
    0x00000009, 0x00000006, 0x00020014, 0x00000028, 0x0004002b, 0x00000006, 0x0000002b, 0x0019660d,
    0x0004002b, 0x00000006, 0x0000002d, 0x3c6ef35f, 0x0004002b, 0x00000015, 0x00000030, 0x00000001,
    0x0004002b, 0x00000006, 0x00000035, 0x00000040, 0x0004002b, 0x00000006, 0x00000036, 0x00000001,
    0x0006002c, 0x00000009, 0x00000037, 0x00000035, 0x00000036, 0x00000036, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008,
    0x00000007, 0x0004003b, 0x00000007, 0x00000010, 0x00000007, 0x0004003b, 0x00000007, 0x0000001b,
    0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006,
    0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000017,
    0x00000008, 0x00060041, 0x00000018, 0x00000019, 0x00000014, 0x00000016, 0x00000017, 0x0004003d,
    0x00000006, 0x0000001a, 0x00000019, 0x0003003e, 0x00000010, 0x0000001a, 0x0003003e, 0x0000001b,
    0x0000000c, 0x000200f9, 0x0000001c, 0x000200f8, 0x0000001c, 0x000400f6, 0x0000001e, 0x0000001f,
    0x00000000, 0x000200f9, 0x00000020, 0x000200f8, 0x00000020, 0x0004003d, 0x00000006, 0x00000021,
    0x0000001b, 0x00050041, 0x00000025, 0x00000026, 0x00000024, 0x00000016, 0x0004003d, 0x00000006,
    0x00000027, 0x00000026, 0x000500b0, 0x00000028, 0x00000029, 0x00000021, 0x00000027, 0x000400fa,
    0x00000029, 0x0000001d, 0x0000001e, 0x000200f8, 0x0000001d, 0x0004003d, 0x00000006, 0x0000002a,
    0x00000010, 0x00050084, 0x00000006, 0x0000002c, 0x0000002a, 0x0000002b, 0x00050080, 0x00000006,
    0x0000002e, 0x0000002c, 0x0000002d, 0x0003003e, 0x00000010, 0x0000002e, 0x000200f9, 0x0000001f,
    0x000200f8, 0x0000001f, 0x0004003d, 0x00000006, 0x0000002f, 0x0000001b, 0x00050080, 0x00000006,
    0x00000031, 0x0000002f, 0x00000030, 0x0003003e, 0x0000001b, 0x00000031, 0x000200f9, 0x0000001c,
    0x000200f8, 0x0000001e, 0x0004003d, 0x00000006, 0x00000032, 0x00000008, 0x0004003d, 0x00000006,
    0x00000033, 0x00000010, 0x00060041, 0x00000018, 0x00000034, 0x00000014, 0x00000016, 0x00000032,
    0x0003003e, 0x00000034, 0x00000033, 0x000100fd, 0x00010038,
];

/// Where a copy runs.
#[derive(Clone, Copy, PartialEq)]
enum On {
    /// The graphics family's first queue.
    Graphics,
    /// The other family's queue: transfers alone, or compute work.
    Other,
}

/// What a copy moves between.
#[derive(Clone, Copy)]
enum Kind {
    Image,
    Buffer,
}

/// Which way a copy goes.
#[derive(Clone, Copy)]
enum Way {
    /// Host memory into device memory.
    Up,
    /// Device memory into host memory.
    Down,
}

/// A queue and the command pool of its family, for the main thread.
struct Lane {
    family: u32,
    queue: vk::Queue,
    pool: vk::CommandPool,
    /// The family writes timestamps.
    timed: bool,
    label: &'static str,
}

/// The test's own device and its objects.
struct Bench {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    graphics: Lane,
    other: Option<Lane>,
    /// The graphics family's second queue, where the load runs.
    load_queue: Option<vk::Queue>,
    /// Nanoseconds a timestamp tick.
    period: f64,
    host: (vk::Buffer, vk::DeviceMemory),
    device_buffer: (vk::Buffer, vk::DeviceMemory),
    image: (vk::Image, vk::DeviceMemory),
    load_buffer: (vk::Buffer, vk::DeviceMemory),
    descriptors: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    /// The load buffer's set, and the device buffer's (the handoffs' dispatch).
    load_set: vk::DescriptorSet,
    device_set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    queries: vk::QueryPool,
    timeline: vk::Semaphore,
    /// The timeline's last value signaled.
    value: u64,
}

/// Unix milliseconds now, on the host's clock.
fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis())
}

/// The 50th and 90th percentiles of `values`, nearest rank.
fn percentiles(mut values: Vec<f64>) -> [f64; 2] {
    if values.is_empty() {
        return [f64::NAN; 2];
    }
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[((values.len() * p).div_ceil(100)).clamp(1, values.len()) - 1];
    [at(50), at(90)]
}

/// The first memory type of `types` that `bits` admits, holds `wanted` and none of `unwanted`,
/// preferring one that also holds `preferred`.
fn memory_type(
    types: &[vk::MemoryType],
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
    unwanted: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let fits = |index: usize| {
        bits & (1 << index) != 0
            && types[index].property_flags.contains(wanted)
            && !types[index].property_flags.intersects(unwanted)
    };
    let all = 0..types.len();
    (all.clone())
        .find(|&index| fits(index) && types[index].property_flags.contains(preferred))
        .or_else(|| all.clone().find(|&index| fits(index)))
        .map(|index| index as u32)
}

impl Bench {
    fn new() -> Bench {
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
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
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
        let kind = |physical: vk::PhysicalDevice| unsafe {
            instance
                .get_physical_device_properties(physical)
                .device_type
        };
        let physical = (types.iter())
            .find_map(|&wanted| physicals.iter().copied().find(|&p| kind(p) == wanted))
            .expect("a device");
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance's; Vulkan 1.1's command.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let properties = properties.properties;
        let name = (properties.device_name_as_c_str())
            .map_or("?".to_owned(), |name| name.to_string_lossy().into_owned());
        let driver_info = (driver.driver_info_as_c_str())
            .map_or("?".to_owned(), |info| info.to_string_lossy().into_owned());
        // SAFETY: the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let graphics = (families.iter())
            .position(|family| family.queue_flags.contains(wanted) && family.queue_count > 0)
            .expect("a family of graphics and compute work") as u32;
        let unit = vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        };
        let besides =
            vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING | vk::QueueFlags::PROTECTED;
        let other_of = |test: &dyn Fn(vk::QueueFlags) -> bool| {
            (families.iter().enumerate()).position(|(index, family)| {
                index as u32 != graphics
                    && family.queue_count > 0
                    && family.min_image_transfer_granularity == unit
                    && test(family.queue_flags)
            })
        };
        let copy = other_of(&|flags| {
            flags.contains(vk::QueueFlags::TRANSFER) && (flags & !besides).is_empty()
        });
        let compute = other_of(&|flags| flags.contains(vk::QueueFlags::COMPUTE));
        let other = (copy.map(|family| (family as u32, "copy")))
            .or(compute.map(|family| (family as u32, "compute")));
        for (index, family) in families.iter().enumerate() {
            println!(
                "COPY-ENGINE family {index} flags {:#x} queues {} timestamp bits {} granularity {}x{}x{}",
                family.queue_flags.as_raw(),
                family.queue_count,
                family.timestamp_valid_bits,
                family.min_image_transfer_granularity.width,
                family.min_image_transfer_granularity.height,
                family.min_image_transfer_granularity.depth,
            );
        }
        let graphics_queues = families[graphics as usize].queue_count.min(2);
        let priorities = [1.0, 1.0];
        let mut queue_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics)
            .queue_priorities(&priorities[..graphics_queues as usize])];
        if let Some((family, _)) = other {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities[..1]),
            );
        }
        let mut features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .host_query_reset(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut features);
        // SAFETY: `info` and what it points at live for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        let lane = |family: u32, index: u32, label: &'static str| {
            // SAFETY: the family and index were created; `info` is valid.
            let (queue, pool) = unsafe {
                (
                    device.get_device_queue(family, index),
                    device
                        .create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                                .queue_family_index(family),
                            None,
                        )
                        .unwrap(),
                )
            };
            Lane {
                family,
                queue,
                pool,
                timed: families[family as usize].timestamp_valid_bits > 0,
                label,
            }
        };
        let graphics_lane = lane(graphics, 0, "graphics");
        let other_lane = other.map(|(family, label)| lane(family, 0, label));
        let load_queue = (graphics_queues > 1).then(|| {
            // SAFETY: the family's second queue was created.
            unsafe { device.get_device_queue(graphics, 1) }
        });
        // SAFETY: the instance's.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_types = &memory.memory_types[..memory.memory_type_count as usize];
        let shared: Vec<u32> = std::iter::once(graphics)
            .chain(other.map(|(family, _)| family))
            .collect();
        let usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let buffer = |size: u64, host: bool| {
            let info = vk::BufferCreateInfo::default().size(size).usage(usage);
            let info = match shared.len() > 1 {
                true => info
                    .sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&shared),
                false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
            };
            // SAFETY: the device is live and `info` valid.
            let buffer = unsafe { device.create_buffer(&info, None) }.unwrap();
            // SAFETY: the buffer is the device's.
            let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
            let index = match host {
                // The host's own memory, which the device reaches across the bus where it has its
                // own.
                true => memory_type(
                    memory_types,
                    needs.memory_type_bits,
                    host_flags,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                )
                .or_else(|| {
                    memory_type(
                        memory_types,
                        needs.memory_type_bits,
                        host_flags,
                        vk::MemoryPropertyFlags::empty(),
                        vk::MemoryPropertyFlags::HOST_CACHED,
                    )
                }),
                false => memory_type(
                    memory_types,
                    needs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    vk::MemoryPropertyFlags::empty(),
                )
                .or_else(|| {
                    memory_type(
                        memory_types,
                        needs.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        vk::MemoryPropertyFlags::empty(),
                        vk::MemoryPropertyFlags::empty(),
                    )
                }),
            }
            .expect("a memory type for the buffer");
            println!(
                "COPY-ENGINE {} buffer memory type {index} flags {:#x} heap {}",
                if host { "host" } else { "device" },
                memory_types[index as usize].property_flags.as_raw(),
                memory_types[index as usize].heap_index,
            );
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(index);
            // SAFETY: as above; the memory is bound once, at offset 0.
            unsafe {
                let memory = device.allocate_memory(&allocate, None).unwrap();
                device.bind_buffer_memory(buffer, memory, 0).unwrap();
                (buffer, memory)
            }
        };
        let host = buffer(MOST, true);
        let device_buffer = buffer(MOST, false);
        let load_buffer = buffer(LOAD_WORDS * 4, false);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: WIDTH,
                height: (MOST / ROW) as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image_info = match shared.len() > 1 {
            true => image_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&shared),
            false => image_info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        };
        // SAFETY: the device is live and every info valid.
        let image = unsafe {
            let image = device.create_image(&image_info, None).unwrap();
            let needs = device.get_image_memory_requirements(image);
            let index = memory_type(
                memory_types,
                needs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
                vk::MemoryPropertyFlags::empty(),
            )
            .expect("a device-local type for the image");
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(needs.size)
                        .memory_type_index(index),
                    None,
                )
                .unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            (image, memory)
        };
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)];
        // SAFETY: the device is live and every info valid.
        let (set_layout, layout, pipeline, descriptors, load_set, device_set) = unsafe {
            let set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .unwrap();
            let layouts = [set_layout];
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .unwrap();
            let module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&LOAD_SPIRV),
                    None,
                )
                .unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, error)| error)
                .unwrap()[0];
            device.destroy_shader_module(module, None);
            let descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&sizes),
                    None,
                )
                .unwrap();
            let both = [set_layout, set_layout];
            let sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptors)
                        .set_layouts(&both),
                )
                .unwrap();
            for (set, buffer) in [(sets[0], load_buffer.0), (sets[1], device_buffer.0)] {
                let infos = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos);
                device.update_descriptor_sets(&[write], &[]);
            }
            (set_layout, layout, pipeline, descriptors, sets[0], sets[1])
        };
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: the device is live and every info valid.
        let (queries, timeline) = unsafe {
            (
                device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .unwrap(),
                device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                        None,
                    )
                    .unwrap(),
            )
        };
        println!(
            "COPY-ENGINE device {name} driver {} {driver_info} graphics family {graphics} ({} queues) other {}",
            driver.driver_id.as_raw(),
            graphics_queues,
            match other {
                Some((family, label)) => format!("{label} family {family}"),
                None => "none".to_owned(),
            },
        );
        let mut bench = Bench {
            _entry: entry,
            instance,
            device,
            graphics: graphics_lane,
            other: other_lane,
            load_queue,
            period: f64::from(properties.limits.timestamp_period),
            host,
            device_buffer,
            image,
            load_buffer,
            descriptors,
            set_layout,
            load_set,
            device_set,
            layout,
            pipeline,
            queries,
            timeline,
            value: 0,
        };
        // The image rests in GENERAL, as the runtime's do.
        bench.run(On::Graphics, |device, command| {
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.0)
                .subresource_range(color_range());
            // SAFETY: recording.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            };
        });
        bench
    }

    fn lane(&self, on: On) -> &Lane {
        match on {
            On::Graphics => &self.graphics,
            On::Other => self.other.as_ref().expect("another family's queue"),
        }
    }

    /// A new primary command buffer of `on`'s pool.
    fn command(&self, on: On) -> vk::CommandBuffer {
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.lane(on).pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is the device's, used by this thread alone.
        unsafe { self.device.allocate_command_buffers(&allocate) }.unwrap()[0]
    }

    /// Record `record` into a new command buffer of `on`, submit it and wait for it.
    fn run(&mut self, on: On, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        let command = self.command(on);
        // SAFETY: a new command buffer, recorded here alone.
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            record(&self.device, command);
            self.device.end_command_buffer(command).unwrap();
        }
        self.submit(on, command, None);
        self.wait();
        // SAFETY: the command buffer completed.
        unsafe {
            (self.device).free_command_buffers(self.lane(on).pool, &[command]);
        }
    }

    /// Submit `command` on `on`'s queue, waiting first for the timeline's `wait` (at every stage)
    /// where given; it signals the timeline's next value.
    fn submit(&mut self, on: On, command: vk::CommandBuffer, wait: Option<u64>) {
        self.value += 1;
        let waits: Vec<u64> = wait.into_iter().collect();
        let wait_semaphores = vec![self.timeline; waits.len()];
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signals = [self.value];
        let mut values = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&waits)
            .signal_semaphore_values(&signals);
        let commands = [command];
        let semaphores = [self.timeline];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&commands)
            .signal_semaphores(&semaphores)
            .push_next(&mut values);
        // SAFETY: the queue is used by this thread alone; the command buffer is executable and
        // not pending; the signal is above the timeline's value.
        unsafe {
            (self.device)
                .queue_submit(self.lane(on).queue, &[submit], vk::Fence::null())
                .unwrap()
        };
    }

    /// Wait on the host for the timeline's last value signaled.
    fn wait(&self) {
        let semaphores = [self.timeline];
        let values = [self.value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is the device's.
        unsafe { self.device.wait_semaphores(&info, TIMEOUT) }.unwrap();
    }

    /// A command buffer of `on` that moves `bytes` `way` between host memory and the image or
    /// device buffer (`kind`), timed by the query pool's two timestamps where the family writes
    /// them.
    fn copy_command(&self, on: On, kind: Kind, way: Way, bytes: u64) -> vk::CommandBuffer {
        let command = self.command(on);
        let timed = self.lane(on).timed;
        let device = &self.device;
        let all = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: WIDTH,
                height: (bytes / ROW) as u32,
                depth: 1,
            });
        let range = vk::BufferCopy::default().size(bytes);
        // SAFETY: a new command buffer, recorded here alone; every object is the device's.
        unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[all],
                &[],
                &[],
            );
            if timed {
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    0,
                );
            }
            match (kind, way) {
                (Kind::Image, Way::Up) => device.cmd_copy_buffer_to_image(
                    command,
                    self.host.0,
                    self.image.0,
                    vk::ImageLayout::GENERAL,
                    &[region],
                ),
                (Kind::Image, Way::Down) => device.cmd_copy_image_to_buffer(
                    command,
                    self.image.0,
                    vk::ImageLayout::GENERAL,
                    self.host.0,
                    &[region],
                ),
                (Kind::Buffer, Way::Up) => {
                    device.cmd_copy_buffer(command, self.host.0, self.device_buffer.0, &[range])
                }
                (Kind::Buffer, Way::Down) => {
                    device.cmd_copy_buffer(command, self.device_buffer.0, self.host.0, &[range])
                }
            }
            if timed {
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    1,
                );
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[to_host],
                &[],
                &[],
            );
            device.end_command_buffer(command).unwrap();
        }
        command
    }

    /// The query pool's two timestamps apart, in milliseconds.
    fn gpu_ms(&self) -> f64 {
        let mut ticks = [0u64; 2];
        // SAFETY: both queries were written by a submission the host saw complete.
        unsafe {
            self.device.get_query_pool_results(
                self.queries,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .unwrap();
        ticks[1].wrapping_sub(ticks[0]) as f64 * self.period / 1e6
    }

    /// Copies of `bytes` `way` on `on`, one submission waited for at a time, for `span` and at
    /// least [`LEAST`] of them: their GPU and wall times in milliseconds.
    fn copies(&mut self, on: On, kind: Kind, way: Way, bytes: u64, span: Duration) -> Phase {
        let command = self.copy_command(on, kind, way, bytes);
        let timed = self.lane(on).timed;
        let from = unix_ms();
        let began = Instant::now();
        let (mut gpu, mut wall) = (Vec::new(), Vec::new());
        while began.elapsed() < span || wall.len() < LEAST {
            // SAFETY: the queries are not in use: the last submission completed.
            unsafe { self.device.reset_query_pool(self.queries, 0, 2) };
            let start = Instant::now();
            self.submit(on, command, None);
            self.wait();
            wall.push(start.elapsed().as_secs_f64() * 1e3);
            if timed {
                gpu.push(self.gpu_ms());
            }
        }
        let to = unix_ms();
        // SAFETY: the command buffer completed.
        unsafe { (self.device).free_command_buffers(self.lane(on).pool, &[command]) };
        Phase {
            from,
            to,
            gpu,
            wall,
        }
    }

    /// A command buffer of the graphics queue that dispatches the load kernel once over the
    /// device buffer's first workgroup (`rounds` 0), after a copy into it where `copy` (bytes)
    /// is given, with a barrier between.
    fn dispatch_command(&self, copy: Option<u64>) -> vk::CommandBuffer {
        let command = self.command(On::Graphics);
        let device = &self.device;
        // SAFETY: a new command buffer, recorded here alone; every object is the device's.
        unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            if let Some(bytes) = copy {
                let range = vk::BufferCopy::default().size(bytes);
                device.cmd_copy_buffer(command, self.host.0, self.device_buffer.0, &[range]);
                let after = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[after],
                    &[],
                    &[],
                );
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.device_set],
                &[],
            );
            device.cmd_push_constants(
                command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &0u32.to_ne_bytes(),
            );
            device.cmd_dispatch(command, 1, 1, 1);
            device.end_command_buffer(command).unwrap();
        }
        command
    }

    /// Handoffs of `bytes` (the module docs), for [`SMALL_SPAN`] and at least [`LEAST`]: on the
    /// graphics queue alone, or the copy on the other queue: their wall times in milliseconds.
    fn handoffs(&mut self, on: On, bytes: u64) -> Phase {
        let (first, then) = match on {
            On::Graphics => (self.dispatch_command(Some(bytes)), None),
            On::Other => {
                let copy = self.command(On::Other);
                let range = vk::BufferCopy::default().size(bytes);
                // SAFETY: a new command buffer, recorded here alone.
                unsafe {
                    (self.device)
                        .begin_command_buffer(copy, &vk::CommandBufferBeginInfo::default())
                        .unwrap();
                    (self.device).cmd_copy_buffer(
                        copy,
                        self.host.0,
                        self.device_buffer.0,
                        &[range],
                    );
                    self.device.end_command_buffer(copy).unwrap();
                }
                (copy, Some(self.dispatch_command(None)))
            }
        };
        let from = unix_ms();
        let began = Instant::now();
        let mut wall = Vec::new();
        while began.elapsed() < SMALL_SPAN || wall.len() < LEAST {
            let start = Instant::now();
            self.submit(on, first, None);
            if let Some(dispatch) = then {
                // The semaphore's signal and wait order the copy's writes before the dispatch:
                // the buffer is concurrent between the two families, so no ownership moves.
                let copied = self.value;
                self.submit(On::Graphics, dispatch, Some(copied));
            }
            self.wait();
            wall.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let to = unix_ms();
        // SAFETY: every submission completed.
        unsafe {
            (self.device).free_command_buffers(self.lane(on).pool, &[first]);
            if let Some(dispatch) = then {
                (self.device).free_command_buffers(self.graphics.pool, &[dispatch]);
            }
        }
        Phase {
            from,
            to,
            gpu: Vec::new(),
            wall,
        }
    }

    /// The load's rounds for a dispatch of about [`LOAD_MS`], and the dispatch's GPU time then,
    /// measured on the graphics queue alone.
    fn calibrate(&mut self) -> (u32, f64) {
        let groups = (LOAD_WORDS / 64) as u32;
        let time = |bench: &mut Bench, rounds: u32| {
            let (queries, layout, pipeline, set) =
                (bench.queries, bench.layout, bench.pipeline, bench.load_set);
            // SAFETY: the queries are not in use.
            unsafe { bench.device.reset_query_pool(queries, 0, 2) };
            bench.run(On::Graphics, |device, command| {
                // SAFETY: recording.
                unsafe {
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        queries,
                        0,
                    );
                    record_load(device, command, (pipeline, layout, set), groups, rounds);
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        queries,
                        1,
                    );
                }
            });
            bench.gpu_ms()
        };
        let probe = 256;
        let ms = time(self, probe).max(0.01);
        let rounds = ((f64::from(probe) * LOAD_MS / ms) as u32).clamp(1, 1 << 22);
        let ms = time(self, rounds);
        (rounds, ms)
    }
}

/// The color aspect's one level and layer.
fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

/// Record one dispatch of the load kernel of `groups` workgroups and `rounds` rounds over `set`.
///
/// # Safety
///
/// `command` is recording, of a family of compute work; the objects are its device's.
unsafe fn record_load(
    device: &ash::Device,
    command: vk::CommandBuffer,
    (pipeline, layout, set): (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSet),
    groups: u32,
    rounds: u32,
) {
    // SAFETY: the caller's contract.
    unsafe {
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        device.cmd_push_constants(
            command,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &rounds.to_ne_bytes(),
        );
        device.cmd_dispatch(command, groups, 1, 1);
    }
}

/// A phase's span and times.
struct Phase {
    from: u128,
    to: u128,
    gpu: Vec<f64>,
    wall: Vec<f64>,
}

impl Phase {
    fn line(&self, bytes: u64) -> String {
        let [gpu50, gpu90] = percentiles(self.gpu.clone());
        let [wall50, wall90] = percentiles(self.wall.clone());
        let rate = bytes as f64 / (gpu50 * 1e6);
        format!(
            "n {} gpu_ms p50 {gpu50:.3} p90 {gpu90:.3} wall_ms p50 {wall50:.3} p90 {wall90:.3} \
             gpu_GB/s {rate:.2} from {} to {}",
            self.wall.len(),
            self.from,
            self.to,
        )
    }
}

/// The load running on the graphics family's second queue, two dispatches in flight, until it is
/// stopped: the dispatches it completed, and their GPU time in nanoseconds.
struct Load {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicU64>,
    busy: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Load {
    fn start(bench: &Bench, queue: vk::Queue, rounds: u32) -> Load {
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicU64::new(0));
        let busy = Arc::new(AtomicU64::new(0));
        let device = bench.device.clone();
        let family = bench.graphics.family;
        let objects = (bench.pipeline, bench.layout, bench.load_set);
        let period = bench.period;
        let (thread_stop, thread_done, thread_busy) = (stop.clone(), done.clone(), busy.clone());
        let thread = std::thread::spawn(move || {
            let groups = (LOAD_WORDS / 64) as u32;
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            // SAFETY: the device is live; every object here is this thread's alone, and the
            // queue is used by it alone.
            unsafe {
                let pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                        None,
                    )
                    .unwrap();
                let commands = device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(2),
                    )
                    .unwrap();
                let queries = device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(4),
                        None,
                    )
                    .unwrap();
                let timeline = device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                        None,
                    )
                    .unwrap();
                for (slot, &command) in commands.iter().enumerate() {
                    let first = 2 * slot as u32;
                    device
                        .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                        .unwrap();
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        queries,
                        first,
                    );
                    record_load(&device, command, objects, groups, rounds);
                    device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        queries,
                        first + 1,
                    );
                    device.end_command_buffer(command).unwrap();
                }
                let mut submitted = [0u64; 2];
                let mut value = 0u64;
                let mut slot = 0;
                while !thread_stop.load(Ordering::Relaxed) {
                    let first = 2 * slot as u32;
                    if submitted[slot] > 0 {
                        let semaphores = [timeline];
                        let values = [submitted[slot]];
                        let info = vk::SemaphoreWaitInfo::default()
                            .semaphores(&semaphores)
                            .values(&values);
                        device.wait_semaphores(&info, TIMEOUT).unwrap();
                        let mut ticks = [0u64; 2];
                        device
                            .get_query_pool_results(
                                queries,
                                first,
                                &mut ticks,
                                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                            )
                            .unwrap();
                        let ns = ticks[1].wrapping_sub(ticks[0]) as f64 * period;
                        thread_busy.fetch_add(ns as u64, Ordering::Relaxed);
                        thread_done.fetch_add(1, Ordering::Relaxed);
                    }
                    device.reset_query_pool(queries, first, 2);
                    value += 1;
                    let signals = [value];
                    let mut values = vk::TimelineSemaphoreSubmitInfo::default()
                        .signal_semaphore_values(&signals);
                    let buffers = [commands[slot]];
                    let semaphores = [timeline];
                    let submit = vk::SubmitInfo::default()
                        .command_buffers(&buffers)
                        .signal_semaphores(&semaphores)
                        .push_next(&mut values);
                    device
                        .queue_submit(queue, &[submit], vk::Fence::null())
                        .unwrap();
                    submitted[slot] = value;
                    slot = 1 - slot;
                }
                let semaphores = [timeline];
                let values = [value];
                let info = vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values);
                device.wait_semaphores(&info, TIMEOUT).unwrap();
                device.destroy_semaphore(timeline, None);
                device.destroy_query_pool(queries, None);
                device.destroy_command_pool(pool, None);
            }
        });
        Load {
            stop,
            done,
            busy,
            thread: Some(thread),
        }
    }

    /// The dispatches completed so far and their GPU time in nanoseconds.
    fn read(&self) -> (u64, u64) {
        (
            self.done.load(Ordering::Relaxed),
            self.busy.load(Ordering::Relaxed),
        )
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The load's rate over a span: dispatches a second, and the GPU time a dispatch in
/// milliseconds.
fn load_rate(before: (u64, u64), after: (u64, u64), seconds: f64) -> (f64, f64) {
    let count = after.0 - before.0;
    let busy = (after.1 - before.1) as f64 / 1e6;
    (count as f64 / seconds, busy / count.max(1) as f64)
}

impl Drop for Bench {
    fn drop(&mut self) {
        // SAFETY: nothing is pending (every submission was waited for, the load thread joined);
        // each object is destroyed once.
        unsafe {
            let device = &self.device;
            device.device_wait_idle().unwrap();
            device.destroy_semaphore(self.timeline, None);
            device.destroy_query_pool(self.queries, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.descriptors, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_image(self.image.0, None);
            device.free_memory(self.image.1, None);
            for (buffer, memory) in [self.host, self.device_buffer, self.load_buffer] {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            device.destroy_command_pool(self.graphics.pool, None);
            if let Some(other) = &self.other {
                device.destroy_command_pool(other.pool, None);
            }
            device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn name(kind: Kind, way: Way) -> &'static str {
    match (kind, way) {
        (Kind::Image, Way::Up) => "image up",
        (Kind::Image, Way::Down) => "image down",
        (Kind::Buffer, Way::Up) => "buffer up",
        (Kind::Buffer, Way::Down) => "buffer down",
    }
}

fn main() {
    let mut bench = Bench::new();
    // The bytes round trip exactly through the image on each queue: a pattern up, zeros over it
    // on the host, and back down.
    let check = 4 << 20;
    let ons: Vec<On> = std::iter::once(On::Graphics)
        .chain(bench.other.as_ref().map(|_| On::Other))
        .collect();
    for &on in &ons {
        // SAFETY: the host memory is host-visible and coherent, of `MOST` bytes, and no
        // submission is pending.
        unsafe {
            let mapped = (bench.device)
                .map_memory(bench.host.1, 0, check, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u32;
            let words = std::slice::from_raw_parts_mut(mapped, (check / 4) as usize);
            for (i, word) in words.iter_mut().enumerate() {
                *word = (i as u32).wrapping_mul(2654435761) ^ (on == On::Other) as u32;
            }
            bench.copies(on, Kind::Image, Way::Up, check, Duration::ZERO);
            words.fill(0);
            bench.copies(on, Kind::Image, Way::Down, check, Duration::ZERO);
            let wrong = (words.iter().enumerate())
                .filter(|&(i, &word)| {
                    word != (i as u32).wrapping_mul(2654435761) ^ (on == On::Other) as u32
                })
                .count();
            (bench.device).unmap_memory(bench.host.1);
            assert_eq!(
                wrong,
                0,
                "{} queue: words that did not round trip",
                bench.lane(on).label
            );
        }
    }
    for &bytes in &SMALL {
        for kind in [Kind::Image, Kind::Buffer] {
            for way in [Way::Up, Way::Down] {
                for &on in &ons {
                    let phase = bench.copies(on, kind, way, bytes, SMALL_SPAN);
                    println!(
                        "COPY-ENGINE copy {} {} {bytes} load off {}",
                        bench.lane(on).label,
                        name(kind, way),
                        phase.line(bytes),
                    );
                }
            }
        }
    }
    for &bytes in &HANDOFF {
        for &on in &ons {
            let phase = bench.handoffs(on, bytes);
            println!(
                "COPY-ENGINE handoff {} {bytes} {}",
                bench.lane(on).label,
                phase.line(bytes),
            );
        }
    }
    let (rounds, ms) = bench.calibrate();
    println!("COPY-ENGINE load rounds {rounds} dispatch_ms {ms:.2}");
    for &bytes in &LARGE {
        for kind in [Kind::Image, Kind::Buffer] {
            for way in [Way::Up, Way::Down] {
                for &on in &ons {
                    let phase = bench.copies(on, kind, way, bytes, LARGE_SPAN);
                    println!(
                        "COPY-ENGINE copy {} {} {bytes} load off {}",
                        bench.lane(on).label,
                        name(kind, way),
                        phase.line(bytes),
                    );
                }
            }
        }
    }
    let Some(queue) = bench.load_queue else {
        println!("COPY-ENGINE load none: the graphics family has one queue");
        return;
    };
    let load = Load::start(&bench, queue, rounds);
    let alone = |load: &Load, label: &str| {
        let (from, before, began) = (unix_ms(), load.read(), Instant::now());
        std::thread::sleep(LARGE_SPAN);
        let (rate, ms) = load_rate(before, load.read(), began.elapsed().as_secs_f64());
        println!(
            "COPY-ENGINE load alone {label} rate {rate:.1}/s dispatch_ms {ms:.2} from {from} to {}",
            unix_ms()
        );
        rate
    };
    let first = alone(&load, "first");
    for &bytes in &LARGE {
        for kind in [Kind::Image, Kind::Buffer] {
            for way in [Way::Up, Way::Down] {
                for &on in &ons {
                    let (before, began) = (load.read(), Instant::now());
                    let phase = bench.copies(on, kind, way, bytes, LARGE_SPAN);
                    let (rate, ms) = load_rate(before, load.read(), began.elapsed().as_secs_f64());
                    println!(
                        "COPY-ENGINE copy {} {} {bytes} load on {} load_rate {rate:.1}/s \
                         ({:.0}% of alone) load_dispatch_ms {ms:.2}",
                        bench.lane(on).label,
                        name(kind, way),
                        phase.line(bytes),
                        100.0 * rate / first,
                    );
                }
            }
        }
    }
    alone(&load, "last");
}
