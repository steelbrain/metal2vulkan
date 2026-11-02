//! The Vulkan-side plumbing a consumer's video work needs, on a device: extra queues for the
//! caller (`DeviceOptions::extra_queues`, `DeviceHandles::extra_queues`), the video extensions'
//! feature structures, a shared event's semaphore ordering the caller's raw submissions against
//! command buffers' `encodeWaitForEvent:` and `encodeSignalEvent:` (`SharedEvent::vk_wait`,
//! `SharedEvent::reserve_vk_signal`), and a texture over a plane of the caller's NV12 image
//! (`Device::new_texture_with_vk_image_plane`).
//!
//! The raw submissions go to the extra queue where the device made one, else to a command
//! queue's `VkQueue` (`CommandQueue::with_vk_queue`). Each prints what the device reported
//! (`VIDEO-INTEROP:` lines).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, DeviceKind,
    DeviceOptions, ExtraQueue, Library, MessageSink, PipelineOptions, Region, ResourceOptions,
    ShaderCacheOptions, Size, Texture, TextureDescriptor, VkBufferMemory, VkEventSignal,
    VkEventWait,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The NV12 image: its luma plane's extent (the chroma plane's is half of each).
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;

/// The video extensions a consumer passes (`optional_device_extensions`).
const VIDEO_EXTENSIONS: [&std::ffi::CStr; 8] = [
    c"VK_KHR_video_queue",
    c"VK_KHR_video_decode_queue",
    c"VK_KHR_video_decode_h264",
    c"VK_KHR_video_decode_h265",
    c"VK_KHR_video_maintenance1",
    c"VK_KHR_video_maintenance2",
    c"VK_KHR_unified_image_layouts",
    c"VK_KHR_synchronization2",
];

/// The device the cfg selects, its shader cache in `cache`, with an extra compute queue and an
/// extra video decode queue, the video extensions where it has them, and its messages kept in
/// `messages`.
fn device(cache: &Path, messages: &Arc<Mutex<Vec<String>>>) -> Device {
    let kept = Arc::clone(messages);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        extra_queues: vec![vk::QueueFlags::COMPUTE, vk::QueueFlags::VIDEO_DECODE_KHR],
        optional_device_extensions: VIDEO_EXTENSIONS
            .iter()
            .map(|&name| CString::from(name))
            .collect(),
        messages: Some(MessageSink::new(move |message| {
            kept.lock().unwrap().push(message.text.clone());
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The families of the device's queue family properties.
fn families(device: &Device) -> Vec<vk::QueueFamilyProperties> {
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    }
}

/// The extra queue made for `flags`, if any.
fn extra(device: &Device, flags: vk::QueueFlags) -> Option<ExtraQueue> {
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    raw.extra_queues.iter().find(|q| q.flags == flags).copied()
}

/// Each extra queue is a queue of a family whose flags contain the asked ones, none of the
/// runtime's own, which takes a submission; one the device has no family for is left out and
/// reported in an informational message, and the device is made all the same. A device with a
/// video decode family makes that queue and enables the video extensions' features it has.
#[test]
fn extra_queues_are_made_or_reported_and_the_video_features_enabled() {
    let scratch = Scratch::new("video-interop-queues");
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, &messages);
    let families = families(&device);
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    eprintln!(
        "VIDEO-INTEROP: {}: families {:?}, the runtime's {:?}, extra {:?}",
        device.name(),
        (families.iter())
            .map(|f| (f.queue_flags.as_raw(), f.queue_count))
            .collect::<Vec<_>>(),
        raw.queue_families,
        raw.extra_queues
    );
    let decode_family =
        (families.iter()).position(|f| f.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR));
    for asked in [vk::QueueFlags::COMPUTE, vk::QueueFlags::VIDEO_DECODE_KHR] {
        match extra(&device, asked) {
            Some(queue) => {
                let flags = families[queue.family as usize].queue_flags;
                assert!(flags.contains(asked), "{queue:?}");
                assert_ne!(queue.queue, vk::Queue::null());
                // An empty submission with a fence: the queue takes work.
                // SAFETY: the queue is the device's, submitted to by this test alone; the
                // fence is waited for and destroyed here.
                unsafe {
                    let fence = raw
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .unwrap();
                    raw.device.queue_submit(queue.queue, &[], fence).unwrap();
                    raw.device
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .unwrap();
                    raw.device.destroy_fence(fence, None);
                }
            }
            None => {
                let text = messages.lock().unwrap().join("\n");
                assert!(
                    text.contains("metal2vulkan-extra-queues-absent"),
                    "{:?}: no message: {text}",
                    crate::device::handles::QueueFlagsText(asked)
                );
            }
        }
    }
    assert_eq!(
        extra(&device, vk::QueueFlags::VIDEO_DECODE_KHR).is_some(),
        decode_family.is_some(),
        "a video decode queue exactly where the device has a decode family"
    );
    if raw.queue_families.len() + families.len() > 2 {
        assert!(
            extra(&device, vk::QueueFlags::COMPUTE).is_some(),
            "a device of more than one queue makes the compute one"
        );
    }
    // The video extensions the device enabled, and the features it reports of them.
    let enabled: Vec<&str> = (VIDEO_EXTENSIONS.iter())
        .filter(|name| device.enables_extension(name))
        .map(|name| name.to_str().unwrap())
        .collect();
    let mut maintenance1 = vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if enabled.contains(&"VK_KHR_video_maintenance1") {
        features = features.push_next(&mut maintenance1);
        // SAFETY: the device lists the extension; the chain lives for the call.
        unsafe {
            raw.instance
                .get_physical_device_features2(raw.physical_device, &mut features)
        };
        assert_eq!(maintenance1.video_maintenance1, vk::TRUE);
    }
    eprintln!(
        "VIDEO-INTEROP: {}: video extensions enabled {enabled:?}, videoMaintenance1 {}",
        device.name(),
        maintenance1.video_maintenance1
    );
    if decode_family.is_some() {
        assert!(enabled.contains(&"VK_KHR_video_decode_queue"));
    }
}

/// A queue for the test's raw submissions: an extra queue of the device's where it made one,
/// else a command queue's `VkQueue`.
enum RawQueue {
    Extra(ExtraQueue),
    Lent(CommandQueue),
}

impl RawQueue {
    fn of(device: &Device) -> RawQueue {
        match extra(device, vk::QueueFlags::COMPUTE) {
            Some(queue) => RawQueue::Extra(queue),
            None => RawQueue::Lent(device.new_command_queue().unwrap()),
        }
    }

    /// `f` with the `VkQueue` and its family.
    fn with<R>(&self, f: impl FnOnce(vk::Queue, u32) -> R) -> R {
        match self {
            RawQueue::Extra(queue) => f(queue.queue, queue.family),
            // SAFETY: `f` submits and returns, keeping no handle and calling no runtime.
            RawQueue::Lent(queue) => unsafe { queue.with_vk_queue(f) },
        }
    }

    fn family(&self) -> u32 {
        self.with(|_, family| family)
    }
}

/// One raw command buffer of the test's, recorded by `record` and submitted to `queue` after a
/// wait for `waits` and with `signals` of timeline semaphores, its completion waited for on
/// `fence`; everything it made destroyed after.
fn submit_raw(
    device: &Device,
    queue: &RawQueue,
    waits: &[(vk::Semaphore, u64)],
    signals: &[(vk::Semaphore, u64)],
    record: impl FnOnce(&ash::Device, vk::CommandBuffer),
) {
    // SAFETY: the handles are used while the device lives, every object made here is destroyed
    // here after its work completed, and the command buffer's allocation is made with the
    // loader's word in place (`Device::with_loader_word`).
    unsafe {
        let raw = device.raw();
        let vk_device = raw.device;
        let pool = vk_device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue.family()),
                None,
            )
            .unwrap();
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = device
            .with_loader_word(|| vk_device.allocate_command_buffers(&info))
            .unwrap();
        let command_buffer = buffers[0];
        vk_device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        record(vk_device, command_buffer);
        vk_device.end_command_buffer(command_buffer).unwrap();
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = waits.iter().map(|w| w.1).collect();
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signal_semaphores: Vec<vk::Semaphore> = signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<u64> = signals.iter().map(|s| s.1).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        let fence = vk_device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        queue.with(|vk_queue, _| vk_device.queue_submit(vk_queue, &[submit], fence).unwrap());
        vk_device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        vk_device.destroy_fence(fence, None);
        vk_device.destroy_command_pool(pool, None);
    }
}

/// The queue families the test's own objects are shared among: the device's and the raw
/// queue's.
fn shared_families(device: &Device, raw_queue: &RawQueue) -> Vec<u32> {
    // SAFETY: the handles are read while the device lives.
    let mut families = unsafe { device.raw() }.queue_families.to_vec();
    if !families.contains(&raw_queue.family()) {
        families.push(raw_queue.family());
    }
    families
}

/// A buffer of the test's own, of the usage the device's buffers take, in host-visible coherent
/// memory mapped whole, concurrent among `families` where they are more than one: what a raw
/// submission to an extra queue of a family the device's own objects are not shared with uses.
struct CallerBuffer<'a> {
    device: &'a Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    type_index: u32,
    contents: NonNull<u8>,
    size: u64,
    usage: vk::BufferUsageFlags,
    families: Vec<u32>,
}

impl<'a> CallerBuffer<'a> {
    fn new(device: &'a Device, size: u64, families: Vec<u32>) -> CallerBuffer<'a> {
        let usage = vk::BufferUsageFlags::from(device.vk_buffer_usage());
        let info = CallerBuffer::info_of(size, usage, &families);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // SAFETY: a valid create info; the memory is a host-visible type the buffer takes,
        // allocated with device addresses where its usage names them, bound once and mapped
        // whole once; both are destroyed in `Drop`.
        unsafe {
            let raw = device.raw();
            let buffer = raw.device.create_buffer(&info, None).unwrap();
            let requirements = raw.device.get_buffer_memory_requirements(buffer);
            let properties = raw
                .instance
                .get_physical_device_memory_properties(raw.physical_device);
            let type_index = (0..properties.memory_type_count)
                .find(|&at| {
                    requirements.memory_type_bits & (1 << at) != 0
                        && properties.memory_types[at as usize]
                            .property_flags
                            .contains(host)
                })
                .unwrap();
            let mut address = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(type_index);
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                allocate = allocate.push_next(&mut address);
            }
            let memory = raw.device.allocate_memory(&allocate, None).unwrap();
            raw.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let mapped = raw
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap();
            CallerBuffer {
                device,
                buffer,
                memory,
                type_index,
                contents: NonNull::new(mapped.cast()).unwrap(),
                size,
                usage,
                families,
            }
        }
    }

    fn info_of(
        size: u64,
        usage: vk::BufferUsageFlags,
        families: &[u32],
    ) -> vk::BufferCreateInfo<'_> {
        let info = vk::BufferCreateInfo::default().size(size).usage(usage);
        match families.len() > 1 {
            true => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
            false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }
    }

    /// A `Shared` Metal buffer over it (`Device::new_buffer_with_vk_buffer`).
    ///
    /// # Safety
    ///
    /// The Metal buffer, and every command buffer that uses it, ends before `self`.
    unsafe fn wrapped(&self) -> Buffer {
        let info = CallerBuffer::info_of(self.size, self.usage, &self.families);
        let memory = VkBufferMemory::new(self.memory, 0, self.type_index, Some(self.contents));
        // SAFETY: made with `info`, bound whole at offset 0 of `memory`, of device addresses
        // where its usage names them, mapped whole; the caller's contract keeps it past the
        // Metal buffer, and the test orders its raw submissions by the event's semaphore.
        unsafe {
            self.device.new_buffer_with_vk_buffer(
                self.buffer,
                &info,
                memory,
                self.size,
                ResourceOptions::shared(),
            )
        }
        .unwrap()
    }

    /// Its bytes, set to `bytes` from byte 0.
    fn set(&self, bytes: &[u8]) {
        assert!(bytes.len() as u64 <= self.size);
        // SAFETY: mapped whole, coherent, and used by no command now.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.contents.as_ptr(), bytes.len())
        };
    }
}

impl Drop for CallerBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: every command that used them completed, and their Metal buffers ended.
        unsafe {
            let raw = self.device.raw();
            raw.device.destroy_buffer(self.buffer, None);
            raw.device.free_memory(self.memory, None);
        }
    }
}

/// A raw submission waits for a command buffer's `encodeSignalEvent:` and signals what a later
/// command buffer's `encodeWaitForEvent:` waits for, through the event's semaphore: its counter is
/// a signal index, not Metal's value (a wait for value 10 is counter 1), a wait for a signal not
/// yet submitted is `NotYet`, and the device's watcher sees the raw signal land, which releases
/// the later command buffer and raises the event's value. The raw copy reads what the first
/// command buffer's fill wrote, and the second command buffer's copy what the raw copy wrote; the
/// second command buffer has not completed before the raw signal. The two buffers the raw copy
/// uses are the test's own, shared with the raw queue's family, wrapped as Metal buffers.
#[test]
fn a_raw_submission_is_ordered_by_a_shared_event_s_semaphore() {
    let scratch = Scratch::new("video-interop-event");
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, &messages);
    let raw_queue = RawQueue::of(&device);
    let queue = device.new_command_queue().unwrap();
    const BYTES: u64 = 4096;
    let families = shared_families(&device, &raw_queue);
    let source = CallerBuffer::new(&device, BYTES, families.clone());
    let middle = CallerBuffer::new(&device, BYTES, families);
    source.set(&[0; BYTES as usize]);
    middle.set(&[0; BYTES as usize]);
    // SAFETY: the Metal buffers and the command buffers using them end before the test's own.
    let (filled, copied) = unsafe { (source.wrapped(), middle.wrapped()) };
    let read = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    let hold = device.new_shared_event().unwrap();

    // The first command buffer, held until the host sets `hold`, fills and signals 10.
    let first = queue.command_buffer().unwrap();
    first.encode_wait_for_event((&hold).into(), 1).unwrap();
    let mut blit = first.blit_command_encoder().unwrap();
    blit.fill_buffer(&filled, 0..BYTES, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    first.encode_signal_event((&event).into(), 10).unwrap();
    first.commit().unwrap();
    assert_eq!(event.vk_wait(10), VkEventWait::NotYet, "not submitted yet");
    hold.set_signaled_value(1);
    first.wait_until_scheduled().unwrap();
    let VkEventWait::Counter(wait) = event.vk_wait(10) else {
        panic!("a submitted signal's counter: {:?}", event.vk_wait(10));
    };
    assert_eq!(wait, 1, "the first signal's index, not its value");
    // SAFETY: the signal is submitted below, once, after a wait for `after`.
    let reserved = unsafe { event.reserve_vk_signal(20) }.unwrap();
    let VkEventSignal::Counter { counter, after } = reserved else {
        panic!("a signal above every value is kept: {reserved:?}");
    };
    assert_eq!((counter, after), (2, Some(1)));
    // SAFETY: as above; a value below one reserved is dropped.
    assert_eq!(
        unsafe { event.reserve_vk_signal(15) }.unwrap(),
        VkEventSignal::Dropped
    );
    assert_eq!(event.vk_wait(20), VkEventWait::Counter(2));

    // The second command buffer waits for 20, which only the raw signal reaches.
    let second = queue.command_buffer().unwrap();
    second.encode_wait_for_event((&event).into(), 20).unwrap();
    let mut blit = second.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&copied, 0, &read, 0, BYTES)
        .unwrap();
    blit.end_encoding().unwrap();
    second.commit().unwrap();
    assert!(
        second
            .wait_until_completed_for(Duration::from_millis(300))
            .is_err(),
        "the second command buffer ran before the raw signal"
    );

    // SAFETY: the semaphore is used while the event lives, signalled only as reserved.
    let semaphore = unsafe { event.vk_semaphore() };
    let (from, to) = (source.buffer, middle.buffer);
    submit_raw(
        &device,
        &raw_queue,
        &[(semaphore, wait.max(after.unwrap_or(0)))],
        &[(semaphore, counter)],
        |vk_device, command_buffer| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: BYTES,
            };
            // SAFETY: both buffers live and take transfers; the command buffer is recording.
            unsafe { vk_device.cmd_copy_buffer(command_buffer, from, to, &[region]) };
        },
    );
    assert!(
        second.wait_until_completed_for(WAIT).is_ok(),
        "the command buffer waiting for the raw signal never completed"
    );
    assert_eq!(second.status(), CommandBufferStatus::Completed);
    let contents = read.contents().unwrap().as_ptr();
    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents, BYTES as usize) };
    assert!(
        bytes.iter().all(|&b| b == 0x5a),
        "the fill reached the read"
    );
    assert!(event.wait_until_signaled_value(20, WAIT));
    assert_eq!(event.signaled_value(), 20);
    eprintln!(
        "VIDEO-INTEROP: {}: value 10 was counter {wait}, value 20 counter {counter}",
        device.name()
    );
    drop((first, second, filled, copied));
}

/// The library of the fixture at `path` under `validation/fixtures/public/`.
fn library(device: &Device, path: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(path);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

fn invert(device: &Device) -> ComputePipelineState {
    let function = library(device, "examples/runtime_examples.ll")
        .new_function_with_name("invert_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// Luma texel `(x, y)`'s code.
fn luma(x: u32, y: u32) -> u8 {
    (x * 3 + y * 7 + 16) as u8
}

/// Chroma texel `(x, y)`'s codes, Cb then Cr.
fn chroma(x: u32, y: u32) -> [u8; 2] {
    [(x * 5 + y * 11 + 64) as u8, (255 - x * 2 - y * 3) as u8]
}

/// The caller's NV12 image, its memory, and its creation's queue families.
struct Nv12 {
    image: vk::Image,
    memory: vk::DeviceMemory,
    families: Vec<u32>,
    usage: vk::ImageUsageFlags,
}

impl Nv12 {
    /// The image, where the device makes an optimal NV12 image of sampled and transfer usage,
    /// mutable and of extended usage, shared among `families`; else `None`, printed.
    fn new(device: &Device, families: Vec<u32>) -> Option<Nv12> {
        // SAFETY: the handles are used while the device lives.
        let raw = unsafe { device.raw() };
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let flags = vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
        let format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
        // SAFETY: the physical device is the instance's.
        let supported = unsafe {
            raw.instance.get_physical_device_image_format_properties(
                raw.physical_device,
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                usage,
                flags,
            )
        };
        if let Err(error) = supported {
            eprintln!(
                "VIDEO-INTEROP: {}: no NV12 image of the test's usage ({error:?}): skipped",
                device.name()
            );
            return None;
        }
        let concurrent = families.len() > 1;
        let info = Nv12::info(&families, usage, concurrent);
        // SAFETY: a valid create info of a supported image; the memory is a type the image
        // takes, bound once, and both are destroyed in `Drop`.
        unsafe {
            let image = raw.device.create_image(&info, None).unwrap();
            let requirements = raw.device.get_image_memory_requirements(image);
            let properties = raw
                .instance
                .get_physical_device_memory_properties(raw.physical_device);
            let type_index = (0..properties.memory_type_count)
                .find(|&at| {
                    requirements.memory_type_bits & (1 << at) != 0
                        && properties.memory_types[at as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .unwrap();
            let memory = raw
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(type_index),
                    None,
                )
                .unwrap();
            raw.device.bind_image_memory(image, memory, 0).unwrap();
            Some(Nv12 {
                image,
                memory,
                families,
                usage,
            })
        }
    }

    /// The create info of the image, shared among `families` where `concurrent`.
    fn info(
        families: &[u32],
        usage: vk::ImageUsageFlags,
        concurrent: bool,
    ) -> vk::ImageCreateInfo<'_> {
        let info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .extent(vk::Extent3D {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        match concurrent {
            true => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
            false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }
    }

    /// The texture over `plane` of the image, of `format` at the plane's extent.
    fn texture(&self, device: &Device, plane: u32, format: PixelFormat) -> Texture {
        let shift = plane.min(1);
        let mut descriptor = TextureDescriptor::texture_2d(
            format,
            u64::from(WIDTH >> shift),
            u64::from(HEIGHT >> shift),
            false,
        );
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let info = Nv12::info(&self.families, self.usage, self.families.len() > 1);
        // SAFETY: the image was made with `info`, is bound, rests in `GENERAL` (moved there by
        // the fill before any command buffer uses it), and outlives the textures, which the test
        // drops before it.
        unsafe { device.new_texture_with_vk_image_plane(&descriptor, self.image, &info, plane) }
            .unwrap()
    }
}

/// Drops the image and its memory once nothing uses them.
struct Owned<'a>(&'a Device, Nv12);

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        // SAFETY: every command that used them completed, and their textures were dropped.
        unsafe {
            let raw = self.0.raw();
            raw.device.destroy_image(self.1.image, None);
            raw.device.free_memory(self.1.memory, None);
        }
    }
}

/// A texture over each plane of the caller's NV12 image reads the plane's bytes in place: the
/// test fills each plane with its own codes from a buffer on a raw queue (plane aspects), leaves
/// the image in `GENERAL`, signals a shared event's reserved counter, and a command buffer that
/// waits for it runs `invert_texels` over each plane texture into a texture of its own; every
/// texel read back is the plane's code inverted.
#[test]
fn a_texture_over_each_plane_of_an_nv12_image_reads_its_bytes() {
    let scratch = Scratch::new("video-interop-planes");
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, &messages);
    let raw_queue = RawQueue::of(&device);
    let Some(nv12) = Nv12::new(&device, shared_families(&device, &raw_queue)) else {
        return;
    };
    let nv12 = Owned(&device, nv12);
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let mut bytes = vec![0u8; w * h + w * h / 2];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            bytes[(y * WIDTH + x) as usize] = luma(x, y);
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let at = w * h + (y * WIDTH + 2 * x) as usize;
            bytes[at..at + 2].copy_from_slice(&chroma(x, y));
        }
    }
    let staging = CallerBuffer::new(&device, bytes.len() as u64, vec![raw_queue.family()]);
    staging.set(&bytes);
    let event = device.new_shared_event().unwrap();
    // SAFETY: the signal is submitted below, once.
    let VkEventSignal::Counter { counter, after } =
        (unsafe { event.reserve_vk_signal(1) }).unwrap()
    else {
        panic!("the first signal is kept");
    };
    assert_eq!((counter, after), (1, None));
    // SAFETY: the semaphore is used while the event lives, signalled only as reserved.
    let semaphore = unsafe { event.vk_semaphore() };
    let staging_raw = staging.buffer;
    let image = nv12.1.image;
    submit_raw(
        &device,
        &raw_queue,
        &[],
        &[(semaphore, counter)],
        |vk_device, cb| {
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
            };
            let to_general = barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            let plane = |aspect, offset: usize, extent: (u32, u32)| vk::BufferImageCopy {
                buffer_offset: offset as u64,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: extent.0,
                    height: extent.1,
                    depth: 1,
                },
            };
            let regions = [
                plane(vk::ImageAspectFlags::PLANE_0, 0, (WIDTH, HEIGHT)),
                plane(
                    vk::ImageAspectFlags::PLANE_1,
                    w * h,
                    (WIDTH / 2, HEIGHT / 2),
                ),
            ];
            let written = barrier(
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ,
            );
            // SAFETY: the image and the buffer live; the command buffer is recording.
            unsafe {
                vk_device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                vk_device.cmd_copy_buffer_to_image(
                    cb,
                    staging_raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
                vk_device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[written],
                );
            }
        },
    );

    let luma_texture = nv12.1.texture(&device, 0, PixelFormat::R8Unorm);
    let chroma_texture = nv12.1.texture(&device, 1, PixelFormat::RG8Unorm);
    // A copy of a plane is the runtime's gap, refused typed.
    let queue = device.new_command_queue().unwrap();
    let refused = queue.command_buffer().unwrap();
    let mut blit = refused.blit_command_encoder().unwrap();
    let scratch_texture = {
        let d = TextureDescriptor::texture_2d(PixelFormat::R8Unorm, 8, 8, false);
        device.new_texture(&d).unwrap()
    };
    let copy = blit.copy_from_texture_to_texture(
        &luma_texture,
        0,
        0,
        crate::Origin::default(),
        Size::new(8, 8, 1),
        &scratch_texture,
        0,
        0,
        crate::Origin::default(),
    );
    assert_eq!(
        copy,
        Err(crate::Error::NotImplemented(
            crate::RuntimeGap::VkImagePlaneCopy
        ))
    );
    drop(blit);
    drop(refused);

    let invert = invert(&device);
    let output = |w: u32, h: u32| {
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w.into(), h.into(), false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        d.resource_options = ResourceOptions::shared();
        device.new_texture(&d).unwrap()
    };
    let (luma_out, chroma_out) = (output(WIDTH, HEIGHT), output(WIDTH / 2, HEIGHT / 2));
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event((&event).into(), 1)
        .unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&invert).unwrap();
    for (source, target, (w, h)) in [
        (&luma_texture, &luma_out, (WIDTH, HEIGHT)),
        (&chroma_texture, &chroma_out, (WIDTH / 2, HEIGHT / 2)),
    ] {
        encoder.set_texture(Some(source), 0).unwrap();
        encoder.set_texture(Some(target), 1).unwrap();
        encoder
            .dispatch_threads(Size::new(w.into(), h.into(), 1), Size::new(8, 8, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "the command buffer never completed"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let read = |texture: &Texture, w: u32, h: u32| {
        let mut texels = vec![0u8; (w * h * 4) as usize];
        texture
            .get_bytes(
                &mut texels,
                u64::from(w * 4),
                0,
                Region::new_2d(0, 0, w.into(), h.into()),
                0,
                0,
            )
            .unwrap();
        texels
    };
    let luma_texels = read(&luma_out, WIDTH, HEIGHT);
    let chroma_texels = read(&chroma_out, WIDTH / 2, HEIGHT / 2);
    let mut wrong = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = ((y * WIDTH + x) * 4) as usize;
            let want = [255 - luma(x, y), 255, 255, 255];
            if luma_texels[at..at + 4] != want {
                wrong.push(("luma", x, y, luma_texels[at..at + 4].to_vec()));
            }
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let at = ((y * WIDTH / 2 + x) * 4) as usize;
            let [cb, cr] = chroma(x, y);
            let want = [255 - cb, 255 - cr, 255, 255];
            if chroma_texels[at..at + 4] != want {
                wrong.push(("chroma", x, y, chroma_texels[at..at + 4].to_vec()));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} texels differ, the first {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(4)]
    );
    eprintln!(
        "VIDEO-INTEROP: {}: NV12 {WIDTH}x{HEIGHT}, both planes read in place",
        device.name()
    );
    drop((command_buffer, luma_texture, chroma_texture));
    drop(nv12);
}
