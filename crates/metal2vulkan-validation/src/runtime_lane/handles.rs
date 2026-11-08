//! Device tests of the raw-handle accessors on the device the cfg selects: each handle the runtime
//! lends names the object it says, read back through Vulkan itself.

use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{ResourceOptions, TextureDescriptor};
use std::ffi::CStr;

/// `Device::raw` gives the loader entry, the instance, the physical device, the device and its
/// queue family, with the runtime's function tables: the instance's commands read the device's
/// identity as `properties()` states it. A buffer's `VkBuffer`, a texture's and its
/// view's `VkImage`, and a shared event's timeline semaphore are live objects of that device; a
/// texture buffer has no image. A queue's `VkQueue`, lent under the runtime's lock, takes the
/// test's own submission. Once everything is dropped, the validation layer reported nothing.
#[test]
fn the_raw_handles_name_the_runtimes_objects() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // SAFETY: the handles are used only below, while the device lives; nothing here
        // destroys one, and the test submits only through `with_vk_queue`.
        let raw = unsafe { device.raw() };
        let (instance, vk_device) = (raw.instance, raw.device);
        // The entry is the one the instance was made through.
        // SAFETY: the instance's entry, asked for one of its own commands.
        let enumerate = unsafe {
            (raw.entry.static_fn().get_instance_proc_addr)(
                instance.handle(),
                c"vkEnumeratePhysicalDevices".as_ptr(),
            )
        };
        assert!(enumerate.is_some());
        // SAFETY: a physical device of the instance.
        let props = unsafe { instance.get_physical_device_properties(raw.physical_device) };
        let p = device.properties();
        assert_eq!(
            (props.vendor_id, props.device_id),
            (p.vendor_id, p.device_id)
        );
        // SAFETY: `device_name` is a NUL-terminated array.
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        assert_eq!(name.to_string_lossy(), device.name());
        assert!(raw.instance_api_version.packed() >= vk::API_VERSION_1_2);
        // SAFETY: a physical device of the instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(raw.physical_device) };
        let flags = families[raw.queue_family as usize].queue_flags;
        assert!(flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE));
        eprintln!(
            "runtime handles: {} (vendor {:#06x}, device {:#06x}), instance API {}.{}, \
             queue family {}",
            device.name(),
            props.vendor_id,
            props.device_id,
            raw.instance_api_version.major,
            raw.instance_api_version.minor,
            raw.queue_family
        );

        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        // SAFETY: read only while the buffer lives.
        let (vk_buffer, offset) = unsafe { buffer.vk_buffer() };
        assert_eq!(offset, 0);
        // SAFETY: a live buffer of the device.
        let needs = unsafe { vk_device.get_buffer_memory_requirements(vk_buffer) };
        assert!(needs.size >= 256, "{}", needs.size);

        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        let view = texture
            .new_texture_view(PixelFormat::RGBA8UnormSRGB)
            .unwrap();
        // SAFETY: read only while the texture lives.
        let (image, viewed) = unsafe { (texture.vk_image(), view.vk_image()) };
        let image = image.expect("a texture with an image");
        assert_eq!(viewed, Some(image));
        // SAFETY: a live image of the device.
        let needs = unsafe { vk_device.get_image_memory_requirements(image) };
        assert!(needs.size >= 16 * 16 * 4, "{}", needs.size);
        let texels = device.new_buffer(1024, ResourceOptions::shared()).unwrap();
        let d = TextureDescriptor::texture_buffer(
            PixelFormat::R32Uint,
            64,
            ResourceOptions::shared(),
            TextureUsage::SHADER_READ,
        );
        let texture_buffer = texels.new_texture(&d, 0, 256).unwrap();
        // SAFETY: read only while the texture lives.
        assert_eq!(unsafe { texture_buffer.vk_image() }, None);

        let event = device.new_shared_event().unwrap();
        let other = device.new_shared_event().unwrap();
        // SAFETY: read only while the events live; the test never signals them.
        let (semaphore, other_semaphore) = unsafe { (event.vk_semaphore(), other.vk_semaphore()) };
        assert_ne!(semaphore, other_semaphore);
        // SAFETY: a live timeline semaphore of the device.
        let counter = unsafe { vk_device.get_semaphore_counter_value(semaphore) };
        assert_eq!(counter, Ok(0));

        let queue = device.new_command_queue().unwrap();
        // SAFETY: a fence of the test's own, destroyed below once its submission completed.
        let fence = unsafe { vk_device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .expect("a fence");
        // SAFETY: the queue is used only inside `f`, under the runtime's lock; the submission
        // has no work and signals the test's unsignaled fence.
        let (family, submitted) = unsafe {
            queue.with_vk_queue(|_, vk_queue, family| {
                (
                    family,
                    vk_device.queue_submit(vk_queue, &[vk::SubmitInfo::default()], fence),
                )
            })
        };
        assert_eq!(raw.queue_families.first(), Some(&raw.queue_family));
        assert!(
            raw.queue_families.contains(&family),
            "{family} among {:?}",
            raw.queue_families
        );
        submitted.expect("the test's own submission");
        wait::vulkan::wait_for_fence(
            vk_device,
            fence,
            SubmissionBackend::vulkan(TEST_DEVICE),
            "the raw-handle test's own submission",
        )
        .expect("the submission completes");
        // SAFETY: the test's fence; its submission completed.
        unsafe { vk_device.destroy_fence(fence, None) };
    }
    layer_reported_nothing_at_all(&messages);
}
