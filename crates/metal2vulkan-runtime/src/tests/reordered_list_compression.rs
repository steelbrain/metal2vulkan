//! What a render target's channel-order sibling formats in its image's format list cost the
//! driver's lossless compression: a measurement, not a check. Ignored, printed as `REORDERED-DCC`
//! lines.
//!
//! An `RGBA8Unorm` or `BGRA8Unorm` texture that renders lists its sibling's formats and that
//! sibling's sRGB twin's in its image (`image_plan::reordered_formats`), so a pass renders into such
//! a view of the image itself (`texture/view_shadow.rs`). For each of the two, the runtime's own
//! plan of a 2048x2048 `Private` texture of `ShaderRead` and `RenderTarget` usage, made without
//! `PixelFormatView`, is made three ways on a second `VkDevice` of the runtime's physical device
//! that enables `VK_EXT_image_compression_control`: as planned (the sibling's formats listed),
//! without the sibling's formats (its own and its twin's), and not mutable at all, the control
//! that says whether the driver compresses such an image at all. Each image's
//! `VkImageCompressionPropertiesEXT::imageCompressionFlags` (`vkGetImageSubresourceLayout2`) is
//! printed: `DEFAULT` is the driver's default compression, `DISABLED` none. A device without the
//! extension prints that and nothing else.
//!
//! Run it under `--cfg m2v_gpu` (or `m2v_cpu_icd`):
//! `cargo nextest run -p metal2vulkan-runtime --lib --run-ignored all
//! -E 'test(reordered_list_compression)' --no-capture`.

use super::blit_holds::{device, Scratch};
use crate::image_plan::{reordered_views, view_formats, ImagePlan};
use crate::sharing::Sharing;
use crate::{ResourceOptions, TextureDescriptor};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::{Format, ImageCreateFlags};
use std::collections::BTreeSet;

/// The side of each texture, in texels.
const SIDE: u64 = 2048;

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn reordered_list_compression() {
    let scratch = Scratch::new("reordered-dcc");
    let device = device(&scratch.0);
    let name = device.name();
    // SAFETY: the handles are used while the device lives; the second device is the test's own
    // and destroyed below.
    let handles = unsafe { device.raw() };
    let (instance, physical) = (handles.instance, handles.physical_device);
    let extension = ash::ext::image_compression_control::NAME;
    // SAFETY: the physical device is the instance's.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
    let lists = listed
        .iter()
        .any(|p| p.extension_name_as_c_str() == Ok(extension));
    let mut asked = vk::PhysicalDeviceImageCompressionControlFeaturesEXT::default();
    if lists {
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut asked);
        // SAFETY: the physical device is the instance's, which is of Vulkan 1.1 or later.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
    }
    if asked.image_compression_control == vk::FALSE {
        eprintln!("REORDERED-DCC on {name}: no VK_EXT_image_compression_control");
        return;
    }
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(handles.queue_family)
        .queue_priorities(&priorities)];
    let extensions = [extension.as_ptr()];
    let mut enabled = vk::PhysicalDeviceImageCompressionControlFeaturesEXT::default()
        .image_compression_control(true);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .push_next(&mut enabled);
    // SAFETY: the physical device is the instance's; it lists the extension and its feature.
    let second = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    let functions = ash::ext::image_compression_control::Device::new(instance, &second);
    let profile = device.compile_device().profile();
    let sharing = device.shared().sharing();
    for pf in [PixelFormat::RGBA8Unorm, PixelFormat::BGRA8Unorm] {
        let texture = device
            .new_texture(&TextureDescriptor {
                pixel_format: pf,
                width: SIDE,
                height: SIDE,
                usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                resource_options: ResourceOptions::private(),
                ..TextureDescriptor::default()
            })
            .unwrap();
        let planned = texture.image_plan().unwrap().clone();
        let sibling: BTreeSet<Format> = reordered_views(pf)
            .into_iter()
            .flat_map(|view| view_formats(view, planned.usage, profile))
            .collect();
        let mut apart = planned.clone();
        apart.view_formats.retain(|f| !sibling.contains(f));
        let mut fixed = planned.clone();
        fixed.view_formats.clear();
        fixed.flags = fixed.flags.difference(ImageCreateFlags::MUTABLE_FORMAT);
        for (form, plan) in [
            ("as planned", &planned),
            ("without the sibling's formats", &apart),
            ("not mutable", &fixed),
        ] {
            let compression = compression(&second, &functions, plan, sharing);
            eprintln!(
                "REORDERED-DCC on {name}: {pf:?} {form}: {compression:?} (flags {:?}, formats {:?}, \
                 usage {:?})",
                plan.flags, plan.view_formats, plan.usage
            );
        }
    }
    // SAFETY: every image made on the second device was destroyed.
    unsafe { second.destroy_device(None) };
}

/// The compression the driver of `device` (which enables `VK_EXT_image_compression_control`)
/// gives an image of `plan`, shared as `sharing`, of its first level and layer.
fn compression(
    device: &ash::Device,
    functions: &ash::ext::image_compression_control::Device,
    plan: &ImagePlan,
    sharing: Sharing,
) -> vk::ImageCompressionFlagsEXT {
    let formats = plan.ash_view_formats();
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = plan.create_info(sharing, &formats, &mut list);
    // SAFETY: the plan is the runtime's on the same physical device; the image is destroyed
    // below.
    let image = unsafe { device.create_image(&info, None) }.unwrap();
    let subresource = vk::ImageSubresource2EXT::default().image_subresource(vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    });
    let mut properties = vk::ImageCompressionPropertiesEXT::default();
    {
        let mut layout = vk::SubresourceLayout2EXT::default().push_next(&mut properties);
        // SAFETY: the image is the device's, of one color aspect; the extension is enabled.
        unsafe { functions.get_image_subresource_layout2(image, &subresource, &mut layout) };
    }
    // SAFETY: the image is the device's and nothing uses it.
    unsafe { device.destroy_image(image, None) };
    properties.image_compression_flags
}
