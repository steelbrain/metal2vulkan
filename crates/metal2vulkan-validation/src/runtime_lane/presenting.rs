//! Device tests of step 10 on the device the cfg selects: presenting a texture the runtime rendered
//! through a swapchain the test makes on a device it adopts, the caller's images around the
//! runtime's batches, binary semaphores both ways, shared texture handles, the resource ID a
//! texture over the caller's image takes, and each refusal typed.
//!
//! Two tests make their own instance and device, as a consumer that adopts its device does
//! ([`Adopted`]): the validation layer, a debug-utils messenger that collects what the layer
//! reports (`Messages`' rule), and, for presenting, `VK_EXT_headless_surface` where the loader
//! lists it. A venue without a headless surface, a queue that presents to it, or a swapchain the
//! runtime's texture can be is a typed gap ([`PresentGap`]), printed with the measured reason;
//! the caller's-image test still runs there.

use super::degrade_notices::{assert_made_once, WRAPPED};
use super::shared_kinds;
use super::{
    layer_reported_nothing_at_all, loader, read_buffer, validated_device, validated_device_with,
    Messages, ScratchCache, VALIDATION_LAYER,
};
use crate::candidate::LoaderWord;
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::{DeviceType, TEST_DEVICE};
use ash::vk;
use metal2vulkan::device::{DevicePolicy, EnableSet, VkExtension};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, Device, Error, HazardTrackingMode, IdPlacement,
    Invalid, LoadAction, Nil, Origin, Problem, RawDevice, ResourceId, ResourceIdKind,
    ResourceOptions, SemaphoreProblem, ShaderCacheOptions, Size, StorageMode, StoreAction, Texture,
    TextureDescriptor, VkImageLayouts, WrappedImageProblem,
};
use std::collections::BTreeSet;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The test's images: 16 × 16 texels of four bytes.
const SIDE: u32 = 16;

/// Why a venue cannot present through a headless surface, as measured.
enum PresentGap {
    /// The loader lists no `VK_EXT_headless_surface` (or no `VK_KHR_surface`).
    NoHeadlessSurface,
    /// The device lists no `VK_KHR_swapchain`.
    NoSwapchain,
    /// The device's graphics and compute family cannot present to the surface; `presenting`
    /// names the device's families that can.
    NoPresentQueue { family: u32, presenting: Vec<u32> },
    /// The surface offers neither `B8G8R8A8_UNORM` nor `R8G8B8A8_UNORM`.
    NoFormat { offered: Vec<i32> },
    /// The surface's images lack usage the runtime's render target needs.
    SurfaceUsage { missing: vk::ImageUsageFlags },
}

impl std::fmt::Display for PresentGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PresentGap::NoHeadlessSurface => {
                f.write_str("NoHeadlessSurface: the loader lists no VK_EXT_headless_surface")
            }
            PresentGap::NoSwapchain => {
                f.write_str("NoSwapchain: the device lists no VK_KHR_swapchain")
            }
            PresentGap::NoPresentQueue { family, presenting } => write!(
                f,
                "NoPresentQueue: queue family {family} cannot present to a headless surface; \
                 the families that can: {presenting:?}"
            ),
            PresentGap::NoFormat { offered } => write!(
                f,
                "NoFormat: the surface offers formats {offered:?}, neither B8G8R8A8_UNORM nor \
                 R8G8B8A8_UNORM"
            ),
            PresentGap::SurfaceUsage { missing } => write!(
                f,
                "SurfaceUsage: the surface's images lack usage {:#x}",
                missing.as_raw()
            ),
        }
    }
}

/// What the validation layer reported on the test's own instance: every error, and every warning
/// of the validation type (`Messages`' rule). Bounded.
#[derive(Default)]
struct Reported(Mutex<Vec<String>>);

impl Reported {
    fn take(&self) -> Vec<String> {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.drain(..).collect()
    }
}

/// The test instance's messenger callback: keeps what fails a run.
unsafe extern "system" fn report(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    let fails = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        || (severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
            && kind.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION));
    if fails && !data.is_null() && !user.is_null() {
        // SAFETY: the instance's messenger was made with a `Reported` that outlives it, and the
        // layer passes a valid callback data structure.
        let (reported, text) = unsafe {
            (
                &*user.cast::<Reported>().cast_const(),
                (*data)
                    .message_as_c_str()
                    .map(|text| text.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            )
        };
        let mut reported = reported.0.lock().unwrap_or_else(|p| p.into_inner());
        if reported.len() < 64 {
            reported.push(text);
        }
    }
    vk::FALSE
}

/// A surface the test presents to, and the commands it needs.
struct Presenting {
    surface_fn: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_fn: ash::khr::swapchain::Device,
}

/// A device the test made itself, as a consumer that adopts its device makes it: a Vulkan 1.2
/// instance with the validation layer and a messenger, one queue of the first graphics and
/// compute family, timeline semaphores, and only `VK_KHR_portability_subset` (where listed) of the
/// compile layer's vocabulary, with `VK_KHR_swapchain` when it presents. `gpu_times` adopts one
/// too: it names no calibrated timestamps.
pub(super) struct Adopted {
    entry: ash::Entry,
    instance: ash::Instance,
    debug: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    reported: Box<Reported>,
    physical: vk::PhysicalDevice,
    family: u32,
    name: String,
    memory: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    word: LoaderWord,
    enabled: EnableSet,
    present: Result<Presenting, PresentGap>,
    /// Whether the device enabled its portability subset's `imageViewFormatReinterpretation`,
    /// which the runtime that adopts it is told (`RawDevice::view_format_reinterpretation`).
    view_reinterpretation: bool,
    /// Whether the device enabled `hostImageCopy`, which the runtime that adopts it is told
    /// (`RawDevice::host_image_copy`).
    host_image_copy: bool,
}

impl Adopted {
    /// A new instance and device of the kind the cfg selects; with `present`, a headless surface
    /// and the swapchain commands where the venue has them. A GPU is opened only while the hang
    /// tripwire is absent. Where the device has a portability subset, it is enabled without
    /// `imageViewFormatReinterpretation`.
    pub(super) fn open(present: bool) -> Result<Adopted, String> {
        Adopted::open_with(present, false, false)
    }

    /// As [`Adopted::open`] without presenting, a portability subset enabled with
    /// `imageViewFormatReinterpretation` where the subset has it.
    pub(super) fn open_reinterpreting() -> Result<Adopted, String> {
        Adopted::open_with(false, true, false)
    }

    /// As [`Adopted::open`] without presenting, `hostImageCopy` enabled where the device has it:
    /// with `VK_EXT_host_image_copy`, `VK_KHR_copy_commands2` and `VK_KHR_format_feature_flags2`,
    /// which the Vulkan 1.2 instance needs, where the device lists the three.
    pub(super) fn open_host_copying() -> Result<Adopted, String> {
        Adopted::open_with(false, false, true)
    }

    /// Whether the device enabled `hostImageCopy`.
    pub(super) fn copies_host_images(&self) -> bool {
        self.host_image_copy
    }

    /// Whether the device enabled its portability subset's `imageViewFormatReinterpretation`.
    pub(super) fn reinterprets(&self) -> bool {
        self.view_reinterpretation
    }

    /// Whether the device enabled a portability subset.
    pub(super) fn subset(&self) -> bool {
        self.enabled
            .extensions
            .contains(&VkExtension::KhrPortabilitySubset)
    }

    fn open_with(present: bool, reinterpret: bool, host_copy: bool) -> Result<Adopted, String> {
        super::fatal_signals::install();
        if TEST_DEVICE == DeviceType::Gpu {
            crate::gpu::tripwire::refuse_if_tripped()?;
        }
        // SAFETY: opening the loader library runs only its initializers.
        let entry = unsafe {
            match loader() {
                Some(path) => ash::Entry::load_from(path),
                None => ash::Entry::load(),
            }
        }
        .map_err(|error| format!("no Vulkan loader: {error}"))?;
        // SAFETY: the entry is live.
        let listed: BTreeSet<CString> =
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|error| format!("instance extensions: {error}"))?
                .iter()
                .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
                .collect();
        let has = |name: &CStr| listed.contains(name);
        let portability = has(ash::khr::portability_enumeration::NAME);
        let headless = has(ash::khr::surface::NAME) && has(ash::ext::headless_surface::NAME);
        let mut extensions = vec![ash::ext::debug_utils::NAME.as_ptr()];
        if portability {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }
        if present && headless {
            extensions.push(ash::khr::surface::NAME.as_ptr());
            extensions.push(ash::ext::headless_surface::NAME.as_ptr());
        }
        let layers = [VALIDATION_LAYER.as_ptr()];
        let reported = Box::<Reported>::default();
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(report))
            .user_data(std::ptr::from_ref::<Reported>(&reported).cast_mut().cast());
        let application = vk::ApplicationInfo::default()
            .application_name(c"metal2vulkan runtime lane")
            .api_version(vk::API_VERSION_1_2);
        let mut chained = messenger_info;
        let info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(if portability {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::empty()
            })
            .push_next(&mut chained);
        // SAFETY: `info` and everything it points at live for the call.
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|error| format!("create instance: {error}"))?;
        let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
        // SAFETY: the instance enabled `VK_EXT_debug_utils`; `reported` outlives the messenger.
        let messenger = unsafe { debug.create_debug_utils_messenger(&messenger_info, None) }
            .map_err(|error| format!("create messenger: {error}"))?;

        // SAFETY: the instance is live.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| format!("enumerate physical devices: {error}"))?;
        let rank = |ty: vk::PhysicalDeviceType| match (TEST_DEVICE, ty) {
            (DeviceType::Cpu, vk::PhysicalDeviceType::CPU) => Some(0),
            (DeviceType::Gpu, vk::PhysicalDeviceType::DISCRETE_GPU) => Some(0),
            (DeviceType::Gpu, vk::PhysicalDeviceType::INTEGRATED_GPU) => Some(1),
            (DeviceType::Gpu, vk::PhysicalDeviceType::VIRTUAL_GPU) => Some(2),
            _ => None,
        };
        let (physical, props) = physical_devices
            .iter()
            .filter_map(|&physical| {
                // SAFETY: enumerated from the instance.
                let props = unsafe { instance.get_physical_device_properties(physical) };
                rank(props.device_type).map(|rank| (rank, physical, props))
            })
            .min_by_key(|&(rank, ..)| rank)
            .map(|(_, physical, props)| (physical, props))
            .ok_or_else(|| format!("no {} device", TEST_DEVICE.as_str()))?;
        // SAFETY: `device_name` is a NUL-terminated array.
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: enumerated from the instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let family = families
            .iter()
            .position(|f| f.queue_count > 0 && f.queue_flags.contains(wanted))
            .ok_or("no graphics and compute queue family")? as u32;
        // SAFETY: enumerated from the instance.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        // SAFETY: enumerated from the instance.
        let device_listed: BTreeSet<CString> =
            unsafe { instance.enumerate_device_extension_properties(physical) }
                .map_err(|error| format!("device extensions: {error}"))?
                .iter()
                .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
                .collect();
        let device_has = |name: &CStr| device_listed.contains(name);

        let surface = if !present || !headless {
            Err(PresentGap::NoHeadlessSurface)
        } else if !device_has(ash::khr::swapchain::NAME) {
            Err(PresentGap::NoSwapchain)
        } else {
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let headless_fn = ash::ext::headless_surface::Instance::new(&entry, &instance);
            // SAFETY: the instance enabled both extensions.
            let surface = unsafe {
                headless_fn
                    .create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None)
            }
            .map_err(|error| format!("create headless surface: {error}"))?;
            let mut presenting = Vec::new();
            for index in 0..families.len() as u32 {
                // SAFETY: a surface of the instance and a family of the device.
                let supported = unsafe {
                    surface_fn.get_physical_device_surface_support(physical, index, surface)
                }
                .map_err(|error| format!("surface support: {error}"))?;
                if supported {
                    presenting.push(index);
                }
            }
            if presenting.contains(&family) {
                Ok((surface_fn, surface))
            } else {
                // SAFETY: nothing was made from the surface.
                unsafe { surface_fn.destroy_surface(surface, None) };
                Err(PresentGap::NoPresentQueue { family, presenting })
            }
        };

        let mut enabled = EnableSet::new();
        let mut device_extensions = Vec::new();
        let subset = device_has(ash::khr::portability_subset::NAME);
        if subset {
            enabled.extensions.insert(VkExtension::KhrPortabilitySubset);
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        if surface.is_ok() {
            device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        // The subset's `imageViewFormatReinterpretation`, where asked for and the subset has it.
        let view_reinterpretation = reinterpret && subset && {
            let mut offered = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut offered);
            // SAFETY: enumerated from the instance, a Vulkan 1.2 one; the device lists the
            // subset, whose features structure is chained.
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
            offered.image_view_format_reinterpretation == vk::TRUE
        };
        // `hostImageCopy`, where asked for and the device lists the extensions a Vulkan 1.2
        // instance needs for it and has the feature.
        let host_names = [
            ash::ext::host_image_copy::NAME,
            ash::khr::copy_commands2::NAME,
            ash::khr::format_feature_flags2::NAME,
        ];
        let host_image_copy = host_copy && host_names.iter().all(|&name| device_has(name)) && {
            let mut offered = vk::PhysicalDeviceHostImageCopyFeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut offered);
            // SAFETY: enumerated from the instance, a Vulkan 1.2 one; the device lists
            // `VK_EXT_host_image_copy`, whose features structure is chained.
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
            offered.host_image_copy == vk::TRUE
        };
        if host_image_copy {
            device_extensions.extend(host_names.map(CStr::as_ptr));
        }
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut timeline = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut portability = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default()
            .image_view_format_reinterpretation(true);
        let mut host = vk::PhysicalDeviceHostImageCopyFeaturesEXT::default().host_image_copy(true);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut timeline);
        if view_reinterpretation {
            info = info.push_next(&mut portability);
        }
        if host_image_copy {
            info = info.push_next(&mut host);
        }
        // SAFETY: `info` and everything it points at live for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }
            .map_err(|error| format!("create device: {error}"))?;
        // SAFETY: the loader just returned the handle; `close` destroys it last.
        let word = unsafe { LoaderWord::of_device(device.handle()) };
        let present = surface.map(|(surface_fn, surface)| {
            #[expect(
                clippy::disallowed_methods,
                reason = "the loader word was recorded from the handle just returned"
            )]
            let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
            Presenting {
                surface_fn,
                surface,
                swapchain_fn,
            }
        });
        Ok(Adopted {
            entry,
            instance,
            debug,
            messenger,
            reported,
            physical,
            family,
            name,
            memory,
            device,
            word,
            enabled,
            present,
            view_reinterpretation,
            host_image_copy,
        })
    }

    /// The runtime device that adopts this device, its shader cache in `cache`.
    pub(super) fn runtime(&self, cache: &Path) -> Device {
        let mut raw = RawDevice::new(
            self.entry.static_fn().get_instance_proc_addr,
            self.instance.handle(),
            metal2vulkan::device::ApiVersion::V1_2,
            self.physical,
            self.device.handle(),
            self.family,
            self.enabled.clone(),
        );
        raw.shader_cache = ShaderCacheOptions::in_directory(cache);
        raw.view_format_reinterpretation = self.view_reinterpretation;
        raw.host_image_copy = self.host_image_copy;
        // SAFETY: the instance and device were made as `raw` states (the subset's
        // `imageViewFormatReinterpretation` and `hostImageCopy`, with the extensions a Vulkan 1.2
        // instance needs for it, enabled exactly where it says), with timeline semaphores; they
        // outlive the runtime device (dropped before `close`); the test uses the queue only
        // through `with_vk_queue`.
        unsafe { Device::from_raw(raw, DevicePolicy::default()) }.expect("the runtime adopts it")
    }

    /// A memory type of `bits` with `flags`.
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        memory_type(&self.memory, bits, flags)
    }

    /// Destroy the surface, the device, the messenger and the instance, in that order, and give
    /// what the validation layer reported.
    pub(super) fn close(self) -> Vec<String> {
        // SAFETY: nothing made from them remains; the device's word is put back before the
        // loader's `vkDestroyDevice` reads it.
        unsafe {
            if let Ok(present) = &self.present {
                present.surface_fn.destroy_surface(present.surface, None);
            }
            self.word.restore();
            #[expect(
                clippy::disallowed_methods,
                reason = "the device the test made, after the loader word is put back"
            )]
            self.device.destroy_device(None);
            self.debug
                .destroy_debug_utils_messenger(self.messenger, None);
            self.instance.destroy_instance(None);
        }
        self.reported.take()
    }
}

/// `descriptor`'s 2D texture of `format`, `RenderTarget` usage and `Private` storage, `SIDE`
/// texels square.
fn target(format: PixelFormat) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(format, u64::from(SIDE), u64::from(SIDE), false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// What `vkCreateImage` takes for a `SIDE`-square 2D image of `format` and `usage`.
pub(super) fn image_info(
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
}

/// `info` shared as a runtime device's own images are among its queue families `families`
/// (`DeviceHandles::queue_families`): concurrent among them where there are several, as a
/// caller's image the runtime wraps must be there; as it is on a device of one family.
pub(super) fn shared_among<'a>(
    info: vk::ImageCreateInfo<'a>,
    families: &'a [u32],
) -> vk::ImageCreateInfo<'a> {
    match families.len() > 1 {
        true => info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(families),
        false => info,
    }
}

/// Encode a render pass clearing `texture` to `color` into `command_buffer`.
fn clear(command_buffer: &CommandBuffer, texture: &Texture, [r, g, b, a]: [f64; 4]) {
    let mut pass = metal2vulkan_runtime::RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(r, g, b, a);
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
}

/// Encode a blit copying `texture`'s texels into `buffer` (four bytes a texel).
fn copy_out(command_buffer: &CommandBuffer, texture: &Texture, buffer: &Buffer) {
    let side = u64::from(SIDE);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        buffer,
        0,
        side * 4,
        side * side * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

/// `SIDE × SIDE` texels of `texel`.
fn filled(texel: [u8; 4]) -> Vec<u8> {
    texel.repeat((SIDE * SIDE) as usize)
}

/// A new binary semaphore of `device`.
fn binary_semaphore(device: &ash::Device) -> vk::Semaphore {
    // SAFETY: the device is live.
    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap()
}

/// The runtime presents what it rendered through the test's swapchain on a device the test
/// made and the runtime adopted: each swapchain image is a texture over the caller's image
/// (`UNDEFINED` at first, `PRESENT_SRC_KHR` between the runtime's batches); each frame acquires
/// an image, and one command buffer waits for the acquire, clears the image to the frame's color,
/// copies it into a buffer and signals the image's semaphore, which the present waits for once
/// the command buffer is scheduled. Twice as many frames as images are presented, each into the
/// image the swapchain hands out, its bytes the frame's color; how often each image came back is
/// printed.
/// Dropping the textures leaves the images to the swapchain, whose destruction the validation
/// layer accepts: the runtime destroyed none of them. Where the venue cannot present headless,
/// the gap is printed with its measured reason.
#[test]
fn a_rendered_texture_presents_through_the_tests_swapchain() {
    let cache = ScratchCache::new().unwrap();
    let adopted = Adopted::open(true).unwrap();
    let present = match &adopted.present {
        Ok(present) => present,
        Err(gap) => {
            eprintln!("runtime present: gap {gap} on {}", adopted.name);
            let reported = adopted.close();
            assert!(
                reported.is_empty(),
                "the validation layer reported: {reported:#?}"
            );
            return;
        }
    };
    // SAFETY: a surface of the instance.
    let (caps, formats) = unsafe {
        (
            present
                .surface_fn
                .get_physical_device_surface_capabilities(adopted.physical, present.surface)
                .unwrap(),
            present
                .surface_fn
                .get_physical_device_surface_formats(adopted.physical, present.surface)
                .unwrap(),
        )
    };
    let chosen = [
        (vk::Format::B8G8R8A8_UNORM, PixelFormat::BGRA8Unorm),
        (vk::Format::R8G8B8A8_UNORM, PixelFormat::RGBA8Unorm),
    ]
    .into_iter()
    .find_map(|(format, pf)| {
        formats
            .iter()
            .find(|f| f.format == format)
            .map(|f| (*f, pf))
    });
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let missing = usage & !caps.supported_usage_flags;
    let gap = match chosen {
        None => Some(PresentGap::NoFormat {
            offered: formats.iter().map(|f| f.format.as_raw()).collect(),
        }),
        Some(_) if !missing.is_empty() => Some(PresentGap::SurfaceUsage { missing }),
        Some(_) => None,
    };
    if let Some(gap) = gap {
        eprintln!("runtime present: gap {gap} on {}", adopted.name);
        let reported = adopted.close();
        assert!(
            reported.is_empty(),
            "the validation layer reported: {reported:#?}"
        );
        return;
    }
    let (surface_format, pixel_format) = chosen.expect("a format");
    let extent = vk::Extent2D {
        width: SIDE,
        height: SIDE,
    };
    let fits = caps.current_extent.width == u32::MAX
        || (caps.current_extent.width == SIDE && caps.current_extent.height == SIDE);
    assert!(fits, "a headless surface takes the swapchain's extent");
    let count = caps.min_image_count.max(2);
    let count = match caps.max_image_count {
        0 => count,
        max => count.min(max),
    };
    let alpha = [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ]
    .into_iter()
    .find(|&alpha| caps.supported_composite_alpha.contains(alpha))
    .expect("a composite alpha");
    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(present.surface)
        .min_image_count(count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(alpha)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true);
    let swapchain_fn = &present.swapchain_fn;
    // SAFETY: the surface is the instance's, and the device presents to it from `family`.
    let swapchain = unsafe { swapchain_fn.create_swapchain(&info, None) }.unwrap();
    // SAFETY: the swapchain is live.
    let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }.unwrap();
    let device = adopted.runtime(cache.path());
    let queue = device.new_command_queue().unwrap();
    // The images, as `vkCreateSwapchainKHR` made them.
    let made = image_info(surface_format.format, usage);
    let textures: Vec<Texture> = images
        .iter()
        .map(|&image| {
            // SAFETY: a live image of the device, made as `made` states, in `UNDEFINED` now; the
            // swapchain outlives the texture; the test presents it only after the runtime's
            // batch that uses it is scheduled, waiting for that batch's signal.
            unsafe {
                device.new_texture_with_vk_image(
                    &target(pixel_format),
                    image,
                    &made,
                    VkImageLayouts::new(
                        vk::ImageLayout::UNDEFINED.into(),
                        vk::ImageLayout::PRESENT_SRC_KHR.into(),
                    ),
                )
            }
            .unwrap()
        })
        .collect();
    let acquired = binary_semaphore(&adopted.device);
    let rendered: Vec<vk::Semaphore> = images
        .iter()
        .map(|_| binary_semaphore(&adopted.device))
        .collect();
    let readback = device
        .new_buffer(u64::from(SIDE * SIDE * 4), ResourceOptions::shared())
        .unwrap();
    let colors: [([f64; 4], [u8; 4]); 3] = [
        ([1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        ([0.0, 1.0, 0.0, 1.0], [0, 255, 0, 255]),
        ([0.0, 0.0, 1.0, 1.0], [0, 0, 255, 255]),
    ];
    let frames = 2 * images.len();
    let mut presented = vec![0; images.len()];
    for frame in 0..frames {
        // SAFETY: the swapchain is live; `acquired` is unsignaled (the last command buffer that
        // waited for it completed).
        let (index, _) = unsafe {
            swapchain_fn.acquire_next_image(
                swapchain,
                WAIT.as_nanos() as u64,
                acquired,
                vk::Fence::null(),
            )
        }
        .unwrap();
        let index = index as usize;
        let (color, rgba) = colors[frame % colors.len()];
        let command_buffer = queue.command_buffer().unwrap();
        // SAFETY: binary semaphores of the device: the acquire's signal was submitted before
        // the commit; the image's semaphore is unsignaled, the present that waited for it last
        // done (the image was acquired again).
        unsafe { command_buffer.encode_wait_for_vk_semaphore(acquired) }.unwrap();
        clear(&command_buffer, &textures[index], color);
        copy_out(&command_buffer, &textures[index], &readback);
        // SAFETY: as above.
        unsafe { command_buffer.encode_signal_vk_semaphore(rendered[index]) }.unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        let wait = [rendered[index]];
        let swapchains = [swapchain];
        let indices = [index as u32];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the queue is used only inside the closure, under the runtime's lock; the
        // semaphore's signal is submitted (the command buffer is scheduled).
        let shown = unsafe {
            queue.with_vk_queue(|_, vk_queue, _| swapchain_fn.queue_present(vk_queue, &info))
        };
        shown.unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let texel = match pixel_format {
            PixelFormat::BGRA8Unorm => [rgba[2], rgba[1], rgba[0], rgba[3]],
            _ => rgba,
        };
        let bytes = read_buffer(&readback, 0, readback.length()).unwrap();
        assert_eq!(bytes, filled(texel), "frame {frame}, image {index}");
        presented[index] += 1;
    }
    // SAFETY: under the runtime's lock; every submission there completed but the presents,
    // which a fence submitted after them orders.
    unsafe {
        queue.with_vk_queue(|_, vk_queue, _| {
            wait::vulkan::drain_queue(
                &adopted.device,
                vk_queue,
                SubmissionBackend::vulkan(TEST_DEVICE),
            )
        })
    }
    .unwrap();
    eprintln!(
        "runtime present: {frames} frames presented through {} {:?} images of a headless \
         swapchain on {}, each image {:?} times",
        images.len(),
        metal2vulkan::vk::Format::from(surface_format.format),
        adopted.name,
        presented
    );
    drop(textures);
    drop(readback);
    drop(queue);
    drop(device);
    // SAFETY: nothing uses them: every command buffer and present completed.
    unsafe {
        swapchain_fn.destroy_swapchain(swapchain, None);
        adopted.device.destroy_semaphore(acquired, None);
        for semaphore in rendered {
            adopted.device.destroy_semaphore(semaphore, None);
        }
    }
    let reported = adopted.close();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// An image of `device` for `info`, bound to new memory of a type `memory` gives with `flags`.
///
/// # Safety
///
/// `device` is live; the caller destroys the image and frees the memory after their last use.
pub(super) unsafe fn made_image(
    device: &ash::Device,
    memory: &vk::PhysicalDeviceMemoryProperties,
    info: &vk::ImageCreateInfo<'_>,
) -> (vk::Image, vk::DeviceMemory) {
    // SAFETY: the caller's contract.
    unsafe {
        let image = device.create_image(info, None).unwrap();
        let needs = device.get_image_memory_requirements(image);
        let index = memory_type(
            memory,
            needs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(index);
        let bound = device.allocate_memory(&allocation, None).unwrap();
        device.bind_image_memory(image, bound, 0).unwrap();
        (image, bound)
    }
}

/// A memory type of `memory` among `bits` with `flags`.
fn memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    (0..memory.memory_type_count)
        .find(|&i| {
            bits & (1 << i) != 0
                && memory.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
        .expect("a memory type")
}

/// Commit `command_buffer` and wait for it to complete without an error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A texture over an image the test made and keeps (`UNDEFINED` at first, `TRANSFER_SRC_OPTIMAL`
/// between the runtime's batches), on a device the test made and the runtime adopted. The runtime
/// clears it and signals a binary semaphore; the test's own submission waits for it, copies the
/// image from `TRANSFER_SRC_OPTIMAL` into a buffer of its own and signals a second binary
/// semaphore. A runtime command buffer with no encoders, on another logical queue, waits for it
/// and signals a third, for which the next one waits before it copies the image out; another
/// clears it again and copies it out. The bytes are each clear's. Dropping the texture leaves the image to the test, which destroys it and its memory,
/// and the validation layer reported nothing.
#[test]
fn the_callers_image_comes_home_for_its_own_submission() {
    let cache = ScratchCache::new().unwrap();
    let adopted = Adopted::open(false).unwrap();
    let vkd = &adopted.device;
    let backend = SubmissionBackend::vulkan(TEST_DEVICE);
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let info = image_info(vk::Format::R8G8B8A8_UNORM, usage);
    let size = u64::from(SIDE * SIDE * 4);
    // SAFETY: the device is live; the test destroys what it makes after its last use.
    let (image, image_memory) = unsafe { made_image(vkd, &adopted.memory, &info) };
    // SAFETY: as above.
    let (host, host_memory, mapped) = unsafe {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = vkd.create_buffer(&info, None).unwrap();
        let needs = vkd.get_buffer_memory_requirements(buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(adopted.memory_type(needs.memory_type_bits, flags));
        let memory = vkd.allocate_memory(&allocation, None).unwrap();
        vkd.bind_buffer_memory(buffer, memory, 0).unwrap();
        let mapped = vkd
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .unwrap();
        (buffer, memory, mapped)
    };
    let device = adopted.runtime(cache.path());
    // SAFETY: a live image of the device, made with `info`, bound, in `UNDEFINED` now; it
    // outlives the texture; the test uses it between the runtime's batches only in
    // `TRANSFER_SRC_OPTIMAL`, ordered by the binary semaphores.
    let texture = unsafe {
        device.new_texture_with_vk_image(
            &target(PixelFormat::RGBA8Unorm),
            image,
            &info,
            VkImageLayouts::new(
                vk::ImageLayout::UNDEFINED.into(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL.into(),
            ),
        )
    }
    .unwrap();
    // SAFETY: read only while the texture lives.
    assert_eq!(unsafe { texture.vk_image() }, Some(image));
    let rendered = binary_semaphore(vkd);
    let copied = binary_semaphore(vkd);
    let relayed = binary_semaphore(vkd);
    let first = device.new_command_queue().unwrap();
    let second = device.new_command_queue().unwrap();
    let readback = device.new_buffer(size, ResourceOptions::shared()).unwrap();

    let clearing = first.command_buffer().unwrap();
    clear(&clearing, &texture, [1.0, 0.0, 0.0, 1.0]);
    // SAFETY: a binary semaphore of the device, unsignaled, which the test's submission waits
    // for once the command buffer is scheduled.
    unsafe { clearing.encode_signal_vk_semaphore(rendered) }.unwrap();
    clearing.commit().unwrap();
    clearing.wait_until_scheduled().unwrap();

    // SAFETY: the device is live; the pool and its command buffer are the test's own.
    let pool = unsafe {
        vkd.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(adopted.family),
            None,
        )
    }
    .unwrap();
    let allocation = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the device is live; the loader serves the call after the word is put back.
    let own = unsafe {
        adopted.word.restore();
        #[expect(
            clippy::disallowed_methods,
            reason = "the device the test made, after the loader word is put back"
        )]
        let own = vkd.allocate_command_buffers(&allocation);
        own
    }
    .unwrap()[0];
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        });
    let to_host = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    // SAFETY: the test's command buffer, recorded once; the image is in `TRANSFER_SRC_OPTIMAL`
    // when the copy runs (after the runtime's batch, which the submission waits for).
    unsafe {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        vkd.begin_command_buffer(own, &begin).unwrap();
        vkd.cmd_copy_image_to_buffer(
            own,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            host,
            &[region],
        );
        vkd.cmd_pipeline_barrier(
            own,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[to_host],
            &[],
            &[],
        );
        vkd.end_command_buffer(own).unwrap();
    }
    // SAFETY: the device is live.
    let fence = unsafe { vkd.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    let waits = [rendered];
    let stages = [vk::PipelineStageFlags::TRANSFER];
    let buffers = [own];
    let signals = [copied];
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .command_buffers(&buffers)
        .signal_semaphores(&signals);
    // SAFETY: under the runtime's lock of the queue; `rendered`'s signal is submitted (the
    // runtime's command buffer is scheduled); `copied` is unsignaled; the fence is the test's.
    unsafe { first.with_vk_queue(|_, queue, _| vkd.queue_submit(queue, &[submit], fence)) }
        .unwrap();
    wait::vulkan::wait_for_fence(vkd, fence, backend, "the test's own copy").unwrap();
    // SAFETY: the mapping covers the buffer's bytes, which the completed copy wrote.
    let own_bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize) };
    assert_eq!(own_bytes, filled([255, 0, 0, 255]));
    clearing.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(clearing.error(), None);

    // A command buffer with no encoders passes the caller's signal on (one submission of its
    // wait and its signal), and the next one, on the other queue and committed after it, waits
    // for the relayed signal: the scheduler holds it until the relay is submitted.
    let relay = second.command_buffer().unwrap();
    // SAFETY: binary semaphores of the device: `copied`'s signal the test submitted before the
    // commit, `relayed` unsignaled; one wait consumes each.
    unsafe {
        relay.encode_wait_for_vk_semaphore(copied).unwrap();
        relay.encode_signal_vk_semaphore(relayed).unwrap();
    }
    let reading = first.command_buffer().unwrap();
    // SAFETY: as above: the relay, committed first, signals it.
    unsafe { reading.encode_wait_for_vk_semaphore(relayed) }.unwrap();
    copy_out(&reading, &texture, &readback);
    relay.commit().unwrap();
    reading.commit().unwrap();
    for command_buffer in [&relay, &reading] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    let bytes = read_buffer(&readback, 0, size).unwrap();
    assert_eq!(bytes, filled([255, 0, 0, 255]));

    let again = first.command_buffer().unwrap();
    clear(&again, &texture, [0.0, 1.0, 0.0, 1.0]);
    copy_out(&again, &texture, &readback);
    run(&again);
    let bytes = read_buffer(&readback, 0, size).unwrap();
    assert_eq!(bytes, filled([0, 255, 0, 255]));
    eprintln!(
        "runtime present: the caller's image came home for its own submission on {}",
        adopted.name
    );

    drop(texture);
    drop(readback);
    drop((first, second, clearing, relay, reading, again));
    drop(device);
    // SAFETY: nothing uses them: every submission completed and the runtime device is gone.
    unsafe {
        vkd.destroy_fence(fence, None);
        vkd.destroy_command_pool(pool, None);
        vkd.destroy_semaphore(rendered, None);
        vkd.destroy_semaphore(copied, None);
        vkd.destroy_semaphore(relayed, None);
        vkd.unmap_memory(host_memory);
        vkd.destroy_buffer(host, None);
        vkd.free_memory(host_memory, None);
        vkd.destroy_image(image, None);
        vkd.free_memory(image_memory, None);
    }
    let reported = adopted.close();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// On a device the runtime created, a texture over an image the test made through
/// `Device::raw` (`UNDEFINED` at first, `TRANSFER_DST_OPTIMAL` between batches) is cleared on one
/// queue and copied out on another, three times with the roles swapped, with no event or fence:
/// hazard tracking orders the moves and the uses across the queues, and each copy holds its
/// clear's bytes. The image rests out of `GENERAL`, which the device announces once
/// (`Degradation::WrappedImageLayoutTransitions`). The number of distinct `VkQueue`s the two
/// queues submit to is printed.
#[test]
fn wrapped_images_order_across_queues() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // SAFETY: used only while the device lives; the image is destroyed after its texture
        // and every command buffer that used it.
        let raw = unsafe { device.raw() };
        // SAFETY: a physical device of the instance.
        let memory = unsafe {
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device)
        };
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let info = shared_among(
            image_info(vk::Format::R8G8B8A8_UNORM, usage),
            raw.queue_families,
        );
        // SAFETY: the device is live.
        let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
        // SAFETY: as in `the_callers_image_comes_home_for_its_own_submission`; nothing else
        // uses the image.
        let texture = unsafe {
            device.new_texture_with_vk_image(
                &target(PixelFormat::RGBA8Unorm),
                image,
                &info,
                VkImageLayouts::new(
                    vk::ImageLayout::UNDEFINED.into(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL.into(),
                ),
            )
        }
        .unwrap();
        let queues = [
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        ];
        let handles: BTreeSet<vk::Queue> = queues
            .iter()
            // SAFETY: the handle is only compared.
            .map(|queue| unsafe { queue.with_vk_queue(|_, vk_queue, _| vk_queue) })
            .collect();
        let size = u64::from(SIDE * SIDE * 4);
        let readback = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        let colors = [
            ([1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
            ([0.0, 0.0, 1.0, 1.0], [0, 0, 255, 255]),
            ([1.0, 1.0, 1.0, 0.0], [255, 255, 255, 0]),
        ];
        for (round, (color, texel)) in colors.into_iter().enumerate() {
            let (writer, reader) = (&queues[round % 2], &queues[1 - round % 2]);
            let clearing = writer.command_buffer().unwrap();
            clear(&clearing, &texture, color);
            clearing.commit().unwrap();
            let reading = reader.command_buffer().unwrap();
            copy_out(&reading, &texture, &readback);
            run(&reading);
            clearing.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(clearing.error(), None);
            let bytes = read_buffer(&readback, 0, size).unwrap();
            assert_eq!(bytes, filled(texel), "round {round}");
        }
        eprintln!(
            "runtime present: a wrapped image ordered across two queues on {} VkQueue(s)",
            handles.len()
        );
        assert_made_once(&messages, &WRAPPED, &shared_kinds(&device));
        drop(texture);
        // SAFETY: nothing uses them: the texture is gone and every command buffer completed.
        unsafe {
            raw.device.destroy_image(image, None);
            raw.device.free_memory(image_memory, None);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A texture over an image the test makes through `Device::raw` (`UNDEFINED` at first, `GENERAL`
/// between the runtime's batches), of [`target`]'s `RGBA8Unorm` descriptor placed `placement`,
/// with the image and its memory, which [`destroy_callers`] destroys once nothing uses them.
fn callers_texture(
    device: &Device,
    placement: IdPlacement,
) -> (Texture, vk::Image, vk::DeviceMemory) {
    // SAFETY: used only while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let info = shared_among(
        image_info(vk::Format::R8G8B8A8_UNORM, usage),
        raw.queue_families,
    );
    // SAFETY: the device is live; `destroy_callers` destroys them after their last use.
    let (image, bound) = unsafe { made_image(raw.device, &memory, &info) };
    let mut descriptor = target(PixelFormat::RGBA8Unorm);
    descriptor.placement = placement;
    // SAFETY: a live image of the device, made with `info`, bound, in `UNDEFINED` now; it
    // outlives the texture and every command buffer that uses it, and nothing else uses it.
    let texture = unsafe {
        device.new_texture_with_vk_image(
            &descriptor,
            image,
            &info,
            VkImageLayouts::new(
                vk::ImageLayout::UNDEFINED.into(),
                vk::ImageLayout::GENERAL.into(),
            ),
        )
    }
    .unwrap();
    (texture, image, bound)
}

/// Destroy `image` and its memory `bound`, which [`callers_texture`] made on `device`.
///
/// # Safety
///
/// Nothing uses them: their texture is gone and every command buffer that used it completed and
/// is gone.
unsafe fn destroy_callers(device: &Device, image: vk::Image, bound: vk::DeviceMemory) {
    // SAFETY: used only while the device lives; the caller's contract.
    unsafe {
        let raw = device.raw();
        raw.device.destroy_image(image, None);
        raw.device.free_memory(bound, None);
    }
}

/// A texture over the caller's image placed [`IdPlacement::Outside`] takes no resource ID, as a
/// texture `new_texture` makes there does (`resource_ids::an_outside_texture_takes_no_id`), so a
/// consumer's swapchain textures move no program texture's ID. After a reservation of 20000 it
/// answers 0, and the plain texture made next takes 20001, the ID it takes on a device with no
/// caller's texture; one placed `Any` takes 20001 itself and moves the plain texture to 20002.
/// It starts no space: a reservation after it counts. It renders and copies as one placed `Any`
/// does: what a clear of it wrote is what a copy of it reads.
#[test]
fn a_callers_image_placed_outside_takes_no_id() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let plain = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 4, 4, false);
    let reserve = |device: &Device| {
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(20000)])
            .unwrap()
    };
    // On a new device: the reservation, a texture over the caller's image at `placement` where
    // there is one, and a plain texture; the IDs they answer.
    let ids = |placement: Option<IdPlacement>| -> (Option<u64>, u64) {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        reserve(&device);
        let callers = placement.map(|placement| callers_texture(&device, placement));
        let next = device.new_texture(&plain).unwrap();
        let answered = (
            callers
                .as_ref()
                .map(|(texture, ..)| texture.gpu_resource_id().0),
            next.gpu_resource_id().0,
        );
        if let Some((texture, image, bound)) = callers {
            drop(texture);
            // SAFETY: the texture is gone and no command buffer used it.
            unsafe { destroy_callers(&device, image, bound) };
        }
        answered
    };
    let without = ids(None);
    let any = ids(Some(IdPlacement::Any));
    let outside = ids(Some(IdPlacement::Outside));
    eprintln!(
        "runtime ids: the plain texture after a reservation of 20000 answers {} with no \
         caller's texture; {:?} after one placed Any; {:?} after one placed Outside",
        without.1, any, outside
    );
    assert_eq!(without, (None, 20001));
    assert_eq!(any, (Some(20001), 20002), "a caller's texture placed Any");
    assert_eq!(
        outside,
        (Some(0), 20001),
        "a caller's texture placed Outside"
    );
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let (texture, image, bound) = callers_texture(&device, IdPlacement::Outside);
        reserve(&device);
        let next = device.new_texture(&plain).unwrap();
        assert_eq!(
            [texture.gpu_resource_id().0, next.gpu_resource_id().0],
            [0, 20001],
            "a reservation after a caller's texture placed Outside"
        );
        let queue = device.new_command_queue().unwrap();
        let size = u64::from(SIDE * SIDE * 4);
        let readback = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        clear(&command_buffer, &texture, [0.0, 1.0, 0.0, 1.0]);
        copy_out(&command_buffer, &texture, &readback);
        run(&command_buffer);
        assert_eq!(
            read_buffer(&readback, 0, size).unwrap(),
            filled([0, 255, 0, 255])
        );
        drop((texture, command_buffer, queue));
        // SAFETY: the texture is gone, and the command buffer that used it completed and is gone.
        unsafe { destroy_callers(&device, image, bound) };
    }
    layer_reported_nothing_at_all(&messages);
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// Each misuse of the interop calls is refused typed: a null semaphore and the runtime's own
/// timeline semaphores, binary semaphore commands with an encoder open or after commit; a
/// wrapped image of `Shared` storage, a null image, an image whose format or usage contradicts
/// the descriptor, a moving image with untracked hazards, a home layout of `UNDEFINED`, and a
/// texture buffer. A shared texture is shareable and its handle opens a texture over the same
/// image, which hazard tracking orders as one: a clear through one is what a copy through the
/// other reads. A view, an ordinary texture, and another device's open are refused, and so is
/// `Shared` storage as a shared texture; a texture buffer is made shareable
/// (`buffer_textures::a_shared_texture_buffer_opens_over_its_texels`).
#[test]
fn interop_refusals_and_shared_textures() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // SAFETY: used only while the device lives.
        let raw = unsafe { device.raw() };
        let queue = device.new_command_queue().unwrap();

        let command_buffer = queue.command_buffer().unwrap();
        let event = device.new_shared_event().unwrap();
        // SAFETY: read only while the event lives.
        let timeline = unsafe { event.vk_semaphore() };
        let binary = binary_semaphore(raw.device);
        // SAFETY: each call is refused before it records anything.
        unsafe {
            assert_eq!(
                command_buffer.encode_wait_for_vk_semaphore(vk::Semaphore::null()),
                Err(invalid(
                    "encode_wait_for_vk_semaphore",
                    Problem::Semaphore(SemaphoreProblem::Null)
                ))
            );
            assert_eq!(
                command_buffer.encode_signal_vk_semaphore(timeline),
                Err(invalid(
                    "encode_signal_vk_semaphore",
                    Problem::Semaphore(SemaphoreProblem::Timeline)
                ))
            );
            let blit = command_buffer.blit_command_encoder().unwrap();
            assert!(matches!(
                command_buffer.encode_signal_vk_semaphore(binary),
                Err(Error::Invalid(_))
            ));
            blit.end_encoding().unwrap();
        }
        run(&command_buffer);
        // SAFETY: refused before it records anything.
        assert!(matches!(
            unsafe { command_buffer.encode_wait_for_vk_semaphore(binary) },
            Err(Error::Invalid(_))
        ));

        // SAFETY: a physical device of the instance.
        let memory = unsafe {
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device)
        };
        let full = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let info = shared_among(
            image_info(vk::Format::R8G8B8A8_UNORM, full),
            raw.queue_families,
        );
        let narrow = shared_among(
            image_info(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            raw.queue_families,
        );
        // SAFETY: the device is live; the images are destroyed below, never wrapped.
        let ((image, image_memory), (narrow_image, narrow_memory)) = unsafe {
            (
                made_image(raw.device, &memory, &info),
                made_image(raw.device, &memory, &narrow),
            )
        };
        const CALL: &str = "new_texture_with_vk_image";
        let moving = VkImageLayouts::new(
            vk::ImageLayout::UNDEFINED.into(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL.into(),
        );
        let wrap = |descriptor: &TextureDescriptor,
                    image: vk::Image,
                    info: &vk::ImageCreateInfo<'_>,
                    layouts: VkImageLayouts| {
            // SAFETY: each call is refused before the texture would exist.
            unsafe { device.new_texture_with_vk_image(descriptor, image, info, layouts) }
                .map(|_| ())
        };
        let refused = |problem| Err(invalid(CALL, Problem::WrappedImage(problem)));
        let mut shared = target(PixelFormat::RGBA8Unorm);
        shared.resource_options = ResourceOptions::shared();
        assert_eq!(
            wrap(&shared, image, &info, moving),
            Err(invalid(CALL, Problem::StorageMode))
        );
        let rgba = target(PixelFormat::RGBA8Unorm);
        assert_eq!(
            wrap(&rgba, vk::Image::null(), &info, moving),
            refused(WrappedImageProblem::NullImage)
        );
        assert_eq!(
            wrap(&target(PixelFormat::BGRA8Unorm), image, &info, moving),
            refused(WrappedImageProblem::Format {
                image: vk::Format::R8G8B8A8_UNORM.into(),
                texture: vk::Format::B8G8R8A8_UNORM.into(),
            })
        );
        assert!(matches!(
            wrap(&rgba, narrow_image, &narrow, moving),
            Err(Error::Invalid(Invalid {
                problem: Problem::WrappedImage(WrappedImageProblem::Usage { .. }),
                ..
            }))
        ));
        let mut untracked = rgba;
        untracked.resource_options = metal2vulkan_runtime::ResourceOptions::new(
            StorageMode::Private,
            metal2vulkan_runtime::CpuCacheMode::DefaultCache,
            HazardTrackingMode::Untracked,
        );
        assert_eq!(
            wrap(&untracked, image, &info, moving),
            refused(WrappedImageProblem::Untracked)
        );
        assert_eq!(
            wrap(
                &rgba,
                image,
                &info,
                VkImageLayouts::new(
                    vk::ImageLayout::GENERAL.into(),
                    vk::ImageLayout::UNDEFINED.into()
                )
            ),
            refused(WrappedImageProblem::Layout {
                layout: vk::ImageLayout::UNDEFINED.into()
            })
        );
        let texels = TextureDescriptor::texture_buffer(
            PixelFormat::RGBA8Unorm,
            64,
            ResourceOptions::private(),
            TextureUsage::SHADER_READ,
        );
        assert_eq!(
            wrap(&texels, image, &info, moving),
            refused(WrappedImageProblem::ImageType {
                image: vk::ImageType::TYPE_2D.into()
            })
        );
        // SAFETY: never wrapped; nothing uses them.
        unsafe {
            raw.device.destroy_image(image, None);
            raw.device.free_memory(image_memory, None);
            raw.device.destroy_image(narrow_image, None);
            raw.device.free_memory(narrow_memory, None);
            raw.device.destroy_semaphore(binary, None);
        }

        let mut descriptor = target(PixelFormat::RGBA8Unorm);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let original = device.new_shared_texture(&descriptor).unwrap();
        assert!(original.is_shareable());
        let handle = original.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        assert!(opened.is_shareable());
        assert_eq!((opened.width(), opened.height()), (16, 16));
        // SAFETY: read only while the textures live.
        unsafe { assert_eq!(opened.vk_image(), original.vk_image()) };
        let other_queue = device.new_command_queue().unwrap();
        let clearing = queue.command_buffer().unwrap();
        clear(&clearing, &original, [0.0, 0.0, 1.0, 1.0]);
        clearing.commit().unwrap();
        let size = u64::from(SIDE * SIDE * 4);
        let readback = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        let reading = other_queue.command_buffer().unwrap();
        copy_out(&reading, &opened, &readback);
        run(&reading);
        clearing.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            read_buffer(&readback, 0, size).unwrap(),
            filled([0, 0, 255, 255])
        );
        let view = original
            .new_texture_view(PixelFormat::RGBA8UnormSRGB)
            .unwrap();
        let ordinary = device.new_texture(&descriptor).unwrap();
        for texture in [&view, &ordinary] {
            assert!(!texture.is_shareable());
            assert_eq!(
                texture.new_shared_texture_handle().unwrap_err(),
                Error::Nil(Nil::new("new_shared_texture_handle", Problem::NotShareable))
            );
        }
        let mut not_private = descriptor;
        not_private.resource_options = ResourceOptions::shared();
        assert_eq!(
            device.new_shared_texture(&not_private).unwrap_err(),
            Error::Nil(Nil::new("new_shared_texture", Problem::StorageMode))
        );
        assert!(device.new_shared_texture(&texels).unwrap().is_shareable());
        let other = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        assert_eq!(
            other
                .new_shared_texture_with_handle(&handle, IdPlacement::Any)
                .unwrap_err(),
            invalid("new_shared_texture_with_handle", Problem::ForeignObject)
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A device made with the presenting extensions whose feature structures the runtime knows
/// (`VK_KHR_swapchain` with `VK_EXT_swapchain_maintenance1` and
/// `VK_EXT_present_mode_fifo_latest_ready`, each where the device lists it) has their features
/// enabled, under the validation layer, which reports nothing; where the device lacks one, it is
/// refused, naming the feature. Without the instance extensions they require (`VK_KHR_surface`
/// for `VK_KHR_swapchain`), the options are refused, naming the instance extension, before an
/// instance is made, so the layer reports nothing of them either.
#[test]
fn listed_presenting_extensions_are_made_with_their_features() {
    /// `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR`, which `ash 0.38` predates.
    #[repr(C)]
    struct FifoLatestReady {
        s_type: vk::StructureType,
        p_next: *mut c_void,
        present_mode_fifo_latest_ready: vk::Bool32,
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let probe = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // SAFETY: read while the device lives; only instance-level queries of its physical
        // device are made.
        let raw = unsafe { probe.raw() };
        let (instance, physical) = (raw.instance, raw.physical_device);
        // SAFETY: a physical device of the instance.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical) }
            .expect("the device's extensions");
        let lists = |name: &CStr| {
            listed
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(name))
        };
        let maintenance = c"VK_EXT_swapchain_maintenance1";
        let fifo = c"VK_EXT_present_mode_fifo_latest_ready";
        let mut swapchain = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
        let mut latest = FifoLatestReady {
            s_type: vk::StructureType::from_raw(1_000_361_000),
            p_next: std::ptr::null_mut(),
            present_mode_fifo_latest_ready: vk::FALSE,
        };
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if lists(maintenance) {
            features = features.push_next(&mut swapchain);
        }
        if lists(fifo) {
            latest.p_next = features.p_next;
            features.p_next = std::ptr::from_mut(&mut latest).cast();
        }
        // SAFETY: the chain lives for the call, of structures of extensions the device lists.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        // Every venue of the lane lists `VK_KHR_swapchain` (the presenting test's gap otherwise).
        let mut wanted = vec![ash::khr::swapchain::NAME.to_owned()];
        let mut missing = None;
        for (name, feature, has) in [
            (
                maintenance,
                "swapchainMaintenance1",
                swapchain.swapchain_maintenance1,
            ),
            (
                fifo,
                "presentModeFifoLatestReady",
                latest.present_mode_fifo_latest_ready,
            ),
        ] {
            if lists(name) {
                wanted.push(name.to_owned());
                if has == vk::FALSE && missing.is_none() {
                    missing = Some(feature);
                }
            }
        }
        // The instance extensions the device extensions require, as a consumer that presents
        // enables them: `VK_KHR_surface` (`VK_KHR_swapchain`'s), and with it
        // `VK_KHR_get_surface_capabilities2` and `VK_EXT_surface_maintenance1`
        // (`VK_EXT_swapchain_maintenance1`'s).
        // SAFETY: an instance-level query of the loader the device was made through.
        let instance_listed = unsafe { raw.entry.enumerate_instance_extension_properties(None) }
            .expect("the loader's instance extensions");
        let surface = [
            ash::khr::surface::NAME,
            ash::khr::get_surface_capabilities2::NAME,
            ash::ext::surface_maintenance1::NAME,
        ]
        .into_iter()
        .filter(|&name| {
            instance_listed
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(name))
        })
        .map(CStr::to_owned)
        .collect::<Vec<_>>();
        drop(probe);
        // The instance extensions the runtime refuses the device extensions without, of those
        // the loader does not list (`VK_KHR_surface` always lists where the lane presents).
        let unlisted = (lists(maintenance)
            && !surface
                .iter()
                .any(|name| name.as_c_str() == ash::ext::surface_maintenance1::NAME))
        .then_some("VK_EXT_surface_maintenance1");
        let refused = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.device_extensions.extend(wanted.iter().cloned());
        })
        .map(drop)
        .expect_err("options refused without the surface instance extensions");
        assert!(
            refused.contains("VK_KHR_swapchain") && refused.contains("VK_KHR_surface"),
            "{refused}"
        );
        let made = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.instance_extensions.extend(surface.iter().cloned());
            options.device_extensions.extend(wanted.iter().cloned());
        });
        eprintln!(
            "API device extensions {wanted:?}, instance extensions {surface:?}, missing feature \
             {missing:?}"
        );
        match unlisted.or(missing) {
            None => drop(made.expect("a device with the extensions' features")),
            Some(missing) => {
                let refused = made.map(drop).expect_err("a refused device");
                assert!(refused.contains(missing), "{refused}");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
