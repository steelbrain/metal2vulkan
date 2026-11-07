//! Device tests of the caller's own device commands: loaded through `Device::device_proc_addr` and
//! called, where the Vulkan loader serves them itself, inside `Device::with_loader_word`, after
//! MoltenVK overwrote the device's loader word, on the device the cfg selects.
//!
//! On MoltenVK the overwrite is MoltenVK's own: with its performance tracking logging each
//! activity at once, a submission's activity is a message naming the device, which the relay's
//! debug-utils messenger hears, and MoltenVK writes `ICD_LOADER_MAGIC` over the device's first
//! word as it names it there (the test requires the magic). Elsewhere the test writes the magic
//! itself, as MoltenVK would. A swapchain needs a headless surface that the device's primary
//! queue family presents to; a venue without one prints its gap and only loads.

use super::{
    layer_reported_nothing_at_all, nothing_reported_without_the_layer, validated_device,
    validated_device_with, Messages, ScratchCache,
};
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::TEST_DEVICE;
use ash::vk::{self, Handle};
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{Device, LayerSetting, LayerSettingValue, ResourceOptions};
use std::collections::BTreeSet;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// `ICD_LOADER_MAGIC` (`vk_icd.h`): what a driver initializes a dispatchable object's first word
/// to, and what MoltenVK 1.4.1 writes back over it as it names the object in a message.
const ICD_LOADER_MAGIC: usize = 0x01CD_C0DE;

/// The swapchain's images: 16 × 16 texels.
const SIDE: u32 = 16;

/// MoltenVK's settings under which every activity it tracks is logged at once, a message naming
/// the device (MoltenVK 1.4.1, `MVKDevice::updateActivityPerformance`): `performanceTracking`, a
/// `VkBool32`, and `activityPerformanceLoggingStyle`, whose 32 bits MoltenVK reads from the
/// setting's value (`MVKInstance::initMVKConfig`), so a `VkBool32` of 1 is
/// `MVK_CONFIG_ACTIVITY_PERFORMANCE_LOGGING_STYLE_IMMEDIATE`. Other drivers ignore them.
fn immediate_tracking() -> [LayerSetting; 2] {
    [
        c"MVK_CONFIG_PERFORMANCE_TRACKING",
        c"MVK_CONFIG_ACTIVITY_PERFORMANCE_LOGGING_STYLE",
    ]
    .map(|name| {
        LayerSetting::new(
            c"MoltenVK".to_owned(),
            name.to_owned(),
            LayerSettingValue::Bool(true),
        )
    })
}

/// The device's first pointer-sized word.
fn first_word(device: &Device) -> *mut usize {
    // SAFETY: only the handle is read.
    let handle = unsafe { device.raw() }.device.handle().as_raw();
    std::ptr::with_exposed_provenance_mut(handle as usize)
}

/// The device's first word now.
fn word(device: &Device) -> usize {
    // SAFETY: a dispatchable handle points at its object's first pointer-sized entry, which lives
    // as long as the device.
    unsafe { first_word(device).read_volatile() }
}

/// The device's first word overwritten with the magic: on MoltenVK by MoltenVK, as it logs a
/// submission's activities naming the device; elsewhere by the test.
///
/// On MoltenVK two command buffers run, one after the other. The runtime puts the word back
/// before every call of its own that the loader serves (`loader_word.rs`), and a command buffer's
/// completion gives its command pool back, which makes the pool a secondary command buffer
/// (`vkAllocateCommandBuffers`, which the loader serves) where it has none (`spares.rs`): that
/// happens before `wait_until_completed_for` returns. The first command buffer leaves its pool
/// among the spares with a secondary, so the second takes a pool that needs none, and its
/// completion calls nothing the loader serves: the magic MoltenVK wrote as it ran stays.
fn overwrite(device: &Device, moltenvk: bool) {
    if moltenvk {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(256, ResourceOptions::private()).unwrap();
        for _ in 0..2 {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..256, 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
    } else {
        // SAFETY: the device's first word, which only the loader reads and the runtime puts back
        // before every call of its own the loader serves.
        unsafe { first_word(device).write_volatile(ICD_LOADER_MAGIC) };
    }
    assert_eq!(
        word(device),
        ICD_LOADER_MAGIC,
        "the device's first word holds ICD_LOADER_MAGIC (moltenvk: {moltenvk})"
    );
}

/// The base address of the loaded image (the library) that holds `address`.
fn image_of(address: *const c_void) -> (usize, String) {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: `info` is writable; `dladdr` only reads the process's image list.
    let found = unsafe { libc::dladdr(address, info.as_mut_ptr()) };
    assert_ne!(found, 0, "dladdr found no image holding {address:?}");
    // SAFETY: `dladdr` filled `info`.
    let info = unsafe { info.assume_init() };
    let name = if info.dli_fname.is_null() {
        String::new()
    } else {
        // SAFETY: a NUL-terminated path the image list keeps.
        unsafe { CStr::from_ptr(info.dli_fname) }
            .to_string_lossy()
            .into_owned()
    };
    (info.dli_fbase as usize, name)
}

/// What the probe device's loader and device list, by name.
struct Listed {
    instance: BTreeSet<CString>,
    device: BTreeSet<CString>,
}

/// The extensions of the loader and of the device the cfg selects.
fn listed(messages: &Messages, cache: &Path) -> Listed {
    let probe = validated_device(TEST_DEVICE, messages, cache).unwrap();
    // SAFETY: only instance-level queries of the probe's physical device are made, while it lives.
    let raw = unsafe { probe.raw() };
    let names = |properties: Vec<vk::ExtensionProperties>| -> BTreeSet<CString> {
        properties
            .iter()
            .filter_map(|p| p.extension_name_as_c_str().ok().map(CStr::to_owned))
            .collect()
    };
    // SAFETY: the loader the probe was made through.
    let instance =
        names(unsafe { raw.entry.enumerate_instance_extension_properties(None) }.unwrap());
    // SAFETY: a physical device of the instance.
    let device = names(
        unsafe {
            raw.instance
                .enumerate_device_extension_properties(raw.physical_device)
        }
        .unwrap(),
    );
    Listed { instance, device }
}

/// A device made through `validated_device_with`, with MoltenVK's immediate tracking, the
/// headless surface's instance extensions and `VK_KHR_swapchain` where `headless`,
/// `VK_EXT_metal_objects` where the device lists it, and the validation layer where `layered`.
fn make(
    messages: &Messages,
    cache: &Path,
    listed: &Listed,
    headless: bool,
    layered: bool,
) -> Device {
    validated_device_with(TEST_DEVICE, messages, cache, |options| {
        if !layered {
            options.layers.clear();
        }
        options.layer_settings.extend(immediate_tracking());
        if headless {
            options.instance_extensions.extend([
                ash::khr::surface::NAME.to_owned(),
                ash::ext::headless_surface::NAME.to_owned(),
            ]);
            options
                .device_extensions
                .push(ash::khr::swapchain::NAME.to_owned());
        }
        if listed.device.contains(ash::ext::metal_objects::NAME) {
            options
                .device_extensions
                .push(ash::ext::metal_objects::NAME.to_owned());
        }
    })
    .unwrap()
}

/// On MoltenVK, a device is made, primed and used while MoltenVK names it in a message at each
/// activity it tracks (its immediate performance logging, heard by the relay's messenger), so
/// its first word holds the magic from the priming's own submission on: the runtime puts the
/// word back before every call the loader serves, priming's command buffer allocations
/// included. Without that restore the device's creation died by `SIGSEGV` at `0x1cdc0de` in the
/// loader's `vkAllocateCommandBuffers` (an M4 under MoltenVK 1.4.1). Then a
/// command buffer is committed on a new queue (an allocation through the loader with the magic
/// in the word) and the device is dropped (`vkDestroyDevice`, likewise). Elsewhere nothing writes
/// the word, and the test says so and ends.
#[test]
fn a_device_is_made_and_used_while_moltenvk_names_it_in_each_message() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.layer_settings.extend(immediate_tracking());
        })
        .unwrap();
        if device.properties().description.driver_id != Some(DriverId::MOLTENVK) {
            eprintln!("API device commands: not MoltenVK, nothing writes the word; not run");
            return;
        }
        overwrite(&device, true);
        overwrite(&device, true);
    }
    nothing_reported_without_the_layer(&messages, "nothing MoltenVK said fails");
}

/// The caller's swapchain commands load through `Device::device_proc_addr` after the device's
/// first word was overwritten, and make a swapchain over a headless surface through
/// `vkCreateSwapchainKHR` inside `Device::with_loader_word`, with and without the validation
/// layer, which reports nothing. Without a layer, `vkCreateSwapchainKHR` is the loader's own
/// terminator and the swapchain's other commands are the driver's own, which run with the magic
/// in the word: its images, an acquire, its destruction, and on MoltenVK
/// `vkExportMetalObjectsEXT`. With the layer, every call is made inside `with_loader_word`.
#[test]
fn the_caller_s_commands_load_and_make_a_swapchain_after_moltenvk_overwrote_the_device_s_word() {
    for layered in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let listed = listed(&messages, cache.path());
            let headless = listed.instance.contains(ash::khr::surface::NAME)
                && listed.instance.contains(ash::ext::headless_surface::NAME)
                && listed.device.contains(ash::khr::swapchain::NAME);
            let device = make(&messages, cache.path(), &listed, headless, layered);
            let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
            eprintln!(
                "API device commands: layered {layered}, MoltenVK {moltenvk}, headless {headless}"
            );
            caller_s_commands(&device, moltenvk, headless, layered);
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// The test's steps on `device` (the test's docs).
fn caller_s_commands(device: &Device, moltenvk: bool, headless: bool, layered: bool) {
    let load = |name: &CStr| {
        device
            .device_proc_addr(name)
            .map_or(std::ptr::null(), |command| command as *const c_void)
    };
    overwrite(device, moltenvk);
    let unknown = device.device_proc_addr(c"vkM2vNoSuchCommand");
    assert!(unknown.is_none(), "an unknown name loads as null");
    if !headless {
        eprintln!("API device commands: gap, no headless surface or VK_KHR_swapchain; loads only");
        return;
    }
    overwrite(device, moltenvk);
    let swapchain_fn = ash::khr::swapchain::DeviceFn::load(load);
    // SAFETY: only handles are read, while the device lives.
    let raw = unsafe { device.raw() };
    let handle = raw.device.handle();
    if !layered {
        let loader = image_of(raw.entry.static_fn().get_instance_proc_addr as *const c_void);
        let create = image_of(swapchain_fn.create_swapchain_khr as *const c_void);
        let own = [
            (
                "vkAcquireNextImageKHR",
                image_of(swapchain_fn.acquire_next_image_khr as *const c_void),
            ),
            (
                "vkQueuePresentKHR",
                image_of(swapchain_fn.queue_present_khr as *const c_void),
            ),
            (
                "vkGetSwapchainImagesKHR",
                image_of(swapchain_fn.get_swapchain_images_khr as *const c_void),
            ),
            (
                "vkDestroySwapchainKHR",
                image_of(swapchain_fn.destroy_swapchain_khr as *const c_void),
            ),
        ];
        eprintln!(
            "API device commands: loader {}; vkCreateSwapchainKHR {}; {}",
            loader.1,
            create.1,
            own.iter()
                .map(|(name, image)| format!("{name} {}", image.1))
                .collect::<Vec<_>>()
                .join("; ")
        );
        assert_eq!(
            create.0, loader.0,
            "vkCreateSwapchainKHR is the loader's own"
        );
        for (name, image) in &own {
            assert_ne!(image.0, loader.0, "{name} is the driver's own");
        }
    }

    let surface_fn = ash::khr::surface::Instance::new(raw.entry, raw.instance);
    let headless_fn = ash::ext::headless_surface::Instance::new(raw.entry, raw.instance);
    // SAFETY: the instance enabled both extensions.
    let surface = unsafe {
        headless_fn.create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None)
    }
    .unwrap();
    // SAFETY: a surface of the instance and a family of the device.
    let presents = unsafe {
        surface_fn.get_physical_device_surface_support(
            raw.physical_device,
            raw.queue_family,
            surface,
        )
    }
    .unwrap();
    if !presents {
        eprintln!(
            "API device commands: gap, queue family {} presents to no headless surface",
            raw.queue_family
        );
        // SAFETY: nothing was made from the surface.
        unsafe { surface_fn.destroy_surface(surface, None) };
        return;
    }
    // SAFETY: a surface of the instance, of a physical device that presents to it.
    let caps = unsafe {
        surface_fn.get_physical_device_surface_capabilities(raw.physical_device, surface)
    }
    .unwrap();
    // SAFETY: as above.
    let formats =
        unsafe { surface_fn.get_physical_device_surface_formats(raw.physical_device, surface) }
            .unwrap();
    let format = formats[0];
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
        .surface(surface)
        .min_image_count(count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(vk::Extent2D {
            width: SIDE,
            height: SIDE,
        })
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(alpha)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true);

    overwrite(device, moltenvk);
    let mut swapchain = vk::SwapchainKHR::null();
    let created = device.with_loader_word(|| {
        // SAFETY: the surface is the instance's, the device presents to it from its primary
        // family, and the loader's terminator reads the word put back just now.
        unsafe {
            (swapchain_fn.create_swapchain_khr)(handle, &info, std::ptr::null(), &mut swapchain)
        }
    });
    assert_eq!(created, vk::Result::SUCCESS, "vkCreateSwapchainKHR");

    let images = |swapchain_fn: &ash::khr::swapchain::DeviceFn| -> u32 {
        let mut count = 0;
        // SAFETY: a live swapchain of the device; the count alone is asked for.
        let asked = unsafe {
            (swapchain_fn.get_swapchain_images_khr)(
                handle,
                swapchain,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(asked, vk::Result::SUCCESS, "vkGetSwapchainImagesKHR");
        count
    };
    if layered {
        let count = device.with_loader_word(|| images(&swapchain_fn));
        assert!(count >= 1, "the swapchain has images");
        device.with_loader_word(|| {
            // SAFETY: a live swapchain of the device, no image of which is in use.
            unsafe { (swapchain_fn.destroy_swapchain_khr)(handle, swapchain, std::ptr::null()) }
        });
    } else {
        // The driver's own commands, with the magic in the word: nothing puts it back, so
        // whatever loads through `device_proc_addr`, which puts it back, loads before.
        let metal_fn = moltenvk.then(|| ash::ext::metal_objects::DeviceFn::load(load));
        overwrite(device, moltenvk);
        assert!(images(&swapchain_fn) >= 1, "the swapchain has images");
        // SAFETY: a fence of the caller's own, destroyed below.
        let fence = unsafe {
            raw.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .unwrap();
        let mut index = 0;
        // SAFETY: a live swapchain of the device with no image acquired, and an unsignaled fence.
        let acquired = unsafe {
            (swapchain_fn.acquire_next_image_khr)(
                handle,
                swapchain,
                WAIT.as_nanos() as u64,
                vk::Semaphore::null(),
                fence,
                &mut index,
            )
        };
        assert!(
            matches!(acquired, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR),
            "vkAcquireNextImageKHR: {acquired:?}"
        );
        // The fence the acquire signals, under the bounded wait's hang protocol.
        wait::vulkan::wait_for_fence(
            raw.device,
            fence,
            SubmissionBackend::vulkan(TEST_DEVICE),
            "the test's acquire",
        )
        .unwrap();
        if let Some(metal_fn) = metal_fn {
            let mut metal_device = vk::ExportMetalDeviceInfoEXT::default();
            let mut export = vk::ExportMetalObjectsInfoEXT::default().push_next(&mut metal_device);
            // SAFETY: the device enabled `VK_EXT_metal_objects`; the chain lives for the call.
            unsafe { (metal_fn.export_metal_objects_ext)(handle, &mut export) };
            assert!(!metal_device.mtl_device.is_null(), "the MTLDevice");
        }
        assert_eq!(
            word(device),
            ICD_LOADER_MAGIC,
            "the driver's own commands ran with the magic in the word"
        );
        // SAFETY: a live swapchain of the device; its acquired image is no longer in use.
        unsafe { (swapchain_fn.destroy_swapchain_khr)(handle, swapchain, std::ptr::null()) };
        // SAFETY: signaled, waited for, and used by nothing else.
        unsafe { raw.device.destroy_fence(fence, None) };
    }
    // SAFETY: the swapchain over it is destroyed.
    unsafe { surface_fn.destroy_surface(surface, None) };
}

/// The devices [`the_device_s_checks_run_beside_the_caller_s_loader_served_calls`] makes, each
/// over a cache of its own, so that each runs its checks.
const BESIDE_DEVICES: usize = 5;

/// The caller's loads through `Device::device_proc_addr` (each a `vkGetDeviceProcAddr` the
/// loader serves, reading the device's word) between two looks at whether the checks answered.
const LOADS_A_ROUND: usize = 64;

/// On MoltenVK, with each activity logged at once, the device's checks that run beside the
/// program on threads of their own (its submission checks, its in-place check, its queue-switch
/// check where its driver lends two `VkQueue`s it can time) run while the caller's thread makes
/// calls the loader serves, [`LOADS_A_ROUND`] loads of a command through
/// `Device::device_proc_addr` at a time, until every check answered. The caller's thread makes no
/// GPU work of its own, so every write of the magic over the device's word comes from the checks'
/// work. Each check holds the device's loader word across its submissions, and across its whole
/// steps where the runtime found the word overwritten (`loader_word.rs` of the runtime), so no
/// load reads the magic as its table: no `SIGSEGV`, on [`BESIDE_DEVICES`] devices. Printed (`API
/// device commands beside the checks`): each device's loads, and which checks ran. Elsewhere
/// nothing writes the word, and the test says so and ends.
#[test]
fn the_device_s_checks_run_beside_the_caller_s_loader_served_calls() {
    for made in 0..BESIDE_DEVICES {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                options.layers.clear();
                options.layer_settings.extend(immediate_tracking());
                options.policy.host_page_textures =
                    metal2vulkan::device::HostPageTexturePolicy::InPlace;
            })
            .unwrap();
            if device.properties().description.driver_id != Some(DriverId::MOLTENVK) {
                eprintln!(
                    "API device commands beside the checks: not MoltenVK, nothing writes the \
                     word; not run"
                );
                return;
            }
            let answered = |device: &Device| {
                device.settle_submit_checks(Duration::ZERO)
                    && device.settle_in_place_checks(Duration::ZERO)
                    && device.settle_queue_switch_check(Duration::ZERO)
            };
            let start = std::time::Instant::now();
            let mut loads = 0;
            while !answered(&device) {
                assert!(start.elapsed() < WAIT, "the device's checks never answered");
                for _ in 0..LOADS_A_ROUND {
                    let loaded = device.device_proc_addr(c"vkGetDeviceQueue");
                    assert!(loaded.is_some(), "vkGetDeviceQueue loads");
                }
                loads += LOADS_A_ROUND;
            }
            let counts = device.work_counts();
            eprintln!(
                "API device commands beside the checks: device {made}, {loads} loads: \
                 submission checks run {}, in-place checks run {}, queue-switch checks run {}; \
                 {}; {}",
                counts.submit_checks_run,
                counts.in_place_checks_run,
                counts.queue_switch_checks_run,
                device.submit_check_report(),
                device.queue_switch_check_report()
            );
            assert_eq!(counts.submit_checks_run, 1, "the submission checks ran");
        }
        nothing_reported_without_the_layer(&messages, "nothing MoltenVK said fails");
    }
}
