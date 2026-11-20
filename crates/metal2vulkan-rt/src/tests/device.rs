//! The instance, the device and its counters: that the handles cross threads, that an object's
//! handle is two words, and, under a device cfg, that every device of the cfg's kind opens,
//! presents on the queue its family offers, is torn down with its gauges at zero, and enables the
//! consumer's extensions and features it offers.

use crate::{
    BlockId, BlockInfo, BlockMappings, Buffer, CommandBuffer, CommandBufferError,
    CommandBufferStatus, CommandQueue, ConsumerFeature, Counters, Device, DeviceExtension,
    DeviceOptions, Event, Gauges, Instance, Metrics, PhysicalDevice, SharedEvent,
};

#[test]
fn the_handles_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Instance>();
    send_sync::<PhysicalDevice>();
    send_sync::<Device>();
    send_sync::<DeviceOptions>();
    send_sync::<DeviceExtension>();
    send_sync::<ConsumerFeature>();
    send_sync::<Counters>();
    send_sync::<BlockId>();
    send_sync::<BlockInfo>();
    send_sync::<Buffer>();
    send_sync::<std::sync::Arc<dyn BlockMappings>>();
    send_sync::<CommandQueue>();
    send_sync::<CommandBuffer>();
    send_sync::<CommandBufferStatus>();
    send_sync::<CommandBufferError>();
    send_sync::<Event>();
    send_sync::<SharedEvent>();
}

/// A buffer's, a texture's and a sampler state's handle is two words, the object and the device,
/// as an event's and a queue's are: what a command buffer keeps of them is the object alone.
#[test]
fn an_object_handle_is_the_object_and_the_device() {
    use crate::{SamplerState, Texture};
    let two = 2 * std::mem::size_of::<usize>();
    assert_eq!(std::mem::size_of::<Buffer>(), two);
    assert_eq!(std::mem::size_of::<Texture>(), two);
    assert_eq!(std::mem::size_of::<SamplerState>(), two);
    assert_eq!(std::mem::size_of::<Event>(), two);
    assert_eq!(std::mem::size_of::<CommandQueue>(), two);
}

#[test]
fn new_counters_read_zero_from_every_clone() {
    let counters = Counters::new();
    let clone = counters.clone();
    drop(counters);
    assert_eq!(clone.metrics(), Metrics::default());
    assert_eq!(clone.gauges(), Gauges::default());
}

/// Every device of the cfg's kind the loader lists opens (CPU devices under `m2v_cpu_icd`, the
/// others under `m2v_gpu`; the legs pick one with `VK_DRIVER_FILES` and `MESA_VK_DEVICE_SELECT`).
/// Its queue family does graphics and compute, and its memory types have the flags the floor asks
/// of each; the presenting queue is the family's second queue where it has one and the runtime's
/// own where it has one, and it runs a submission from another thread. Dropped before its instance
/// or after, the device is torn down with every gauge at zero.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_of_the_cfgs_kind_opens_presents_and_tears_down_to_zero() {
    use crate::floor::{PRIVATE_MEMORY, RING_MEMORY, SHARED_MEMORY};
    use ash::vk;

    // The instance's handle drops before what it made: its watch checks the log after them.
    let (instance, watch) = crate::tests::validation::instance(&[]).split();
    let physicals = instance.physical_devices().expect("the physical devices");
    let mut opened = Vec::new();
    for physical in &physicals {
        let (_, vk_instance) = instance.raw();
        // SAFETY: `physical` is one of `instance`'s.
        let properties = unsafe { vk_instance.get_physical_device_properties(physical.raw()) };
        let cpu = properties.device_type == vk::PhysicalDeviceType::CPU;
        if cpu != cfg!(m2v_cpu_icd) {
            continue;
        }
        opened.push(physical.clone());
        let name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let device = Device::new(physical, DeviceOptions::default())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let counters = device.counters();

        let raw = device.raw();
        assert_eq!(raw.physical_device, physical.raw());
        // SAFETY: as above.
        let families =
            unsafe { vk_instance.get_physical_device_queue_family_properties(physical.raw()) };
        let family = families[raw.queue_family_index as usize];
        assert!(
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
            "{name}: family {} has flags {:#x}",
            raw.queue_family_index,
            family.queue_flags.as_raw()
        );
        let own = family.queue_count >= 2;
        // SAFETY: the device was made with this family's first queue, and its second where `own`.
        let expected = unsafe {
            raw.device
                .get_device_queue(raw.queue_family_index, u32::from(own))
        };
        // The memory types are the device's, with the flags the floor asks of each.
        let memory_types = device.memory_types();
        // SAFETY: as above.
        let memory = unsafe { vk_instance.get_physical_device_memory_properties(physical.raw()) };
        for (index, flags) in [
            (memory_types.private, PRIVATE_MEMORY),
            (memory_types.shared, SHARED_MEMORY),
            (memory_types.ring, RING_MEMORY),
        ] {
            let kind = memory.memory_types_as_slice()[index as usize];
            assert!(
                kind.property_flags.contains(flags),
                "{name}: memory type {index} has flags {:#x}, not {:#x}",
                kind.property_flags.as_raw(),
                flags.as_raw()
            );
        }
        assert_eq!(
            device.has_unified_memory(),
            matches!(
                properties.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
            )
        );

        // The presenting queue runs a submission, from a thread other than the device's maker.
        let presenting = device.clone();
        let submitted = std::thread::spawn(move || {
            let raw = presenting.raw();
            // SAFETY: the device is live; the fence is destroyed below.
            let fence = unsafe {
                raw.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .expect("a fence");
            let queue = presenting.with_present_queue(|queue| {
                // SAFETY: the queue is held for the call; an empty batch signals the fence.
                unsafe { raw.device.queue_submit2(queue, &[], fence) }.expect("an empty submit");
                queue
            });
            // SAFETY: the fence was submitted; it is not in use once waited for.
            unsafe {
                raw.device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .expect("the fence signals");
                raw.device.destroy_fence(fence, None);
            }
            queue
        })
        .join()
        .expect("the presenting thread");
        assert_eq!(submitted, expected, "{name}: the presenting queue");
        // Set 1's two default samplers, and nothing else.
        assert_eq!(counters.gauges().device_samplers, 2, "{name}");

        drop(device);
        let gauges = counters.gauges();
        for (kind, value) in [
            ("buffers", gauges.buffers),
            ("textures", gauges.textures),
            ("samplers", gauges.samplers),
            ("pipelines", gauges.pipelines),
            ("command_buffers", gauges.command_buffers),
            ("descriptor_slots", gauges.descriptor_slots),
            ("blocks", gauges.blocks),
            ("events", gauges.events),
            ("device_samplers", gauges.device_samplers),
        ] {
            println!("M2V-MATRIX gauge device_teardown/{kind}@drop {value}");
        }
        assert_eq!(gauges, Gauges::default(), "{name}: gauges after teardown");
        println!(
            "device: {name} opened; presenting on {} queue; memory types {:?}; unified memory {}",
            if own { "its own" } else { "the runtime's" },
            memory_types,
            properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                || properties.device_type == vk::PhysicalDeviceType::CPU
        );
    }
    assert!(
        !opened.is_empty(),
        "the loader lists no device of this cfg's kind"
    );
    // A physical device holds its instance, and a device holds both: with the instance's handle
    // and the physical devices dropped first, the device still opens and is torn down last.
    let physical = opened[0].clone();
    drop((instance, physicals, opened));
    let device = Device::new(&physical, DeviceOptions::default())
        .expect("a device after the instance's handle dropped");
    drop(physical);
    drop(device);
    drop(watch);
}

/// A device torn down while its cache directory's removal thread runs waits for it: opened on a
/// directory with two dead generations, its removal paused 200 ms (the test seam
/// `cache::REMOVAL_PAUSE`), and dropped at once, the device's every gauge reads zero,
/// `Gauges::threads` among them, and the dead generations are gone.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_joins_its_cache_directory_s_removal_at_teardown() {
    use crate::cache::{Store, REMOVAL_PAUSE};
    use std::sync::atomic::Ordering::Relaxed;

    REMOVAL_PAUSE.store(200, Relaxed);
    use crate::tests::devices_of_the_cfgs_kind;

    let instance = crate::tests::validation::instance(&[]);
    let devices = devices_of_the_cfgs_kind(&instance);
    assert!(!devices.is_empty(), "no device of this cfg's kind");
    for (name, physical) in &devices {
        let dir = std::env::temp_dir().join(format!("m2v-rt-removal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Generations 1 to 4 made through a store of its own, 4 young; then 1 and 2, dead, refilled
        // as a run cut short in their removal leaves them.
        let store = Store::open(dir.clone(), 1000, Counters::new());
        for n in 0..16_u32 {
            assert!(store.put(&format!("translations/{n:064x}"), &[1; 100]));
        }
        store.settle();
        drop(store);
        for number in [1, 2] {
            let generation = dir.join(format!("g{number}"));
            std::fs::create_dir_all(&generation).expect("the generation");
            for n in 0..100 {
                std::fs::write(generation.join(format!("dead{n:04}")), [0; 10]).expect("a file");
            }
        }
        let options = DeviceOptions {
            pipeline_cache_dir: Some(dir.clone()),
            ..DeviceOptions::default()
        };
        let device =
            Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"));
        let counters = device.counters();
        drop(device);
        assert_eq!(
            counters.gauges(),
            Gauges::default(),
            "{name}: gauges after teardown"
        );
        assert!(
            !dir.join("g1").exists() && !dir.join("g2").exists(),
            "{name}: the dead generations removed by the teardown's end"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The consumer's extensions: the instance and every device of the cfg's kind enable each one they
/// offer, once, after the floor's (and the instance's after the portability listing, where the
/// loader offers it), and pass over the rest; with the tests' own driver, each one the loader or
/// that driver offers, then direct driver loading. `VK_EXT_swapchain_maintenance1` goes
/// with its feature structure, and only where the instance has its dependency.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_of_the_cfgs_kind_enables_the_consumers_extensions_it_offers() {
    use crate::floor;
    use crate::tests::devices_of_the_cfgs_kind;
    use ash::vk;
    use std::ffi::CStr;

    let missing: &CStr = c"VK_M2V_no_such_extension";
    let surface = [
        ash::khr::get_surface_capabilities2::NAME,
        ash::ext::surface_maintenance1::NAME,
    ];
    let instance = crate::tests::validation::instance(&[
        surface[0],
        missing,
        surface[1],
        floor::INSTANCE_EXTENSIONS[0],
        surface[0],
    ]);
    let (entry, vk_instance) = instance.raw();
    // SAFETY: the loader is loaded.
    let mut offered = unsafe { entry.enumerate_instance_extension_properties(None) }
        .expect("the instance extensions");
    // Where the tests give the loader a driver of their own, its extensions count too: the
    // loader does not ask such a driver for them.
    let direct = crate::tests::validation::direct_driver();
    if let Some(driver) = direct {
        // SAFETY: the driver's library stays loaded for the process.
        let own = unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr: driver,
            })
        };
        // SAFETY: the driver is loaded.
        offered.extend(
            unsafe { own.enumerate_instance_extension_properties(None) }
                .expect("the driver's instance extensions"),
        );
    }
    let offers = |name: &CStr| {
        offered
            .iter()
            .any(|properties| properties.extension_name_as_c_str() == Ok(name))
    };
    let mut expected: Vec<&CStr> = floor::INSTANCE_EXTENSIONS.to_vec();
    expected.extend(
        (floor::OPTIONAL_INSTANCE_EXTENSIONS.iter())
            .chain(&surface)
            .copied()
            .filter(|&name| offers(name)),
    );
    if direct.is_some() {
        expected.push(ash::lunarg::direct_driver_loading::NAME);
    }
    let enabled: Vec<&CStr> = instance.extensions().iter().map(|n| n.as_c_str()).collect();
    assert_eq!(enabled, expected, "the instance's extensions");
    let maintenance = enabled.contains(&ash::ext::surface_maintenance1::NAME);

    let devices = devices_of_the_cfgs_kind(&instance);
    assert!(
        !devices.is_empty(),
        "the loader lists no device of this cfg's kind"
    );
    for (name, physical) in &devices {
        // SAFETY: `physical` is one of `instance`'s.
        let listed = unsafe { vk_instance.enumerate_device_extension_properties(physical.raw()) }
            .expect("the device extensions");
        let lists = |name: &CStr| {
            listed
                .iter()
                .any(|properties| properties.extension_name_as_c_str() == Ok(name))
        };
        let mut answer = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut answer)
            .push_next(&mut vulkan11);
        // SAFETY: as above; `answer` and `vulkan11` live through the call.
        unsafe { vk_instance.get_physical_device_features2(physical.raw(), &mut features) };
        let ycbcr = vulkan11.sampler_ycbcr_conversion == vk::TRUE;
        let swapchain = ash::ext::swapchain_maintenance1::NAME;
        let offered = lists(swapchain) && answer.swapchain_maintenance1 == vk::TRUE;
        // The border swizzle's structure is read only from a device that lists its extension.
        let border = ash::ext::border_color_swizzle::NAME;
        let mut swizzles = vk::PhysicalDeviceBorderColorSwizzleFeaturesEXT::default();
        if lists(border) {
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut swizzles);
            // SAFETY: as above; `swizzles` lives through the call.
            unsafe { vk_instance.get_physical_device_features2(physical.raw(), &mut features) };
        }
        let from_image = swizzles.border_color_swizzle_from_image == vk::TRUE;

        let mut asked = vec![
            DeviceExtension::new(missing),
            DeviceExtension::new(floor::DEVICE_EXTENSIONS[0]),
        ];
        if maintenance {
            asked.push(DeviceExtension::with_features(
                swapchain,
                vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                    .swapchain_maintenance1(true),
                |answer| answer.swapchain_maintenance1 == vk::TRUE,
            ));
        }
        let options = DeviceOptions {
            extensions: asked,
            features: vec![
                ConsumerFeature::SamplerYcbcrConversion,
                ConsumerFeature::SamplerYcbcrConversion,
            ],
            ..DeviceOptions::default()
        };
        let device =
            Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut expected: Vec<&CStr> = floor::DEVICE_EXTENSIONS.to_vec();
        expected.extend(
            floor::OPTIONAL_EXTENSIONS
                .iter()
                .copied()
                .filter(|&name| lists(name)),
        );
        // The attachment reads' extensions, where device creation took them.
        if device.core().feedback.is_some() {
            expected.extend(floor::ATTACHMENT_READ_EXTENSIONS);
        }
        // The border swizzle's, where the device reports it from the image.
        if from_image {
            expected.extend(floor::BORDER_SWIZZLE_EXTENSIONS);
        }
        if maintenance && offered {
            expected.push(swapchain);
        }
        let enabled: Vec<&CStr> = device.extensions().iter().map(|n| n.as_c_str()).collect();
        assert_eq!(enabled, expected, "{name}: the device's extensions");
        println!("extensions: {name}: {enabled:?}");
        // The consumer's feature, asked twice, is enabled once where the device reports it.
        let granted: &[ConsumerFeature] = if ycbcr {
            &[ConsumerFeature::SamplerYcbcrConversion]
        } else {
            &[]
        };
        assert_eq!(device.features(), granted, "{name}: the device's features");
        println!("features: {name}: {:?}", device.features());
    }
}

/// The translator's profile on every device of the cfg's kind holds only what device creation
/// enabled: the floor's `robustImageAccess` and `nullDescriptor`, and none of the robustness
/// features the runtime leaves off, whatever the device reports.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_of_the_cfgs_kind_gives_the_translator_only_what_it_enables() {
    use crate::tests::devices_of_the_cfgs_kind;
    use metal2vulkan::device::VkFeature;

    let instance = crate::tests::validation::instance(&[]);
    let devices = devices_of_the_cfgs_kind(&instance);
    assert!(
        !devices.is_empty(),
        "the loader lists no device of this cfg's kind"
    );
    for (name, physical) in &devices {
        let device = Device::new(physical, DeviceOptions::default())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let profile = device.core().translator.profile();
        let enabled = &profile.enabled().features;
        for on in [
            VkFeature::RobustImageAccess,
            VkFeature::RobustImageAccess2,
            VkFeature::NullDescriptor,
        ] {
            assert!(enabled.contains(&on), "{name}: {on:?}");
        }
        for off in [
            VkFeature::RobustBufferAccess,
            VkFeature::RobustBufferAccess2,
        ] {
            assert!(!enabled.contains(&off), "{name}: {off:?}");
        }
        // An image read out of range is the device's; a texel buffer's is still the shader's.
        assert!(profile.device_reads_texels(false), "{name}: image reads");
        assert!(!profile.device_reads_texels(true), "{name}: texel buffers");
        println!(
            "profile: {name}: robustImageAccess, robustImageAccess2 and nullDescriptor alone of \
             the robustness features"
        );
    }
}
