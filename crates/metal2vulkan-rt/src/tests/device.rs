//! The instance, the device and its counters: that the handles cross threads, and, under a device
//! cfg, that every device of the cfg's kind opens, presents on the queue its family offers, is torn
//! down with its gauges at zero, and enables the consumer's extensions it offers.

use crate::{
    BlockId, BlockInfo, BlockMappings, Buffer, Counters, Device, DeviceExtension, DeviceOptions,
    Gauges, Instance, Metrics, PhysicalDevice,
};

#[test]
fn the_handles_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Instance>();
    send_sync::<PhysicalDevice>();
    send_sync::<Device>();
    send_sync::<DeviceOptions>();
    send_sync::<DeviceExtension>();
    send_sync::<Counters>();
    send_sync::<BlockId>();
    send_sync::<BlockInfo>();
    send_sync::<Buffer>();
    send_sync::<std::sync::Arc<dyn BlockMappings>>();
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

    let instance = Instance::new(&[]).expect("an instance at the floor");
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
        ] {
            println!("M2V-MATRIX gauge device_teardown/{kind} {value}");
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
}

/// The consumer's extensions: the instance and every device of the cfg's kind enable each one they
/// offer, once, after the floor's, and pass over the rest. `VK_EXT_swapchain_maintenance1` goes
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
    let instance = Instance::new(&[
        surface[0],
        missing,
        surface[1],
        floor::INSTANCE_EXTENSIONS[0],
        surface[0],
    ])
    .expect("an instance at the floor");
    let (entry, vk_instance) = instance.raw();
    // SAFETY: the loader is loaded.
    let offered = unsafe { entry.enumerate_instance_extension_properties(None) }
        .expect("the instance extensions");
    let offers = |name: &CStr| {
        offered
            .iter()
            .any(|properties| properties.extension_name_as_c_str() == Ok(name))
    };
    let mut expected: Vec<&CStr> = floor::INSTANCE_EXTENSIONS.to_vec();
    expected.extend(surface.iter().copied().filter(|&name| offers(name)));
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
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut answer);
        // SAFETY: as above; `answer` lives through the call.
        unsafe { vk_instance.get_physical_device_features2(physical.raw(), &mut features) };
        let swapchain = ash::ext::swapchain_maintenance1::NAME;
        let offered = lists(swapchain) && answer.swapchain_maintenance1 == vk::TRUE;

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
            ..DeviceOptions::default()
        };
        let device =
            Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut expected: Vec<&CStr> = floor::DEVICE_EXTENSIONS.to_vec();
        if maintenance && offered {
            expected.push(swapchain);
        }
        let enabled: Vec<&CStr> = device.extensions().iter().map(|n| n.as_c_str()).collect();
        assert_eq!(enabled, expected, "{name}: the device's extensions");
        println!("extensions: {name}: {enabled:?}");
    }
}
