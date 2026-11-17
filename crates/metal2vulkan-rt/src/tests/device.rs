//! The instance, the device and its counters: which queue family and queues a device takes, that
//! the handles cross threads, and, under a device cfg, that every device of the cfg's kind opens,
//! presents on the queue its family offers, and is torn down with its gauges at zero.

use crate::device::queue_plan;
use crate::{Counters, Device, Gauges, Instance, Metrics, PhysicalDevice};
use ash::vk;

fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
    vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        ..Default::default()
    }
}

#[test]
fn the_runtime_takes_the_first_family_that_does_graphics_and_compute() {
    let g = vk::QueueFlags::GRAPHICS;
    let c = vk::QueueFlags::COMPUTE;
    let t = vk::QueueFlags::TRANSFER;
    let cases = [
        // A family with many queues gives the runtime one and presenting one.
        (vec![family(g | c | t, 16)], Some((0, 2))),
        // A family with one queue is shared by the runtime and presenting.
        (vec![family(g | c | t, 1)], Some((0, 1))),
        // Compute-only and graphics-only families are passed over.
        (
            vec![family(c | t, 2), family(g, 1), family(g | c, 4)],
            Some((2, 2)),
        ),
        // The first family that does both wins over a later one with more queues.
        (vec![family(g | c, 1), family(g | c | t, 8)], Some((0, 1))),
        // A family that reports no queue is passed over.
        (vec![family(g | c, 0), family(g | c, 3)], Some((1, 2))),
        // No family does both.
        (vec![family(c | t, 4), family(g | t, 1)], None),
        (vec![], None),
    ];
    for (families, expected) in cases {
        let shown: Vec<(u32, u32)> = families
            .iter()
            .map(|family| (family.queue_flags.as_raw(), family.queue_count))
            .collect();
        assert_eq!(
            queue_plan(&families),
            expected,
            "flags and counts {shown:x?}"
        );
    }
}

#[test]
fn the_handles_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Instance>();
    send_sync::<PhysicalDevice>();
    send_sync::<Device>();
    send_sync::<Counters>();
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
        let device = Device::new(physical).unwrap_or_else(|error| panic!("{name}: {error}"));
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
    let device = Device::new(&physical).expect("a device after the instance's handle dropped");
    drop(physical);
    drop(device);
}
