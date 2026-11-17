//! The floor's check: each missing item is named, the optional features decide
//! `Capabilities`, memory types are chosen by flags and by what buffers can use, the README states
//! what the code checks, and, under a device cfg, every device the loader lists is above the floor.

use crate::floor::{
    choose, decide, flag_names, limits, vulkan_name, BelowFloor, Features, MemoryTypes, Missing,
    Queried, DEVICE_EXTENSIONS, INSTANCE_EXTENSIONS, MEMORY_FLAG_NAMES, OPTIONAL,
    OPTIONAL_EXTENSIONS, PRIVATE_MEMORY, REQUIRED, RING_MEMORY, SHARED_MEMORY, TIMESTAMP_BITS,
    VERSION,
};
use ash::vk;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Memory with the given types, each in heap 0.
fn memory(types: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
    let mut memory = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: types.len() as u32,
        memory_heap_count: 1,
        ..Default::default()
    };
    for (slot, &flags) in memory.memory_types.iter_mut().zip(types) {
        *slot = vk::MemoryType::default().property_flags(flags);
    }
    memory
}

const DEVICE_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

/// A device with exactly the floor: Vulkan 1.3.0, the floor's extensions, every required feature,
/// every limit at its minimum, one queue family with graphics, compute and 36-bit timestamps, the
/// two memory types the runtime allocates from after a device-local one, and no optional feature.
fn at_the_floor() -> Queried {
    let mut features = Features::default();
    features.vulkan12.buffer_device_address = vk::TRUE;
    features.core.shader_int64 = vk::TRUE;
    features.vulkan12.runtime_descriptor_array = vk::TRUE;
    features.vulkan12.descriptor_binding_partially_bound = vk::TRUE;
    features
        .vulkan12
        .descriptor_binding_update_unused_while_pending = vk::TRUE;
    features
        .vulkan12
        .descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
    features
        .vulkan12
        .descriptor_binding_storage_image_update_after_bind = vk::TRUE;
    features.core.shader_sampled_image_array_dynamic_indexing = vk::TRUE;
    features.core.shader_storage_image_array_dynamic_indexing = vk::TRUE;
    features
        .vulkan12
        .shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
    features
        .vulkan12
        .shader_storage_image_array_non_uniform_indexing = vk::TRUE;
    features.core.shader_storage_image_write_without_format = vk::TRUE;
    features.core.shader_int16 = vk::TRUE;
    features.vulkan12.shader_int8 = vk::TRUE;
    features.vulkan11.storage_buffer16_bit_access = vk::TRUE;
    features.vulkan12.storage_buffer8_bit_access = vk::TRUE;
    features.vulkan12.scalar_block_layout = vk::TRUE;
    features.vulkan12.uniform_buffer_standard_layout = vk::TRUE;
    let layout = &mut features.workgroup_layout;
    layout.workgroup_memory_explicit_layout = vk::TRUE;
    layout.workgroup_memory_explicit_layout_scalar_block_layout = vk::TRUE;
    layout.workgroup_memory_explicit_layout8_bit_access = vk::TRUE;
    layout.workgroup_memory_explicit_layout16_bit_access = vk::TRUE;
    features.vulkan12.vulkan_memory_model = vk::TRUE;
    features.vulkan12.vulkan_memory_model_device_scope = vk::TRUE;
    features.vulkan13.subgroup_size_control = vk::TRUE;
    features.vulkan13.compute_full_subgroups = vk::TRUE;
    features.vulkan13.robust_image_access = vk::TRUE;
    features.vulkan12.timeline_semaphore = vk::TRUE;
    features.vulkan13.synchronization2 = vk::TRUE;
    features.vulkan13.dynamic_rendering = vk::TRUE;

    Queried {
        version: vk::make_api_version(0, 1, 3, 0),
        limits: vk::PhysicalDeviceLimits::default()
            .max_sampler_allocation_count(4000)
            .max_color_attachments(8)
            .max_push_constants_size(128)
            .max_compute_work_group_invocations(1024)
            .max_compute_work_group_size([1024, 1024, 64])
            .timestamp_compute_and_graphics(true),
        vulkan12: vk::PhysicalDeviceVulkan12Properties::default()
            .max_per_stage_descriptor_update_after_bind_sampled_images(65_536)
            .max_descriptor_set_update_after_bind_sampled_images(65_536)
            .max_per_stage_descriptor_update_after_bind_storage_images(4096)
            .max_descriptor_set_update_after_bind_storage_images(4096)
            .max_per_stage_descriptor_update_after_bind_samplers(4000)
            .max_descriptor_set_update_after_bind_samplers(4000)
            .max_per_stage_update_after_bind_resources(65_536 + 4096 + 8)
            .max_update_after_bind_descriptors_in_all_pools(65_536 + 4096 + 4000),
        features,
        extensions: DEVICE_EXTENSIONS
            .iter()
            .map(|&name| name.to_owned())
            .collect(),
        families: vec![vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            queue_count: 1,
            timestamp_valid_bits: 36,
            ..Default::default()
        }],
        memory: memory(&[DEVICE_LOCAL, SHARED_MEMORY, RING_MEMORY]),
    }
}

/// The one item `decide` names for `device`.
fn only_missing(device: &Queried) -> Missing {
    match decide(device) {
        Ok(_) => panic!("the device is above the floor"),
        Err(BelowFloor { missing }) => {
            assert_eq!(missing.len(), 1, "{missing:?}");
            missing.into_iter().next().expect("one item")
        }
    }
}

fn feature(structure: &'static str, member: &str) -> Missing {
    Missing::Feature {
        structure,
        member: member.to_string(),
    }
}

/// Whether `features` holds `feature` at `vk::TRUE`.
fn holds(features: &Features, feature: &crate::floor::Feature) -> bool {
    let mut features = *features;
    *(feature.field)(&mut features) == vk::TRUE
}

/// The translator's description is restricted to what device creation enables: a feature or an
/// extension the device reports beyond the floor is outside the enable set, so the translator's
/// profile never relies on it, and the floor's `robustImageAccess` is in it.
#[test]
fn the_translator_sees_only_what_device_creation_enables() {
    use crate::floor::shader_enables;
    use metal2vulkan::device::{VkExtension, VkFeature};
    let mut device = at_the_floor();
    device.features.core.geometry_shader = vk::TRUE;
    device.features.vulkan12.sampler_mirror_clamp_to_edge = vk::TRUE;
    device.features.vulkan13.robust_image_access = vk::TRUE;
    device
        .extensions
        .push(ash::ext::robustness2::NAME.to_owned());
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    let enables = shader_enables(&capabilities);
    assert!(enables.features.contains(&VkFeature::RobustImageAccess));
    for reported in [
        VkFeature::GeometryShader,
        VkFeature::SamplerMirrorClampToEdge,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess2,
        VkFeature::RobustBufferAccess,
    ] {
        assert!(!enables.features.contains(&reported), "{reported:?}");
    }
    assert!(!enables.extensions.contains(&VkExtension::ExtRobustness2));
}

#[test]
fn a_device_at_the_floor_is_above_it() {
    let mut device = at_the_floor();
    // Features beyond the floor are not enabled.
    device.features.core.geometry_shader = vk::TRUE;
    device.features.vulkan11.storage_input_output16 = vk::TRUE;
    device.features.vulkan12.sampler_mirror_clamp_to_edge = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert!(!capabilities.float16);
    assert!(!capabilities.storage_read_without_format);
    assert!(!capabilities.buffer_int64_atomics);
    assert!(!capabilities.shared_int64_atomics);
    assert!(!capabilities.float64);
    assert!(!capabilities.maximal_reconvergence);
    assert!(!capabilities.subgroup_uniform_control_flow);
    assert_eq!(capabilities.extensions, DEVICE_EXTENSIONS);
    assert_eq!(capabilities.queue_family, 0);
    assert_eq!(
        choose(&capabilities.memory, u32::MAX),
        Ok(MemoryTypes {
            private: 0,
            shared: 1,
            ring: 2
        })
    );
    let enabled = &capabilities.features;
    for required in REQUIRED {
        assert!(
            holds(enabled, required),
            "{} is not enabled",
            required.member
        );
    }
    for optional in OPTIONAL {
        assert!(!holds(enabled, optional), "{} is enabled", optional.member);
    }
    assert_eq!(enabled.core.geometry_shader, vk::FALSE);
    assert_eq!(enabled.vulkan11.storage_input_output16, vk::FALSE);
    assert_eq!(enabled.vulkan12.sampler_mirror_clamp_to_edge, vk::FALSE);
    assert_eq!(enabled.vulkan13.robust_image_access, vk::TRUE);
}

#[test]
fn each_missing_feature_is_named() {
    const V11: &str = "VkPhysicalDeviceVulkan11Features";
    const V12: &str = "VkPhysicalDeviceVulkan12Features";
    const V13: &str = "VkPhysicalDeviceVulkan13Features";
    const CORE: &str = "VkPhysicalDeviceFeatures";
    const LAYOUT: &str = "VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR";
    type Clear = fn(&mut Features);
    let cases: [(Clear, Missing); 30] = [
        (
            |f| f.vulkan12.buffer_device_address = vk::FALSE,
            feature(V12, "bufferDeviceAddress"),
        ),
        (
            |f| f.core.shader_int64 = vk::FALSE,
            feature(CORE, "shaderInt64"),
        ),
        (
            |f| f.vulkan12.runtime_descriptor_array = vk::FALSE,
            feature(V12, "runtimeDescriptorArray"),
        ),
        (
            |f| f.vulkan12.descriptor_binding_partially_bound = vk::FALSE,
            feature(V12, "descriptorBindingPartiallyBound"),
        ),
        (
            |f| f.vulkan12.descriptor_binding_update_unused_while_pending = vk::FALSE,
            feature(V12, "descriptorBindingUpdateUnusedWhilePending"),
        ),
        (
            |f| {
                f.vulkan12
                    .descriptor_binding_sampled_image_update_after_bind = vk::FALSE
            },
            feature(V12, "descriptorBindingSampledImageUpdateAfterBind"),
        ),
        (
            |f| {
                f.vulkan12
                    .descriptor_binding_storage_image_update_after_bind = vk::FALSE
            },
            feature(V12, "descriptorBindingStorageImageUpdateAfterBind"),
        ),
        (
            |f| f.core.shader_sampled_image_array_dynamic_indexing = vk::FALSE,
            feature(CORE, "shaderSampledImageArrayDynamicIndexing"),
        ),
        (
            |f| f.core.shader_storage_image_array_dynamic_indexing = vk::FALSE,
            feature(CORE, "shaderStorageImageArrayDynamicIndexing"),
        ),
        (
            |f| f.vulkan12.shader_sampled_image_array_non_uniform_indexing = vk::FALSE,
            feature(V12, "shaderSampledImageArrayNonUniformIndexing"),
        ),
        (
            |f| f.vulkan12.shader_storage_image_array_non_uniform_indexing = vk::FALSE,
            feature(V12, "shaderStorageImageArrayNonUniformIndexing"),
        ),
        (
            |f| f.core.shader_storage_image_write_without_format = vk::FALSE,
            feature(CORE, "shaderStorageImageWriteWithoutFormat"),
        ),
        (
            |f| f.core.shader_int16 = vk::FALSE,
            feature(CORE, "shaderInt16"),
        ),
        (
            |f| f.vulkan12.shader_int8 = vk::FALSE,
            feature(V12, "shaderInt8"),
        ),
        (
            |f| f.vulkan11.storage_buffer16_bit_access = vk::FALSE,
            feature(V11, "storageBuffer16BitAccess"),
        ),
        (
            |f| f.vulkan12.storage_buffer8_bit_access = vk::FALSE,
            feature(V12, "storageBuffer8BitAccess"),
        ),
        (
            |f| f.vulkan12.scalar_block_layout = vk::FALSE,
            feature(V12, "scalarBlockLayout"),
        ),
        (
            |f| f.vulkan12.uniform_buffer_standard_layout = vk::FALSE,
            feature(V12, "uniformBufferStandardLayout"),
        ),
        (
            |f| f.workgroup_layout.workgroup_memory_explicit_layout = vk::FALSE,
            feature(LAYOUT, "workgroupMemoryExplicitLayout"),
        ),
        (
            |f| {
                f.workgroup_layout
                    .workgroup_memory_explicit_layout_scalar_block_layout = vk::FALSE
            },
            feature(LAYOUT, "workgroupMemoryExplicitLayoutScalarBlockLayout"),
        ),
        (
            |f| {
                f.workgroup_layout
                    .workgroup_memory_explicit_layout8_bit_access = vk::FALSE
            },
            feature(LAYOUT, "workgroupMemoryExplicitLayout8BitAccess"),
        ),
        (
            |f| {
                f.workgroup_layout
                    .workgroup_memory_explicit_layout16_bit_access = vk::FALSE
            },
            feature(LAYOUT, "workgroupMemoryExplicitLayout16BitAccess"),
        ),
        (
            |f| f.vulkan12.vulkan_memory_model = vk::FALSE,
            feature(V12, "vulkanMemoryModel"),
        ),
        (
            |f| f.vulkan12.vulkan_memory_model_device_scope = vk::FALSE,
            feature(V12, "vulkanMemoryModelDeviceScope"),
        ),
        (
            |f| f.vulkan12.timeline_semaphore = vk::FALSE,
            feature(V12, "timelineSemaphore"),
        ),
        (
            |f| f.vulkan13.subgroup_size_control = vk::FALSE,
            feature(V13, "subgroupSizeControl"),
        ),
        (
            |f| f.vulkan13.compute_full_subgroups = vk::FALSE,
            feature(V13, "computeFullSubgroups"),
        ),
        (
            |f| f.vulkan13.robust_image_access = vk::FALSE,
            feature(V13, "robustImageAccess"),
        ),
        (
            |f| f.vulkan13.synchronization2 = vk::FALSE,
            feature(V13, "synchronization2"),
        ),
        (
            |f| f.vulkan13.dynamic_rendering = vk::FALSE,
            feature(V13, "dynamicRendering"),
        ),
    ];
    assert_eq!(
        cases.len(),
        REQUIRED.len(),
        "a required feature has no case"
    );
    for (clear, expected) in cases {
        let mut device = at_the_floor();
        clear(&mut device.features);
        assert_eq!(only_missing(&device), expected);
    }
}

#[test]
fn each_low_limit_is_named() {
    type Lower = fn(&mut Queried);
    let cases: [(Lower, &str, u32); 16] = [
        (
            |d| {
                d.vulkan12
                    .max_per_stage_descriptor_update_after_bind_sampled_images = 65_535
            },
            "maxPerStageDescriptorUpdateAfterBindSampledImages",
            65_536,
        ),
        (
            |d| {
                d.vulkan12
                    .max_descriptor_set_update_after_bind_sampled_images = 65_535
            },
            "maxDescriptorSetUpdateAfterBindSampledImages",
            65_536,
        ),
        (
            |d| {
                d.vulkan12
                    .max_per_stage_descriptor_update_after_bind_storage_images = 4095
            },
            "maxPerStageDescriptorUpdateAfterBindStorageImages",
            4096,
        ),
        (
            |d| {
                d.vulkan12
                    .max_descriptor_set_update_after_bind_storage_images = 4095
            },
            "maxDescriptorSetUpdateAfterBindStorageImages",
            4096,
        ),
        (
            |d| {
                d.vulkan12
                    .max_per_stage_descriptor_update_after_bind_samplers = 3999
            },
            "maxPerStageDescriptorUpdateAfterBindSamplers",
            4000,
        ),
        (
            |d| d.vulkan12.max_descriptor_set_update_after_bind_samplers = 3999,
            "maxDescriptorSetUpdateAfterBindSamplers",
            4000,
        ),
        (
            |d| d.limits.max_sampler_allocation_count = 3999,
            "maxSamplerAllocationCount",
            4000,
        ),
        (
            |d| d.vulkan12.max_per_stage_update_after_bind_resources = 65_536 + 4096 + 7,
            "maxPerStageUpdateAfterBindResources",
            65_536 + 4096 + 8,
        ),
        (
            |d| d.vulkan12.max_update_after_bind_descriptors_in_all_pools = 65_536 + 4096 + 3999,
            "maxUpdateAfterBindDescriptorsInAllPools",
            65_536 + 4096 + 4000,
        ),
        (
            |d| d.limits.max_push_constants_size = 127,
            "maxPushConstantsSize",
            128,
        ),
        (
            |d| d.limits.max_compute_work_group_invocations = 1023,
            "maxComputeWorkGroupInvocations",
            1024,
        ),
        (
            |d| d.limits.max_compute_work_group_size[0] = 1023,
            "maxComputeWorkGroupSize[0]",
            1024,
        ),
        (
            |d| d.limits.max_compute_work_group_size[1] = 1023,
            "maxComputeWorkGroupSize[1]",
            1024,
        ),
        (
            |d| d.limits.max_compute_work_group_size[2] = 63,
            "maxComputeWorkGroupSize[2]",
            64,
        ),
        (
            |d| d.limits.timestamp_compute_and_graphics = vk::FALSE,
            "timestampComputeAndGraphics",
            1,
        ),
        (
            |d| d.families[0].timestamp_valid_bits = 35,
            "timestampValidBits",
            36,
        ),
    ];
    for (lower, name, needed) in cases {
        let mut device = at_the_floor();
        lower(&mut device);
        assert_eq!(
            only_missing(&device),
            Missing::Limit {
                name,
                device: needed - 1,
                needed,
            }
        );
    }
}

#[test]
fn each_missing_extension_is_named() {
    for &extension in DEVICE_EXTENSIONS {
        let mut device = at_the_floor();
        device
            .extensions
            .retain(|name| name.as_c_str() != extension);
        assert_eq!(only_missing(&device), Missing::Extension(extension));
    }
}

#[test]
fn a_device_without_a_graphics_and_compute_family_is_named() {
    let mut device = at_the_floor();
    device.families = [vk::QueueFlags::GRAPHICS, vk::QueueFlags::COMPUTE]
        .into_iter()
        .map(|queue_flags| vk::QueueFamilyProperties {
            queue_flags,
            queue_count: 1,
            timestamp_valid_bits: 64,
            ..Default::default()
        })
        .collect();
    let missing = only_missing(&device);
    assert_eq!(missing, Missing::QueueFamily);
    assert_eq!(
        missing.to_string(),
        "a queue family with graphics and compute"
    );
}

#[test]
fn the_first_graphics_and_compute_family_is_the_queue_family() {
    let mut device = at_the_floor();
    let family = device.families[0];
    let mut transfer = family;
    transfer.queue_flags = vk::QueueFlags::TRANSFER;
    let mut compute = family;
    compute.queue_flags = vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
    device.families = vec![transfer, compute, family, family];
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert_eq!(capabilities.queue_family, 2);
    // Only that family's timestamps count.
    device.families[2].timestamp_valid_bits = 0;
    assert_eq!(
        only_missing(&device),
        Missing::Limit {
            name: "timestampValidBits",
            device: 0,
            needed: TIMESTAMP_BITS,
        }
    );
}

#[test]
fn each_missing_memory_type_is_named() {
    let shared = Missing::MemoryType {
        flags: "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | \
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT"
            .to_string(),
    };
    let ring = Missing::MemoryType {
        flags: "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | \
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT"
            .to_string(),
    };
    let host_coherent =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let cases = [
        // Host memory that is not cached is not Shared memory.
        (vec![DEVICE_LOCAL, host_coherent, RING_MEMORY], &shared),
        // Device-local host-visible memory that is not coherent is not the ring's.
        (
            vec![
                DEVICE_LOCAL,
                SHARED_MEMORY,
                DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ],
            &ring,
        ),
    ];
    for (types, expected) in cases {
        let mut device = at_the_floor();
        device.memory = memory(&types);
        assert_eq!(&only_missing(&device), expected);
    }
    let mut device = at_the_floor();
    device.memory = memory(&[DEVICE_LOCAL]);
    let Err(below) = decide(&device) else {
        panic!("the device is above the floor");
    };
    assert_eq!(below.missing, [shared.clone(), ring.clone()]);
    // Vulkan guarantees a device-local type; a device without one is named for it too.
    device.memory = memory(&[SHARED_MEMORY]);
    let Err(below) = decide(&device) else {
        panic!("the device is above the floor");
    };
    assert_eq!(
        below.missing,
        [
            Missing::MemoryType {
                flags: "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT".to_string(),
            },
            ring,
        ]
    );
    assert_eq!(
        shared.to_string(),
        "a memory type with VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | \
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT for buffers"
    );
}

#[test]
fn memory_types_are_the_first_with_their_flags_that_buffers_can_use() {
    // Device-local, host-coherent, two cached, then two device-local host-visible types, as an AMD
    // device lists them.
    let host_coherent =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let device = memory(&[
        DEVICE_LOCAL,
        host_coherent,
        SHARED_MEMORY,
        SHARED_MEMORY,
        RING_MEMORY,
        RING_MEMORY,
    ]);
    assert_eq!(
        choose(&device, u32::MAX),
        Ok(MemoryTypes {
            private: 0,
            shared: 2,
            ring: 4
        })
    );
    assert_eq!(
        choose(&device, !((1 << 2) | (1 << 4))),
        Ok(MemoryTypes {
            private: 0,
            shared: 3,
            ring: 5
        })
    );
    // Private memory is the first device-local type buffers can use, whatever its other flags.
    assert_eq!(
        choose(&device, !(1 | (1 << 4))),
        Ok(MemoryTypes {
            private: 5,
            shared: 2,
            ring: 5
        })
    );
    // One type with every flag serves all three, as on a CPU device.
    let all = memory(&[SHARED_MEMORY | RING_MEMORY]);
    assert_eq!(
        choose(&all, 1),
        Ok(MemoryTypes {
            private: 0,
            shared: 0,
            ring: 0
        })
    );
    // Bits past the device's types choose nothing.
    assert_eq!(
        choose(&device, !0b11_1111).map_err(|missing| missing.len()),
        Err(3)
    );
}

#[test]
fn memory_flags_are_named_as_vulkan_names_them() {
    let mut named = vk::MemoryPropertyFlags::empty();
    for (flag, name) in MEMORY_FLAG_NAMES {
        assert_eq!(flag_names(flag), name);
        named |= flag;
    }
    assert!(
        named.contains(PRIVATE_MEMORY | SHARED_MEMORY | RING_MEMORY),
        "a flag the floor asks for has no name"
    );
    assert_eq!(
        flag_names(RING_MEMORY),
        "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | \
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT"
    );
}

#[test]
fn a_device_below_vulkan_1_3_is_named_by_its_version() {
    let mut device = at_the_floor();
    device.version = vk::make_api_version(0, 1, 2, 198);
    device.features = Features::default();
    let missing = only_missing(&device);
    assert_eq!(
        missing,
        Missing::Version(vk::make_api_version(0, 1, 2, 198))
    );
    assert_eq!(missing.to_string(), "Vulkan 1.2.198, below 1.3");
}

#[test]
fn every_missing_item_is_named_at_once() {
    let mut device = at_the_floor();
    device.extensions.clear();
    device.features.vulkan13.dynamic_rendering = vk::FALSE;
    device.features.core.shader_int64 = vk::FALSE;
    device.limits.max_push_constants_size = 64;
    device.families[0].queue_flags = vk::QueueFlags::COMPUTE;
    device.memory = memory(&[DEVICE_LOCAL, SHARED_MEMORY]);
    let Err(below) = decide(&device) else {
        panic!("the device is above the floor");
    };
    assert_eq!(
        below.missing,
        [
            Missing::Extension(ash::khr::swapchain::NAME),
            Missing::Extension(ash::khr::workgroup_memory_explicit_layout::NAME),
            feature("VkPhysicalDeviceFeatures", "shaderInt64"),
            feature("VkPhysicalDeviceVulkan13Features", "dynamicRendering"),
            Missing::Limit {
                name: "maxPushConstantsSize",
                device: 64,
                needed: 128,
            },
            Missing::QueueFamily,
            Missing::MemoryType {
                flags: flag_names(RING_MEMORY),
            },
        ]
    );
    assert_eq!(
        below.to_string(),
        "the device is below the runtime's floor: it lacks VK_KHR_swapchain; \
         VK_KHR_workgroup_memory_explicit_layout; VkPhysicalDeviceFeatures.shaderInt64; VkPhysicalDeviceVulkan13Features.dynamicRendering; \
         maxPushConstantsSize 64, below 128; a queue family with graphics and compute; a memory \
         type with VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | \
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT for buffers"
    );
}

/// An optional extension is enabled where the device lists it, after the floor's.
#[test]
fn an_optional_extension_is_enabled_where_listed() {
    for &extension in OPTIONAL_EXTENSIONS {
        let mut device = at_the_floor();
        device.extensions.push(extension.to_owned());
        let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
        let mut expected = DEVICE_EXTENSIONS.to_vec();
        expected.push(extension);
        assert_eq!(capabilities.extensions, expected);
    }
}

#[test]
fn the_optional_features_decide_the_capabilities() {
    type Set = fn(&mut Features);
    let cases: [(Set, &str); 7] = [
        (|f| f.vulkan12.shader_float16 = vk::TRUE, "float16"),
        (
            |f| f.core.shader_storage_image_read_without_format = vk::TRUE,
            "storage_read_without_format",
        ),
        (
            |f| f.vulkan12.shader_buffer_int64_atomics = vk::TRUE,
            "buffer_int64_atomics",
        ),
        (
            |f| f.vulkan12.shader_shared_int64_atomics = vk::TRUE,
            "shared_int64_atomics",
        ),
        (|f| f.core.shader_float64 = vk::TRUE, "float64"),
        (
            |f| f.maximal_reconvergence.shader_maximal_reconvergence = vk::TRUE,
            "maximal_reconvergence",
        ),
        (
            |f| {
                f.subgroup_uniform_control_flow
                    .shader_subgroup_uniform_control_flow = vk::TRUE
            },
            "subgroup_uniform_control_flow",
        ),
    ];
    assert_eq!(
        cases.len(),
        OPTIONAL.len(),
        "an optional feature has no case"
    );
    for (index, (set, answer)) in cases.into_iter().enumerate() {
        let mut device = at_the_floor();
        set(&mut device.features);
        let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
        let answers = [
            ("float16", capabilities.float16),
            (
                "storage_read_without_format",
                capabilities.storage_read_without_format,
            ),
            ("buffer_int64_atomics", capabilities.buffer_int64_atomics),
            ("shared_int64_atomics", capabilities.shared_int64_atomics),
            ("float64", capabilities.float64),
            ("maximal_reconvergence", capabilities.maximal_reconvergence),
            (
                "subgroup_uniform_control_flow",
                capabilities.subgroup_uniform_control_flow,
            ),
        ];
        for (name, value) in answers {
            assert_eq!(
                value,
                name == answer,
                "{name} when only {answer} is offered"
            );
        }
        for (other, optional) in OPTIONAL.iter().enumerate() {
            assert_eq!(
                holds(&capabilities.features, optional),
                other == index,
                "{} enabled when only {answer} is offered",
                optional.member
            );
        }
    }
}

/// A table row's cells, each trimmed and stripped of code quotes.
fn cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().trim_matches('`').to_string())
        .collect()
}

/// The data rows of the section's table whose header row is `header`.
fn table(section: &[&str], header: &[&str]) -> Vec<Vec<String>> {
    let start = section
        .iter()
        .position(|&line| line.starts_with('|') && cells(line) == header)
        .unwrap_or_else(|| panic!("the Floor section has no table headed {header:?}"));
    section[start + 2..]
        .iter()
        .take_while(|line| line.starts_with('|'))
        .map(|&line| cells(line))
        .collect()
}

#[test]
fn the_readme_states_the_floor() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let section: Vec<&str> = readme
        .lines()
        .skip_while(|line| *line != "## Floor")
        .skip(1)
        .take_while(|line| !line.starts_with("## "))
        .collect();
    assert!(!section.is_empty(), "the README has no `## Floor` section");
    let sentence = format!(
        "The runtime requires Vulkan {}.{}",
        vk::api_version_major(VERSION),
        vk::api_version_minor(VERSION)
    );
    assert!(
        section.iter().any(|line| line.contains(&sentence)),
        "the Floor section must say `{sentence}`"
    );

    let stated: BTreeSet<(String, String)> = table(&section, &["Extension", "Where", "Used for"])
        .into_iter()
        .map(|row| (row[0].clone(), row[1].clone()))
        .collect();
    let code: BTreeSet<(String, String)> = INSTANCE_EXTENSIONS
        .iter()
        .map(|name| (name, "instance"))
        .chain(DEVICE_EXTENSIONS.iter().map(|name| (name, "device")))
        .chain(
            OPTIONAL_EXTENSIONS
                .iter()
                .map(|name| (name, "device, where listed")),
        )
        .map(|(name, place)| (name.to_string_lossy().into_owned(), place.to_string()))
        .collect();
    assert_eq!(stated, code, "the README's extensions");

    for (header, features) in [
        (["Feature", "Structure", "Used for"], REQUIRED),
        (["Optional feature", "Structure", "Used for"], OPTIONAL),
    ] {
        let stated: BTreeSet<(String, String)> = table(&section, &header)
            .into_iter()
            .map(|row| (row[1].clone(), row[0].clone()))
            .collect();
        let code: BTreeSet<(String, String)> = features
            .iter()
            .map(|feature| (feature.structure.to_string(), vulkan_name(feature.member)))
            .collect();
        assert_eq!(code.len(), features.len(), "a feature listed twice");
        assert_eq!(stated, code, "the README's {}s", header[0].to_lowercase());
    }

    // The per-stage resource limit's README minimum leaves out the colour attachments.
    let mut device = at_the_floor();
    device.limits.max_color_attachments = 0;
    let stated: BTreeSet<(String, u32)> = table(&section, &["Limit", "Minimum", "Used for"])
        .into_iter()
        .map(|row| {
            let minimum = row[1]
                .parse()
                .unwrap_or_else(|error| panic!("the minimum of {}: {error}", row[0]));
            (row[0].clone(), minimum)
        })
        .collect();
    let code: BTreeSet<(String, u32)> = limits(&device)
        .into_iter()
        .map(|(name, _, needed)| (name.to_string(), needed))
        .chain([("timestampValidBits".to_string(), TIMESTAMP_BITS)])
        .collect();
    assert_eq!(stated, code, "the README's limits");

    let stated: BTreeSet<String> = table(&section, &["Memory type", "Flags", "Used for"])
        .into_iter()
        .map(|row| {
            let flags: Vec<&str> = row[1]
                .split(", ")
                .map(|flag| flag.trim_matches('`'))
                .collect();
            flags.join(" | ")
        })
        .collect();
    let code: BTreeSet<String> = [PRIVATE_MEMORY, SHARED_MEMORY, RING_MEMORY]
        .into_iter()
        .map(flag_names)
        .collect();
    assert_eq!(stated, code, "the README's memory types");
}

#[test]
fn vulkan_names_follow_the_registry() {
    assert_eq!(vulkan_name("buffer_device_address"), "bufferDeviceAddress");
    assert_eq!(vulkan_name("shader_int64"), "shaderInt64");
    assert_eq!(vulkan_name("synchronization2"), "synchronization2");
}

/// Every device of the cfg's kind the machine's Vulkan loader lists is above the floor: CPU devices
/// under `m2v_cpu_icd`, the others under `m2v_gpu`. Each is created with the features and
/// extensions the floor returns, and its memory types are chosen on it. The device legs pick one
/// device per run with `VK_DRIVER_FILES` and, for Mesa, `MESA_VK_DEVICE_SELECT`.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_the_loader_lists_is_above_the_floor() {
    use crate::floor::{check, memory_types, Capabilities};
    use std::ffi::c_char;

    /// Creates a device from `capabilities` as the runtime does, and chooses its memory types.
    ///
    /// # Safety
    ///
    /// `physical` is one of `instance`'s, and `capabilities` is its floor check's answer.
    unsafe fn memory_on_a_device(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        capabilities: &Capabilities,
    ) -> Result<MemoryTypes, String> {
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(capabilities.queue_family)
            .queue_priorities(&priorities)];
        let Features {
            core,
            mut vulkan11,
            mut vulkan12,
            mut vulkan13,
            mut workgroup_layout,
            mut maximal_reconvergence,
            mut subgroup_uniform_control_flow,
        } = capabilities.features;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut workgroup_layout);
        let enabled = |name: &std::ffi::CStr| capabilities.extensions.contains(&name);
        if enabled(ash::khr::shader_maximal_reconvergence::NAME) {
            features = features.push_next(&mut maximal_reconvergence);
        }
        if enabled(ash::khr::shader_subgroup_uniform_control_flow::NAME) {
            features = features.push_next(&mut subgroup_uniform_control_flow);
        }
        let extensions: Vec<*const c_char> = capabilities
            .extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features);
        // SAFETY: the caller's contract; `info` and what it points to live through the call.
        let device = unsafe { instance.create_device(physical, &info, None) }
            .map_err(|error| format!("creating the device with the floor's features: {error}"))?;
        // SAFETY: `device` was created from `physical` with `capabilities.features`.
        let memory = unsafe { memory_types(&device, capabilities) };
        // SAFETY: nothing made from `device` is alive.
        unsafe { device.destroy_device(None) };
        memory.map_err(|below| below.to_string())
    }

    // SAFETY: loading the Vulkan loader runs its initialisation, which the test process allows.
    let entry = unsafe { ash::Entry::load() }.expect("the Vulkan loader");
    let extensions: Vec<*const c_char> = INSTANCE_EXTENSIONS
        .iter()
        .map(|name| name.as_ptr())
        .collect();
    let application = vk::ApplicationInfo::default().api_version(VERSION);
    let info = vk::InstanceCreateInfo::default()
        .application_info(&application)
        .enabled_extension_names(&extensions);
    // SAFETY: `info` and what it points to live through the call.
    let instance = unsafe { entry.create_instance(&info, None) }
        .expect("an instance with the floor's version and instance extensions");
    // SAFETY: `instance` is live.
    let physicals = unsafe { instance.enumerate_physical_devices() }.expect("the physical devices");
    let mut checked = 0;
    let mut below = Vec::new();
    for &physical in &physicals {
        let mut driver = vk::PhysicalDeviceVulkan12Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: `physical` is one of `instance`'s, which is a Vulkan 1.3 instance.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let cpu = properties.properties.device_type == vk::PhysicalDeviceType::CPU;
        if cpu != cfg!(m2v_cpu_icd) {
            continue;
        }
        checked += 1;
        let api = properties.properties.api_version;
        let name = properties
            .properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let info = driver
            .driver_info_as_c_str()
            .map(|info| info.to_string_lossy().into_owned())
            .unwrap_or_default();
        let version = format!(
            "Vulkan {}.{}.{}",
            vk::api_version_major(api),
            vk::api_version_minor(api),
            vk::api_version_patch(api)
        );
        // SAFETY: as above.
        let capabilities = match unsafe { check(&instance, physical) } {
            Ok(capabilities) => capabilities,
            Err(refusal) => {
                below.push(format!("{name} ({version}, {info}): {refusal}"));
                continue;
            }
        };
        // SAFETY: as above, and `capabilities` is `physical`'s.
        let memory = match unsafe { memory_on_a_device(&instance, physical, &capabilities) } {
            Ok(memory) => memory,
            Err(refusal) => {
                below.push(format!("{name} ({version}, {info}): {refusal}"));
                continue;
            }
        };
        let flags = |index: u32| {
            capabilities.memory.memory_types_as_slice()[index as usize]
                .property_flags
                .as_raw()
        };
        println!(
            "floor: {name} ({version}, {info}) is above the floor; float16 {}, storage reads \
             without format {}, 64-bit atomics on buffers {} and threadgroup memory {}, float64 \
             {}; queue family {}; Private memory type {} (flags {:#x}), Shared memory type {} \
             (flags {:#x}), ring memory type {} (flags {:#x})",
            capabilities.float16,
            capabilities.storage_read_without_format,
            capabilities.buffer_int64_atomics,
            capabilities.shared_int64_atomics,
            capabilities.float64,
            capabilities.queue_family,
            memory.private,
            flags(memory.private),
            memory.shared,
            flags(memory.shared),
            memory.ring,
            flags(memory.ring)
        );
    }
    // SAFETY: nothing made from `instance` is alive.
    unsafe { instance.destroy_instance(None) };
    assert!(checked > 0, "the loader lists no device of this cfg's kind");
    assert!(below.is_empty(), "below the floor: {below:#?}");
}
