//! The floor's check: each missing item is named, the optional features decide
//! `Capabilities`, memory types are chosen by flags and by what buffers can use, the README states
//! what the code checks, and, under a device cfg, every device the loader lists is above the floor.

use crate::floor::{
    choose, decide, flag_names, limits, vulkan_name, BelowFloor, Features, MemoryTypes, Missing,
    Queried, ATTACHMENT_READS, ATTACHMENT_READ_EXTENSIONS, DEVICE_EXTENSIONS, INSTANCE_EXTENSIONS,
    MEMORY_FLAG_NAMES, OPTIONAL, OPTIONAL_EXTENSIONS, OPTIONAL_INSTANCE_EXTENSIONS, PORTABILITY,
    PRIVATE_MEMORY, REQUIRED, RING_MEMORY, SHARED_MEMORY, STEP_RATES, TIMESTAMP_BITS, TIME_DOMAINS,
    VERSION,
};
use ash::vk;
use std::collections::BTreeSet;
use std::fs;

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
    features.core.sampler_anisotropy = vk::TRUE;
    features.core.shader_int16 = vk::TRUE;
    features.vulkan12.shader_int8 = vk::TRUE;
    features.vulkan11.storage_buffer16_bit_access = vk::TRUE;
    features.vulkan12.storage_buffer8_bit_access = vk::TRUE;
    features.vulkan12.scalar_block_layout = vk::TRUE;
    features.vulkan12.uniform_buffer_standard_layout = vk::TRUE;
    features.vulkan12.vulkan_memory_model = vk::TRUE;
    features.vulkan12.vulkan_memory_model_device_scope = vk::TRUE;
    features.vulkan13.subgroup_size_control = vk::TRUE;
    features.vulkan13.compute_full_subgroups = vk::TRUE;
    features.vulkan13.robust_image_access = vk::TRUE;
    features.robustness2.robust_image_access2 = vk::TRUE;
    features.core.independent_blend = vk::TRUE;
    features.core.fragment_stores_and_atomics = vk::TRUE;
    features.core.tessellation_shader = vk::TRUE;
    features.vulkan11.multiview = vk::TRUE;
    features.vulkan11.shader_draw_parameters = vk::TRUE;
    features.vulkan12.shader_subgroup_extended_types = vk::TRUE;
    features.vulkan13.shader_demote_to_helper_invocation = vk::TRUE;
    features.vulkan13.shader_terminate_invocation = vk::TRUE;
    features.vulkan13.shader_zero_initialize_workgroup_memory = vk::TRUE;
    features.vulkan12.timeline_semaphore = vk::TRUE;
    features.vulkan12.host_query_reset = vk::TRUE;
    features.vulkan13.synchronization2 = vk::TRUE;
    features.vulkan13.dynamic_rendering = vk::TRUE;
    features.robustness2.null_descriptor = vk::TRUE;

    Queried {
        version: vk::make_api_version(0, 1, 3, 0),
        limits: vk::PhysicalDeviceLimits::default()
            .max_sampler_allocation_count(4000)
            .max_color_attachments(8)
            .max_push_constants_size(256)
            .max_compute_work_group_invocations(1024)
            .max_compute_work_group_size([1024, 1024, 64])
            .timestamp_compute_and_graphics(true)
            .standard_sample_locations(true),
        vulkan12: vk::PhysicalDeviceVulkan12Properties::default()
            .max_per_stage_descriptor_update_after_bind_sampled_images(65_536)
            .max_descriptor_set_update_after_bind_sampled_images(65_536)
            .max_per_stage_descriptor_update_after_bind_storage_images(65_536)
            .max_descriptor_set_update_after_bind_storage_images(65_536)
            .max_per_stage_descriptor_update_after_bind_samplers(3 * 4000)
            .max_descriptor_set_update_after_bind_samplers(3 * 4000)
            .max_per_stage_update_after_bind_resources(2 * 65_536 + 8)
            .max_update_after_bind_descriptors_in_all_pools(2 * 65_536 + 3 * 4000),
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
        time_domains: TIME_DOMAINS.to_vec(),
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

/// The instance enables the floor's extensions; an optional one or the consumer's where the loader
/// or one of the consumer's drivers lists it, once each; and direct driver loading only where the
/// consumer gives drivers, which a loader that does not offer it refuses by type.
#[test]
fn an_instance_enables_what_the_loader_or_the_drivers_offer_and_direct_loading_with_drivers_alone()
{
    use crate::instance::enabled;
    use crate::Error;
    use std::ffi::{CStr, CString};
    let (surface, metal) = (ash::khr::surface::NAME, ash::ext::metal_surface::NAME);
    let (direct, portability) = (
        ash::lunarg::direct_driver_loading::NAME,
        ash::khr::portability_enumeration::NAME,
    );
    let owned = |names: &[&CStr]| {
        names
            .iter()
            .map(|&name| name.to_owned())
            .collect::<Vec<CString>>()
    };
    let (loader, drivers) = (owned(&[direct, portability]), owned(&[surface, metal]));
    let names = |enabled: Result<Vec<CString>, Error>| enabled.expect("the extensions");
    // No driver: the consumer's extension the loader does not list is left out, and direct
    // loading too, though the loader offers it.
    assert_eq!(
        names(enabled(&loader, &[], &[metal], false)),
        owned(&[surface, portability])
    );
    // Drivers: what they list counts, each name once, direct loading among them.
    assert_eq!(
        names(enabled(&loader, &drivers, &[metal, metal, direct], true)),
        owned(&[surface, portability, metal, direct])
    );
    assert_eq!(
        names(enabled(&loader, &drivers, &[], true)),
        owned(&[surface, portability, direct])
    );
    assert!(matches!(
        enabled(&owned(&[portability]), &drivers, &[], true),
        Err(Error::NoDirectDriverLoading)
    ));
}

/// The consumer's loader is the library at its path, alone: a path that holds none is refused by
/// type, with no search of the platform's standard names after it, which would find the
/// machine's loader where it has one.
#[test]
fn a_consumer_s_loader_is_its_path_alone() {
    use crate::{Error, Instance};
    let missing = std::env::temp_dir().join("m2v-no-vulkan-loader/libvulkan");
    assert!(
        !missing.exists(),
        "{}: a library at the test's path",
        missing.display()
    );
    // SAFETY: no library is at the path, so nothing loads.
    match unsafe { Instance::with_drivers(Some(&missing), &[], &[]) } {
        Err(Error::Loader(_)) => {}
        Err(error) => panic!("a path with no loader: {error}"),
        Ok(_) => panic!("an instance from a path with no loader: a standard name was searched"),
    }
}

/// The translator's description is restricted to what device creation enables: a feature or an
/// extension the device reports beyond the floor is outside the enable set, so the translator's
/// profile never relies on it, and the floor's `robustImageAccess`, `robustImageAccess2`,
/// `independentBlend`, `fragmentStoresAndAtomics` and `nullDescriptor` are in it.
/// `VK_EXT_robustness2`'s robust buffer access stays off where the device reports it.
#[test]
fn the_translator_sees_only_what_device_creation_enables() {
    use crate::floor::shader_enables;
    use metal2vulkan::device::{VkExtension, VkFeature};
    let mut device = at_the_floor();
    device.features.core.geometry_shader = vk::TRUE;
    device.features.vulkan12.sampler_mirror_clamp_to_edge = vk::TRUE;
    device.features.vulkan13.robust_image_access = vk::TRUE;
    device.features.robustness2.robust_buffer_access2 = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert_eq!(
        capabilities.features.robustness2.robust_image_access2,
        vk::TRUE
    );
    assert_eq!(
        capabilities.features.robustness2.robust_buffer_access2,
        vk::FALSE
    );
    let enables = shader_enables(&capabilities);
    assert!(enables.features.contains(&VkFeature::RobustImageAccess));
    assert!(enables.features.contains(&VkFeature::RobustImageAccess2));
    assert!(enables.features.contains(&VkFeature::IndependentBlend));
    assert!(enables
        .features
        .contains(&VkFeature::FragmentStoresAndAtomics));
    assert!(enables.features.contains(&VkFeature::TessellationShader));
    assert!(enables.features.contains(&VkFeature::Multiview));
    assert!(enables
        .features
        .contains(&VkFeature::ShaderSubgroupExtendedTypes));
    assert!(enables
        .features
        .contains(&VkFeature::ShaderDemoteToHelperInvocation));
    assert!(enables
        .features
        .contains(&VkFeature::ShaderTerminateInvocation));
    assert!(enables
        .features
        .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory));
    assert!(enables.features.contains(&VkFeature::NullDescriptor));
    for reported in [
        VkFeature::GeometryShader,
        VkFeature::SamplerMirrorClampToEdge,
        VkFeature::RobustBufferAccess2,
        VkFeature::RobustBufferAccess,
    ] {
        assert!(!enables.features.contains(&reported), "{reported:?}");
    }
    // `nullDescriptor` comes with the extension, which device creation enables.
    assert!(enables.extensions.contains(&VkExtension::ExtRobustness2));
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
    for optional in OPTIONAL.iter().chain(STEP_RATES) {
        assert!(!holds(enabled, optional), "{} is enabled", optional.member);
    }
    assert_eq!(enabled.core.geometry_shader, vk::FALSE);
    assert_eq!(enabled.vulkan11.storage_input_output16, vk::FALSE);
    assert_eq!(enabled.vulkan12.sampler_mirror_clamp_to_edge, vk::FALSE);
    assert_eq!(enabled.vulkan13.robust_image_access, vk::TRUE);
    assert_eq!(enabled.core.independent_blend, vk::TRUE);
    assert_eq!(enabled.core.fragment_stores_and_atomics, vk::TRUE);
    assert_eq!(enabled.core.tessellation_shader, vk::TRUE);
    assert_eq!(enabled.vulkan11.multiview, vk::TRUE);
    assert_eq!(enabled.vulkan11.shader_draw_parameters, vk::TRUE);
    assert_eq!(enabled.vulkan12.shader_subgroup_extended_types, vk::TRUE);
    assert_eq!(
        enabled.vulkan13.shader_demote_to_helper_invocation,
        vk::TRUE
    );
    assert_eq!(enabled.vulkan13.shader_terminate_invocation, vk::TRUE);
    assert_eq!(
        enabled.vulkan13.shader_zero_initialize_workgroup_memory,
        vk::TRUE
    );
}

#[test]
fn each_missing_feature_is_named() {
    const V11: &str = "VkPhysicalDeviceVulkan11Features";
    const V12: &str = "VkPhysicalDeviceVulkan12Features";
    const V13: &str = "VkPhysicalDeviceVulkan13Features";
    const CORE: &str = "VkPhysicalDeviceFeatures";
    const ROBUSTNESS2: &str = "VkPhysicalDeviceRobustness2FeaturesEXT";
    type Clear = fn(&mut Features);
    let cases: [(Clear, Missing); 39] = [
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
            |f| f.vulkan12.host_query_reset = vk::FALSE,
            feature(V12, "hostQueryReset"),
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
            |f| f.core.independent_blend = vk::FALSE,
            feature(CORE, "independentBlend"),
        ),
        (
            |f| f.core.fragment_stores_and_atomics = vk::FALSE,
            feature(CORE, "fragmentStoresAndAtomics"),
        ),
        (
            |f| f.core.tessellation_shader = vk::FALSE,
            feature(CORE, "tessellationShader"),
        ),
        (
            |f| f.vulkan11.multiview = vk::FALSE,
            feature(V11, "multiview"),
        ),
        (
            |f| f.vulkan11.shader_draw_parameters = vk::FALSE,
            feature(V11, "shaderDrawParameters"),
        ),
        (
            |f| f.vulkan12.shader_subgroup_extended_types = vk::FALSE,
            feature(V12, "shaderSubgroupExtendedTypes"),
        ),
        (
            |f| f.vulkan13.shader_demote_to_helper_invocation = vk::FALSE,
            feature(V13, "shaderDemoteToHelperInvocation"),
        ),
        (
            |f| f.vulkan13.shader_terminate_invocation = vk::FALSE,
            feature(V13, "shaderTerminateInvocation"),
        ),
        (
            |f| f.vulkan13.shader_zero_initialize_workgroup_memory = vk::FALSE,
            feature(V13, "shaderZeroInitializeWorkgroupMemory"),
        ),
        (
            |f| f.vulkan13.synchronization2 = vk::FALSE,
            feature(V13, "synchronization2"),
        ),
        (
            |f| f.vulkan13.dynamic_rendering = vk::FALSE,
            feature(V13, "dynamicRendering"),
        ),
        (
            |f| f.robustness2.null_descriptor = vk::FALSE,
            feature(ROBUSTNESS2, "nullDescriptor"),
        ),
        (
            |f| f.robustness2.robust_image_access2 = vk::FALSE,
            feature(ROBUSTNESS2, "robustImageAccess2"),
        ),
        (
            |f| f.core.sampler_anisotropy = vk::FALSE,
            feature(CORE, "samplerAnisotropy"),
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
    let cases: [(Lower, &str, u32); 18] = [
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
                    .max_per_stage_descriptor_update_after_bind_storage_images = 65_535
            },
            "maxPerStageDescriptorUpdateAfterBindStorageImages",
            65_536,
        ),
        (
            |d| {
                d.vulkan12
                    .max_descriptor_set_update_after_bind_storage_images = 65_535
            },
            "maxDescriptorSetUpdateAfterBindStorageImages",
            65_536,
        ),
        (
            |d| {
                d.vulkan12
                    .max_per_stage_descriptor_update_after_bind_samplers = 11_999
            },
            "maxPerStageDescriptorUpdateAfterBindSamplers",
            12_000,
        ),
        (
            |d| d.vulkan12.max_descriptor_set_update_after_bind_samplers = 11_999,
            "maxDescriptorSetUpdateAfterBindSamplers",
            12_000,
        ),
        (
            |d| d.limits.max_sampler_allocation_count = 3999,
            "maxSamplerAllocationCount",
            4000,
        ),
        (
            |d| d.vulkan12.max_per_stage_update_after_bind_resources = 2 * 65_536 + 7,
            "maxPerStageUpdateAfterBindResources",
            2 * 65_536 + 8,
        ),
        (
            |d| d.vulkan12.max_update_after_bind_descriptors_in_all_pools = 2 * 65_536 + 11_999,
            "maxUpdateAfterBindDescriptorsInAllPools",
            2 * 65_536 + 12_000,
        ),
        (
            |d| d.limits.max_push_constants_size = 255,
            "maxPushConstantsSize",
            256,
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
            |d| d.limits.standard_sample_locations = vk::FALSE,
            "standardSampleLocations",
            1,
        ),
        (
            |d| d.families[0].timestamp_valid_bits = 35,
            "timestampValidBits",
            36,
        ),
        (
            |d| d.limits.max_color_attachments = 7,
            "maxColorAttachments",
            8,
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
            Missing::Extension(ash::ext::robustness2::NAME),
            Missing::Extension(ash::ext::calibrated_timestamps::NAME),
            feature("VkPhysicalDeviceFeatures", "shaderInt64"),
            feature("VkPhysicalDeviceVulkan13Features", "dynamicRendering"),
            Missing::Limit {
                name: "maxPushConstantsSize",
                device: 64,
                needed: 256,
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
         VK_EXT_robustness2; VK_EXT_calibrated_timestamps; VkPhysicalDeviceFeatures.shaderInt64; VkPhysicalDeviceVulkan13Features.dynamicRendering; \
         maxPushConstantsSize 64, below 256; a queue family with graphics and compute; a memory \
         type with VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | \
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT for buffers"
    );
}

/// An optional extension is enabled where the device lists it, after the floor's; the portability
/// subset with the portability features the floor asks of a device that lists it.
#[test]
fn an_optional_extension_is_enabled_where_listed() {
    for &extension in OPTIONAL_EXTENSIONS {
        let mut device = at_the_floor();
        device.extensions.push(extension.to_owned());
        if extension == ash::khr::portability_subset::NAME {
            for feature in PORTABILITY {
                *(feature.field)(&mut device.features) = vk::TRUE;
            }
        }
        let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
        let mut expected = DEVICE_EXTENSIONS.to_vec();
        expected.push(extension);
        assert_eq!(capabilities.extensions, expected);
    }
}

/// A device that lists the portability subset is asked for the portability features, each missing
/// one named, and is given exactly those; a device that does not list it is not asked.
#[test]
fn a_device_that_lists_the_portability_subset_is_asked_for_its_features() {
    const PORTABILITY_FEATURES: &str = "VkPhysicalDevicePortabilitySubsetFeaturesKHR";
    let subset = ash::khr::portability_subset::NAME;
    let unlisted = decide(&at_the_floor()).unwrap_or_else(|below| panic!("{below}"));
    assert!(!unlisted.extensions.contains(&subset));
    assert!(PORTABILITY
        .iter()
        .all(|feature| !holds(&unlisted.features, feature)));

    let mut device = at_the_floor();
    device.extensions.push(subset.to_owned());
    let below = decide(&device).err().expect("below the floor");
    assert_eq!(
        below.missing,
        [
            feature(PORTABILITY_FEATURES, "imageViewFormatSwizzle"),
            feature(PORTABILITY_FEATURES, "mutableComparisonSamplers"),
            feature(PORTABILITY_FEATURES, "multisampleArrayImage"),
        ]
    );

    let mut device = at_the_floor();
    device.extensions.push(subset.to_owned());
    // Every member the device reports: only the floor's are enabled.
    let reported = &mut device.features.portability;
    reported.constant_alpha_color_blend_factors = vk::TRUE;
    reported.events = vk::TRUE;
    reported.image_view_format_swizzle = vk::TRUE;
    reported.mutable_comparison_samplers = vk::TRUE;
    reported.multisample_array_image = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert!(capabilities.extensions.contains(&subset));
    let enabled = capabilities.features.portability;
    assert_eq!(
        (
            enabled.image_view_format_swizzle,
            enabled.mutable_comparison_samplers,
            enabled.multisample_array_image,
            enabled.constant_alpha_color_blend_factors,
            enabled.events,
        ),
        (vk::TRUE, vk::TRUE, vk::TRUE, vk::FALSE, vk::FALSE)
    );
}

/// The translator is told what device creation enables of the portability subset: the extension,
/// which its description of a device that lists it must enable, and the portability features the
/// floor asks for, but no other the device reports. A device that does not list it gives neither.
#[test]
fn the_translator_sees_the_portability_subset_device_creation_enables() {
    use crate::floor::shader_enables;
    use metal2vulkan::device::{VkExtension, VkFeature};
    let mut device = at_the_floor();
    device
        .extensions
        .push(ash::khr::portability_subset::NAME.to_owned());
    for feature in PORTABILITY {
        *(feature.field)(&mut device.features) = vk::TRUE;
    }
    device
        .features
        .portability
        .constant_alpha_color_blend_factors = vk::TRUE;
    let enables = shader_enables(&decide(&device).unwrap_or_else(|below| panic!("{below}")));
    assert!(enables
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset));
    for feature in [
        VkFeature::ImageViewFormatSwizzle,
        VkFeature::MutableComparisonSamplers,
        VkFeature::MultisampleArrayImage,
    ] {
        assert!(enables.features.contains(&feature), "{feature:?}");
    }
    assert!(!enables
        .features
        .contains(&VkFeature::ConstantAlphaColorBlendFactors));

    let unlisted =
        shader_enables(&decide(&at_the_floor()).unwrap_or_else(|below| panic!("{below}")));
    assert!(!unlisted
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset));
    assert!(!unlisted
        .features
        .contains(&VkFeature::ImageViewFormatSwizzle));
}

/// A device that lists a vertex attribute divisor extension, the KHR or only the EXT, and reports
/// the step-rate feature is given it, and the translator sees both; the zero divisor beside it is
/// not enabled, and a device that lists the extension without the feature is given the extension
/// alone.
#[test]
fn the_step_rate_feature_is_enabled_where_reported() {
    use crate::floor::shader_enables;
    use metal2vulkan::device::{VkExtension, VkFeature};
    let divisor = ash::khr::vertex_attribute_divisor::NAME;
    let mut device = at_the_floor();
    device.extensions.push(divisor.to_owned());
    let unreported = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert!(unreported.extensions.contains(&divisor));
    assert!(STEP_RATES
        .iter()
        .all(|feature| !holds(&unreported.features, feature)));
    assert!(!shader_enables(&unreported)
        .features
        .contains(&VkFeature::VertexAttributeInstanceRateDivisor));

    let reported = &mut device.features.divisor;
    reported.vertex_attribute_instance_rate_divisor = vk::TRUE;
    reported.vertex_attribute_instance_rate_zero_divisor = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    let enabled = &capabilities.features.divisor;
    assert_eq!(enabled.vertex_attribute_instance_rate_divisor, vk::TRUE);
    assert_eq!(
        enabled.vertex_attribute_instance_rate_zero_divisor,
        vk::FALSE
    );
    let enables = shader_enables(&capabilities);
    assert!(enables
        .features
        .contains(&VkFeature::VertexAttributeInstanceRateDivisor));
    assert!(enables
        .extensions
        .contains(&VkExtension::KhrVertexAttributeDivisor));

    let older = ash::ext::vertex_attribute_divisor::NAME;
    let mut device = at_the_floor();
    device.extensions.push(older.to_owned());
    device
        .features
        .divisor
        .vertex_attribute_instance_rate_divisor = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert!(capabilities.extensions.contains(&older));
    assert!(!capabilities.extensions.contains(&divisor));
    assert_eq!(
        capabilities
            .features
            .divisor
            .vertex_attribute_instance_rate_divisor,
        vk::TRUE
    );
    let enables = shader_enables(&capabilities);
    assert!(enables
        .features
        .contains(&VkFeature::VertexAttributeInstanceRateDivisor));
    assert!(enables
        .extensions
        .contains(&VkExtension::ExtVertexAttributeDivisor));
}

/// A device reads attachments where its descriptor limits hold the sampled-texture array's second
/// elements: through the attachment feedback loop where it lists both extensions and reports both
/// features, which device creation then enables, and through a copy of the image where it does
/// not, enabling neither. A device without the room reads no attachment, whatever it reports.
#[test]
fn the_attachment_reads_take_the_feedback_loop_where_reported_and_a_copy_elsewhere() {
    let roomy = |device: &mut Queried| {
        let descriptors = &mut device.vulkan12;
        descriptors.max_per_stage_descriptor_update_after_bind_sampled_images = 2 * 65_536;
        descriptors.max_descriptor_set_update_after_bind_sampled_images = 2 * 65_536;
        descriptors.max_per_stage_update_after_bind_resources = 3 * 65_536 + 8;
        descriptors.max_update_after_bind_descriptors_in_all_pools = 3 * 65_536 + 3 * 4000;
    };
    let looped = |device: &mut Queried| {
        (device.extensions).extend(
            ATTACHMENT_READ_EXTENSIONS
                .iter()
                .map(|&name| name.to_owned()),
        );
        device
            .features
            .feedback_loop
            .attachment_feedback_loop_layout = vk::TRUE;
        (device.features.feedback_loop_dynamic).attachment_feedback_loop_dynamic_state = vk::TRUE;
    };
    let answer = |device: &Queried| {
        let capabilities = decide(device).unwrap_or_else(|below| panic!("{below}"));
        let enabled = (ATTACHMENT_READ_EXTENSIONS.iter())
            .filter(|name| capabilities.extensions.contains(name))
            .count();
        let features = (ATTACHMENT_READS.iter())
            .filter(|feature| holds(&capabilities.features, feature))
            .count();
        (
            capabilities.attachment_reads,
            capabilities.feedback_loop,
            enabled,
            features,
        )
    };
    let mut device = at_the_floor();
    assert_eq!(answer(&device), (false, false, 0, 0), "at the floor");
    looped(&mut device);
    assert_eq!(
        answer(&device),
        (false, false, 0, 0),
        "no room, the loop reported"
    );
    let mut device = at_the_floor();
    roomy(&mut device);
    assert_eq!(
        answer(&device),
        (true, false, 0, 0),
        "room, no loop: the copy"
    );
    device
        .extensions
        .extend((ATTACHMENT_READ_EXTENSIONS.iter()).map(|&name| name.to_owned()));
    assert_eq!(
        answer(&device),
        (true, false, 0, 0),
        "room, the extensions listed and the features unreported: the copy"
    );
    let mut device = at_the_floor();
    roomy(&mut device);
    looped(&mut device);
    assert_eq!(answer(&device), (true, true, 2, 2), "room and the loop");
}

/// A set layout the created device does not support is named by its set.
#[test]
fn a_set_layout_the_device_does_not_support_is_named() {
    assert_eq!(
        Missing::SetLayout { set: 1 }.to_string(),
        "support for descriptor set 1's layout (vkGetDescriptorSetLayoutSupport)"
    );
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
    let path = crate::tests::crate_dir().join("README.md");
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
        .chain(
            OPTIONAL_INSTANCE_EXTENSIONS
                .iter()
                .map(|name| (name, "instance, where offered")),
        )
        .chain(DEVICE_EXTENSIONS.iter().map(|name| (name, "device")))
        .chain(
            OPTIONAL_EXTENSIONS
                .iter()
                .map(|name| (name, "device, where listed")),
        )
        .chain(
            ATTACHMENT_READ_EXTENSIONS
                .iter()
                .map(|name| (name, "device, with the feedback loop")),
        )
        .map(|(name, place)| (name.to_string_lossy().into_owned(), place.to_string()))
        .collect();
    assert_eq!(stated, code, "the README's extensions");

    for (header, features) in [
        (["Feature", "Structure", "Used for"], REQUIRED),
        (["Optional feature", "Structure", "Used for"], OPTIONAL),
        (
            ["Portability feature", "Structure", "Used for"],
            PORTABILITY,
        ),
        (
            ["Attachment-read feature", "Structure", "Used for"],
            ATTACHMENT_READS,
        ),
        (["Step-rate feature", "Structure", "Used for"], STEP_RATES),
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

/// Every device of the cfg's kind the machine's Vulkan loader lists to the runtime's instance
/// (portability devices too, where the loader offers their listing) is above the floor: CPU devices
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
            mut maximal_reconvergence,
            mut subgroup_uniform_control_flow,
            mut robustness2,
            mut portability,
            mut feedback_loop,
            mut feedback_loop_dynamic,
            mut divisor,
        } = capabilities.features;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan11)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut robustness2);
        let enabled = |name: &std::ffi::CStr| capabilities.extensions.contains(&name);
        if enabled(ash::khr::shader_maximal_reconvergence::NAME) {
            features = features.push_next(&mut maximal_reconvergence);
        }
        if enabled(ash::khr::shader_subgroup_uniform_control_flow::NAME) {
            features = features.push_next(&mut subgroup_uniform_control_flow);
        }
        if enabled(ash::khr::portability_subset::NAME) {
            features = features.push_next(&mut portability);
        }
        if capabilities.feedback_loop {
            features = features
                .push_next(&mut feedback_loop)
                .push_next(&mut feedback_loop_dynamic);
        }
        if enabled(ash::khr::vertex_attribute_divisor::NAME)
            || enabled(ash::ext::vertex_attribute_divisor::NAME)
        {
            features = features.push_next(&mut divisor);
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

    let runtime = crate::tests::validation::instance(&[]);
    let (entry, instance) = runtime.raw();
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
        let capabilities = match unsafe { check(entry, instance, physical) } {
            Ok(capabilities) => capabilities,
            Err(refusal) => {
                below.push(format!("{name} ({version}, {info}): {refusal}"));
                continue;
            }
        };
        // SAFETY: as above, and `capabilities` is `physical`'s.
        let memory = match unsafe { memory_on_a_device(instance, physical, &capabilities) } {
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
             (flags {:#x}), ring memory type {} (flags {:#x}); extensions enabled {:?}; \
             portability features enabled: imageViewFormatSwizzle {}, \
             mutableComparisonSamplers {}, multisampleArrayImage {}",
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
            flags(memory.ring),
            capabilities.extensions,
            capabilities.features.portability.image_view_format_swizzle,
            capabilities
                .features
                .portability
                .mutable_comparison_samplers,
            capabilities.features.portability.multisample_array_image
        );
        println!(
            "floor: {name}: attachment reads {}, through the feedback loop {}",
            capabilities.attachment_reads, capabilities.feedback_loop
        );
    }
    drop(runtime);
    assert!(checked > 0, "the loader lists no device of this cfg's kind");
    assert!(below.is_empty(), "below the floor: {below:#?}");
}

/// A device that does not calibrate the host clock's time domain with its own is below the floor,
/// named for the domain; one that calibrates neither is named for both, its own first.
#[test]
fn a_device_that_does_not_calibrate_the_host_clock_is_below_the_floor() {
    let mut device = at_the_floor();
    device.time_domains = vec![vk::TimeDomainKHR::DEVICE];
    let host = crate::floor::time_domain_name(crate::HOST_DOMAIN);
    let missing = only_missing(&device);
    assert_eq!(missing, Missing::TimeDomain(host));
    assert_eq!(
        missing.to_string(),
        format!("the calibrateable time domain {host}")
    );
    device.time_domains.clear();
    let Err(below) = decide(&device) else {
        panic!("the device is above the floor");
    };
    assert_eq!(
        below.missing,
        [
            Missing::TimeDomain("VK_TIME_DOMAIN_DEVICE_EXT"),
            Missing::TimeDomain(host)
        ]
    );
    assert!(TIME_DOMAINS[1] == crate::HOST_DOMAIN);
}
