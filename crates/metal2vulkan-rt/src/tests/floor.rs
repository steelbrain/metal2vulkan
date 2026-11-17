//! The floor's check: each missing item is named, the four optional features decide
//! `Capabilities`, the README states what the code checks, and, under a device cfg, every device
//! the loader lists is above the floor.

use crate::floor::{
    decide, limits, vulkan_name, BelowFloor, Features, Missing, Queried, DEVICE_EXTENSIONS,
    INSTANCE_EXTENSIONS, OPTIONAL, REQUIRED, VERSION,
};
use ash::vk;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// A device with exactly the floor: Vulkan 1.3.0, the swapchain extension, every required feature,
/// every limit at its minimum, and no optional feature.
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
            .max_compute_work_group_size([1024, 1024, 64]),
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
        extensions: vec![ash::khr::swapchain::NAME.to_owned()],
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

#[test]
fn a_device_at_the_floor_is_above_it() {
    let mut device = at_the_floor();
    // Features beyond the floor are not enabled.
    device.features.core.geometry_shader = vk::TRUE;
    device.features.vulkan12.scalar_block_layout = vk::TRUE;
    device.features.vulkan13.robust_image_access = vk::TRUE;
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    assert!(!capabilities.float16);
    assert!(!capabilities.storage_read_without_format);
    assert!(!capabilities.buffer_int64_atomics);
    assert!(!capabilities.shared_int64_atomics);
    assert!(!capabilities.float64);
    assert_eq!(capabilities.extensions, DEVICE_EXTENSIONS);
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
    assert_eq!(enabled.vulkan12.scalar_block_layout, vk::FALSE);
    assert_eq!(enabled.vulkan13.robust_image_access, vk::FALSE);
}

#[test]
fn each_missing_feature_is_named() {
    const V12: &str = "VkPhysicalDeviceVulkan12Features";
    const V13: &str = "VkPhysicalDeviceVulkan13Features";
    const CORE: &str = "VkPhysicalDeviceFeatures";
    type Clear = fn(&mut Features);
    let cases: [(Clear, Missing); 15] = [
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
            |f| f.vulkan12.timeline_semaphore = vk::FALSE,
            feature(V12, "timelineSemaphore"),
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
    let cases: [(Lower, &str, u32); 14] = [
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
fn a_missing_extension_is_named() {
    let mut device = at_the_floor();
    device.extensions.clear();
    assert_eq!(
        only_missing(&device),
        Missing::Extension(ash::khr::swapchain::NAME)
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
    let Err(below) = decide(&device) else {
        panic!("the device is above the floor");
    };
    assert_eq!(
        below.missing,
        [
            Missing::Extension(ash::khr::swapchain::NAME),
            feature("VkPhysicalDeviceFeatures", "shaderInt64"),
            feature("VkPhysicalDeviceVulkan13Features", "dynamicRendering"),
            Missing::Limit {
                name: "maxPushConstantsSize",
                device: 64,
                needed: 128,
            },
        ]
    );
    assert_eq!(
        below.to_string(),
        "the device is below the runtime's floor: it lacks VK_KHR_swapchain; \
         VkPhysicalDeviceFeatures.shaderInt64; VkPhysicalDeviceVulkan13Features.dynamicRendering; \
         maxPushConstantsSize 64, below 128"
    );
}

#[test]
fn the_four_optional_features_decide_the_capabilities() {
    type Set = fn(&mut Features);
    let cases: [(Set, &str); 5] = [
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
        .collect();
    assert_eq!(stated, code, "the README's limits");
}

#[test]
fn vulkan_names_follow_the_registry() {
    assert_eq!(vulkan_name("buffer_device_address"), "bufferDeviceAddress");
    assert_eq!(vulkan_name("shader_int64"), "shaderInt64");
    assert_eq!(vulkan_name("synchronization2"), "synchronization2");
}

/// Every device the machine's Vulkan loader lists is above the floor. The device legs pick one
/// device per run with `VK_DRIVER_FILES` and, for Mesa, `MESA_VK_DEVICE_SELECT`.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn every_device_the_loader_lists_is_above_the_floor() {
    use crate::floor::check;
    use std::ffi::c_char;

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
    let mut below = Vec::new();
    for &physical in &physicals {
        let mut driver = vk::PhysicalDeviceVulkan12Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: `physical` is one of `instance`'s, which is a Vulkan 1.3 instance.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
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
        match unsafe { check(&instance, physical) } {
            Ok(capabilities) => println!(
                "floor: {name} ({version}, {info}) is above the floor; float16 {}, storage reads \
                 without format {}, 64-bit atomics on buffers {} and threadgroup memory {}, \
                 float64 {}",
                capabilities.float16,
                capabilities.storage_read_without_format,
                capabilities.buffer_int64_atomics,
                capabilities.shared_int64_atomics,
                capabilities.float64
            ),
            Err(refusal) => below.push(format!("{name} ({version}, {info}): {refusal}")),
        }
    }
    // SAFETY: nothing made from `instance` is alive.
    unsafe { instance.destroy_instance(None) };
    assert!(!physicals.is_empty(), "the loader lists no device");
    assert!(below.is_empty(), "below the floor: {below:#?}");
}
