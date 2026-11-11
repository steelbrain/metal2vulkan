//! The profiles layer's checks over the committed profiles, device-free.

use super::*;
use crate::engine::{committed_profiles, load_description};
use metal2vulkan::device::{FormatProperties, FormatSupport};
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::vk;
use std::collections::BTreeMap;

/// What a context restricted to committed profile `stem` refuses of a device described by
/// `device` before it reads the device through the layer: the profile's presentation gap, else
/// what the profile states beyond the device at the context's version.
fn gaps_of(stem: &str, device: &DeviceDescription) -> Vec<String> {
    let restriction = ProfileRestriction::committed(stem).unwrap();
    match restriction.presentation_gap() {
        Some(gap) => vec![gap],
        None => gaps_at(&restriction, device, restriction.context_version()),
    }
}

const AT_1_2: ApiVersion = ApiVersion::V1_2;

fn committed(stem: &str) -> DeviceDescription {
    load_description(stem).unwrap()
}

fn tool(layer: &CStr) -> ActiveTool {
    ActiveTool {
        name: "a tool".into(),
        version: "1.0.0".into(),
        purposes: 0,
        layer: layer.to_string_lossy().into_owned(),
    }
}

#[test]
fn every_committed_profile_is_a_restriction() {
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        assert_eq!(restriction.stem, stem);
        assert_eq!(
            restriction.profile_name,
            format!("VP_M2V_{}", stem.replace('-', "_"))
        );
        assert!(restriction.file.is_absolute(), "{stem}");
        assert_eq!(restriction.sha256.len(), 64, "{stem}");
        assert_eq!(restriction.description, committed(&stem), "{stem}");
    }
    let unknown = ProfileRestriction::committed("no-such-profile").unwrap_err();
    assert!(unknown.contains("committed profiles: "), "{unknown}");
    assert!(unknown.contains("vk12-floor"), "{unknown}");
    assert!(restrict_devices_to("no-such-profile").is_err());
}

/// The layer lists every extension promoted to core at or below P's version; the context hides
/// the ones P does not list. `swiftshader` (1.3) does not list
/// `VK_EXT_texture_compression_astc_hdr` (promoted in 1.3), which SwiftShader lacks. It lists
/// `VK_KHR_format_feature_flags2` but, a gpuinfo report without them, states no
/// `VkFormatProperties3`, so that is hidden too. A `vulkaninfo` capture with its supplement
/// (`gate-lavapipe`) states them, and keeps the extension.
#[test]
fn the_extensions_a_profile_does_not_list_or_describe_are_hidden() {
    let restriction = ProfileRestriction::committed("swiftshader").unwrap();
    for listed in ["VK_EXT_image_robustness", FORMAT_FEATURE_FLAGS2] {
        assert!(restriction.extension_names.contains(listed), "{listed}");
    }
    assert!(!restriction
        .extension_names
        .contains("VK_EXT_texture_compression_astc_hdr"));
    assert!(!restriction.states_format_flags2);
    let presented = [
        "VK_EXT_image_robustness",
        "VK_EXT_texture_compression_astc_hdr",
        FORMAT_FEATURE_FLAGS2,
    ];
    assert_eq!(
        restriction.hidden(presented),
        BTreeSet::from([
            "VK_EXT_texture_compression_astc_hdr".to_owned(),
            FORMAT_FEATURE_FLAGS2.to_owned()
        ])
    );
    // The context translates for P without it, whose bits it does not report.
    assert!(restriction.description.khr_format_feature_flags2);
    assert!(!restriction.read_at(AT_1_2).has_format_feature_flags2());
    // A report or a capture that states VkFormatProperties3 keeps the extension.
    for stem in ["nvidia-rtx50-linux", "gate-lavapipe"] {
        let stating = ProfileRestriction::committed(stem).unwrap();
        assert!(stating.states_format_flags2, "{stem}");
        assert!(
            stating.extension_names.contains(FORMAT_FEATURE_FLAGS2),
            "{stem}"
        );
        assert!(stating.hidden([FORMAT_FEATURE_FLAGS2]).is_empty(), "{stem}");
        assert!(
            stating.read_at(AT_1_2).has_format_feature_flags2(),
            "{stem}"
        );
    }
    // Each extension the product reads from a profile is the reading of a name it lists.
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        assert!(
            restriction.extension_names.len() >= restriction.description.extensions.len(),
            "{stem}"
        );
    }
}

/// Every member of every committed profile has a subset rule (the check fails closed on a member
/// it has none for), and every profile is within itself.
#[test]
fn every_committed_profile_is_within_itself() {
    for stem in committed_profiles().unwrap() {
        for description in [committed(&stem), read_at(&committed(&stem), AT_1_2)] {
            let gaps = exceeds(&description, &description);
            assert!(gaps.is_empty(), "{stem}: {gaps:?}");
            assert!(differences(&description, &description).is_empty(), "{stem}");
        }
    }
}

/// A profile that states features a device lacks is named member by member: the floor with four
/// optional features SwiftShader lacks (`shaderInt16`, 8- and 16-bit storage access, and formatless
/// storage-image reads) names each of them on SwiftShader.
#[test]
fn a_profile_beyond_the_device_is_named_member_by_member() {
    let lacked = [
        VkFeature::ShaderInt16,
        VkFeature::StorageBuffer8BitAccess,
        VkFeature::StorageBuffer16BitAccess,
        VkFeature::ShaderStorageImageReadWithoutFormat,
    ];
    let mut beyond = read_at(&committed("vk12-floor"), AT_1_2);
    beyond.features.extend(lacked);
    let swiftshader = read_at(&committed("swiftshader"), AT_1_2);
    let gaps = exceeds(&beyond, &swiftshader);
    for feature in lacked {
        assert!(
            gaps.contains(&format!("feature {feature:?}")),
            "{feature:?}: {gaps:?}"
        );
    }
    // The committed x86 lavapipe has more than the floor; the reverse does not hold.
    let floor = read_at(&committed("vk12-floor"), AT_1_2);
    let lavapipe = read_at(&committed("lavapipe"), AT_1_2);
    assert!(exceeds(&floor, &lavapipe).is_empty());
    assert!(!exceeds(&lavapipe, &floor).is_empty());
}

/// The floor is Vulkan 1.2 core, which every Vulkan 1.2 device has: Intel's Iris Xe on ANV, which
/// reads no storage image without its format, takes it as the other Linux venue captures do.
#[test]
fn every_device_capture_takes_the_floor() {
    let floor = read_at(&committed("vk12-floor"), AT_1_2);
    for stem in [
        "intel-irisxe-rplp-anv",
        "intel-uhd770-rpls-anv",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
        "gate-lavapipe",
    ] {
        let device = read_at(&committed(stem), AT_1_2);
        assert_eq!(exceeds(&floor, &device), Vec::<String>::new(), "{stem}");
    }
}

#[test]
fn each_kind_of_member_is_bounded_in_its_direction() {
    let device = read_at(&committed("gate-lavapipe"), AT_1_2);
    let gaps = |change: &dyn Fn(&mut DeviceDescription)| {
        let mut profile = device.clone();
        change(&mut profile);
        exceeds(&profile, &device)
    };
    let one = |change: &dyn Fn(&mut DeviceDescription), expected: &str| {
        let found = gaps(change);
        assert_eq!(found.len(), 1, "{expected}: {found:?}");
        assert!(found[0].starts_with(expected), "{expected}: {found:?}");
    };
    // Less than the device is within it.
    assert!(gaps(&|p| p.limits.max_push_constants_size -= 4).is_empty());
    assert!(gaps(&|p| p.limits.min_storage_buffer_offset_alignment *= 2).is_empty());
    assert!(gaps(&|p| {
        p.features.remove(&VkFeature::ShaderInt64);
    })
    .is_empty());
    // More is named.
    one(
        &|p| p.limits.max_push_constants_size += 4,
        "limits.max_push_constants_size",
    );
    one(
        &|p| p.limits.max_compute_work_group_size[1] += 1,
        "limits.max_compute_work_group_size.1",
    );
    one(
        &|p| p.limits.min_storage_buffer_offset_alignment /= 2,
        "limits.min_storage_buffer_offset_alignment",
    );
    one(
        &|p| p.limits.min_texel_offset -= 1,
        "limits.min_texel_offset",
    );
    one(
        &|p| {
            p.limits.max_sampler_anisotropy_bits =
                (f32::from_bits(p.limits.max_sampler_anisotropy_bits) * 2.0).to_bits()
        },
        "limits.max_sampler_anisotropy_bits",
    );
    // A stride rule a device without one does not have is stricter, so within it; a profile
    // without the rule on a device with one states more.
    assert!(gaps(&|p| p.limits.min_vertex_input_binding_stride_alignment = Some(4)).is_empty());
    let mut ruled = device.clone();
    ruled.limits.min_vertex_input_binding_stride_alignment = Some(4);
    let found = exceeds(&device, &ruled);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].starts_with("limits.min_vertex_input_binding_stride_alignment"),
        "{found:?}"
    );
    one(
        &|p| p.subgroup.supported_operations |= vk::SubgroupFeatureFlags::from_raw(1 << 30),
        "subgroup.supported_operations",
    );
    one(&|p| p.subgroup.max_size *= 2, "subgroup.max_size");
    one(&|p| p.subgroup.min_size /= 2, "subgroup.min_size");
    // The generated-commands facts: a profile without them is within a device with them, more
    // sequences or another bound stage is named, and a profile that states them on a device without
    // them states more.
    assert!(device.generated_commands.is_some());
    assert!(gaps(&|p| p.generated_commands = None).is_empty());
    one(
        &|p| {
            if let Some(facts) = &mut p.generated_commands {
                facts.max_indirect_sequence_count += 1;
            }
        },
        "generated_commands.max_indirect_sequence_count",
    );
    one(
        &|p| {
            if let Some(facts) = &mut p.generated_commands {
                facts.pipeline_binding_stages |= vk::ShaderStageFlags::from_raw(1 << 30);
            }
        },
        "generated_commands.pipeline_binding_stages",
    );
    let mut without = device.clone();
    without.generated_commands = None;
    let found = exceeds(&device, &without);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("generated_commands "), "{found:?}");
    one(
        &|p| p.api_version = ApiVersion { major: 1, minor: 3 },
        "api_version",
    );
    one(
        &|p| {
            p.extensions.insert(VkExtension::KhrPortabilitySubset);
        },
        "extension KhrPortabilitySubset",
    );
    one(
        &|p| {
            let format = vk::Format::R8G8B8A8_UNORM;
            let mut properties = p.formats.formats.get(&format).copied().unwrap_or_default();
            properties.buffer |= vk::FormatFeatureFlags2::from_raw(1 << 30);
            p.formats.formats.insert(format, properties);
        },
        "formats.VK_FORMAT_R8G8B8A8_UNORM.buffer",
    );
    one(
        &|p| {
            p.formats.formats.insert(
                vk::Format::from_raw(1_000_000_000),
                FormatProperties::default(),
            );
            let format = vk::Format::from_raw(1_000_000_001);
            let mut properties = FormatProperties::default();
            properties.optimal_tiling = vk::FormatFeatureFlags2::from_raw(1);
            p.formats.formats.insert(format, properties);
        },
        "formats.VkFormat(1000000001).optimal_tiling",
    );
    // The subgroup size describes execution, not capacity: not compared.
    assert!(gaps(&|p| p.subgroup.size = 1).is_empty());
    // Identity is not a capability.
    assert!(gaps(&|p| p.vendor_id ^= 1).is_empty());
}

#[test]
fn differences_name_each_member_with_both_values() {
    let expected = read_at(&committed("gate-lavapipe"), AT_1_2);
    assert!(differences(&expected, &expected).is_empty());
    let mut queried = expected.clone();
    queried.limits.max_push_constants_size += 4;
    queried.features.remove(&VkFeature::ShaderInt64);
    queried.subgroup.size = 8;
    // Identity is compared apart, native against queried.
    queried.vendor_id ^= 1;
    let found = differences(&expected, &queried);
    let paths: Vec<&str> = found.iter().map(|d| d.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "features.ShaderInt64",
            "limits.max_push_constants_size",
            "subgroup.size"
        ]
    );
    assert_eq!(found[0].expected, "listed");
    assert_eq!(found[0].queried, "absent");
    assert_eq!(
        found[1].to_string(),
        format!(
            "limits.max_push_constants_size: expected {}, queried {}",
            expected.limits.max_push_constants_size,
            expected.limits.max_push_constants_size + 4
        )
    );
    assert_eq!(found[2].queried, "8");
}

/// A context at API 1.2 reads a later profile at 1.2: the version, and only what 1.2 exposes with
/// the listed extensions.
#[test]
fn a_later_profile_is_read_at_the_instance_version() {
    let roadmap = committed("roadmap-2022");
    assert!(roadmap.api_version > AT_1_2);
    let read = read_at(&roadmap, AT_1_2);
    assert_eq!(read.api_version, AT_1_2);
    assert!(read.features.is_subset(&roadmap.features));
    assert!(read.features.len() < roadmap.features.len());
    for feature in &read.features {
        assert!(
            feature.exposure().exposed(AT_1_2, &read.extensions),
            "{feature:?}"
        );
    }
    let queried = FormatSupport::queries_for(AT_1_2, &read.extensions);
    for format in read.formats.formats.keys() {
        assert!(queried.contains(format), "{format:?}");
    }
    // A 1.2 profile reads as itself, less the formats the product never asks about: the floor
    // states the scaled vertex formats, which no Metal format maps to.
    let floor = committed("vk12-floor");
    let read = read_at(&floor, AT_1_2);
    assert_eq!(read.features, floor.features);
    assert_eq!(read.limits, floor.limits);
    assert!(read.formats.formats.len() < floor.formats.formats.len());
    assert!(!read.formats.formats.contains_key(&vk::Format::R8_USCALED));
}

#[test]
fn a_restricted_context_runs_only_on_a_device_that_has_and_reports_its_profile() {
    let restriction = ProfileRestriction::committed("vk12-floor").unwrap();
    let native = read_at(&committed("gate-lavapipe"), AT_1_2);
    let tools = [tool(c"VK_LAYER_KHRONOS_validation"), tool(PROFILES_LAYER)];
    // What the layer reports: the profile as read at 1.2, with the members it keeps (the
    // identity and `KEPT`) the device's.
    let profile = restriction.read_at(AT_1_2);
    let mut queried = profile.clone();
    queried.driver_id = native.driver_id;
    queried.driver_version = native.driver_version;
    queried.vendor_id = native.vendor_id;
    queried.device_id = native.device_id;
    queried.device_name.clone_from(&native.device_name);
    queried.device_type = native.device_type;
    queried.float_controls.denorm_independence = native.float_controls.denorm_independence;
    queried.float_controls.rounding_independence = native.float_controls.rounding_independence;
    queried.rgba4_opaque_black_swizzled = native.rgba4_opaque_black_swizzled;
    assert_ne!(
        queried.float_controls.denorm_independence,
        profile.float_controls.denorm_independence
    );
    assert_eq!(check_within(&restriction, &native, AT_1_2), Ok(()));
    let check = |native: &DeviceDescription, tools: &[ActiveTool], queried: &DeviceDescription| {
        check_restricted(&restriction, native, tools, queried, AT_1_2)
    };
    // The context translates for the profile, not for the kept members' device values, on the
    // device that executes it: the identity is the device's, which the layer keeps.
    let mut translated = profile.clone();
    translated.driver_id = native.driver_id;
    translated.driver_version = native.driver_version;
    translated.vendor_id = native.vendor_id;
    translated.device_id = native.device_id;
    translated.device_name.clone_from(&native.device_name);
    translated.device_type = native.device_type;
    assert_ne!(translated.driver_id, profile.driver_id);
    assert_eq!(check(&native, &tools, &queried), Ok(translated.clone()));
    // So a driver workaround follows the driver that runs the module: the profile on lavapipe's
    // identity carries lavapipe's (and at the gate's Mesa 26.0.8 the misplaced mesh index stores
    // measured there), and the same profile on SwiftShader's identity checks its buffer atomics in
    // the shader.
    let workarounds = |description: &DeviceDescription| {
        metal2vulkan::device::Device::new(
            description.clone(),
            metal2vulkan::device::DevicePolicy::default(),
        )
        .expect("device")
        .profile()
        .workarounds()
        .to_vec()
    };
    assert_eq!(
        workarounds(&translated),
        [
            metal2vulkan::device::DriverWorkaround::ClippedTrianglesSplitPixelFragments,
            metal2vulkan::device::DriverWorkaround::MeshIndexComponentStoresMisplaced,
            metal2vulkan::device::DriverWorkaround::ClippedCoverageNotMetals,
            metal2vulkan::device::DriverWorkaround::TessellatorNotMetals,
        ]
    );
    let mut swiftshader_native = native.clone();
    swiftshader_native.driver_id = Some(vk::DriverId::GOOGLE_SWIFTSHADER);
    let mut swiftshader_queried = queried.clone();
    swiftshader_queried.driver_id = swiftshader_native.driver_id;
    let on_swiftshader = check(&swiftshader_native, &tools, &swiftshader_queried).unwrap();
    assert_eq!(
        workarounds(&on_swiftshader),
        [
            metal2vulkan::device::DriverWorkaround::BufferAtomicsUncontained,
            metal2vulkan::device::DriverWorkaround::ClippedCoverageNotMetals,
            metal2vulkan::device::DriverWorkaround::TessellatorNotMetals,
        ]
    );
    assert!(matches!(
        check(&native, &tools[..1], &queried),
        Err(RestrictionError::Inactive { .. })
    ));
    // SwiftShader takes the floor (Vulkan 1.2's required features and minimum limits); a device
    // short of one of its limits does not.
    let swiftshader = read_at(&committed("swiftshader"), AT_1_2);
    assert_eq!(check_within(&restriction, &swiftshader, AT_1_2), Ok(()));
    let mut short = swiftshader;
    short.limits.max_bound_descriptor_sets = 3;
    assert!(matches!(
        check_within(&restriction, &short, AT_1_2),
        Err(RestrictionError::Exceeds { .. })
    ));
    let mut renamed = queried.clone();
    renamed.device_type = Some(vk::PhysicalDeviceType::DISCRETE_GPU);
    let Err(RestrictionError::Kept { fields, .. }) = check(&native, &tools, &renamed) else {
        panic!("the layer renamed the device");
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].path, "device_type");
    // A kept member may read between P and the device (the reader's default where P does not
    // expose its structure), but not beyond the device.
    let mut set = queried.clone();
    set.float_controls.rounding_independence = profile.float_controls.rounding_independence;
    assert_eq!(check(&native, &tools, &set), Ok(translated.clone()));
    let mut weaker = native.clone();
    weaker.float_controls.rounding_independence = Independence::None;
    let Err(RestrictionError::Kept { fields, .. }) = check(&weaker, &tools, &queried) else {
        panic!("the layer reported more than the device has");
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].path, "float_controls.rounding_independence");
    assert_eq!(fields[0].expected, r#"between "None" and "None""#);
    let mut other = queried.clone();
    other.limits.max_bound_descriptor_sets += 1;
    let Err(error @ RestrictionError::Mismatch { .. }) = check(&native, &tools, &other) else {
        panic!("a queried description that is not the profile");
    };
    assert!(
        error
            .to_string()
            .contains("limits.max_bound_descriptor_sets: expected 4, queried 5"),
        "{error}"
    );
    // The layer answers a color attachment's sample counts from the device, as it keeps the `KEPT`
    // members: they may read between P's and the device's, not beyond the device's.
    let format = vk::Format::R8G8B8A8_UNORM;
    let four = vk::SampleCountFlags::COUNT_1 | vk::SampleCountFlags::COUNT_4;
    let mut counted_native = native.clone();
    let mut counted = queried.clone();
    for description in [&mut counted_native, &mut counted] {
        description
            .formats
            .formats
            .get_mut(&format)
            .expect("a color attachment format")
            .color_attachment_sample_counts = four;
    }
    assert_eq!(
        check(&counted_native, &tools, &counted),
        Ok(translated.clone())
    );
    let mut lacking = native.clone();
    lacking
        .formats
        .formats
        .get_mut(&format)
        .expect("a color attachment format")
        .color_attachment_sample_counts = vk::SampleCountFlags::COUNT_1;
    let Err(RestrictionError::Kept { fields, .. }) = check(&lacking, &tools, &counted) else {
        panic!("the layer reported counts the device lacks");
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(
        fields[0].path,
        "formats.VK_FORMAT_R8G8B8A8_UNORM.color_attachment_sample_counts"
    );
}

/// Every [`KEPT`] path is a capability member of the description with a subset rule.
#[test]
fn every_kept_member_is_a_bounded_capability_member() {
    let description = committed("gate-lavapipe");
    let members = capability_members(&description);
    for path in KEPT {
        let segments: Vec<String> = path.split('.').map(str::to_owned).collect();
        let value = segments
            .iter()
            .try_fold(&members, |value, segment| value.get(segment))
            .unwrap_or_else(|| panic!("{path} is not a member"));
        assert!(!value.is_object(), "{path}");
        assert!(bound(&segments).is_some(), "{path}");
    }
}

/// A restriction the device cannot take is a typed refusal, recorded with the device as it is,
/// never a failure without a record (which a re-record would retry without end). Pascal's committed
/// profile against the RTX 50 description refuses the three members the RTX 5080's restricted
/// context refused; the checks before the context reads the device agree; and the refusal is a
/// device refusal whose record is well formed, in the restricted slot, with the translator
/// fingerprint its own environment's profile fingerprint derives (the currency rule's product
/// check).
#[test]
fn a_restriction_the_device_cannot_take_is_a_recorded_refusal() {
    use crate::candidate::engine::{engine_fingerprint_of, engine_slot, ENGINE_EXECUTOR_ABI};
    use crate::check::admission::{ExecutorError, NotSupported};
    use crate::coverage::Leg;
    use crate::observation::Backend;
    use crate::refusal::{Refusal, RefusalClass, RefusalObservation};

    let restriction: &'static ProfileRestriction = Box::leak(Box::new(
        ProfileRestriction::committed("nvidia-pascal-linux").unwrap(),
    ));
    let native = committed("nvidia-rtx50-linux");
    let Err(RestrictionError::Exceeds { gaps, .. }) = check_within(restriction, &native, AT_1_2)
    else {
        panic!("the RTX 50 description takes Pascal's profile");
    };
    for member in [
        "limits.max_compute_work_group_invocations 1536 beyond the device's 1024",
        "limits.max_compute_work_group_size.0 1536 beyond the device's 1024",
        "subgroup.max_compute_workgroup_subgroups 48 beyond the device's 32",
    ] {
        assert!(gaps.iter().any(|gap| gap == member), "{member}: {gaps:?}");
    }
    assert_eq!(gaps_of("nvidia-pascal-linux", &native), gaps);
    assert_eq!(gaps_of("nvidia-rtx50-linux", &native), Vec::<String>::new());

    let refused = super::super::device::RefusedRestriction {
        restriction,
        gaps: gaps.clone(),
        native,
        identity: serde_json::json!({
            "device": "a GPU",
            "device_id": 1,
            "vendor_id": 0x10de,
            "driver_version": 2,
            "api_version": 3,
            "driver_id": 4,
            "driver_name": "a driver",
            "driver_info": "1.0",
        }),
    };
    let attempt = super::super::compute::restriction_refusal(&refused, "engine-compute");
    let not_supported = match attempt.result {
        Err(ExecutorError::NotSupported(not_supported)) => not_supported,
        other => panic!("{:?}", other.err()),
    };
    assert_eq!(
        not_supported,
        NotSupported::Restriction {
            profile: "nvidia-pascal-linux".into(),
            gaps: gaps.clone(),
        }
    );
    assert_eq!(
        not_supported.outcome(),
        "not_supported{profile_exceeds_device}"
    );
    assert!(not_supported
        .to_string()
        .contains("profile nvidia-pascal-linux states 3 capabilities the device lacks"));
    let opened = attempt.refused_on.expect("the refusal names the device");
    let environment = &opened.environment;
    assert_eq!(environment["device"], "a GPU");
    assert_eq!(environment["executor"], "engine-compute");
    assert_eq!(
        environment["profiles_layer"]["profile"],
        "nvidia-pascal-linux"
    );
    assert_eq!(environment["robustness"]["leg"], "stated");
    let hex = environment["profile_fingerprint"].as_str().unwrap();
    let mut fingerprint = [0u8; 32];
    for (byte, pair) in fingerprint.iter_mut().zip(hex.as_bytes().chunks(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    assert_eq!(
        engine_fingerprint_of(&fingerprint),
        opened.translator_fingerprint
    );
    let record = RefusalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: "a".repeat(64),
        air_sha256: "b".repeat(64),
        input_sha256: "c".repeat(64),
        translator_fingerprint: opened.translator_fingerprint.clone(),
        backend: Backend::Vulkan,
        environment_id: engine_slot(
            "nvidia-rtx5080-laptop",
            Some("nvidia-pascal-linux"),
            Leg::Stated,
        ),
        environment: opened.environment.clone(),
        executor_abi: ENGINE_EXECUTOR_ABI.into(),
        refusal: Refusal::of(&not_supported),
        artifact: None,
    };
    record.validate_content().unwrap();
    assert_eq!(record.refusal.class, RefusalClass::Device);
}

/// A venue run stops on its device's identity or translation key, never on a description member
/// no translation reads: lavapipe reports its `VK_EXT_device_generated_commands` stage masks as
/// `VK_SHADER_STAGE_ALL`, whose undefined bits a `vulkaninfo` capture cannot state, so its
/// committed profile reads back the defined bits only; the run reports the two masks and proceeds.
/// A member translation reads moves the key and stops it, with every differing member as the
/// explanation; another driver stops it on the identity.
#[test]
fn a_venue_run_stops_on_identity_or_key_and_reports_unread_members() {
    let venue = crate::c3::VenueProfile::committed("gate-lavapipe").unwrap();
    let allowed = std::slice::from_ref(&venue.identity);
    assert_eq!(
        venue_check(&venue, allowed, &venue.identity, &venue.description).unwrap(),
        crate::c3::VenueCheck::default()
    );

    let mut live = venue.description.clone();
    let generated = live
        .generated_commands
        .as_mut()
        .expect("lavapipe lists VK_EXT_device_generated_commands");
    assert_eq!(generated.supported_stages.as_raw(), 557_055);
    generated.supported_stages = vk::ShaderStageFlags::ALL;
    generated.pipeline_binding_stages = vk::ShaderStageFlags::ALL;
    let check = venue_check(&venue, allowed, &venue.identity, &live).unwrap();
    assert!(check.stops.is_empty(), "{check:?}");
    assert_eq!(check.unread.len(), 2, "{check:?}");
    assert!(
        check
            .unread
            .iter()
            .all(|member| member.starts_with("generated_commands.")),
        "{check:?}"
    );

    let mut lacking = live.clone();
    assert!(lacking.features.remove(&VkFeature::ShaderInt16));
    let check = venue_check(&venue, allowed, &venue.identity, &lacking).unwrap();
    assert!(check.unread.is_empty(), "{check:?}");
    assert!(check.stops[0].starts_with("translation key: "), "{check:?}");
    assert!(
        check
            .stops
            .contains(&"features.ShaderInt16: expected listed, queried absent".to_string()),
        "{check:?}"
    );
    assert_eq!(check.stops.len(), 4, "{check:?}");

    let mut other = venue.identity.clone();
    other.driver_info = "Mesa 26.0.9".into();
    let check = venue_check(&venue, allowed, &other, &venue.description).unwrap();
    assert_eq!(check.unread, Vec::<String>::new());
    assert_eq!(check.stops.len(), 1, "{check:?}");
    assert!(
        check.stops[0].starts_with("identity driver_info: "),
        "{check:?}"
    );
}

/// One venue, two devices: `intel-irisxe-rplp-anv` allows the Iris Xe (RPL-P) it was first captured
/// on (ANV 26.2.6) and the UHD Graphics 770 (RPL-S, ANV 26.2.4). Their committed captures describe
/// the same capabilities but for `maxMemoryAllocationSize`, which the driver sizes from its host's
/// memory and no translation reads (`DeviceDescription::allocation`): both derive the venue's
/// translation key, so a run on either proceeds and only reports the size. A device the venue does
/// not allow stops on its identity, each field named against the nearest allowed identity; a venue
/// allowing the RPL-P alone stops the RPL-S on its name and device ID.
#[test]
fn one_intel_venue_allows_both_devices_and_stops_any_other() {
    let venue = crate::c3::VenueProfile::committed("intel-irisxe-rplp-anv").unwrap();
    let rpls = crate::c3::VenueProfile::committed("intel-uhd770-rpls-anv").unwrap();
    assert_eq!(rpls.fingerprint, venue.fingerprint);
    let described = differences(&venue.description, &rpls.description)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    assert_eq!(described.len(), 1, "{described:?}");
    assert!(
        described[0].starts_with("allocation.max_memory_allocation_size: "),
        "{described:?}"
    );
    let allowed = [venue.identity.clone(), rpls.identity.clone()];
    for (device, unread) in [(&venue, Vec::new()), (&rpls, described.clone())] {
        let check = venue_check(&venue, &allowed, &device.identity, &device.description).unwrap();
        assert_eq!(
            check,
            crate::c3::VenueCheck {
                stops: Vec::new(),
                unread
            },
            "{}",
            device.stem
        );
    }
    let mut other = rpls.identity.clone();
    other.device_id = 0xa788;
    let check = venue_check(&venue, &allowed, &other, &rpls.description).unwrap();
    assert_eq!(
        check.stops,
        [format!(
            "identity device_id: {} ({})",
            rpls.identity.device_id, other.device_id
        )]
    );
    let check = venue_check(
        &venue,
        std::slice::from_ref(&venue.identity),
        &rpls.identity,
        &rpls.description,
    )
    .unwrap();
    assert_eq!(
        check.stops,
        [
            format!(
                "identity device: {} ({})",
                venue.identity.device, rpls.identity.device
            ),
            format!(
                "identity device_id: {} ({})",
                venue.identity.device_id, rpls.identity.device_id
            ),
        ]
    );
}
/// The restricted context requires the description it queries through the layer to be the profile
/// as the product reads it, so every member the product reads must come from the file. A
/// requirements document omits what its profile does not require; the layer then reports the
/// device's own value for a member it omits (of a structure the context queries) and a feature it
/// omits as absent, where the product reads the version's guarantee (or none) and a feature the
/// version requires as present. Our probe ran the RTX 5080 restricted to
/// `lunarg-desktop-baseline-2023` as extracted (none of its stated members beyond the device): the
/// context refused all four runs, in exactly these six members. The document lists
/// `VK_EXT_subgroup_size_control` and `VK_EXT_vertex_attribute_divisor` but states none of their
/// properties, and omits `shaderSubgroupExtendedTypes`, which Vulkan 1.2 requires. Three of the six
/// move the translation key, so no narrower comparison would let the run proceed. Its regeneration
/// now states them ([`a_completed_requirements_document_reads_as_before`]), and the product's
/// reading, which this reconstruction compares against, is unchanged.
#[test]
fn the_layer_reports_the_device_value_of_a_member_a_requirements_document_omits() {
    let restriction = ProfileRestriction::committed("lunarg-desktop-baseline-2023").unwrap();
    let native = read_at(&committed("nvidia-rtx5080-laptop"), AT_1_2);
    assert_eq!(check_within(&restriction, &native, AT_1_2), Ok(()));
    let profile = restriction.read_at(AT_1_2);
    assert!(profile.has(VkFeature::ShaderSubgroupExtendedTypes));
    // What the layer reported on the RTX 5080: the profile, but for the device's identity and
    // kept members, the omitted feature absent, and the omitted properties the device's.
    let mut queried = profile.clone();
    queried.driver_id = native.driver_id;
    queried.driver_version = native.driver_version;
    queried.vendor_id = native.vendor_id;
    queried.device_id = native.device_id;
    queried.device_name.clone_from(&native.device_name);
    queried.device_type = native.device_type;
    queried.float_controls.denorm_independence = native.float_controls.denorm_independence;
    queried.float_controls.rounding_independence = native.float_controls.rounding_independence;
    queried.rgba4_opaque_black_swizzled = native.rgba4_opaque_black_swizzled;
    queried
        .features
        .remove(&VkFeature::ShaderSubgroupExtendedTypes);
    queried.limits.max_vertex_attrib_divisor = native.limits.max_vertex_attrib_divisor;
    queried.subgroup.min_size = native.subgroup.min_size;
    queried.subgroup.max_size = native.subgroup.max_size;
    queried.subgroup.max_compute_workgroup_subgroups =
        native.subgroup.max_compute_workgroup_subgroups;
    queried.subgroup.required_size_stages = native.subgroup.required_size_stages;
    let tools = [tool(c"VK_LAYER_KHRONOS_validation"), tool(PROFILES_LAYER)];
    let Err(RestrictionError::Mismatch { fields, .. }) =
        check_restricted(&restriction, &native, &tools, &queried, AT_1_2)
    else {
        panic!("the RTX 5080 reported lunarg-desktop-baseline-2023 as the product reads it");
    };
    let reported = fields.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        reported,
        [
            "features.ShaderSubgroupExtendedTypes: expected listed, queried absent",
            "limits.max_vertex_attrib_divisor: expected 0, queried 4294967295",
            "subgroup.max_compute_workgroup_subgroups: expected 0, queried 32",
            "subgroup.max_size: expected 0, queried 32",
            "subgroup.min_size: expected 0, queried 32",
            "subgroup.required_size_stages: expected 0, queried 16383",
        ]
    );
    // Each of the first three alone moves the translation key; the size-control range does not,
    // since the profile does not enable subgroup size control.
    let key =
        |description: &DeviceDescription| crate::c3::VenueProfile::key_of(description).unwrap();
    let moves = |edit: &dyn Fn(&mut DeviceDescription)| {
        let mut one = profile.clone();
        edit(&mut one);
        key(&one) != key(&profile)
    };
    assert!(moves(&|d| {
        d.features.remove(&VkFeature::ShaderSubgroupExtendedTypes);
    }));
    assert!(moves(&|d| d.limits.max_vertex_attrib_divisor = 4294967295));
    assert!(moves(&|d| d.subgroup.max_compute_workgroup_subgroups = 32));
    assert!(!moves(&|d| {
        d.subgroup.min_size = 32;
        d.subgroup.max_size = 32;
        d.subgroup.required_size_stages = vk::ShaderStageFlags::from_raw(16383);
    }));
}

/// `scripts/regen-device-profiles` completes `lunarg-desktop-baseline-2023` for the layer
/// (`m2v_layer_presentation`): it states the feature its version requires and the five properties
/// the product reads of the extensions it lists, at the values the product reads, so the layer
/// reports the profile as the product reads it on any device. The block changes nothing the product
/// reads: without it the profile reads the same, with the same translation key, so every
/// translation for it, golden, and census cell stays as it was.
#[test]
fn a_completed_requirements_document_reads_as_before() {
    let mut completed = Vec::new();
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        if !restriction.completed_for_layer {
            continue;
        }
        completed.push(stem.clone());
        assert!(restriction.requirements_only, "{stem}");
        // Presentable at its own version, which the context runs at (`roadmap-2022` at 1.3).
        assert_eq!(restriction.presentation_gap(), None, "{stem}");
        let text = std::fs::read_to_string(&restriction.file).unwrap();
        let mut document: Value = serde_json::from_str(&text).unwrap();
        let block = document["capabilities"][LAYER_PRESENTATION_BLOCK].clone();
        document["capabilities"]
            .as_object_mut()
            .unwrap()
            .remove(LAYER_PRESENTATION_BLOCK);
        document["profiles"][&restriction.profile_name]["capabilities"]
            .as_array_mut()
            .unwrap()
            .retain(|name| name.as_str() != Some(LAYER_PRESENTATION_BLOCK));
        let without = DeviceDescription::from_khronos_json(
            &document.to_string(),
            Some(&restriction.profile_name),
        )
        .unwrap();
        assert_eq!(without, restriction.description, "{stem}");
        assert_eq!(
            crate::c3::VenueProfile::key_of(&without).unwrap(),
            crate::c3::VenueProfile::key_of(&restriction.description).unwrap(),
            "{stem}"
        );
        if stem == "lunarg-desktop-baseline-2023" {
            assert_eq!(
                block,
                serde_json::json!({
                    "features": {
                        "VkPhysicalDeviceVulkan12Features": {
                            "subgroupBroadcastDynamicId": true,
                            "shaderSubgroupExtendedTypes": true
                        }
                    },
                    "properties": {
                        "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT": {
                            "minSubgroupSize": 0,
                            "maxSubgroupSize": 0,
                            "maxComputeWorkgroupSubgroups": 0,
                            "requiredSubgroupSizeStages": []
                        },
                        "VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT": {
                            "maxVertexAttribDivisor": 0
                        }
                    }
                })
            );
        }
    }
    assert_eq!(completed, ["lunarg-desktop-baseline-2023", "roadmap-2022"]);
}

/// `ProfileRestriction` `stem` as its regeneration would have written it without its
/// `m2v_layer_presentation` block (a requirements document as extracted).
fn without_block(stem: &str) -> ProfileRestriction {
    let restriction = ProfileRestriction::committed(stem).unwrap();
    let text = std::fs::read_to_string(&restriction.file).unwrap();
    let mut document: Value = serde_json::from_str(&text).unwrap();
    document["capabilities"]
        .as_object_mut()
        .unwrap()
        .remove(LAYER_PRESENTATION_BLOCK);
    document["profiles"][&restriction.profile_name]["capabilities"]
        .as_array_mut()
        .unwrap()
        .retain(|name| name.as_str() != Some(LAYER_PRESENTATION_BLOCK));
    ProfileRestriction::of_document(
        stem,
        restriction.file.clone(),
        document.to_string().as_bytes(),
    )
    .unwrap()
}

/// The core device-property structures a venue capture states, by the minor version that made
/// each core: a context at that version or later queries it.
fn core_minor(structure: &str) -> Option<u32> {
    match structure {
        "VkPhysicalDeviceProperties" => Some(0),
        "VkPhysicalDeviceVulkan11Properties" => Some(1),
        "VkPhysicalDeviceVulkan12Properties" => Some(2),
        "VkPhysicalDeviceVulkan13Properties" => Some(3),
        "VkPhysicalDeviceVulkan14Properties" => Some(4),
        _ => None,
    }
}

/// The members the product reads (`src/device/khronos.rs`) of each extension's device-property
/// structure: a context queries them where the profile lists the extension.
const EXTENSION_MEMBERS: [(&str, &[&str]); 8] = [
    (
        "VK_EXT_subgroup_size_control",
        &[
            "minSubgroupSize",
            "maxSubgroupSize",
            "maxComputeWorkgroupSubgroups",
            "requiredSubgroupSizeStages",
        ],
    ),
    (
        "VK_EXT_vertex_attribute_divisor",
        &["maxVertexAttribDivisor"],
    ),
    (
        "VK_KHR_vertex_attribute_divisor",
        &["maxVertexAttribDivisor", "supportsNonZeroFirstInstance"],
    ),
    ("VK_KHR_maintenance4", &["maxBufferSize"]),
    ("VK_KHR_push_descriptor", &["maxPushDescriptors"]),
    (
        "VK_KHR_portability_subset",
        &["minVertexInputBindingStrideAlignment"],
    ),
    (
        "VK_KHR_acceleration_structure",
        &[
            "maxPerStageDescriptorAccelerationStructures",
            "maxDescriptorSetAccelerationStructures",
        ],
    ),
    (
        "VK_EXT_device_generated_commands",
        &[
            "maxIndirectPipelineCount",
            "maxIndirectSequenceCount",
            "maxIndirectCommandsTokenCount",
            "maxIndirectCommandsTokenOffset",
            "maxIndirectCommandsIndirectStride",
            "supportedIndirectCommandsShaderStages",
            "supportedIndirectCommandsShaderStagesPipelineBinding",
        ],
    ),
];

/// The capability blocks the one profile of `document` lists.
fn profile_blocks(document: &Value) -> Vec<&Value> {
    let profile = document["profiles"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    profile["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| &document["capabilities"][name.as_str().unwrap()])
        .collect()
}

/// What `VK_LAYER_KHRONOS_profiles` reports, as a document the product reads, when it restricts
/// the device a venue's `capture` (which states its device's every member) describes to the
/// profile `restriction`, for a context at Vulkan 1.`minor`: the profile's version, extensions,
/// and every member it states; each format at the last entry the profile's blocks state of it;
/// each feature it omits as absent (the layer's
/// `DEFAULT_FEATURE_VALUES_FALSE`); and the device's own value of each property it omits of a
/// structure the context queries (a core one of the context's version or earlier, or the
/// structure of an extension it lists), including the device's identity.
fn layer_report(restriction: &ProfileRestriction, capture: &Value, minor: u32) -> String {
    let text = std::fs::read_to_string(&restriction.file).unwrap();
    let mut document: Value = serde_json::from_str(&text).unwrap();
    if !restriction.completed_for_layer {
        // `without_block`'s document: the file less the block.
        document["capabilities"]
            .as_object_mut()
            .unwrap()
            .remove(LAYER_PRESENTATION_BLOCK);
        document["profiles"][&restriction.profile_name]["capabilities"]
            .as_array_mut()
            .unwrap()
            .retain(|name| name.as_str() != Some(LAYER_PRESENTATION_BLOCK));
    }
    let mut features = BTreeSet::new();
    let mut properties = BTreeSet::new();
    for block in profile_blocks(&document) {
        for members in block["features"]
            .as_object()
            .into_iter()
            .flat_map(|s| s.values())
        {
            features.extend(members.as_object().unwrap().keys().cloned());
        }
        for members in block["properties"]
            .as_object()
            .into_iter()
            .flat_map(|s| s.values())
        {
            for (member, value) in members.as_object().unwrap() {
                match (member.as_str(), value.as_object()) {
                    ("limits", Some(limits)) => properties.extend(limits.keys().cloned()),
                    _ => {
                        properties.insert(member.clone());
                    }
                }
            }
        }
    }
    let read_by_extension: BTreeSet<&str> = EXTENSION_MEMBERS
        .iter()
        .filter(|(extension, _)| restriction.extension_names.contains(*extension))
        .flat_map(|(_, members)| members.iter().copied())
        .collect();
    let mut device = serde_json::Map::new();
    let (mut omitted_features, mut omitted_properties) =
        (serde_json::Map::new(), serde_json::Map::new());
    for block in profile_blocks(capture) {
        for (structure, members) in block["features"].as_object().into_iter().flatten() {
            for member in members.as_object().unwrap().keys() {
                if !features.contains(member) {
                    omitted_features
                        .entry(structure.clone())
                        .or_insert_with(|| Value::Object(serde_json::Map::new()))[member] =
                        Value::Bool(false);
                }
            }
        }
        for (structure, members) in block["properties"].as_object().into_iter().flatten() {
            let core = core_minor(structure).is_some_and(|core| core <= minor);
            for (member, value) in members.as_object().unwrap() {
                let entry = omitted_properties
                    .entry(structure.clone())
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                if let ("limits", Some(limits)) = (member.as_str(), value.as_object()) {
                    for (limit, value) in limits {
                        if core && !properties.contains(limit) {
                            let nested = entry
                                .as_object_mut()
                                .unwrap()
                                .entry("limits")
                                .or_insert_with(|| Value::Object(serde_json::Map::new()));
                            nested[limit] = value.clone();
                        }
                    }
                } else if (core || read_by_extension.contains(member.as_str()))
                    && !properties.contains(member)
                {
                    entry[member] = value.clone();
                }
            }
        }
    }
    device.insert("features".into(), Value::Object(omitted_features));
    device.insert("properties".into(), Value::Object(omitted_properties));
    // The layer keeps one block's entry of a format, the last of the profile's blocks that states
    // it, where the product's reader takes the union of every block's flags (`roadmap-2022`'s
    // storage-only conditional entries replaced its required ones).
    let mut formats = serde_json::Map::new();
    for block in profile_blocks(&document) {
        for (format, entry) in block["formats"].as_object().into_iter().flatten() {
            formats.insert(format.clone(), entry.clone());
        }
    }
    for block in document["capabilities"]
        .as_object_mut()
        .unwrap()
        .values_mut()
    {
        block.as_object_mut().unwrap().remove("formats");
    }
    device.insert("formats".into(), Value::Object(formats));
    document["capabilities"]["m2v_test_device"] = Value::Object(device);
    document["profiles"][&restriction.profile_name]["capabilities"]
        .as_array_mut()
        .unwrap()
        .push(Value::from("m2v_test_device"));
    document.to_string()
}

/// What a context at Vulkan 1.`minor` restricted to `restriction` on the venue whose capture is
/// `venue` decides: its first check, that the device has what the profile states
/// ([`check_within`]), then, after the layer reports [`layer_report`], its check of that report
/// against the profile as the product reads it ([`check_restricted`]).
fn presented(
    restriction: &ProfileRestriction,
    venue: &str,
    minor: u32,
) -> Result<DeviceDescription, RestrictionError> {
    let at = ApiVersion { major: 1, minor };
    let file = crate::engine::profiles_dir().join(format!("{venue}.json"));
    let capture: Value = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
    let report = layer_report(restriction, &capture, minor);
    let mut queried = read_at(
        &DeviceDescription::from_khronos_json(&report, Some(&restriction.profile_name)).unwrap(),
        at,
    );
    // The context hides what the layer adds that the profile does not describe (`hidden`).
    queried.khr_format_feature_flags2 &= restriction.states_format_flags2;
    let native = read_at(&committed(venue), at);
    check_within(restriction, &native, at)?;
    let tools = [tool(c"VK_LAYER_KHRONOS_validation"), tool(PROFILES_LAYER)];
    check_restricted(restriction, &native, &tools, &queried, at)
}

/// A completed requirements document is presented as the product reads it on every venue, at its
/// own version (the context a restricted run of it opens): modelled from each venue's capture, the
/// layer's report of it passes the restricted context's check. The model is the evidence's: for
/// `lunarg-desktop-baseline-2023` as extracted it gives exactly the six members the RTX 5080's
/// context reported, and the completed one matched on RADV, ANV, and the RTX 5080. `roadmap-2022`
/// as extracted omits the core members the product reads that the layer then reports at each
/// device's value; completed, it omits none.
#[test]
fn a_completed_requirements_document_presents_as_the_product_reads_it_on_every_venue() {
    let venues = crate::c3::Venues::load(&crate::c3::Venues::committed_path()).unwrap();
    // Every device a venue allows, by its own capture (the Intel venue's two).
    let venue_profiles: BTreeSet<String> = venues
        .venues
        .iter()
        .flat_map(|venue| venue.devices.iter().map(|device| device.profile.clone()))
        .collect();
    assert_eq!(venue_profiles.len(), 7, "{venue_profiles:?}");
    let Err(RestrictionError::Mismatch { fields, .. }) = presented(
        &without_block("lunarg-desktop-baseline-2023"),
        "nvidia-rtx5080-laptop",
        2,
    ) else {
        panic!("the model presented lunarg-desktop-baseline-2023 as extracted");
    };
    assert_eq!(
        fields.iter().map(ToString::to_string).collect::<Vec<_>>(),
        [
            "features.ShaderSubgroupExtendedTypes: expected listed, queried absent",
            "limits.max_vertex_attrib_divisor: expected 0, queried 4294967295",
            "subgroup.max_compute_workgroup_subgroups: expected 0, queried 32",
            "subgroup.max_size: expected 0, queried 32",
            "subgroup.min_size: expected 0, queried 32",
            "subgroup.required_size_stages: expected 0, queried 16383",
        ]
    );
    let Err(RestrictionError::Mismatch { fields, .. }) =
        presented(&without_block("roadmap-2022"), "nvidia-rtx5080-laptop", 3)
    else {
        panic!("the model presented roadmap-2022 as extracted");
    };
    assert!(
        fields
            .iter()
            .any(|field| field.path == "limits.max_image_dimension_3d"),
        "{fields:?}"
    );
    let mut takes = BTreeMap::new();
    for stem in ["lunarg-desktop-baseline-2023", "roadmap-2022"] {
        let restriction = ProfileRestriction::committed(stem).unwrap();
        let minor = restriction.context_version().minor;
        for venue in &venue_profiles {
            match presented(&restriction, venue, minor) {
                Ok(_) => takes
                    .entry(stem)
                    .or_insert_with(Vec::new)
                    .push(venue.as_str()),
                // The device lacks what the profile states: a typed refusal, before the layer
                // reports anything.
                Err(RestrictionError::Exceeds { .. }) => {}
                Err(error) => panic!("{stem} on {venue}: {error}"),
            }
        }
    }
    // MoltenVK lacks two of lunarg's members and lavapipe 15; at Vulkan 1.3 every venue but
    // MoltenVK has what `roadmap-2022` states.
    assert_eq!(
        takes["lunarg-desktop-baseline-2023"],
        [
            "amd-780m-radv",
            "intel-irisxe-rplp-anv",
            "intel-uhd770-rpls-anv",
            "nvidia-rtx5080-laptop"
        ]
    );
    assert_eq!(
        takes["roadmap-2022"],
        [
            "amd-780m-radv",
            "gate-lavapipe",
            "gate-lavapipe-x86_64",
            "intel-irisxe-rplp-anv",
            "intel-uhd770-rpls-anv",
            "nvidia-rtx5080-laptop"
        ]
    );
}

/// A requirements document the layer cannot present as the product reads it, on any device, is
/// refused by the plan and the run by the same check before any device is read: it is that one
/// gap, a tooling gap, and
/// `restrict_devices_to` and a batch (`check_presentable`) refuse it up front. One its regeneration
/// did not complete for the layer is one (the safety net), named with the version the context would
/// run at. A completed one is presentable at its own version, which the context runs at, and
/// whether a device has what it states is the first check's, at that version:
/// `lunarg-desktop-baseline-2023` states nothing beyond the RTX 5080 and format features beyond
/// lavapipe's, and `roadmap-2022`, Vulkan 1.3, nothing beyond either. Every other committed
/// profile states every member the product reads: a device report or capture its device's every
/// member, a derived profile its base version's every member and the ones the base leaves open.
#[test]
fn an_unpresentable_requirements_document_is_refused_before_any_device_is_read() {
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        let requirements =
            ["lunarg-desktop-baseline-2023", "roadmap-2022"].contains(&stem.as_str());
        assert_eq!(restriction.requirements_only, requirements, "{stem}");
        assert_eq!(restriction.completed_for_layer, requirements, "{stem}");
        assert_eq!(restriction.presentation_gap(), None, "{stem}");
    }
    let nvidia = committed("nvidia-rtx5080-laptop");
    let lavapipe = committed("gate-lavapipe-x86_64");
    for (stem, minor) in [("lunarg-desktop-baseline-2023", 2), ("roadmap-2022", 3)] {
        let uncompleted = without_block(stem);
        assert_eq!(uncompleted.context_version().minor, minor, "{stem}");
        let gap = uncompleted.presentation_gap().unwrap();
        assert!(
            gap.starts_with(&format!(
                "tooling gap: the executor's Vulkan 1.{minor} context cannot present \
                 requirements document {stem} as the product reads it (provenance source \
                 khronos, not completed by m2v_layer_presentation: "
            )),
            "{gap}"
        );
    }
    presentable("roadmap-2022").unwrap();
    super::super::check_presentable("roadmap-2022").unwrap();
    for device in [&nvidia, &lavapipe] {
        assert_eq!(gaps_of("roadmap-2022", device), Vec::<String>::new());
    }

    super::super::check_presentable("lunarg-desktop-baseline-2023").unwrap();
    assert_eq!(
        gaps_of("lunarg-desktop-baseline-2023", &nvidia),
        Vec::<String>::new()
    );
    let beyond = gaps_of("lunarg-desktop-baseline-2023", &lavapipe);
    assert!(
        beyond
            .iter()
            .any(|gap| gap.starts_with("formats.VK_FORMAT_R8_SNORM.optimal_tiling 0xdd83")),
        "{beyond:?}"
    );
    assert_eq!(gaps_of("vk12-floor", &lavapipe), Vec::<String>::new());
    super::super::check_presentable("vk12-floor").unwrap();
}
