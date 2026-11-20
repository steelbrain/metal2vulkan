//! The profiles layer's checks over the committed profiles, device-free.

use super::*;
use crate::engine::{committed_profiles, load_description};
use metal2vulkan::device::{FormatProperties, FormatSupport};
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::vk;

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
    assert!(unknown.contains("runtime-floor"), "{unknown}");
    assert!(restrict_devices_to("no-such-profile").is_err());
}

/// The layer lists every extension promoted to core at or below P's version; the context hides
/// the ones P does not list. `intel-arl-anv` (1.4) does not list
/// `VK_EXT_texture_compression_astc_hdr` (promoted in 1.3), which the device lacks. It lists
/// `VK_KHR_format_feature_flags2` but, a gpuinfo report without them, states no
/// `VkFormatProperties3`, so that is hidden too. A `vulkaninfo` capture with its supplement
/// (`gate-lavapipe`) states them, and keeps the extension.
#[test]
fn the_extensions_a_profile_does_not_list_or_describe_are_hidden() {
    let restriction = ProfileRestriction::committed("intel-arl-anv").unwrap();
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
/// optional features it lacks (geometry shaders, 64-bit floats, 64-bit buffer atomics, and
/// formatless storage-image reads) names each of them on the floor.
#[test]
fn a_profile_beyond_the_device_is_named_member_by_member() {
    let lacked = [
        VkFeature::GeometryShader,
        VkFeature::ShaderFloat64,
        VkFeature::ShaderBufferInt64Atomics,
        VkFeature::ShaderStorageImageReadWithoutFormat,
    ];
    let floor = read_at(&committed("runtime-floor"), AT_1_2);
    for feature in lacked {
        assert!(!floor.features.contains(&feature), "{feature:?}");
    }
    let mut beyond = floor.clone();
    beyond.features.extend(lacked);
    let gaps = exceeds(&beyond, &floor);
    for feature in lacked {
        assert!(
            gaps.contains(&format!("feature {feature:?}")),
            "{feature:?}: {gaps:?}"
        );
    }
    // The committed x86 lavapipe has more than the floor; the reverse does not hold.
    let lavapipe = read_at(&committed("lavapipe"), AT_1_2);
    assert!(exceeds(&floor, &lavapipe).is_empty());
    assert!(!exceeds(&lavapipe, &floor).is_empty());
}

/// The floor is Vulkan 1.2 core, which every Vulkan 1.2 device has: Intel's Iris Xe on ANV, which
/// reads no storage image without its format, takes it as the other Linux venue captures do.
#[test]
fn every_device_capture_takes_the_floor() {
    let floor = read_at(&committed("runtime-floor"), AT_1_2);
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
    let floor = committed("runtime-floor");
    assert!(floor.api_version > AT_1_2);
    let read = read_at(&floor, AT_1_2);
    assert_eq!(read.api_version, AT_1_2);
    assert!(read.features.is_subset(&floor.features));
    assert!(read.features.len() < floor.features.len());
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
    // Less the formats the product never asks about: the floor states the scaled vertex formats,
    // which no Metal format maps to.
    assert!(read.formats.formats.len() < floor.formats.formats.len());
    assert!(!read.formats.formats.contains_key(&vk::Format::R8_USCALED));
}

#[test]
fn a_restricted_context_runs_only_on_a_device_that_has_and_reports_its_profile() {
    let restriction = ProfileRestriction::committed("runtime-floor").unwrap();
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
    // A profile is every driver's: the profile on lavapipe's identity and on SwiftShader's is one.
    let profile_of = |description: &DeviceDescription| {
        metal2vulkan::device::Device::from_queries(
            description.clone(),
            metal2vulkan::device::DevicePolicy::default(),
        )
        .expect("device")
        .profile()
        .fingerprint()
    };
    let mut swiftshader_native = native.clone();
    swiftshader_native.driver_id = Some(vk::DriverId::GOOGLE_SWIFTSHADER);
    let mut swiftshader_queried = queried.clone();
    swiftshader_queried.driver_id = swiftshader_native.driver_id;
    let on_swiftshader = check(&swiftshader_native, &tools, &swiftshader_queried).unwrap();
    assert_eq!(profile_of(&on_swiftshader), profile_of(&translated));
    assert!(matches!(
        check(&native, &tools[..1], &queried),
        Err(RestrictionError::Inactive { .. })
    ));
    // SwiftShader takes the floor (Vulkan 1.2's required features and minimum limits); a device
    // short of one of its limits does not.
    let swiftshader = read_at(&committed("runtime-floor"), AT_1_2);
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

/// Two Raptor Lake Xe-LP devices, one translation key: the Iris Xe of RPL-P (ANV 26.2.6), which
/// `intel-irisxe-rplp-anv` was captured on, and the UHD Graphics 770 of RPL-S (ANV 26.2.4). Their
/// committed captures describe the same capabilities but for `maxMemoryAllocationSize`, which the
/// driver sizes from its host's memory and no translation reads (`DeviceDescription::allocation`),
/// so the two derive one key. The venue allows the RPL-P alone: the RPL-P runs there; the RPL-S
/// stops on its name, device ID and driver build, and its allocation size is read as unread.
#[test]
fn the_intel_venue_allows_the_rplp_alone_and_stops_the_rpls() {
    let venue = crate::c3::VenueProfile::committed("intel-irisxe-rplp-anv").unwrap();
    let rpls = crate::c3::VenueProfile::committed("intel-uhd770-rpls-anv").unwrap();
    let described = differences(&venue.description, &rpls.description)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    assert_eq!(described.len(), 1, "{described:?}");
    assert!(
        described[0].starts_with("allocation.max_memory_allocation_size: "),
        "{described:?}"
    );
    assert_eq!(rpls.fingerprint, venue.fingerprint);
    let allowed = std::slice::from_ref(&venue.identity);
    let check = venue_check(&venue, allowed, &venue.identity, &venue.description).unwrap();
    assert_eq!(
        check,
        crate::c3::VenueCheck {
            stops: Vec::new(),
            unread: Vec::new()
        }
    );
    let check = venue_check(&venue, allowed, &rpls.identity, &rpls.description).unwrap();
    let identity = [
        format!(
            "identity device: {} ({})",
            venue.identity.device, rpls.identity.device
        ),
        format!(
            "identity device_id: {} ({})",
            venue.identity.device_id, rpls.identity.device_id
        ),
        format!(
            "identity driver_version: {} ({})",
            venue.identity.driver_version, rpls.identity.driver_version
        ),
        format!(
            "identity driver_info: {} ({})",
            venue.identity.driver_info, rpls.identity.driver_info
        ),
    ];
    assert_eq!(check.stops, identity, "{check:?}");
    // Its key is the venue's, so its host's allocation size is unread, not a stop.
    assert_eq!(check.unread, described, "{check:?}");
}
