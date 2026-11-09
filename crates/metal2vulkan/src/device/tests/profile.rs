//! `Device::new`, profile derivation, the fingerprint, and `requirements ⊆ profile`.

use super::*;
use crate::device::profile::{BufferAccess, Containment, GuestIsolation, Independence, Robustness};
use crate::device::requirements::{Requirements, StageCreateFlag, VkLimit, VkProperty};
use crate::error::{Capability, Feature, Needs};

pub(crate) fn device(stem: &str) -> Device {
    Device::new(profile(stem), DevicePolicy::default()).expect("committed profile derives")
}

#[test]
fn every_committed_profile_derives() {
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let dev = Device::new(desc.clone(), DevicePolicy::default())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let f = dev.profile().fields();
        assert_eq!(f.api_version, desc.api_version, "{stem}");
        assert_eq!(f.isolation, GuestIsolation::On, "{stem}");
        assert!(
            matches!(f.containment, Containment::On { .. }),
            "{stem}: {:?}",
            f.containment
        );
        // Deterministic.
        let again = Device::new(desc, DevicePolicy::default()).unwrap();
        assert_eq!(dev.profile(), again.profile(), "{stem}");
        assert_eq!(dev.profile().fingerprint(), again.profile().fingerprint());
        // A profile satisfies the requirements of everything it enabled.
        let all = Requirements {
            features: f.enabled.features.clone(),
            extensions: f.enabled.extensions.clone(),
            properties: f.enabled.properties.clone(),
            denorm_independence: f.float_controls.denorm_independence,
            rounding_independence: f.float_controls.rounding_independence,
            ..Requirements::default()
        };
        assert_eq!(dev.profile().check(&all), Ok(()), "{stem}");
        // The extended dynamic states a pipeline makes are exactly what it requires.
        assert_eq!(
            f.dynamic_vertex_stride,
            f.enabled
                .features
                .contains(&VkFeature::ExtendedDynamicState),
            "{stem}"
        );
    }
}

/// `occlusionQueryPrecise` is read from each committed profile whose source states it, and enabled
/// there: every venue's source does; the derived Vulkan 1.2 floor, whose base
/// (`VP_LUNARG_minimum_requirements_1_2`) does not require it, leaves it unstated.
#[test]
fn occlusion_query_precise_follows_each_source() {
    for (stem, text) in committed_profiles() {
        let stated = text.contains("\"occlusionQueryPrecise\": true");
        assert_eq!(stated, stem != "vk12-floor", "{stem}");
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        assert_eq!(
            desc.features.contains(&VkFeature::OcclusionQueryPrecise),
            stated,
            "{stem}"
        );
        let dev = Device::new(desc, DevicePolicy::default()).unwrap();
        let enabled = &dev.profile().fields().enabled.features;
        assert_eq!(
            enabled.contains(&VkFeature::OcclusionQueryPrecise),
            stated,
            "{stem}"
        );
    }
}

/// `VK_EXT_depth_bias_control`'s three named features are read from each committed profile that
/// lists the extension, member by member as its source states them, and enabled there with the
/// extension; a profile whose source does not list it has none.
#[test]
fn depth_bias_control_follows_each_source() {
    use crate::device::requirements::VkExtension;
    let features = [
        ("depthBiasControl", VkFeature::DepthBiasControl),
        ("floatRepresentation", VkFeature::FloatRepresentation),
        ("depthBiasExact", VkFeature::DepthBiasExact),
    ];
    let mut listed = 0;
    for (stem, text) in committed_profiles() {
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let mut stated = None;
        find_structure(
            &json,
            "VkPhysicalDeviceDepthBiasControlFeaturesEXT",
            &mut stated,
        );
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let has_extension = desc.extensions.contains(&VkExtension::ExtDepthBiasControl);
        assert_eq!(has_extension, stated.is_some(), "{stem}");
        let dev = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
        let enabled = &dev.profile().fields().enabled;
        for (member, feature) in features {
            let want = stated
                .as_ref()
                .is_some_and(|s| s.get(member) == Some(&serde_json::Value::Bool(true)));
            assert_eq!(desc.features.contains(&feature), want, "{stem} {member}");
            assert_eq!(enabled.features.contains(&feature), want, "{stem} {member}");
        }
        assert_eq!(
            enabled
                .extensions
                .contains(&VkExtension::ExtDepthBiasControl),
            has_extension,
            "{stem}"
        );
        listed += usize::from(has_extension);
    }
    // RADV, ANV, NVIDIA and lavapipe list it; MoltenVK, SwiftShader and the requirements
    // documents do not.
    assert!(
        listed >= 10,
        "{listed} profiles list VK_EXT_depth_bias_control"
    );
}

/// The first object named `name` in `v`, depth first.
fn find_structure(v: &serde_json::Value, name: &str, out: &mut Option<serde_json::Value>) {
    if out.is_some() {
        return;
    }
    match v {
        serde_json::Value::Object(map) => {
            for (k, child) in map {
                if k == name {
                    *out = Some(child.clone());
                    return;
                }
                find_structure(child, name, out);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                find_structure(child, name, out);
            }
        }
        _ => {}
    }
}

#[test]
fn memory_profiles_follow_bda_and_int64() {
    // roadmap-2022 requires device addresses but not shaderInt64.
    for stem in ["vk12-floor", "roadmap-2022"] {
        assert_eq!(
            device(stem).profile().fields().buffer_access,
            BufferAccess::Floor,
            "{stem}"
        );
    }
    for stem in [
        "moltenvk-m3",
        "nvidia-rtx50-linux",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "lavapipe",
    ] {
        assert_eq!(
            device(stem).profile().fields().buffer_access,
            BufferAccess::Addressed,
            "{stem}"
        );
        assert_eq!(device(stem).profile().addressed_needs(), vec![], "{stem}");
    }
}

/// A Profile F refusal names what the device lacks of Profile A, never a feature it has.
#[test]
fn addressed_needs_name_only_the_missing_features() {
    let bda = Capability::Feature(VkFeature::BufferDeviceAddress);
    let int64 = Capability::Feature(VkFeature::ShaderInt64);
    assert_eq!(
        device("vk12-floor").profile().addressed_needs(),
        vec![bda, int64.clone()]
    );
    // roadmap-2022 has device addresses; only the 64-bit integers are missing.
    assert_eq!(
        device("roadmap-2022").profile().addressed_needs(),
        vec![int64]
    );
}

#[test]
fn vk12_floor_derives_a_floor_profile() {
    let dev = device("vk12-floor");
    let f = dev.profile().fields();
    assert_eq!(
        f.containment,
        Containment::On {
            robust: Robustness::Rba
        }
    );
    assert!(f.descriptors.uniform_buffer_standard_layout);
    assert!(!f.leaf.vertex_stores_and_atomics);
    assert!(!f.leaf.fragment_stores_and_atomics);
    assert!(!f.dynamic_vertex_stride);
    // The Vulkan 1.2 Required Format Support tables: what they require unconditionally is
    // supported, and a usage they leave conditional is refused (a typed Format capability).
    assert!(!f.vertex_formats.formats.is_empty());
    assert!(!f.storage_formats.formats.is_empty());
    assert!(!f.image_formats.formats.is_empty());
    let req = |format, usage| Requirements {
        formats: [(format, usage)].into(),
        ..Requirements::default()
    };
    assert_eq!(
        dev.profile()
            .check(&req(vk::Format::R32_UINT, FormatUsage::StorageImage)),
        Ok(())
    );
    assert_eq!(
        dev.profile()
            .check(&req(vk::Format::R8_UNORM, FormatUsage::StorageImage)),
        Err(Capability::Format {
            format: vk::Format::R8_UNORM,
            usage: FormatUsage::StorageImage
        })
    );
    assert!(!f.subgroup.kernel.pinnable);
    assert_eq!(f.subgroup.kernel.api_size, 1);
}

/// A vertex function writes `Layer` and `ViewportIndex` through the two Vulkan 1.2 features
/// alone: `VK_EXT_shader_viewport_index_layer` without them grants neither, since at the 1.2
/// floor a module declares the features' capabilities and the core never enables the extension.
#[test]
fn vertex_layer_and_viewport_outputs_follow_the_features_not_the_extension() {
    let mut desc = profile("moltenvk-m3");
    assert!(desc
        .extensions
        .contains(&VkExtension::ExtShaderViewportIndexLayer));
    let full = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
    let f = full.profile().fields();
    assert!(f.fragment.output_layer && f.fragment.output_viewport_index);
    assert!(desc.features.remove(&VkFeature::ShaderOutputLayer));
    assert!(desc.features.remove(&VkFeature::ShaderOutputViewportIndex));
    let lacking = Device::new(desc, DevicePolicy::default()).unwrap();
    let f = lacking.profile().fields();
    assert!(!f.fragment.output_layer);
    assert!(!f.fragment.output_viewport_index);
}

#[test]
fn robustness2_selects_the_rba2_encoding() {
    assert_eq!(
        device("amd-rdna3-radv").profile().fields().containment,
        Containment::On {
            robust: Robustness::Rba2
        }
    );
    // MoltenVK exposes VK_EXT_robustness2 but reports robustBufferAccess2 = false.
    let f = device("moltenvk-m3");
    assert_eq!(
        f.profile().fields().containment,
        Containment::On {
            robust: Robustness::Rba
        }
    );
    let m = f.profile().fields();
    assert!(m.dynamic_vertex_stride);
    assert!(m.descriptors.push_descriptor);
    assert!(f
        .profile()
        .supports_format(vk::Format::R8G8B8A8_UNORM, FormatUsage::SampledImage));
    assert!(!f
        .profile()
        .supports_format(vk::Format::R8G8B8A8_USCALED, FormatUsage::SampledImage));
    assert!(f
        .profile()
        .supports_format(vk::Format::R8G8B8A8_USCALED, FormatUsage::VertexBuffer));
}

#[test]
fn below_1_2_and_contradictory_policies_are_refused() {
    let mut desc = profile("vk12-floor");
    desc.api_version = ApiVersion { major: 1, minor: 1 };
    match Device::new(desc, DevicePolicy::default()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VulkanVersion);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::ApiVersion(ApiVersion::V1_2)]])
            );
        }
        other => panic!("{other:?}"),
    }
    let mut unsafe_on = DevicePolicy {
        containment: ContainmentPolicy::UnsafeOff,
        ..DevicePolicy::default()
    };
    assert!(matches!(
        Device::new(profile("vk12-floor"), unsafe_on),
        Err(Error::InvalidDescriptor(
            DescriptorError::DeviceDescription { .. }
        ))
    ));
    unsafe_on.isolation = GuestIsolation::Off;
    let dev = Device::new(profile("vk12-floor"), unsafe_on).unwrap();
    assert_eq!(dev.profile().fields().containment, Containment::UnsafeOff);
}

#[test]
fn fingerprint_covers_every_policy_field() {
    let desc = profile("lavapipe");
    let base = DevicePolicy::default();
    let fp = |p: DevicePolicy| {
        Device::new(desc.clone(), p)
            .unwrap()
            .profile()
            .fingerprint()
    };
    let reference = fp(base);
    let mut variants = Vec::new();
    let mut p = base;
    p.isolation = GuestIsolation::Off;
    variants.push(("isolation", p));
    p.containment = ContainmentPolicy::UnsafeOff;
    variants.push(("containment", p));
    let mut seen = vec![reference];
    for (name, p) in variants {
        let h = fp(p);
        assert!(!seen.contains(&h), "{name} did not change the fingerprint");
        seen.push(h);
    }
    // Description-derived fields are covered too.
    let mut fewer = desc.clone();
    fewer.features.remove(&VkFeature::ShaderInt8);
    assert_ne!(
        Device::new(fewer, base).unwrap().profile().fingerprint(),
        reference
    );
    let mut limits = desc;
    limits.limits.max_texel_offset += 1;
    assert_ne!(
        Device::new(limits, base).unwrap().profile().fingerprint(),
        reference
    );
}

/// The fingerprint streams the `Debug` rendering into the hash; it must equal the hash of the
/// materialized rendering, so every fingerprint (and every key and cache built on one) is unchanged
/// by streaming.
#[test]
fn streamed_fingerprint_equals_the_hash_of_the_rendering() {
    use sha2::{Digest, Sha256};
    for (stem, _) in committed_profiles() {
        let dev = device(&stem);
        let mut h = Sha256::new();
        h.update(b"metal2vulkan/device-profile/v1\0");
        h.update(format!("{:?}", dev.profile().fields()).as_bytes());
        let materialized: crate::cache::Hash256 = h.finalize().into();
        assert_eq!(dev.profile().fingerprint(), materialized, "{stem}");
    }
}

/// Two committed profiles share a fingerprint only when their descriptions differ in nothing a
/// translation reads: at most in the device's identity (vendor, device, driver version, type) and
/// its allocation sizes, as two captures of one GPU family on one driver do
/// (`intel-irisxe-rplp-anv` and `intel-uhd770-rpls-anv`, the one Intel venue), two of one GPU
/// under its two `deviceID`s (`amd-780m-hawkpoint-radv` and `amd-780m-radv`, the Radeon 780M of
/// Hawk Point and of Phoenix).
#[test]
fn fingerprints_distinguish_committed_profiles() {
    let translated = |stem: &str| {
        let mut d = profile(stem);
        (d.vendor_id, d.device_id, d.driver_version, d.device_type) = (0, 0, 0, None);
        d.device_name.clear();
        d.allocation = AllocationLimits::default();
        d
    };
    let mut seen = std::collections::BTreeMap::new();
    let mut shared = Vec::new();
    for (stem, _) in committed_profiles() {
        let h = device(&stem).profile().fingerprint();
        if let Some(other) = seen.insert(h, stem.clone()) {
            assert_eq!(
                translated(&stem),
                translated(&other),
                "{stem} and {other} share a fingerprint but describe different translation inputs"
            );
            shared.push((other, stem));
        }
    }
    assert_eq!(
        shared,
        [
            (
                "amd-780m-hawkpoint-radv".to_string(),
                "amd-780m-radv".to_string()
            ),
            (
                "intel-irisxe-rplp-anv".to_string(),
                "intel-uhd770-rpls-anv".to_string()
            ),
        ],
        "the committed profiles that share a fingerprint"
    );
}

/// The allocation sizes are no profile field: they follow the machine's installed memory on
/// integrated GPUs, and no translation reads them, so a description that differs only in them
/// derives the same profile and fingerprint on every committed profile, while the Metal-facing
/// `maxBufferLength` of a device that reaches buffers by address follows them, as Metal's does.
#[test]
fn allocation_sizes_are_no_profile_field() {
    for (stem, _) in committed_profiles() {
        let desc = profile(&stem);
        let base = device(&stem);
        for (memory, buffer) in [
            (1 << 30, 0),
            (50_382_059_520, 1 << 32),
            (9_534_832_640, 9_534_832_640),
        ] {
            let mut other = desc.clone();
            other.allocation = AllocationLimits::new(memory, buffer);
            let dev = Device::new(other, DevicePolicy::default()).unwrap();
            assert_eq!(dev.profile(), base.profile(), "{stem}");
            assert_eq!(
                dev.profile().fingerprint(),
                base.profile().fingerprint(),
                "{stem}"
            );
            assert_eq!(
                dev.allocation(),
                &AllocationLimits::new(memory, buffer),
                "{stem}"
            );
            let length = dev.metal_properties().max_buffer_length;
            if base.profile().fields().buffer_access == BufferAccess::Addressed {
                let want = match buffer {
                    0 => memory,
                    cap => cap.min(memory),
                };
                assert_eq!(length, want, "{stem}");
            } else {
                assert_eq!(length, base.metal_properties().max_buffer_length, "{stem}");
            }
        }
    }
}

#[test]
fn check_reports_the_first_missing_capability() {
    let dev = device("vk12-floor");
    let p = dev.profile();
    assert_eq!(p.check(&Requirements::default()), Ok(()));
    let need = |r: Requirements| p.check(&r);
    assert_eq!(
        need(Requirements {
            api_version: Some(ApiVersion::V1_3),
            ..Requirements::default()
        }),
        Err(Capability::ApiVersion(ApiVersion::V1_3))
    );
    assert_eq!(
        need(Requirements {
            features: [VkFeature::RobustBufferAccess, VkFeature::ShaderInt64].into(),
            ..Requirements::default()
        }),
        Err(Capability::Feature(VkFeature::ShaderInt64))
    );
    assert_eq!(
        need(Requirements {
            extensions: [VkExtension::ExtRobustness2].into(),
            ..Requirements::default()
        }),
        Err(Capability::Extension(VkExtension::ExtRobustness2))
    );
    assert_eq!(
        need(Requirements {
            properties: [VkProperty::SubgroupQuadOperationsInAllStages].into(),
            ..Requirements::default()
        }),
        Err(Capability::Property(
            VkProperty::SubgroupQuadOperationsInAllStages
        ))
    );
    // Max limits must be at least, alignments at most, the requirement.
    let limit = |l: VkLimit, v: u64| {
        need(Requirements {
            limits: [(l, v)].into(),
            ..Requirements::default()
        })
    };
    assert_eq!(limit(VkLimit::MaxComputeSharedMemorySize, 16384), Ok(()));
    assert_eq!(
        limit(VkLimit::MaxComputeSharedMemorySize, 32768),
        Err(Capability::Limit {
            limit: VkLimit::MaxComputeSharedMemorySize,
            required: 32768,
            available: 16384
        })
    );
    assert_eq!(limit(VkLimit::MinStorageBufferOffsetAlignment, 256), Ok(()));
    assert!(limit(VkLimit::MinStorageBufferOffsetAlignment, 64).is_err());
    assert_eq!(limit(VkLimit::MinTexelOffset, 8), Ok(()));
    assert!(limit(VkLimit::MinTexelOffset, 9).is_err());
    assert_eq!(limit(VkLimit::MaxComputeWorkGroupSize(2), 64), Ok(()));
    assert!(limit(VkLimit::MaxComputeWorkGroupSize(2), 65).is_err());
    assert_eq!(limit(VkLimit::MaxSamplerLodBias, 2), Ok(()));
    assert!(limit(VkLimit::MaxSamplerLodBias, 3).is_err());
    // Float-controls independence (ported from src/vulkan/float_controls.rs).
    assert_eq!(
        need(Requirements {
            denorm_independence: Independence::Only32Bit,
            ..Requirements::default()
        }),
        Err(Capability::Property(
            VkProperty::DenormIndependence32BitOnly
        ))
    );
    assert_eq!(
        need(Requirements {
            rounding_independence: Independence::All,
            ..Requirements::default()
        }),
        Err(Capability::Property(VkProperty::RoundingIndependenceAll))
    );
    // Stage flags need their enabling features.
    assert_eq!(
        need(Requirements {
            stage_flags: [StageCreateFlag::RequireFullSubgroups].into(),
            ..Requirements::default()
        }),
        Err(Capability::Feature(VkFeature::ComputeFullSubgroups))
    );
}

#[test]
fn independence_admits_weaker_needs() {
    use Independence::{All, None, Only32Bit};
    for (have, need, ok) in [
        (None, None, true),
        (None, Only32Bit, false),
        (None, All, false),
        (Only32Bit, None, true),
        (Only32Bit, Only32Bit, true),
        (Only32Bit, All, false),
        (All, None, true),
        (All, Only32Bit, true),
        (All, All, true),
    ] {
        assert_eq!(have.admits(need), ok, "{have:?} admits {need:?}");
    }
}

#[test]
fn device_is_send_sync_and_cheap_to_clone() {
    fn send_sync<T: Send + Sync + Clone>() {}
    send_sync::<Device>();
    send_sync::<DeviceDescription>();
    send_sync::<DevicePolicy>();
}

/// The storage-without-format usages: per format and per image or buffer from
/// `VkFormatFeatureFlags2` on a device that reports it, with the image bits the specification
/// implies for its listed formats; the device's features for every storage format otherwise.
#[test]
fn storage_without_format_follows_each_format() {
    use vk::FormatFeatureFlags2 as F;
    use FormatUsage as U;
    let usages = |dev: &Device, format: vk::Format| -> Vec<FormatUsage> {
        [
            U::StorageImageReadWithoutFormat,
            U::StorageImageWriteWithoutFormat,
            U::StorageTexelBufferReadWithoutFormat,
            U::StorageTexelBufferWriteWithoutFormat,
        ]
        .into_iter()
        .filter(|&u| dev.profile().supports_format(format, u))
        .collect()
    };
    let storage = |optimal: F, buffer: F| crate::device::FormatProperties {
        optimal_tiling: F::STORAGE_IMAGE | optimal,
        buffer: F::STORAGE_TEXEL_BUFFER | buffer,
        ..Default::default()
    };
    // The floor states neither formatless feature: the test grants both.
    let mut desc = profile("vk12-floor");
    desc.features
        .insert(VkFeature::ShaderStorageImageReadWithoutFormat);
    desc.features
        .insert(VkFeature::ShaderStorageImageWriteWithoutFormat);
    assert!(!desc.has_format_feature_flags2());
    desc.formats.formats.clear();
    desc.formats
        .formats
        .insert(vk::Format::R32_UINT, storage(F::empty(), F::empty()));
    desc.formats
        .formats
        .insert(vk::Format::B8G8R8A8_UNORM, storage(F::empty(), F::empty()));
    // Without `VkFormatProperties3` the features decide, images and texel buffers alike, for the
    // formats of the specification's list only: `B8G8R8A8_UNORM` is not on it.
    let all = vec![
        U::StorageImageReadWithoutFormat,
        U::StorageImageWriteWithoutFormat,
        U::StorageTexelBufferReadWithoutFormat,
        U::StorageTexelBufferWriteWithoutFormat,
    ];
    let dev = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
    assert_eq!(usages(&dev, vk::Format::R32_UINT), all);
    assert_eq!(usages(&dev, vk::Format::B8G8R8A8_UNORM), []);
    let mut write_only = desc.clone();
    write_only
        .features
        .remove(&VkFeature::ShaderStorageImageReadWithoutFormat);
    let dev = Device::new(write_only, DevicePolicy::default()).unwrap();
    assert_eq!(
        usages(&dev, vk::Format::R32_UINT),
        [
            U::StorageImageWriteWithoutFormat,
            U::StorageTexelBufferWriteWithoutFormat
        ]
    );
    // With it, the bits decide: a listed format's image bits are implied by the features, a
    // texel buffer's never are, and an unlisted format has only the bits it states. The
    // extension at 1.2 reports them as Vulkan 1.3 does.
    desc.khr_format_feature_flags2 = true;
    assert!(desc.has_format_feature_flags2());
    let dev = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
    assert_eq!(
        usages(&dev, vk::Format::R32_UINT),
        [
            U::StorageImageReadWithoutFormat,
            U::StorageImageWriteWithoutFormat
        ]
    );
    assert_eq!(usages(&dev, vk::Format::B8G8R8A8_UNORM), []);
    desc.formats.formats.insert(
        vk::Format::B8G8R8A8_UNORM,
        storage(
            F::STORAGE_WRITE_WITHOUT_FORMAT,
            F::STORAGE_READ_WITHOUT_FORMAT,
        ),
    );
    let dev = Device::new(desc.clone(), DevicePolicy::default()).unwrap();
    assert_eq!(
        usages(&dev, vk::Format::B8G8R8A8_UNORM),
        [
            U::StorageImageWriteWithoutFormat,
            U::StorageTexelBufferReadWithoutFormat
        ]
    );
    // The implication needs the feature.
    desc.features
        .remove(&VkFeature::ShaderStorageImageReadWithoutFormat);
    let dev = Device::new(desc, DevicePolicy::default()).unwrap();
    assert_eq!(
        usages(&dev, vk::Format::R32_UINT),
        [U::StorageImageWriteWithoutFormat]
    );

    // SwiftShader (Vulkan 1.3, no `VkFormatProperties3` in its document): its `R32_UINT` image is
    // written without a format, as the specification implies, and its texel buffer is not, which
    // its driver reports (`VUID-vkCmdDispatch-OpTypeImage-07029`); it has no read feature.
    let swiftshader = device("swiftshader");
    assert!(profile("swiftshader").has_format_feature_flags2());
    assert_eq!(
        usages(&swiftshader, vk::Format::R32_UINT),
        [U::StorageImageWriteWithoutFormat]
    );
    // lavapipe's document states `VkFormatProperties3` with all four bits.
    assert_eq!(usages(&device("lavapipe"), vk::Format::R32_UINT), all);
}

/// A MoltenVK driver on an Apple GPU carries
/// `DriverWorkaround::EarlyTestsDropMultisampledImageWrites` and
/// `DriverWorkaround::DiscardDropsSimdgroupStores` (the defects are the GPU's, which MoltenVK
/// reaches), keyed on the driver and the vendor and on nothing else: MoltenVK on another vendor's
/// GPU, and another driver on an Apple vendor ID, do not. (MoltenVK 1.4.1's
/// `BothBarycentricFormsRejected`, `CompositeInsertsWrittenInPlace`, `LoadsForwardedIntoLoops` and
/// `LocalizedLoadsForwardedPastAtomicsAndCalls`, and `FunctionsInlinedAtEveryCall`, are its
/// SPIRV-Cross's, on every vendor; its `MemoryWritingFragmentsRunPerSample`, is measured on Apple's
/// GPUs.)
#[test]
fn moltenvk_on_apple_carries_the_early_tests_image_write_defect() {
    use crate::device::profile::DriverWorkaround as W;
    let workarounds = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .clone()
    };
    let mut desc = profile("moltenvk-m3");
    assert_eq!(desc.driver_id, Some(vk::DriverId::MOLTENVK));
    assert_eq!(desc.vendor_id, 0x106B);
    assert_eq!(
        workarounds(&desc),
        [
            W::EarlyTestsDropMultisampledImageWrites,
            W::BothBarycentricFormsRejected,
            W::MemoryWritingFragmentsRunPerSample,
            W::CompositeInsertsWrittenInPlace,
            W::FunctionsInlinedAtEveryCall,
            W::LoadsForwardedIntoLoops,
            W::LocalizedLoadsForwardedPastAtomicsAndCalls,
            W::DiscardDropsSimdgroupStores
        ]
    );
    desc.vendor_id = 0x1002;
    assert_eq!(
        workarounds(&desc),
        [
            W::BothBarycentricFormsRejected,
            W::CompositeInsertsWrittenInPlace,
            W::FunctionsInlinedAtEveryCall,
            W::LoadsForwardedIntoLoops,
            W::LocalizedLoadsForwardedPastAtomicsAndCalls,
            W::ClippedCoverageNotMetals,
            W::TessellatorNotMetals
        ]
    );
    desc.vendor_id = 0x106B;
    desc.driver_id = Some(vk::DriverId::MESA_RADV);
    // Every rasterizer but Apple's hands a clipped primitive's fragments other coverage than
    // Metal's pieces, and every tessellator but Apple's orders a patch's triangles its own way;
    // RADV's own interpolation facts and its fused half conversion hold on any vendor.
    assert_eq!(
        workarounds(&desc),
        [
            W::HalfConversionFusedWithArithmetic,
            W::InterpolationNeedsTheWholeQuad,
            W::PerspectiveReadsExtrapolated,
            W::ClippedCoverageNotMetals,
            W::TessellatorNotMetals
        ]
    );
}

/// `DriverWorkaround::BothBarycentricFormsRejected`,
/// `DriverWorkaround::CompositeInsertsWrittenInPlace`, `DriverWorkaround::LoadsForwardedIntoLoops`,
/// `DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls` and
/// `DriverWorkaround::FunctionsInlinedAtEveryCall`, its SPIRV-Cross's, are keyed on MoltenVK by
/// the measured table (`driver_facts::DEFECTS`): carried at every release, and not by another
/// driver.
#[test]
fn moltenvk_carries_its_spirv_cross_defects_at_every_release() {
    use crate::device::profile::DriverWorkaround;
    for defect in [
        DriverWorkaround::BothBarycentricFormsRejected,
        DriverWorkaround::CompositeInsertsWrittenInPlace,
        DriverWorkaround::FunctionsInlinedAtEveryCall,
        DriverWorkaround::LoadsForwardedIntoLoops,
        DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls,
    ] {
        let carries = |desc: &DeviceDescription| {
            Device::new(desc.clone(), DevicePolicy::default())
                .unwrap()
                .profile()
                .fields()
                .workarounds
                .contains(&defect)
        };
        {
            let stem = "moltenvk-m3";
            let mut desc = profile(stem);
            assert!(carries(&desc), "{stem} {defect:?}");
            for version in [10400, 10500] {
                desc.driver_version = version;
                assert!(carries(&desc), "{stem} {version} {defect:?}");
            }
            desc.driver_id = Some(vk::DriverId::MESA_LLVMPIPE);
            assert!(!carries(&desc), "{stem} under lavapipe {defect:?}");
        }
        for (stem, text) in committed_profiles() {
            let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
            let moltenvk = desc.driver_id == Some(vk::DriverId::MOLTENVK);
            assert_eq!(carries(&desc), moltenvk, "{stem} {defect:?}");
        }
    }
}

/// `DriverWorkaround::MemoryWritingFragmentsRunPerSample` is keyed on MoltenVK and Apple's vendor
/// ID by the measured table (`driver_facts::DEFECTS`): carried at every release, and not on another
/// vendor's GPU, nor by another driver.
#[test]
fn moltenvk_on_apple_carries_the_per_sample_writer_defect_at_every_release() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::MemoryWritingFragmentsRunPerSample;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&defect)
    };
    {
        let stem = "moltenvk-m3";
        let mut desc = profile(stem);
        assert!(carries(&desc), "{stem}");
        for version in [10400, 10500] {
            desc.driver_version = version;
            assert!(carries(&desc), "{stem} {version}");
        }
        desc.vendor_id = 0x1002;
        assert!(!carries(&desc), "{stem} on another vendor");
        desc.vendor_id = 0x106B;
        desc.driver_id = Some(vk::DriverId::MESA_LLVMPIPE);
        assert!(!carries(&desc), "{stem} under lavapipe");
    }
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let apple_moltenvk =
            desc.driver_id == Some(vk::DriverId::MOLTENVK) && desc.vendor_id == 0x106B;
        assert_eq!(carries(&desc), apple_moltenvk, "{stem}");
    }
}

/// `DriverWorkaround::InterpolationNeedsTheWholeQuad` and
/// `DriverWorkaround::PerspectiveReadsExtrapolated` are keyed on RADV by the measured table
/// (`driver_facts::DEFECTS`): each carried at every release, whatever the vendor ID (the rows name
/// none), and not by another driver.
#[test]
fn radv_carries_its_interpolation_facts_at_every_release() {
    use crate::device::profile::DriverWorkaround;
    for fact in [
        DriverWorkaround::InterpolationNeedsTheWholeQuad,
        DriverWorkaround::PerspectiveReadsExtrapolated,
    ] {
        let carries = |desc: &DeviceDescription| {
            Device::new(desc.clone(), DevicePolicy::default())
                .unwrap()
                .profile()
                .fields()
                .workarounds
                .contains(&fact)
        };
        let mut desc = profile("amd-780m-radv");
        assert_eq!(desc.driver_id, Some(vk::DriverId::MESA_RADV));
        assert!(carries(&desc), "{fact:?}");
        desc.driver_version += 1;
        assert!(carries(&desc), "{fact:?} at another release");
        desc.vendor_id = 0x8086;
        assert!(carries(&desc), "{fact:?} on another vendor's ID");
        desc.driver_id = Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA);
        assert!(!carries(&desc), "{fact:?} under ANV");
        for (stem, text) in committed_profiles() {
            let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
            let radv = desc.driver_id == Some(vk::DriverId::MESA_RADV);
            assert_eq!(carries(&desc), radv, "{fact:?} on {stem}");
        }
    }
}

/// The profile holds the upper bound of the device's `pointSizeRange` in whole pixels, rounded
/// down, so a description's two-decimal JSON (2047.94, 8191.88) and the device's own `f32`
/// (2047.9375, 8191.875) hold the same; 0 for a range that is not above zero.
#[test]
fn the_profile_holds_the_largest_whole_point_size() {
    for (stem, max) in [
        ("nvidia-rtx5080-laptop", 2047),
        ("nvidia-rtx50-linux", 2047),
        ("amd-780m-radv", 8191),
        ("amd-rdna3-radv", 8191),
        ("intel-irisxe-rplp-anv", 255),
        ("lavapipe", 256),
        ("moltenvk-m3", 511),
        ("vk12-floor", 1),
    ] {
        assert_eq!(
            device(stem).profile().fields().limits.max_point_size,
            max,
            "{stem}"
        );
    }
    for (high, whole) in [
        (2047.9375f32, 2047),
        (2047.94, 2047),
        (8191.875, 8191),
        (8191.88, 8191),
        (511.0, 511),
        (0.0, 0),
        (-1.0, 0),
        (f32::NAN, 0),
        (f32::INFINITY, u32::MAX),
    ] {
        assert_eq!(crate::device::whole_point_size(high), whole, "{high}");
    }
}

/// Every driver carries `DriverWorkaround::ClippedCoverageNotMetals` and
/// `DriverWorkaround::TessellatorNotMetals` but MoltenVK on an Apple GPU, which draws through Metal
/// and splits a clipped primitive and orders a patch's triangles as Metal does: MoltenVK on another
/// vendor's GPU carries them, and so does every other driver on an Apple vendor ID.
#[test]
fn every_rasterizer_and_tessellator_but_apples_answer_their_own_way() {
    use crate::device::profile::DriverWorkaround;
    for answer in [
        DriverWorkaround::ClippedCoverageNotMetals,
        DriverWorkaround::TessellatorNotMetals,
    ] {
        let carries = |desc: &DeviceDescription| {
            Device::new(desc.clone(), DevicePolicy::default())
                .unwrap()
                .profile()
                .fields()
                .workarounds
                .contains(&answer)
        };
        for stem in [
            "nvidia-rtx5080-laptop",
            "amd-rdna3-radv",
            "intel-irisxe-rplp-anv",
            "lavapipe",
            "swiftshader",
        ] {
            assert!(carries(&profile(stem)), "{answer:?} {stem}");
        }
        let mut desc = profile("moltenvk-m3");
        assert!(!carries(&desc), "{answer:?}: MoltenVK on an Apple GPU");
        desc.vendor_id = 0x1002;
        assert!(
            carries(&desc),
            "{answer:?}: MoltenVK on another vendor's GPU"
        );
        desc.vendor_id = 0x106B;
        desc.driver_id = Some(vk::DriverId::MESA_RADV);
        assert!(
            carries(&desc),
            "{answer:?}: another driver on an Apple vendor ID"
        );
    }
}

/// Lavapipe carries `DriverWorkaround::ClippedTrianglesSplitPixelFragments` (the defect is the
/// driver's rasterizer, so whatever vendor ID it reports), keyed on the driver alone: the same
/// description under another driver does not.
#[test]
fn lavapipe_carries_the_split_pixel_fragments_defect() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
    let workarounds = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .clone()
    };
    // The committed lavapipe's release (Mesa 26.2.3) also carries the mesh index defect, kept
    // past the release measured with it (`lavapipe_carries_the_mesh_index_defect_from_its_
    // measured_release`).
    // And, as every rasterizer and tessellator but Apple's, `ClippedCoverageNotMetals` and
    // `TessellatorNotMetals`.
    let both = [
        defect,
        DriverWorkaround::MeshIndexComponentStoresMisplaced,
        DriverWorkaround::ClippedCoverageNotMetals,
        DriverWorkaround::TessellatorNotMetals,
    ];
    let mut desc = profile("lavapipe");
    assert_eq!(desc.driver_id, Some(vk::DriverId::MESA_LLVMPIPE));
    assert_eq!(workarounds(&desc), both);
    desc.vendor_id = 0x1002;
    assert_eq!(workarounds(&desc), both);
    desc.driver_id = Some(vk::DriverId::MESA_RADV);
    assert!(!workarounds(&desc).contains(&defect));
}

/// `DriverWorkaround::MeshIndexComponentStoresMisplaced` is keyed on lavapipe by the measured
/// table (`driver_facts::DEFECTS`): carried at every release, whatever the vendor ID (the row
/// names none), and not by another driver.
#[test]
fn lavapipe_carries_the_mesh_index_defect_at_every_release() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::MeshIndexComponentStoresMisplaced;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&defect)
    };
    let mut desc = profile("gate-lavapipe-x86_64");
    assert_eq!(desc.driver_id, Some(vk::DriverId::MESA_LLVMPIPE));
    assert!(carries(&desc));
    desc.driver_version += 1;
    assert!(carries(&desc), "at another release");
    desc.vendor_id = 0x8086;
    assert!(carries(&desc), "on another vendor's ID");
    desc.driver_id = Some(vk::DriverId::MESA_RADV);
    assert!(!carries(&desc), "under RADV");
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let lavapipe = desc.driver_id == Some(vk::DriverId::MESA_LLVMPIPE);
        assert_eq!(carries(&desc), lavapipe, "{stem}");
    }
}

/// `DriverWorkaround::NarrowMaskAfterSignedClamp` is keyed on NVIDIA's driver on Ampere's GA10x
/// GPUs and on nothing else. No committed profile describes one, so none carries it, and each
/// translates as before. The RTX 5080 laptop GPU's description under the RTX 3090's device ID
/// (0x2204, the GPU measured) and under another GA10x ID (the RTX 3070, 0x2484) carries it; under
/// its own ID (Blackwell, measured right), or with the 3090's ID under another driver, it does not.
#[test]
fn only_nvidia_on_ampere_ga10x_carries_the_narrow_mask_defect() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::NarrowMaskAfterSignedClamp;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&defect)
    };
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        assert!(!carries(&desc), "{stem}");
    }
    let mut desc = profile("nvidia-rtx5080-laptop");
    assert_eq!(desc.driver_id, Some(vk::DriverId::NVIDIA_PROPRIETARY));
    assert!(!crate::device::NVIDIA_AMPERE_GA10X.contains(&desc.device_id));
    for ampere in [0x2204, 0x2484] {
        desc.device_id = ampere;
        assert!(carries(&desc), "{ampere:#x}");
    }
    desc.driver_id = Some(vk::DriverId::MESA_NVK);
    assert!(!carries(&desc));
}

/// `DriverWorkaround::HalfConversionPairFolded` is keyed on NVIDIA's driver whatever its GPU and on
/// nothing else: every committed NVIDIA profile carries it (the RTX 3090's generation and the RTX
/// 50 series alike) and no other committed profile does; the RTX 5080 laptop GPU's description
/// under Mesa's NVK does not. Where it is carried, a profile whose float controls would make the
/// conversion Metal's rounds a `half` in another form (`narrow::controlled_modes` is `None`).
#[test]
fn nvidia_carries_the_folded_half_conversion_pair() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::HalfConversionPairFolded;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&defect)
    };
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let nvidia = desc.driver_id == Some(vk::DriverId::NVIDIA_PROPRIETARY);
        assert_eq!(carries(&desc), nvidia, "{stem}");
    }
    let mut desc = profile("nvidia-rtx5080-laptop");
    let p = Device::new(desc.clone(), DevicePolicy::default())
        .unwrap()
        .profile()
        .clone();
    let half = p.fields().float_controls.f16;
    assert!(half.rounding_rte && half.denorm_preserve && half.signed_zero_inf_nan_preserve);
    assert!(crate::lower::arith::narrow::controlled_modes(&p).is_none());
    desc.device_id = 0x2204;
    assert!(carries(&desc));
    desc.driver_id = Some(vk::DriverId::MESA_NVK);
    assert!(!carries(&desc));
}

/// `DriverWorkaround::HalfConversionFusedWithArithmetic` is keyed on RADV whatever its GPU and on
/// nothing else: every committed RADV profile carries it (the Radeon 780M, 680M and RX 7800 XT)
/// and no other committed profile does; the 780M's description under another driver does not.
/// Such a profile keeps the conversion under the float controls (`QuantizeForm::Controlled`).
#[test]
fn radv_carries_the_fused_half_conversion() {
    use crate::device::profile::DriverWorkaround;
    let defect = DriverWorkaround::HalfConversionFusedWithArithmetic;
    let profile_of = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .clone()
    };
    let mut radv = 0;
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let on_radv = desc.driver_id == Some(vk::DriverId::MESA_RADV);
        let p = profile_of(&desc);
        assert_eq!(p.workarounds().contains(&defect), on_radv, "{stem}");
        if on_radv {
            radv += 1;
            assert_eq!(
                crate::lower::arith::narrow::quantize_form(&p),
                crate::spirv::builder::QuantizeForm::Controlled,
                "{stem}"
            );
        }
    }
    assert!(radv >= 3, "{radv} RADV profiles");
    let mut desc = profile("amd-780m-radv");
    assert!(profile_of(&desc).workarounds().contains(&defect));
    desc.driver_id = Some(vk::DriverId::AMD_OPEN_SOURCE);
    assert!(!profile_of(&desc).workarounds().contains(&defect));
}

/// `DriverWorkaround::RegisterAllocatorOverflowsLargeModules` is keyed on ANV, whatever its GPU
/// and release; every committed ANV profile carries it, and no other profile does. The UHD 770's
/// description under RADV or NVIDIA's driver loses it.
#[test]
fn register_allocator_overflow_is_keyed_on_anv() {
    use crate::device::profile::DriverWorkaround;
    let fact = DriverWorkaround::RegisterAllocatorOverflowsLargeModules;
    let profile_of = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .clone()
    };
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let anv = desc.driver_id == Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA);
        let p = profile_of(&desc);
        assert_eq!(p.workarounds().contains(&fact), anv, "{stem}");
    }
    let mut desc = profile("intel-irisxe-rplp-anv");
    assert!(profile_of(&desc).workarounds().contains(&fact));
    for other in [vk::DriverId::MESA_RADV, vk::DriverId::NVIDIA_PROPRIETARY] {
        desc.driver_id = Some(other);
        let p = profile_of(&desc);
        assert!(!p.workarounds().contains(&fact), "{other:?}");
    }
}

/// The Function storage a thread holds is the measured value of the device's identity, and on every
/// committed profile of a device not measured the assumed floor, 16 KiB. It is keyed on the driver,
/// the vendor, and the device ID together: the Radeon 780M's description under the RX 7900 XTX's
/// device ID (a measured RADV GPU of no committed capture) takes the XTX's value, and under another
/// device ID (an unmeasured RADV GPU, Navi 21's 0x73bf) or another driver is assumed. The Intel
/// venue's two devices take one value, so they keep one translation key.
#[test]
fn function_storage_is_the_measured_value_of_the_device() {
    use crate::device::profile::FunctionStorage::{Assumed, Measured};
    let kib = 1024;
    let measured = [
        ("amd-780m-hawkpoint-radv", 96 * kib),
        ("amd-780m-radv", 96 * kib),
        ("intel-irisxe-rplp-anv", 32 * kib),
        ("intel-uhd770-rpls-anv", 32 * kib),
        ("nvidia-rtx5080-laptop", 32 * kib),
        ("moltenvk-m3", 192 * kib),
        ("gate-lavapipe", 48 * kib),
        ("gate-lavapipe-x86_64", 48 * kib),
        ("lavapipe", 48 * kib),
    ];
    let storage = |desc: DeviceDescription| {
        Device::new(desc, DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .function_storage
    };
    for (stem, _) in committed_profiles() {
        let want = measured
            .iter()
            .find(|(m, _)| *m == stem)
            .map_or(Assumed(16 * kib), |&(_, b)| Measured(b));
        assert_eq!(storage(profile(&stem)), want, "{stem}");
    }
    let mut desc = profile("amd-780m-radv");
    desc.device_id = 0x744c;
    assert_eq!(storage(desc.clone()), Measured(96 * kib));
    desc.device_id = 0x73bf;
    assert_eq!(storage(desc.clone()), Assumed(16 * kib));
    desc.device_id = 0x15bf;
    desc.driver_id = Some(vk::DriverId::AMD_OPEN_SOURCE);
    assert_eq!(storage(desc), Assumed(16 * kib));
}

/// `DriverWorkaround::VertexGroupingNotApples` is keyed on ANV, for every GPU it runs at every
/// release: every committed ANV profile carries it, and another driver does not.
#[test]
fn anv_carries_the_vertex_grouping_at_every_release() {
    use crate::device::profile::DriverWorkaround;
    let fact = DriverWorkaround::VertexGroupingNotApples;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&fact)
    };
    let mut desc = profile("intel-irisxe-rplp-anv");
    assert_eq!(desc.driver_id, Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA));
    assert!(carries(&desc));
    desc.device_id = 0x56a0;
    assert!(carries(&desc), "on another Intel GPU");
    desc.driver_version += 1;
    assert!(carries(&desc), "at another release");
    desc.driver_id = Some(vk::DriverId::MESA_RADV);
    assert!(!carries(&desc), "under RADV");
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let anv = desc.driver_id == Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA);
        assert_eq!(carries(&desc), anv, "{stem}");
    }
    assert!(carries(&profile("intel-uhd770-rpls-anv")));
    assert!(carries(&profile("intel-arl-anv")));
    assert!(carries(&profile("intel-irisxe-anv")));
}

/// `DriverWorkaround::DontInlineKeepsCalls`, a cost fact, is keyed on the GPUs measured with it:
/// NVIDIA's driver on the RTX 3090 and on the RTX 5080 Laptop GPU (`scripts/making-bench`), at
/// every release. Another GPU of the vendor (the desktop RTX 50's committed profile, Pascal's),
/// another vendor's ID, or another driver (NVK, Mesa's, MoltenVK) does not carry it.
#[test]
fn nvidia_keeps_dont_inline_calls_on_its_measured_gpu() {
    use crate::device::profile::DriverWorkaround;
    let fact = DriverWorkaround::DontInlineKeepsCalls;
    let carries = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .unwrap()
            .profile()
            .fields()
            .workarounds
            .contains(&fact)
    };
    let mut desc = profile("nvidia-rtx5080-laptop");
    assert_eq!(desc.driver_id, Some(vk::DriverId::NVIDIA_PROPRIETARY));
    assert!(
        carries(&desc),
        "on the RTX 5080 Laptop GPU, where it was measured"
    );
    desc.device_id = crate::device::driver_facts::NVIDIA_RTX_3090;
    assert!(carries(&desc), "on the RTX 3090, where it was measured");
    desc.driver_version += 1 << 6;
    assert!(carries(&desc), "at another release");
    desc.vendor_id = 0x1002;
    assert!(!carries(&desc), "on another vendor's ID");
    desc.vendor_id = 0x10de;
    desc.driver_id = Some(vk::DriverId::MESA_NVK);
    assert!(!carries(&desc), "under NVK");
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        assert_eq!(
            carries(&desc),
            stem == "nvidia-rtx5080-laptop",
            "{stem}: the RTX 5080 Laptop GPU's is the one committed profile of a measured GPU"
        );
    }
    assert!(!carries(&profile("nvidia-pascal-linux")));
    assert!(!carries(&profile("amd-780m-radv")));
    assert!(!carries(&profile("moltenvk-m3")));
}
