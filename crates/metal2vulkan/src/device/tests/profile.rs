//! `Device::from_queries`, profile derivation, the fingerprint, and `requirements ⊆ profile`.

use super::*;
use crate::device::profile::{BufferAccess, HandleDomain, Independence, PointerDomain};
use crate::device::requirements::{Requirements, StageCreateFlag, VkLimit, VkProperty};
use crate::error::{Capability, Feature, Needs};

pub(crate) fn device(stem: &str) -> Device {
    Device::from_queries(profile(stem), DevicePolicy::RESIDENT_TABLES)
        .expect("committed profile derives")
}

#[test]
fn every_committed_profile_derives() {
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let f = dev.profile().fields();
        assert_eq!(f.api_version, desc.api_version, "{stem}");
        // Deterministic.
        let again = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap();
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

/// `occlusionQueryPrecise` is read from each committed profile, and enabled there: every venue's
/// source states it, and the runtime floor's block (`m2v_runtime_floor`) requires it.
#[test]
fn occlusion_query_precise_follows_each_source() {
    for (stem, text) in committed_profiles() {
        let stated = text.contains("\"occlusionQueryPrecise\": true");
        assert!(stated, "{stem}");
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        assert_eq!(
            desc.features.contains(&VkFeature::OcclusionQueryPrecise),
            stated,
            "{stem}"
        );
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap();
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
        let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
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
    // Without either, below the runtime's floor, which requires both.
    for without in [
        &[VkFeature::BufferDeviceAddress, VkFeature::ShaderInt64][..],
        &[VkFeature::ShaderInt64],
    ] {
        let dev =
            Device::from_queries(floor_without(without), DevicePolicy::RESIDENT_TABLES).unwrap();
        assert_eq!(
            dev.profile().fields().buffer_access,
            BufferAccess::Floor,
            "{without:?}"
        );
    }
    for stem in [
        "runtime-floor",
        "moltenvk-m3",
        "nvidia-rtx50-linux",
        "amd-780m-radv",
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

/// A Profile F refusal names what the device lacks of Profile A, never a feature it has: on
/// `runtime-floor` without either feature, and without `shaderInt64` alone.
#[test]
fn addressed_needs_name_only_the_missing_features() {
    let bda = Capability::Feature(VkFeature::BufferDeviceAddress);
    let int64 = Capability::Feature(VkFeature::ShaderInt64);
    let needs = |without: &[VkFeature]| {
        Device::from_queries(floor_without(without), DevicePolicy::RESIDENT_TABLES)
            .unwrap()
            .profile()
            .addressed_needs()
    };
    assert_eq!(
        needs(&[VkFeature::BufferDeviceAddress, VkFeature::ShaderInt64]),
        vec![bda, int64.clone()]
    );
    assert_eq!(needs(&[VkFeature::ShaderInt64]), vec![int64]);
}

/// `runtime-floor` derives the profile of a device at the runtime's floor: addressed buffers,
/// stores and atomics in every stage, and the formats the Vulkan 1.2 Required Format Support
/// tables require of it.
#[test]
fn runtime_floor_derives_the_floor_profile() {
    let dev = device("runtime-floor");
    let f = dev.profile().fields();
    assert_eq!(f.buffer_access, BufferAccess::Addressed);
    assert!(f.descriptors.uniform_buffer_standard_layout);
    assert!(f.leaf.vertex_stores_and_atomics);
    assert!(f.leaf.fragment_stores_and_atomics);
    // What the tables require of it is supported, and a usage they leave to a feature it does not
    // require is refused (a typed Format capability).
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
        Ok(())
    );
    assert_eq!(
        dev.profile().check(&req(
            vk::Format::D32_SFLOAT,
            FormatUsage::DepthStencilAttachment
        )),
        Err(Capability::Format {
            format: vk::Format::D32_SFLOAT,
            usage: FormatUsage::DepthStencilAttachment
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
    let full = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
    let f = full.profile().fields();
    assert!(f.fragment.output_layer && f.fragment.output_viewport_index);
    assert!(desc.features.remove(&VkFeature::ShaderOutputLayer));
    assert!(desc.features.remove(&VkFeature::ShaderOutputViewportIndex));
    let lacking = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap();
    let f = lacking.profile().fields();
    assert!(!f.fragment.output_layer);
    assert!(!f.fragment.output_viewport_index);
}

/// `moltenvk-m3` derives a dynamic vertex stride and its formats' usages as it reports them.
#[test]
fn moltenvk_m3_derives_its_stride_and_format_usages() {
    let f = device("moltenvk-m3");
    let m = f.profile().fields();
    assert!(m.dynamic_vertex_stride);
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
fn below_1_2_is_refused() {
    let mut desc = profile("runtime-floor");
    desc.api_version = ApiVersion { major: 1, minor: 1 };
    match Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VulkanVersion);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::ApiVersion(ApiVersion::V1_2)]])
            );
        }
        other => panic!("{other:?}"),
    }
}

/// The identity domains come as a pair (`InvalidDescriptor` for one alone) and need
/// `bufferDeviceAddress` and `shaderInt64` (`Unsupported{IdentityDomains}` naming both, on a
/// device without `shaderInt64`). What they lower is `tests::engine_identity`'s.
#[test]
fn the_identity_domains_come_as_a_pair_on_a_device_with_addresses() {
    let identity = DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::RESIDENT_TABLES
    };
    for half in [
        DevicePolicy {
            pointer_domain: PointerDomain::Identity,
            ..DevicePolicy::RESIDENT_TABLES
        },
        DevicePolicy {
            handle_domain: HandleDomain::Identity,
            ..DevicePolicy::RESIDENT_TABLES
        },
    ] {
        assert!(
            matches!(
                Device::from_queries(profile("lavapipe"), half),
                Err(Error::InvalidDescriptor(
                    DescriptorError::DeviceDescription { .. }
                ))
            ),
            "{half:?}"
        );
    }
    // `runtime-floor` without `shaderInt64`: device addresses, no 64-bit integers.
    match Device::from_queries(floor_without(&[VkFeature::ShaderInt64]), identity) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::IdentityDomains);
            let both = [VkFeature::BufferDeviceAddress, VkFeature::ShaderInt64];
            let both = both.map(Capability::Feature).to_vec();
            assert_eq!(u.needs, Needs::AnyOf(vec![both]));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn fingerprint_covers_every_policy_field() {
    let desc = profile("lavapipe");
    let base = DevicePolicy::RESIDENT_TABLES;
    let fp = |p: DevicePolicy| {
        Device::from_queries(desc.clone(), p)
            .unwrap()
            .profile()
            .fingerprint()
    };
    let reference = fp(base);
    let mut variants = Vec::new();
    let identity = DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..base
    };
    variants.push(("domains", identity));
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
        Device::from_queries(fewer, base)
            .unwrap()
            .profile()
            .fingerprint(),
        reference
    );
    let mut limits = desc;
    limits.limits.max_texel_offset += 1;
    assert_ne!(
        Device::from_queries(limits, base)
            .unwrap()
            .profile()
            .fingerprint(),
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
/// its allocation sizes, as two captures of one GPU under its two `deviceID`s do
/// (`amd-780m-hawkpoint-radv` and `amd-780m-radv`, the Radeon 780M of Hawk Point and of Phoenix),
/// and as one GPU family on one driver does (`intel-irisxe-rplp-anv` and `intel-uhd770-rpls-anv`).
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
            let dev = Device::from_queries(other, DevicePolicy::RESIDENT_TABLES).unwrap();
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
    let dev = device("runtime-floor");
    let p = dev.profile();
    assert_eq!(p.check(&Requirements::default()), Ok(()));
    let need = |r: Requirements| p.check(&r);
    let v1_4 = ApiVersion { major: 1, minor: 4 };
    assert_eq!(
        need(Requirements {
            api_version: Some(v1_4),
            ..Requirements::default()
        }),
        Err(Capability::ApiVersion(v1_4))
    );
    assert_eq!(
        need(Requirements {
            features: [VkFeature::RobustBufferAccess, VkFeature::ShaderFloat64].into(),
            ..Requirements::default()
        }),
        Err(Capability::Feature(VkFeature::ShaderFloat64))
    );
    assert_eq!(
        need(Requirements {
            extensions: [VkExtension::ExtShaderStencilExport].into(),
            ..Requirements::default()
        }),
        Err(Capability::Extension(VkExtension::ExtShaderStencilExport))
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
    // Stage flags need their enabling features, which the floor requires: the floor without
    // `computeFullSubgroups`, a device below it, names it.
    let without = Device::from_queries(
        floor_without(&[VkFeature::ComputeFullSubgroups]),
        DevicePolicy::RESIDENT_TABLES,
    )
    .unwrap();
    assert_eq!(
        without.profile().check(&Requirements {
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
    // `runtime-floor` as a Vulkan 1.2 context without `VK_KHR_format_feature_flags2` reads it, with
    // both formatless features.
    let mut desc = crate::pipeline::testing::description_at_1_2("runtime-floor", false).unwrap();
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
    let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
    assert_eq!(usages(&dev, vk::Format::R32_UINT), all);
    assert_eq!(usages(&dev, vk::Format::B8G8R8A8_UNORM), []);
    let mut write_only = desc.clone();
    write_only
        .features
        .remove(&VkFeature::ShaderStorageImageReadWithoutFormat);
    let dev = Device::from_queries(write_only, DevicePolicy::RESIDENT_TABLES).unwrap();
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
    let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
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
    let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
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
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap();
    assert_eq!(
        usages(&dev, vk::Format::R32_UINT),
        [U::StorageImageWriteWithoutFormat]
    );

    // lavapipe's document states `VkFormatProperties3` with all four bits.
    assert_eq!(usages(&device("lavapipe"), vk::Format::R32_UINT), all);
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
        ("intel-irisxe-rplp-anv", 255),
        ("lavapipe", 256),
        ("moltenvk-m3", 511),
        ("runtime-floor", 63),
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

/// Through `Device::from_queries`, a profile's `half` rounding follows its queries alone: every
/// committed profile with `shaderFloat16` and the 16-bit float controls (`narrow::half_controls`)
/// rounds an `f32` to `half` through the controlled conversion (`narrow::quantize_form` is
/// `Controlled`), NVIDIA's included.
#[test]
fn from_queries_half_rounding_follows_the_queries_alone() {
    use crate::lower::arith::narrow::{half_controls, quantize_form};
    use crate::spirv::builder::QuantizeForm;
    let mut controlled = Vec::new();
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).unwrap();
        let p = device.profile();
        if half_controls(p) {
            assert_eq!(quantize_form(p), QuantizeForm::Controlled, "{stem}");
            controlled.push(stem);
        }
    }
    for stem in [
        "nvidia-rtx5080-laptop",
        "nvidia-rtx50-linux",
        "amd-780m-radv",
    ] {
        assert!(
            controlled.iter().any(|s| s == stem),
            "{stem}: {controlled:?}"
        );
    }
}
