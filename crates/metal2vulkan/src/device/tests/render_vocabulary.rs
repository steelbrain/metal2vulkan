//! The render runtime's vocabulary (ARTIFACT 180), member by member against each committed
//! profile's source: every committed profile states each member it can have, and the description
//! reads, derives and enables exactly what the source states.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::device::requirements::{ResolveMode, VkProperty};
use serde_json::Value;

/// Every value stated under the key `member` in `v`, depth first: a member a profile states in
/// several structures (`VkPhysicalDeviceVulkan12Properties` and the structure it absorbed) or
/// blocks is stated in each.
fn stated(v: &Value, member: &str, out: &mut Vec<Value>) {
    match v {
        Value::Object(map) => {
            for (key, child) in map {
                if key == member {
                    out.push(child.clone());
                }
                stated(child, member, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                stated(child, member, out);
            }
        }
        _ => {}
    }
}

/// The values `json`'s capability blocks state for `member`.
fn capability_values(json: &Value, member: &str) -> Vec<Value> {
    let mut out = Vec::new();
    stated(&json["capabilities"], member, &mut out);
    out
}

/// Whether some block states `member` true in a structure named `structure`.
fn stated_true(json: &Value, structure: &str, member: &str) -> bool {
    let mut structures = Vec::new();
    stated(&json["capabilities"], structure, &mut structures);
    structures
        .iter()
        .any(|s| s.get(member) == Some(&Value::Bool(true)))
}

/// The union of the flag names `json` states for `member`.
fn stated_flags(json: &Value, member: &str) -> BTreeSet<String> {
    capability_values(json, member)
        .iter()
        .flat_map(|v| v.as_array().cloned().unwrap_or_default())
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect()
}

/// The eight features ARTIFACT 180 adds, `imageViewFormatSwizzle` (ARTIFACT 203),
/// `provokingVertexLast`, `multiDraw` and `pointPolygons`, each with its structure, its member,
/// and the extension that exposes it
/// (`None`: `VkPhysicalDeviceFeatures`, Vulkan 1.0).
const FEATURES: [(&str, &str, VkFeature, Option<VkExtension>); 12] = [
    (
        "VkPhysicalDeviceFeatures",
        "fullDrawIndexUint32",
        VkFeature::FullDrawIndexUint32,
        None,
    ),
    (
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR",
        "multisampleArrayImage",
        VkFeature::MultisampleArrayImage,
        Some(VkExtension::KhrPortabilitySubset),
    ),
    (
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR",
        "imageView2DOn3DImage",
        VkFeature::ImageView2DOn3DImage,
        Some(VkExtension::KhrPortabilitySubset),
    ),
    (
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR",
        "separateStencilMaskRef",
        VkFeature::SeparateStencilMaskRef,
        Some(VkExtension::KhrPortabilitySubset),
    ),
    (
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR",
        "imageViewFormatSwizzle",
        VkFeature::ImageViewFormatSwizzle,
        Some(VkExtension::KhrPortabilitySubset),
    ),
    (
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR",
        "pointPolygons",
        VkFeature::PointPolygons,
        Some(VkExtension::KhrPortabilitySubset),
    ),
    (
        "VkPhysicalDeviceDepthClipEnableFeaturesEXT",
        "depthClipEnable",
        VkFeature::DepthClipEnable,
        Some(VkExtension::ExtDepthClipEnable),
    ),
    (
        "VkPhysicalDeviceDepthClampControlFeaturesEXT",
        "depthClampControl",
        VkFeature::DepthClampControl,
        Some(VkExtension::ExtDepthClampControl),
    ),
    (
        "VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT",
        "primitiveTopologyListRestart",
        VkFeature::PrimitiveTopologyListRestart,
        Some(VkExtension::ExtPrimitiveTopologyListRestart),
    ),
    (
        "VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT",
        "primitiveTopologyPatchListRestart",
        VkFeature::PrimitiveTopologyPatchListRestart,
        Some(VkExtension::ExtPrimitiveTopologyListRestart),
    ),
    (
        "VkPhysicalDeviceProvokingVertexFeaturesEXT",
        "provokingVertexLast",
        VkFeature::ProvokingVertexLast,
        Some(VkExtension::ExtProvokingVertex),
    ),
    (
        "VkPhysicalDeviceMultiDrawFeaturesEXT",
        "multiDraw",
        VkFeature::MultiDraw,
        Some(VkExtension::ExtMultiDraw),
    ),
];

/// Each feature is read, and enabled, exactly where the profile's source states it true in an
/// exposed structure (or its version requires it: `fullDrawIndexUint32` from Vulkan 1.4). Where
/// the source lists the feature's extension it states the member, true or false; a member a
/// source does not state reads as absent, and is refused. The extensions are enabled exactly
/// where the source lists them. Every committed profile states `fullDrawIndexUint32`.
#[test]
fn render_features_follow_each_source() {
    let mut listed: BTreeMap<VkExtension, usize> = BTreeMap::new();
    for (stem, text) in committed_profiles() {
        let json: Value = serde_json::from_str(&text).unwrap();
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
        let enabled = &dev.profile().fields().enabled;
        for (structure, member, feature, extension) in FEATURES {
            let exposed = extension.is_none_or(|e| desc.extensions.contains(&e));
            let mut members = Vec::new();
            stated(&json["capabilities"], structure, &mut members);
            let states = members.iter().any(|m| m.get(member).is_some());
            if extension.is_some() {
                assert_eq!(states, exposed, "{stem}: {structure}.{member}");
            } else {
                assert!(states, "{stem}: {member}");
            }
            let required = required_features(desc.api_version).any(|f| f == feature);
            let want = exposed && (stated_true(&json, structure, member) || required);
            assert_eq!(desc.features.contains(&feature), want, "{stem}: {member}");
            assert_eq!(
                enabled.features.contains(&feature),
                want,
                "{stem}: {member}"
            );
        }
        for extension in [
            VkExtension::ExtDepthClipEnable,
            VkExtension::ExtDepthClampControl,
            VkExtension::ExtPrimitiveTopologyListRestart,
            VkExtension::ExtProvokingVertex,
            VkExtension::ExtSampleLocations,
            VkExtension::ExtMultiDraw,
        ] {
            let has = desc.extensions.contains(&extension);
            assert_eq!(
                enabled.extensions.contains(&extension),
                has,
                "{stem}: {extension:?}"
            );
            *listed.entry(extension).or_default() += usize::from(has);
        }
    }
    // RADV, ANV, NVIDIA and lavapipe list depth clipping and list restart; RADV, ANV and NVIDIA
    // the clamp control; every Linux venue and MoltenVK the sample locations; every native
    // profile (RADV, ANV, NVIDIA, lavapipe) the provoking vertex; RADV, ANV, NVIDIA
    // and lavapipe the multi-draw.
    for (extension, at_least) in [
        (VkExtension::ExtDepthClipEnable, 10),
        (VkExtension::ExtDepthClampControl, 5),
        (VkExtension::ExtPrimitiveTopologyListRestart, 10),
        (VkExtension::ExtProvokingVertex, 13),
        (VkExtension::ExtSampleLocations, 10),
        (VkExtension::ExtMultiDraw, 10),
    ] {
        let count = listed.get(&extension).copied().unwrap_or(0);
        assert!(count >= at_least, "{count} profiles list {extension:?}");
    }
}

/// The depth/stencil resolve properties and the two limits are stated by every committed profile
/// (a device report its device's; the derived floor its version's, which
/// the regeneration states where its source omits them), and the description derives exactly
/// the stated modes and booleans.
#[test]
fn render_properties_and_limits_follow_each_source() {
    let name = |mode: ResolveMode| match mode {
        ResolveMode::SampleZero => "VK_RESOLVE_MODE_SAMPLE_ZERO_BIT",
        ResolveMode::Average => "VK_RESOLVE_MODE_AVERAGE_BIT",
        ResolveMode::Min => "VK_RESOLVE_MODE_MIN_BIT",
        ResolveMode::Max => "VK_RESOLVE_MODE_MAX_BIT",
    };
    for (stem, text) in committed_profiles() {
        let json: Value = serde_json::from_str(&text).unwrap();
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
        let properties = &dev.profile().fields().enabled.properties;
        let depth = stated_flags(&json, "supportedDepthResolveModes");
        let stencil = stated_flags(&json, "supportedStencilResolveModes");
        // Vulkan 1.2 requires sample zero of both, and every profile states it.
        assert!(depth.contains(name(ResolveMode::SampleZero)), "{stem}");
        assert!(stencil.contains(name(ResolveMode::SampleZero)), "{stem}");
        for mode in ResolveMode::ALL {
            assert_eq!(
                properties.contains(&VkProperty::DepthResolveMode(mode)),
                depth.contains(name(mode)),
                "{stem}: depth {mode:?}"
            );
            assert_eq!(
                properties.contains(&VkProperty::StencilResolveMode(mode)),
                stencil.contains(name(mode)),
                "{stem}: stencil {mode:?}"
            );
        }
        for (member, property) in [
            ("independentResolveNone", VkProperty::IndependentResolveNone),
            ("independentResolve", VkProperty::IndependentResolve),
        ] {
            let values = capability_values(&json, member);
            assert!(!values.is_empty(), "{stem} states no {member}");
            let on = values.iter().any(|v| v == &Value::Bool(true));
            assert_eq!(properties.contains(&property), on, "{stem}: {member}");
        }
        for (member, value) in [
            (
                "maxDrawIndexedIndexValue",
                desc.limits.max_draw_indexed_index_value,
            ),
            (
                "maxTessellationPatchSize",
                desc.limits.max_tessellation_patch_size,
            ),
        ] {
            let values = capability_values(&json, member);
            let largest = values.iter().filter_map(Value::as_u64).max();
            assert_eq!(largest, Some(u64::from(value)), "{stem}: {member}");
        }
        if desc.features.contains(&VkFeature::FullDrawIndexUint32) {
            assert_eq!(desc.limits.max_draw_indexed_index_value, u32::MAX, "{stem}");
        } else {
            assert!(
                desc.limits.max_draw_indexed_index_value >= (1 << 24) - 1,
                "{stem}"
            );
        }
    }
}

/// Each sample count a profile states for a color attachment format (its
/// `m2v_color_attachment_sample_counts` entry) derives one
/// `VkProperty::ColorAttachmentSampleCount`, and nothing else derives one: a profile without the
/// entry has none.
#[test]
fn each_stated_color_attachment_sample_count_derives_its_property() {
    let mut with_counts = 0;
    for (stem, text) in committed_profiles() {
        let json: Value = serde_json::from_str(&text).unwrap();
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let dev = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
        let derived: BTreeSet<(vk::Format, u32)> = dev
            .profile()
            .fields()
            .enabled
            .properties
            .iter()
            .filter_map(|p| match *p {
                VkProperty::ColorAttachmentSampleCount(format, count) => Some((format, count)),
                _ => None,
            })
            .collect();
        let stated: BTreeSet<(vk::Format, u32)> = desc
            .formats
            .formats
            .iter()
            .flat_map(|(&format, p)| {
                (0..7)
                    .map(|shift| 1u32 << shift)
                    .filter(|&count| {
                        p.color_attachment_sample_counts
                            .contains(vk::SampleCountFlags::from_raw(count))
                    })
                    .map(move |count| (format, count))
            })
            .collect();
        assert_eq!(derived, stated, "{stem}");
        let entry = json.get("m2v_color_attachment_sample_counts").is_some();
        assert_eq!(!stated.is_empty(), entry, "{stem}");
        with_counts += usize::from(entry);
    }
    assert!(with_counts > 0);
}

/// A document that states neither the resolve modes nor `maxDrawIndexedIndexValue` reads its
/// version's guarantees: sample zero for both aspects, and `2^24 - 1`, or `2^32 - 1` where it
/// states `fullDrawIndexUint32`. The booleans it omits read as absent.
#[test]
fn an_omitted_resolve_mode_or_index_value_reads_its_versions_guarantee() {
    let doc = |features: &str| {
        format!(
            r#"{{"capabilities": {{"base": {{"features": {{"VkPhysicalDeviceFeatures": {{{features}}}}},
                "extensions": {{"VK_EXT_sample_locations": 1}}}}}},
              "profiles": {{"P": {{"api-version": "1.2.0", "capabilities": ["base"]}}}}}}"#
        )
    };
    let plain = DeviceDescription::from_khronos_json(&doc(""), None).unwrap();
    let l = &plain.limits;
    assert_eq!(
        l.supported_depth_resolve_modes,
        vk::ResolveModeFlags::SAMPLE_ZERO
    );
    assert_eq!(
        l.supported_stencil_resolve_modes,
        vk::ResolveModeFlags::SAMPLE_ZERO
    );
    assert_eq!(l.max_draw_indexed_index_value, (1 << 24) - 1);
    assert_eq!(l.max_tessellation_patch_size, 0);
    assert!(!l.independent_resolve_none && !l.independent_resolve);
    let full =
        DeviceDescription::from_khronos_json(&doc(r#""fullDrawIndexUint32": true"#), None).unwrap();
    assert_eq!(full.limits.max_draw_indexed_index_value, u32::MAX);
}
