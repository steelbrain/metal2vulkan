//! The Vulkan 1.2 specification's Required Format Support tables over the committed profiles.
//!
//! `scripts/regen-device-profiles/required_formats.py` reads the tables from the pinned
//! specification source. The regeneration writes every unconditional entry as the capability
//! block `m2v_vulkan12_required_formats` of each requirements profile that states no formats of
//! its own, the entries a feature it requires makes unconditional as one more block per feature,
//! and every conditional requirement into its `provenance.required_formats.conditional`.

use std::collections::BTreeMap;

use serde_json::Value;

use super::*;

const BLOCK: &str = "m2v_vulkan12_required_formats";

/// Format → the `VkFormatProperties` bits required of it.
type Table = BTreeMap<vk::Format, FormatProperties>;

/// `{member: [bit names]}` as `VkFormatProperties`.
fn properties(members: &Value) -> FormatProperties {
    let flags = |member: &str| {
        let mut flags = vk::FormatFeatureFlags::empty();
        for bit in members
            .get(member)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = bit.as_str().expect("a bit name");
            flags |= vk::FormatFeatureFlags::from_name(name)
                .unwrap_or_else(|| panic!("{name} is not a VkFormatFeatureFlagBits name"));
        }
        flags
    };
    let known = [
        "linearTilingFeatures",
        "optimalTilingFeatures",
        "bufferFeatures",
    ];
    for member in members.as_object().expect("an object").keys() {
        assert!(known.contains(&member.as_str()), "unknown member {member}");
    }
    FormatProperties::from_legacy(
        flags("linearTilingFeatures"),
        flags("optimalTilingFeatures"),
        flags("bufferFeatures"),
    )
}

/// A `{format: {member: [bits]}}` table (`wrapped`: each entry is `{"VkFormatProperties": ..}`, as
/// in a capability block).
fn table(value: &Value, wrapped: bool) -> Table {
    value
        .as_object()
        .expect("a format table")
        .iter()
        .map(|(name, entry)| {
            let format = vk::Format::from_name(name)
                .unwrap_or_else(|| panic!("{name} is not a VkFormat name"));
            let members = if wrapped {
                &entry["VkFormatProperties"]
            } else {
                entry
            };
            (format, properties(members))
        })
        .collect()
}

/// The bits of `need` that `have` lacks, per format.
fn missing(have: &FormatSupport, need: &Table) -> Table {
    need.iter()
        .filter_map(|(&format, n)| {
            let h = have.formats.get(&format).copied().unwrap_or_default();
            let gap = FormatProperties {
                linear_tiling: n.linear_tiling.difference(h.linear_tiling),
                optimal_tiling: n.optimal_tiling.difference(h.optimal_tiling),
                buffer: n.buffer.difference(h.buffer),
                color_attachment_sample_counts: n
                    .color_attachment_sample_counts
                    .difference(h.color_attachment_sample_counts),
            };
            (gap != FormatProperties::default()).then_some((format, gap))
        })
        .collect()
}

/// The feature a conditional requirement names (a `VkPhysicalDeviceFeatures` member).
fn condition(member: &str) -> VkFeature {
    match member {
        "shaderStorageImageExtendedFormats" => VkFeature::ShaderStorageImageExtendedFormats,
        other => panic!("no VkFeature for the condition `{other}`: add it"),
    }
}

/// One conditional requirement: on a device with `if_feature` (every device when `None`), at
/// least one table of `any_of` holds in full.
struct Conditional {
    tables: Vec<String>,
    if_feature: Option<VkFeature>,
    any_of: Vec<Table>,
}

/// The requirements, read from the committed profiles that carry them; every carrier must agree.
fn reference(docs: &[(String, Value)]) -> (Table, Vec<Conditional>) {
    let carriers: Vec<&(String, Value)> = docs
        .iter()
        .filter(|(_, doc)| doc["capabilities"].get(BLOCK).is_some())
        .collect();
    let first = carriers
        .first()
        .expect("some committed profile carries the tables");
    for (stem, doc) in &carriers {
        assert_eq!(
            doc["capabilities"][BLOCK], first.1["capabilities"][BLOCK],
            "{stem}"
        );
        assert_eq!(
            doc["provenance"]["required_formats"]["conditional"],
            first.1["provenance"]["required_formats"]["conditional"],
            "{stem}"
        );
    }
    let unconditional = table(&first.1["capabilities"][BLOCK]["formats"], true);
    let conditional = first.1["provenance"]["required_formats"]["conditional"]
        .as_array()
        .expect("a conditional list")
        .iter()
        .map(|r| Conditional {
            tables: r["tables"]
                .as_array()
                .expect("tables")
                .iter()
                .map(|t| t.as_str().expect("an anchor").to_string())
                .collect(),
            if_feature: r
                .get("if_feature")
                .map(|f| condition(f.as_str().expect("a name"))),
            any_of: r["any_of"]
                .as_array()
                .expect("alternatives")
                .iter()
                .map(|t| table(t, false))
                .collect(),
        })
        .collect();
    (unconditional, conditional)
}

/// The tables are the specification's: spot checks against the published Vulkan 1.2 tables.
#[test]
fn the_tables_read_the_specification() {
    let docs: Vec<(String, Value)> = committed_profiles()
        .into_iter()
        .map(|(stem, text)| (stem, serde_json::from_str(&text).expect("JSON")))
        .collect();
    let (unconditional, conditional) = reference(&docs);
    use vk::FormatFeatureFlags2 as F;
    let transfer = F::TRANSFER_SRC | F::TRANSFER_DST;
    let rgba8 = unconditional[&vk::Format::R8G8B8A8_UNORM];
    assert_eq!(
        rgba8.optimal_tiling,
        transfer
            | F::SAMPLED_IMAGE
            | F::BLIT_SRC
            | F::SAMPLED_IMAGE_FILTER_LINEAR
            | F::STORAGE_IMAGE
            | F::COLOR_ATTACHMENT
            | F::BLIT_DST
            | F::COLOR_ATTACHMENT_BLEND
    );
    assert_eq!(
        rgba8.buffer,
        F::VERTEX_BUFFER | F::UNIFORM_TEXEL_BUFFER | F::STORAGE_TEXEL_BUFFER
    );
    assert!(rgba8.linear_tiling.is_empty());
    let r32 = unconditional[&vk::Format::R32_UINT];
    assert!(r32.optimal_tiling.contains(F::STORAGE_IMAGE_ATOMIC));
    assert!(r32.buffer.contains(F::STORAGE_TEXEL_BUFFER_ATOMIC));
    // A format required only as vertex input.
    assert_eq!(
        unconditional[&vk::Format::R32G32B32_SFLOAT],
        FormatProperties {
            buffer: F::VERTEX_BUFFER,
            ..FormatProperties::default()
        }
    );
    // D16 is the one depth format every device can attach; the others are choices.
    assert!(unconditional[&vk::Format::D16_UNORM]
        .optimal_tiling
        .contains(F::DEPTH_STENCIL_ATTACHMENT));
    assert!(!unconditional[&vk::Format::D32_SFLOAT]
        .optimal_tiling
        .contains(F::DEPTH_STENCIL_ATTACHMENT));
    assert_eq!(unconditional.len(), 56);
    // Storage on 8/16-bit and packed formats needs shaderStorageImageExtendedFormats.
    let extended: Vec<&Conditional> = conditional
        .iter()
        .filter(|r| r.if_feature == Some(VkFeature::ShaderStorageImageExtendedFormats))
        .collect();
    assert_eq!(extended.len(), 4);
    assert!(extended.iter().any(|r| r.any_of[0]
        .get(&vk::Format::R8_UNORM)
        .is_some_and(|p| p.optimal_tiling == F::STORAGE_IMAGE)));
    // Two depth choices and one choice between the compressed families, on every device.
    let choices: Vec<(&[String], usize)> = conditional
        .iter()
        .filter(|r| r.if_feature.is_none())
        .map(|r| (r.tables.as_slice(), r.any_of.len()))
        .collect();
    assert_eq!(choices.len(), 3);
    assert!(choices.contains(&(
        &[
            "formats-mandatory-features-astc".to_string(),
            "formats-mandatory-features-bcn".to_string(),
            "formats-mandatory-features-etc".to_string()
        ][..],
        3
    )));
}

/// Every committed profile covers the tables. A device report states one device, which meets
/// every requirement that applies to it (including a choice between formats or tables); the
/// derived floor states a lower bound, which cannot hold a choice, so only its unconditional and
/// feature-conditioned entries are checked.
#[test]
fn committed_profiles_cover_the_vulkan_1_2_required_formats() {
    let docs: Vec<(String, Value)> = committed_profiles()
        .into_iter()
        .map(|(stem, text)| (stem, serde_json::from_str(&text).expect("JSON")))
        .collect();
    let (unconditional, conditional) = reference(&docs);
    let mut gaps: BTreeMap<String, Table> = BTreeMap::new();
    let mut failures = Vec::new();
    for (stem, text) in committed_profiles() {
        let doc = &docs.iter().find(|(s, _)| *s == stem).expect("parsed").1;
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("parses");
        let report = matches!(
            doc["provenance"]["source"].as_str(),
            Some("gpuinfo" | "vulkaninfo")
        );
        let gap = missing(&desc.formats, &unconditional);
        if !gap.is_empty() {
            gaps.insert(stem.clone(), gap);
        }
        for r in &conditional {
            let applies = match r.if_feature {
                Some(feature) => desc.has(feature),
                None => report,
            };
            if applies
                && !r
                    .any_of
                    .iter()
                    .any(|t| missing(&desc.formats, t).is_empty())
            {
                failures.push(format!("{stem}: {:?} ({:?})", r.tables, r.if_feature));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(gaps.is_empty(), "{gaps:#?}");
}
