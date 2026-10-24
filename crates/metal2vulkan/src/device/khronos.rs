//! Khronos profiles-schema JSON → [`DeviceDescription`].
//!
//! Reads one profile of a `profiles-0.8` document (the committed `validation/profiles/*.json`
//! and any Vulkan-Profiles / gpuinfo export). The profile's capability blocks are merged in
//! order; a member that appears twice keeps the more capable value (`max*` the larger, `min*` the
//! smaller, booleans `or`, flag lists the union), which is how a profile's layered requirement
//! blocks combine.
//!
//! Only structures this reader knows are read, so an unrelated vendor structure can never supply
//! a value. A feature counts only when its structure is exposed: a core `VkPhysicalDeviceVulkanXY`
//! structure needs `api-version ≥ X.Y`, and an extension structure needs its extension listed
//! (or the core version that promoted it). Members the base version requires unconditionally
//! ([`required_features`]) are added when the document does not mention them.
//!
//! A limit, subgroup property, or depth/stencil resolve mode set the document omits takes the
//! value its API version guarantees (`required_limits`; `maxDrawIndexedIndexValue` is `2^32 - 1`
//! where the document states `fullDrawIndexUint32`, which the specification makes exact): profile
//! documents state only what they raise. Otherwise absence means "not supported"
//! (validation/profiles/README.md): unlisted features and format usages are unavailable, other
//! missing `max*` limits are 0, other missing alignments `u64::MAX`, and other missing booleans
//! and flag sets (`independentResolve`, the sample-location properties) state nothing.
//!
//! Where a profile's device departs from the specifications, which no Vulkan query reports,
//! is not the document's to state: its `driverID`, `vendorID` and `driverVersion` name the
//! driver and release whose measured deviations translation takes (`device::driver_facts`). A
//! document that still holds the former top-level `m2v_deviations` object is refused, so a
//! statement it makes is never silently dropped.
//!
//! A venue profile's document also states each color attachment format's sample counts, which no
//! profiles-schema structure holds, in a top-level `m2v_color_attachment_sample_counts` object
//! keyed by profile name (`FormatProperties::color_attachment_sample_counts`). A document without
//! it states none for any format.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::device::description::{
    required_features, AllocationLimits, DeviceDescription, FormatProperties, FormatSupport,
    GeneratedCommandsProperties, SubgroupProperties,
};
use crate::device::profile::{FloatControls, FloatWidthControls, Independence, Limits, MeshLimits};
use crate::device::requirements::{ApiVersion, VkExtension, VkFeature};
use crate::error::{DescriptorError, Error};
use crate::vk;

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidDescriptor(DescriptorError::DeviceDescription {
        detail: detail.into(),
    })
}

impl DeviceDescription {
    /// Parses one profile of a Khronos profiles-schema JSON document. `profile` names it; `None`
    /// requires the document to hold exactly one profile.
    pub fn from_khronos_json(
        json: &str,
        profile: Option<&str>,
    ) -> Result<DeviceDescription, Error> {
        let doc: Value =
            serde_json::from_str(json).map_err(|e| invalid(format!("profile JSON: {e}")))?;
        let profiles = doc
            .get("profiles")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("profile JSON has no `profiles` object"))?;
        let (name, prof) = match profile {
            Some(name) => (
                name,
                profiles
                    .get(name)
                    .ok_or_else(|| invalid(format!("profile `{name}` not in document")))?,
            ),
            None => {
                let mut it = profiles.iter();
                match (it.next(), it.next()) {
                    (Some((name, prof)), None) => (name.as_str(), prof),
                    _ => {
                        return Err(invalid(format!(
                            "document holds {} profiles; name one",
                            profiles.len()
                        )))
                    }
                }
            }
        };
        let api_version = parse_version(
            prof.get("api-version")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("profile `{name}` has no api-version")))?,
        )?;
        let blocks = doc
            .get("capabilities")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("profile JSON has no `capabilities` object"))?;
        let mut merged = Merged::default();
        for block in prof
            .get("capabilities")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("profile `{name}` has no capabilities list")))?
        {
            // A list entry may be an array of alternatives in the schema; a description is one
            // concrete device, so alternatives are refused rather than guessed.
            let block = block
                .as_str()
                .ok_or_else(|| invalid("capability alternatives are not a single device"))?;
            let caps = blocks
                .get(block)
                .and_then(Value::as_object)
                .ok_or_else(|| invalid(format!("capability block `{block}` missing")))?;
            merged.add(caps)?;
        }
        if doc.get("m2v_deviations").is_some() {
            return Err(invalid(
                "`m2v_deviations` is no longer read: a driver's deviations are measured facts of \
                 its release",
            ));
        }
        let mut desc = merged.describe(api_version)?;
        color_attachment_sample_counts(&doc, name, &mut desc)?;
        Ok(desc)
    }
}

/// The document's top-level object of each profile's color attachment sample counts.
const SAMPLE_COUNTS: &str = "m2v_color_attachment_sample_counts";

/// Reads the profile's entry in the document's top-level `m2v_color_attachment_sample_counts`
/// object: for each format, the `VkSampleCountFlagBits` names of the sample counts a color
/// attachment of it takes ([`FormatProperties::color_attachment_sample_counts`]), as the venue
/// capture's `corpus-vulkan --capture-formats` supplement measured them. No profiles-schema
/// structure holds `VkImageFormatProperties`. A document without the object, or without an entry
/// for this profile, states none. A format outside the pinned registry, one the profile does not
/// state as a color attachment, and a sample count outside the registry are refused, never
/// ignored.
///
/// [`FormatProperties::color_attachment_sample_counts`]:
/// crate::device::FormatProperties::color_attachment_sample_counts
fn color_attachment_sample_counts(
    doc: &Value,
    profile: &str,
    desc: &mut DeviceDescription,
) -> Result<(), Error> {
    let Some(all) = doc.get(SAMPLE_COUNTS) else {
        return Ok(());
    };
    let all = all
        .as_object()
        .ok_or_else(|| invalid(format!("`{SAMPLE_COUNTS}` is not an object")))?;
    let Some(entry) = all.get(profile) else {
        return Ok(());
    };
    let entry = entry
        .as_object()
        .ok_or_else(|| invalid(format!("`{SAMPLE_COUNTS}.{profile}` is not an object")))?;
    for (name, counts) in entry {
        let at = format!("`{SAMPLE_COUNTS}.{profile}.{name}`");
        let format = vk::Format::from_name(name)
            .ok_or_else(|| invalid(format!("{at}: a format outside the pinned registry")))?;
        let slot = desc
            .formats
            .formats
            .get_mut(&format)
            .filter(|p| {
                p.optimal_tiling
                    .contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT)
            })
            .ok_or_else(|| {
                invalid(format!(
                    "{at}: the profile states no color attachment of it"
                ))
            })?;
        let counts = counts
            .as_array()
            .ok_or_else(|| invalid(format!("{at} is not a list")))?;
        for count in counts {
            let bit = count
                .as_str()
                .and_then(vk::SampleCountFlags::from_name)
                .ok_or_else(|| invalid(format!("{at}: {count} is no sample count")))?;
            slot.color_attachment_sample_counts |= bit;
        }
    }
    Ok(())
}

fn parse_version(text: &str) -> Result<ApiVersion, Error> {
    let mut parts = text.split('.');
    let mut next = || -> Result<u32, Error> {
        parts
            .next()
            .and_then(|p| p.parse().ok())
            .ok_or_else(|| invalid(format!("bad api-version `{text}`")))
    };
    let major = next()?;
    let minor = next()?;
    Ok(ApiVersion { major, minor })
}

/// The capability blocks of one profile, merged.
#[derive(Default)]
struct Merged {
    extensions: BTreeSet<String>,
    /// structure name → member → value (explicit values only).
    features: BTreeMap<String, BTreeMap<String, bool>>,
    /// Flattened property members of the known property structures (limits included).
    properties: BTreeMap<String, Value>,
    formats: BTreeMap<String, FormatProperties>,
}

impl Merged {
    fn add(&mut self, caps: &Map<String, Value>) -> Result<(), Error> {
        if let Some(exts) = caps.get("extensions") {
            let exts = exts
                .as_object()
                .ok_or_else(|| invalid("`extensions` is not an object"))?;
            self.extensions.extend(exts.keys().cloned());
        }
        if let Some(features) = caps.get("features") {
            let features = features
                .as_object()
                .ok_or_else(|| invalid("`features` is not an object"))?;
            for (structure, members) in features {
                let members = members
                    .as_object()
                    .ok_or_else(|| invalid(format!("`{structure}` is not an object")))?;
                let slot = self.features.entry(structure.clone()).or_default();
                for (member, value) in members {
                    // Non-boolean members (e.g. pointers dropped by the converter) are ignored.
                    if let Some(v) = value.as_bool() {
                        let e = slot.entry(member.clone()).or_insert(false);
                        *e |= v;
                    }
                }
            }
        }
        if let Some(properties) = caps.get("properties") {
            let properties = properties
                .as_object()
                .ok_or_else(|| invalid("`properties` is not an object"))?;
            for (structure, members) in properties {
                if !PROPERTY_STRUCTS.contains(&structure.as_str()) {
                    continue;
                }
                let members = members
                    .as_object()
                    .ok_or_else(|| invalid(format!("`{structure}` is not an object")))?;
                for (member, value) in members {
                    if member == "limits" {
                        let limits = value
                            .as_object()
                            .ok_or_else(|| invalid("`limits` is not an object"))?;
                        for (limit, value) in limits {
                            merge_into(&mut self.properties, limit, value);
                        }
                    } else {
                        merge_into(&mut self.properties, member, value);
                    }
                }
            }
        }
        if let Some(formats) = caps.get("formats") {
            let formats = formats
                .as_object()
                .ok_or_else(|| invalid("`formats` is not an object"))?;
            for (format, props) in formats {
                let slot = self.formats.entry(format.clone()).or_default();
                // `VkFormatProperties` (also as `VkFormatProperties2::formatProperties`) states the
                // low 32 bits; `VkFormatProperties3` all of `VkFormatFeatureFlags2`.
                let legacy = [
                    props.get("VkFormatProperties"),
                    props
                        .get("VkFormatProperties2")
                        .or_else(|| props.get("VkFormatProperties2KHR"))
                        .and_then(|p| p.get("formatProperties")),
                ];
                for p in legacy.into_iter().flatten() {
                    slot.linear_tiling |= format_flags(p.get("linearTilingFeatures"))?;
                    slot.optimal_tiling |= format_flags(p.get("optimalTilingFeatures"))?;
                    slot.buffer |= format_flags(p.get("bufferFeatures"))?;
                }
                let flags2 = props
                    .get("VkFormatProperties3")
                    .or_else(|| props.get("VkFormatProperties3KHR"));
                if let Some(p) = flags2 {
                    slot.linear_tiling |= format_flags2(p.get("linearTilingFeatures"))?;
                    slot.optimal_tiling |= format_flags2(p.get("optimalTilingFeatures"))?;
                    slot.buffer |= format_flags2(p.get("bufferFeatures"))?;
                }
            }
        }
        Ok(())
    }

    fn describe(mut self, api_version: ApiVersion) -> Result<DeviceDescription, Error> {
        let index_value_omitted = !self.properties.contains_key("maxDrawIndexedIndexValue");
        // Members the document omits take the value every device of its version guarantees.
        for (name, value) in required_limits(api_version) {
            self.properties.entry(name.to_string()).or_insert(value);
        }
        let mut desc = DeviceDescription::new(api_version);
        for name in &self.extensions {
            if let Some(ext) = extension_from_name(name) {
                desc.extensions.insert(ext);
            }
        }
        // Features: explicit values from exposed structures; `true` anywhere wins.
        let mut explicit: BTreeMap<VkFeature, bool> = BTreeMap::new();
        for (structure, members) in &self.features {
            let Some(route) = feature_route(structure) else {
                continue;
            };
            if !route.exposed(api_version, &self.extensions) {
                continue;
            }
            for (member, &value) in members {
                if let Some(feature) = feature_from_member(member) {
                    let e = explicit.entry(feature).or_insert(false);
                    *e |= value;
                }
            }
        }
        for feature in required_features(api_version) {
            explicit.entry(feature).or_insert(true);
        }
        desc.features = explicit
            .into_iter()
            .filter_map(|(f, on)| on.then_some(f))
            .collect();
        // Extended dynamic state and its second extension are core in 1.3 without a feature
        // bit; before 1.3 each extension's own feature bit (read above) decides.
        if api_version >= ApiVersion::V1_3 {
            desc.features.insert(VkFeature::ExtendedDynamicState);
            desc.features.insert(VkFeature::ExtendedDynamicState2);
        }

        let p = Props(&self.properties);
        desc.driver_id = p.enum_value("driverID", vk::DriverId::from_name)?;
        desc.driver_version = p.u32("driverVersion")?;
        desc.vendor_id = p.u32("vendorID")?;
        desc.device_id = p.u32("deviceID")?;
        desc.device_name = p.string("deviceName");
        desc.device_type = p.enum_value("deviceType", vk::PhysicalDeviceType::from_name)?;
        desc.limits = limits(&p, api_version, &self.extensions)?;
        desc.allocation = AllocationLimits::new(
            p.u64_or("maxMemoryAllocationSize", 0)?,
            p.u64_or("maxBufferSize", 0)?,
        );
        // "If this feature is supported, `maxDrawIndexedIndexValue` must be `2^32 - 1`"
        // (`fullDrawIndexUint32`): a document that states the feature and omits the limit states
        // that value.
        if index_value_omitted && desc.features.contains(&VkFeature::FullDrawIndexUint32) {
            desc.limits.max_draw_indexed_index_value = u32::MAX;
        }
        desc.subgroup = SubgroupProperties {
            size: p.u32_any(&["subgroupSize"])?,
            supported_stages: p.flags_any(
                &["subgroupSupportedStages", "supportedStages"],
                vk::ShaderStageFlags::from_name,
            )?,
            supported_operations: p.flags_any(
                &["subgroupSupportedOperations", "supportedOperations"],
                vk::SubgroupFeatureFlags::from_name,
            )?,
            quad_operations_in_all_stages: p.bool_any(&[
                "subgroupQuadOperationsInAllStages",
                "quadOperationsInAllStages",
            ])?,
            min_size: p.u32("minSubgroupSize")?,
            max_size: p.u32("maxSubgroupSize")?,
            max_compute_workgroup_subgroups: p.u32("maxComputeWorkgroupSubgroups")?,
            required_size_stages: p.flags_any(
                &["requiredSubgroupSizeStages"],
                vk::ShaderStageFlags::from_name,
            )?,
        };
        desc.float_controls = float_controls(&p)?;
        desc.filter_minmax_single_component_formats =
            p.bool_any(&["filterMinmaxSingleComponentFormats"])?;
        // `VK_KHR_maintenance10`'s property, read where the extension is exposed.
        desc.rgba4_opaque_black_swizzled =
            if Route::Ext(&["VK_KHR_maintenance10"], None).exposed(api_version, &self.extensions) {
                p.bool("rgba4OpaqueBlackSwizzled")?
            } else {
                false
            };
        // A Vulkan 1.2 property (`VkPhysicalDeviceVulkan12Properties`, and the descriptor
        // indexing structure it absorbed), so every description's version exposes it; a document
        // that does not state it reads false, which holds it to the rule it gates.
        desc.robust_buffer_access_update_after_bind =
            p.bool("robustBufferAccessUpdateAfterBind")?;
        // `VkFormatProperties3` is core in 1.3 and the extension's before.
        desc.khr_format_feature_flags2 = self.extensions.contains("VK_KHR_format_feature_flags2");
        // `VK_EXT_device_generated_commands` where the document lists it with the EXT
        // structure's `deviceGeneratedCommands` true; a document that omits a stage list states no
        // stage.
        let generated = self.extensions.contains("VK_EXT_device_generated_commands")
            && self
                .features
                .get("VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT")
                .and_then(|members| members.get("deviceGeneratedCommands"))
                .copied()
                .unwrap_or(false);
        desc.generated_commands = if generated {
            Some(GeneratedCommandsProperties::new(
                [
                    p.u32("maxIndirectPipelineCount")?,
                    p.u32("maxIndirectSequenceCount")?,
                    p.u32("maxIndirectCommandsTokenCount")?,
                    p.u32("maxIndirectCommandsTokenOffset")?,
                    p.u32("maxIndirectCommandsIndirectStride")?,
                ],
                p.flags_any(
                    &["supportedIndirectCommandsShaderStages"],
                    vk::ShaderStageFlags::from_name,
                )?,
                p.flags_any(
                    &["supportedIndirectCommandsShaderStagesPipelineBinding"],
                    vk::ShaderStageFlags::from_name,
                )?,
            ))
        } else {
            None
        };
        let mut formats = FormatSupport::default();
        for (name, props) in self.formats {
            // Formats outside the pinned registry cannot be named by the core; skip them.
            if let Some(format) = vk::Format::from_name(&name) {
                formats.formats.insert(format, props);
            }
        }
        desc.formats = formats;
        Ok(desc)
    }
}

/// Merges one member into the flattened map, keeping the more capable value.
fn merge_into(map: &mut BTreeMap<String, Value>, name: &str, value: &Value) {
    let merged = match map.get(name) {
        None => value.clone(),
        Some(old) => merge_value(name, old, value),
    };
    map.insert(name.to_string(), merged);
}

fn merge_value(name: &str, old: &Value, new: &Value) -> Value {
    match (old, new) {
        (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
        (Value::Number(_), Value::Number(_)) => {
            let (a, b) = (number_key(old), number_key(new));
            let keep_old = if name.starts_with("max") {
                a > b
            } else if name.starts_with("min") {
                a < b
            } else {
                false
            };
            if keep_old {
                old.clone()
            } else {
                new.clone()
            }
        }
        (Value::Array(a), Value::Array(b))
            if a.len() == b.len() && a.iter().all(Value::is_number) =>
        {
            Value::Array(
                a.iter()
                    .zip(b)
                    .map(|(x, y)| merge_value(name, x, y))
                    .collect(),
            )
        }
        (Value::Array(a), Value::Array(b)) if a.iter().chain(b).all(Value::is_string) => {
            let set: BTreeSet<&str> = a.iter().chain(b).filter_map(Value::as_str).collect();
            Value::Array(set.into_iter().map(|s| Value::String(s.into())).collect())
        }
        _ => new.clone(),
    }
}

/// A total order key for JSON numbers (exact for integers).
fn number_key(v: &Value) -> (i128, f64) {
    if let Some(u) = v.as_u64() {
        (i128::from(u), 0.0)
    } else if let Some(i) = v.as_i64() {
        (i128::from(i), 0.0)
    } else {
        let f = v.as_f64().unwrap_or(0.0);
        (f.floor() as i128, f - f.floor())
    }
}

/// A `VkFormatProperties` member's list of `VkFormatFeatureFlagBits` names, as the low 32 bits of
/// `VkFormatFeatureFlags2`.
fn format_flags(v: Option<&Value>) -> Result<vk::FormatFeatureFlags2, Error> {
    let mut flags = vk::FormatFeatureFlags::empty();
    for name in feature_names(v)? {
        // Bits outside the pinned registry are dropped.
        if let Some(bit) = vk::FormatFeatureFlags::from_name(name) {
            flags |= bit;
        }
    }
    Ok(crate::device::description::widen(flags))
}

/// A `VkFormatProperties3` member's list of `VkFormatFeatureFlagBits2` names.
fn format_flags2(v: Option<&Value>) -> Result<vk::FormatFeatureFlags2, Error> {
    let mut flags = vk::FormatFeatureFlags2::empty();
    for name in feature_names(v)? {
        // Bits outside the pinned registry are dropped.
        if let Some(bit) = vk::FormatFeatureFlags2::from_name(name) {
            flags |= bit;
        }
    }
    Ok(flags)
}

/// The names of a format feature list (none for an absent member).
fn feature_names(v: Option<&Value>) -> Result<Vec<&str>, Error> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    let list = v
        .as_array()
        .ok_or_else(|| invalid("format features are not a list"))?;
    list.iter()
        .map(|name| {
            name.as_str()
                .ok_or_else(|| invalid("format feature is not a string"))
        })
        .collect()
}

/// Flattened property members.
struct Props<'a>(&'a BTreeMap<String, Value>);

impl Props<'_> {
    fn get(&self, name: &str) -> Option<&Value> {
        self.0.get(name)
    }

    /// The string member `name`, empty where absent or not a string.
    fn string(&self, name: &str) -> String {
        self.get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }

    fn u64_or(&self, name: &str, absent: u64) -> Result<u64, Error> {
        match self.get(name) {
            None => Ok(absent),
            Some(v) => v
                .as_u64()
                .or_else(|| v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
                .ok_or_else(|| invalid(format!("`{name}` is not an unsigned integer"))),
        }
    }

    fn u32(&self, name: &str) -> Result<u32, Error> {
        let v = self.u64_or(name, 0)?;
        // Saturate: a limit beyond u32 (e.g. gpuinfo's unbounded counts) is at least u32::MAX.
        Ok(u32::try_from(v).unwrap_or(u32::MAX))
    }

    fn u32_any(&self, names: &[&str]) -> Result<u32, Error> {
        for name in names {
            if self.get(name).is_some() {
                return self.u32(name);
            }
        }
        Ok(0)
    }

    fn i32(&self, name: &str) -> Result<i32, Error> {
        match self.get(name) {
            None => Ok(0),
            Some(v) => v
                .as_i64()
                .and_then(|i| i32::try_from(i).ok())
                .ok_or_else(|| invalid(format!("`{name}` is not a 32-bit integer"))),
        }
    }

    fn f32_bits(&self, name: &str) -> Result<u32, Error> {
        match self.get(name) {
            None => Ok(0f32.to_bits()),
            Some(v) => v
                .as_f64()
                .map(|f| (f as f32).to_bits())
                .ok_or_else(|| invalid(format!("`{name}` is not a number"))),
        }
    }

    /// The upper bound of the 2-element range `name`, rounded down to a whole number (0 where
    /// it is absent): `Limits::max_point_size`.
    fn max_point_size(&self, name: &str) -> Result<u32, Error> {
        // Vulkan's bound without `largePoints`: `pointSizeRange` (1.0, 1.0).
        let Some(v) = self.get(name) else {
            return Ok(1);
        };
        let high = v
            .as_array()
            .filter(|l| l.len() == 2)
            .and_then(|l| l[1].as_f64())
            .ok_or_else(|| invalid(format!("`{name}` is not a 2-element list of numbers")))?;
        Ok(crate::device::whole_point_size(high as f32))
    }

    fn u32x3(&self, name: &str) -> Result<[u32; 3], Error> {
        let Some(v) = self.get(name) else {
            return Ok([0; 3]);
        };
        let list = v
            .as_array()
            .filter(|l| l.len() == 3)
            .ok_or_else(|| invalid(format!("`{name}` is not a 3-element list")))?;
        let mut out = [0u32; 3];
        for (slot, v) in out.iter_mut().zip(list) {
            *slot = v
                .as_u64()
                .map(|u| u32::try_from(u).unwrap_or(u32::MAX))
                .ok_or_else(|| invalid(format!("`{name}` holds a non-integer")))?;
        }
        Ok(out)
    }

    fn bool(&self, name: &str) -> Result<bool, Error> {
        self.bool_any(&[name])
    }

    fn bool_any(&self, names: &[&str]) -> Result<bool, Error> {
        for name in names {
            if let Some(v) = self.get(name) {
                return v
                    .as_bool()
                    .ok_or_else(|| invalid(format!("`{name}` is not a boolean")));
            }
        }
        Ok(false)
    }

    fn flags_any<F: Copy + std::ops::BitOrAssign + Default>(
        &self,
        names: &[&str],
        from_name: fn(&str) -> Option<F>,
    ) -> Result<F, Error> {
        let mut flags = F::default();
        for name in names {
            let Some(v) = self.get(name) else {
                continue;
            };
            let list = v
                .as_array()
                .ok_or_else(|| invalid(format!("`{name}` is not a flag list")))?;
            for bit in list {
                let bit = bit
                    .as_str()
                    .ok_or_else(|| invalid(format!("`{name}` holds a non-string")))?;
                // Bits the pinned registry does not name are dropped (the converter already
                // drops them; see provenance.unmapped).
                if let Some(bit) = from_name(bit) {
                    flags |= bit;
                }
            }
        }
        // Every spelling present contributes (the core and extension structures agree).
        Ok(flags)
    }

    fn enum_value<E>(
        &self,
        name: &str,
        from_name: fn(&str) -> Option<E>,
    ) -> Result<Option<E>, Error> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(from_name(s)),
            Some(_) => Err(invalid(format!("`{name}` is not an enumerant name"))),
        }
    }
}

fn limits(p: &Props<'_>, api: ApiVersion, extensions: &BTreeSet<String>) -> Result<Limits, Error> {
    let align = |name: &str| p.u64_or(name, u64::MAX);
    // `supportsNonZeroFirstInstance` exists where the KHR divisor (or 1.4) is exposed. An
    // EXT-only divisor cannot report it, and drivers implementing the EXT do carry the
    // restriction (lavapipe exposes the EXT and reports `false`), so it reads `false`.
    let supports_non_zero_first_instance =
        if Route::Ext(&["VK_KHR_vertex_attribute_divisor"], Some(4)).exposed(api, extensions) {
            p.bool("supportsNonZeroFirstInstance")?
        } else {
            false
        };
    // The portability subset's stride rule exists exactly where the extension is exposed (and
    // so must be enabled); an unreported alignment is the most restrictive.
    let min_vertex_input_binding_stride_alignment =
        if Route::Ext(&["VK_KHR_portability_subset"], None).exposed(api, extensions) {
            let a = align("minVertexInputBindingStrideAlignment")?;
            Some(u32::try_from(a).unwrap_or(u32::MAX))
        } else {
            None
        };
    // The acceleration-structure limits exist where `VK_KHR_acceleration_structure` is exposed;
    // an omitted one there is the required minimum, 16.
    // The sample-location properties exist where `VK_EXT_sample_locations` is exposed; elsewhere
    // no count takes custom locations.
    let sample_locations = Route::Ext(&["VK_EXT_sample_locations"], None).exposed(api, extensions);
    // The provoking-vertex property exists where `VK_EXT_provoking_vertex` is exposed.
    let provoking_vertex = Route::Ext(&["VK_EXT_provoking_vertex"], None).exposed(api, extensions);
    let accel = Route::Ext(&["VK_KHR_acceleration_structure"], None).exposed(api, extensions);
    let accel_limit = |name: &str| -> Result<u32, Error> {
        if accel {
            Ok(u32::try_from(p.u64_or(name, 16)?).unwrap_or(u32::MAX))
        } else {
            Ok(0)
        }
    };
    Ok(Limits {
        max_compute_work_group_size: p.u32x3("maxComputeWorkGroupSize")?,
        max_compute_work_group_count: p.u32x3("maxComputeWorkGroupCount")?,
        max_compute_work_group_invocations: p.u32("maxComputeWorkGroupInvocations")?,
        max_compute_shared_memory_size: p.u32("maxComputeSharedMemorySize")?,
        max_push_constants_size: p.u32("maxPushConstantsSize")?,
        max_push_descriptors: p.u32("maxPushDescriptors")?,
        max_bound_descriptor_sets: p.u32("maxBoundDescriptorSets")?,
        max_per_stage_descriptor_samplers: p.u32("maxPerStageDescriptorSamplers")?,
        max_per_stage_descriptor_uniform_buffers: p.u32("maxPerStageDescriptorUniformBuffers")?,
        max_per_stage_descriptor_storage_buffers: p.u32("maxPerStageDescriptorStorageBuffers")?,
        max_per_stage_descriptor_sampled_images: p.u32("maxPerStageDescriptorSampledImages")?,
        max_per_stage_descriptor_storage_images: p.u32("maxPerStageDescriptorStorageImages")?,
        max_per_stage_descriptor_input_attachments: p
            .u32("maxPerStageDescriptorInputAttachments")?,
        max_per_stage_resources: p.u32("maxPerStageResources")?,
        max_descriptor_set_samplers: p.u32("maxDescriptorSetSamplers")?,
        max_descriptor_set_uniform_buffers: p.u32("maxDescriptorSetUniformBuffers")?,
        max_descriptor_set_storage_buffers: p.u32("maxDescriptorSetStorageBuffers")?,
        max_descriptor_set_sampled_images: p.u32("maxDescriptorSetSampledImages")?,
        max_descriptor_set_storage_images: p.u32("maxDescriptorSetStorageImages")?,
        max_descriptor_set_input_attachments: p.u32("maxDescriptorSetInputAttachments")?,
        max_descriptor_set_update_after_bind_samplers: p
            .u32("maxDescriptorSetUpdateAfterBindSamplers")?,
        max_descriptor_set_update_after_bind_sampled_images: p
            .u32("maxDescriptorSetUpdateAfterBindSampledImages")?,
        max_descriptor_set_update_after_bind_storage_images: p
            .u32("maxDescriptorSetUpdateAfterBindStorageImages")?,
        max_descriptor_set_update_after_bind_storage_buffers: p
            .u32("maxDescriptorSetUpdateAfterBindStorageBuffers")?,
        max_per_stage_descriptor_update_after_bind_samplers: p
            .u32("maxPerStageDescriptorUpdateAfterBindSamplers")?,
        max_per_stage_descriptor_update_after_bind_sampled_images: p
            .u32("maxPerStageDescriptorUpdateAfterBindSampledImages")?,
        max_per_stage_descriptor_update_after_bind_storage_images: p
            .u32("maxPerStageDescriptorUpdateAfterBindStorageImages")?,
        max_per_stage_descriptor_update_after_bind_storage_buffers: p
            .u32("maxPerStageDescriptorUpdateAfterBindStorageBuffers")?,
        max_per_stage_update_after_bind_resources: p.u32("maxPerStageUpdateAfterBindResources")?,
        max_storage_buffer_range: p.u32("maxStorageBufferRange")?,
        max_uniform_buffer_range: p.u32("maxUniformBufferRange")?,
        min_storage_buffer_offset_alignment: align("minStorageBufferOffsetAlignment")?,
        min_uniform_buffer_offset_alignment: align("minUniformBufferOffsetAlignment")?,
        min_texel_buffer_offset_alignment: align("minTexelBufferOffsetAlignment")?,
        max_texel_buffer_elements: p.u32("maxTexelBufferElements")?,
        max_sampler_lod_bias_bits: p.f32_bits("maxSamplerLodBias")?,
        max_sampler_anisotropy_bits: p.f32_bits("maxSamplerAnisotropy")?,
        min_texel_offset: p.i32("minTexelOffset")?,
        max_texel_offset: p.u32("maxTexelOffset")?,
        min_texel_gather_offset: p.i32("minTexelGatherOffset")?,
        max_texel_gather_offset: p.u32("maxTexelGatherOffset")?,
        min_interpolation_offset_bits: p.f32_bits("minInterpolationOffset")?,
        max_interpolation_offset_bits: p.f32_bits("maxInterpolationOffset")?,
        max_image_dimension_1d: p.u32("maxImageDimension1D")?,
        max_image_dimension_2d: p.u32("maxImageDimension2D")?,
        max_image_dimension_3d: p.u32("maxImageDimension3D")?,
        max_image_dimension_cube: p.u32("maxImageDimensionCube")?,
        max_image_array_layers: p.u32("maxImageArrayLayers")?,
        max_vertex_input_attributes: p.u32("maxVertexInputAttributes")?,
        max_vertex_input_bindings: p.u32("maxVertexInputBindings")?,
        max_vertex_input_attribute_offset: p.u32("maxVertexInputAttributeOffset")?,
        max_vertex_input_binding_stride: p.u32("maxVertexInputBindingStride")?,
        max_vertex_output_components: p.u32("maxVertexOutputComponents")?,
        max_tessellation_generation_level: p.u32("maxTessellationGenerationLevel")?,
        max_tessellation_evaluation_output_components: p
            .u32("maxTessellationEvaluationOutputComponents")?,
        max_fragment_input_components: p.u32("maxFragmentInputComponents")?,
        max_fragment_output_attachments: p.u32("maxFragmentOutputAttachments")?,
        max_fragment_dual_src_attachments: p.u32("maxFragmentDualSrcAttachments")?,
        max_fragment_combined_output_resources: p.u32("maxFragmentCombinedOutputResources")?,
        max_color_attachments: p.u32("maxColorAttachments")?,
        max_clip_distances: p.u32("maxClipDistances")?,
        max_cull_distances: p.u32("maxCullDistances")?,
        max_combined_clip_and_cull_distances: p.u32("maxCombinedClipAndCullDistances")?,
        max_vertex_attrib_divisor: p.u32("maxVertexAttribDivisor")?,
        max_per_stage_descriptor_acceleration_structures: accel_limit(
            "maxPerStageDescriptorAccelerationStructures",
        )?,
        max_descriptor_set_acceleration_structures: accel_limit(
            "maxDescriptorSetAccelerationStructures",
        )?,
        supports_non_zero_first_instance,
        min_vertex_input_binding_stride_alignment,
        framebuffer_color_sample_counts: p.flags_any(
            &["framebufferColorSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        framebuffer_depth_sample_counts: p.flags_any(
            &["framebufferDepthSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        // Absent is 0: no view count is known to be supported.
        max_multiview_view_count: p.u32("maxMultiviewViewCount")?,
        max_multiview_instance_index: p.u32("maxMultiviewInstanceIndex")?,
        framebuffer_integer_color_sample_counts: p.flags_any(
            &["framebufferIntegerColorSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        framebuffer_stencil_sample_counts: p.flags_any(
            &["framebufferStencilSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        framebuffer_no_attachments_sample_counts: p.flags_any(
            &["framebufferNoAttachmentsSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        storage_image_sample_counts: p.flags_any(
            &["storageImageSampleCounts"],
            vk::SampleCountFlags::from_name,
        )?,
        max_framebuffer_width: p.u32("maxFramebufferWidth")?,
        max_framebuffer_height: p.u32("maxFramebufferHeight")?,
        max_framebuffer_layers: p.u32("maxFramebufferLayers")?,
        max_draw_indexed_index_value: p.u32("maxDrawIndexedIndexValue")?,
        max_tessellation_patch_size: p.u32("maxTessellationPatchSize")?,
        max_geometry_input_components: p.u32("maxGeometryInputComponents")?,
        max_geometry_output_components: p.u32("maxGeometryOutputComponents")?,
        max_geometry_total_output_components: p.u32("maxGeometryTotalOutputComponents")?,
        supported_depth_resolve_modes: p.flags_any(
            &["supportedDepthResolveModes"],
            vk::ResolveModeFlags::from_name,
        )?,
        supported_stencil_resolve_modes: p.flags_any(
            &["supportedStencilResolveModes"],
            vk::ResolveModeFlags::from_name,
        )?,
        independent_resolve_none: p.bool("independentResolveNone")?,
        independent_resolve: p.bool("independentResolve")?,
        sample_location_sample_counts: if sample_locations {
            p.flags_any(
                &["sampleLocationSampleCounts"],
                vk::SampleCountFlags::from_name,
            )?
        } else {
            vk::SampleCountFlags::default()
        },
        variable_sample_locations: sample_locations && p.bool("variableSampleLocations")?,
        provoking_vertex_mode_per_pipeline: provoking_vertex
            && p.bool("provokingVertexModePerPipeline")?,
        mesh: if Route::Ext(&["VK_EXT_mesh_shader"], None).exposed(api, extensions) {
            mesh_limits(p)?
        } else {
            MeshLimits::default()
        },
        max_point_size: p.max_point_size("pointSizeRange")?,
    })
}

/// `VkPhysicalDeviceMeshShaderPropertiesEXT`, read where `VK_EXT_mesh_shader` is exposed.
fn mesh_limits(p: &Props<'_>) -> Result<MeshLimits, Error> {
    Ok(MeshLimits {
        max_task_work_group_total_count: p.u32("maxTaskWorkGroupTotalCount")?,
        max_task_work_group_count: p.u32x3("maxTaskWorkGroupCount")?,
        max_task_work_group_invocations: p.u32("maxTaskWorkGroupInvocations")?,
        max_task_work_group_size: p.u32x3("maxTaskWorkGroupSize")?,
        max_task_payload_size: p.u32("maxTaskPayloadSize")?,
        max_task_shared_memory_size: p.u32("maxTaskSharedMemorySize")?,
        max_task_payload_and_shared_memory_size: p.u32("maxTaskPayloadAndSharedMemorySize")?,
        max_mesh_work_group_total_count: p.u32("maxMeshWorkGroupTotalCount")?,
        max_mesh_work_group_count: p.u32x3("maxMeshWorkGroupCount")?,
        max_mesh_work_group_invocations: p.u32("maxMeshWorkGroupInvocations")?,
        max_mesh_work_group_size: p.u32x3("maxMeshWorkGroupSize")?,
        max_mesh_shared_memory_size: p.u32("maxMeshSharedMemorySize")?,
        max_mesh_payload_and_shared_memory_size: p.u32("maxMeshPayloadAndSharedMemorySize")?,
        max_mesh_output_memory_size: p.u32("maxMeshOutputMemorySize")?,
        max_mesh_payload_and_output_memory_size: p.u32("maxMeshPayloadAndOutputMemorySize")?,
        max_mesh_output_components: p.u32("maxMeshOutputComponents")?,
        max_mesh_output_vertices: p.u32("maxMeshOutputVertices")?,
        max_mesh_output_primitives: p.u32("maxMeshOutputPrimitives")?,
        max_mesh_output_layers: p.u32("maxMeshOutputLayers")?,
        max_mesh_multiview_view_count: p.u32("maxMeshMultiviewViewCount")?,
    })
}

fn float_controls(p: &Props<'_>) -> Result<FloatControls, Error> {
    let width = |bits: &str| -> Result<FloatWidthControls, Error> {
        Ok(FloatWidthControls {
            denorm_preserve: p.bool(&format!("shaderDenormPreserveFloat{bits}"))?,
            denorm_flush_to_zero: p.bool(&format!("shaderDenormFlushToZeroFloat{bits}"))?,
            rounding_rte: p.bool(&format!("shaderRoundingModeRTEFloat{bits}"))?,
            rounding_rtz: p.bool(&format!("shaderRoundingModeRTZFloat{bits}"))?,
            signed_zero_inf_nan_preserve: p
                .bool(&format!("shaderSignedZeroInfNanPreserveFloat{bits}"))?,
        })
    };
    let independence = |name: &str| -> Result<Independence, Error> {
        let raw = p.enum_value(name, vk::ShaderFloatControlsIndependence::from_name)?;
        Ok(match raw {
            Some(vk::ShaderFloatControlsIndependence::ALL) => Independence::All,
            Some(vk::ShaderFloatControlsIndependence::INDEPENDENCE_32_BIT_ONLY) => {
                Independence::Only32Bit
            }
            // `NONE`, an unknown value, or absence: the most constrained contract.
            _ => Independence::None,
        })
    };
    Ok(FloatControls {
        f16: width("16")?,
        f32: width("32")?,
        f64: width("64")?,
        denorm_independence: independence("denormBehaviorIndependence")?,
        rounding_independence: independence("roundingModeIndependence")?,
    })
}

/// The limits and subgroup properties every device of `version` guarantees, for the members the
/// core reads: the Vulkan "Required Limits" table as transcribed by Vulkan-Profiles
/// `VP_LUNARG_minimum_requirements_1_2` (the source of the committed `vk12-floor` profile), plus
/// `maxBufferSize ≥ 2^30` from Vulkan 1.3. A Khronos *profile* document states only what it
/// raises above its base version, so an omitted member means this value, never "unsupported".
/// Device reports list every member and are not affected.
fn required_limits(version: ApiVersion) -> Vec<(&'static str, Value)> {
    let n = |v: u64| Value::from(v);
    let f = |v: f64| Value::from(v);
    let list = |names: &[&str]| Value::Array(names.iter().map(|s| Value::from(*s)).collect());
    let mut out = vec![
        ("maxImageDimension1D", n(4096)),
        ("maxImageDimension2D", n(4096)),
        ("maxImageDimension3D", n(256)),
        ("maxImageDimensionCube", n(4096)),
        ("maxImageArrayLayers", n(256)),
        ("maxTexelBufferElements", n(65536)),
        ("maxUniformBufferRange", n(16384)),
        ("maxStorageBufferRange", n(1 << 27)),
        ("maxPushConstantsSize", n(128)),
        ("maxBoundDescriptorSets", n(4)),
        ("maxPerStageDescriptorSamplers", n(16)),
        ("maxPerStageDescriptorUniformBuffers", n(12)),
        ("maxPerStageDescriptorStorageBuffers", n(4)),
        ("maxPerStageDescriptorSampledImages", n(16)),
        ("maxPerStageDescriptorStorageImages", n(4)),
        ("maxPerStageDescriptorInputAttachments", n(4)),
        ("maxPerStageResources", n(128)),
        ("maxDescriptorSetSamplers", n(96)),
        ("maxDescriptorSetUniformBuffers", n(72)),
        ("maxDescriptorSetStorageBuffers", n(24)),
        ("maxDescriptorSetSampledImages", n(96)),
        ("maxDescriptorSetStorageImages", n(24)),
        ("maxDescriptorSetInputAttachments", n(4)),
        ("maxVertexInputAttributes", n(16)),
        ("maxVertexInputBindings", n(16)),
        ("maxVertexInputAttributeOffset", n(2047)),
        ("maxVertexInputBindingStride", n(2048)),
        ("maxVertexOutputComponents", n(64)),
        ("maxFragmentInputComponents", n(64)),
        ("maxFragmentOutputAttachments", n(4)),
        ("maxFragmentDualSrcAttachments", n(0)),
        ("maxFragmentCombinedOutputResources", n(4)),
        ("maxComputeSharedMemorySize", n(16384)),
        ("maxComputeWorkGroupCount", Value::from(vec![65535u64; 3])),
        ("maxComputeWorkGroupInvocations", n(128)),
        (
            "maxComputeWorkGroupSize",
            Value::from(vec![128u64, 128, 64]),
        ),
        ("maxSamplerLodBias", f(2.0)),
        ("maxSamplerAnisotropy", f(1.0)),
        ("minTexelBufferOffsetAlignment", n(256)),
        ("minUniformBufferOffsetAlignment", n(256)),
        ("minStorageBufferOffsetAlignment", n(256)),
        ("minTexelOffset", Value::from(-8i64)),
        ("maxTexelOffset", n(7)),
        ("minTexelGatherOffset", Value::from(-8i64)),
        ("maxTexelGatherOffset", n(7)),
        ("minInterpolationOffset", f(0.0)),
        ("maxInterpolationOffset", f(0.0)),
        (
            "framebufferColorSampleCounts",
            list(&["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]),
        ),
        (
            "framebufferDepthSampleCounts",
            list(&["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]),
        ),
        (
            "framebufferStencilSampleCounts",
            list(&["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]),
        ),
        (
            "framebufferNoAttachmentsSampleCounts",
            list(&["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]),
        ),
        (
            "framebufferIntegerColorSampleCounts",
            list(&["VK_SAMPLE_COUNT_1_BIT"]),
        ),
        ("maxColorAttachments", n(4)),
        ("maxFramebufferWidth", n(4096)),
        ("maxFramebufferHeight", n(4096)),
        ("maxFramebufferLayers", n(256)),
        // `2^32 - 1` where the document states `fullDrawIndexUint32` (`Merged::describe`).
        ("maxDrawIndexedIndexValue", n((1 << 24) - 1)),
        // Without `shaderStorageImageMultisample`; a device with it states its own counts.
        ("storageImageSampleCounts", list(&["VK_SAMPLE_COUNT_1_BIT"])),
        ("maxClipDistances", n(0)),
        ("maxCullDistances", n(0)),
        ("maxCombinedClipAndCullDistances", n(0)),
        ("subgroupSize", n(1)),
        (
            "subgroupSupportedStages",
            list(&["VK_SHADER_STAGE_COMPUTE_BIT"]),
        ),
        (
            "subgroupSupportedOperations",
            list(&["VK_SUBGROUP_FEATURE_BASIC_BIT"]),
        ),
        ("maxMemoryAllocationSize", n(1 << 30)),
        // `VkPhysicalDeviceDepthStencilResolveProperties` (core in 1.2, the floor):
        // "`VK_RESOLVE_MODE_SAMPLE_ZERO_BIT` must be included in the set" of each.
        (
            "supportedDepthResolveModes",
            list(&["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]),
        ),
        (
            "supportedStencilResolveModes",
            list(&["VK_RESOLVE_MODE_SAMPLE_ZERO_BIT"]),
        ),
    ];
    if version >= ApiVersion::V1_3 {
        out.push(("maxBufferSize", n(1 << 30)));
    }
    out
}

/// The property structures read (core, and the extension structures core ones replaced).
const PROPERTY_STRUCTS: &[&str] = &[
    "VkPhysicalDeviceProperties",
    "VkPhysicalDeviceVulkan11Properties",
    "VkPhysicalDeviceVulkan12Properties",
    "VkPhysicalDeviceVulkan13Properties",
    "VkPhysicalDeviceVulkan14Properties",
    "VkPhysicalDeviceSubgroupProperties",
    "VkPhysicalDeviceMaintenance3Properties",
    "VkPhysicalDeviceMaintenance3PropertiesKHR",
    "VkPhysicalDeviceDriverProperties",
    "VkPhysicalDeviceDriverPropertiesKHR",
    "VkPhysicalDeviceFloatControlsProperties",
    "VkPhysicalDeviceFloatControlsPropertiesKHR",
    "VkPhysicalDeviceDescriptorIndexingProperties",
    "VkPhysicalDeviceDescriptorIndexingPropertiesEXT",
    "VkPhysicalDeviceSamplerFilterMinmaxProperties",
    "VkPhysicalDeviceSamplerFilterMinmaxPropertiesEXT",
    "VkPhysicalDeviceSubgroupSizeControlProperties",
    "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT",
    "VkPhysicalDeviceMaintenance4Properties",
    "VkPhysicalDeviceMaintenance4PropertiesKHR",
    "VkPhysicalDevicePushDescriptorProperties",
    "VkPhysicalDevicePushDescriptorPropertiesKHR",
    "VkPhysicalDeviceVertexAttributeDivisorProperties",
    "VkPhysicalDeviceVertexAttributeDivisorPropertiesKHR",
    "VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT",
    "VkPhysicalDevicePortabilitySubsetPropertiesKHR",
    "VkPhysicalDeviceMaintenance10PropertiesKHR",
    "VkPhysicalDeviceAccelerationStructurePropertiesKHR",
    "VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT",
    "VkPhysicalDeviceDepthStencilResolveProperties",
    "VkPhysicalDeviceDepthStencilResolvePropertiesKHR",
    "VkPhysicalDeviceSampleLocationsPropertiesEXT",
    "VkPhysicalDeviceMeshShaderPropertiesEXT",
    "VkPhysicalDeviceProvokingVertexPropertiesEXT",
];

/// How a feature structure is exposed.
#[derive(Clone, Copy)]
enum Route {
    /// A core structure, valid from this version.
    Core(u32),
    /// An extension structure: valid when any extension that defines it (or an alias of it) is
    /// available, and from the core version that promoted it. Exporters name a structure by its
    /// newest alias, so every defining extension is listed.
    Ext(&'static [&'static str], Option<u32>),
}

impl Route {
    fn exposed(self, api: ApiVersion, extensions: &BTreeSet<String>) -> bool {
        let core = |minor: u32| api >= ApiVersion { major: 1, minor };
        match self {
            Route::Core(minor) => core(minor),
            Route::Ext(exts, promoted) => {
                exts.iter().any(|e| extensions.contains(*e)) || promoted.is_some_and(core)
            }
        }
    }
}

/// The feature structures read, by every spelling the schema and gpuinfo use.
fn feature_route(structure: &str) -> Option<Route> {
    use Route::{Core, Ext};
    Some(match structure {
        "VkPhysicalDeviceFeatures" => Core(0),
        "VkPhysicalDeviceVulkan11Features" | "VkPhysicalDeviceVulkan12Features" => Core(2),
        "VkPhysicalDeviceVulkan13Features" => Core(3),
        "VkPhysicalDeviceVulkan14Features" => Core(4),
        "VkPhysicalDevice16BitStorageFeatures" | "VkPhysicalDevice16BitStorageFeaturesKHR" => {
            Ext(&["VK_KHR_16bit_storage"], Some(1))
        }
        "VkPhysicalDeviceMultiviewFeatures" | "VkPhysicalDeviceMultiviewFeaturesKHR" => {
            Ext(&["VK_KHR_multiview"], Some(1))
        }
        "VkPhysicalDeviceShaderDrawParametersFeatures"
        | "VkPhysicalDeviceShaderDrawParameterFeatures" => {
            Ext(&["VK_KHR_shader_draw_parameters"], Some(1))
        }
        "VkPhysicalDevice8BitStorageFeatures" | "VkPhysicalDevice8BitStorageFeaturesKHR" => {
            Ext(&["VK_KHR_8bit_storage"], Some(2))
        }
        "VkPhysicalDeviceShaderAtomicInt64Features"
        | "VkPhysicalDeviceShaderAtomicInt64FeaturesKHR" => {
            Ext(&["VK_KHR_shader_atomic_int64"], Some(2))
        }
        "VkPhysicalDeviceShaderFloat16Int8Features"
        | "VkPhysicalDeviceShaderFloat16Int8FeaturesKHR"
        | "VkPhysicalDeviceFloat16Int8FeaturesKHR" => Ext(&["VK_KHR_shader_float16_int8"], Some(2)),
        "VkPhysicalDeviceDescriptorIndexingFeatures"
        | "VkPhysicalDeviceDescriptorIndexingFeaturesEXT" => {
            Ext(&["VK_EXT_descriptor_indexing"], Some(2))
        }
        "VkPhysicalDeviceScalarBlockLayoutFeatures"
        | "VkPhysicalDeviceScalarBlockLayoutFeaturesEXT" => {
            Ext(&["VK_EXT_scalar_block_layout"], Some(2))
        }
        "VkPhysicalDeviceUniformBufferStandardLayoutFeatures"
        | "VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR" => {
            Ext(&["VK_KHR_uniform_buffer_standard_layout"], Some(2))
        }
        "VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures"
        | "VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR" => {
            Ext(&["VK_KHR_shader_subgroup_extended_types"], Some(2))
        }
        // The EXT buffer-device-address structure is a different extension with different
        // capture semantics; only the KHR/core one is read.
        "VkPhysicalDeviceBufferDeviceAddressFeatures"
        | "VkPhysicalDeviceBufferDeviceAddressFeaturesKHR" => {
            Ext(&["VK_KHR_buffer_device_address"], Some(2))
        }
        "VkPhysicalDeviceVulkanMemoryModelFeatures"
        | "VkPhysicalDeviceVulkanMemoryModelFeaturesKHR" => {
            Ext(&["VK_KHR_vulkan_memory_model"], Some(2))
        }
        "VkPhysicalDeviceImageRobustnessFeatures"
        | "VkPhysicalDeviceImageRobustnessFeaturesEXT" => {
            Ext(&["VK_EXT_image_robustness"], Some(3))
        }
        "VkPhysicalDeviceSubgroupSizeControlFeatures"
        | "VkPhysicalDeviceSubgroupSizeControlFeaturesEXT" => {
            Ext(&["VK_EXT_subgroup_size_control"], Some(3))
        }
        "VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures"
        | "VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT" => {
            Ext(&["VK_EXT_shader_demote_to_helper_invocation"], Some(3))
        }
        "VkPhysicalDeviceShaderTerminateInvocationFeatures"
        | "VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR" => {
            Ext(&["VK_KHR_shader_terminate_invocation"], Some(3))
        }
        "VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures"
        | "VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR" => {
            Ext(&["VK_KHR_zero_initialize_workgroup_memory"], Some(3))
        }
        "VkPhysicalDeviceSynchronization2Features"
        | "VkPhysicalDeviceSynchronization2FeaturesKHR" => {
            Ext(&["VK_KHR_synchronization2"], Some(3))
        }
        "VkPhysicalDeviceMaintenance4Features" | "VkPhysicalDeviceMaintenance4FeaturesKHR" => {
            Ext(&["VK_KHR_maintenance4"], Some(3))
        }
        "VkPhysicalDeviceShaderSubgroupRotateFeatures"
        | "VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR" => {
            Ext(&["VK_KHR_shader_subgroup_rotate"], Some(4))
        }
        "VkPhysicalDeviceDynamicRenderingLocalReadFeatures"
        | "VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR" => {
            Ext(&["VK_KHR_dynamic_rendering_local_read"], Some(4))
        }
        "VkPhysicalDeviceMaintenance5Features" | "VkPhysicalDeviceMaintenance5FeaturesKHR" => {
            Ext(&["VK_KHR_maintenance5"], Some(4))
        }
        "VkPhysicalDeviceVertexAttributeDivisorFeatures"
        | "VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR"
        | "VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT" => Ext(
            &[
                "VK_KHR_vertex_attribute_divisor",
                "VK_EXT_vertex_attribute_divisor",
            ],
            Some(4),
        ),
        "VkPhysicalDeviceRobustness2FeaturesEXT" | "VkPhysicalDeviceRobustness2FeaturesKHR" => {
            Ext(&["VK_EXT_robustness2", "VK_KHR_robustness2"], None)
        }
        "VkPhysicalDeviceMeshShaderFeaturesEXT" => Ext(&["VK_EXT_mesh_shader"], None),
        "VkPhysicalDeviceShaderAtomicFloatFeaturesEXT" => {
            Ext(&["VK_EXT_shader_atomic_float"], None)
        }
        "VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT" => {
            Ext(&["VK_EXT_shader_atomic_float2"], None)
        }
        "VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT" => {
            Ext(&["VK_EXT_shader_image_atomic_int64"], None)
        }
        "VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR" => {
            Ext(&["VK_KHR_workgroup_memory_explicit_layout"], None)
        }
        "VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR" => {
            Ext(&["VK_KHR_shader_maximal_reconvergence"], None)
        }
        "VkPhysicalDeviceShaderQuadControlFeaturesKHR" => {
            Ext(&["VK_KHR_shader_quad_control"], None)
        }
        "VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR" => {
            Ext(&["VK_KHR_shader_subgroup_uniform_control_flow"], None)
        }
        "VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR"
        | "VkPhysicalDeviceComputeShaderDerivativesFeaturesNV" => Ext(
            &[
                "VK_KHR_compute_shader_derivatives",
                "VK_NV_compute_shader_derivatives",
            ],
            None,
        ),
        "VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR" => {
            Ext(&["VK_KHR_fragment_shader_barycentric"], None)
        }
        "VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT" => {
            Ext(&["VK_EXT_fragment_shader_interlock"], None)
        }
        "VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT"
        | "VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM" => Ext(
            &[
                "VK_EXT_rasterization_order_attachment_access",
                "VK_ARM_rasterization_order_attachment_access",
            ],
            None,
        ),
        "VkPhysicalDeviceExtendedDynamicStateFeaturesEXT" => {
            Ext(&["VK_EXT_extended_dynamic_state"], None)
        }
        "VkPhysicalDeviceExtendedDynamicState2FeaturesEXT" => {
            Ext(&["VK_EXT_extended_dynamic_state2"], None)
        }
        // Not promoted to a core version.
        "VkPhysicalDeviceExtendedDynamicState3FeaturesEXT" => {
            Ext(&["VK_EXT_extended_dynamic_state3"], None)
        }
        "VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT" => {
            Ext(&["VK_EXT_attachment_feedback_loop_layout"], None)
        }
        "VkPhysicalDeviceMaintenance8FeaturesKHR" => Ext(&["VK_KHR_maintenance8"], None),
        "VkPhysicalDeviceShaderFmaFeaturesKHR" => Ext(&["VK_KHR_shader_fma"], None),
        "VkPhysicalDeviceBorderColorSwizzleFeaturesEXT" => {
            Ext(&["VK_EXT_border_color_swizzle"], None)
        }
        "VkPhysicalDeviceDepthBiasControlFeaturesEXT" => Ext(&["VK_EXT_depth_bias_control"], None),
        "VkPhysicalDeviceDepthClipEnableFeaturesEXT" => Ext(&["VK_EXT_depth_clip_enable"], None),
        "VkPhysicalDeviceDepthClampControlFeaturesEXT" => {
            Ext(&["VK_EXT_depth_clamp_control"], None)
        }
        "VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT" => {
            Ext(&["VK_EXT_primitive_topology_list_restart"], None)
        }
        "VkPhysicalDeviceProvokingVertexFeaturesEXT" => Ext(&["VK_EXT_provoking_vertex"], None),
        "VkPhysicalDeviceLineRasterizationFeaturesEXT"
        | "VkPhysicalDeviceLineRasterizationFeaturesKHR" => Ext(
            &["VK_EXT_line_rasterization", "VK_KHR_line_rasterization"],
            None,
        ),
        "VkPhysicalDeviceRayQueryFeaturesKHR" => Ext(&["VK_KHR_ray_query"], None),
        "VkPhysicalDeviceAccelerationStructureFeaturesKHR" => {
            Ext(&["VK_KHR_acceleration_structure"], None)
        }
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR" => Ext(&["VK_KHR_portability_subset"], None),
        _ => return None,
    })
}

/// Every feature structure member the reader takes, with the feature it enables: one row per
/// `VkFeature` (`every_feature_has_a_member_name`), so a feature the device query reads is never
/// missing from a profile that lists it. Member names are unique across the structures
/// [`feature_route`] accepts.
pub(super) const FEATURE_MEMBERS: &[(&str, VkFeature)] = {
    use VkFeature as F;
    &[
        ("robustBufferAccess", F::RobustBufferAccess),
        ("robustBufferAccess2", F::RobustBufferAccess2),
        ("robustImageAccess", F::RobustImageAccess),
        ("robustImageAccess2", F::RobustImageAccess2),
        ("nullDescriptor", F::NullDescriptor),
        ("shaderInt8", F::ShaderInt8),
        ("shaderInt16", F::ShaderInt16),
        ("shaderInt64", F::ShaderInt64),
        ("shaderFloat16", F::ShaderFloat16),
        ("shaderFloat64", F::ShaderFloat64),
        ("storageBuffer8BitAccess", F::StorageBuffer8BitAccess),
        (
            "uniformAndStorageBuffer8BitAccess",
            F::UniformAndStorageBuffer8BitAccess,
        ),
        ("storagePushConstant8", F::StoragePushConstant8),
        ("storageBuffer16BitAccess", F::StorageBuffer16BitAccess),
        (
            "uniformAndStorageBuffer16BitAccess",
            F::UniformAndStorageBuffer16BitAccess,
        ),
        ("storagePushConstant16", F::StoragePushConstant16),
        ("storageInputOutput16", F::StorageInputOutput16),
        ("shaderBufferInt64Atomics", F::ShaderBufferInt64Atomics),
        ("shaderSharedInt64Atomics", F::ShaderSharedInt64Atomics),
        ("shaderImageInt64Atomics", F::ShaderImageInt64Atomics),
        ("shaderBufferFloat32Atomics", F::ShaderBufferFloat32Atomics),
        (
            "shaderBufferFloat32AtomicAdd",
            F::ShaderBufferFloat32AtomicAdd,
        ),
        (
            "shaderBufferFloat32AtomicMinMax",
            F::ShaderBufferFloat32AtomicMinMax,
        ),
        ("shaderSharedFloat32Atomics", F::ShaderSharedFloat32Atomics),
        (
            "shaderSharedFloat32AtomicAdd",
            F::ShaderSharedFloat32AtomicAdd,
        ),
        (
            "shaderSharedFloat32AtomicMinMax",
            F::ShaderSharedFloat32AtomicMinMax,
        ),
        ("shaderImageFloat32Atomics", F::ShaderImageFloat32Atomics),
        (
            "shaderImageFloat32AtomicAdd",
            F::ShaderImageFloat32AtomicAdd,
        ),
        (
            "shaderImageFloat32AtomicMinMax",
            F::ShaderImageFloat32AtomicMinMax,
        ),
        ("bufferDeviceAddress", F::BufferDeviceAddress),
        ("scalarBlockLayout", F::ScalarBlockLayout),
        (
            "uniformBufferStandardLayout",
            F::UniformBufferStandardLayout,
        ),
        ("vulkanMemoryModel", F::VulkanMemoryModel),
        (
            "vulkanMemoryModelDeviceScope",
            F::VulkanMemoryModelDeviceScope,
        ),
        (
            "workgroupMemoryExplicitLayout",
            F::WorkgroupMemoryExplicitLayout,
        ),
        (
            "workgroupMemoryExplicitLayout8BitAccess",
            F::WorkgroupMemoryExplicitLayout8BitAccess,
        ),
        (
            "workgroupMemoryExplicitLayout16BitAccess",
            F::WorkgroupMemoryExplicitLayout16BitAccess,
        ),
        (
            "shaderZeroInitializeWorkgroupMemory",
            F::ShaderZeroInitializeWorkgroupMemory,
        ),
        (
            "vertexPipelineStoresAndAtomics",
            F::VertexPipelineStoresAndAtomics,
        ),
        ("fragmentStoresAndAtomics", F::FragmentStoresAndAtomics),
        ("shaderDrawParameters", F::ShaderDrawParameters),
        ("geometryShader", F::GeometryShader),
        (
            "shaderTessellationAndGeometryPointSize",
            F::ShaderTessellationAndGeometryPointSize,
        ),
        ("tessellationShader", F::TessellationShader),
        ("sampleRateShading", F::SampleRateShading),
        ("dualSrcBlend", F::DualSrcBlend),
        ("logicOp", F::LogicOp),
        ("independentBlend", F::IndependentBlend),
        ("shaderClipDistance", F::ShaderClipDistance),
        ("shaderCullDistance", F::ShaderCullDistance),
        ("shaderOutputLayer", F::ShaderOutputLayer),
        ("shaderOutputViewportIndex", F::ShaderOutputViewportIndex),
        ("multiview", F::Multiview),
        (
            "multiviewTessellationShader",
            F::MultiviewTessellationShader,
        ),
        ("multiviewGeometryShader", F::MultiviewGeometryShader),
        ("fragmentShaderBarycentric", F::FragmentShaderBarycentric),
        (
            "fragmentShaderPixelInterlock",
            F::FragmentShaderPixelInterlock,
        ),
        (
            "fragmentShaderSampleInterlock",
            F::FragmentShaderSampleInterlock,
        ),
        (
            "shaderDemoteToHelperInvocation",
            F::ShaderDemoteToHelperInvocation,
        ),
        ("shaderTerminateInvocation", F::ShaderTerminateInvocation),
        ("dynamicRenderingLocalRead", F::DynamicRenderingLocalRead),
        (
            "rasterizationOrderColorAttachmentAccess",
            F::RasterizationOrderColorAttachmentAccess,
        ),
        (
            "rasterizationOrderDepthAttachmentAccess",
            F::RasterizationOrderDepthAttachmentAccess,
        ),
        (
            "rasterizationOrderStencilAttachmentAccess",
            F::RasterizationOrderStencilAttachmentAccess,
        ),
        (
            "vertexAttributeInstanceRateDivisor",
            F::VertexAttributeInstanceRateDivisor,
        ),
        (
            "vertexAttributeInstanceRateZeroDivisor",
            F::VertexAttributeInstanceRateZeroDivisor,
        ),
        ("drawIndirectFirstInstance", F::DrawIndirectFirstInstance),
        ("multiDrawIndirect", F::MultiDrawIndirect),
        ("depthClamp", F::DepthClamp),
        ("depthBiasClamp", F::DepthBiasClamp),
        ("fillModeNonSolid", F::FillModeNonSolid),
        ("imageCubeArray", F::ImageCubeArray),
        ("shaderImageGatherExtended", F::ShaderImageGatherExtended),
        ("shaderResourceMinLod", F::ShaderResourceMinLod),
        ("shaderResourceResidency", F::ShaderResourceResidency),
        ("samplerAnisotropy", F::SamplerAnisotropy),
        ("samplerFilterMinmax", F::SamplerFilterMinmax),
        ("drawIndirectCount", F::DrawIndirectCount),
        ("samplerMirrorClampToEdge", F::SamplerMirrorClampToEdge),
        (
            "shaderStorageImageMultisample",
            F::ShaderStorageImageMultisample,
        ),
        (
            "shaderStorageImageExtendedFormats",
            F::ShaderStorageImageExtendedFormats,
        ),
        (
            "shaderStorageImageReadWithoutFormat",
            F::ShaderStorageImageReadWithoutFormat,
        ),
        (
            "shaderStorageImageWriteWithoutFormat",
            F::ShaderStorageImageWriteWithoutFormat,
        ),
        ("runtimeDescriptorArray", F::RuntimeDescriptorArray),
        (
            "descriptorBindingPartiallyBound",
            F::DescriptorBindingPartiallyBound,
        ),
        (
            "descriptorBindingVariableDescriptorCount",
            F::DescriptorBindingVariableDescriptorCount,
        ),
        (
            "descriptorBindingSampledImageUpdateAfterBind",
            F::DescriptorBindingSampledImageUpdateAfterBind,
        ),
        (
            "descriptorBindingStorageImageUpdateAfterBind",
            F::DescriptorBindingStorageImageUpdateAfterBind,
        ),
        (
            "descriptorBindingStorageBufferUpdateAfterBind",
            F::DescriptorBindingStorageBufferUpdateAfterBind,
        ),
        (
            "descriptorBindingUniformTexelBufferUpdateAfterBind",
            F::DescriptorBindingUniformTexelBufferUpdateAfterBind,
        ),
        (
            "descriptorBindingStorageTexelBufferUpdateAfterBind",
            F::DescriptorBindingStorageTexelBufferUpdateAfterBind,
        ),
        (
            "descriptorBindingUpdateUnusedWhilePending",
            F::DescriptorBindingUpdateUnusedWhilePending,
        ),
        (
            "shaderSampledImageArrayNonUniformIndexing",
            F::ShaderSampledImageArrayNonUniformIndexing,
        ),
        (
            "shaderStorageImageArrayNonUniformIndexing",
            F::ShaderStorageImageArrayNonUniformIndexing,
        ),
        (
            "shaderStorageBufferArrayNonUniformIndexing",
            F::ShaderStorageBufferArrayNonUniformIndexing,
        ),
        (
            "shaderUniformTexelBufferArrayNonUniformIndexing",
            F::ShaderUniformTexelBufferArrayNonUniformIndexing,
        ),
        (
            "shaderStorageTexelBufferArrayNonUniformIndexing",
            F::ShaderStorageTexelBufferArrayNonUniformIndexing,
        ),
        ("subgroupSizeControl", F::SubgroupSizeControl),
        ("computeFullSubgroups", F::ComputeFullSubgroups),
        (
            "shaderSubgroupExtendedTypes",
            F::ShaderSubgroupExtendedTypes,
        ),
        ("shaderMaximalReconvergence", F::ShaderMaximalReconvergence),
        ("shaderQuadControl", F::ShaderQuadControl),
        (
            "shaderSubgroupUniformControlFlow",
            F::ShaderSubgroupUniformControlFlow,
        ),
        ("shaderSubgroupRotate", F::ShaderSubgroupRotate),
        (
            "shaderSubgroupRotateClustered",
            F::ShaderSubgroupRotateClustered,
        ),
        (
            "computeDerivativeGroupQuads",
            F::ComputeDerivativeGroupQuads,
        ),
        (
            "computeDerivativeGroupLinear",
            F::ComputeDerivativeGroupLinear,
        ),
        ("maintenance4", F::Maintenance4),
        ("maintenance8", F::Maintenance8),
        ("maintenance5", F::Maintenance5),
        ("occlusionQueryPrecise", F::OcclusionQueryPrecise),
        ("depthBiasControl", F::DepthBiasControl),
        ("floatRepresentation", F::FloatRepresentation),
        ("depthBiasExact", F::DepthBiasExact),
        ("shaderFmaFloat16", F::ShaderFmaFloat16),
        ("shaderFmaFloat32", F::ShaderFmaFloat32),
        ("synchronization2", F::Synchronization2),
        ("extendedDynamicState", F::ExtendedDynamicState),
        ("extendedDynamicState2", F::ExtendedDynamicState2),
        (
            "extendedDynamicState3ColorWriteMask",
            F::ExtendedDynamicState3ColorWriteMask,
        ),
        (
            "attachmentFeedbackLoopLayout",
            F::AttachmentFeedbackLoopLayout,
        ),
        (
            "shaderUniformBufferArrayDynamicIndexing",
            F::ShaderUniformBufferArrayDynamicIndexing,
        ),
        (
            "shaderSampledImageArrayDynamicIndexing",
            F::ShaderSampledImageArrayDynamicIndexing,
        ),
        (
            "shaderStorageBufferArrayDynamicIndexing",
            F::ShaderStorageBufferArrayDynamicIndexing,
        ),
        (
            "shaderStorageImageArrayDynamicIndexing",
            F::ShaderStorageImageArrayDynamicIndexing,
        ),
        (
            "shaderUniformTexelBufferArrayDynamicIndexing",
            F::ShaderUniformTexelBufferArrayDynamicIndexing,
        ),
        (
            "shaderStorageTexelBufferArrayDynamicIndexing",
            F::ShaderStorageTexelBufferArrayDynamicIndexing,
        ),
        (
            "shaderInputAttachmentArrayDynamicIndexing",
            F::ShaderInputAttachmentArrayDynamicIndexing,
        ),
        ("shaderBufferFloat16Atomics", F::ShaderBufferFloat16Atomics),
        (
            "shaderBufferFloat16AtomicAdd",
            F::ShaderBufferFloat16AtomicAdd,
        ),
        (
            "shaderBufferFloat16AtomicMinMax",
            F::ShaderBufferFloat16AtomicMinMax,
        ),
        ("shaderSharedFloat16Atomics", F::ShaderSharedFloat16Atomics),
        (
            "shaderSharedFloat16AtomicAdd",
            F::ShaderSharedFloat16AtomicAdd,
        ),
        (
            "shaderSharedFloat16AtomicMinMax",
            F::ShaderSharedFloat16AtomicMinMax,
        ),
        ("shaderBufferFloat64Atomics", F::ShaderBufferFloat64Atomics),
        (
            "shaderBufferFloat64AtomicAdd",
            F::ShaderBufferFloat64AtomicAdd,
        ),
        (
            "shaderBufferFloat64AtomicMinMax",
            F::ShaderBufferFloat64AtomicMinMax,
        ),
        ("shaderSharedFloat64Atomics", F::ShaderSharedFloat64Atomics),
        (
            "shaderSharedFloat64AtomicAdd",
            F::ShaderSharedFloat64AtomicAdd,
        ),
        (
            "shaderSharedFloat64AtomicMinMax",
            F::ShaderSharedFloat64AtomicMinMax,
        ),
        (
            "workgroupMemoryExplicitLayoutScalarBlockLayout",
            F::WorkgroupMemoryExplicitLayoutScalarBlockLayout,
        ),
        ("multiViewport", F::MultiViewport),
        ("borderColorSwizzle", F::BorderColorSwizzle),
        (
            "borderColorSwizzleFromImage",
            F::BorderColorSwizzleFromImage,
        ),
        ("rayQuery", F::RayQuery),
        ("taskShader", F::TaskShader),
        ("meshShader", F::MeshShader),
        ("accelerationStructure", F::AccelerationStructure),
        (
            "constantAlphaColorBlendFactors",
            F::ConstantAlphaColorBlendFactors,
        ),
        (
            "vertexAttributeAccessBeyondStride",
            F::VertexAttributeAccessBeyondStride,
        ),
        ("samplerMipLodBias", F::SamplerMipLodBias),
        ("mutableComparisonSamplers", F::MutableComparisonSamplers),
        (
            "shaderSampleRateInterpolationFunctions",
            F::ShaderSampleRateInterpolationFunctions,
        ),
        ("multisampleArrayImage", F::MultisampleArrayImage),
        ("imageView2DOn3DImage", F::ImageView2DOn3DImage),
        ("separateStencilMaskRef", F::SeparateStencilMaskRef),
        ("imageViewFormatSwizzle", F::ImageViewFormatSwizzle),
        ("fullDrawIndexUint32", F::FullDrawIndexUint32),
        ("depthClipEnable", F::DepthClipEnable),
        ("depthClampControl", F::DepthClampControl),
        (
            "primitiveTopologyListRestart",
            F::PrimitiveTopologyListRestart,
        ),
        (
            "primitiveTopologyPatchListRestart",
            F::PrimitiveTopologyPatchListRestart,
        ),
        ("provokingVertexLast", F::ProvokingVertexLast),
        ("bresenhamLines", F::BresenhamLines),
        ("alphaToOne", F::AlphaToOne),
    ]
};

/// The feature a structure member enables ([`FEATURE_MEMBERS`]).
fn feature_from_member(member: &str) -> Option<VkFeature> {
    FEATURE_MEMBERS
        .iter()
        .find(|(name, _)| *name == member)
        .map(|&(_, feature)| feature)
}

/// The typed extension for a registry extension name.
pub(crate) fn extension_from_name(name: &str) -> Option<VkExtension> {
    use VkExtension as E;
    Some(match name {
        "VK_EXT_robustness2" | "VK_KHR_robustness2" => E::ExtRobustness2,
        "VK_EXT_image_robustness" => E::ExtImageRobustness,
        "VK_EXT_shader_atomic_float" => E::ExtShaderAtomicFloat,
        "VK_EXT_shader_atomic_float2" => E::ExtShaderAtomicFloat2,
        "VK_EXT_shader_image_atomic_int64" => E::ExtShaderImageAtomicInt64,
        "VK_EXT_shader_stencil_export" => E::ExtShaderStencilExport,
        "VK_EXT_shader_demote_to_helper_invocation" => E::ExtShaderDemoteToHelperInvocation,
        "VK_EXT_shader_viewport_index_layer" => E::ExtShaderViewportIndexLayer,
        "VK_EXT_fragment_shader_interlock" => E::ExtFragmentShaderInterlock,
        "VK_EXT_subgroup_size_control" => E::ExtSubgroupSizeControl,
        "VK_EXT_rasterization_order_attachment_access"
        | "VK_ARM_rasterization_order_attachment_access" => {
            E::ExtRasterizationOrderAttachmentAccess
        }
        "VK_EXT_vertex_attribute_divisor" => E::ExtVertexAttributeDivisor,
        "VK_EXT_external_memory_host" => E::ExtExternalMemoryHost,
        "VK_EXT_extended_dynamic_state" => E::ExtExtendedDynamicState,
        "VK_EXT_extended_dynamic_state2" => E::ExtExtendedDynamicState2,
        "VK_EXT_extended_dynamic_state3" => E::ExtExtendedDynamicState3,
        "VK_KHR_format_feature_flags2" => E::KhrFormatFeatureFlags2,
        "VK_EXT_post_depth_coverage" => E::ExtPostDepthCoverage,
        "VK_EXT_attachment_feedback_loop_layout" => E::ExtAttachmentFeedbackLoopLayout,
        "VK_EXT_mutable_descriptor_type" | "VK_VALVE_mutable_descriptor_type" => {
            E::ExtMutableDescriptorType
        }
        "VK_EXT_descriptor_buffer" => E::ExtDescriptorBuffer,
        "VK_EXT_mesh_shader" => E::ExtMeshShader,
        "VK_KHR_shader_terminate_invocation" => E::KhrShaderTerminateInvocation,
        "VK_KHR_shader_maximal_reconvergence" => E::KhrShaderMaximalReconvergence,
        "VK_KHR_shader_quad_control" => E::KhrShaderQuadControl,
        "VK_KHR_shader_subgroup_uniform_control_flow" => E::KhrShaderSubgroupUniformControlFlow,
        "VK_KHR_shader_subgroup_rotate" => E::KhrShaderSubgroupRotate,
        "VK_KHR_fragment_shader_barycentric" => E::KhrFragmentShaderBarycentric,
        "VK_KHR_compute_shader_derivatives" | "VK_NV_compute_shader_derivatives" => {
            E::KhrComputeShaderDerivatives
        }
        "VK_KHR_workgroup_memory_explicit_layout" => E::KhrWorkgroupMemoryExplicitLayout,
        "VK_KHR_zero_initialize_workgroup_memory" => E::KhrZeroInitializeWorkgroupMemory,
        "VK_KHR_dynamic_rendering_local_read" => E::KhrDynamicRenderingLocalRead,
        "VK_KHR_vertex_attribute_divisor" => E::KhrVertexAttributeDivisor,
        "VK_KHR_push_descriptor" => E::KhrPushDescriptor,
        "VK_KHR_maintenance4" => E::KhrMaintenance4,
        "VK_KHR_maintenance8" => E::KhrMaintenance8,
        "VK_KHR_shader_fma" => E::KhrShaderFma,
        "VK_EXT_depth_range_unrestricted" => E::ExtDepthRangeUnrestricted,
        "VK_EXT_depth_bias_control" => E::ExtDepthBiasControl,
        "VK_EXT_depth_clip_enable" => E::ExtDepthClipEnable,
        "VK_EXT_depth_clamp_control" => E::ExtDepthClampControl,
        "VK_EXT_sample_locations" => E::ExtSampleLocations,
        "VK_EXT_primitive_topology_list_restart" => E::ExtPrimitiveTopologyListRestart,
        "VK_EXT_provoking_vertex" => E::ExtProvokingVertex,
        "VK_EXT_line_rasterization" | "VK_KHR_line_rasterization" => E::ExtLineRasterization,
        "VK_KHR_synchronization2" => E::KhrSynchronization2,
        "VK_KHR_shader_untyped_pointers" => E::KhrShaderUntypedPointers,
        "VK_KHR_cooperative_matrix" => E::KhrCooperativeMatrix,
        "VK_KHR_shader_bfloat16" => E::KhrShaderBfloat16,
        "VK_KHR_ray_query" => E::KhrRayQuery,
        "VK_QCOM_tile_shading" => E::QcomTileShading,
        "VK_KHR_maintenance5" => E::KhrMaintenance5,
        "VK_EXT_texture_compression_astc_hdr" => E::ExtTextureCompressionAstcHdr,
        "VK_IMG_format_pvrtc" => E::ImgFormatPvrtc,
        "VK_EXT_border_color_swizzle" => E::ExtBorderColorSwizzle,
        "VK_EXT_custom_border_color" => E::ExtCustomBorderColor,
        "VK_KHR_portability_subset" => E::KhrPortabilitySubset,
        "VK_KHR_dynamic_rendering" => E::KhrDynamicRendering,
        "VK_KHR_acceleration_structure" => E::KhrAccelerationStructure,
        "VK_KHR_deferred_host_operations" => E::KhrDeferredHostOperations,
        _ => return None,
    })
}
