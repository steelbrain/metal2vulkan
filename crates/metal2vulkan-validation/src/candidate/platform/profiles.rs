//! `VK_LAYER_KHRONOS_profiles` under the engine executor.
//!
//! A command restricts every Vulkan context it opens to one committed device profile
//! ([`restrict_devices_to`], from its `--profiles-layer P` argument). A restricted context stacks
//! the profiles layer below the validation layer, configured through `VK_EXT_layer_settings` with
//! the committed file, so the device reports P and the validation layer checks against it; the ICD
//! still executes as itself. Before its device is created the context checks, each failure a
//! [`RestrictionError`]:
//!
//! - the device as it is (read through the layer with its emulation off) has every capability P
//!   states ([`exceeds`]): the layer reports P's value whether or not the device has it, so it
//!   cannot enforce this itself;
//! - the layer is active (`vkGetPhysicalDeviceToolPropertiesEXT`); of the members it keeps from
//!   the device, the identity is the device's and each of [`KEPT`], and each format's color
//!   attachment sample counts ([`kept_sample_counts`]), lies between P's value and the device's;
//!   and every other member of the description the context queries through it equals P as a
//!   context at its instance version reads it ([`read_at`], [`differences`]).
//!
//! The context then translates for P as read at its instance version: its queried description
//! but for the kept members, where P is within the device.
//!
//! Every context, restricted or not, reads the active tools and records them with its rows. A
//! context the profiles layer restricts without a request (an environment that stacks it) opens
//! and runs, but the store refuses its rows (`CandidateObservation::validate_content`), so a row
//! run under a profile is never recorded as a native row. A restricted row records the layer and
//! P in its environment, its census names P and the device that executed, and it is stored in P's
//! own slot.

use ash::vk;
use metal2vulkan::device::ApiVersion;
use metal2vulkan::device::{DeviceDescription, Independence};
use metal2vulkan_runtime::{LayerSetting, LayerSettingValue};
use serde_json::Value;
use std::collections::BTreeSet;
use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;

pub use crate::acceptance::read_at;
pub(super) use crate::candidate::device_facts::{active_tools, ActiveTool};

/// The profiles layer, as the candidate rows' validation names it.
pub(super) use crate::observation::PROFILES_LAYER;

/// What the restricted context simulates: what the layer can simulate that the description
/// states, so that the context checks it against the device ([`check_within`]) and against what
/// the layer reports ([`check_restricted`]). Not video, and not queue families: the description
/// states none, so P's could be held to neither, and the device's own pass through. The API
/// version is among it: below the layer the instance runs at P's version, without which the layer
/// would not read P's `VkPhysicalDeviceVulkan13*` and later structures, which its extension
/// structures take their values from. In exchange the layer lists every extension promoted to
/// core at or below that version whether or not P lists it, so the context removes those again
/// ([`ProfilesLayer::Restrict`]).
const SIMULATED: [&CStr; 5] = [
    c"SIMULATE_API_VERSION_BIT",
    c"SIMULATE_FEATURES_BIT",
    c"SIMULATE_PROPERTIES_BIT",
    c"SIMULATE_EXTENSIONS_BIT",
    c"SIMULATE_FORMATS_BIT",
];

/// The capability members the layer keeps from the device whatever a profile states: `vk.xml`
/// gives them `limittype="exact"`, which the pinned layer never sets. Through the layer each
/// reads as the device's value, or as the reader's default where P does not expose its
/// structure; P is within the device on them ([`exceeds`]), and the queried value must lie
/// between the two ([`check_restricted`]). The driver's measured deviations are no member: they
/// follow the identity (`driverID`, `vendorID`, `driverVersion`), which the layer keeps from the
/// device, so the context translates with the executing driver's, as it takes that driver's
/// workarounds.
pub(super) const KEPT: [&str; 3] = [
    "float_controls.denorm_independence",
    "float_controls.rounding_independence",
    "rgba4_opaque_black_swizzled",
];

/// A committed device profile a context is restricted to.
#[derive(Clone, Debug)]
pub struct ProfileRestriction {
    /// The file stem under `validation/profiles`, which the census and the store slot name.
    pub stem: String,
    /// The one profile the file holds (`VP_M2V_<stem>`).
    pub profile_name: String,
    /// The file the layer reads.
    pub file: PathBuf,
    /// SHA-256 of the file's bytes, recorded with every row run under it.
    pub sha256: String,
    /// The committed description, as the Khronos JSON reader builds it.
    pub description: DeviceDescription,
    /// The extension names the file's capability blocks list (every block belongs to its one
    /// profile).
    pub extension_names: BTreeSet<String>,
    /// Whether a format entry of the file states `VkFormatProperties3`: without one the profile
    /// says nothing about the `VkFormatFeatureFlags2`-only bits ([`ProfileRestriction::hidden`]).
    pub states_format_flags2: bool,
    /// Whether the file is a requirements document extracted from Vulkan-Profiles
    /// (`provenance.source` `khronos`), which states only what its profile requires.
    pub requirements_only: bool,
    /// Whether its profile lists the block its regeneration completes a requirements document with
    /// ([`LAYER_PRESENTATION_BLOCK`]); a requirements document without it the layer cannot present
    /// as the product reads it ([`ProfileRestriction::presentation_gap`]).
    pub completed_for_layer: bool,
}

/// The provenance kind of a requirements document extracted verbatim from Vulkan-Profiles
/// (`scripts/regen-device-profiles`): the other kinds state every member the product reads (a
/// device report or capture states its device's every member; a derived profile states its base
/// version's every member and, in `m2v_unrequired_properties`, the ones the base leaves open).
const REQUIREMENTS_SOURCE: &str = "khronos";

/// The capability block `scripts/regen-device-profiles` completes a requirements document with
/// (`LAYER_PRESENTATION_BLOCK` there): each feature its core version (up to 1.2) requires that it
/// omits, and each member the product reads of the device-property structures of the extensions it
/// lists that it omits, at the value the product reads.
pub const LAYER_PRESENTATION_BLOCK: &str = "m2v_layer_presentation";

/// The extension whose `VkFormatProperties3` carries the format feature bits `VkFormatProperties`
/// cannot (the storage reads and writes without a format, depth-comparison sampling).
const FORMAT_FEATURE_FLAGS2: &str = "VK_KHR_format_feature_flags2";

impl ProfileRestriction {
    /// The committed profile `stem`; an unknown stem names the committed ones.
    pub fn committed(stem: &str) -> Result<ProfileRestriction, String> {
        let committed = crate::engine::committed_profiles()?;
        if !committed.iter().any(|known| known == stem) {
            return Err(format!(
                "unknown profile {stem:?}; committed profiles: {}",
                committed.join(", ")
            ));
        }
        let file = crate::engine::profiles_dir().join(format!("{stem}.json"));
        let bytes = std::fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        Self::of_document(stem, file, &bytes)
    }

    /// The profile document `bytes`, read from `file`, as restriction `stem`.
    fn of_document(stem: &str, file: PathBuf, bytes: &[u8]) -> Result<ProfileRestriction, String> {
        let text =
            std::str::from_utf8(bytes).map_err(|error| format!("{}: {error}", file.display()))?;
        let document: Value =
            serde_json::from_str(text).map_err(|error| format!("{}: {error}", file.display()))?;
        let names = document
            .get("profiles")
            .and_then(Value::as_object)
            .map(|profiles| profiles.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let [profile_name] = <[String; 1]>::try_from(names).map_err(|names| {
            format!(
                "{} holds {} profiles; a restriction needs exactly one",
                file.display(),
                names.len()
            )
        })?;
        let description = DeviceDescription::from_khronos_json(text, Some(&profile_name))
            .map_err(|error| format!("{stem}: {error}"))?;
        let blocks = || {
            document
                .get("capabilities")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|blocks| blocks.values())
        };
        let extension_names = blocks()
            .filter_map(|block| block.get("extensions").and_then(Value::as_object))
            .flat_map(|extensions| extensions.keys().cloned())
            .collect();
        let states_format_flags2 = blocks()
            .filter_map(|block| block.get("formats").and_then(Value::as_object))
            .flat_map(|formats| formats.values())
            .any(|format| {
                format.get("VkFormatProperties3").is_some()
                    || format.get("VkFormatProperties3KHR").is_some()
            });
        let requirements_only = document
            .get("provenance")
            .and_then(|provenance| provenance.get("source"))
            .and_then(Value::as_str)
            == Some(REQUIREMENTS_SOURCE);
        let completed_for_layer = document
            .get("profiles")
            .and_then(|profiles| profiles.get(&profile_name))
            .and_then(|profile| profile.get("capabilities"))
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|block| block.as_str() == Some(LAYER_PRESENTATION_BLOCK))
            });
        Ok(ProfileRestriction {
            stem: stem.into(),
            profile_name,
            file,
            sha256: crate::hash::sha256_bytes(bytes),
            description,
            extension_names,
            states_format_flags2,
            requirements_only,
            completed_for_layer,
        })
    }

    /// The API version a context restricted to this profile runs at: the profile's own, with Vulkan
    /// 1.2 its floor. Its instance is created at it, the device's description and the profile are
    /// read at it ([`check_within`], [`check_restricted`]), and the context's feature chain links
    /// what is core at it (`enables::Available::at`), so the context translates for the profile the
    /// product reads, which reads a profile at its own version: read at an earlier one, a Vulkan
    /// 1.3 document drops each feature and member only that version's core states (`roadmap-2022`
    /// moves its translation key and ten golden cells). A profile at Vulkan 1.2 runs at 1.2, as
    /// every restricted context did before.
    pub fn context_version(&self) -> ApiVersion {
        self.description.api_version.max(ApiVersion::V1_2)
    }

    /// Why the layer cannot present this profile as the product reads it, if it cannot: a tooling
    /// gap, not a device's. A restriction to it is then refused before any device is read
    /// ([`restrict_devices_to`]). The restricted context requires the description it queries through the layer to be the profile
    /// as the product reads it ([`check_restricted`]), so every member the product reads must come
    /// from the file, on every device. A requirements document
    /// ([`ProfileRestriction::requirements_only`]) states only what its profile requires: the
    /// product reads a member it omits as its version's guarantee, or as none, and a feature its
    /// version requires as present, while the layer reports the device's own value for an omitted
    /// member (of a structure the context queries) and an omitted feature as absent. Under
    /// `lunarg-desktop-baseline-2023` as extracted, the RTX 5080's context
    /// found `shaderSubgroupExtendedTypes` absent and the five `VK_EXT_subgroup_size_control` and
    /// `VK_EXT_vertex_attribute_divisor` properties the document lists the extensions of but does
    /// not state at the device's values. Its regeneration now states every such member
    /// ([`ProfileRestriction::completed_for_layer`]), at the profile's own version, which the
    /// context runs at ([`ProfileRestriction::context_version`]); one that does not is refused.
    pub fn presentation_gap(&self) -> Option<String> {
        if !self.requirements_only || self.completed_for_layer {
            return None;
        }
        let version = self.context_version();
        Some(format!(
            "tooling gap: the executor's Vulkan {}.{} context cannot present requirements \
             document {} as the product reads it (provenance source {REQUIREMENTS_SOURCE}, not \
             completed by {LAYER_PRESENTATION_BLOCK}: it states only what the profile requires, \
             and the profiles layer reports the device's own value for each member it omits and \
             each feature it omits as absent, where the product reads the version's guarantee or \
             none, and a feature the version requires as present)",
            version.major, version.minor, self.stem
        ))
    }

    /// Of the extension names a device `presents` through the layer, the ones the restricted
    /// context hides:
    ///
    /// - every one P does not list: what the layer adds for P's version;
    /// - `VK_KHR_format_feature_flags2` where P states no `VkFormatProperties3`. The layer then
    ///   derives each format's `VkFormatProperties3` from its `VkFormatProperties` alone, so the
    ///   bits only the former carries read clear whatever the device has, and the validation
    ///   layer would hold a storage image without a format, or depth-comparison sampling, to
    ///   them. Hidden, it applies the rules `VkFormatProperties` states, and so does the product,
    ///   which reads the bits where the extension or Vulkan 1.3 reports them: the context
    ///   translates for P without the extension ([`ProfileRestriction::read_at`]).
    pub(super) fn hidden<'a>(
        &self,
        presents: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        presents
            .into_iter()
            .filter(|name| {
                !self.extension_names.contains(*name)
                    || (*name == FORMAT_FEATURE_FLAGS2 && !self.states_format_flags2)
            })
            .map(str::to_owned)
            .collect()
    }

    /// P as a context opened at `instance` (the instance's API version) presents it: P
    /// [`read_at`] `instance`, without `VK_KHR_format_feature_flags2` where the context hides it
    /// ([`ProfileRestriction::hidden`]).
    pub fn read_at(&self, instance: ApiVersion) -> DeviceDescription {
        let mut read = read_at(&self.description, instance);
        read.khr_format_feature_flags2 &= self.states_format_flags2;
        read
    }
}

/// The restriction of this process's contexts: decided once, by [`restrict_devices_to`] or by the
/// first context opened without one.
static RESTRICTION: OnceLock<Option<ProfileRestriction>> = OnceLock::new();

/// Restrict every Vulkan context this process opens to committed profile `stem` (a command's
/// `--profiles-layer P`). It must be called before any context opens: the choice is made once per
/// process, and a context already opened without it, or an earlier restriction, refuses it.
pub fn restrict_devices_to(stem: &str) -> Result<(), String> {
    let restriction = presentable(stem)?;
    RESTRICTION.set(Some(restriction)).map_err(|_| {
        "the device restriction is already decided: pass --profiles-layer once, before any \
         device opens"
            .to_string()
    })
}

/// Committed profile `stem` as a restriction the layer can present
/// ([`ProfileRestriction::presentation_gap`]): what a command restricted to it (`--profiles-layer
/// P`, a batch before its first run) refuses otherwise, before any device is read.
pub fn presentable(stem: &str) -> Result<ProfileRestriction, String> {
    let restriction = ProfileRestriction::committed(stem)?;
    match restriction.presentation_gap() {
        Some(gap) => Err(gap),
        None => Ok(restriction),
    }
}

/// The restriction of every context this process opens ([`restrict_devices_to`]); reading it
/// decides that there is none if none was set.
pub(super) fn current_restriction() -> Option<&'static ProfileRestriction> {
    RESTRICTION.get_or_init(|| None).as_ref()
}

/// The committed profile of [`current_restriction`]: what an engine device worker must
/// restrict its own contexts to before it opens a device.
pub fn requested_restriction() -> Option<String> {
    current_restriction().map(|restriction| restriction.stem.clone())
}

/// How an instance uses the profiles layer.
#[derive(Clone, Copy, Debug)]
pub(super) enum ProfilesLayer<'a> {
    /// The layer restricts the device to the profile, and hides these extensions: the ones it
    /// presents for the profile's version that the profile does not list
    /// ([`ProfileRestriction::hidden`], read through an instance that hides none).
    Restrict(&'a ProfileRestriction, &'a BTreeSet<String>),
    /// The layer passes the device through (`profile_emulation` off): the native probe. A profile
    /// an environment names is not loaded either, since its file and name are read only with
    /// emulation on.
    PassThrough,
}

/// One `VkLayerSettingEXT` value list, owned.
enum SettingValues {
    Bool(vk::Bool32),
    Strings {
        /// The strings `pointers` points into, kept alive with them.
        owned: Vec<CString>,
        pointers: Vec<*const c_char>,
    },
}

/// The profiles layer's settings for one instance, owned for as long as the create info that
/// points into them lives.
pub(super) struct ProfilesSettings {
    values: Vec<(&'static CStr, SettingValues)>,
}

impl ProfilesSettings {
    pub(super) fn new(layer: ProfilesLayer<'_>) -> Result<ProfilesSettings, String> {
        let strings = |values: &[&CStr]| {
            let owned: Vec<CString> = values.iter().map(|&value| value.to_owned()).collect();
            let pointers = owned.iter().map(|value| value.as_ptr()).collect();
            SettingValues::Strings { owned, pointers }
        };
        // The pinned layer emulates a portability device by default: it adds
        // `VK_KHR_portability_subset` and hides the formats such a device lacks (scaled and
        // three-component vertex formats, the packed ABGR formats, ...), whether or not it loads a
        // profile. Neither describes the device or the profile, so both instances turn it off.
        let portability = (c"emulate_portability", SettingValues::Bool(vk::FALSE));
        let values = match layer {
            ProfilesLayer::PassThrough => {
                vec![
                    (c"profile_emulation", SettingValues::Bool(vk::FALSE)),
                    portability,
                ]
            }
            ProfilesLayer::Restrict(restriction, hidden) => {
                let file = CString::new(restriction.file.to_string_lossy().into_owned())
                    .map_err(|_| format!("{} contains NUL", restriction.file.display()))?;
                let name = CString::new(restriction.profile_name.as_str())
                    .map_err(|_| format!("{:?} contains NUL", restriction.profile_name))?;
                let hidden = hidden
                    .iter()
                    .map(|name| {
                        CString::new(name.as_str()).map_err(|_| format!("{name:?} contains NUL"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let hidden: Vec<&CStr> = hidden.iter().map(CString::as_c_str).collect();
                let mut values = vec![
                    (c"profile_emulation", SettingValues::Bool(vk::TRUE)),
                    (c"profile_file", strings(&[&file])),
                    (c"profile_name", strings(&[&name])),
                    (c"simulate_capabilities", strings(&SIMULATED)),
                    // A feature P does not state reads false.
                    (
                        c"default_feature_values",
                        strings(&[c"DEFAULT_FEATURE_VALUES_FALSE"]),
                    ),
                    // The context's own checks decide (`check_within`, `check_restricted`). The
                    // layer's are not usable for it: with false defaults it compares P's features
                    // with the defaulted values, not the device's, and it compares an `exact`
                    // member P states with the device's in both directions. With its failure path
                    // on, enumeration failed under `gate-lavapipe` on the device it was captured
                    // from, after warning, among others, that the stated `deviceType` and
                    // `driverID` read as 0. So it neither fails nor reports.
                    (c"debug_fail_on_error", SettingValues::Bool(vk::FALSE)),
                    (c"debug_reports", strings(&[])),
                    portability,
                ];
                // An empty list is left out rather than passed as a setting with no values.
                if !hidden.is_empty() {
                    values.push((c"exclude_device_extensions", strings(&hidden)));
                }
                values
            }
        };
        Ok(ProfilesSettings { values })
    }

    /// The settings as a runtime device's options take them (`DeviceOptions::layer_settings`),
    /// owned.
    pub(super) fn runtime_settings(&self) -> Vec<LayerSetting> {
        self.values
            .iter()
            .map(|(name, values)| {
                let value = match values {
                    SettingValues::Bool(value) => LayerSettingValue::Bool(*value == vk::TRUE),
                    SettingValues::Strings { owned, .. } => {
                        LayerSettingValue::Strings(owned.clone())
                    }
                };
                LayerSetting::new(PROFILES_LAYER.to_owned(), (*name).to_owned(), value)
            })
            .collect()
    }

    /// The settings, pointing into `self`.
    pub(super) fn settings(&self) -> Vec<vk::LayerSettingEXT<'_>> {
        self.values
            .iter()
            .map(|(name, values)| {
                let mut setting = vk::LayerSettingEXT::default()
                    .layer_name(PROFILES_LAYER)
                    .setting_name(name);
                // `LayerSettingEXT::values` counts bytes, not values; set the count directly.
                match values {
                    SettingValues::Bool(value) => {
                        setting = setting.ty(vk::LayerSettingTypeEXT::BOOL32);
                        setting.value_count = 1;
                        setting.p_values = std::ptr::from_ref(value).cast();
                    }
                    SettingValues::Strings { pointers, .. } => {
                        setting = setting.ty(vk::LayerSettingTypeEXT::STRING);
                        setting.value_count = pointers.len() as u32;
                        setting.p_values = pointers.as_ptr().cast();
                    }
                }
                setting
            })
            .collect()
    }
}

/// The profiles layer's tool among `tools`, if it is active.
pub(super) fn profiles_tool(tools: &[ActiveTool]) -> Option<&ActiveTool> {
    tools
        .iter()
        .find(|tool| tool.layer.as_bytes() == PROFILES_LAYER.to_bytes())
}

/// Why a context cannot run restricted to a profile. Every variant fails the context: nothing
/// runs, and no row is recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestrictionError {
    /// The profiles layer is not installed.
    LayerMissing,
    /// The restriction was requested, but the active tools do not include the profiles layer.
    Inactive { profile: String },
    /// The profile states a capability the device lacks (`exceeds`).
    Exceeds { profile: String, gaps: Vec<String> },
    /// Under the layer a member it keeps from the device (the device's identity, [`KEPT`]) is
    /// not the native device's.
    Kept {
        profile: String,
        fields: Vec<Difference>,
    },
    /// Under the layer the queried description is not the committed profile.
    Mismatch {
        profile: String,
        fields: Vec<Difference>,
    },
}

impl fmt::Display for RestrictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |items: &mut dyn Iterator<Item = String>| items.collect::<Vec<_>>().join("; ");
        match self {
            RestrictionError::LayerMissing => write!(
                f,
                "{} is not installed; a restricted context never runs without it",
                PROFILES_LAYER.to_string_lossy()
            ),
            RestrictionError::Inactive { profile } => write!(
                f,
                "profile {profile}: the active tools do not include {}",
                PROFILES_LAYER.to_string_lossy()
            ),
            RestrictionError::Exceeds { profile, gaps } => write!(
                f,
                "profile {profile} states {} capabilities the device lacks: {}",
                gaps.len(),
                list(&mut gaps.iter().cloned())
            ),
            RestrictionError::Kept { profile, fields } => write!(
                f,
                "under profile {profile} the device changed members the layer keeps from it: {}",
                list(&mut fields.iter().map(Difference::to_string))
            ),
            RestrictionError::Mismatch { profile, fields } => write!(
                f,
                "under profile {profile} the queried description differs from the committed \
                 profile in {} members: {}",
                fields.len(),
                list(&mut fields.iter().map(Difference::to_string))
            ),
        }
    }
}

/// One member whose value differs: `path` in the description's terms, the committed (or native)
/// value, and the queried one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    pub path: String,
    pub expected: String,
    pub queried: String,
}

impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: expected {}, queried {}",
            self.path, self.expected, self.queried
        )
    }
}

/// The first check of a restricted context, on the device as it is: P, read at `instance` (the
/// instance's API version), states nothing `native` lacks ([`gaps_at`]).
pub(super) fn check_within(
    restriction: &ProfileRestriction,
    native: &DeviceDescription,
    instance: ApiVersion,
) -> Result<(), RestrictionError> {
    let gaps = gaps_at(restriction, native, instance);
    if gaps.is_empty() {
        Ok(())
    } else {
        Err(RestrictionError::Exceeds {
            profile: restriction.stem.clone(),
            gaps,
        })
    }
}

/// What P states that `device` lacks, both read as a context opened at `instance` reads them: the
/// one comparison [`check_within`] makes of the device as it is.
fn gaps_at(
    restriction: &ProfileRestriction,
    device: &DeviceDescription,
    instance: ApiVersion,
) -> Vec<String> {
    exceeds(&restriction.read_at(instance), &read_at(device, instance))
}

/// The checks of a restricted context before its device is created, after [`check_within`]: the
/// layer active, the members it keeps from the device the native device's, and every other member
/// of the queried description equal to P read at `instance`. Returns the description the context
/// translates for: P read at `instance` ([`ProfileRestriction::read_at`]) on the device that
/// executes it, whose identity the layer keeps. The product keys its driver workarounds on that
/// identity, so they follow the driver that runs the module, not the one P was captured on
/// (SwiftShader's buffer atomics).
pub(super) fn check_restricted(
    restriction: &ProfileRestriction,
    native: &DeviceDescription,
    tools: &[ActiveTool],
    queried: &DeviceDescription,
    instance: ApiVersion,
) -> Result<DeviceDescription, RestrictionError> {
    let profile = restriction.stem.clone();
    if profiles_tool(tools).is_none() {
        return Err(RestrictionError::Inactive { profile });
    }
    let expected = restriction.read_at(instance);
    let mut changed = identity_differences(native, queried);
    let native_read = read_at(native, instance);
    changed.extend(kept_outside(
        &capability_members(&expected),
        &capability_members(queried),
        &capability_members(&native_read),
    ));
    changed.extend(kept_sample_counts(&expected, queried, &native_read));
    if !changed.is_empty() {
        return Err(RestrictionError::Kept {
            profile,
            fields: changed,
        });
    }
    let kept = |difference: &Difference| {
        KEPT.contains(&difference.path.as_str()) || difference.path.ends_with(SAMPLE_COUNTS_FIELD)
    };
    let fields: Vec<Difference> = differences(&expected, queried)
        .into_iter()
        .filter(|difference| !kept(difference))
        .collect();
    if !fields.is_empty() {
        return Err(RestrictionError::Mismatch { profile, fields });
    }
    let mut translated = expected;
    translated.driver_id = native.driver_id;
    translated.driver_version = native.driver_version;
    translated.vendor_id = native.vendor_id;
    translated.device_id = native.device_id;
    translated.device_name.clone_from(&native.device_name);
    translated.device_type = native.device_type;
    Ok(translated)
}

/// A format's color attachment sample counts, as [`differences`] names the member.
const SAMPLE_COUNTS_FIELD: &str = ".color_attachment_sample_counts";

/// The formats whose color attachment sample counts `queried` states do not lie between the
/// profile's (`expected`) and the native device's: no profiles-schema structure holds
/// `VkImageFormatProperties`, so the layer answers the image format query from the device, as it
/// keeps the [`KEPT`] members, and P's counts must lie within the device's ([`exceeds`]).
fn kept_sample_counts(
    expected: &DeviceDescription,
    queried: &DeviceDescription,
    native: &DeviceDescription,
) -> Vec<Difference> {
    let counts = |description: &DeviceDescription, format| {
        description
            .formats
            .formats
            .get(format)
            .map(|p| p.color_attachment_sample_counts)
            .unwrap_or_default()
    };
    let formats: BTreeSet<_> = [expected, queried, native]
        .iter()
        .flat_map(|description| description.formats.formats.keys())
        .collect();
    formats
        .into_iter()
        .filter_map(|format| {
            let (low, value, high) = (
                counts(expected, format),
                counts(queried, format),
                counts(native, format),
            );
            (!value.contains(low) || !high.contains(value)).then(|| Difference {
                path: format!("formats.{format:?}{SAMPLE_COUNTS_FIELD}"),
                expected: format!("between {:#x} and {:#x}", low.as_raw(), high.as_raw()),
                queried: format!("{:#x}", value.as_raw()),
            })
        })
        .collect()
}

/// The [`KEPT`] members whose `queried` value does not lie between the profile's (`expected`)
/// and the native device's under the member's [`Bound`]: the layer either presented more than the
/// device has or less than P states.
fn kept_outside(expected: &Value, queried: &Value, native: &Value) -> Vec<Difference> {
    let member = |value: &Value, path: &[String]| {
        path.iter()
            .try_fold(value, |value, key| value.get(key))
            .cloned()
            .unwrap_or(Value::Null)
    };
    KEPT.iter()
        .filter_map(|path| {
            let path: Vec<String> = path.split('.').map(str::to_owned).collect();
            let rule = bound(&path).expect("every kept member has a subset rule");
            let (low, value, high) = (
                member(expected, &path),
                member(queried, &path),
                member(native, &path),
            );
            (!within(rule, &low, &value) || !within(rule, &value, &high)).then(|| Difference {
                path: path.join("."),
                expected: format!("between {low} and {high}"),
                queried: value.to_string(),
            })
        })
        .collect()
}

/// The members that name the device rather than describe its capabilities: `vk.xml` gives them
/// `limittype="noauto"`, and the layer keeps them from the device ([`check_restricted`]).
fn identity(description: &DeviceDescription) -> [(&'static str, String); 6] {
    [
        ("driver_id", format!("{:?}", description.driver_id)),
        ("driver_version", description.driver_version.to_string()),
        ("vendor_id", description.vendor_id.to_string()),
        ("device_id", description.device_id.to_string()),
        ("device_name", format!("{:?}", description.device_name)),
        ("device_type", format!("{:?}", description.device_type)),
    ]
}

fn identity_differences(
    native: &DeviceDescription,
    queried: &DeviceDescription,
) -> Vec<Difference> {
    identity(native)
        .into_iter()
        .zip(identity(queried))
        .filter(|((_, a), (_, b))| a != b)
        .map(|((path, expected), (_, queried))| Difference {
            path: path.into(),
            expected,
            queried,
        })
        .collect()
}

/// `description`'s capability members as JSON: everything but its identity, its sets (features
/// and extensions), and its format table, which are compared typed.
fn capability_members(description: &DeviceDescription) -> Value {
    let mut value = serde_json::to_value(description).expect("a description serializes");
    if let Some(object) = value.as_object_mut() {
        for key in [
            "driver_id",
            "driver_version",
            "vendor_id",
            "device_id",
            "device_name",
            "device_type",
            "extensions",
            "features",
            "formats",
        ] {
            object.remove(key);
        }
    }
    value
}

/// The check of a venue run's device against the venue's committed profile `venue` and the device
/// identities the venue allows, `allowed`: `live` is the device's identity and `read` its
/// description as the run's context queried it, read at API 1.2 as the profile is ([`read_at`]).
/// The run stops on an identity none of `allowed` is, naming each field it differs in from the
/// nearest allowed identity (the one it differs from in fewest fields, the first of those), and on
/// a translation key ([`crate::c3::VenueProfile::key_of`]) that is not the profile's, with every
/// description member that differs (both directions, [`differences`]) beside it as the explanation;
/// members that differ while the key holds are no translation's input (an allowed device's
/// allocation sizes among them), and are only reported.
pub fn venue_check(
    venue: &crate::c3::VenueProfile,
    allowed: &[crate::c3::DeviceIdentity],
    live: &crate::c3::DeviceIdentity,
    read: &DeviceDescription,
) -> Result<crate::c3::VenueCheck, String> {
    if allowed.is_empty() {
        return Err(format!(
            "venue profile {}: the venue allows no device",
            venue.stem
        ));
    }
    let mut stops = allowed
        .iter()
        .map(|identity| identity.differences(live))
        .min_by_key(Vec::len)
        .unwrap_or_default()
        .into_iter()
        .map(|difference| format!("identity {difference}"))
        .collect::<Vec<_>>();
    let described = differences(&venue.description, read)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let key = crate::c3::VenueProfile::key_of(read)?;
    if key == venue.fingerprint {
        return Ok(crate::c3::VenueCheck {
            stops,
            unread: described,
        });
    }
    stops.push(format!("translation key: {} ({key})", venue.fingerprint));
    stops.extend(described);
    Ok(crate::c3::VenueCheck {
        stops,
        unread: Vec::new(),
    })
}

/// Every capability member where `queried` differs from `expected` (identity excluded): added and
/// missing features and extensions, each format's differing feature flags, and every other member
/// by its path.
pub fn differences(expected: &DeviceDescription, queried: &DeviceDescription) -> Vec<Difference> {
    let mut out = Vec::new();
    set_differences("features", &expected.features, &queried.features, &mut out);
    set_differences(
        "extensions",
        &expected.extensions,
        &queried.extensions,
        &mut out,
    );
    let formats: BTreeSet<_> = expected
        .formats
        .formats
        .keys()
        .chain(queried.formats.formats.keys())
        .copied()
        .collect();
    for format in formats {
        let a = expected
            .formats
            .formats
            .get(&format)
            .copied()
            .unwrap_or_default();
        let b = queried
            .formats
            .formats
            .get(&format)
            .copied()
            .unwrap_or_default();
        for (field, a, b) in [
            ("linear_tiling", a.linear_tiling, b.linear_tiling),
            ("optimal_tiling", a.optimal_tiling, b.optimal_tiling),
            ("buffer", a.buffer, b.buffer),
        ] {
            if a != b {
                out.push(Difference {
                    path: format!("formats.{format:?}.{field}"),
                    expected: format!("{:#x}", a.as_raw()),
                    queried: format!("{:#x}", b.as_raw()),
                });
            }
        }
        let (a, b) = (
            a.color_attachment_sample_counts,
            b.color_attachment_sample_counts,
        );
        if a != b {
            out.push(Difference {
                path: format!("formats.{format:?}{SAMPLE_COUNTS_FIELD}"),
                expected: format!("{:#x}", a.as_raw()),
                queried: format!("{:#x}", b.as_raw()),
            });
        }
    }
    let mut path = Vec::new();
    json_differences(
        &mut path,
        &capability_members(expected),
        &capability_members(queried),
        &mut out,
    );
    out
}

fn set_differences<T: Ord + fmt::Debug>(
    what: &str,
    expected: &BTreeSet<T>,
    queried: &BTreeSet<T>,
    out: &mut Vec<Difference>,
) {
    for (expected_has, item) in expected
        .difference(queried)
        .map(|item| (true, item))
        .chain(queried.difference(expected).map(|item| (false, item)))
    {
        out.push(Difference {
            path: format!("{what}.{item:?}"),
            expected: if expected_has { "listed" } else { "absent" }.into(),
            queried: if expected_has { "absent" } else { "listed" }.into(),
        });
    }
}

fn json_differences(path: &mut Vec<String>, a: &Value, b: &Value, out: &mut Vec<Difference>) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            for key in keys {
                path.push(key.clone());
                json_differences(
                    path,
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                    out,
                );
                path.pop();
            }
        }
        (a, b) if a != b => out.push(Difference {
            path: path.join("."),
            expected: a.to_string(),
            queried: b.to_string(),
        }),
        _ => {}
    }
}

/// How a member a profile states must relate to the device's for the device to have it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    /// A maximum: the profile's value is at most the device's.
    AtMost,
    /// A minimum (an alignment, a negative offset): at least the device's; a missing (`null`)
    /// value states no requirement, the most capable.
    AtLeast,
    /// `AtMost` of an `f32` stored as its bits.
    FloatAtMost,
    /// `AtLeast` of an `f32` stored as its bits.
    FloatAtLeast,
    /// Flag bits: the profile's are a subset of the device's.
    Flags,
    /// A capability boolean: the profile's true needs the device's true.
    Implies,
    /// A `VkShaderFloatControlsIndependence`: no more independent than the device's.
    Independence,
    /// Describes execution, not capacity (the subgroup size; how lines rasterize, which the
    /// runtime reads to choose how it draws a wide line and never to refuse one): not compared.
    Execution,
    /// A size-control bound that a device without subgroup size control reports as 0 (no value):
    /// the profile's range must lie within the device's.
    SubgroupMin,
    /// An optional member (the generated-commands facts): a profile that states it needs a device
    /// that does; where both do, its members have their own rules.
    Present,
}

/// The [`Bound`] of the capability member at `path` (the description's serialized member names),
/// or `None` for a member this check has no rule for: then the check fails rather than skip it.
fn bound(path: &[String]) -> Option<Bound> {
    let path: Vec<&str> = path.iter().map(String::as_str).collect();
    Some(match path.as_slice() {
        // `VK_EXT_mesh_shader`'s limits, all maxima.
        ["limits", "mesh", name, ..] if name.starts_with("max_") => Bound::AtMost,
        ["limits", name, ..] => return limit_bound(name),
        // The allocation sizes: no translation reads them, but a restricted run presents them to
        // the runtime, so a profile's must be within the device's.
        ["allocation", "max_memory_allocation_size" | "max_buffer_size"] => Bound::AtMost,
        ["subgroup", "size"] => Bound::Execution,
        ["subgroup", "min_size"] => Bound::SubgroupMin,
        ["subgroup", "max_size" | "max_compute_workgroup_subgroups"] => Bound::AtMost,
        ["subgroup", "supported_stages" | "supported_operations" | "required_size_stages"] => {
            Bound::Flags
        }
        ["subgroup", "quad_operations_in_all_stages"] => Bound::Implies,
        ["float_controls", "f16" | "f32" | "f64", _] => Bound::Implies,
        ["float_controls", "denorm_independence" | "rounding_independence"] => Bound::Independence,
        ["filter_minmax_single_component_formats"
        | "rgba4_opaque_black_swizzled"
        | "robust_buffer_access_update_after_bind"
        | "khr_format_feature_flags2"] => Bound::Implies,
        ["generated_commands"] => Bound::Present,
        ["generated_commands", "max_indirect_pipeline_count"
        | "max_indirect_sequence_count"
        | "max_indirect_commands_token_count"
        | "max_indirect_commands_token_offset"
        | "max_indirect_commands_indirect_stride"] => Bound::AtMost,
        ["generated_commands", "supported_stages" | "pipeline_binding_stages"] => Bound::Flags,
        _ => return None,
    })
}

/// A limit's [`Bound`], from Vulkan's naming: `max*` limits are maxima, `min*Alignment` limits and
/// the negative minimum offsets are minima, sample counts and resolve modes are flags, and the
/// boolean properties are implications.
fn limit_bound(name: &str) -> Option<Bound> {
    Some(match name {
        "min_interpolation_offset_bits" => Bound::FloatAtLeast,
        "line_width_range_bits"
        | "line_width_granularity_bits"
        | "strict_lines"
        | "non_strict_wide_lines_use_parallelogram"
        | "sub_pixel_precision_bits" => Bound::Execution,
        "min_texel_offset" | "min_texel_gather_offset" => Bound::AtLeast,
        "supports_non_zero_first_instance"
        | "independent_resolve_none"
        | "independent_resolve"
        | "variable_sample_locations"
        | "provoking_vertex_mode_per_pipeline" => Bound::Implies,
        "framebuffer_color_sample_counts"
        | "framebuffer_depth_sample_counts"
        | "framebuffer_integer_color_sample_counts"
        | "framebuffer_stencil_sample_counts"
        | "framebuffer_no_attachments_sample_counts"
        | "storage_image_sample_counts"
        | "supported_depth_resolve_modes"
        | "supported_stencil_resolve_modes"
        | "sample_location_sample_counts" => Bound::Flags,
        name if name.starts_with("max_") && name.ends_with("_bits") => Bound::FloatAtMost,
        name if name.starts_with("min_") && name.ends_with("_alignment") => Bound::AtLeast,
        name if name.starts_with("max_") => Bound::AtMost,
        _ => return None,
    })
}

/// What `profile` states that `device` does not have, one entry per member: a higher API version,
/// a feature or extension the device does not list, format feature bits or color attachment sample
/// counts it lacks, and every other capability member beyond the device's by its [`Bound`]. A
/// member without a rule is reported as such, never skipped. Identity is not a capability (the layer keeps it, [`check_restricted`]).
pub fn exceeds(profile: &DeviceDescription, device: &DeviceDescription) -> Vec<String> {
    let mut gaps = Vec::new();
    if profile.api_version > device.api_version {
        gaps.push(format!(
            "api_version {}.{} above the device's {}.{}",
            profile.api_version.major,
            profile.api_version.minor,
            device.api_version.major,
            device.api_version.minor
        ));
    }
    gaps.extend(
        profile
            .features
            .difference(&device.features)
            .map(|feature| format!("feature {feature:?}")),
    );
    gaps.extend(
        profile
            .extensions
            .difference(&device.extensions)
            .map(|extension| format!("extension {extension:?}")),
    );
    for (format, wanted) in &profile.formats.formats {
        let has = device
            .formats
            .formats
            .get(format)
            .copied()
            .unwrap_or_default();
        for (field, wanted, has) in [
            ("linear_tiling", wanted.linear_tiling, has.linear_tiling),
            ("optimal_tiling", wanted.optimal_tiling, has.optimal_tiling),
            ("buffer", wanted.buffer, has.buffer),
        ] {
            if !has.contains(wanted) {
                gaps.push(format!(
                    "formats.{format:?}.{field} {:#x} beyond the device's {:#x}",
                    wanted.as_raw(),
                    has.as_raw()
                ));
            }
        }
        // A color attachment's sample counts are flags too.
        let (wanted, has) = (
            wanted.color_attachment_sample_counts,
            has.color_attachment_sample_counts,
        );
        if !has.contains(wanted) {
            gaps.push(format!(
                "formats.{format:?}{SAMPLE_COUNTS_FIELD} {:#x} beyond the device's {:#x}",
                wanted.as_raw(),
                has.as_raw()
            ));
        }
    }
    let mut path = Vec::new();
    let mut profile_members = capability_members(profile);
    let mut device_members = capability_members(device);
    for members in [&mut profile_members, &mut device_members] {
        if let Some(object) = members.as_object_mut() {
            object.remove("api_version");
        }
    }
    json_exceeds(
        &mut path,
        &profile_members,
        &device_members,
        None,
        &mut gaps,
    );
    gaps
}

fn json_exceeds(
    path: &mut Vec<String>,
    profile: &Value,
    device: &Value,
    inherited: Option<Bound>,
    gaps: &mut Vec<String>,
) {
    if let (Value::Object(profile), Value::Object(device)) = (profile, device) {
        for (key, value) in profile {
            path.push(key.clone());
            json_exceeds(
                path,
                value,
                device.get(key).unwrap_or(&Value::Null),
                inherited,
                gaps,
            );
            path.pop();
        }
        return;
    }
    let Some(rule) = inherited.or_else(|| bound(path)) else {
        gaps.push(format!(
            "{} has no subset rule (the check knows no rule for this member)",
            path.join(".")
        ));
        return;
    };
    if let (Value::Array(profile), Value::Array(device)) = (profile, device) {
        for (index, (p, d)) in profile.iter().zip(device).enumerate() {
            path.push(index.to_string());
            json_exceeds(path, p, d, Some(rule), gaps);
            path.pop();
        }
        if profile.len() != device.len() {
            gaps.push(format!("{} has another length", path.join(".")));
        }
        return;
    }
    if !within(rule, profile, device) {
        gaps.push(format!(
            "{} {profile} beyond the device's {device}",
            path.join(".")
        ));
    }
}

/// Whether the device's `device` value provides the profile's `profile` under `rule`.
fn within(rule: Bound, profile: &Value, device: &Value) -> bool {
    let integer = |value: &Value| {
        value
            .as_i64()
            .map(i128::from)
            .or_else(|| value.as_u64().map(i128::from))
    };
    let float = |value: &Value| {
        value
            .as_u64()
            .and_then(|bits| u32::try_from(bits).ok())
            .map(f32::from_bits)
    };
    match rule {
        Bound::Execution => true,
        Bound::Implies => !(profile.as_bool() == Some(true) && device.as_bool() != Some(true)),
        Bound::AtMost => match (integer(profile), integer(device)) {
            (Some(p), Some(d)) => p <= d,
            _ => false,
        },
        Bound::AtLeast => match (profile, device) {
            (Value::Null, Value::Null) => true,
            (_, Value::Null) => true,
            (Value::Null, _) => false,
            _ => matches!((integer(profile), integer(device)), (Some(p), Some(d)) if p >= d),
        },
        Bound::FloatAtMost => {
            matches!((float(profile), float(device)), (Some(p), Some(d)) if p <= d)
        }
        Bound::FloatAtLeast => {
            matches!((float(profile), float(device)), (Some(p), Some(d)) if p >= d)
        }
        Bound::Flags => match (integer(profile), integer(device)) {
            (Some(p), Some(d)) => p & !d == 0,
            _ => false,
        },
        Bound::Independence => {
            let parse = |value: &Value| serde_json::from_value::<Independence>(value.clone()).ok();
            match (parse(profile), parse(device)) {
                (Some(p), Some(d)) => d.admits(p),
                _ => false,
            }
        }
        // A profile without size control states no range; a device without it has none to give.
        Bound::SubgroupMin => match (integer(profile), integer(device)) {
            (Some(0), _) => true,
            (Some(p), Some(d)) => d != 0 && p >= d,
            _ => false,
        },
        Bound::Present => profile.is_null() || !device.is_null(),
    }
}

#[cfg(test)]
mod tests;
