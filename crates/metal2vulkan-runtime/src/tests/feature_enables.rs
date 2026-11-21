//! What the translator's profile reads against what device creation enables: the translator's
//! description of a device states every feature and extension creation enables that the
//! translator names, and the translator's profile reads no feature a device reports that creation
//! does not enable, but those the README's `Not enabled` table records (`## Floor`), each with the
//! form or the refusal a program meets without it.

use super::floor::{at_the_floor, table};
use crate::floor::{
    decide, shader_enables, shader_extension, vulkan_name, Capabilities, Queried,
    ATTACHMENT_READ_EXTENSIONS, BARYCENTRIC_EXTENSIONS, BORDER_SWIZZLE_EXTENSIONS,
    DEVICE_EXTENSIONS, ENABLED, INDIRECT_COMMANDS, OPTIONAL_EXTENSIONS,
};
use ash::vk;
use metal2vulkan::device::{
    Device, DeviceDescription, DevicePolicy, DeviceProfile, EnableSet, VkExtension, VkFeature,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::fs;
use std::path::PathBuf;

/// The translator's feature of the Vulkan member `member` (`shaderInt64`), where its vocabulary
/// names one: the vocabulary names each feature by its member, capitalized
/// (`VkFeature::ShaderInt64`).
fn vocabulary(member: &str) -> Option<VkFeature> {
    let mut chars = member.chars();
    let first = chars.next()?;
    let name: String = first.to_uppercase().chain(chars).collect();
    VkFeature::ALL
        .iter()
        .copied()
        .find(|feature| format!("{feature:?}") == name)
}

/// Every extension device creation may enable.
fn creation_extensions() -> BTreeSet<&'static CStr> {
    (DEVICE_EXTENSIONS.iter())
        .chain(OPTIONAL_EXTENSIONS)
        .chain(ATTACHMENT_READ_EXTENSIONS)
        .chain(BORDER_SWIZZLE_EXTENSIONS)
        .chain(BARYCENTRIC_EXTENSIONS)
        .chain(INDIRECT_COMMANDS)
        .copied()
        .collect()
}

/// The data rows of the README's table headed `header` under its subsection `title`.
fn readme_table(title: &str, header: &[&str]) -> Vec<Vec<String>> {
    let path = crate::tests::crate_dir().join("README.md");
    let readme = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let heading = format!("### {title}");
    let section: Vec<&str> = readme
        .lines()
        .skip_while(|line| *line != heading)
        .skip(1)
        .take_while(|line| !line.starts_with("## ") && !line.starts_with("### "))
        .collect();
    assert!(!section.is_empty(), "the README has no `{heading}`");
    table(&section, header)
}

/// The items of the README's `Enabled, not stated` table of the kind `kind` (`feature` or
/// `extension`), by their Vulkan names.
fn unstated(kind: &str) -> BTreeSet<String> {
    readme_table(
        "Enabled, not stated",
        &["Item", "Kind", "Why it is not stated"],
    )
    .into_iter()
    .filter(|row| row[1].starts_with(kind))
    .map(|row| row[0].clone())
    .collect()
}

/// The README's `Not enabled` table: each feature it records, by the translator's name.
fn recorded() -> BTreeSet<VkFeature> {
    readme_table("Not enabled", &["Feature", "Structure", "Without it"])
        .into_iter()
        .map(|row| {
            vocabulary(&row[0])
                .unwrap_or_else(|| panic!("{}: not a feature the translator names", row[0]))
        })
        .collect()
}

/// Each feature of a table device creation enables, by the translator's name, where it names
/// one.
fn creation_features() -> BTreeSet<VkFeature> {
    ENABLED
        .iter()
        .copied()
        .flatten()
        .filter_map(|feature| vocabulary(&vulkan_name(feature.member)))
        .collect()
}

/// Every entry of a table device creation enables carries the translator's name for its member,
/// where the translator names it, so a feature creation enables reaches the translator's
/// description wherever creation enables it; and a device that reports every one of them gives
/// the translator each. The exceptions are the README's `Enabled, not stated` features, each of
/// which creation enables, the translator names, and the description does not state.
#[test]
fn every_feature_creation_enables_reaches_the_translator() {
    let unstated = unstated("feature");
    let mut met = BTreeSet::new();
    let mut device = at_the_floor();
    for feature in ENABLED.iter().copied().flatten() {
        *(feature.field)(&mut device.features) = vk::TRUE;
    }
    device.extensions = creation_extensions()
        .into_iter()
        .map(CStr::to_owned)
        .collect();
    let capabilities = decide(&device).unwrap_or_else(|below| panic!("{below}"));
    let enables = shader_enables(&capabilities);
    let mut named = 0;
    for feature in ENABLED.iter().copied().flatten() {
        let member = vulkan_name(feature.member);
        let name = vocabulary(&member);
        if unstated.contains(&member) {
            let name = name.unwrap_or_else(|| {
                panic!("{member}: in `Enabled, not stated`, and the translator does not name it")
            });
            assert_eq!(feature.translator, None, "{member}: stated");
            assert!(!enables.features.contains(&name), "{name:?} is passed");
            met.insert(member);
            continue;
        }
        assert_eq!(
            feature.translator, name,
            "{}.{member}: the translator's name",
            feature.structure
        );
        let mut features = capabilities.features;
        if let Some(name) = name.filter(|_| *(feature.field)(&mut features) == vk::TRUE) {
            assert!(enables.features.contains(&name), "{name:?} is not passed");
            named += 1;
        }
    }
    assert!(named > 60, "{named} named features enabled");
    assert_eq!(
        met, unstated,
        "the `Enabled, not stated` features that creation enables"
    );
}

/// Each extension device creation may enable reaches the translator's description where the
/// translator names it, as the translator itself reads its name: a profile document that lists it
/// alone gives the translator's description that one extension. The exceptions are the README's
/// `Enabled, not stated` extensions, each of which the translator names and the description does
/// not state.
#[test]
fn every_extension_creation_enables_reaches_the_translator() {
    let unstated = unstated("extension");
    let mut met = BTreeSet::new();
    let path = metal2vulkan_devtools::validation_dir().join("profiles/runtime-floor.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let mut doc: Value = serde_json::from_str(&text).expect("the floor profile");
    for block in doc["capabilities"]
        .as_object_mut()
        .expect("its capabilities")
        .values_mut()
    {
        if let Some(block) = block.as_object_mut() {
            block.remove("extensions");
        }
    }
    let profile = doc["profiles"]
        .as_object()
        .and_then(|profiles| profiles.keys().next().cloned())
        .expect("its profile");
    let read = |doc: &Value| {
        DeviceDescription::from_khronos_json(&doc.to_string(), None)
            .unwrap_or_else(|error| panic!("the floor profile: {error}"))
            .extensions
    };
    let bare = read(&doc);
    for name in creation_extensions() {
        let mut listed = doc.clone();
        listed["capabilities"]["m2v_listed"] =
            serde_json::json!({ "extensions": { name.to_string_lossy(): 1 } });
        listed["profiles"][&profile]["capabilities"]
            .as_array_mut()
            .expect("the profile's capabilities")
            .push(Value::from("m2v_listed"));
        let read: Vec<VkExtension> = read(&listed).difference(&bare).copied().collect();
        let vulkan = name.to_string_lossy().into_owned();
        if unstated.contains(&vulkan) {
            assert!(
                !read.is_empty(),
                "{vulkan}: in `Enabled, not stated`, and the translator does not name it"
            );
            assert_eq!(shader_extension(name), None, "{vulkan}: stated");
            met.insert(vulkan);
            continue;
        }
        assert_eq!(
            shader_extension(name),
            read.first().copied(),
            "{name:?}: the translator reads {read:?}"
        );
    }
    assert_eq!(
        met, unstated,
        "the `Enabled, not stated` extensions that creation may enable"
    );
}

/// A committed device profile as the floor reads a device: its extensions, with the floor's that the
/// translator does not name (a profile document states no window-system extension), and each
/// feature of a table creation enables where the document states it, or, where it does not, where
/// the translator's description of it holds it (a member the profile's Vulkan version requires).
/// The rest is the floor's device.
fn queried(doc: &Value, description: &DeviceDescription) -> Queried {
    let blocks: Vec<&serde_json::Map<String, Value>> = doc["capabilities"]
        .as_object()
        .map(|blocks| blocks.values().filter_map(Value::as_object).collect())
        .unwrap_or_default();
    let mut device = at_the_floor();
    device.version = vk::make_api_version(
        0,
        description.api_version.major,
        description.api_version.minor,
        0,
    );
    device.extensions = blocks
        .iter()
        .filter_map(|block| block.get("extensions").and_then(Value::as_object))
        .flat_map(|extensions| extensions.keys())
        .map(|name| CString::new(name.as_str()).expect("an extension name"))
        .chain(
            (DEVICE_EXTENSIONS.iter())
                .filter(|name| shader_extension(name).is_none())
                .map(|name| CStr::to_owned(name)),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for feature in ENABLED.iter().copied().flatten() {
        let member = vulkan_name(feature.member);
        let stated = blocks
            .iter()
            .filter_map(|block| block.get("features").and_then(Value::as_object))
            .flat_map(|structures| structures.values())
            .filter_map(|members| members.get(&member).and_then(Value::as_bool))
            .next();
        let on = stated
            .or_else(|| vocabulary(&member).map(|name| description.has(name)))
            .unwrap_or(false);
        *(feature.field)(&mut device.features) = if on { vk::TRUE } else { vk::FALSE };
    }
    device
}

/// The translator's profile without its enabled set and its fingerprint: what translation reads
/// of the profile beside the set ([`DeviceProfile::fingerprint`] renders the same fields).
fn facts(profile: &DeviceProfile) -> String {
    let text = format!("{profile:?}");
    let text = match text.find(", fingerprint:") {
        Some(end) => &text[..end],
        None => &text[..],
    };
    let Some(start) = text.find("enabled: EnabledCaps {") else {
        return text.to_owned();
    };
    let mut depth = 0usize;
    let mut end = text.len();
    for (at, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = start + at + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    format!("{}{}", &text[..start], &text[end..])
}

/// The enable set `enables` with `feature`, its exposing extension where the device lists one
/// and the core version does not expose it, and what that extension requires.
fn with_feature(
    enables: &EnableSet,
    feature: VkFeature,
    description: &DeviceDescription,
) -> EnableSet {
    let api = description.api_version;
    let mut set = enables.clone();
    set.features.insert(feature);
    let exposure = feature.exposure();
    if !exposure.exposed(api, &set.extensions) {
        if let Some(&extension) = (exposure.extensions_at(api).iter())
            .find(|extension| description.extensions.contains(extension))
        {
            let mut pending = vec![extension];
            while let Some(extension) = pending.pop() {
                if set.extensions.insert(extension) {
                    pending.extend(extension.requires(api));
                }
            }
        }
    }
    set
}

/// The features `description`'s device reports, that creation (`capabilities`) does not enable
/// and whose enabling changes the translator's profile, each with the device's name.
fn read_but_withheld(
    name: &str,
    description: &DeviceDescription,
    capabilities: &Capabilities,
    read: &mut BTreeMap<VkFeature, Vec<String>>,
) -> Result<(), String> {
    let enables = shader_enables(capabilities);
    let created = description
        .clone()
        .with_enabled(&enables)
        .map_err(|error| {
            format!("{name}: the translator's description refuses creation's enables: {error}")
        })?;
    let base = Device::from_queries(created.clone(), DevicePolicy::default())
        .map_err(|error| format!("{name}: no translator for creation's enables: {error}"))?;
    let base = facts(base.profile());
    for &feature in VkFeature::ALL {
        if !description.has(feature) || created.has(feature) {
            continue;
        }
        // A feature Vulkan forbids beside what creation enables (a co-enable rule) is one no
        // device made by the runtime could have.
        let Ok(with) =
            description
                .clone()
                .with_enabled(&with_feature(&enables, feature, description))
        else {
            continue;
        };
        let Ok(device) = Device::from_queries(with, DevicePolicy::default()) else {
            continue;
        };
        if facts(device.profile()) != base {
            read.entry(feature).or_default().push(name.to_owned());
        }
    }
    Ok(())
}

/// The committed device profiles, each with its file's stem.
fn committed_profiles() -> Vec<(String, String)> {
    let dir: PathBuf = metal2vulkan_devtools::validation_dir().join("profiles");
    let mut profiles: Vec<(String, String)> = fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .map(|path| {
            let stem = path
                .file_stem()
                .expect("a stem")
                .to_string_lossy()
                .into_owned();
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            (stem, text)
        })
        .collect();
    profiles.sort();
    profiles
}

/// The features creation never enables that the translator's profile reads: on each committed
/// device profile at or above the floor, each feature the device reports that creation does not
/// enable, whose enabling changes the profile, is one the README's `Not enabled` table records;
/// and each recorded feature is one no table of creation's enables, and one the profile reads
/// where a profile reports it.
#[test]
fn the_profile_reads_no_feature_creation_withholds_unrecorded() {
    let mut read = BTreeMap::new();
    let mut reported = BTreeSet::new();
    let mut checked = Vec::new();
    let mut failures = Vec::new();
    for (name, text) in committed_profiles() {
        let Ok(description) = DeviceDescription::from_khronos_json(&text, None) else {
            continue;
        };
        let doc: Value = serde_json::from_str(&text).expect("a profile document");
        let Ok(capabilities) = decide(&queried(&doc, &description)) else {
            continue;
        };
        reported.extend(description.features.iter().copied());
        match read_but_withheld(&name, &description, &capabilities, &mut read) {
            Ok(()) => checked.push(name),
            Err(failure) => failures.push(failure),
        }
    }
    println!("feature-enables: profiles checked {checked:?}");
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(checked.len() >= 5, "{checked:?}");
    judge(&read, &reported);
}

/// Each feature `read` names is in the README's `Not enabled` table or its `Enabled, not stated`
/// one: a feature creation enables only where its own rule says, and never states, is withheld
/// wherever that rule does not hold.
fn all_recorded(read: &BTreeMap<VkFeature, Vec<String>>) {
    for (feature, devices) in read {
        println!("feature-enables: {feature:?} changes the profile on {devices:?}");
    }
    let recorded = recorded();
    let unstated: BTreeSet<VkFeature> = unstated("feature")
        .iter()
        .filter_map(|member| vocabulary(member))
        .collect();
    let unrecorded: Vec<&VkFeature> = read
        .keys()
        .filter(|f| !recorded.contains(f) && !unstated.contains(f))
        .collect();
    assert!(
        unrecorded.is_empty(),
        "read by the translator's profile, withheld by creation and in neither the README's `Not \
         enabled` table nor its `Enabled, not stated` one: {unrecorded:?}"
    );
}

/// Each feature `read` names is recorded in the README's `Not enabled` table or its `Enabled, not
/// stated` one, and each feature `Not enabled` records is not a feature creation enables, and is
/// in `read` wherever some device reports it.
fn judge(read: &BTreeMap<VkFeature, Vec<String>>, reported: &BTreeSet<VkFeature>) {
    let recorded = recorded();
    all_recorded(read);
    let creation = creation_features();
    let enabled: Vec<&VkFeature> = recorded.intersection(&creation).collect();
    assert!(
        enabled.is_empty(),
        "recorded as not enabled, but creation enables {enabled:?}"
    );
    let unread: Vec<&VkFeature> = recorded
        .iter()
        .filter(|f| reported.contains(f) && !read.contains_key(f))
        .collect();
    assert!(
        unread.is_empty(),
        "recorded as not enabled, reported by a device, and not read by the profile: {unread:?}"
    );
}

/// As [`the_profile_reads_no_feature_creation_withholds_unrecorded`], on each device of the cfg's
/// kind the machine's Vulkan loader lists, as the runtime queries it.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn the_profile_reads_no_feature_the_device_reports_that_creation_withholds_unrecorded() {
    use crate::floor::check;
    use metal2vulkan::device::ApiVersion;
    let runtime = crate::tests::validation::instance(&[]);
    let (entry, instance) = runtime.raw();
    // SAFETY: `instance` is live.
    let physicals = unsafe { instance.enumerate_physical_devices() }.expect("the physical devices");
    let mut read = BTreeMap::new();
    let mut checked = 0;
    for physical in physicals {
        // SAFETY: `physical` is one of `instance`'s.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        if (properties.device_type == vk::PhysicalDeviceType::CPU) != cfg!(m2v_cpu_icd) {
            continue;
        }
        let name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // SAFETY: as above.
        let Ok(capabilities) = (unsafe { check(entry, instance, physical) }) else {
            continue;
        };
        // SAFETY: as above, at the floor's version, which the instance was created at or above.
        let description = unsafe {
            DeviceDescription::query(
                instance,
                physical,
                ApiVersion::from_packed(crate::floor::VERSION),
            )
        }
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        read_but_withheld(&name, &description, &capabilities, &mut read)
            .unwrap_or_else(|failure| panic!("{failure}"));
        checked += 1;
    }
    drop(runtime);
    assert!(checked > 0, "the loader lists no device of this cfg's kind");
    all_recorded(&read);
}
