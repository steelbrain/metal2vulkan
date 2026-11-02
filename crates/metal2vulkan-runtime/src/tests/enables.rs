use crate::enables::{
    host_image_copy_names, linear_color_attachment_names, names_of, Enables, Promotion, CORE_1_3,
    MEMBERS,
};
use crate::error::{Error, RuntimeGap};
use ash::vk;
use metal2vulkan::device::{
    ApiVersion, DeviceDescription, DevicePolicy, EnableSet, VkExtension, VkFeature,
};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};

/// Vulkan 1.4.
const V1_4: ApiVersion = ApiVersion { major: 1, minor: 4 };

/// Every committed profile's recommendation at its version capped by `instance` (an instance of
/// Vulkan 1.2 caps every device at 1.2), as `(file, set, version)`.
fn recommendations(instance: ApiVersion) -> Vec<(std::path::PathBuf, EnableSet, ApiVersion)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let Ok(mut description) = DeviceDescription::from_khronos_json(&text, None) else {
            continue;
        };
        description.api_version = description.api_version.min(instance);
        let set = description.recommended_enables(&DevicePolicy::default());
        out.push((path, set, description.api_version));
    }
    assert!(out.len() > 3, "read {} committed profiles", out.len());
    out
}

/// Every feature and extension a committed profile's recommendation enables has a member and a
/// name, and the whole set makes a create info, at an instance of Vulkan 1.2 and at the
/// profile's own version (most committed profiles are 1.3 or 1.4, and a 1.3 one may present a
/// promoted feature without its extension, as the profiles layer does): a device the runtime
/// creates enables all of it.
#[test]
fn every_recommended_enable_has_a_member_and_a_name() {
    for instance in [ApiVersion::V1_2, V1_4] {
        for (path, set, api) in recommendations(instance) {
            for feature in &set.features {
                assert!(
                    MEMBERS
                        .iter()
                        .chain(CORE_1_3)
                        .any(|member| member.feature == *feature),
                    "{path:?}: no member enables {feature:?}"
                );
            }
            let listed: BTreeSet<CString> = set
                .extensions
                .iter()
                .flat_map(|&extension| names_of(extension))
                .map(|name| name.to_owned())
                .collect();
            if let Err(error) = Enables::of(&set, api, &listed, &[]) {
                panic!("{path:?} at {api:?}: {error:?}");
            }
        }
    }
}

/// Every feature the enable rules can produce has a member: a device that lists every vocabulary
/// feature (`VkFeature::ALL`) and every extension exposing one, with what those require, recommends
/// at each version from 1.2 to 1.4 a set whose every feature has a member and which makes a create
/// info. So a feature the rules newly enable without a member here fails this test, not a device's
/// creation (the committed profiles' recommendations cover only the features their captures read).
#[test]
fn every_feature_the_rules_enable_has_a_member() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/roadmap-2022.json");
    let text = std::fs::read_to_string(path).unwrap();
    for api in [ApiVersion::V1_2, ApiVersion::V1_3, V1_4] {
        let mut extensions: BTreeSet<VkExtension> = VkFeature::ALL
            .iter()
            .flat_map(|feature| feature.exposure().extensions_at(ApiVersion::V1_2))
            .copied()
            .collect();
        let required: Vec<VkExtension> = extensions
            .iter()
            .flat_map(|extension| extension.requires(ApiVersion::V1_2))
            .collect();
        extensions.extend(required);
        let mut description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        description.api_version = api;
        description.features = VkFeature::ALL.iter().copied().collect();
        description.extensions = extensions;
        description.robust_buffer_access_update_after_bind = true;
        let set = description.recommended_enables(&DevicePolicy::default());
        for feature in &set.features {
            assert!(
                MEMBERS
                    .iter()
                    .chain(CORE_1_3)
                    .any(|member| member.feature == *feature),
                "{api:?}: the rules enable {feature:?}, which no member sets"
            );
        }
        let listed: BTreeSet<CString> = set
            .extensions
            .iter()
            .flat_map(|&extension| names_of(extension))
            .map(|name| name.to_owned())
            .collect();
        if let Err(error) = Enables::of(&set, api, &listed, &[]) {
            panic!("{api:?}: {error:?}");
        }
        assert!(
            set.features.len() > 100,
            "{api:?}: only {} enabled",
            set.features.len()
        );
    }
}

/// The type of each structure `enables`' create info chains, in chain order, raw (`ash` types
/// print nothing).
fn chained(enables: &mut Enables) -> Vec<i32> {
    enables.with_create_info(&[], |info| {
        let mut types = Vec::new();
        let mut next = info.p_next.cast::<vk::BaseInStructure>();
        while !next.is_null() {
            // SAFETY: every structure of the chain lives for the call, and each begins with
            // `sType` and `pNext`.
            let base = unsafe { &*next };
            types.push(base.s_type.as_raw());
            next = base.p_next;
        }
        types
    })
}

/// A device that presents a promoted feature without listing its extension (the profiles layer
/// at a 1.3 profile; any device of Vulkan 1.3 or later may): from the version that promoted it,
/// a member 1.3 promoted is set in `VkPhysicalDeviceVulkan13Features`, never beside its
/// extension structure, and a member 1.4 promoted in its own structure, core from 1.4; below
/// that version the feature is refused, as before. Extended dynamic state, core in 1.3 without a
/// member, sets nothing there.
#[test]
fn a_promoted_feature_without_its_extension_is_enabled_from_its_core_version() {
    let mut set = EnableSet::new();
    set.features.extend([
        VkFeature::RobustImageAccess,
        VkFeature::ShaderZeroInitializeWorkgroupMemory,
        VkFeature::Maintenance5,
        VkFeature::ExtendedDynamicState,
    ]);
    let none = BTreeSet::new();
    let mut enables = Enables::of(&set, V1_4, &none, &[]).unwrap();
    let types = chained(&mut enables);
    assert!(
        types.contains(&vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES.as_raw()),
        "{types:?}"
    );
    assert!(
        types.contains(&vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR.as_raw()),
        "{types:?}"
    );
    for absent in [
        vk::StructureType::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
        vk::StructureType::PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES,
        vk::StructureType::PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
    ] {
        let absent = absent.as_raw();
        assert!(!types.contains(&absent), "{absent} in {types:?}");
    }
    let v13 = enables.with_create_info(&[], |info| {
        let mut next = info.p_next.cast::<vk::BaseInStructure>();
        while !next.is_null() {
            // SAFETY: as in `chained`.
            let base = unsafe { &*next };
            if base.s_type == vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES {
                // SAFETY: a structure of this type is the 1.3 features'.
                let v13 = unsafe { *next.cast::<vk::PhysicalDeviceVulkan13Features<'static>>() };
                return Some((
                    v13.robust_image_access,
                    v13.shader_zero_initialize_workgroup_memory,
                    v13.synchronization2,
                ));
            }
            next = base.p_next;
        }
        None
    });
    assert_eq!(v13, Some((vk::TRUE, vk::TRUE, vk::FALSE)));
    // Below the version that promoted it, a feature without its extension is refused.
    let refused = |feature: VkFeature, api: ApiVersion| {
        let mut set = EnableSet::new();
        set.features.insert(feature);
        Enables::of(&set, api, &none, &[]).err()
    };
    let gap = |feature| Some(Error::NotImplemented(RuntimeGap::FeatureEnable(feature)));
    assert_eq!(
        refused(VkFeature::RobustImageAccess, ApiVersion::V1_2),
        gap(VkFeature::RobustImageAccess)
    );
    assert_eq!(
        refused(VkFeature::Maintenance5, ApiVersion::V1_3),
        gap(VkFeature::Maintenance5)
    );
    // And with it, in its extension structure.
    let mut set = EnableSet::new();
    set.features.insert(VkFeature::RobustImageAccess);
    set.extensions.insert(VkExtension::ExtImageRobustness);
    let listed = names_of(VkExtension::ExtImageRobustness)
        .map(|name| name.to_owned())
        .collect();
    let types = chained(&mut Enables::of(&set, ApiVersion::V1_2, &listed, &[]).unwrap());
    assert!(types.contains(&vk::StructureType::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES.as_raw()));
    assert!(!types.contains(&vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES.as_raw()));
}

/// Each vocabulary feature has one member, and `CORE_1_3` holds one for exactly the features
/// whose structure Vulkan 1.3 took over with its members (`Structure::promotion`).
#[test]
fn members_are_unique() {
    let features: BTreeSet<_> = MEMBERS.iter().map(|member| member.feature).collect();
    assert_eq!(features.len(), MEMBERS.len());
    let core: BTreeSet<_> = CORE_1_3.iter().map(|member| member.feature).collect();
    assert_eq!(core.len(), CORE_1_3.len());
    let promoted: BTreeSet<_> = MEMBERS
        .iter()
        .filter(|member| member.structure.promotion() == Some(Promotion::Members(ApiVersion::V1_3)))
        .map(|member| member.feature)
        .collect();
    assert_eq!(core, promoted);
}

/// The runtime's promotions (`Structure::promotion`) are the ones the description applies (the
/// registry's): at each version from 1.2 to 1.4, on a device that presents every feature the
/// runtime has a member for and lists no extension, the runtime enables every feature the
/// description recommends there, refuses every other the description gives a member, and sets
/// nothing for a feature the description keeps without one.
#[test]
fn promotions_are_the_descriptions() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/roadmap-2022.json");
    let text = std::fs::read_to_string(path).unwrap();
    let all: BTreeSet<VkFeature> = MEMBERS.iter().map(|member| member.feature).collect();
    let none = BTreeSet::new();
    let of = |features: &[VkFeature], api| {
        let mut set = EnableSet::new();
        set.features.extend(features.iter().copied());
        Enables::of(&set, api, &none, &[])
    };
    for api in [ApiVersion::V1_2, ApiVersion::V1_3, V1_4] {
        let mut description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        description.api_version = api;
        description.features = all.clone();
        description.extensions.clear();
        description.robust_buffer_access_update_after_bind = true;
        let recommended = description.recommended_enables(&DevicePolicy::default());
        assert!(recommended.extensions.is_empty(), "{api:?}");
        if let Err(error) = Enables::of(&recommended, api, &none, &[]) {
            panic!("{api:?}: {error:?}");
        }
        let memberless = description
            .with_enabled(&EnableSet::new())
            .unwrap()
            .features;
        let unset = chained(&mut of(&[], api).unwrap());
        for &feature in &all {
            if memberless.contains(&feature) {
                let chain = chained(&mut of(&[feature], api).unwrap());
                assert_eq!(chain, unset, "{feature:?} at {api:?}");
            } else if !recommended.features.contains(&feature) {
                assert_eq!(
                    of(&[feature], api).err(),
                    Some(Error::NotImplemented(RuntimeGap::FeatureEnable(feature))),
                    "{feature:?} at {api:?}"
                );
            }
        }
        if api >= ApiVersion::V1_3 {
            for feature in [
                VkFeature::ExtendedDynamicState,
                VkFeature::ExtendedDynamicState2,
            ] {
                assert!(
                    memberless.contains(&feature),
                    "the scan reads nothing at {api:?}: {feature:?}"
                );
            }
        }
        assert!(
            recommended.features.len() > 60,
            "{api:?}: only {} recommended",
            recommended.features.len()
        );
    }
}

/// A set the runtime cannot enable exactly is refused, never trimmed.
#[test]
fn an_enable_the_device_cannot_take_is_refused() {
    let member = MEMBERS
        .iter()
        .find(|member| !member.structure.extensions().is_empty())
        .unwrap();
    let mut set = EnableSet::new();
    set.features.insert(member.feature);
    assert_eq!(
        Enables::of(&set, ApiVersion::V1_2, &BTreeSet::new(), &[]).err(),
        Some(Error::NotImplemented(RuntimeGap::FeatureEnable(
            member.feature
        ))),
        "a feature without the extension exposing its structure"
    );
    let extension = member.structure.extensions()[0];
    let mut set = EnableSet::new();
    set.extensions.insert(extension);
    assert_eq!(
        Enables::of(&set, ApiVersion::V1_2, &BTreeSet::new(), &[]).err(),
        Some(Error::NotImplemented(RuntimeGap::ExtensionEnable(
            extension
        ))),
        "an extension the device does not list"
    );
}

/// The runtime's own `imageViewFormatReinterpretation`, enabled on a device whose portability
/// subset has it, is set in the subset's structure, chained into the create info.
#[test]
fn view_reinterpretation_is_enabled_in_the_portability_subset() {
    let mut set = EnableSet::new();
    set.extensions.insert(VkExtension::KhrPortabilitySubset);
    let listed = names_of(VkExtension::KhrPortabilitySubset)
        .map(|name| name.to_owned())
        .collect();
    let mut enables = Enables::of(&set, ApiVersion::V1_2, &listed, &[]).unwrap();
    let subset = |enables: &mut Enables| {
        enables.with_create_info(&[], |info| {
            let mut next = info.p_next.cast::<vk::BaseInStructure>();
            while !next.is_null() {
                // SAFETY: every structure of the chain lives for the call, and each begins with
                // `sType` and `pNext`.
                let base = unsafe { &*next };
                if base.s_type == vk::StructureType::PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
                {
                    // SAFETY: a structure of this type is the subset's.
                    let subset =
                        unsafe { &*next.cast::<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>() };
                    return Some(subset.image_view_format_reinterpretation);
                }
                next = base.p_next;
            }
            None
        })
    };
    assert_eq!(
        subset(&mut enables),
        None,
        "nothing asked for the subset's structure"
    );
    enables.enable_view_reinterpretation();
    assert_eq!(subset(&mut enables), Some(vk::TRUE));
}

/// The runtime's own `hostImageCopy` is set in its own structure, chained into the create info only
/// once enabled, with the extensions the device's version needs: `VK_EXT_host_image_copy` and the
/// two it requires below Vulkan 1.3, the extension alone at 1.3, and none from 1.4, which made it
/// core.
#[test]
fn host_image_copy_is_enabled_in_its_own_structure() {
    let expected: [(ApiVersion, &[&CStr]); 3] = [
        (
            ApiVersion::V1_2,
            &[
                c"VK_EXT_host_image_copy",
                c"VK_KHR_copy_commands2",
                c"VK_KHR_format_feature_flags2",
            ],
        ),
        (ApiVersion::V1_3, &[c"VK_EXT_host_image_copy"]),
        (V1_4, &[]),
    ];
    // The structure's member where chained, and the extension names.
    let read = |enables: &mut Enables| {
        enables.with_create_info(&[], |info| {
            let mut member = None;
            let mut next = info.p_next.cast::<vk::BaseInStructure>();
            while !next.is_null() {
                // SAFETY: every structure of the chain lives for the call, and each begins with
                // `sType` and `pNext`.
                let base = unsafe { &*next };
                if base.s_type
                    == <vk::PhysicalDeviceHostImageCopyFeaturesEXT as vk::TaggedStructure>::STRUCTURE_TYPE
                {
                    // SAFETY: a structure of this type is the host image copy features'.
                    let host =
                        unsafe { &*next.cast::<vk::PhysicalDeviceHostImageCopyFeaturesEXT>() };
                    member = Some(host.host_image_copy);
                }
                next = base.p_next;
            }
            let names: Vec<CString> = (0..info.enabled_extension_count as usize)
                // SAFETY: the create info's names are its count of live C strings.
                .map(|i| unsafe { CStr::from_ptr(*info.pp_enabled_extension_names.add(i)) })
                .map(CStr::to_owned)
                .collect();
            (member, names)
        })
    };
    for (api, names) in expected {
        assert_eq!(host_image_copy_names(api), names, "{api:?}");
        let mut enables = Enables::of(&EnableSet::new(), api, &BTreeSet::new(), &[]).unwrap();
        assert_eq!(read(&mut enables), (None, Vec::new()), "{api:?}");
        enables.enable_host_image_copy(host_image_copy_names(api));
        let names: Vec<CString> = names.iter().map(|&name| name.to_owned()).collect();
        assert_eq!(read(&mut enables), (Some(vk::TRUE), names), "{api:?}");
    }
}

/// The runtime's own `linearColorAttachment` is set in its own structure, chained into the create
/// info only once enabled, with `VK_NV_linear_color_attachment`.
#[test]
fn linear_color_attachment_is_enabled_in_its_own_structure() {
    let read = |enables: &mut Enables| {
        enables.with_create_info(&[], |info| {
            let mut member = None;
            let mut next = info.p_next.cast::<vk::BaseInStructure>();
            while !next.is_null() {
                // SAFETY: every structure of the chain lives for the call, and each begins with
                // `sType` and `pNext`.
                let base = unsafe { &*next };
                if base.s_type
                    == <vk::PhysicalDeviceLinearColorAttachmentFeaturesNV as vk::TaggedStructure>::STRUCTURE_TYPE
                {
                    // SAFETY: a structure of this type is the linear color attachment features'.
                    let linear = unsafe {
                        &*next.cast::<vk::PhysicalDeviceLinearColorAttachmentFeaturesNV>()
                    };
                    member = Some(linear.linear_color_attachment);
                }
                next = base.p_next;
            }
            let names: Vec<CString> = (0..info.enabled_extension_count as usize)
                // SAFETY: the create info's names are its count of live C strings.
                .map(|i| unsafe { CStr::from_ptr(*info.pp_enabled_extension_names.add(i)) })
                .map(CStr::to_owned)
                .collect();
            (member, names)
        })
    };
    let api = ApiVersion::V1_3;
    let mut enables = Enables::of(&EnableSet::new(), api, &BTreeSet::new(), &[]).unwrap();
    assert_eq!(read(&mut enables), (None, Vec::new()));
    enables.enable_linear_color_attachment(linear_color_attachment_names(api));
    let names = vec![c"VK_NV_linear_color_attachment".to_owned()];
    assert_eq!(read(&mut enables), (Some(vk::TRUE), names));
    // Below Vulkan 1.3, with the extension it requires, once beside host image copy's.
    let api = ApiVersion::V1_2;
    let mut enables = Enables::of(&EnableSet::new(), api, &BTreeSet::new(), &[]).unwrap();
    enables.enable_host_image_copy(host_image_copy_names(api));
    enables.enable_linear_color_attachment(linear_color_attachment_names(api));
    let names: Vec<CString> = [
        c"VK_EXT_host_image_copy",
        c"VK_KHR_copy_commands2",
        c"VK_KHR_format_feature_flags2",
        c"VK_NV_linear_color_attachment",
    ]
    .iter()
    .map(|&name| name.to_owned())
    .collect();
    assert_eq!(read(&mut enables), (Some(vk::TRUE), names));
}

/// A caller's extension whose feature structure the runtime knows (`VK_KHR_`/`VK_EXT_swapchain_
/// maintenance1`, `VK_KHR_`/`VK_EXT_present_mode_fifo_latest_ready`) takes its structure, once
/// for both names, chained with every feature set where the device has them; a device without
/// one is refused, naming the extension as listed and the feature; another extension takes none.
#[test]
fn a_listed_extension_s_features_are_enabled_or_the_device_refused() {
    use crate::enables::listed_structures;
    use crate::error::ExtensionFeature;
    let names =
        |names: &[&CStr]| -> Vec<CString> { names.iter().map(|&name| name.to_owned()).collect() };
    let swapchain = 1_000_275_000;
    let fifo = 1_000_361_000;
    let listed = listed_structures(&names(&[
        c"VK_KHR_swapchain",
        c"VK_EXT_swapchain_maintenance1",
        c"VK_KHR_swapchain_maintenance1",
        c"VK_EXT_present_mode_fifo_latest_ready",
    ]));
    assert_eq!(
        listed.len(),
        2,
        "one structure per extension, whatever its names"
    );
    assert_eq!(listed[0].0, c"VK_EXT_swapchain_maintenance1");
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    let before = chained(&mut enables);
    enables.enable_listed(&listed, |_| true).unwrap();
    let after = chained(&mut enables);
    assert!(after.contains(&swapchain) && after.contains(&fifo));
    assert_eq!(after.len(), before.len() + 2);
    // Each structure's member is set.
    let members = enables.with_create_info(&[], |info| {
        let mut set = Vec::new();
        let mut next = info.p_next.cast::<vk::BaseInStructure>();
        while !next.is_null() {
            // SAFETY: every structure of the chain lives for the call, and each begins with
            // `sType` and `pNext`.
            let base = unsafe { &*next };
            if [swapchain, fifo].contains(&base.s_type.as_raw()) {
                // SAFETY: both structures hold one `VkBool32` member after `pNext`.
                let member = unsafe { *next.cast::<u8>().add(16).cast::<vk::Bool32>() };
                set.push((base.s_type.as_raw(), member));
            }
            next = base.p_next;
        }
        set
    });
    assert_eq!(members.len(), 2);
    assert!(members.iter().all(|&(_, member)| member == vk::TRUE));
    // A device without the feature: refused, naming the extension as listed.
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    assert_eq!(
        enables.enable_listed(&listed, |feature| feature != "presentModeFifoLatestReady"),
        Err(Error::ExtensionFeature(ExtensionFeature {
            extension: "VK_EXT_present_mode_fifo_latest_ready",
            feature: "presentModeFifoLatestReady",
        }))
    );
    // Another extension takes no structure.
    assert!(listed_structures(&names(&[c"VK_KHR_swapchain"])).is_empty());
}

/// An optional extension (`DeviceOptions::optional_device_extensions`) whose feature structure
/// the runtime knows is enabled, with its structure and every feature set, only where the device
/// has every feature; one the device lacks a feature of is left out, never refused; one of no
/// known structure is enabled as it is.
#[test]
fn an_optional_extension_is_enabled_only_with_its_features() {
    let names =
        |names: &[&CStr]| -> Vec<CString> { names.iter().map(|&name| name.to_owned()).collect() };
    let swapchain = 1_000_275_000;
    let fifo = 1_000_361_000;
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    let before = chained(&mut enables);
    enables.enable_where_featured(
        &names(&[
            c"VK_KHR_swapchain",
            c"VK_EXT_swapchain_maintenance1",
            c"VK_EXT_present_mode_fifo_latest_ready",
        ]),
        |feature| feature != "presentModeFifoLatestReady",
    );
    let after = chained(&mut enables);
    assert!(after.contains(&swapchain) && !after.contains(&fifo));
    assert_eq!(after.len(), before.len() + 1);
    let enabled: Vec<&CStr> = enables.names().collect();
    assert_eq!(
        enabled,
        [c"VK_KHR_swapchain", c"VK_EXT_swapchain_maintenance1"]
    );
}

/// The video extensions whose feature structures the runtime knows (`VK_KHR_video_maintenance1`,
/// `VK_KHR_video_maintenance2`, `VK_KHR_unified_image_layouts`) take their structures with every
/// feature the device has set; `unifiedImageLayoutsVideo`, which Vulkan requires of no device
/// listing the extension, stays disabled where the device lacks it, without a refusal, and a
/// device without `unifiedImageLayouts` is refused (listed) or leaves the extension out
/// (optional).
#[test]
fn video_extensions_take_their_features_the_optional_one_where_the_device_has_it() {
    use crate::enables::listed_structures;
    use crate::error::ExtensionFeature;
    let names =
        |names: &[&CStr]| -> Vec<CString> { names.iter().map(|&name| name.to_owned()).collect() };
    let (maintenance1, maintenance2, unified) = (1_000_515_000, 1_000_586_000, 1_000_527_000);
    // The `VkBool32` members after `sType` and `pNext` of each chained video structure.
    let members = |enables: &mut Enables| {
        enables.with_create_info(&[], |info| {
            let mut set = Vec::new();
            let mut next = info.p_next.cast::<vk::BaseInStructure>();
            while !next.is_null() {
                // SAFETY: every structure of the chain lives for the call, and each begins with
                // `sType` and `pNext`.
                let base = unsafe { &*next };
                let raw = base.s_type.as_raw();
                let count = match raw {
                    _ if raw == unified => 2,
                    _ if raw == maintenance1 || raw == maintenance2 => 1,
                    _ => 0,
                };
                // SAFETY: each such structure holds `count` `VkBool32` members after `pNext`.
                let bools: Vec<vk::Bool32> = (0..count)
                    .map(|at| unsafe { *next.cast::<u8>().add(16 + 4 * at).cast::<vk::Bool32>() })
                    .collect();
                if count > 0 {
                    set.push((raw, bools));
                }
                next = base.p_next;
            }
            set.sort();
            set
        })
    };
    let listed = listed_structures(&names(&[
        c"VK_KHR_video_queue",
        c"VK_KHR_video_maintenance1",
        c"VK_KHR_video_maintenance2",
        c"VK_KHR_unified_image_layouts",
    ]));
    assert_eq!(listed.len(), 3, "the video queue takes no structure");
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    enables.enable_listed(&listed, |_| true).unwrap();
    assert_eq!(
        members(&mut enables),
        [
            (maintenance1, vec![vk::TRUE]),
            (unified, vec![vk::TRUE, vk::TRUE]),
            (maintenance2, vec![vk::TRUE]),
        ]
    );
    // Without `unifiedImageLayoutsVideo`: the extension stays, its video member disabled.
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    enables
        .enable_listed(&listed, |feature| feature != "unifiedImageLayoutsVideo")
        .unwrap();
    assert!(members(&mut enables).contains(&(unified, vec![vk::TRUE, vk::FALSE])));
    // Without `unifiedImageLayouts`: refused when listed, left out when optional.
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    assert_eq!(
        enables.enable_listed(&listed, |feature| feature != "unifiedImageLayouts"),
        Err(Error::ExtensionFeature(ExtensionFeature {
            extension: "VK_KHR_unified_image_layouts",
            feature: "unifiedImageLayouts",
        }))
    );
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    enables.enable_where_featured(
        &names(&[
            c"VK_KHR_video_maintenance1",
            c"VK_KHR_unified_image_layouts",
        ]),
        |feature| !feature.starts_with("unifiedImageLayouts"),
    );
    assert_eq!(members(&mut enables), [(maintenance1, vec![vk::TRUE])]);
    let enabled: Vec<&CStr> = enables.names().collect();
    assert_eq!(enabled, [c"VK_KHR_video_maintenance1"]);
    // An optional one whose video member alone is missing is enabled.
    let mut enables =
        Enables::of(&EnableSet::new(), ApiVersion::V1_3, &BTreeSet::new(), &[]).unwrap();
    enables.enable_where_featured(&names(&[c"VK_KHR_unified_image_layouts"]), |feature| {
        feature != "unifiedImageLayoutsVideo"
    });
    assert_eq!(
        members(&mut enables),
        [(unified, vec![vk::TRUE, vk::FALSE])]
    );
}
