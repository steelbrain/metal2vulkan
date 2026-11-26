use super::*;
use crate::acceptance::{derived_profile, opened_description};
use crate::coverage::ROBUSTNESS_FEATURES;
use crate::engine::{committed_profiles, load_description};
use metal2vulkan::device::ApiVersion;
use metal2vulkan::device::{Device, DevicePolicy};
use std::collections::BTreeMap;

/// What a committed profile states that the engine's vocabulary cannot carry, by the names the
/// Vulkan registry uses: a `DeviceDescription` holds only the vocabulary, so a member with no
/// [`VkFeature`] ([`Member::registry`]) and an extension name with no [`VkExtension`] are read
/// from the profile's own capture here. Every feature member of every capability block, each
/// supported where any block states it so, and every extension name.
struct Capture {
    members: BTreeMap<String, bool>,
    extensions: BTreeSet<String>,
}

impl Capture {
    /// Committed profile `stem`'s capture.
    fn of(stem: &str) -> Capture {
        let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let blocks = json["capabilities"]
            .as_object()
            .unwrap_or_else(|| panic!("{}: no capability block", path.display()));
        assert!(!blocks.is_empty(), "{}: no capability", path.display());
        let mut members: BTreeMap<String, bool> = BTreeMap::new();
        let mut extensions = BTreeSet::new();
        for block in blocks.values() {
            if let Some(names) = block["extensions"].as_object() {
                extensions.extend(names.keys().cloned());
            }
            for structure in block["features"]
                .as_object()
                .into_iter()
                .flat_map(|features| features.values())
            {
                for (name, value) in structure.as_object().into_iter().flatten() {
                    if let Some(stated) = value.as_bool() {
                        let supported = members.entry(name.clone()).or_default();
                        *supported = *supported || stated;
                    }
                }
            }
        }
        Capture {
            members,
            extensions,
        }
    }

    /// Whether the capture states the member the registry calls `name` supported. A member no
    /// block states is not supported: a capture holds every member of every structure it read, so
    /// an absent one is a structure the device does not expose (the requirement documents,
    /// `roadmap-2022` and `vk12-floor`, state almost none).
    fn states(&self, name: &str) -> bool {
        self.members.get(name).copied().unwrap_or(false)
    }

    /// Whether the capture lists the device extension `name`.
    fn lists(&self, name: &CStr) -> bool {
        self.extensions
            .contains(name.to_str().expect("every extension name is UTF-8"))
    }
}

/// What a device described by `description` supports, as a chain: every member the description
/// lists, and every member the engine's vocabulary does not name as `capture`, the device's own,
/// states it. Claiming the latter supported would model a device no profile describes:
/// `moltenvk-m3` states every line-rasterization member false, and `swiftshader` states
/// `textureCompressionASTC_HDR` false.
fn supported_by(description: &DeviceDescription, capture: &Capture) -> FeatureChain {
    let mut chain = FeatureChain::default();
    for member in MEMBERS {
        let supported = match member.feature {
            Some(feature) => description.features.contains(&feature),
            None => capture.states(registry_name(member)),
        };
        if supported {
            member.set(&mut chain);
        }
    }
    chain
}

/// The registry name of a member the engine's vocabulary does not name
/// (`a_member_the_vocabulary_does_not_name_carries_its_registry_name` holds every such row to
/// having one).
fn registry_name(member: &Member) -> &'static str {
    member.registry.unwrap_or_else(|| {
        panic!(
            "{:?}.{} has no feature and no registry name",
            member.structure, member.name
        )
    })
}

/// A device the committed profile `stem` describes, as a context at `version` models it: what it
/// supports ([`supported_by`]) and what it lists ([`listed_at`]), both read from the profile's own
/// capture where the vocabulary cannot carry them.
fn modelled(
    stem: &str,
    description: &DeviceDescription,
    version: ApiVersion,
) -> (FeatureChain, Available) {
    let capture = Capture::of(stem);
    (
        supported_by(description, &capture),
        listed_at(version, description, &capture),
    )
}

/// `description` as a context's query reads it: at the instance's API 1.2, so a feature only a
/// later core version exposes (without the extension listed) is not reported.
fn at_1_2(mut description: DeviceDescription) -> DeviceDescription {
    let api = ApiVersion::V1_2;
    description.api_version = api;
    let extensions = description.extensions.clone();
    description
        .features
        .retain(|&feature| feature.exposure().exposed(api, &extensions));
    description
}

/// Every committed profile's context: at API 1.2 as a device context queries it, and at the
/// profile's own version as a restricted context opens it. Each is named `<stem> <kind>` and given
/// with its enables and what its device supports.
fn contexts() -> Vec<(String, Enables, FeatureChain)> {
    use super::super::profiles::ProfileRestriction;
    let mut out = Vec::new();
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        let version = restriction.context_version();
        let contexts = [
            (
                "device",
                at_1_2(load_description(&stem).unwrap()),
                ApiVersion::V1_2,
            ),
            ("restricted", restriction.read_at(version), version),
        ];
        for (kind, description, version) in contexts {
            let (supported, available) = modelled(&stem, &description, version);
            let enables = engine_list(&description, &supported, &available)
                .unwrap_or_else(|error| panic!("{stem} {kind}: {error}"));
            out.push((format!("{stem} {kind}"), enables, supported));
        }
    }
    out
}

/// The extensions a device described by `description` lists, as a context at `version` sees them:
/// every name whose vocabulary extension the description reports, and every name the vocabulary
/// does not carry that `capture`, the device's own, lists (a `DeviceDescription` keeps only the
/// vocabulary, so those two names are read there).
fn listed_at(version: ApiVersion, description: &DeviceDescription, capture: &Capture) -> Available {
    Available::at(
        version,
        EXTENSIONS
            .iter()
            .filter(|extension| match extension.vocabulary {
                Some(name) => description.extensions.contains(&name),
                None => capture.lists(extension.name),
            })
            .map(|extension| extension.name.to_owned()),
    )
}

/// Every member the engine's vocabulary does not name carries its registry name, and no member it
/// names carries one. That name is the only way a model of a device can read what the device
/// states about the member, so a `None` row without one is a member nothing can report
/// unsupported. Every name is one some committed capture states, so a misspelling cannot read as
/// an unsupported member on every device.
#[test]
fn a_member_the_vocabulary_does_not_name_carries_its_registry_name() {
    let mut unnamed = 0;
    let mut names = BTreeSet::new();
    for member in MEMBERS {
        match (member.feature, member.registry) {
            (Some(feature), Some(name)) => {
                panic!("{feature:?} is in the vocabulary and carries the registry name {name:?}")
            }
            (Some(_), None) => {}
            (None, None) => panic!(
                "{:?}.{} has no feature and no registry name",
                member.structure, member.name
            ),
            (None, Some(name)) => {
                assert!(names.insert(name), "{name} is two members' registry name");
                unnamed += 1;
            }
        }
    }
    assert_eq!(unnamed, 12, "members the vocabulary does not name");
    // Every one of them is a name some committed capture states: a misspelled name would read as
    // a device that does not support the member, on every device.
    let captures: Vec<Capture> = committed_profiles()
        .unwrap()
        .iter()
        .map(|stem| Capture::of(stem))
        .collect();
    for name in names {
        assert!(
            captures
                .iter()
                .any(|capture| capture.members.contains_key(name)),
            "no committed profile's capture states {name:?}: the registry spells it otherwise, so \
             every model reads the member as unsupported"
        );
    }
}

/// No context enables a member its device does not report: the invariant the enable lists keep on
/// a real device, where `supported` is the driver's own `VkPhysicalDeviceFeatures2`
/// ([`FeatureChain::query`]), held here over every committed profile and both kinds of context.
/// `vkCreateDevice` fails with `VK_ERROR_FEATURE_NOT_PRESENT` over a feature it is
/// asked for and the device does not support, so a context whose chain sets a member the queried
/// chain does not report opens no device at all.
#[test]
fn no_context_enables_a_member_the_device_does_not_report() {
    let mut checked = 0;
    for (name, enables, supported) in contexts() {
        for member in MEMBERS {
            if member.is_set(&enables.chain) {
                assert!(
                    member.is_set(&supported),
                    "{name}: the context enables {:?}.{}, which the device does not report",
                    member.structure,
                    member.name
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 1000, "{checked} enabled members checked");
}

/// No committed profile's context, at API 1.2 or restricted at its profile's version, enables a
/// feature or an extension the runtime never enables (`metal2vulkan_runtime::floor::enable_set`):
/// every member it sets is a feature of the runtime's set (or one the runtime's version has as core
/// with nothing to enable), none is a member the engine's vocabulary does not name, and every
/// extension it enables is one of the set's or one exposing a feature it enables below that
/// feature's core version. Ray queries, acceleration structures and deferred host operations,
/// which ten profiles list, are among what no context enables.
#[test]
fn no_context_enables_what_the_runtime_never_enables() {
    let runtime = metal2vulkan_runtime::floor::enable_set();
    let version = ApiVersion::from_packed(metal2vulkan_runtime::floor::VERSION);
    let mut listing_ray_queries = 0;
    let contexts = contexts();
    for (name, enables, supported) in &contexts {
        for member in MEMBERS.iter().filter(|member| enabled(enables, member)) {
            let feature = member.feature.unwrap_or_else(|| {
                panic!(
                    "{name}: enables {:?}.{}, which the vocabulary does not name",
                    member.structure, member.name
                )
            });
            assert!(
                runtime.features.contains(&feature) || !feature.exposure().has_member(version),
                "{name}: enables {feature:?}, which the runtime never enables"
            );
        }
        let set = enabled_set(enables);
        for &extension in &enables.extensions {
            let vocabulary = EXTENSIONS
                .iter()
                .find(|named| named.name == extension)
                .and_then(|named| named.vocabulary)
                .unwrap_or_else(|| panic!("{name}: enables {extension:?}, outside the vocabulary"));
            let exposes = (set.features.iter()).any(|feature| {
                (feature.exposure().extensions_at(enables.version)).contains(&vocabulary)
            });
            assert!(
                runtime.extensions.contains(&vocabulary) || exposes,
                "{name}: enables {vocabulary:?}, which the runtime never enables"
            );
        }
        for never in [
            c"VK_KHR_ray_query",
            c"VK_KHR_acceleration_structure",
            c"VK_KHR_deferred_host_operations",
        ] {
            assert!(!enables.extensions.contains(never), "{name}: {never:?}");
        }
        if member_of(F::RayQuery).is_some_and(|member| member.is_set(supported)) {
            listing_ray_queries += 1;
        }
    }
    assert!(contexts.len() >= 30, "{} contexts", contexts.len());
    assert!(
        listing_ray_queries > 0,
        "no context's device has ray queries"
    );
}

#[test]
fn every_member_reads_and_writes_its_own_bit() {
    for (index, member) in MEMBERS.iter().enumerate() {
        let mut chain = FeatureChain::default();
        member.set(&mut chain);
        for (other_index, other) in MEMBERS.iter().enumerate() {
            assert_eq!(
                other.is_set(&chain),
                index == other_index,
                "{:?} / {:?}",
                member.feature,
                other.feature
            );
        }
    }
}

/// One member per name in each vocabulary, one name table entry per extension name, and every
/// feature of the product's vocabulary is one the table can enable.
#[test]
fn every_feature_has_one_member_and_every_name_one_entry() {
    let mut named = BTreeSet::new();
    for feature in MEMBERS.iter().filter_map(|member| member.feature) {
        assert!(named.insert(feature), "{feature:?} is listed twice");
    }
    let mut fields = BTreeSet::new();
    for member in MEMBERS {
        assert!(
            fields.insert((member.structure, member.name)),
            "{:?}.{} is listed twice",
            member.structure,
            member.name
        );
    }
    let mut names = BTreeSet::new();
    for extension in EXTENSIONS {
        assert!(names.insert(extension.name), "{:?} twice", extension.name);
    }
    // Every structure's exposing extensions and requirements are names the table carries.
    for structure in Structure::EXTENSIONS {
        for name in structure.extensions().iter().chain(structure.requires()) {
            assert!(names.contains(name), "{structure:?}: {name:?}");
        }
    }
    for feature in VkFeature::ALL {
        assert!(named.contains(feature), "{feature:?} has no member");
    }
}

/// A structure is chained, and its members enabled, only through an extension the device lists
/// together with what that extension requires; the first listed alternative name is the one
/// enabled.
#[test]
fn a_member_is_enabled_only_through_a_listed_extension() {
    let unmet = Available::at(
        ApiVersion::V1_2,
        [
            c"VK_KHR_maintenance5".to_owned(),
            c"VK_KHR_dynamic_rendering_local_read".to_owned(),
            c"VK_EXT_shader_atomic_float2".to_owned(),
            c"VK_EXT_border_color_swizzle".to_owned(),
        ],
    );
    assert_eq!(unmet.exposing(Structure::Maintenance5), None);
    assert_eq!(unmet.exposing(Structure::LocalRead), None);
    assert_eq!(unmet.exposing(Structure::AtomicFloat2), None);
    assert_eq!(unmet.exposing(Structure::BorderColorSwizzle), None);
    assert!(unmet.exposed().is_empty());

    let alternative = Available::at(
        ApiVersion::V1_2,
        [
            c"VK_NV_compute_shader_derivatives".to_owned(),
            c"VK_EXT_vertex_attribute_divisor".to_owned(),
            c"VK_KHR_vertex_attribute_divisor".to_owned(),
        ],
    );
    assert_eq!(
        alternative.exposing(Structure::Derivatives),
        Some(Some(c"VK_NV_compute_shader_derivatives"))
    );
    assert_eq!(
        alternative.exposing(Structure::Divisor),
        Some(Some(c"VK_KHR_vertex_attribute_divisor"))
    );
    assert_eq!(alternative.exposing(Structure::V12), Some(None));

    // `link` chains a structure only through a name already enabled, with its requirements.
    let mut enables = Enables::new(ApiVersion::V1_2);
    assert!(enables.link(Structure::V11));
    assert!(!enables.link(Structure::BorderColorSwizzle));
    enables.extensions.insert(c"VK_EXT_border_color_swizzle");
    assert!(!enables.link(Structure::BorderColorSwizzle));
    enables.extensions.insert(c"VK_EXT_custom_border_color");
    assert!(enables.link(Structure::BorderColorSwizzle));
    assert_eq!(
        enables.linked,
        BTreeSet::from([Structure::BorderColorSwizzle])
    );
}

/// On every committed profile (as a context's API-1.2 query reads it), the engine's context
/// enables exactly what the runtime enables of it (`engine::runtime_enables`), in the one closed
/// set `with_enabled` accepts and `Device::from_queries` derives from: its opened profile leaves
/// images to the device where the profile states image robustness. No
/// member the vocabulary does not name is enabled (`textureCompressionASTC_HDR` among them).
#[test]
fn the_list_enables_what_the_runtime_enables_of_the_description() {
    let stems = committed_profiles().unwrap();
    assert!(!stems.is_empty());
    for stem in stems {
        let description = at_1_2(load_description(&stem).unwrap());
        let (supported, available) = modelled(&stem, &description, ApiVersion::V1_2);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        assert!(enables.unmet_robustness.is_empty(), "{stem}");
        let set = enabled_set(&enables);
        assert_eq!(set, crate::engine::runtime_enables(&description), "{stem}");
        let opened = opened_description(description.clone(), &set)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        let profile = Device::from_queries(opened, DevicePolicy::default())
            .unwrap_or_else(|error| panic!("{stem}: {error}"))
            .profile()
            .clone();
        let states = |feature| description.features.contains(&feature);
        assert_eq!(
            profile.images().robust_image_access,
            states(F::RobustImageAccess) || states(F::RobustImageAccess2),
            "{stem}"
        );
        assert!(!enables.linked.contains(&Structure::AstcHdr), "{stem}");
        assert_eq!(
            enables.chain.astc_hdr.texture_compression_astc_hdr,
            vk::FALSE
        );
    }
}

/// The engine's context enables the robustness features the device's description states that
/// the runtime enables (the image ones), through their structures: `robustImageAccess2` through
/// `VK_EXT_robustness2`'s, `robustImageAccess` through `VK_EXT_image_robustness`'s, and neither
/// buffer feature, with `nullDescriptor` where the device states it. A stated feature the device
/// cannot enable is not dropped: no robustness feature is enabled then, and the context names it
/// unmet, which refuses every run on it.
#[test]
fn the_context_enables_the_robustness_the_description_states_or_names_it_unmet() {
    let enabled_on = |stem: &str, unreported: Option<VkFeature>| {
        let description = at_1_2(load_description(stem).unwrap());
        let (mut supported, available) = modelled(stem, &description, ApiVersion::V1_2);
        if let Some(feature) = unreported {
            *(member_of(feature).unwrap().write)(&mut supported) = vk::FALSE;
        }
        engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"))
    };
    let robustness = |enables: &Enables| {
        enabled_set(enables)
            .features
            .into_iter()
            .filter(|feature| ROBUSTNESS_FEATURES.contains(feature))
            .collect::<BTreeSet<_>>()
    };
    let images = [F::RobustImageAccess, F::RobustImageAccess2];
    let lavapipe = enabled_on("lavapipe", None);
    assert_eq!(robustness(&lavapipe), images.into());
    assert!(lavapipe.linked.contains(&Structure::Robustness2));
    assert!(lavapipe.linked.contains(&Structure::ImageRobustness));
    assert_eq!(lavapipe.chain.core.robust_buffer_access, vk::FALSE);
    assert_eq!(lavapipe.chain.robustness2.robust_buffer_access2, vk::FALSE);
    assert_eq!(lavapipe.chain.robustness2.robust_image_access2, vk::TRUE);
    assert_eq!(
        lavapipe.chain.image_robustness.robust_image_access,
        vk::TRUE
    );
    assert_eq!(lavapipe.chain.robustness2.null_descriptor, vk::TRUE);
    let moltenvk = enabled_on("moltenvk-m3", None);
    assert_eq!(robustness(&moltenvk), images.into());
    // The runtime floor read at API 1.2: `robustImageAccess` is core from 1.3 and the floor lists
    // no `VK_EXT_image_robustness`, so only `robustImageAccess2` serves images.
    let floor = enabled_on("runtime-floor", None);
    assert_eq!(robustness(&floor), [F::RobustImageAccess2].into());

    let unmet = enabled_on("lavapipe", Some(F::RobustImageAccess));
    assert_eq!(unmet.unmet_robustness, [F::RobustImageAccess].into());
    assert_eq!(robustness(&unmet), BTreeSet::new());
}

/// Where a device lists `VK_EXT_extended_dynamic_state`, the engine's context enables it and
/// `extendedDynamicState` through its structure, so the opened profile makes the extended states
/// dynamic and a render pipeline's requirement of them is met, as the runtime's version has it as
/// core; a device without the extension gets neither.
#[test]
fn the_list_enables_extended_dynamic_state_where_the_device_lists_it() {
    let lists = |stem: &str| {
        let description = at_1_2(load_description(stem).unwrap());
        let (supported, available) = modelled(stem, &description, ApiVersion::V1_2);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        (description, enables)
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let (description, enables) = lists(stem);
        assert!(description.has(F::ExtendedDynamicState), "{stem}");
        assert!(enables.linked.contains(&Structure::ExtendedDynamicState));
        assert!(enables
            .extensions
            .contains(c"VK_EXT_extended_dynamic_state"));
        assert_eq!(
            enables.chain.extended_dynamic_state.extended_dynamic_state,
            vk::TRUE
        );
        let set = enabled_set(&enables);
        assert!(set.features.contains(&F::ExtendedDynamicState), "{stem}");
        assert!(set
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState));
        let profile = derived_profile(opened_description(description, &set).unwrap()).unwrap();
        assert!(
            profile
                .enabled()
                .features
                .contains(&F::ExtendedDynamicState),
            "{stem}"
        );
    }
    let (description, enables) = lists("runtime-floor");
    assert!(!description.has(F::ExtendedDynamicState));
    assert!(!enabled_set(&enables)
        .features
        .contains(&F::ExtendedDynamicState));
    let profile =
        derived_profile(opened_description(description, &enabled_set(&enables)).unwrap()).unwrap();
    assert!(!profile
        .enabled()
        .features
        .contains(&F::ExtendedDynamicState));
}

/// Where a device lists `VK_EXT_extended_dynamic_state2`, the engine's context at API 1.2
/// enables it and `extendedDynamicState2` through its structure, so the opened profile makes a
/// vertex pipeline's topology and restart dynamic as the runtime's does; a device without the
/// extension gets neither.
#[test]
fn the_list_enables_extended_dynamic_state2_where_the_device_lists_it() {
    let lists = |stem: &str| {
        let description = at_1_2(load_description(stem).unwrap());
        let (supported, available) = modelled(stem, &description, ApiVersion::V1_2);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        (description, enables)
    };
    for stem in ["lavapipe", "moltenvk-m3", "amd-780m-radv"] {
        let (description, enables) = lists(stem);
        assert!(description.has(F::ExtendedDynamicState2), "{stem}");
        assert!(enables.linked.contains(&Structure::ExtendedDynamicState2));
        assert!(enables
            .extensions
            .contains(c"VK_EXT_extended_dynamic_state2"));
        assert_eq!(
            enables
                .chain
                .extended_dynamic_state2
                .extended_dynamic_state2,
            vk::TRUE
        );
        let set = enabled_set(&enables);
        assert!(set.features.contains(&F::ExtendedDynamicState2), "{stem}");
        assert!(set
            .extensions
            .contains(&VkExtension::ExtExtendedDynamicState2));
        let profile = derived_profile(opened_description(description, &set).unwrap()).unwrap();
        assert!(
            profile
                .enabled()
                .features
                .contains(&F::ExtendedDynamicState2),
            "{stem}"
        );
    }
    let (description, enables) = lists("runtime-floor");
    assert!(!description.has(F::ExtendedDynamicState2));
    assert!(!enabled_set(&enables)
        .features
        .contains(&F::ExtendedDynamicState2));
}

/// On every committed profile (as a context's API-1.2 query reads it), the list is a set the
/// device can create: `with_enabled` accepts it. It enables neither buffer robustness feature (the
/// runtime enables none), so VUID-VkDeviceCreateInfo-robustBufferAccess-10247 never applies, and
/// the texel buffers' update-after-bind features, which the runtime requires, are enabled wherever
/// the profile states them.
#[test]
fn the_list_enables_no_buffer_robustness_and_keeps_the_texel_buffers_update_after_bind() {
    for stem in committed_profiles().unwrap() {
        let description = at_1_2(load_description(&stem).unwrap());
        let (supported, available) = modelled(&stem, &description, ApiVersion::V1_2);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        let set = enabled_set(&enables);
        let robust: Vec<&Member> = MEMBERS
            .iter()
            .filter(|m| matches!(m.name, "robust_buffer_access" | "robust_buffer_access2"))
            .collect();
        assert_eq!(robust.len(), 2, "{stem}");
        for member in robust {
            assert!(!enabled(&enables, member), "{stem}: {}", member.name);
        }
        for feature in [
            F::DescriptorBindingUniformTexelBufferUpdateAfterBind,
            F::DescriptorBindingStorageTexelBufferUpdateAfterBind,
        ] {
            assert_eq!(
                set.features.contains(&feature),
                description.features.contains(&feature),
                "{stem}: {feature:?}"
            );
        }
        let opened = opened_description(description.clone(), &set)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        Device::from_queries(opened, DevicePolicy::default())
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
    }
}

/// A description and a device's own lists that disagree open no context: a recommended feature
/// the chain does not report, or a recommended extension the device lists no name of.
#[test]
fn a_disagreeing_description_is_refused() {
    let description = at_1_2(load_description("lavapipe").unwrap());
    let (supported, available) = modelled("lavapipe", &description, ApiVersion::V1_2);
    let mut unreported = supported;
    *(member_of(F::ShaderInt64).unwrap().write)(&mut unreported) = vk::FALSE;
    let error = engine_list(&description, &unreported, &available)
        .err()
        .unwrap();
    assert!(error.contains("ShaderInt64"), "{error}");
    let unlisted = Available::at(
        ApiVersion::V1_2,
        EXTENSIONS
            .iter()
            .filter(|named| named.vocabulary != Some(VkExtension::ExtRobustness2))
            .filter(|named| available.has(named.name))
            .map(|named| named.name.to_owned()),
    );
    let error = engine_list(&description, &supported, &unlisted)
        .err()
        .unwrap();
    assert!(error.contains("ExtRobustness2"), "{error}");
}

/// The executor's MoltenVK device (the Q1 refusals): its context enables subgroup size control
/// and the memory model, so the opened kernel stage can require full subgroups, and the opened
/// profile carries them; it enables no `fragmentShaderSampleInterlock`, which MoltenVK states and
/// the runtime never enables.
#[test]
fn the_moltenvk_context_opens_full_subgroups_and_the_memory_model() {
    let description = at_1_2(load_description("moltenvk-m3").unwrap());
    let (supported, available) = modelled("moltenvk-m3", &description, ApiVersion::V1_2);
    let enables = engine_list(&description, &supported, &available).unwrap();
    let set = enabled_set(&enables);
    for feature in [
        F::SubgroupSizeControl,
        F::ComputeFullSubgroups,
        F::VulkanMemoryModel,
        F::VulkanMemoryModelDeviceScope,
        F::Maintenance8,
    ] {
        assert!(set.features.contains(&feature), "{feature:?}");
    }
    assert!(description.has(F::FragmentShaderSampleInterlock));
    assert!(!set.features.contains(&F::FragmentShaderSampleInterlock));
    assert!(set
        .extensions
        .contains(&VkExtension::ExtSubgroupSizeControl));
    assert!(enables.linked.contains(&Structure::SubgroupSizeControl));
    let profile = derived_profile(opened_description(description, &set).unwrap()).unwrap();
    assert!(profile.subgroups().kernel.full_subgroups);
    assert!(profile.enabled().features.contains(&F::VulkanMemoryModel));
}

/// The opened context enables what the runtime enables of the device (lavapipe and MoltenVK both
/// list subgroup size control and have the memory model), under the validation layer without a
/// message, and its opened description is restricted to exactly what it enabled. None of its
/// stated robustness is unmet.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn the_opened_contexts_enable_their_lists() {
    use super::super::device::with_validated_context;
    let device = crate::gpu::TEST_DEVICE;
    let (new, messages, opened, unmet) = with_validated_context(device, |context| {
        Ok((
            context.enabled.clone(),
            context.take_messages(),
            context.opened_description.clone(),
            context.unmet_robustness.clone(),
        ))
    })
    .unwrap();
    assert!(messages.is_empty(), "{messages:?}");
    assert_eq!(unmet, BTreeSet::new());
    let opened = opened.unwrap();
    assert_eq!(opened.features, new.features);
    assert_eq!(opened.extensions, new.extensions);
    for feature in [
        F::SubgroupSizeControl,
        F::ComputeFullSubgroups,
        F::VulkanMemoryModel,
        F::VulkanMemoryModelDeviceScope,
    ] {
        assert!(new.features.contains(&feature), "{feature:?}");
    }
    let runtime = metal2vulkan_runtime::floor::enable_set();
    let version = ApiVersion::from_packed(metal2vulkan_runtime::floor::VERSION);
    for feature in &new.features {
        assert!(
            runtime.features.contains(feature) || !feature.exposure().has_member(version),
            "{feature:?}"
        );
    }
}

/// Where the device lists `VK_KHR_shader_fma` (moltenvk-m3), the context enables neither it nor
/// `shaderFmaFloat32` (the runtime never does), and the opened description states neither, so a
/// `float` `fma` compiles to the device's `Fma` as on the runtime.
#[test]
fn shader_fma_is_not_enabled_where_the_device_lists_it() {
    let description = at_1_2(load_description("moltenvk-m3").unwrap());
    assert!(description.extensions.contains(&VkExtension::KhrShaderFma));
    let (supported, available) = modelled("moltenvk-m3", &description, ApiVersion::V1_2);
    let enables = engine_list(&description, &supported, &available).unwrap();
    assert_eq!(enables.chain.shader_fma.shader_fma_float32, vk::FALSE);
    assert!(!enables.extensions.contains(c"VK_KHR_shader_fma"));
    let set = enabled_set(&enables);
    let opened = opened_description(description, &set).unwrap();
    assert!(!opened.features.contains(&F::ShaderFmaFloat32));
    assert!(!opened.extensions.contains(&VkExtension::KhrShaderFma));
}

/// No engine context enables `VK_EXT_depth_range_unrestricted`, which the runtime never enables,
/// whether the device lists it or not: the opened description does not state it, so a module
/// writing depth texels compiled for it records `RestrictedDepthRange`, as on the runtime.
#[test]
fn depth_range_unrestricted_is_not_enabled_where_the_device_lists_it() {
    use metal2vulkan::device::VkExtension as X;
    let name = c"VK_EXT_depth_range_unrestricted";
    for (stem, listed) in [
        ("amd-780m-radv", true),
        ("lavapipe", true),
        ("moltenvk-m3", false),
    ] {
        let description = at_1_2(load_description(stem).unwrap());
        assert_eq!(
            description
                .extensions
                .contains(&X::ExtDepthRangeUnrestricted),
            listed,
            "{stem}"
        );
        let (supported, available) = modelled(stem, &description, ApiVersion::V1_2);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        assert!(!enables.extensions.contains(name), "{stem}");
        let set = enabled_set(&enables);
        assert!(
            !set.extensions.contains(&X::ExtDepthRangeUnrestricted),
            "{stem}"
        );
        let opened = opened_description(description.clone(), &set)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        assert!(
            !opened.extensions.contains(&X::ExtDepthRangeUnrestricted),
            "{stem}"
        );
    }
}

/// Vulkan 1.4.
const V1_4: ApiVersion = ApiVersion { major: 1, minor: 4 };

/// Every structure the model chains: the base ones and every extension structure.
fn every_structure() -> impl Iterator<Item = Structure> {
    [Structure::Core, Structure::V11, Structure::V12]
        .into_iter()
        .chain(Structure::EXTENSIONS)
}

/// A context at Vulkan 1.2, every unrestricted context and every one restricted to a profile of
/// that version, exposes, chains, and enables exactly as before contexts had a version. Only the
/// base structures are core at 1.2, every extension requirement stands, and
/// [`Available::exposing`] and [`Enables::link`] give what the former rules (restated here) give
/// over extension lists this test builds: none, all, each structure's names alone, and each with
/// its requirements.
#[test]
fn a_context_at_vulkan_1_2_exposes_and_links_as_before() {
    fn former_exposing(
        names: &BTreeSet<&'static CStr>,
        structure: Structure,
    ) -> Option<Option<&'static CStr>> {
        let alternatives = structure.extensions();
        if alternatives.is_empty() {
            return Some(None);
        }
        if !structure.requires().iter().all(|name| names.contains(name)) {
            return None;
        }
        alternatives
            .iter()
            .copied()
            .find(|name| names.contains(name))
            .map(Some)
    }
    /// `(linked, extensions)` of a former `Enables`.
    type Former = (BTreeSet<Structure>, BTreeSet<&'static CStr>);
    fn former_link(state: &mut Former, structure: Structure) -> bool {
        let alternatives = structure.extensions();
        if alternatives.is_empty() {
            return true;
        }
        let exposed = alternatives.iter().any(|name| state.1.contains(name))
            && structure
                .requires()
                .iter()
                .all(|name| state.1.contains(name));
        if exposed {
            state.0.insert(structure);
        }
        exposed
    }

    for structure in every_structure() {
        assert_eq!(
            structure.core_at(ApiVersion::V1_2),
            structure.extensions().is_empty(),
            "{structure:?}"
        );
        assert_eq!(
            structure.requires_at(ApiVersion::V1_2).collect::<Vec<_>>(),
            structure.requires(),
            "{structure:?}"
        );
    }
    let every: BTreeSet<&'static CStr> = every_structure()
        .flat_map(|structure| structure.extensions().iter().chain(structure.requires()))
        .copied()
        .collect();
    let mut lists = vec![BTreeSet::new(), every];
    for structure in Structure::EXTENSIONS {
        let alone: BTreeSet<&'static CStr> = structure.extensions().iter().copied().collect();
        let mut required = alone.clone();
        required.extend(structure.requires().iter().copied());
        lists.extend([alone, required]);
    }
    for names in &lists {
        let available = Available::at(
            ApiVersion::V1_2,
            names.iter().map(|name| (*name).to_owned()),
        );
        let mut linking = Enables::new(ApiVersion::V1_2);
        linking.extensions = names.clone();
        let mut former_linked: Former = (BTreeSet::new(), names.clone());
        for structure in every_structure() {
            assert_eq!(
                available.exposing(structure),
                former_exposing(names, structure),
                "{structure:?} {names:?}"
            );
            assert_eq!(
                linking.link(structure),
                former_link(&mut former_linked, structure),
                "{structure:?} {names:?}"
            );
        }
        assert_eq!(
            (linking.linked, linking.extensions),
            former_linked,
            "{names:?}"
        );
    }
}

/// The structures a context chains as core from a later version are the ones the product reads as
/// core from it. Every member whose feature the product reads from a core structure, or one
/// promoted by 1.2, belongs to a base structure; every member of a structure a later version
/// promoted belongs to one core from that version ([`Structure::core_since`]); and a member only an
/// extension exposes, or one whose functionality a version makes core without a member (extended
/// dynamic state), belongs to a structure no version makes core. Each structure's requirements at
/// each version are the closure of what the product's `VkExtension::requires` names at that version
/// for each of its extensions (`VK_KHR_dynamic_rendering` core from 1.3).
#[test]
fn a_structure_is_core_from_the_version_the_product_reads_it_core() {
    use metal2vulkan::device::Exposure;
    for member in MEMBERS {
        let Some(feature) = member.feature else {
            continue;
        };
        let structure = member.structure;
        match feature.exposure() {
            Exposure::Core(_) => assert!(structure.is_base(), "{feature:?}"),
            Exposure::Promoted { version, .. } if version <= ApiVersion::V1_2 => {
                assert!(structure.is_base(), "{feature:?}")
            }
            Exposure::Promoted { version, .. } => {
                assert_eq!(structure.core_since(), Some(version), "{feature:?}")
            }
            Exposure::Extension(_) | Exposure::Implied { .. } => {
                assert!(!structure.is_base(), "{feature:?}");
                assert_eq!(structure.core_since(), None, "{feature:?}");
            }
            other => panic!("{feature:?}: an exposure this test does not model: {other:?}"),
        }
    }
    let name_of = |extension: VkExtension| {
        EXTENSIONS
            .iter()
            .find(|named| named.vocabulary == Some(extension))
            .map(|named| named.name)
            .unwrap_or_else(|| panic!("{extension:?} has a name"))
    };
    let mut checked = 0;
    for structure in Structure::EXTENSIONS {
        for version in [ApiVersion::V1_2, ApiVersion::V1_3, V1_4] {
            if structure.core_at(version) {
                continue;
            }
            let ours: BTreeSet<&CStr> = structure.requires_at(version).collect();
            for named in EXTENSIONS
                .iter()
                .filter(|named| structure.extensions().contains(&named.name))
            {
                let Some(extension) = named.vocabulary else {
                    continue;
                };
                let mut closure = BTreeSet::new();
                let mut open = extension.requires(version);
                while let Some(required) = open.pop() {
                    if closure.insert(name_of(required)) {
                        open.extend(required.requires(version));
                    }
                }
                assert_eq!(ours, closure, "{structure:?} {:?} {version:?}", named.name);
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

/// A context restricted to a committed profile runs at the profile's version
/// (`ProfileRestriction::context_version`) and translates for the profile read there. At that
/// version its engine list is what the runtime enables of the profile, the one closed set
/// `with_enabled` accepts and `Device::from_queries` derives from, as at 1.2
/// ([`the_list_enables_what_the_runtime_enables_of_the_description`]), and no structure core at
/// the version is chained through its extension.
#[test]
fn a_restricted_context_enables_what_the_runtime_enables_at_its_profiles_version() {
    use super::super::profiles::ProfileRestriction;
    for stem in committed_profiles().unwrap() {
        let restriction = ProfileRestriction::committed(&stem).unwrap();
        let version = restriction.context_version();
        assert_eq!(
            version,
            restriction.description.api_version.max(ApiVersion::V1_2),
            "{stem}"
        );
        let description = restriction.read_at(version);
        let (supported, available) = modelled(&stem, &description, version);
        let enables = engine_list(&description, &supported, &available)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        assert_eq!(enables.version, version, "{stem}");
        assert!(enables.unmet_robustness.is_empty(), "{stem}");
        let set = enabled_set(&enables);
        assert_eq!(set, crate::engine::runtime_enables(&description), "{stem}");
        let opened = opened_description(description.clone(), &set)
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        Device::from_queries(opened, DevicePolicy::default())
            .unwrap_or_else(|error| panic!("{stem}: {error}"));
        for structure in enables.linked.iter().filter(|s| s.core_at(version)) {
            assert!(
                structure
                    .extensions()
                    .iter()
                    .all(|name| !enables.extensions.contains(name)),
                "{stem} {structure:?}"
            );
        }
    }
}

/// A fake Vulkan instance for the cross-check of the executor's feature structures against the
/// product's ([`the_executor_reads_and_enables_each_promoted_feature_where_the_product_reads_it`]):
/// a `vkGetInstanceProcAddr` whose physical-device entry points answer for one device, of
/// [`fake::Device::api_version`] listing [`fake::Device::extensions`], and record the type of
/// every structure chained onto `vkGetPhysicalDeviceFeatures2`. No loader, ICD, or device.
mod fake {
    use ash::vk;
    use std::cell::RefCell;
    use std::ffi::{c_char, CStr, CString};

    #[derive(Default)]
    pub(super) struct Device {
        pub(super) api_version: u32,
        pub(super) extensions: Vec<CString>,
        pub(super) chained: Vec<vk::StructureType>,
    }

    thread_local! {
        pub(super) static DEVICE: RefCell<Device> = RefCell::new(Device::default());
    }

    unsafe extern "system" fn properties(
        _: vk::PhysicalDevice,
        out: *mut vk::PhysicalDeviceProperties,
    ) {
        let api_version = DEVICE.with(|device| device.borrow().api_version);
        // SAFETY: the caller hands a writable structure.
        unsafe {
            out.write(vk::PhysicalDeviceProperties {
                api_version,
                ..Default::default()
            })
        };
    }

    unsafe extern "system" fn properties2(
        _: vk::PhysicalDevice,
        out: *mut vk::PhysicalDeviceProperties2<'_>,
    ) {
        let api_version = DEVICE.with(|device| device.borrow().api_version);
        // SAFETY: the caller hands a writable structure; its chain is left as handed.
        unsafe { (*out).properties.api_version = api_version };
    }

    unsafe extern "system" fn features2(
        _: vk::PhysicalDevice,
        out: *mut vk::PhysicalDeviceFeatures2<'_>,
    ) {
        // SAFETY: every chained structure begins with `sType` and `pNext`.
        let mut next = unsafe { (*out).p_next }.cast::<vk::BaseOutStructure<'_>>();
        while !next.is_null() {
            // SAFETY: as above.
            let ty = unsafe { (*next).s_type };
            DEVICE.with(|device| device.borrow_mut().chained.push(ty));
            // SAFETY: as above.
            next = unsafe { (*next).p_next };
        }
    }

    unsafe extern "system" fn extensions(
        _: vk::PhysicalDevice,
        _: *const c_char,
        count: *mut u32,
        out: *mut vk::ExtensionProperties,
    ) -> vk::Result {
        DEVICE.with(|device| {
            let names = &device.borrow().extensions;
            if out.is_null() {
                // SAFETY: the caller hands a writable count.
                unsafe { *count = names.len() as u32 };
                return vk::Result::SUCCESS;
            }
            // SAFETY: as above, and `out` holds `*count` entries.
            let written = (unsafe { *count } as usize).min(names.len());
            for (i, name) in names.iter().take(written).enumerate() {
                let mut entry = vk::ExtensionProperties::default();
                for (dst, &src) in entry
                    .extension_name
                    .iter_mut()
                    .zip(name.to_bytes_with_nul())
                {
                    *dst = src as c_char;
                }
                // SAFETY: as above.
                unsafe { out.add(i).write(entry) };
            }
            // SAFETY: as above.
            unsafe { *count = written as u32 };
            if written < names.len() {
                vk::Result::INCOMPLETE
            } else {
                vk::Result::SUCCESS
            }
        })
    }

    unsafe extern "system" fn format_properties(
        _: vk::PhysicalDevice,
        _: vk::Format,
        out: *mut vk::FormatProperties,
    ) {
        // SAFETY: the caller hands a writable structure.
        unsafe { out.write(vk::FormatProperties::default()) };
    }

    unsafe extern "system" fn format_properties2(
        _: vk::PhysicalDevice,
        _: vk::Format,
        out: *mut vk::FormatProperties2<'_>,
    ) {
        // SAFETY: the caller hands a writable structure; its chain is left as handed.
        unsafe { (*out).format_properties = vk::FormatProperties::default() };
    }

    pub(super) unsafe extern "system" fn get_instance_proc_addr(
        _: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        // SAFETY (every arm): `name` is a NUL-terminated entry point name, and the loader casts
        // the pointer back to the named entry point's type, which is the returned function's.
        unsafe {
            match CStr::from_ptr(name).to_bytes() {
                b"vkGetPhysicalDeviceProperties" => Some(std::mem::transmute::<
                    vk::PFN_vkGetPhysicalDeviceProperties,
                    unsafe extern "system" fn(),
                >(properties)),
                b"vkGetPhysicalDeviceProperties2" => Some(std::mem::transmute::<
                    vk::PFN_vkGetPhysicalDeviceProperties2,
                    unsafe extern "system" fn(),
                >(properties2)),
                b"vkGetPhysicalDeviceFeatures2" => Some(std::mem::transmute::<
                    vk::PFN_vkGetPhysicalDeviceFeatures2,
                    unsafe extern "system" fn(),
                >(features2)),
                b"vkEnumerateDeviceExtensionProperties" => Some(std::mem::transmute::<
                    vk::PFN_vkEnumerateDeviceExtensionProperties,
                    unsafe extern "system" fn(),
                >(extensions)),
                b"vkGetPhysicalDeviceFormatProperties" => Some(std::mem::transmute::<
                    vk::PFN_vkGetPhysicalDeviceFormatProperties,
                    unsafe extern "system" fn(),
                >(format_properties)),
                b"vkGetPhysicalDeviceFormatProperties2" => {
                    Some(std::mem::transmute::<
                        vk::PFN_vkGetPhysicalDeviceFormatProperties2,
                        unsafe extern "system" fn(),
                    >(format_properties2))
                }
                _ => None,
            }
        }
    }
}

/// The structure type of each structure a version up to 1.4 promoted.
fn promoted_structure_type(structure: Structure) -> vk::StructureType {
    let chain = FeatureChain::default();
    match structure {
        Structure::SubgroupSizeControl => chain.subgroup_size_control.s_type,
        Structure::Demote => chain.demote.s_type,
        Structure::Terminate => chain.terminate.s_type,
        Structure::ZeroInitialize => chain.zero_initialize.s_type,
        Structure::Synchronization2 => chain.synchronization2.s_type,
        Structure::Maintenance4 => chain.maintenance4.s_type,
        Structure::ImageRobustness => chain.image_robustness.s_type,
        Structure::AstcHdr => chain.astc_hdr.s_type,
        Structure::SubgroupRotate => chain.subgroup_rotate.s_type,
        Structure::LocalRead => chain.local_read.s_type,
        Structure::Divisor => chain.divisor.s_type,
        Structure::Maintenance5 => chain.maintenance5.s_type,
        Structure::FloatControls2 => chain.float_controls2.s_type,
        Structure::PipelineRobustness => chain.pipeline_robustness.s_type,
        other => panic!("{other:?} is not a promoted structure"),
    }
}

/// For each feature the product reads whose structure a later version promoted, the executor's
/// context queries it, and enables it, through the structure the product's query reads it from, at
/// Vulkan 1.2, 1.3, and 1.4: the structure itself through its extension at 1.2,
/// `VkPhysicalDeviceVulkan13Features` for 1.3's from 1.3, and 1.4's own structures at 1.4. Both
/// queries run against one fake instance this test owns, which records the structures chained; the
/// enabled chain is read from the create info the executor hands its device's creation. And each
/// member `VkPhysicalDeviceVulkan13Features` carries reaches the device and comes back as itself
/// alone.
#[test]
fn the_executor_reads_and_enables_each_promoted_feature_where_the_product_reads_it() {
    use ash::vk::Handle;
    let promoted: Vec<Structure> = Structure::EXTENSIONS
        .into_iter()
        .filter(|structure| structure.core_since().is_some())
        .filter(|structure| {
            MEMBERS
                .iter()
                .any(|member| member.structure == *structure && member.feature.is_some())
        })
        .collect();
    // Raw structure types: ash's types print only with its `debug` feature.
    let v13 = FeatureChain::default().v13.s_type.as_raw();
    let watched: BTreeSet<i32> = promoted
        .iter()
        .map(|structure| promoted_structure_type(*structure).as_raw())
        .chain([v13])
        .collect();
    let names: Vec<CString> = every_structure()
        .flat_map(|structure| structure.extensions().iter().chain(structure.requires()))
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let static_fn = ash::StaticFn {
        get_instance_proc_addr: fake::get_instance_proc_addr,
    };
    // SAFETY: the fake table answers every entry point either query calls.
    let instance = unsafe { ash::Instance::load(&static_fn, vk::Instance::from_raw(1)) };
    let physical = vk::PhysicalDevice::from_raw(1);
    let chained = || -> BTreeSet<i32> {
        fake::DEVICE.with(|device| {
            device
                .borrow_mut()
                .chained
                .drain(..)
                .map(|ty| ty.as_raw())
                .filter(|ty| watched.contains(ty))
                .collect()
        })
    };
    for version in [ApiVersion::V1_2, ApiVersion::V1_3, V1_4] {
        fake::DEVICE.with(|device| {
            let mut device = device.borrow_mut();
            device.api_version = vk::make_api_version(0, version.major, version.minor, 0);
            device.extensions = names.clone();
            device.chained.clear();
        });
        // SAFETY: the fake device answers for any handle.
        let _ = unsafe { DeviceDescription::query(&instance, physical, version) };
        let product = chained();
        assert_eq!(
            product.contains(&v13),
            version >= ApiVersion::V1_3,
            "{version:?}"
        );
        let available = Available::at(version, names.iter().cloned());
        // SAFETY: as above.
        let supported = unsafe { FeatureChain::query(&instance, physical, &available) };
        assert_eq!(chained(), product, "the query at {version:?}");
        let linked: BTreeSet<Structure> = promoted.iter().copied().collect();
        let mut chain = supported;
        let enabled = chain.with_create_info(&[], &[], &linked, version, |info| {
            let mut types = BTreeSet::new();
            let mut next = info.p_next.cast::<vk::BaseInStructure<'_>>();
            while !next.is_null() {
                // SAFETY: every chained structure begins with `sType` and `pNext`.
                let ty = unsafe { (*next).s_type }.as_raw();
                if watched.contains(&ty) {
                    types.insert(ty);
                }
                // SAFETY: as above.
                next = unsafe { (*next).p_next };
            }
            types
        });
        assert_eq!(enabled, product, "the enabled chain at {version:?}");
    }

    let carried: Vec<&Member> = MEMBERS
        .iter()
        .filter(|member| member.structure.in_vulkan13())
        .collect();
    let all: BTreeSet<Structure> = carried.iter().map(|member| member.structure).collect();
    for member in &carried {
        let mut chain = FeatureChain::default();
        member.set(&mut chain);
        let mut back = FeatureChain {
            v13: chain.gather_v13(&all),
            ..FeatureChain::default()
        };
        back.spread_v13();
        for other in &carried {
            assert_eq!(
                other.is_set(&back),
                std::ptr::eq(*other, *member),
                "{:?} through {:?}",
                other.structure,
                member.structure
            );
        }
    }
}

/// The runtime probe's context enables each extension it measures exactly where the device lists
/// it, with `VK_KHR_image_format_list` beside the modifier extension only below Vulkan 1.2, where
/// that requirement is not core.
#[test]
fn the_runtime_probe_enables_what_it_measures_where_listed() {
    let host = ash::ext::external_memory_host::NAME;
    let modifier = ash::ext::image_drm_format_modifier::NAME;
    let format_list = ash::khr::image_format_list::NAME;
    let v1_1 = ApiVersion { major: 1, minor: 1 };
    for version in [v1_1, ApiVersion::V1_2, ApiVersion::V1_3] {
        for listed in [
            vec![],
            vec![host],
            vec![modifier],
            vec![modifier, format_list],
            vec![host, modifier, format_list],
        ] {
            let available = Available::at(version, listed.iter().map(|name| (*name).to_owned()));
            let mut enables = Enables::new(version);
            let before = enables.extensions.clone();
            let enabled = enable_runtime_probe(&mut enables, &available);
            let below_1_2 = version < ApiVersion::V1_2;
            let expect_modifier =
                listed.contains(&modifier) && (!below_1_2 || listed.contains(&format_list));
            assert_eq!(enabled.external_memory_host, listed.contains(&host));
            assert_eq!(enabled.drm_format_modifier, expect_modifier);
            let mut expected = before;
            if enabled.external_memory_host {
                expected.insert(host);
            }
            if expect_modifier {
                expected.insert(modifier);
                if below_1_2 {
                    expected.insert(format_list);
                }
            }
            assert_eq!(enables.extensions, expected, "{version:?} {listed:?}");
        }
    }
}

/// What a context enables, as `vkCreateDevice` receives it: the API version, the structures
/// chained, every member set in a chained structure (by its structure and field), the extension
/// names, and the robustness features left unmet. One line each, for [`OPENED_CHAINS`].
fn chain_lines(enables: &Enables) -> Vec<String> {
    let mut lines = vec![format!("  version {:?}", enables.version)];
    lines.extend(
        enables
            .linked
            .iter()
            .map(|structure| format!("  linked {structure:?}")),
    );
    lines.extend(
        MEMBERS
            .iter()
            .filter(|member| enabled(enables, member))
            .map(|member| format!("  member {:?}.{}", member.structure, member.name)),
    );
    lines.extend(
        enables
            .extensions
            .iter()
            .map(|name| format!("  extension {}", name.to_string_lossy())),
    );
    lines.extend(
        enables
            .unmet_robustness
            .iter()
            .map(|feature| format!("  unmet {feature:?}")),
    );
    lines
}

/// Every committed profile's context ([`contexts`]), named `<stem> <kind>`, with its
/// [`chain_lines`].
fn opened_chains() -> Vec<(String, Vec<String>)> {
    contexts()
        .into_iter()
        .map(|(name, enables, _)| (name, chain_lines(&enables)))
        .collect()
}

/// One line of [`OPENED_CHAINS`]: a context's name, the SHA-256 of its [`chain_lines`] (each
/// ending in a newline), and how many members and extensions it enables.
fn pinned_line(name: &str, lines: &[String]) -> String {
    let text = lines
        .iter()
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    let count = |prefix: &str| lines.iter().filter(|line| line.starts_with(prefix)).count();
    format!(
        "{name} {} members={} extensions={}",
        crate::hash::sha256_bytes(text.as_bytes()),
        count("  member "),
        count("  extension ")
    )
}

/// The pinned [`opened_chains`], written by [`bless_opened_chains`].
const OPENED_CHAINS: &str = "src/candidate/platform/enables/opened_chains.txt";

/// What every committed profile's context enables is pinned, chain by chain: a change to the enable
/// lists, or to the tables that name what they enable, that moves one bit of a context's chain or
/// one extension name fails here, printing that context's chain as it is now. The opened profile,
/// and so every row's translator fingerprint, derives from that chain. Bless with `cargo nextest
/// run -p metal2vulkan-validation --run-ignored ignored-only -E 'test(bless_opened_chains)'`.
#[test]
fn every_committed_profiles_context_enables_its_pinned_chain() {
    let path = metal2vulkan_devtools::validation_dir().join(OPENED_CHAINS);
    let pinned = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let pinned = pinned.lines().collect::<Vec<_>>();
    let chains = opened_chains();
    let moved = chains
        .iter()
        .filter(|(name, lines)| !pinned.contains(&pinned_line(name, lines).as_str()))
        .map(|(name, lines)| format!("{name}:\n{}", lines.join("\n")))
        .collect::<Vec<_>>();
    assert!(
        moved.is_empty(),
        "{OPENED_CHAINS}: {} contexts enable another chain than pinned (bless_opened_chains):\n{}",
        moved.len(),
        moved.join("\n")
    );
    assert_eq!(
        pinned.len(),
        chains.len(),
        "{OPENED_CHAINS}: contexts pinned and now"
    );
}

/// Rewrite [`OPENED_CHAINS`].
#[test]
#[ignore]
fn bless_opened_chains() {
    let path = metal2vulkan_devtools::validation_dir().join(OPENED_CHAINS);
    let text = opened_chains()
        .iter()
        .map(|(name, lines)| format!("{}\n", pinned_line(name, lines)))
        .collect::<String>();
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{path:?}: {e}"));
}
