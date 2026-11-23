//! Enable sets: `recommended_enables`, `with_enabled`, `required_enables`, and the exposure
//! invariant `Device::from_queries` checks.

use std::collections::BTreeSet;

use super::*;
use crate::device::enables::{self, Exposure, FEATURE_NEEDS};
use crate::device::requirements::{Requirements, StageCreateFlag};
use crate::error::Capability;

fn set<T: Ord + Copy>(items: &[T]) -> BTreeSet<T> {
    items.iter().copied().collect()
}

fn enables(features: &[VkFeature], extensions: &[VkExtension]) -> EnableSet {
    let mut out = EnableSet::new();
    out.features = set(features);
    out.extensions = set(extensions);
    out
}

fn requiring(features: &[VkFeature]) -> Requirements {
    Requirements {
        features: set(features),
        ..Requirements::default()
    }
}

fn description_error(result: Result<DeviceDescription, Error>) -> String {
    match result {
        Err(Error::InvalidDescriptor(DescriptorError::DeviceDescription { detail })) => detail,
        other => panic!("expected a DeviceDescription error, got {other:?}"),
    }
}

/// A 1.2 description, `runtime-floor` as a Vulkan 1.2 context reads it, exposing only what
/// `extensions` add to it.
fn floor_with(extensions: &[VkExtension], features: &[VkFeature]) -> DeviceDescription {
    let mut d = crate::pipeline::testing::description_at_1_2("runtime-floor", false)
        .expect("runtime-floor");
    assert_eq!(d.api_version, ApiVersion::V1_2);
    d.extensions.extend(extensions.iter().copied());
    d.features.extend(features.iter().copied());
    d
}

/// On every committed profile the recommendation lists only what the description lists, is closed
/// (`with_enabled` accepts it, `Device::from_queries` accepts the result), and is a fixed point:
/// the device made from it needs exactly it, as does the device made from the full description.
#[test]
fn recommendations_are_closed_fixed_points_on_every_committed_profile() {
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let rec = desc.recommended_enables();
        assert!(rec.features.is_subset(&desc.features), "{stem}");
        assert!(rec.extensions.is_subset(&desc.extensions), "{stem}");
        for &f in &rec.features {
            let x = f.exposure();
            assert!(
                x.exposed(desc.api_version, &rec.extensions),
                "{stem}: {f:?}"
            );
        }
        // Every listed feature with a member is recommended (the committed profiles list only
        // exposed features whose extensions' requirements are listed), except the features of the
        // extensions the core reads nothing of: `VK_EXT_depth_bias_control`, `VK_KHR_shader_fma`,
        // `VK_EXT_shader_float8` and `VK_EXT_shader_image_atomic_int64`.
        let unread = [
            VkFeature::DepthBiasControl,
            VkFeature::FloatRepresentation,
            VkFeature::DepthBiasExact,
            VkFeature::ShaderFmaFloat16,
            VkFeature::ShaderFmaFloat32,
            VkFeature::ShaderFloat8,
            VkFeature::ShaderImageInt64Atomics,
            VkFeature::RasterizationOrderColorAttachmentAccess,
            VkFeature::RasterizationOrderDepthAttachmentAccess,
            VkFeature::RasterizationOrderStencilAttachmentAccess,
        ];
        for e in [
            VkExtension::ExtDepthBiasControl,
            VkExtension::KhrShaderFma,
            VkExtension::ExtShaderFloat8,
            VkExtension::ExtShaderImageAtomicInt64,
            VkExtension::ExtRasterizationOrderAttachmentAccess,
        ] {
            assert!(!rec.extensions.contains(&e), "{stem}: {e:?}");
        }
        for &f in &desc.features {
            assert_eq!(
                rec.features.contains(&f),
                f.exposure().has_member(desc.api_version) && !unread.contains(&f),
                "{stem}: {f:?}"
            );
        }
        let restricted = desc
            .clone()
            .with_enabled(&rec)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let dev = Device::from_queries(restricted, DevicePolicy::RESIDENT_TABLES)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(dev.required_enables(), rec, "{stem}");
        let full = Device::from_queries(desc.clone(), DevicePolicy::RESIDENT_TABLES).unwrap();
        assert_eq!(full.required_enables(), rec, "{stem}");
        assert_eq!(
            rec.extensions.contains(&VkExtension::KhrPortabilitySubset),
            desc.extensions.contains(&VkExtension::KhrPortabilitySubset),
            "{stem}"
        );
    }
    let m = profile("moltenvk-m3").recommended_enables();
    assert!(m.extensions.contains(&VkExtension::KhrPortabilitySubset));
}

/// The recommendation names the exposing extension of promoted features below their version,
/// omits consumed extensions a version makes redundant, and omits refusal-only extensions.
#[test]
fn recommendations_follow_the_device_version() {
    let d = floor_with(
        &[
            VkExtension::KhrShaderTerminateInvocation,
            VkExtension::ExtShaderViewportIndexLayer,
            VkExtension::ExtMutableDescriptorType,
            VkExtension::ExtMeshShader,
        ],
        &[
            VkFeature::ShaderTerminateInvocation,
            VkFeature::TaskShader,
            VkFeature::MeshShader,
        ],
    );
    let rec = d.recommended_enables();
    assert!(rec.features.contains(&VkFeature::ShaderTerminateInvocation));
    assert!(rec
        .extensions
        .contains(&VkExtension::KhrShaderTerminateInvocation));
    // Promoted to 1.2; the core never enables refusal-only extensions.
    assert!(!rec
        .extensions
        .contains(&VkExtension::ExtShaderViewportIndexLayer));
    assert!(!rec
        .extensions
        .contains(&VkExtension::ExtMutableDescriptorType));
    // The core consumes `VK_EXT_mesh_shader`: listed, it is enabled with the features it exposes.
    assert!(rec.extensions.contains(&VkExtension::ExtMeshShader));
    assert!(rec.features.contains(&VkFeature::TaskShader));
    assert!(rec.features.contains(&VkFeature::MeshShader));

    let mut v13 = d.clone();
    v13.api_version = ApiVersion { major: 1, minor: 3 };
    v13.features.insert(VkFeature::ExtendedDynamicState);
    let rec13 = v13.recommended_enables();
    assert!(rec13
        .features
        .contains(&VkFeature::ShaderTerminateInvocation));
    assert!(!rec13
        .extensions
        .contains(&VkExtension::KhrShaderTerminateInvocation));
    // Core in 1.3 without a feature member: nothing to enable.
    assert!(!rec13.features.contains(&VkFeature::ExtendedDynamicState));
    let restricted = v13.with_enabled(&rec13).unwrap();
    assert!(restricted.has(VkFeature::ExtendedDynamicState));
}

/// Below Vulkan 1.3 the core consumes `VK_EXT_extended_dynamic_state2`, so a device listing it with
/// its feature is told to enable both, and the profile made from them has the feature; from 1.3 the
/// feature is core without a member, kept with nothing to enable.
#[test]
fn extended_dynamic_state2_is_enabled_below_1_3_and_implied_from_it() {
    let d = floor_with(
        &[
            VkExtension::ExtExtendedDynamicState,
            VkExtension::ExtExtendedDynamicState2,
        ],
        &[
            VkFeature::ExtendedDynamicState,
            VkFeature::ExtendedDynamicState2,
        ],
    );
    let rec = d.recommended_enables();
    for (extension, feature) in [
        (
            VkExtension::ExtExtendedDynamicState,
            VkFeature::ExtendedDynamicState,
        ),
        (
            VkExtension::ExtExtendedDynamicState2,
            VkFeature::ExtendedDynamicState2,
        ),
    ] {
        assert!(rec.extensions.contains(&extension), "{extension:?}");
        assert!(rec.features.contains(&feature), "{feature:?}");
    }
    let dev = Device::from_queries(
        d.clone().with_enabled(&rec).unwrap(),
        DevicePolicy::RESIDENT_TABLES,
    )
    .unwrap();
    let enabled = &dev.profile().fields().enabled;
    assert!(enabled.features.contains(&VkFeature::ExtendedDynamicState2));
    assert!(enabled
        .extensions
        .contains(&VkExtension::ExtExtendedDynamicState2));

    let mut v13 = d.clone();
    v13.api_version = ApiVersion { major: 1, minor: 3 };
    v13.extensions.clear();
    let rec13 = v13.recommended_enables();
    assert!(!rec13.features.contains(&VkFeature::ExtendedDynamicState2));
    assert!(!rec13
        .extensions
        .contains(&VkExtension::ExtExtendedDynamicState2));
    assert!(v13
        .with_enabled(&rec13)
        .unwrap()
        .has(VkFeature::ExtendedDynamicState2));
}

/// A listed extension whose requirement is not listed is not recommended, and neither are the
/// features only it exposes; a listed requirement is added with it.
#[test]
fn recommendations_close_extension_requirements() {
    let float2 = [
        VkFeature::ShaderBufferFloat32AtomicMinMax,
        VkFeature::ShaderSharedFloat32AtomicMinMax,
    ];
    let lonely = floor_with(&[VkExtension::ExtShaderAtomicFloat2], &float2);
    let rec = lonely.recommended_enables();
    assert!(!rec.extensions.contains(&VkExtension::ExtShaderAtomicFloat2));
    assert!(rec.features.is_disjoint(&set(&float2)));

    let both = floor_with(
        &[
            VkExtension::ExtShaderAtomicFloat2,
            VkExtension::ExtShaderAtomicFloat,
        ],
        &float2,
    );
    let rec = both.recommended_enables();
    assert!(rec.features.is_superset(&set(&float2)));
    assert!(rec.extensions.is_superset(&set(&[
        VkExtension::ExtShaderAtomicFloat2,
        VkExtension::ExtShaderAtomicFloat,
    ])));

    let swizzle = floor_with(
        &[
            VkExtension::ExtBorderColorSwizzle,
            VkExtension::ExtCustomBorderColor,
        ],
        &[VkFeature::BorderColorSwizzle],
    );
    let rec = swizzle.recommended_enables();
    assert!(rec.features.contains(&VkFeature::BorderColorSwizzle));
    // Not consumed on its own, enabled as the swizzle extension's requirement.
    assert!(rec.extensions.contains(&VkExtension::ExtCustomBorderColor));
}

#[test]
fn with_enabled_refuses_unsound_sets() {
    let d = floor_with(
        &[
            VkExtension::ExtRobustness2,
            VkExtension::ExtShaderAtomicFloat2,
            VkExtension::ExtShaderAtomicFloat,
        ],
        &[VkFeature::RobustImageAccess2],
    );
    // Not listed.
    let e = description_error(
        d.clone()
            .with_enabled(&enables(&[VkFeature::Maintenance8], &[])),
    );
    assert!(e.contains("Maintenance8"), "{e}");
    let e = description_error(
        d.clone()
            .with_enabled(&enables(&[], &[VkExtension::KhrMaintenance8])),
    );
    assert!(e.contains("KhrMaintenance8"), "{e}");
    // Enabled without its structure's extension.
    let e = description_error(
        d.clone()
            .with_enabled(&enables(&[VkFeature::RobustImageAccess2], &[])),
    );
    assert!(e.contains("ExtRobustness2"), "{e}");
    // An extension without its requirement.
    let e = description_error(
        d.clone()
            .with_enabled(&enables(&[], &[VkExtension::ExtShaderAtomicFloat2])),
    );
    assert!(e.contains("ExtShaderAtomicFloat"), "{e}");
    // The portability subset is mandatory where listed.
    let m = profile("moltenvk-m3");
    let e = description_error(m.clone().with_enabled(&EnableSet::new()));
    assert!(e.contains("KhrPortabilitySubset"), "{e}");
    m.with_enabled(&enables(&[], &[VkExtension::KhrPortabilitySubset]))
        .unwrap();
}

/// `with_enabled` refuses a set breaking a co-enable rule, a feature without its companion, naming
/// its VUID, and accepts the two together.
#[test]
fn with_enabled_refuses_sets_breaking_a_co_enable_rule() {
    let d = floor_with(
        &[],
        &[
            VkFeature::MultiviewTessellationShader,
            VkFeature::MultiviewGeometryShader,
        ],
    );
    for &(f, needs, vuid) in FEATURE_NEEDS {
        let e = description_error(d.clone().with_enabled(&enables(&[f], &[])));
        assert!(e.contains(vuid) && e.contains(&format!("{needs:?}")), "{e}");
        d.clone().with_enabled(&enables(&[f, needs], &[])).unwrap();
    }
}

/// `Device::from_queries` refuses a description listing a feature without its companion; every
/// committed profile lists both or neither. No companion needs one itself, which the one-pass
/// recommendation relies on.
#[test]
fn device_new_refuses_a_feature_without_its_companion() {
    for &(f, needs, vuid) in FEATURE_NEEDS {
        assert!(FEATURE_NEEDS.iter().all(|&(g, _, _)| g != needs));
        for (stem, text) in committed_profiles() {
            let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
            assert!(!desc.has(f) || desc.has(needs), "{stem}: {f:?}");
        }
        let mut d = floor_with(&[], &[f]);
        d.features.remove(&needs);
        let e = match Device::from_queries(d.clone(), DevicePolicy::RESIDENT_TABLES) {
            Err(Error::InvalidDescriptor(DescriptorError::DeviceDescription { detail })) => detail,
            other => panic!("{other:?}"),
        };
        assert!(e.contains(vuid), "{e}");
        d.features.insert(needs);
        Device::from_queries(d, DevicePolicy::RESIDENT_TABLES).unwrap();
    }
}

/// `Device::from_queries` refuses a description listing `shaderQuadControl` without
/// `shaderMaximalReconvergence`, which every device with `VK_KHR_shader_quad_control` supports, and
/// makes it with both; every committed profile that lists quad control lists both.
#[test]
fn device_new_refuses_quad_control_without_maximal_reconvergence() {
    let mut quads = 0;
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        if desc.has(VkFeature::ShaderQuadControl) {
            assert!(desc.has(VkFeature::ShaderMaximalReconvergence), "{stem}");
            quads += 1;
        }
    }
    assert!(quads > 0, "no committed profile lists quad control");
    let mut d = floor_with(
        &[
            VkExtension::KhrShaderMaximalReconvergence,
            VkExtension::KhrShaderQuadControl,
        ],
        &[VkFeature::ShaderQuadControl],
    );
    let e = match Device::from_queries(d.clone(), DevicePolicy::RESIDENT_TABLES) {
        Err(Error::InvalidDescriptor(DescriptorError::DeviceDescription { detail })) => detail,
        other => panic!("{other:?}"),
    };
    assert!(
        e.contains("ShaderQuadControl without ShaderMaximalReconvergence"),
        "{e}"
    );
    d.features.insert(VkFeature::ShaderMaximalReconvergence);
    let device = Device::from_queries(d, DevicePolicy::RESIDENT_TABLES).unwrap();
    let model = &device.profile().fields().subgroup;
    assert!(model.quad_control && model.maximal_reconvergence);
}

/// The restriction keeps exactly the enabled features and extensions and drops format entries
/// whose value the enabled extensions do not expose.
#[test]
fn with_enabled_restricts_features_extensions_and_formats() {
    let listed = [
        VkExtension::ExtRobustness2,
        VkExtension::KhrMaintenance5,
        VkExtension::KhrDynamicRendering,
    ];
    let mut d = floor_with(
        &listed,
        &[VkFeature::RobustImageAccess2, VkFeature::NullDescriptor],
    );
    let a8 = vk::Format::A8_UNORM;
    let r8 = vk::Format::R8_UNORM;
    let entry = *d.formats.formats.get(&r8).expect("the floor lists R8");
    d.formats.formats.insert(a8, entry);
    let kept = d
        .clone()
        .with_enabled(&enables(
            &[
                VkFeature::RobustImageAccess2,
                VkFeature::UniformBufferStandardLayout,
            ],
            &listed,
        ))
        .unwrap();
    assert_eq!(
        kept.features,
        set(&[
            VkFeature::RobustImageAccess2,
            VkFeature::UniformBufferStandardLayout
        ])
    );
    assert_eq!(kept.extensions, set(&listed));
    assert!(kept.formats.formats.contains_key(&a8));
    let dropped = d
        .with_enabled(&enables(&[VkFeature::UniformBufferStandardLayout], &[]))
        .unwrap();
    assert!(dropped.extensions.is_empty());
    assert!(!dropped.formats.formats.contains_key(&a8));
    assert!(dropped.formats.formats.contains_key(&r8));
}

/// Every extension-only feature's requirement names its extension on every committed profile
/// that lists the feature, through `required_enables` both of the bare feature and of the
/// derivation's closed requirements; a promoted feature names an exposing extension exactly
/// below its version (the finding: `maintenance8` without `VK_KHR_maintenance8`).
#[test]
fn required_enables_name_the_exposing_extension_of_every_feature() {
    let mut checked_extension_only = 0;
    for (stem, text) in committed_profiles() {
        let desc = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let api = desc.api_version;
        for &f in VkFeature::ALL.iter().filter(|&&f| desc.has(f)) {
            let bare = desc
                .required_enables(&requiring(&[f]))
                .unwrap_or_else(|c| panic!("{stem}: {f:?}: {c:?}"));
            let mut closed = requiring(&[f]);
            crate::spirv::requirements::close_extensions(&mut closed);
            assert_eq!(
                desc.required_enables(&closed),
                Ok(bare.clone()),
                "{stem}: {f:?}"
            );
            let x = f.exposure();
            assert_eq!(
                bare.features.contains(&f),
                x.has_member(api),
                "{stem}: {f:?}"
            );
            let named = x.extensions_at(api);
            assert_eq!(
                named.iter().any(|e| bare.extensions.contains(e)),
                !named.is_empty(),
                "{stem}: {f:?}: {bare:?}"
            );
            if let Exposure::Extension(&[e]) = x {
                assert!(bare.extensions.contains(&e), "{stem}: {f:?}");
                checked_extension_only += 1;
            }
        }
    }
    assert!(checked_extension_only > 0);
    let m8 = floor_with(&[VkExtension::KhrMaintenance8], &[VkFeature::Maintenance8]);
    assert_eq!(
        m8.required_enables(&requiring(&[VkFeature::Maintenance8])),
        Ok(enables(
            &[VkFeature::Maintenance8],
            &[VkExtension::KhrMaintenance8]
        ))
    );
}

#[test]
fn required_enables_follow_version_dependencies_formats_and_flags() {
    // A promoted feature below its version names its extension; at its version, nothing.
    let terminate = [VkFeature::ShaderTerminateInvocation];
    let d = floor_with(&[VkExtension::KhrShaderTerminateInvocation], &terminate);
    assert_eq!(
        d.required_enables(&requiring(&terminate)),
        Ok(enables(
            &terminate,
            &[VkExtension::KhrShaderTerminateInvocation]
        ))
    );
    let mut v13 = d.clone();
    v13.api_version = ApiVersion { major: 1, minor: 3 };
    assert_eq!(
        v13.required_enables(&requiring(&terminate)),
        Ok(enables(&terminate, &[]))
    );
    // Stage create flags need their features.
    let mut sized = floor_with(
        &[VkExtension::ExtSubgroupSizeControl],
        &[
            VkFeature::ComputeFullSubgroups,
            VkFeature::SubgroupSizeControl,
        ],
    );
    let flags = Requirements {
        stage_flags: set(&[StageCreateFlag::RequireFullSubgroups]),
        ..Requirements::default()
    };
    assert_eq!(
        sized.required_enables(&flags),
        Ok(enables(
            &[VkFeature::ComputeFullSubgroups],
            &[VkExtension::ExtSubgroupSizeControl]
        ))
    );
    sized.features.remove(&VkFeature::ComputeFullSubgroups);
    assert_eq!(
        sized.required_enables(&flags),
        Err(Capability::Feature(VkFeature::ComputeFullSubgroups))
    );
    // Extension requirements close transitively.
    let float2 = [VkFeature::ShaderBufferFloat32AtomicMinMax];
    let atomics = floor_with(
        &[
            VkExtension::ExtShaderAtomicFloat2,
            VkExtension::ExtShaderAtomicFloat,
        ],
        &float2,
    );
    assert_eq!(
        atomics.required_enables(&requiring(&float2)),
        Ok(enables(
            &float2,
            &[
                VkExtension::ExtShaderAtomicFloat2,
                VkExtension::ExtShaderAtomicFloat
            ]
        ))
    );
    let lonely = floor_with(&[VkExtension::ExtShaderAtomicFloat2], &float2);
    assert_eq!(
        lonely.required_enables(&requiring(&float2)),
        Err(Capability::Extension(VkExtension::ExtShaderAtomicFloat))
    );
    // A format enum value below its core version needs its extension.
    let a8 = Requirements {
        formats: set(&[(vk::Format::A8_UNORM, FormatUsage::SampledImage)]),
        ..Requirements::default()
    };
    assert_eq!(
        floor_with(&[], &[]).required_enables(&a8),
        Err(Capability::Extension(VkExtension::KhrMaintenance5))
    );
    let m5 = floor_with(
        &[
            VkExtension::KhrMaintenance5,
            VkExtension::KhrDynamicRendering,
        ],
        &[],
    );
    assert_eq!(
        m5.required_enables(&a8),
        Ok(enables(
            &[],
            &[
                VkExtension::KhrMaintenance5,
                VkExtension::KhrDynamicRendering
            ]
        ))
    );
    // Core formats need nothing.
    let rgba = Requirements {
        formats: set(&[(vk::Format::R8G8B8A8_UNORM, FormatUsage::SampledImage)]),
        ..Requirements::default()
    };
    assert_eq!(
        floor_with(&[], &[]).required_enables(&rgba),
        Ok(EnableSet::new())
    );
    // The API version and missing listed items are the first missing capability.
    let newer = Requirements {
        api_version: Some(ApiVersion { major: 1, minor: 3 }),
        ..Requirements::default()
    };
    assert_eq!(
        floor_with(&[], &[]).required_enables(&newer),
        Err(Capability::ApiVersion(ApiVersion { major: 1, minor: 3 }))
    );
    let named = Requirements {
        extensions: set(&[VkExtension::ExtShaderStencilExport]),
        ..Requirements::default()
    };
    assert_eq!(
        floor_with(&[], &[]).required_enables(&named),
        Err(Capability::Extension(VkExtension::ExtShaderStencilExport))
    );
    // The portability subset is part of every set on a device that lists it.
    assert_eq!(
        profile("moltenvk-m3").required_enables(&Requirements::default()),
        Ok(enables(&[], &[VkExtension::KhrPortabilitySubset]))
    );
}

/// `Device::from_queries` refuses a description listing a feature it does not expose.
#[test]
fn device_new_refuses_unexposed_features() {
    let mut d = profile("runtime-floor");
    d.features.insert(VkFeature::Maintenance8);
    let e = match Device::from_queries(d.clone(), DevicePolicy::RESIDENT_TABLES) {
        Err(Error::InvalidDescriptor(DescriptorError::DeviceDescription { detail })) => detail,
        other => panic!("{other:?}"),
    };
    assert!(
        e.contains("Maintenance8") && e.contains("KhrMaintenance8"),
        "{e}"
    );
    d.extensions.insert(VkExtension::KhrMaintenance8);
    Device::from_queries(d, DevicePolicy::RESIDENT_TABLES).unwrap();
}

/// The profile's format tables hold only formats the description exposes: a queried description
/// answers for the formats of extensions the device supports, the consumer may not enable them.
/// Every committed profile lists only formats its own version or listed extensions expose, and
/// every one keeps its entry but a format only extensions outside the product's vocabulary expose
/// (a vulkaninfo capture's `VK_NV_optical_flow` formats), which a queried description never asks
/// about.
#[test]
fn profile_format_tables_hold_only_exposed_formats() {
    for (stem, text) in committed_profiles() {
        let document: serde_json::Value = serde_json::from_str(&text).unwrap();
        let listed: BTreeSet<String> = document["capabilities"]
            .as_object()
            .into_iter()
            .flat_map(|blocks| blocks.values())
            .filter_map(|block| block.get("extensions")?.as_object())
            .flat_map(|extensions| extensions.keys().cloned())
            .collect();
        let d = DeviceDescription::from_khronos_json(&text, None).unwrap();
        for &f in d.formats.formats.keys() {
            let exposure = f.exposure().unwrap_or_else(|| panic!("{stem}: {f:?}"));
            assert!(
                exposure.exposed(d.api_version, |name| listed.contains(name)),
                "{stem}: {f:?} is not exposed by the profile's version or extensions"
            );
            assert!(
                enables::format_exposed(f, d.api_version, &d.extensions)
                    || exposure
                        .extensions
                        .iter()
                        .all(|name| crate::device::khronos::extension_from_name(name).is_none()),
                "{stem}: {f:?} loses its entry"
            );
        }
    }
    let mut d = profile("runtime-floor");
    let entry = d.formats.formats[&vk::Format::R8_UNORM];
    d.formats.formats.insert(vk::Format::A8_UNORM, entry);
    let a8 = |d: &DeviceDescription| {
        Device::from_queries(d.clone(), DevicePolicy::RESIDENT_TABLES)
            .unwrap()
            .profile()
            .supports_format(vk::Format::A8_UNORM, FormatUsage::SampledImage)
    };
    assert!(!a8(&d));
    d.extensions.extend([
        VkExtension::KhrMaintenance5,
        VkExtension::KhrDynamicRendering,
    ]);
    assert!(a8(&d));
}

/// A storage capability without a format is enabled on the device, as the registry allows
/// (`WITHOUT_FORMAT_CAPABILITIES`): the ANV profile, without `shaderStorageImageReadWithoutFormat`,
/// is told to enable `VK_KHR_format_feature_flags2` at Vulkan 1.2, and a module that reads storage
/// without a format then checks and requires that extension in place of the feature; at its own
/// Vulkan 1.4 it requires nothing; without the extension it lacks the feature. MoltenVK, which has
/// the feature, requires the feature.
#[test]
fn a_storage_read_without_a_format_is_enabled_by_the_feature_the_version_or_the_extension() {
    use crate::device::requirements::SpirvCapability;
    use crate::pipeline::testing::description_at_1_2;
    let read = VkFeature::ShaderStorageImageReadWithoutFormat;
    let flags2 = VkExtension::KhrFormatFeatureFlags2;
    let mut reads = Requirements::default();
    reads.spirv_capabilities.insert(SpirvCapability::of(
        crate::spv::Capability::StorageImageReadWithoutFormat,
    ));
    let policy = DevicePolicy::RESIDENT_TABLES;
    let open = |desc: &DeviceDescription| {
        let enabled = desc.recommended_enables();
        let opened = desc.clone().with_enabled(&enabled).unwrap();
        (
            enabled,
            Device::from_queries(opened.clone(), policy).unwrap(),
            opened,
        )
    };

    let anv = description_at_1_2("intel-irisxe-rplp-anv", true).unwrap();
    assert!(!anv.has(read));
    let (enabled, device, opened) = open(&anv);
    assert!(enabled.extensions.contains(&flags2), "{enabled:?}");
    assert!(device.profile().without_format_by_format_bits());
    assert_eq!(device.profile().check(&reads), Ok(()));
    let required = opened.required_enables(&reads).unwrap();
    assert!(required.extensions.contains(&flags2), "{required:?}");
    assert!(!required.features.contains(&read));

    let without = description_at_1_2("intel-irisxe-rplp-anv", false).unwrap();
    let (enabled, device, opened) = open(&without);
    assert!(!enabled.extensions.contains(&flags2));
    assert!(!device.profile().without_format_by_format_bits());
    assert_eq!(
        device.profile().check(&reads),
        Err(Capability::Feature(read))
    );
    assert_eq!(
        opened.required_enables(&reads),
        Err(Capability::Feature(read))
    );

    let whole = profile("intel-irisxe-rplp-anv");
    assert!(whole.api_version >= ApiVersion::V1_3);
    let (enabled, device, opened) = open(&whole);
    assert!(!enabled.extensions.contains(&flags2), "core at 1.3");
    assert_eq!(device.profile().check(&reads), Ok(()));
    let required = opened.required_enables(&reads).unwrap();
    assert!(!required.extensions.contains(&flags2));
    assert!(!required.features.contains(&read));

    let moltenvk = profile("moltenvk-m3");
    assert!(moltenvk.has(read));
    let (_, device, opened) = open(&moltenvk);
    assert_eq!(device.profile().check(&reads), Ok(()));
    assert!(opened
        .required_enables(&reads)
        .unwrap()
        .features
        .contains(&read));
}
