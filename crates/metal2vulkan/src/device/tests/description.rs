//! `DeviceDescription` and the Khronos JSON reader.

use super::*;

const COMMITTED: &[&str] = &[
    "amd-780m-hawkpoint-radv",
    "amd-780m-radv",
    "amd-rdna2-apu-radv",
    "amd-rdna3-radv",
    "gate-lavapipe",
    "gate-lavapipe-x86_64",
    "intel-arl-anv",
    "intel-irisxe-anv",
    "intel-irisxe-rplp-anv",
    "intel-uhd770-rpls-anv",
    "lavapipe",
    "lunarg-desktop-baseline-2023",
    "moltenvk-m3",
    "nvidia-pascal-linux",
    "nvidia-rtx50-linux",
    "nvidia-rtx5080-laptop",
    "roadmap-2022",
    "swiftshader",
    "vk12-floor",
];

#[test]
fn every_committed_profile_parses() {
    let all = committed_profiles();
    let stems: Vec<&str> = all.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(stems, COMMITTED, "the committed profile set changed");
    for (stem, text) in &all {
        let d = DeviceDescription::from_khronos_json(text, None)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(d.api_version >= ApiVersion::V1_2, "{stem}");
        // No committed document reports a feature its version requires as unsupported.
        for f in required_features(d.api_version) {
            assert!(d.has(f), "{stem} lacks required {f:?}");
        }
        assert!(d.limits.max_compute_work_group_invocations >= 128, "{stem}");
        assert!(d.limits.max_push_constants_size >= 128, "{stem}");
        assert!(d.limits.max_storage_buffer_range >= 1 << 27, "{stem}");
        assert!(
            d.limits.min_storage_buffer_offset_alignment <= 256,
            "{stem}"
        );
        assert!(
            d.subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
            "{stem}"
        );
    }
}

/// The Khronos reader names every feature of the vocabulary, as the device query reads every one
/// (`ash_interop`'s `every_feature_has_a_source`), so a device and its captured profile list the
/// same features: validation holds a venue's device to its committed profile (`venue_check`),
/// and `taskShader` and `meshShader`, read from a device but not from its profile, stopped the
/// RADV venue.
#[test]
fn every_feature_has_a_member_name() {
    use crate::device::khronos::FEATURE_MEMBERS;
    use std::collections::BTreeSet;
    let named: BTreeSet<VkFeature> = FEATURE_MEMBERS.iter().map(|&(_, f)| f).collect();
    let unnamed: Vec<&VkFeature> = VkFeature::ALL
        .iter()
        .filter(|f| !named.contains(f))
        .collect();
    assert!(unnamed.is_empty(), "no Khronos member name: {unnamed:?}");
    let names: BTreeSet<&str> = FEATURE_MEMBERS.iter().map(|&(name, _)| name).collect();
    assert_eq!(
        names.len(),
        FEATURE_MEMBERS.len(),
        "a member is listed twice"
    );
}

#[test]
fn device_reports_carry_identity_and_formats() {
    let m = profile("moltenvk-m3");
    assert_eq!(m.driver_id, Some(vk::DriverId::MOLTENVK));
    assert_eq!(m.vendor_id, 4203);
    assert_eq!(m.subgroup.size, 32);
    assert_eq!(m.subgroup.max_size, 32);
    assert!(m.subgroup.required_size_stages.is_empty());
    assert!(m.has(VkFeature::BufferDeviceAddress));
    assert!(m.extensions.contains(&VkExtension::ExtRobustness2));
    assert!(m
        .formats
        .supports(vk::Format::R8G8B8A8_UNORM, FormatUsage::SampledImage));
    assert!(m
        .formats
        .supports(vk::Format::R8G8B8A8_USCALED, FormatUsage::VertexBuffer));
    assert!(!m
        .formats
        .supports(vk::Format::R8G8B8A8_USCALED, FormatUsage::SampledImage));
    let nv = profile("nvidia-rtx50-linux");
    assert_eq!(nv.driver_id, Some(vk::DriverId::NVIDIA_PROPRIETARY));
    assert_eq!((nv.subgroup.min_size, nv.subgroup.max_size), (32, 32));
    assert!(nv.has(VkFeature::ShaderInt64));
}

/// The floor is Vulkan 1.2 core: no optional feature, not even one the product uses where present
/// (narrow integers and storage access, formatless storage images), since no product path needs
/// one.
#[test]
fn vk12_floor_is_vulkan_1_2_core() {
    let d = profile("vk12-floor");
    assert_eq!(d.api_version, ApiVersion::V1_2);
    // Vulkan 1.2 requires uniformBufferStandardLayout.
    assert!(d.has(VkFeature::UniformBufferStandardLayout));
    assert!(d.has(VkFeature::RobustBufferAccess));
    for f in [
        VkFeature::ShaderInt16,
        VkFeature::ShaderInt64,
        VkFeature::BufferDeviceAddress,
        VkFeature::VertexPipelineStoresAndAtomics,
        VkFeature::FragmentStoresAndAtomics,
        VkFeature::SubgroupSizeControl,
        VkFeature::StorageBuffer8BitAccess,
        VkFeature::StorageBuffer16BitAccess,
        VkFeature::ShaderStorageImageReadWithoutFormat,
        VkFeature::ShaderStorageImageWriteWithoutFormat,
    ] {
        assert!(!d.has(f), "{f:?}");
    }
    // The formats are the Vulkan 1.2 Required Format Support tables' unconditional entries: a
    // storage view of a 8-bit format needs shaderStorageImageExtendedFormats, and depth attachments
    // other than D16 and the compressed families are choices.
    assert_eq!(d.formats.formats.len(), 56);
    for usage in [
        FormatUsage::SampledImage,
        FormatUsage::SampledImageFilterLinear,
        FormatUsage::StorageImage,
        FormatUsage::ColorAttachmentBlend,
        FormatUsage::VertexBuffer,
        FormatUsage::StorageTexelBuffer,
    ] {
        assert!(
            d.formats.supports(vk::Format::R8G8B8A8_UNORM, usage),
            "{usage:?}"
        );
    }
    assert!(d
        .formats
        .supports(vk::Format::D16_UNORM, FormatUsage::DepthStencilAttachment));
    for (format, usage) in [
        (vk::Format::R8_UNORM, FormatUsage::StorageImage),
        (vk::Format::D32_SFLOAT, FormatUsage::DepthStencilAttachment),
        (vk::Format::BC1_RGBA_UNORM_BLOCK, FormatUsage::SampledImage),
        (
            vk::Format::R32_SFLOAT,
            FormatUsage::SampledImageFilterLinear,
        ),
    ] {
        assert!(!d.formats.supports(format, usage), "{format:?} {usage:?}");
    }
    assert_eq!(d.driver_id, None);
    assert_eq!(d.limits.max_compute_shared_memory_size, 16384);
    assert_eq!(d.limits.max_compute_work_group_size, [128, 128, 64]);
    assert_eq!(d.limits.min_storage_buffer_offset_alignment, 256);
}

/// `supportsNonZeroFirstInstance` comes from the KHR divisor (or 1.4) properties; an EXT-only
/// divisor or no divisor at all reports `false`.
#[test]
fn divisor_first_instance_support_follows_exposure() {
    for (stem, expected) in [
        ("lavapipe", false),
        ("moltenvk-m3", true),
        ("nvidia-rtx50-linux", true),
        ("lunarg-desktop-baseline-2023", false),
        ("vk12-floor", false),
    ] {
        assert_eq!(
            profile(stem).limits.supports_non_zero_first_instance,
            expected,
            "{stem}"
        );
    }
}

/// `rgba4OpaqueBlackSwizzled` is read where `VK_KHR_maintenance10` is exposed and is an enabled
/// property when true; MoltenVK's capture has no maintenance10, so it is false there.
#[test]
fn rgba4_opaque_black_swizzled_follows_exposure() {
    use crate::device::requirements::VkProperty;
    assert!(profile("lavapipe").rgba4_opaque_black_swizzled);
    assert!(!profile("moltenvk-m3").rgba4_opaque_black_swizzled);
    let props = r#""properties": {"VkPhysicalDeviceMaintenance10PropertiesKHR":
        {"rgba4OpaqueBlackSwizzled": true}}"#;
    let with = doc(
        "1.2.0",
        r#""a""#,
        &format!(r#""a": {{"extensions": {{"VK_KHR_maintenance10": 1}}, {props}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
    assert!(d.rgba4_opaque_black_swizzled);
    let dev = crate::device::Device::new(d, crate::device::DevicePolicy::default()).unwrap();
    assert!(dev
        .profile()
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::Rgba4OpaqueBlackSwizzled));
    let without = doc("1.2.0", r#""a""#, &format!(r#""a": {{{props}}}"#));
    assert!(
        !DeviceDescription::from_khronos_json(&without, None)
            .unwrap()
            .rgba4_opaque_black_swizzled
    );
}

/// A device that exposes only the ARM spelling of rasterization-order attachment access has the
/// same feature as one exposing the EXT spelling: the extension and its feature structure map to
/// one vocabulary entry, so its fragments read their color attachments in rasterization order
/// and the ordered fetch is enabled.
#[test]
fn the_arm_spelling_of_rasterization_order_access_is_the_ext_one() {
    use crate::device::requirements::VkExtension;
    let feature = r#""VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM":
        {"rasterizationOrderColorAttachmentAccess": true}"#;
    let arm = doc(
        "1.2.0",
        r#""a""#,
        &format!(
            r#""a": {{"extensions": {{"VK_ARM_rasterization_order_attachment_access": 1}},
                       "features": {{{feature}}}}}"#
        ),
    );
    let d = DeviceDescription::from_khronos_json(&arm, None).unwrap();
    assert!(d
        .extensions
        .contains(&VkExtension::ExtRasterizationOrderAttachmentAccess));
    assert!(d.has(VkFeature::RasterizationOrderColorAttachmentAccess));
    let dev = crate::device::Device::new(d, crate::device::DevicePolicy::default()).unwrap();
    assert!(
        dev.profile()
            .fields()
            .fragment
            .rasterization_order_color_access
    );
    // The structure without its extension is not exposed.
    let bare = doc(
        "1.2.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{feature}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&bare, None).unwrap();
    assert!(!d.has(VkFeature::RasterizationOrderColorAttachmentAccess));
}

/// `robustBufferAccessUpdateAfterBind` is read from either structure that states it, and a
/// document that does not state it (`roadmap-2022`, which requires nothing of it) reads false.
#[test]
fn robust_buffer_access_update_after_bind_is_read_where_stated() {
    for (stem, stated) in [
        ("lavapipe", true),
        ("moltenvk-m3", true),
        ("nvidia-rtx50-linux", true),
        ("swiftshader", false),
        ("lunarg-desktop-baseline-2023", false),
        ("vk12-floor", false),
        ("roadmap-2022", false),
    ] {
        assert_eq!(
            profile(stem).robust_buffer_access_update_after_bind,
            stated,
            "{stem}"
        );
    }
    for structure in [
        "VkPhysicalDeviceVulkan12Properties",
        "VkPhysicalDeviceDescriptorIndexingProperties",
        "VkPhysicalDeviceDescriptorIndexingPropertiesEXT",
    ] {
        let with = doc(
            "1.2.0",
            r#""a""#,
            &format!(
                r#""a": {{"properties": {{"{structure}":
                    {{"robustBufferAccessUpdateAfterBind": true}}}}}}"#
            ),
        );
        let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
        assert!(d.robust_buffer_access_update_after_bind, "{structure}");
    }
    let unstated = doc("1.2.0", r#""a""#, r#""a": {}"#);
    assert!(
        !DeviceDescription::from_khronos_json(&unstated, None)
            .unwrap()
            .robust_buffer_access_update_after_bind
    );
}

/// A profile's `m2v_color_attachment_sample_counts` entry states each color attachment format's
/// own sample counts, which no profiles-schema structure holds: the reader puts them on the
/// format's properties. An entry for another profile states nothing for this one, and a format
/// outside the registry, one the profile does not state as a color attachment, a sample count
/// outside the registry, and a malformed entry are refused.
#[test]
fn color_attachment_sample_counts_are_read_from_their_block() {
    let block = r#""a": {"formats": {
        "VK_FORMAT_R32G32B32A32_UINT": {"VkFormatProperties": {
            "optimalTilingFeatures": ["VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT"]}},
        "VK_FORMAT_R8_UNORM": {"VkFormatProperties": {
            "optimalTilingFeatures": ["VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"]}}}}"#;
    let with = |counts: &str| {
        let base = doc("1.2.0", r#""a""#, block);
        format!(
            r#"{{"m2v_color_attachment_sample_counts": {counts}, {}"#,
            base.trim_start().strip_prefix('{').unwrap()
        )
    };
    let counts = |json: &str, format: vk::Format| {
        DeviceDescription::from_khronos_json(json, None)
            .unwrap()
            .formats
            .formats[&format]
            .color_attachment_sample_counts
    };
    let uint = vk::Format::R32G32B32A32_UINT;
    let stated = with(
        r#"{"VP_TEST": {"VK_FORMAT_R32G32B32A32_UINT":
            ["VK_SAMPLE_COUNT_1_BIT", "VK_SAMPLE_COUNT_4_BIT"]}}"#,
    );
    assert_eq!(
        counts(&stated, uint),
        vk::SampleCountFlags::COUNT_1 | vk::SampleCountFlags::COUNT_4
    );
    assert!(counts(&doc("1.2.0", r#""a""#, block), uint).is_empty());
    let other = with(r#"{"VP_OTHER": {"VK_FORMAT_R32G32B32A32_UINT": ["VK_SAMPLE_COUNT_4_BIT"]}}"#);
    assert!(counts(&other, uint).is_empty());
    for refused in [
        r#"{"VP_TEST": {"VK_FORMAT_NOT_A_FORMAT": ["VK_SAMPLE_COUNT_1_BIT"]}}"#,
        r#"{"VP_TEST": {"VK_FORMAT_R8_UNORM": ["VK_SAMPLE_COUNT_1_BIT"]}}"#,
        r#"{"VP_TEST": {"VK_FORMAT_R8G8B8A8_UNORM": ["VK_SAMPLE_COUNT_1_BIT"]}}"#,
        r#"{"VP_TEST": {"VK_FORMAT_R32G32B32A32_UINT": ["VK_SAMPLE_COUNT_3_BIT"]}}"#,
        r#"{"VP_TEST": {"VK_FORMAT_R32G32B32A32_UINT": "VK_SAMPLE_COUNT_1_BIT"}}"#,
        r#"{"VP_TEST": []}"#,
        r#"[]"#,
    ] {
        assert!(
            DeviceDescription::from_khronos_json(&with(refused), None).is_err(),
            "{refused}"
        );
    }
}

/// A document that still holds the former top-level `m2v_deviations` object is refused, whatever
/// it states, so a statement is never silently dropped: a driver's deviations are measured facts
/// of its release (`device::driver_facts`).
#[test]
fn a_stated_deviation_is_refused() {
    let base = doc("1.2.0", r#""a""#, r#""a": {}"#);
    assert!(DeviceDescription::from_khronos_json(&base, None).is_ok());
    for deviations in [
        r#"{"VP_TEST": {"metalSampler": {"rule": "r"}}}"#,
        r#"{"VP_OTHER": {}}"#,
        r#"{}"#,
    ] {
        let stated = format!(
            r#"{{"m2v_deviations": {deviations}, {}"#,
            base.trim_start().strip_prefix('{').unwrap()
        );
        assert!(
            DeviceDescription::from_khronos_json(&stated, None).is_err(),
            "{deviations}"
        );
    }
}

/// The portability subset's `minVertexInputBindingStrideAlignment` is read where
/// `VK_KHR_portability_subset` is exposed (the committed MoltenVK capture reports 1), is the most
/// restrictive value when such a document omits it, and is absent on every other device.
#[test]
fn portability_stride_alignment_follows_exposure() {
    assert_eq!(
        profile("moltenvk-m3")
            .limits
            .min_vertex_input_binding_stride_alignment,
        Some(1)
    );
    for stem in [
        "lavapipe",
        "nvidia-rtx50-linux",
        "vk12-floor",
        "amd-rdna3-radv",
    ] {
        assert_eq!(
            profile(stem)
                .limits
                .min_vertex_input_binding_stride_alignment,
            None,
            "{stem}"
        );
    }
    let props = r#""properties": {"VkPhysicalDevicePortabilitySubsetPropertiesKHR":
        {"minVertexInputBindingStrideAlignment": 4}}"#;
    let with = doc(
        "1.2.0",
        r#""a""#,
        &format!(r#""a": {{"extensions": {{"VK_KHR_portability_subset": 1}}, {props}}}"#),
    );
    let l = DeviceDescription::from_khronos_json(&with, None)
        .unwrap()
        .limits;
    assert_eq!(l.min_vertex_input_binding_stride_alignment, Some(4));
    assert!(l.portable_vertex_stride(8) && l.portable_vertex_stride(4));
    assert!(!l.portable_vertex_stride(6) && !l.portable_vertex_stride(0));
    let without = doc("1.2.0", r#""a""#, &format!(r#""a": {{{props}}}"#));
    let l = DeviceDescription::from_khronos_json(&without, None)
        .unwrap()
        .limits;
    assert_eq!(l.min_vertex_input_binding_stride_alignment, None);
    assert!(l.portable_vertex_stride(0) && l.portable_vertex_stride(6));
    let unreported = doc(
        "1.2.0",
        r#""a""#,
        r#""a": {"extensions": {"VK_KHR_portability_subset": 1}}"#,
    );
    let l = DeviceDescription::from_khronos_json(&unreported, None)
        .unwrap()
        .limits;
    assert_eq!(l.min_vertex_input_binding_stride_alignment, Some(u32::MAX));
}

/// The acceleration-structure limits are read from their extension's property structure where
/// the extension is exposed: as reported, the required minimum (16) when omitted, and 0 without
/// the extension.
#[test]
fn acceleration_structure_limits_read_their_property_structure() {
    let lavapipe = profile("lavapipe").limits;
    assert_eq!(
        lavapipe.max_per_stage_descriptor_acceleration_structures,
        1_015_808
    );
    assert_eq!(
        lavapipe.max_descriptor_set_acceleration_structures,
        1_015_808
    );
    let props = r#""properties": {"VkPhysicalDeviceAccelerationStructurePropertiesKHR":
        {"maxPerStageDescriptorAccelerationStructures": 1000,
         "maxDescriptorSetAccelerationStructures": 2000}}"#;
    let read = |blocks: &str| {
        DeviceDescription::from_khronos_json(&doc("1.2.0", r#""a""#, blocks), None)
            .unwrap()
            .limits
    };
    let extension = r#""extensions": {"VK_KHR_acceleration_structure": 13}"#;
    let l = read(&format!(r#""a": {{{extension}, {props}}}"#));
    assert_eq!(l.max_per_stage_descriptor_acceleration_structures, 1000);
    assert_eq!(l.max_descriptor_set_acceleration_structures, 2000);
    let l = read(&format!(r#""a": {{{extension}}}"#));
    assert_eq!(l.max_per_stage_descriptor_acceleration_structures, 16);
    assert_eq!(l.max_descriptor_set_acceleration_structures, 16);
    let l = read(&format!(r#""a": {{{props}}}"#));
    assert_eq!(l.max_per_stage_descriptor_acceleration_structures, 0);
    assert_eq!(l.max_descriptor_set_acceleration_structures, 0);
}

/// `nonStrictWideLinesUseParallelogram` is read where `VK_KHR_maintenance5` is exposed, from its
/// extension's property structure as from the Vulkan 1.4 one (a capture of a Vulkan 1.3 device
/// carries only the first), and is false without the extension or where unstated.
#[test]
fn wide_line_parallelograms_read_the_maintenance5_structure() {
    assert!(
        profile("amd-780m-radv")
            .limits
            .non_strict_wide_lines_use_parallelogram
    );
    assert!(
        !profile("lavapipe")
            .limits
            .non_strict_wide_lines_use_parallelogram
    );
    let read = |api: &str, blocks: &str| {
        DeviceDescription::from_khronos_json(&doc(api, r#""a""#, blocks), None)
            .unwrap()
            .limits
            .non_strict_wide_lines_use_parallelogram
    };
    let extension = r#""extensions": {"VK_KHR_maintenance5": 1}"#;
    for structure in [
        "VkPhysicalDeviceMaintenance5PropertiesKHR",
        "VkPhysicalDeviceMaintenance5Properties",
    ] {
        let props = format!(
            r#""properties": {{"{structure}": {{"nonStrictWideLinesUseParallelogram": true}}}}"#
        );
        assert!(
            read("1.3.0", &format!(r#""a": {{{extension}, {props}}}"#)),
            "{structure}"
        );
        assert!(
            !read("1.3.0", &format!(r#""a": {{{props}}}"#)),
            "{structure}"
        );
    }
    let v14 = r#""properties": {"VkPhysicalDeviceVulkan14Properties":
        {"nonStrictWideLinesUseParallelogram": true}}"#;
    assert!(read("1.4.0", &format!(r#""a": {{{v14}}}"#)));
    assert!(!read("1.3.0", &format!(r#""a": {{{extension}}}"#)));
}

#[test]
fn profile_documents_imply_version_required_features() {
    // The LunarG baseline lists 1.2 structures but not every member 1.2 requires.
    let d = profile("lunarg-desktop-baseline-2023");
    assert!(d.has(VkFeature::ShaderSubgroupExtendedTypes));
    assert!(d.has(VkFeature::UniformBufferStandardLayout));
}

fn doc(api: &str, caps: &str, blocks: &str) -> String {
    format!(
        r#"{{"capabilities": {{{blocks}}},
            "profiles": {{"VP_TEST": {{"api-version": "{api}", "capabilities": [{caps}]}}}}}}"#
    )
}

#[test]
fn extension_structures_need_their_extension_or_promotion() {
    let atomic_float = r#""VkPhysicalDeviceShaderAtomicFloatFeaturesEXT": {"shaderBufferFloat32AtomicAdd": true},
        "VkPhysicalDeviceScalarBlockLayoutFeaturesEXT": {"scalarBlockLayout": true}"#;
    let without = doc(
        "1.1.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{atomic_float}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&without, None).unwrap();
    assert!(!d.has(VkFeature::ShaderBufferFloat32AtomicAdd));
    assert!(!d.has(VkFeature::ScalarBlockLayout));
    let with = doc(
        "1.1.0",
        r#""a""#,
        &format!(
            r#""a": {{"extensions": {{"VK_EXT_shader_atomic_float": 1, "VK_EXT_scalar_block_layout": 1}},
                       "features": {{{atomic_float}}}}}"#
        ),
    );
    let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
    assert!(d.has(VkFeature::ShaderBufferFloat32AtomicAdd));
    assert!(d.has(VkFeature::ScalarBlockLayout));
    // Promoted to 1.2: the extension structure counts on a 1.2 device without the extension.
    let promoted = doc(
        "1.2.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{atomic_float}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&promoted, None).unwrap();
    assert!(d.has(VkFeature::ScalarBlockLayout));
    assert!(!d.has(VkFeature::ShaderBufferFloat32AtomicAdd));
    // A 1.3 structure on a 1.2 device is not exposed.
    let early = doc(
        "1.2.0",
        r#""a""#,
        r#""a": {"features": {"VkPhysicalDeviceVulkan13Features": {"computeFullSubgroups": true}}}"#,
    );
    let d = DeviceDescription::from_khronos_json(&early, None).unwrap();
    assert!(!d.has(VkFeature::ComputeFullSubgroups));
}

#[test]
fn subgroup_rotate_reads_its_extension_structure_and_the_1_4_block() {
    let rotate = r#""VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR":
        {"shaderSubgroupRotate": true, "shaderSubgroupRotateClustered": true}"#;
    let without = doc(
        "1.3.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{rotate}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&without, None).unwrap();
    assert!(!d.has(VkFeature::ShaderSubgroupRotate));
    let with = doc(
        "1.3.0",
        r#""a""#,
        &format!(
            r#""a": {{"extensions": {{"VK_KHR_shader_subgroup_rotate": 1}},
                       "features": {{{rotate}}}}}"#
        ),
    );
    let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
    assert!(d.has(VkFeature::ShaderSubgroupRotate));
    assert!(d.has(VkFeature::ShaderSubgroupRotateClustered));
    // Core in 1.4: the committed 1.4 profiles enable it through `VkPhysicalDeviceVulkan14Features`.
    let d = profile("moltenvk-m3");
    assert!(d.has(VkFeature::ShaderSubgroupRotate));
    assert!(d.has(VkFeature::ShaderSubgroupRotateClustered));
}

/// `maintenance8` is read from its extension structure only when `VK_KHR_maintenance8` is listed
/// (no core version promotes it); the committed device reports that list it enable it.
#[test]
fn maintenance8_reads_its_extension_structure() {
    let m8 = r#""VkPhysicalDeviceMaintenance8FeaturesKHR": {"maintenance8": true}"#;
    let without = doc(
        "1.4.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{m8}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&without, None).unwrap();
    assert!(!d.has(VkFeature::Maintenance8));
    assert!(!d.extensions.contains(&VkExtension::KhrMaintenance8));
    let with = doc(
        "1.2.0",
        r#""a""#,
        &format!(r#""a": {{"extensions": {{"VK_KHR_maintenance8": 1}}, "features": {{{m8}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
    assert!(d.has(VkFeature::Maintenance8));
    assert!(d.extensions.contains(&VkExtension::KhrMaintenance8));
    for (stem, expected) in [
        ("amd-rdna2-apu-radv", true),
        ("amd-rdna3-radv", true),
        ("intel-arl-anv", true),
        ("intel-irisxe-anv", true),
        ("lavapipe", true),
        ("lunarg-desktop-baseline-2023", false),
        ("moltenvk-m3", true),
        ("nvidia-pascal-linux", true),
        ("nvidia-rtx50-linux", true),
        ("roadmap-2022", false),
        ("swiftshader", false),
        ("vk12-floor", false),
    ] {
        assert_eq!(
            profile(stem).has(VkFeature::Maintenance8),
            expected,
            "{stem}"
        );
    }
}

/// The border-color swizzle features are read from their extension structure only when
/// `VK_EXT_border_color_swizzle` is listed, and the named extensions reach the description by their
/// registry names.
#[test]
fn border_color_swizzle_and_format_extensions_are_read() {
    let bcs = r#""VkPhysicalDeviceBorderColorSwizzleFeaturesEXT":
        {"borderColorSwizzle": true, "borderColorSwizzleFromImage": true}"#;
    let without = doc(
        "1.4.0",
        r#""a""#,
        &format!(r#""a": {{"features": {{{bcs}}}}}"#),
    );
    let d = DeviceDescription::from_khronos_json(&without, None).unwrap();
    assert!(!d.has(VkFeature::BorderColorSwizzle));
    assert!(!d.has(VkFeature::BorderColorSwizzleFromImage));
    let with = doc(
        "1.2.0",
        r#""a""#,
        &format!(
            r#""a": {{"extensions": {{"VK_EXT_border_color_swizzle": 1,
                "VK_EXT_custom_border_color": 1}}, "features": {{{bcs}}}}}"#
        ),
    );
    let d = DeviceDescription::from_khronos_json(&with, None).unwrap();
    assert!(d.has(VkFeature::BorderColorSwizzle));
    assert!(d.has(VkFeature::BorderColorSwizzleFromImage));
    assert!(d.extensions.contains(&VkExtension::ExtBorderColorSwizzle));
    assert!(d.extensions.contains(&VkExtension::ExtCustomBorderColor));
    for stem in ["amd-rdna3-radv", "intel-arl-anv", "lavapipe"] {
        assert!(
            profile(stem).has(VkFeature::BorderColorSwizzleFromImage),
            "{stem}"
        );
    }
    let mvk = profile("moltenvk-m3");
    for ext in [
        VkExtension::KhrMaintenance5,
        VkExtension::ExtTextureCompressionAstcHdr,
        VkExtension::ImgFormatPvrtc,
        VkExtension::KhrPortabilitySubset,
    ] {
        assert!(mvk.extensions.contains(&ext), "moltenvk-m3 lists {ext:?}");
    }
    assert!(!profile("vk12-floor")
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset));
}

#[test]
fn explicit_false_overrides_the_version_requirement() {
    let text = doc(
        "1.2.0",
        r#""a""#,
        r#""a": {"features": {"VkPhysicalDeviceVulkan12Features": {"uniformBufferStandardLayout": false}}}"#,
    );
    let d = DeviceDescription::from_khronos_json(&text, None).unwrap();
    assert!(!d.has(VkFeature::UniformBufferStandardLayout));
    assert!(d.has(VkFeature::ShaderSubgroupExtendedTypes));
}

#[test]
fn blocks_merge_to_the_more_capable_value() {
    let text = doc(
        "1.2.0",
        r#""a", "b""#,
        r#""a": {"properties": {"VkPhysicalDeviceProperties": {"limits": {
                    "maxComputeSharedMemorySize": 32768, "minStorageBufferOffsetAlignment": 64,
                    "maxComputeWorkGroupSize": [1024, 64, 64],
                    "framebufferColorSampleCounts": ["VK_SAMPLE_COUNT_1_BIT"]}},
                  "VkPhysicalDeviceVulkan11Properties": {"subgroupSupportedStages": ["VK_SHADER_STAGE_COMPUTE_BIT"]}},
                 "features": {"VkPhysicalDeviceFeatures": {"shaderInt64": false}}},
           "b": {"properties": {"VkPhysicalDeviceProperties": {"limits": {
                    "maxComputeSharedMemorySize": 16384, "minStorageBufferOffsetAlignment": 256,
                    "maxComputeWorkGroupSize": [128, 1024, 1],
                    "framebufferColorSampleCounts": ["VK_SAMPLE_COUNT_4_BIT"]}},
                  "VkPhysicalDeviceVulkan11Properties": {"subgroupSupportedStages": ["VK_SHADER_STAGE_FRAGMENT_BIT"]},
                  "VkPhysicalDeviceUnrelatedVendorPropertiesXYZ": {"maxComputeSharedMemorySize": 1}},
                 "features": {"VkPhysicalDeviceFeatures": {"shaderInt64": true}}}"#,
    );
    let d = DeviceDescription::from_khronos_json(&text, None).unwrap();
    assert_eq!(d.limits.max_compute_shared_memory_size, 32768);
    assert_eq!(d.limits.min_storage_buffer_offset_alignment, 64);
    assert_eq!(d.limits.max_compute_work_group_size, [1024, 1024, 64]);
    assert_eq!(
        d.limits.framebuffer_color_sample_counts,
        vk::SampleCountFlags::COUNT_1 | vk::SampleCountFlags::COUNT_4
    );
    assert_eq!(
        d.subgroup.supported_stages,
        vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT
    );
    assert!(d.has(VkFeature::ShaderInt64));
    // Omitted core limits are the version's required values; other omitted maxima are zero.
    assert_eq!(d.limits.min_uniform_buffer_offset_alignment, 256);
    assert_eq!(d.limits.max_uniform_buffer_range, 16384);
    assert_eq!(d.limits.max_push_descriptors, 0);
    assert_eq!(d.allocation.max_buffer_size, 0);
}

#[test]
fn malformed_documents_are_typed_errors() {
    let bad = |text: &str| match DeviceDescription::from_khronos_json(text, None) {
        Err(Error::InvalidDescriptor(DescriptorError::DeviceDescription { .. })) => {}
        other => panic!("{text}: {other:?}"),
    };
    bad("not json");
    bad(r#"{"capabilities": {}}"#);
    bad(&doc("1.2", r#""missing""#, ""));
    bad(&doc("one.two", r#""a""#, r#""a": {}"#));
    bad(&doc("1.2.0", r#"["a", "b"]"#, r#""a": {}, "b": {}"#));
    bad(&doc(
        "1.2.0",
        r#""a""#,
        r#""a": {"properties": {"VkPhysicalDeviceProperties": {"limits": {"maxComputeWorkGroupSize": [1, 2]}}}}"#,
    ));
    let two = r#"{"capabilities": {"a": {}}, "profiles": {
        "P1": {"api-version": "1.2.0", "capabilities": ["a"]},
        "P2": {"api-version": "1.3.0", "capabilities": ["a"]}}}"#;
    bad(two);
    let d = DeviceDescription::from_khronos_json(two, Some("P2")).unwrap();
    assert_eq!(d.api_version, ApiVersion::V1_3);
    assert!(DeviceDescription::from_khronos_json(two, Some("P3")).is_err());
}

#[test]
fn required_features_are_cumulative() {
    let v12: Vec<_> = required_features(ApiVersion::V1_2).collect();
    let v13: Vec<_> = required_features(ApiVersion::V1_3).collect();
    assert!(v12.contains(&VkFeature::UniformBufferStandardLayout));
    assert!(!v12.contains(&VkFeature::SubgroupSizeControl));
    assert!(v13.starts_with(&v12));
    assert!(v13.contains(&VkFeature::SubgroupSizeControl));
}

/// With feature `serde`, every committed description and a default and a non-default policy
/// survive a JSON round trip unchanged, and the `Device` rebuilt from the decoded pair has the
/// original's profile fingerprint: what an out-of-process compile worker needs to compile for the
/// consumer's device (worker recipe step 1).
#[cfg(feature = "serde")]
#[test]
fn descriptions_and_policies_round_trip_through_serde() {
    let policies = [
        DevicePolicy::default(),
        DevicePolicy::default().exact_answers(),
    ];
    for (stem, text) in committed_profiles() {
        let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let json = serde_json::to_string(&description).unwrap();
        let decoded: DeviceDescription = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, description, "{stem}");
        for policy in policies {
            let policy_json = serde_json::to_string(&policy).unwrap();
            let decoded_policy: DevicePolicy = serde_json::from_str(&policy_json).unwrap();
            assert_eq!(decoded_policy, policy, "{stem}");
            let original = Device::new(description.clone(), policy).unwrap();
            let rebuilt = Device::new(decoded.clone(), decoded_policy).unwrap();
            assert_eq!(
                rebuilt.profile().fingerprint(),
                original.profile().fingerprint(),
                "{stem}"
            );
        }
    }
}
