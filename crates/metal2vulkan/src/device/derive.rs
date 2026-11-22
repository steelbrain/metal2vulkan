//! `Device::from_queries`: the only producer of a `DeviceProfile`.
//!
//! The profile is a pure function of the description's Vulkan queries and the policy: no driver or
//! GPU identity decides anything. Memory's A/F split is the
//! only field that selects a pointer model; every other field selects leaf encodings or feeds the
//! requirement check.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::device::description::{
    AllocationLimits, DeviceDescription, GeneratedCommandsProperties,
};
use crate::device::function_storage;
use crate::device::metal::MetalDeviceProperties;
use crate::device::policy::DevicePolicy;
use crate::device::profile::{
    BufferAccess, DescriptorCaps, DeviceProfile, EnabledCaps, FormatTable, FragmentCaps,
    FunctionStorage, HandleDomain, ImageCaps, Independence, LeafCaps, PointerDomain, ProfileFields,
    Robustness,
};
use crate::device::requirements::{
    ApiVersion, FloatControl, FloatWidth, FormatUsage, ResolveMode, SubgroupOp, VkExtension,
    VkFeature, VkLimit, VkProperty,
};
use crate::device::subgroup::{OpBits, StageSubgroup, SubgroupModel, SubgroupSizes};
use crate::error::{Capability, DescriptorError, Error, Feature, Unsupported};
use crate::pipeline::sym::Stage;
use crate::vk;

/// A device the core translates for: its profile, the Metal properties the consumer reports to
/// the guest, and the policy it was created with. Cheap to clone; `Send + Sync`.
#[derive(Clone, Debug)]
pub struct Device(Arc<DeviceInner>);

#[derive(Debug)]
struct DeviceInner {
    profile: DeviceProfile,
    /// `profile`'s fingerprint, computed once: every key and artifact check takes it.
    fingerprint: crate::cache::Hash256,
    metal: MetalDeviceProperties,
    policy: DevicePolicy,
    /// The description's `robustBufferAccessUpdateAfterBind`, which decides what
    /// `Device::required_enables` may name together. No translation reads it, so it is not a
    /// profile field: two devices that differ only in it share every artifact.
    robust_buffer_access_update_after_bind: bool,
    /// The description's `generated_commands`, which only the device replay reads; not a profile
    /// field, as above.
    generated_commands: Option<GeneratedCommandsProperties>,
    /// The description's allocation sizes, which the Metal-facing `maxBufferLength` reads; not a
    /// profile field, as above.
    allocation: AllocationLimits,
}

impl Device {
    /// The device its Vulkan queries decide, under `policy`: the profile is a function of what
    /// the description states the device supports and the consumer enabled, and of the policy.
    /// The description need not come from a device: one read from a Khronos profiles JSON document
    /// ([`DeviceDescription::from_khronos_json`]) or built field by field
    /// ([`DeviceDescription::new`]) makes a device with no `VkDevice` behind it.
    ///
    /// Refuses a device below Vulkan 1.2 (`Unsupported{VulkanVersion}`), a device whose
    /// `maxPushConstantsSize` is below the 256 bytes of the push window every stage is lowered for
    /// (`Unsupported{PushWindow}`, a `Limit` of `MaxPushConstantsSize`), a description that lists
    /// a feature neither its version nor its listed extensions expose or a feature without the
    /// feature Vulkan requires enabled with it (`InvalidDescriptor`, the invariants
    /// `device::enables` states), a policy that turns containment off while guest isolation
    /// is on (`InvalidDescriptor`), a policy that takes one identity domain without the other
    /// (`InvalidDescriptor`), and the identity domains on a device without `bufferDeviceAddress`
    /// and `shaderInt64` (`Unsupported{IdentityDomains}`).
    ///
    /// No identity decides anything: the description's `driver_id`, `driver_version`,
    /// `vendor_id`, `device_id` and `device_name` are not read. The rasterizer splits clipped
    /// coverage and the tessellator orders triangles as the device does (each module records its
    /// static note, `PrecisionNote::ClippedCoverageSplit` and
    /// `PrecisionNote::TessellationTriangleOrder`), `InterpolateAtOffset` reads from the pixel's center as GLSL.std.450 states, and a thread's
    /// Function storage is the 16 KiB floor. A two-plane YCbCr texture's planes are read in place
    /// on every device, whatever its type. Every stage output is 32-bit: a `half` color output
    /// reaches the device widened to `float`, whatever `storageInputOutput16` says.
    pub fn from_queries(desc: DeviceDescription, policy: DevicePolicy) -> Result<Device, Error> {
        if desc.api_version < ApiVersion::V1_2 {
            return Err(Unsupported::needs(
                Feature::VulkanVersion,
                Capability::ApiVersion(ApiVersion::V1_2),
            )
            .into());
        }
        let push = desc.limits.max_push_constants_size;
        if push < crate::lower::push::FLOOR_BYTES {
            return Err(Unsupported::needs(
                Feature::PushWindow,
                Capability::Limit {
                    limit: VkLimit::MaxPushConstantsSize,
                    required: u64::from(crate::lower::push::FLOOR_BYTES),
                    available: u64::from(push),
                },
            )
            .into());
        }
        crate::device::enables::check_invariants(&desc)?;
        identity_domains(&desc, &policy)?;
        let profile = DeviceProfile::of(derive_fields(&desc, &policy));
        let metal = MetalDeviceProperties::derive(&profile, &desc.allocation);
        Ok(Device(Arc::new(DeviceInner {
            fingerprint: profile.fingerprint(),
            profile,
            metal,
            policy,
            robust_buffer_access_update_after_bind: desc.robust_buffer_access_update_after_bind,
            generated_commands: desc.generated_commands,
            allocation: desc.allocation,
        })))
    }

    /// The translation-relevant profile.
    pub fn profile(&self) -> &DeviceProfile {
        &self.0.profile
    }

    /// The profile's fingerprint ([`DeviceProfile::fingerprint`]), which the profile holds: every
    /// compile key, descriptor key and artifact check takes it, and deriving it renders the whole
    /// profile, so it is derived once where the profile is built.
    pub fn profile_fingerprint(&self) -> crate::cache::Hash256 {
        self.0.fingerprint
    }

    /// The Metal device properties the consumer reports to the guest.
    pub fn metal_properties(&self) -> &MetalDeviceProperties {
        &self.0.metal
    }

    /// The policy the device was created with.
    pub fn policy(&self) -> &DevicePolicy {
        &self.0.policy
    }

    /// The description's `robustBufferAccessUpdateAfterBind`.
    pub(crate) fn robust_buffer_access_update_after_bind(&self) -> bool {
        self.0.robust_buffer_access_update_after_bind
    }

    /// `VK_EXT_device_generated_commands` as the description states the consumer enabled it: the
    /// limits and stages the device replay's generated-commands layout is held to. It is not part
    /// of [`Device::profile`], so it changes no artifact.
    pub fn generated_commands(&self) -> Option<&GeneratedCommandsProperties> {
        self.0.generated_commands.as_ref()
    }

    /// The description's allocation sizes: `maxMemoryAllocationSize` and `maxBufferSize`. They are
    /// not part of [`Device::profile`] (no translation reads them), so they change no artifact and
    /// no profile fingerprint.
    pub fn allocation(&self) -> &AllocationLimits {
        &self.0.allocation
    }
}

/// The identity domains come as a pair, and need device addresses and 64-bit integers: a
/// pointer is a 64-bit device address.
fn identity_domains(desc: &DeviceDescription, policy: &DevicePolicy) -> Result<(), Error> {
    let pointers = policy.pointer_domain == PointerDomain::Identity;
    let handles = policy.handle_domain == HandleDomain::Identity;
    if pointers != handles {
        return Err(Error::InvalidDescriptor(
            DescriptorError::DeviceDescription {
                detail: "the identity pointer and handle domains are taken together".into(),
            },
        ));
    }
    let needs = [VkFeature::BufferDeviceAddress, VkFeature::ShaderInt64];
    if pointers && !needs.iter().all(|&f| desc.has(f)) {
        let all = needs.iter().map(|&f| Capability::Feature(f)).collect();
        return Err(Unsupported::needs_any_of(Feature::IdentityDomains, vec![all]).into());
    }
    Ok(())
}

fn derive_fields(desc: &DeviceDescription, policy: &DevicePolicy) -> ProfileFields {
    let has = |f: VkFeature| desc.has(f);
    let ext = |e: VkExtension| desc.extensions.contains(&e);
    let api = desc.api_version;

    // Memory's A/F split: Profile A needs device addresses and 64-bit integers; scalar block layout
    // is required only where an emitted type needs it, so it does not select the profile.
    let buffer_access = if has(VkFeature::BufferDeviceAddress) && has(VkFeature::ShaderInt64) {
        BufferAccess::Addressed
    } else {
        BufferAccess::Floor
    };
    // The bounds of a descriptor view's access are a leaf choice from the enabled robustness.
    let robustness = if has(VkFeature::RobustBufferAccess2) {
        Robustness::Rba2
    } else if has(VkFeature::RobustBufferAccess) {
        Robustness::Rba
    } else {
        Robustness::None
    };

    let leaf = LeafCaps {
        int8: has(VkFeature::ShaderInt8),
        int16: has(VkFeature::ShaderInt16),
        int64: has(VkFeature::ShaderInt64),
        float16: has(VkFeature::ShaderFloat16),
        float64: has(VkFeature::ShaderFloat64),
        storage_buffer_8bit: has(VkFeature::StorageBuffer8BitAccess),
        storage_buffer_16bit: has(VkFeature::StorageBuffer16BitAccess),
        buffer_int64_atomics: has(VkFeature::ShaderBufferInt64Atomics),
        shared_int64_atomics: has(VkFeature::ShaderSharedInt64Atomics),
        image_int64_atomics: has(VkFeature::ShaderImageInt64Atomics),
        vertex_stores_and_atomics: has(VkFeature::VertexPipelineStoresAndAtomics),
        fragment_stores_and_atomics: has(VkFeature::FragmentStoresAndAtomics),
        zero_initialize_workgroup_memory: has(VkFeature::ShaderZeroInitializeWorkgroupMemory),
        float8: has(VkFeature::ShaderFloat8),
    };
    let images = ImageCaps {
        storage_read_without_format: has(VkFeature::ShaderStorageImageReadWithoutFormat),
        storage_write_without_format: has(VkFeature::ShaderStorageImageWriteWithoutFormat),
        storage_extended_formats: has(VkFeature::ShaderStorageImageExtendedFormats),
        storage_multisample: has(VkFeature::ShaderStorageImageMultisample),
        cube_array: has(VkFeature::ImageCubeArray),
        gather_extended: has(VkFeature::ShaderImageGatherExtended),
        dynamic_sample_offset: has(VkFeature::Maintenance8),
        min_lod: has(VkFeature::ShaderResourceMinLod),
        robust_image_access: has(VkFeature::RobustImageAccess)
            || has(VkFeature::RobustImageAccess2),
        robust_image_access2: has(VkFeature::RobustImageAccess2),
        sampler_anisotropy: has(VkFeature::SamplerAnisotropy),
        sampler_mirror_clamp_to_edge: has(VkFeature::SamplerMirrorClampToEdge),
        compute_derivative_linear: has(VkFeature::ComputeDerivativeGroupLinear),
    };
    let descriptors = DescriptorCaps {
        buffer_device_address: has(VkFeature::BufferDeviceAddress),
        scalar_block_layout: has(VkFeature::ScalarBlockLayout),
        uniform_buffer_standard_layout: has(VkFeature::UniformBufferStandardLayout),
        runtime_descriptor_array: has(VkFeature::RuntimeDescriptorArray),
        partially_bound: has(VkFeature::DescriptorBindingPartiallyBound),
        update_after_bind_sampled_image: has(
            VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
        ),
        update_after_bind_storage_image: has(
            VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
        ),
        update_after_bind_storage_buffer: has(
            VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
        ),
        update_after_bind_uniform_texel_buffer: has(
            VkFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind,
        ),
        update_after_bind_storage_texel_buffer: has(
            VkFeature::DescriptorBindingStorageTexelBufferUpdateAfterBind,
        ),
        update_unused_while_pending: has(VkFeature::DescriptorBindingUpdateUnusedWhilePending),
        non_uniform_sampled_image: has(VkFeature::ShaderSampledImageArrayNonUniformIndexing),
        non_uniform_storage_image: has(VkFeature::ShaderStorageImageArrayNonUniformIndexing),
        non_uniform_storage_buffer: has(VkFeature::ShaderStorageBufferArrayNonUniformIndexing),
        non_uniform_uniform_texel_buffer: has(
            VkFeature::ShaderUniformTexelBufferArrayNonUniformIndexing,
        ),
        non_uniform_storage_texel_buffer: has(
            VkFeature::ShaderStorageTexelBufferArrayNonUniformIndexing,
        ),
        null_descriptor: has(VkFeature::NullDescriptor),
    };
    let fragment = FragmentCaps {
        demote_to_helper: has(VkFeature::ShaderDemoteToHelperInvocation),
        terminate_invocation: has(VkFeature::ShaderTerminateInvocation),
        pixel_interlock: has(VkFeature::FragmentShaderPixelInterlock),
        barycentric: has(VkFeature::FragmentShaderBarycentric),
        stencil_export: ext(VkExtension::ExtShaderStencilExport),
        sample_rate_shading: has(VkFeature::SampleRateShading),
        dual_src_blend: has(VkFeature::DualSrcBlend),
        independent_blend: has(VkFeature::IndependentBlend),
        tessellation_shader: has(VkFeature::TessellationShader),
        // The features alone: at the 1.2 floor they replace `VK_EXT_shader_viewport_index_layer`'s
        // one capability (`ShaderViewportIndexLayerEXT`) with the `ShaderLayer` and
        // `ShaderViewportIndex` capabilities a module declares, and the core never enables the
        // extension (`device::enables`). A description that lists the extension without them
        // cannot write either builtin from a vertex function here.
        output_layer: has(VkFeature::ShaderOutputLayer),
        output_viewport_index: has(VkFeature::ShaderOutputViewportIndex),
        clip_distance: has(VkFeature::ShaderClipDistance),
        rasterization_order_color_access: has(VkFeature::RasterizationOrderColorAttachmentAccess),
        vertex_attribute_divisor: has(VkFeature::VertexAttributeInstanceRateDivisor),
    };

    let (vertex_formats, storage_formats, image_formats) = format_tables(desc);
    ProfileFields {
        buffer_access,
        pointer_domain: policy.pointer_domain,
        handle_domain: policy.handle_domain,
        robustness,
        leaf,
        subgroup: subgroup_model(desc),
        images,
        descriptors,
        fragment,
        float_controls: desc.float_controls,
        vulkan_memory_model: has(VkFeature::VulkanMemoryModel),
        draw_parameters: has(VkFeature::ShaderDrawParameters),
        limits: desc.limits,
        vertex_formats,
        storage_formats,
        image_formats,
        api_version: api,
        // The feature, which both description readers list from 1.3 (core without a member),
        // so the extended dynamic states a pipeline makes are exactly what it requires.
        dynamic_vertex_stride: has(VkFeature::ExtendedDynamicState),
        enabled: enabled_caps(desc),
        // The planes of a two-plane YCbCr texture read in place on every device.
        ycbcr_in_place: true,
        // The floor: no Vulkan query states how much a thread holds.
        function_storage: FunctionStorage::Assumed(function_storage::FLOOR),
        // Vulkan's rule, the device's own reading of the gradients handed to it.
    }
}

/// The subgroup model.
///
/// A stage is `pinnable` when subgroup size control is enabled and the stage is in
/// `requiredSubgroupSizeStages`; its size range is then `[minSubgroupSize, maxSubgroupSize]`.
/// Otherwise SPIR-V 1.5 modules run with the API-constant `subgroupSize`, which is the only size
/// the stage exposes. `full_subgroups` is compute-only (`computeFullSubgroups`).
fn subgroup_model(desc: &DeviceDescription) -> SubgroupModel {
    let sg = &desc.subgroup;
    let control = desc.has(VkFeature::SubgroupSizeControl);
    let ops = OpBits(sg.supported_operations.as_raw() & OpBits(0xff).0);
    let stage = |stage: Stage| {
        let bit = vk_stage(stage).flags();
        let pinnable = control
            && sg.required_size_stages.contains(bit)
            && sg.min_size > 0
            && sg.max_size >= sg.min_size;
        StageSubgroup {
            supported: sg.supported_stages.contains(bit),
            ops,
            api_size: sg.size,
            sizes: if pinnable {
                SubgroupSizes {
                    min: sg.min_size,
                    max: sg.max_size,
                }
            } else {
                SubgroupSizes {
                    min: sg.size,
                    max: sg.size,
                }
            },
            pinnable,
            full_subgroups: stage == Stage::Kernel && desc.has(VkFeature::ComputeFullSubgroups),
        }
    };
    SubgroupModel {
        vertex: stage(Stage::Vertex),
        fragment: stage(Stage::Fragment),
        kernel: stage(Stage::Kernel),
        quad_all_stages: sg.quad_operations_in_all_stages,
        maximal_reconvergence: desc.has(VkFeature::ShaderMaximalReconvergence),
        uniform_control_flow: desc.has(VkFeature::ShaderSubgroupUniformControlFlow),
        quad_control: desc.has(VkFeature::ShaderQuadControl),
        max_compute_workgroup_subgroups: sg.max_compute_workgroup_subgroups,
    }
}

/// Splits the description's format support into the profile's three usage-class tables. Only
/// formats with at least one usage in a class appear in that class's table.
/// The profile's format tables: the description's format properties by usage class, for the
/// formats whose enum value the description's version or extensions expose (a format property
/// query answers for extensions a device supports but the consumer did not enable). The
/// storage-without-format usages follow [`without_format`].
fn format_tables(desc: &DeviceDescription) -> (FormatTable, FormatTable, FormatTable) {
    let mut vertex = FormatTable::default();
    let mut storage = FormatTable::default();
    let mut image = FormatTable::default();
    let exposed = |format: vk::Format| {
        crate::device::enables::format_exposed(format, desc.api_version, &desc.extensions)
    };
    for (&format, props) in desc.formats.formats.iter().filter(|(&f, _)| exposed(f)) {
        for &usage in crate::device::description::ALL_FORMAT_USAGES {
            let supported = match without_format(desc, format, props, usage) {
                Some(supported) => supported,
                None => props.supports(usage),
            };
            if !supported {
                continue;
            }
            let table = match usage_class(usage) {
                UsageClass::Vertex => &mut vertex,
                UsageClass::Storage => &mut storage,
                UsageClass::Image => &mut image,
            };
            table.formats.entry(format).or_default().insert(usage);
        }
    }
    (vertex, storage, image)
}

/// Whether a storage image or storage texel buffer of `format` (`props`) is accessed without a
/// format for a without-format `usage` (`None` for every other usage):
///
/// - On a device with `VkFormatProperties3` ([`DeviceDescription::has_format_feature_flags2`]) the
///   format's `STORAGE_READ_WITHOUT_FORMAT` / `STORAGE_WRITE_WITHOUT_FORMAT` decide, in its
///   optimal-tiling features for an image and its buffer features for a texel buffer
///   (`VUID-vkCmdDispatch-OpTypeImage-07027` to `-07030`). The specification implies the image
///   bits for a format of its "Formats Without Shader Storage Format" list that supports
///   `STORAGE_IMAGE` on a device with `shaderStorageImageReadWithoutFormat` (or `Write`), so a
///   description that states no `VkFormatProperties3` keeps them; nothing implies the buffer
///   bits.
/// - Without it, the features `shaderStorageImage{Read,Write}WithoutFormat` decide, for storage
///   images and storage texel buffers alike, but "only apply to" the formats of that list (the
///   section is not version-gated): a listed format that supports `STORAGE_IMAGE` (an image) or
///   `STORAGE_TEXEL_BUFFER` (a texel buffer). Nothing reports an unlisted format's bits there.
fn without_format(
    desc: &DeviceDescription,
    format: vk::Format,
    props: &crate::device::description::FormatProperties,
    usage: FormatUsage,
) -> Option<bool> {
    use FormatUsage as U;
    let (base, feature) = match usage {
        U::StorageImageReadWithoutFormat => (
            U::StorageImage,
            VkFeature::ShaderStorageImageReadWithoutFormat,
        ),
        U::StorageImageWriteWithoutFormat => (
            U::StorageImage,
            VkFeature::ShaderStorageImageWriteWithoutFormat,
        ),
        U::StorageTexelBufferReadWithoutFormat => (
            U::StorageTexelBuffer,
            VkFeature::ShaderStorageImageReadWithoutFormat,
        ),
        U::StorageTexelBufferWriteWithoutFormat => (
            U::StorageTexelBuffer,
            VkFeature::ShaderStorageImageWriteWithoutFormat,
        ),
        _ => return None,
    };
    let implied = desc.has(feature) && props.supports(base);
    let listed = STORAGE_WITHOUT_FORMAT.contains(&format);
    Some(if desc.has_format_feature_flags2() {
        props.supports(usage) || (base == U::StorageImage && implied && listed)
    } else {
        implied && listed
    })
}

/// The Vulkan specification's "Formats Without Shader Storage Format" list (`formats.adoc`,
/// `formats-without-shader-storage-format`): the only formats `shaderStorageImageReadWithoutFormat`
/// and `shaderStorageImageWriteWithoutFormat` apply to. A device with `VkFormatProperties3` "must
/// support" the matching `VK_FORMAT_FEATURE_2_STORAGE_*_WITHOUT_FORMAT_BIT` for each of them it
/// supports `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT` for. `A8_UNORM` is listed under Vulkan 1.4
/// or `VK_KHR_maintenance5`; a format the device does not expose has no properties to read.
const STORAGE_WITHOUT_FORMAT: &[vk::Format] = &[
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_SNORM,
    vk::Format::R8G8B8A8_UINT,
    vk::Format::R8G8B8A8_SINT,
    vk::Format::R32_UINT,
    vk::Format::R32_SINT,
    vk::Format::R32_SFLOAT,
    vk::Format::R32G32_UINT,
    vk::Format::R32G32_SINT,
    vk::Format::R32G32_SFLOAT,
    vk::Format::R32G32B32A32_UINT,
    vk::Format::R32G32B32A32_SINT,
    vk::Format::R32G32B32A32_SFLOAT,
    vk::Format::R16G16B16A16_UINT,
    vk::Format::R16G16B16A16_SINT,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16_SFLOAT,
    vk::Format::B10G11R11_UFLOAT_PACK32,
    vk::Format::R16_SFLOAT,
    vk::Format::R16G16B16A16_UNORM,
    vk::Format::A2B10G10R10_UNORM_PACK32,
    vk::Format::R16G16_UNORM,
    vk::Format::R8G8_UNORM,
    vk::Format::R16_UNORM,
    vk::Format::R8_UNORM,
    vk::Format::R16G16B16A16_SNORM,
    vk::Format::R16G16_SNORM,
    vk::Format::R8G8_SNORM,
    vk::Format::R16_SNORM,
    vk::Format::R8_SNORM,
    vk::Format::R16G16_SINT,
    vk::Format::R8G8_SINT,
    vk::Format::R16_SINT,
    vk::Format::R8_SINT,
    vk::Format::A2B10G10R10_UINT_PACK32,
    vk::Format::R16G16_UINT,
    vk::Format::R8G8_UINT,
    vk::Format::R16_UINT,
    vk::Format::R8_UINT,
    vk::Format::A8_UNORM,
];

/// Which profile format table holds a usage.
pub(crate) enum UsageClass {
    Vertex,
    Storage,
    Image,
}

pub(crate) fn usage_class(usage: FormatUsage) -> UsageClass {
    match usage {
        FormatUsage::VertexBuffer => UsageClass::Vertex,
        FormatUsage::StorageImage
        | FormatUsage::StorageImageAtomic
        | FormatUsage::StorageTexelBuffer
        | FormatUsage::StorageTexelBufferAtomic
        | FormatUsage::StorageImageReadWithoutFormat
        | FormatUsage::StorageImageWriteWithoutFormat
        | FormatUsage::StorageTexelBufferReadWithoutFormat
        | FormatUsage::StorageTexelBufferWriteWithoutFormat => UsageClass::Storage,
        FormatUsage::UniformTexelBuffer
        | FormatUsage::SampledImage
        | FormatUsage::SampledImageFilterLinear
        | FormatUsage::SampledImageFilterMinmax
        | FormatUsage::ColorAttachment
        | FormatUsage::ColorAttachmentBlend
        | FormatUsage::DepthStencilAttachment => UsageClass::Image,
    }
}

fn enabled_caps(desc: &DeviceDescription) -> EnabledCaps {
    let sg = &desc.subgroup;
    let mut properties = BTreeSet::new();
    for (stage, bit) in STAGE_BITS {
        if sg.supported_stages.contains(bit.flags()) {
            properties.insert(VkProperty::SubgroupStage(stage));
        }
        if desc.has(VkFeature::SubgroupSizeControl) && sg.required_size_stages.contains(bit.flags())
        {
            properties.insert(VkProperty::RequiredSubgroupSizeStage(stage));
        }
    }
    for op in SUBGROUP_OPS {
        if OpBits(sg.supported_operations.as_raw()).contains(OpBits::of(op)) {
            properties.insert(VkProperty::SubgroupOperation(op));
        }
    }
    if sg.quad_operations_in_all_stages {
        properties.insert(VkProperty::SubgroupQuadOperationsInAllStages);
    }
    let fc = &desc.float_controls;
    for (width, w) in [
        (FloatWidth::F16, &fc.f16),
        (FloatWidth::F32, &fc.f32),
        (FloatWidth::F64, &fc.f64),
    ] {
        for (control, on) in [
            (FloatControl::DenormPreserve, w.denorm_preserve),
            (FloatControl::DenormFlushToZero, w.denorm_flush_to_zero),
            (FloatControl::RoundingModeRte, w.rounding_rte),
            (FloatControl::RoundingModeRtz, w.rounding_rtz),
            (
                FloatControl::SignedZeroInfNanPreserve,
                w.signed_zero_inf_nan_preserve,
            ),
        ] {
            if on {
                properties.insert(VkProperty::FloatControl(control, width));
            }
        }
    }
    for (independence, [partial, all]) in [
        (
            fc.denorm_independence,
            [
                VkProperty::DenormIndependence32BitOnly,
                VkProperty::DenormIndependenceAll,
            ],
        ),
        (
            fc.rounding_independence,
            [
                VkProperty::RoundingIndependence32BitOnly,
                VkProperty::RoundingIndependenceAll,
            ],
        ),
    ] {
        if independence.admits(Independence::Only32Bit) {
            properties.insert(partial);
        }
        if independence.admits(Independence::All) {
            properties.insert(all);
        }
    }
    if desc.filter_minmax_single_component_formats {
        properties.insert(VkProperty::FilterMinmaxSingleComponentFormats);
    }
    if desc.rgba4_opaque_black_swizzled {
        properties.insert(VkProperty::Rgba4OpaqueBlackSwizzled);
    }
    // `InterpolateAtOffset` reads from the pixel's center, as GLSL.std.450 states, and the device
    // samples natively.
    properties.insert(VkProperty::InterpolateAtOffsetCentered);
    properties.insert(VkProperty::DeviceSampling);
    let l = &desc.limits;
    let sample_counts: [(vk::SampleCountFlags, fn(u32) -> VkProperty); 6] = [
        (
            l.framebuffer_color_sample_counts,
            VkProperty::FramebufferColorSampleCount,
        ),
        (
            l.framebuffer_integer_color_sample_counts,
            VkProperty::FramebufferIntegerColorSampleCount,
        ),
        (
            l.framebuffer_depth_sample_counts,
            VkProperty::FramebufferDepthSampleCount,
        ),
        (
            l.framebuffer_stencil_sample_counts,
            VkProperty::FramebufferStencilSampleCount,
        ),
        (
            l.framebuffer_no_attachments_sample_counts,
            VkProperty::FramebufferNoAttachmentsSampleCount,
        ),
        (
            l.storage_image_sample_counts,
            VkProperty::StorageImageSampleCount,
        ),
    ];
    for (flags, property) in sample_counts {
        for count in (0..7).map(|shift| 1u32 << shift) {
            if flags.contains(vk::SampleCountFlags::from_raw(count)) {
                properties.insert(property(count));
            }
        }
    }
    // Each color attachment format's own sample counts, where the description states them.
    for (&format, p) in &desc.formats.formats {
        for count in (0..7).map(|shift| 1u32 << shift) {
            if p.color_attachment_sample_counts
                .contains(vk::SampleCountFlags::from_raw(count))
            {
                properties.insert(VkProperty::ColorAttachmentSampleCount(format, count));
            }
        }
    }
    // The depth/stencil resolve properties (Vulkan 1.2).
    for mode in ResolveMode::ALL {
        if l.supported_depth_resolve_modes.contains(mode.flag()) {
            properties.insert(VkProperty::DepthResolveMode(mode));
        }
        if l.supported_stencil_resolve_modes.contains(mode.flag()) {
            properties.insert(VkProperty::StencilResolveMode(mode));
        }
    }
    if l.independent_resolve_none {
        properties.insert(VkProperty::IndependentResolveNone);
    }
    if l.independent_resolve {
        properties.insert(VkProperty::IndependentResolve);
    }
    EnabledCaps {
        features: desc.features.clone(),
        extensions: desc.extensions.clone(),
        properties,
    }
}

/// The Metal stages and their Vulkan stages.
pub(crate) const STAGE_BITS: [(Stage, vk::ShaderStage); 3] = [
    (Stage::Vertex, vk::ShaderStage::VERTEX),
    (Stage::Fragment, vk::ShaderStage::FRAGMENT),
    (Stage::Kernel, vk::ShaderStage::COMPUTE),
];

/// The Vulkan stage of a Metal stage.
pub fn vk_stage(stage: Stage) -> vk::ShaderStage {
    match stage {
        Stage::Vertex => vk::ShaderStage::VERTEX,
        Stage::Fragment => vk::ShaderStage::FRAGMENT,
        Stage::Kernel => vk::ShaderStage::COMPUTE,
        Stage::Object => vk::ShaderStage::TASK_EXT,
        Stage::Mesh => vk::ShaderStage::MESH_EXT,
    }
}

/// Every subgroup operation class the core names.
pub(crate) const SUBGROUP_OPS: [SubgroupOp; 8] = [
    SubgroupOp::Basic,
    SubgroupOp::Vote,
    SubgroupOp::Arithmetic,
    SubgroupOp::Ballot,
    SubgroupOp::Shuffle,
    SubgroupOp::ShuffleRelative,
    SubgroupOp::Clustered,
    SubgroupOp::Quad,
];
