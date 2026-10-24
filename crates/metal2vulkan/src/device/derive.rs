//! `Device::new`: the only producer of a `DeviceProfile`.
//!
//! The profile is a pure function of the description and the policy. Memory's A/F split is the
//! only field that selects a pointer model; every other field selects leaf encodings or feeds the
//! requirement check.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::device::description::{
    AllocationLimits, DeviceDescription, GeneratedCommandsProperties, APPLE_VENDOR_ID,
};
use crate::device::driver_facts::Standing;
use crate::device::function_storage;
use crate::device::metal::MetalDeviceProperties;
use crate::device::policy::{
    ContainmentPolicy, DevicePolicy, FloatMathsPolicy, HalfRoundingPolicy, NanMinMaxPolicy,
    RoundingOrderPolicy, SamplingPolicy, Unorm8LinearFilterPolicy, YCbCrFilterPolicy,
};
use crate::device::profile::{
    BufferAccess, Containment, CubeGradientRoute, CubeGradients, DescriptorCaps, DeviceProfile,
    DriverWorkaround, EnabledCaps, FloatFunctions, FormatTable, FragmentCaps, GuestIsolation,
    HalfRoundingRoute, ImageCaps, Independence, LeafCaps, MinMaxRoute, NanEncoding, ProfileFields,
    Robustness, RoundingOrderRoute, TexelNanRoute,
};
use crate::device::requirements::{
    ApiVersion, FloatControl, FloatWidth, FormatUsage, ResolveMode, SubgroupOp, VkExtension,
    VkFeature, VkProperty,
};
use crate::device::subgroup::{OpBits, StageSubgroup, SubgroupModel, SubgroupSizes};
use crate::error::{Capability, DescriptorError, DriverDeviation, Error, Feature, Unsupported};
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
    /// The description's allocation sizes, which the Metal-facing `maxBufferLength` and the
    /// runtime's heap sizes read; not a profile field, as above.
    allocation: AllocationLimits,
    /// The description the device was made from, kept so a compile over a device's compiler word
    /// bound can rebuild the device under a reduced policy (`Device::fit_render_variant` and the
    /// other kinds' fits); not a profile field, as above.
    description: DeviceDescription,
}

impl Device {
    /// Derives the device profile from what the consumer enabled and its policy.
    ///
    /// Refuses a device below Vulkan 1.2 (`Unsupported{VulkanVersion}`), a description that lists
    /// a feature neither its version nor its listed extensions expose or a feature without the
    /// feature Vulkan requires enabled with it (`InvalidDescriptor`, the invariants
    /// `device::enables` states), and a policy that turns containment off while guest isolation
    /// is on (`InvalidDescriptor`).
    pub fn new(desc: DeviceDescription, policy: DevicePolicy) -> Result<Device, Error> {
        // The driver's measured deviations at its release, never its name.
        let deviations = super::driver_facts::deviations(&desc);
        Device::with_deviations(desc, policy, deviations)
    }

    /// [`Device::new`] with `deviation` taken (at its measured standing where the driver's
    /// release has it) or not, and the release's other deviations as measured: a test's way to
    /// take one route alone, which no driver release in the table gives.
    #[cfg(test)]
    pub(crate) fn with_deviation(
        desc: DeviceDescription,
        policy: DevicePolicy,
        deviation: DriverDeviation,
        taken: bool,
    ) -> Result<Device, Error> {
        let mut deviations = super::driver_facts::deviations(&desc);
        match taken {
            true => {
                deviations.entry(deviation).or_insert(Standing::Measured);
            }
            false => {
                deviations.remove(&deviation);
            }
        }
        Device::with_deviations(desc, policy, deviations)
    }

    fn with_deviations(
        desc: DeviceDescription,
        policy: DevicePolicy,
        deviations: BTreeMap<DriverDeviation, Standing>,
    ) -> Result<Device, Error> {
        if desc.api_version < ApiVersion::V1_2 {
            return Err(Unsupported::needs(
                Feature::VulkanVersion,
                Capability::ApiVersion(ApiVersion::V1_2),
            )
            .into());
        }
        crate::device::enables::check_invariants(&desc)?;
        if policy.containment == ContainmentPolicy::UnsafeOff
            && policy.isolation == GuestIsolation::On
        {
            return Err(Error::InvalidDescriptor(
                DescriptorError::DeviceDescription {
                    detail: "containment UnsafeOff contradicts GuestIsolation::On".into(),
                },
            ));
        }
        let deviations = accepted(deviations, &policy);
        let fields = derive_fields(&desc, &policy, &deviations);
        let profile = DeviceProfile { fields };
        let metal = MetalDeviceProperties::derive(&profile, &desc.allocation);
        Ok(Device(Arc::new(DeviceInner {
            fingerprint: profile.fingerprint(),
            profile,
            metal,
            policy,
            robust_buffer_access_update_after_bind: desc.robust_buffer_access_update_after_bind,
            generated_commands: desc.generated_commands,
            allocation: desc.allocation,
            description: desc,
        })))
    }

    /// The translation-relevant profile.
    pub fn profile(&self) -> &DeviceProfile {
        &self.0.profile
    }

    /// The profile's fingerprint ([`DeviceProfile::fingerprint`]), computed once when the device
    /// was made: every compile key, descriptor key and artifact check takes it, and computing it
    /// renders the whole profile.
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

    /// The description the device was made from, for a compile that rebuilds the device under a
    /// reduced policy to fit a compiler word bound (`Device::fit_render_variant` and the other
    /// kinds' fits).
    pub(crate) fn description(&self) -> &DeviceDescription {
        &self.0.description
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

/// The measured `deviations` the profile takes under `policy`: each, but the 8-bit linear
/// filter's where the policy keeps Metal's filter exact (`Unorm8LinearFilterPolicy::Exact`, the
/// default), so the translation emulates it as on an unmeasured device.
fn accepted(
    mut deviations: BTreeMap<DriverDeviation, Standing>,
    policy: &DevicePolicy,
) -> BTreeMap<DriverDeviation, Standing> {
    if policy.unorm8_linear_filter == Unorm8LinearFilterPolicy::Exact {
        deviations.remove(&DriverDeviation::Unorm8LinearFilter);
    }
    deviations
}

fn derive_fields(
    desc: &DeviceDescription,
    policy: &DevicePolicy,
    deviations: &BTreeMap<DriverDeviation, Standing>,
) -> ProfileFields {
    let has = |f: VkFeature| desc.has(f);
    let ext = |e: VkExtension| desc.extensions.contains(&e);
    let api = desc.api_version;
    let v1_4 = api >= ApiVersion { major: 1, minor: 4 };

    // Memory's A/F split: Profile A needs device addresses and 64-bit integers; scalar block layout
    // is required only where an emitted type needs it, so it does not select the profile.
    let buffer_access = if has(VkFeature::BufferDeviceAddress) && has(VkFeature::ShaderInt64) {
        BufferAccess::Addressed
    } else {
        BufferAccess::Floor
    };
    // The containment encoding is a leaf choice from the enabled robustness.
    let containment = match policy.containment {
        ContainmentPolicy::UnsafeOff => Containment::UnsafeOff,
        ContainmentPolicy::Contained => Containment::On {
            robust: if has(VkFeature::RobustBufferAccess2) {
                Robustness::Rba2
            } else if has(VkFeature::RobustBufferAccess) {
                Robustness::Rba
            } else {
                Robustness::None
            },
        },
    };

    let leaf = LeafCaps {
        int8: has(VkFeature::ShaderInt8),
        int16: has(VkFeature::ShaderInt16),
        int64: has(VkFeature::ShaderInt64),
        float16: has(VkFeature::ShaderFloat16),
        float64: has(VkFeature::ShaderFloat64),
        storage_buffer_8bit: has(VkFeature::StorageBuffer8BitAccess),
        uniform_buffer_8bit: has(VkFeature::UniformAndStorageBuffer8BitAccess),
        push_constant_8bit: has(VkFeature::StoragePushConstant8),
        storage_buffer_16bit: has(VkFeature::StorageBuffer16BitAccess),
        uniform_buffer_16bit: has(VkFeature::UniformAndStorageBuffer16BitAccess),
        push_constant_16bit: has(VkFeature::StoragePushConstant16),
        input_output_16bit: has(VkFeature::StorageInputOutput16),
        workgroup_explicit_layout: has(VkFeature::WorkgroupMemoryExplicitLayout),
        workgroup_explicit_layout_8bit: has(VkFeature::WorkgroupMemoryExplicitLayout8BitAccess),
        workgroup_explicit_layout_16bit: has(VkFeature::WorkgroupMemoryExplicitLayout16BitAccess),
        buffer_int64_atomics: has(VkFeature::ShaderBufferInt64Atomics),
        shared_int64_atomics: has(VkFeature::ShaderSharedInt64Atomics),
        buffer_float32_atomic_add: has(VkFeature::ShaderBufferFloat32AtomicAdd),
        buffer_float32_atomic_min_max: has(VkFeature::ShaderBufferFloat32AtomicMinMax),
        shared_float32_atomic_add: has(VkFeature::ShaderSharedFloat32AtomicAdd),
        shared_float32_atomic_min_max: has(VkFeature::ShaderSharedFloat32AtomicMinMax),
        image_float32_atomic_add: has(VkFeature::ShaderImageFloat32AtomicAdd),
        image_float32_atomic_min_max: has(VkFeature::ShaderImageFloat32AtomicMinMax),
        image_int64_atomics: has(VkFeature::ShaderImageInt64Atomics),
        vertex_stores_and_atomics: has(VkFeature::VertexPipelineStoresAndAtomics),
        fragment_stores_and_atomics: has(VkFeature::FragmentStoresAndAtomics),
        zero_initialize_workgroup_memory: has(VkFeature::ShaderZeroInitializeWorkgroupMemory),
        fma32: has(VkFeature::ShaderFmaFloat32),
        fma16: has(VkFeature::ShaderFmaFloat16),
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
        sampler_filter_minmax: has(VkFeature::SamplerFilterMinmax),
        sampler_mirror_clamp_to_edge: has(VkFeature::SamplerMirrorClampToEdge),
        compute_derivative_quads: has(VkFeature::ComputeDerivativeGroupQuads),
        compute_derivative_linear: has(VkFeature::ComputeDerivativeGroupLinear),
    };
    let descriptors = DescriptorCaps {
        buffer_device_address: has(VkFeature::BufferDeviceAddress),
        scalar_block_layout: has(VkFeature::ScalarBlockLayout),
        uniform_buffer_standard_layout: has(VkFeature::UniformBufferStandardLayout),
        runtime_descriptor_array: has(VkFeature::RuntimeDescriptorArray),
        partially_bound: has(VkFeature::DescriptorBindingPartiallyBound),
        variable_descriptor_count: has(VkFeature::DescriptorBindingVariableDescriptorCount),
        update_after_bind_sampled_image: has(
            VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
        ),
        update_after_bind_storage_image: has(
            VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
        ),
        update_after_bind_storage_buffer: has(
            VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
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
        // Push descriptors are unconditionally required by Vulkan 1.4 (vk.xml VK_VERSION_1_4).
        push_descriptor: ext(VkExtension::KhrPushDescriptor) || v1_4,
        null_descriptor: has(VkFeature::NullDescriptor),
    };
    let fragment = FragmentCaps {
        demote_to_helper: has(VkFeature::ShaderDemoteToHelperInvocation),
        terminate_invocation: has(VkFeature::ShaderTerminateInvocation),
        pixel_interlock: has(VkFeature::FragmentShaderPixelInterlock),
        sample_interlock: has(VkFeature::FragmentShaderSampleInterlock),
        barycentric: has(VkFeature::FragmentShaderBarycentric),
        stencil_export: ext(VkExtension::ExtShaderStencilExport),
        sample_rate_shading: has(VkFeature::SampleRateShading),
        dual_src_blend: has(VkFeature::DualSrcBlend),
        independent_blend: has(VkFeature::IndependentBlend),
        geometry_shader: has(VkFeature::GeometryShader),
        tessellation_shader: has(VkFeature::TessellationShader),
        // The features alone: at the 1.2 floor they replace `VK_EXT_shader_viewport_index_layer`'s
        // one capability (`ShaderViewportIndexLayerEXT`) with the `ShaderLayer` and
        // `ShaderViewportIndex` capabilities a module declares, and the core never enables the
        // extension (`device::enables`). A description that lists the extension without them
        // cannot write either builtin from a vertex function here.
        output_layer: has(VkFeature::ShaderOutputLayer),
        output_viewport_index: has(VkFeature::ShaderOutputViewportIndex),
        clip_distance: has(VkFeature::ShaderClipDistance),
        cull_distance: has(VkFeature::ShaderCullDistance),
        rasterization_order_color_access: has(VkFeature::RasterizationOrderColorAttachmentAccess),
        dynamic_rendering_local_read: has(VkFeature::DynamicRenderingLocalRead),
        vertex_attribute_divisor: has(VkFeature::VertexAttributeInstanceRateDivisor),
        vertex_attribute_zero_divisor: has(VkFeature::VertexAttributeInstanceRateZeroDivisor),
    };

    let (vertex_formats, storage_formats, image_formats) = format_tables(desc);
    let deviates = |d: DriverDeviation| deviations.contains_key(&d);
    ProfileFields {
        buffer_access,
        pointer_domain: policy.pointer_domain,
        handle_domain: policy.handle_domain,
        containment,
        isolation: policy.isolation,
        texel_conversion: policy.texel_conversion,
        half_blend: policy.half_blend,
        half_texel_bits: policy.half_texel_bits,
        // Metal's where a form is measured to be Metal's or the policy asks for Metal's rounding;
        // elsewhere the device's own.
        half_rounding: if policy.half_rounding == HalfRoundingPolicy::Exact
            || (deviates(DriverDeviation::MetalHalfConversion) && leaf.float16)
            || deviates(DriverDeviation::MetalHalfQuantize)
        {
            HalfRoundingRoute::Metal
        } else {
            HalfRoundingRoute::Native
        },
        leaf,
        subgroup: subgroup_model(desc, deviates(DriverDeviation::MetalVertexSimdgroups)),
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
        workarounds: workarounds(desc),
        api_version: api,
        // The feature, which both description readers list from 1.3 (core without a member),
        // so the extended dynamic states a pipeline makes are exactly what it requires.
        dynamic_vertex_stride: has(VkFeature::ExtendedDynamicState),
        enabled: enabled_caps(desc, deviations, policy),
        // The measured deviation; unmeasured, the translation's own functions. Elsewhere native
        // device maths unless the policy asks for Metal's answers.
        float_functions: if deviates(DriverDeviation::MetalFloatFunctions) {
            FloatFunctions::Metal
        } else if policy.float_maths == FloatMathsPolicy::Exact {
            FloatFunctions::Device
        } else {
            FloatFunctions::Native
        },
        // The measured deviation; unmeasured, the device's NaN where the policy takes it, and the
        // module selects Metal's NaN itself where it asks for Metal's.
        nan_encoding: if deviates(DriverDeviation::MetalCanonicalNans) {
            NanEncoding::MetalCanonical
        } else if policy.nan_and_min_max == NanMinMaxPolicy::Exact {
            NanEncoding::Device
        } else {
            NanEncoding::Native
        },
        // Metal's NaN of a texel where the policy asks for Metal's, on every device: no device
        // fact decides it.
        texel_nans: if policy.nan_and_min_max == NanMinMaxPolicy::Exact {
            TexelNanRoute::Metal
        } else {
            TexelNanRoute::Device
        },
        // Metal's filter of a two-plane YCbCr texture through the slot's variant where the policy
        // asks for it: on a device that samples through Metal's own sampler, under the YCbCr
        // filter's switch or Metal's sampling; on one that samples natively, under the switch. A
        // device that emulates Metal's sampler answers it in every module.
        ycbcr_variant: match deviates(DriverDeviation::MetalSampler) {
            true => {
                policy.ycbcr_filter == YCbCrFilterPolicy::Exact
                    || policy.sampling == SamplingPolicy::Exact
            }
            false => {
                policy.sampling == SamplingPolicy::NativeDevice
                    && policy.ycbcr_filter == YCbCrFilterPolicy::Exact
            }
        },
        // Metal's picks where the device's minimum and maximum are measured to be Metal's or its
        // functions are Metal's own; elsewhere the device's where the policy takes them.
        min_max: if deviates(DriverDeviation::MetalMinMax)
            || deviates(DriverDeviation::MetalFloatFunctions)
            || policy.nan_and_min_max == NanMinMaxPolicy::Exact
        {
            MinMaxRoute::Metal
        } else {
            MinMaxRoute::Native
        },
        // Metal's order where the device's functions are Metal's own; elsewhere the device's
        // instructions where the policy takes them.
        rounding_order: if deviates(DriverDeviation::MetalFloatFunctions)
            || policy.rounding_order == RoundingOrderPolicy::Exact
        {
            RoundingOrderRoute::Metal
        } else {
            RoundingOrderRoute::Native
        },
        // The device's measured value, keyed on its identity as the workarounds are, or the floor.
        function_storage: function_storage::of(desc),
        // The measured deviations, one per pipeline kind (the graphics one per macOS release's
        // reading of SPIRV-Cross's fixup). Unmeasured, a MoltenVK driver's route is unknown, since
        // it hands the gradients to Metal by some route; another driver's is Vulkan's rule.
        cube_gradients: {
            let route = |measured: &[(DriverDeviation, CubeGradientRoute)]| match (
                measured.iter().find(|(deviation, _)| deviates(*deviation)),
                desc.driver_id,
            ) {
                (Some(&(_, route)), _) => route,
                (None, Some(vk::DriverId::MOLTENVK)) => CubeGradientRoute::Unmeasured,
                (None, _) => CubeGradientRoute::Vulkan,
            };
            CubeGradients {
                graphics: route(&[
                    (
                        DriverDeviation::GraphicsCubeGradientRoute,
                        CubeGradientRoute::AppleSwizzled,
                    ),
                    (
                        DriverDeviation::GraphicsCubeGradientTangents,
                        CubeGradientRoute::AppleProjected,
                    ),
                ]),
                compute: route(&[(DriverDeviation::CubeGradientRoute, CubeGradientRoute::Metal)]),
            }
        },
        // The deviations taken past their last measured release, which a module records
        // `PrecisionNote::BeyondMeasuredDeviation` for (the coordinator's ruling of 2026-10-06).
        deviations_beyond: deviations
            .iter()
            .filter_map(|(&deviation, &standing)| match standing {
                Standing::Beyond { last } => Some((deviation, last)),
                Standing::Measured => None,
            })
            .collect(),
    }
}

/// The driver workarounds of the driver `desc` names (`driver_id`), on its GPU (`vendor_id`, or its
/// generation's `device_id`s) where the defect is the GPU's, and those its release has by the
/// measured table (`driver_facts::DEFECTS`), in order.
fn workarounds(desc: &DeviceDescription) -> Vec<DriverWorkaround> {
    let mut found = by_identity(desc);
    // Every rasterizer but Apple's hands a clipped primitive's fragments their coverage by the
    // whole primitive (or, lavapipe, by its own pieces), never by Apple's pieces; and every
    // tessellator but Apple's orders a patch's triangles its own way.
    if !(desc.driver_id == Some(vk::DriverId::MOLTENVK) && desc.vendor_id == APPLE_VENDOR_ID) {
        found.push(DriverWorkaround::ClippedCoverageNotMetals);
        found.push(DriverWorkaround::TessellatorNotMetals);
    }
    found.extend(super::driver_facts::defects(desc));
    found.sort();
    found.dedup();
    found
}

/// The workarounds of [`workarounds`] keyed on the driver's identity alone.
fn by_identity(desc: &DeviceDescription) -> Vec<DriverWorkaround> {
    match desc.driver_id {
        Some(vk::DriverId::GOOGLE_SWIFTSHADER) => vec![DriverWorkaround::BufferAtomicsUncontained],
        Some(vk::DriverId::MOLTENVK) if desc.vendor_id == APPLE_VENDOR_ID => vec![
            DriverWorkaround::EarlyTestsDropMultisampledImageWrites,
            DriverWorkaround::DiscardDropsSimdgroupStores,
        ],
        // The uniform buffer's reads, on 128-bit vectors (`subgroupSize` 4) alone.
        Some(vk::DriverId::MESA_LLVMPIPE) if desc.subgroup.size == 4 => vec![
            DriverWorkaround::ClippedTrianglesSplitPixelFragments,
            DriverWorkaround::UniformReadsOutpaceStorageReads,
        ],
        Some(vk::DriverId::MESA_LLVMPIPE) => {
            vec![DriverWorkaround::ClippedTrianglesSplitPixelFragments]
        }
        // A `half` round trip folded on every GPU it drives, and on the 3090's generation a
        // narrow integer's mask dropped.
        Some(vk::DriverId::NVIDIA_PROPRIETARY) => {
            let mut found = vec![DriverWorkaround::HalfConversionPairFolded];
            if super::NVIDIA_AMPERE_GA10X.contains(&desc.device_id) {
                found.push(DriverWorkaround::NarrowMaskAfterSignedClamp);
            }
            found
        }
        // A uniform buffer's reads at different addresses, and a register allocator that faults
        // compiling a module past a word bound, on every GPU it drives.
        Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA) => vec![
            DriverWorkaround::UniformReadsOutpaceStorageReads,
            DriverWorkaround::RegisterAllocatorOverflowsLargeModules,
        ],
        _ => Vec::new(),
    }
}

/// The operation classes measured exact in a vertex function on a driver that takes
/// `DriverDeviation::MetalVertexSimdgroups`: with the vertex stage forced into the profile,
/// MoltenVK 1.4.1 ran `rvc_simd_vertex`'s `simd_sum`, `simd_shuffle`, `simd_shuffle_xor` and
/// `simd_ballot` as Metal does (200 of 200 words on the M4 and the M3 Max), and with the votes
/// added, `gsm2_vertex_votes_shifts`' `simd_all` and `simd_any` over full and partial simdgroups
/// (the four cases matching in both legs on both chips).
///
/// The clustered class stays out: with it added, MoltenVK 1.4.1 refused `gsm2_quad_ops`' vertex
/// functions, whose quad reductions then lowered to it ("Subgroup builtins are not available in
/// this type of function", SPIRV-Cross's MSL conversion). They now take the ballot
/// and shuffles, so no quad or simdgroup operation of a vertex function on the `Direct` tier needs
/// it.
const MEASURED_VERTEX_OPS: OpBits = OpBits(
    OpBits::BASIC.0 | OpBits::VOTE.0 | OpBits::ARITHMETIC.0 | OpBits::BALLOT.0 | OpBits::SHUFFLE.0,
);

/// The subgroup model.
///
/// A stage is `pinnable` when subgroup size control is enabled and the stage is in
/// `requiredSubgroupSizeStages`; its size range is then `[minSubgroupSize, maxSubgroupSize]`.
/// Otherwise SPIR-V 1.5 modules run with the API-constant `subgroupSize`, which is the only size
/// the stage exposes. `full_subgroups` is compute-only (`computeFullSubgroups`).
///
/// Where the driver is measured to run a vertex function's subgroup operations as Metal does
/// though `subgroupSupportedStages` leaves the vertex stage out (`vertex_simdgroups`,
/// `DriverDeviation::MetalVertexSimdgroups`), the vertex stage has subgroups of the device's
/// operations that were measured exact there ([`MEASURED_VERTEX_OPS`]).
fn subgroup_model(desc: &DeviceDescription, vertex_simdgroups: bool) -> SubgroupModel {
    let sg = &desc.subgroup;
    let control = desc.has(VkFeature::SubgroupSizeControl);
    let ops = OpBits(sg.supported_operations.as_raw() & OpBits(0xff).0);
    let stage = |stage: Stage| {
        let bit = vk_stage(stage).flags();
        let pinnable = control
            && sg.required_size_stages.contains(bit)
            && sg.min_size > 0
            && sg.max_size >= sg.min_size;
        let advertised = sg.supported_stages.contains(bit);
        let measured = !advertised && vertex_simdgroups && stage == Stage::Vertex;
        StageSubgroup {
            supported: advertised || measured,
            ops: if measured {
                OpBits(ops.0 & MEASURED_VERTEX_OPS.0)
            } else {
                ops
            },
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

fn enabled_caps(
    desc: &DeviceDescription,
    deviations: &BTreeMap<DriverDeviation, Standing>,
    policy: &DevicePolicy,
) -> EnabledCaps {
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
    // Where `InterpolateAtOffset` reads, and the measured deviations.
    if desc.interpolates_at_offset_centered() {
        properties.insert(VkProperty::InterpolateAtOffsetCentered);
    }
    if deviations.contains_key(&DriverDeviation::SimdgroupLockstep) {
        properties.insert(VkProperty::SimdgroupLockstep);
    }
    if deviations.contains_key(&DriverDeviation::MetalSampler) {
        properties.insert(VkProperty::MetalSampler);
    } else if policy.sampling == SamplingPolicy::NativeDevice {
        // Native device sampling, the policy's default, where the device's sampler is not Metal's.
        properties.insert(VkProperty::DeviceSampling);
    }
    if deviations.contains_key(&DriverDeviation::MetalColorWrites) {
        properties.insert(VkProperty::MetalColorWrites);
    }
    if deviations.contains_key(&DriverDeviation::MetalFloatColorNans) {
        properties.insert(VkProperty::MetalFloatColorNans);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfColorNans) {
        properties.insert(VkProperty::MetalHalfColorNans);
    }
    if deviations.contains_key(&DriverDeviation::HalfColorOutputBits) {
        properties.insert(VkProperty::HalfColorOutputBits);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfFma) {
        properties.insert(VkProperty::MetalHalfFma);
    }
    if deviations.contains_key(&DriverDeviation::MetalFloatDenormals) {
        properties.insert(VkProperty::MetalFloatDenormals);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfQuantize) {
        properties.insert(VkProperty::MetalHalfQuantize);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfConversion) {
        properties.insert(VkProperty::MetalHalfConversion);
    }
    if deviations.contains_key(&DriverDeviation::MetalCenterVaryings) {
        properties.insert(VkProperty::MetalCenterVaryings);
    }
    if deviations.contains_key(&DriverDeviation::MetalSmallFloatColorWrites) {
        properties.insert(VkProperty::MetalSmallFloatColorWrites);
    }
    if deviations.contains_key(&DriverDeviation::MetalNorm16ColorWrites) {
        properties.insert(VkProperty::MetalNorm16ColorWrites);
    }
    if deviations.contains_key(&DriverDeviation::MetalMinMax) {
        properties.insert(VkProperty::MetalMinMax);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfArithmetic) {
        properties.insert(VkProperty::MetalHalfArithmetic);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfStorageWrites) {
        properties.insert(VkProperty::MetalHalfStorageWrites);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfStorageNans) {
        properties.insert(VkProperty::MetalHalfStorageNans);
    }
    if deviations.contains_key(&DriverDeviation::MetalFloatStorageNans) {
        properties.insert(VkProperty::MetalFloatStorageNans);
    }
    if deviations.contains_key(&DriverDeviation::MetalFragmentSimdgroups) {
        properties.insert(VkProperty::MetalFragmentSimdgroups);
    }
    if deviations.contains_key(&DriverDeviation::MetalVertexSimdgroups) {
        properties.insert(VkProperty::MetalVertexSimdgroups);
    }
    if deviations.contains_key(&DriverDeviation::CenterFragCoordDepth) {
        properties.insert(VkProperty::CenterFragCoordDepth);
    }
    if deviations.contains_key(&DriverDeviation::SampleFragCoordDepth) {
        properties.insert(VkProperty::SampleFragCoordDepth);
    }
    if deviations.contains_key(&DriverDeviation::MetalHalfTexelWrites) {
        properties.insert(VkProperty::MetalHalfTexelWrites);
    }
    if deviations.contains_key(&DriverDeviation::MetalNormStorageWrites) {
        properties.insert(VkProperty::MetalNormStorageWrites);
    }
    if deviations.contains_key(&DriverDeviation::Unorm8LinearFilter) {
        properties.insert(VkProperty::Unorm8LinearFilter);
    }
    if deviations.contains_key(&DriverDeviation::MetalDepth16Stores) {
        properties.insert(VkProperty::MetalDepth16Stores);
    }
    if deviations.contains_key(&DriverDeviation::MetalDepth16BiasedStores) {
        properties.insert(VkProperty::MetalDepth16BiasedStores);
    }
    if deviations.contains_key(&DriverDeviation::MetalInterpolation) {
        properties.insert(VkProperty::MetalInterpolation);
    }
    if deviations.contains_key(&DriverDeviation::MetalFirstVertexIndices) {
        properties.insert(VkProperty::MetalFirstVertexIndices);
    }
    if deviations.contains_key(&DriverDeviation::MetalFullQuads) {
        properties.insert(VkProperty::MetalFullQuads);
    }
    if desc
        .driver_facts()
        .contains(&crate::device::DriverFact::DepthImageKeepsValuesOutsideUnitRange)
    {
        properties.insert(VkProperty::DepthImageKeepsValuesOutsideUnitRange);
    }
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
    // The provoking-vertex property holds only where the extension is listed (after
    // `with_enabled`, enabled).
    if desc.extensions.contains(&VkExtension::ExtProvokingVertex)
        && l.provoking_vertex_mode_per_pipeline
    {
        properties.insert(VkProperty::ProvokingVertexModePerPipeline);
    }
    // The sample-location properties hold only where the extension is listed (after
    // `with_enabled`, enabled): a device that has not enabled it sets no locations.
    if desc.extensions.contains(&VkExtension::ExtSampleLocations) {
        if l.variable_sample_locations {
            properties.insert(VkProperty::VariableSampleLocations);
        }
        for count in (0..7).map(|shift| 1u32 << shift) {
            if l.sample_location_sample_counts
                .contains(vk::SampleCountFlags::from_raw(count))
            {
                properties.insert(VkProperty::SampleLocationSampleCount(count));
            }
        }
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
