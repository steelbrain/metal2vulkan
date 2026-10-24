//! `DeviceProfile` and its translation-relevant fields.
//!
//! `Device::new(desc, policy)` is the only producer of a [`DeviceProfile`]. The profile is the
//! normalized projection of a device description plus the policy; lowering selects representations
//! from it and nothing else. Derivation lives in `device::derive`, the fingerprint in
//! [`DeviceProfile::fingerprint`].

use std::collections::{BTreeMap, BTreeSet};

use crate::air::container::Hash256;
use crate::device::requirements::{ApiVersion, FormatUsage, VkExtension, VkFeature, VkProperty};
use crate::device::subgroup::{SimdClass, SubgroupModel, Tier};
use crate::error::{Capability, DriverDeviation, Unsupported};
use crate::memory::contain::Check;
use crate::pipeline::sym::Stage;
use crate::vk;

/// What `spirv-val` validates a module compiled for a device under
/// ([`DeviceProfile::spirv_validation`]): run `spirv-val --target-env <env>` with the relaxations
/// for each environment.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpirvValidation {
    /// The `--target-env` values: `vulkan1.2`, every module's floor, then the device's own API
    /// environment where it is newer (`vulkan1.4` for any device from 1.4, the newest environment
    /// `spirv-val` knows).
    pub target_envs: Vec<&'static str>,
    /// The `spirv-val` arguments that relax a rule a feature the device enables lifts:
    /// `--scalar-block-layout` (`scalarBlockLayout`), `--uniform-buffer-standard-layout`
    /// (`uniformBufferStandardLayout`), `--workgroup-scalar-block-layout`
    /// (`workgroupMemoryExplicitLayoutScalarBlockLayout`), `--allow-localsizeid` (`maintenance4`)
    /// and `--allow-offset-texture-operand` (`maintenance8`), in that order.
    pub relaxations: Vec<&'static str>,
}

/// The `spirv-val` relaxations, each after the feature whose enablement lifts its rule
/// ([`SpirvValidation::relaxations`]).
const SPIRV_VAL_RELAXATIONS: [(VkFeature, &str); 5] = [
    (VkFeature::ScalarBlockLayout, "--scalar-block-layout"),
    (
        VkFeature::UniformBufferStandardLayout,
        "--uniform-buffer-standard-layout",
    ),
    (
        VkFeature::WorkgroupMemoryExplicitLayoutScalarBlockLayout,
        "--workgroup-scalar-block-layout",
    ),
    (VkFeature::Maintenance4, "--allow-localsizeid"),
    (VkFeature::Maintenance8, "--allow-offset-texture-operand"),
];

/// The normalized, translation-relevant device projection. Opaque; built only by `Device::new`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    pub(crate) fields: ProfileFields,
}

impl DeviceProfile {
    /// The profile's fields: internal; the profile is opaque to consumers, who compare
    /// [`DeviceProfile::fingerprint`] and read its facts through the accessors.
    pub(crate) fn fields(&self) -> &ProfileFields {
        &self.fields
    }

    /// What the consumer enabled on the device, as the capability vocabulary
    /// [`Requirements`](crate::device::Requirements) uses: the features, extensions and
    /// properties a module compiled for the profile may rely on.
    pub fn enabled(&self) -> &EnabledCaps {
        &self.fields.enabled
    }

    /// The device limits the profile was built with (`VkPhysicalDeviceLimits` and the extension
    /// properties it reads).
    pub fn limits(&self) -> &Limits {
        &self.fields.limits
    }

    /// The fragment-stage and fixed-function capabilities the profile enables.
    pub fn fragment(&self) -> &FragmentCaps {
        &self.fields.fragment
    }

    /// The image and sampler capabilities the profile enables.
    pub fn images(&self) -> &ImageCaps {
        &self.fields.images
    }

    /// Whether the profile reads and writes a storage image or storage texel buffer through a
    /// declaration without a format by its format's own `STORAGE_READ_WITHOUT_FORMAT` /
    /// `STORAGE_WRITE_WITHOUT_FORMAT` bit, whatever `shaderStorageImage{Read,Write}WithoutFormat`
    /// say: at Vulkan 1.3, or with `VK_KHR_format_feature_flags2` enabled, each an enablement of
    /// the SPIR-V capabilities `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat`
    /// beside the features. The format tables then hold the bits
    /// (`DeviceDescription::has_format_feature_flags2`), which govern each format's access.
    pub(crate) fn without_format_by_format_bits(&self) -> bool {
        self.fields.api_version >= ApiVersion::V1_3
            || self
                .fields
                .enabled
                .extensions
                .contains(&VkExtension::KhrFormatFeatureFlags2)
    }

    /// Whether the profile can read (`read`) and write (`write`) storage through a declaration
    /// without a format at all: by the device's `shaderStorageImage{Read,Write}WithoutFormat`, or
    /// by the formats' own bits ([`DeviceProfile::without_format_by_format_bits`]). Which formats
    /// it may then view is the format tables' (`metal::format::without_format_gaps`).
    pub(crate) fn accesses_without_format(&self, read: bool, write: bool) -> bool {
        let images = &self.fields.images;
        let bits = self.without_format_by_format_bits();
        (!read || images.storage_read_without_format || bits)
            && (!write || images.storage_write_without_format || bits)
    }

    /// The descriptor and addressing capabilities the profile enables: whether it reaches buffers
    /// by device address, binds descriptors after binding, and so on.
    pub fn descriptors(&self) -> &DescriptorCaps {
        &self.fields.descriptors
    }

    /// The Function storage a thread holds correctly on the device: its measured value, or the
    /// floor of a device not measured. No Vulkan property states it; Metal states none either.
    pub fn function_storage(&self) -> FunctionStorage {
        self.fields.function_storage
    }

    /// The profile's subgroup model: each Metal stage's subgroup sizes, operations, and whether it
    /// can pin a size or require full subgroups. What a simdgroup runs as follows from it
    /// ([`DeviceProfile::simd_tier`]); a compiled function states what it requires
    /// (`CompiledFunction::required_subgroup_size`, `requires_full_subgroups`).
    pub fn subgroups(&self) -> &SubgroupModel {
        &self.fields.subgroup
    }

    /// How simdgroups of usage class `class` run in `stage` on this profile: its [`Tier`], or the
    /// feature it lacks. It depends on the class, the stage and the profile, never on a threadgroup
    /// size.
    pub fn simd_tier(&self, class: &SimdClass, stage: Stage) -> Result<Tier, Unsupported> {
        crate::device::subgroup::tier(class, stage, &self.fields.subgroup)
    }

    /// How this profile contains an out-of-bounds access: on, relying on the buffer robustness it
    /// enables, or off.
    pub fn containment(&self) -> Containment {
        self.fields.containment
    }

    /// Whether a module compiled for this profile checks a buffer access in the shader: exactly
    /// when containment is on and no buffer robustness is enabled. Elsewhere the descriptor range
    /// and the device's robustness contain it, or containment is off.
    pub fn checks_buffers_in_shader(&self) -> bool {
        crate::memory::contain::encoding(self) == Check::InShader
    }

    /// Whether a module compiled for this profile checks a buffer atomic in the shader: as
    /// [`DeviceProfile::checks_buffers_in_shader`], and also where the driver's robustness leaves a
    /// storage buffer's atomics uncontained ([`DriverWorkaround::BufferAtomicsUncontained`]).
    pub fn checks_atomics_in_shader(&self) -> bool {
        crate::memory::contain::atomic_encoding(self) == Check::InShader
    }

    /// Whether this profile leaves an out-of-bounds read of an image (`texel_buffer`: of a texel
    /// buffer) to the device, whose robustness answers Metal's out-of-bounds texel; elsewhere the
    /// shader checks the read.
    pub fn device_reads_texels(&self, texel_buffer: bool) -> bool {
        crate::memory::contain::device_reads_texels(self, texel_buffer)
    }

    /// Whether this profile leaves an out-of-bounds write of an image (`texel_buffer`: of a texel
    /// buffer) to the device, whose robustness discards it as Metal does; elsewhere the shader
    /// guards the write.
    pub fn device_drops_texel_writes(&self, texel_buffer: bool) -> bool {
        crate::memory::contain::device_drops_texel_writes(self, texel_buffer)
    }

    /// The driver workarounds the profile takes: the leaf encodings it changes and the refusals of
    /// what the driver is measured to run wrong.
    pub fn workarounds(&self) -> &[DriverWorkaround] {
        &self.fields.workarounds
    }

    /// The stage-module word count a pipeline's largest stage must fit for this device, where its
    /// driver faults compiling a module past it
    /// ([`DriverWorkaround::RegisterAllocatorOverflowsLargeModules`]); `None` where no such bound
    /// is measured. A compile of any pipeline kind over the bound retranslates with the consumer's
    /// exact forms dropped to fit
    /// ([`Device::fit_render_variant`](crate::device::Device::fit_render_variant),
    /// [`fit_mesh_variant`](crate::device::Device::fit_mesh_variant),
    /// [`fit_compute_pipeline`](crate::device::Device::fit_compute_pipeline),
    /// [`fit_tile_pipeline`](crate::device::Device::fit_tile_pipeline)).
    pub fn module_word_bound(&self) -> Option<u32> {
        self.fields
            .workarounds
            .contains(&DriverWorkaround::RegisterAllocatorOverflowsLargeModules)
            .then_some(REGISTER_ALLOCATOR_MODULE_WORD_BOUND)
    }

    /// The driver deviations the profile takes at a driver release newer than the last one
    /// measured, each with that release's `driverVersion` (`device::driver_facts`): a module
    /// compiled for the profile records each as
    /// [`PrecisionNote::BeyondMeasuredDeviation`](crate::pipeline::PrecisionNote::BeyondMeasuredDeviation).
    /// Empty on a measured release.
    pub fn driver_deviations(&self) -> &BTreeMap<DriverDeviation, u32> {
        &self.fields.deviations_beyond
    }

    /// Whether the profile's device answers samples, gathers and comparisons with its own
    /// sampler, with no emulation of Metal's in the module: a device that samples through Metal's
    /// own sampler (`VkProperty::MetalSampler`, MoltenVK on Apple's GPU), whose answers are
    /// Metal's, or one that samples natively (`VkProperty::DeviceSampling`,
    /// `SamplingPolicy::NativeDevice`, the default), whose answers are the device's
    /// ([`PrecisionNote::DeviceSampling`]). The translation's sampler sites and the runtime's
    /// sampler objects (`metal::sampler`) ask it alike.
    ///
    /// [`PrecisionNote::DeviceSampling`]: crate::pipeline::PrecisionNote::DeviceSampling
    pub(crate) fn device_samples(&self) -> bool {
        use crate::device::requirements::VkProperty;
        let properties = &self.fields.enabled.properties;
        properties.contains(&VkProperty::MetalSampler)
            || properties.contains(&VkProperty::DeviceSampling)
    }

    /// Whether the profile takes the driver deviation `deviation` (`device::driver_facts`): its
    /// driver is measured with it at the device's release, or at an older one
    /// ([`DeviceProfile::driver_deviations`] lists those), so the translation relies on it. A
    /// profile that does not take it compiles the conservative route the deviation's
    /// documentation names. A gate lane that checks a deviation on the device asks this before
    /// expecting the deviation's answer.
    pub fn takes_driver_deviation(&self, deviation: DriverDeviation) -> bool {
        let f = &self.fields;
        match deviation {
            DriverDeviation::CubeGradientRoute => {
                f.cube_gradients.compute == CubeGradientRoute::Metal
            }
            DriverDeviation::GraphicsCubeGradientRoute => {
                f.cube_gradients.graphics == CubeGradientRoute::AppleSwizzled
            }
            DriverDeviation::GraphicsCubeGradientTangents => {
                f.cube_gradients.graphics == CubeGradientRoute::AppleProjected
            }
            DriverDeviation::SimdgroupLockstep => f
                .enabled
                .properties
                .contains(&VkProperty::SimdgroupLockstep),
            DriverDeviation::MetalFloatFunctions => f.float_functions == FloatFunctions::Metal,
            DriverDeviation::MetalCanonicalNans => f.nan_encoding == NanEncoding::MetalCanonical,
            DriverDeviation::MetalSampler => {
                f.enabled.properties.contains(&VkProperty::MetalSampler)
            }
            DriverDeviation::MetalColorWrites => {
                f.enabled.properties.contains(&VkProperty::MetalColorWrites)
            }
            DriverDeviation::MetalFloatColorNans => f
                .enabled
                .properties
                .contains(&VkProperty::MetalFloatColorNans),
            DriverDeviation::MetalHalfColorNans => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfColorNans),
            DriverDeviation::HalfColorOutputBits => f
                .enabled
                .properties
                .contains(&VkProperty::HalfColorOutputBits),
            DriverDeviation::MetalHalfFma => {
                f.enabled.properties.contains(&VkProperty::MetalHalfFma)
            }
            DriverDeviation::MetalFloatDenormals => f
                .enabled
                .properties
                .contains(&VkProperty::MetalFloatDenormals),
            DriverDeviation::MetalHalfQuantize => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfQuantize),
            DriverDeviation::MetalHalfConversion => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfConversion),
            DriverDeviation::MetalCenterVaryings => f
                .enabled
                .properties
                .contains(&VkProperty::MetalCenterVaryings),
            DriverDeviation::MetalSmallFloatColorWrites => f
                .enabled
                .properties
                .contains(&VkProperty::MetalSmallFloatColorWrites),
            DriverDeviation::MetalNorm16ColorWrites => f
                .enabled
                .properties
                .contains(&VkProperty::MetalNorm16ColorWrites),
            DriverDeviation::MetalMinMax => f.enabled.properties.contains(&VkProperty::MetalMinMax),
            DriverDeviation::MetalHalfArithmetic => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfArithmetic),
            DriverDeviation::MetalHalfStorageWrites => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfStorageWrites),
            DriverDeviation::MetalHalfStorageNans => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfStorageNans),
            DriverDeviation::MetalFloatStorageNans => f
                .enabled
                .properties
                .contains(&VkProperty::MetalFloatStorageNans),
            DriverDeviation::MetalFragmentSimdgroups => f
                .enabled
                .properties
                .contains(&VkProperty::MetalFragmentSimdgroups),
            DriverDeviation::MetalVertexSimdgroups => f
                .enabled
                .properties
                .contains(&VkProperty::MetalVertexSimdgroups),
            DriverDeviation::CenterFragCoordDepth => f
                .enabled
                .properties
                .contains(&VkProperty::CenterFragCoordDepth),
            DriverDeviation::SampleFragCoordDepth => f
                .enabled
                .properties
                .contains(&VkProperty::SampleFragCoordDepth),
            DriverDeviation::MetalHalfTexelWrites => f
                .enabled
                .properties
                .contains(&VkProperty::MetalHalfTexelWrites),
            DriverDeviation::MetalNormStorageWrites => f
                .enabled
                .properties
                .contains(&VkProperty::MetalNormStorageWrites),
            DriverDeviation::Unorm8LinearFilter => f
                .enabled
                .properties
                .contains(&VkProperty::Unorm8LinearFilter),
            DriverDeviation::MetalDepth16Stores => f
                .enabled
                .properties
                .contains(&VkProperty::MetalDepth16Stores),
            DriverDeviation::MetalDepth16BiasedStores => f
                .enabled
                .properties
                .contains(&VkProperty::MetalDepth16BiasedStores),
            DriverDeviation::MetalInterpolation => f
                .enabled
                .properties
                .contains(&VkProperty::MetalInterpolation),
            DriverDeviation::MetalFirstVertexIndices => f
                .enabled
                .properties
                .contains(&VkProperty::MetalFirstVertexIndices),
            DriverDeviation::MetalFullQuads => {
                f.enabled.properties.contains(&VkProperty::MetalFullQuads)
            }
        }
    }

    /// What this profile lacks of Profile A's device-address requirements: `bufferDeviceAddress`
    /// and `shaderInt64`, whichever are missing, in that order. Empty exactly when the profile
    /// reaches buffers by device address (Profile A). A Profile F refusal that device addresses
    /// would lift needs this one conjunct, so a device that has one of the two features is never
    /// told it needs that feature.
    pub fn addressed_needs(&self) -> Vec<Capability> {
        let f = &self.fields;
        let mut needs = Vec::new();
        if !f.descriptors.buffer_device_address {
            needs.push(Capability::Feature(VkFeature::BufferDeviceAddress));
        }
        if !f.leaf.int64 {
            needs.push(Capability::Feature(VkFeature::ShaderInt64));
        }
        needs
    }

    /// What `spirv-val` validates a module compiled for this profile under (the L3): each of the
    /// target environments with the relaxations. Every module the crate returns for the profile has
    /// requirements within it ([`DeviceProfile::check`]) and is valid under each; the crate's own
    /// tests, this workspace's validation and the command-line tool run exactly this. A consumer
    /// that wants Khronos's own check of what it was handed runs it (the library runs no external
    /// tool).
    pub fn spirv_validation(&self) -> SpirvValidation {
        let api = self.fields.api_version;
        let own = if api >= (ApiVersion { major: 1, minor: 4 }) {
            "vulkan1.4"
        } else if api >= ApiVersion::V1_3 {
            "vulkan1.3"
        } else {
            "vulkan1.2"
        };
        let mut target_envs = vec!["vulkan1.2"];
        if own != "vulkan1.2" {
            target_envs.push(own);
        }
        let enabled = &self.fields.enabled.features;
        SpirvValidation {
            target_envs,
            relaxations: SPIRV_VAL_RELAXATIONS
                .iter()
                .filter(|(feature, _)| enabled.contains(feature))
                .map(|&(_, argument)| argument)
                .collect(),
        }
    }

    /// A SHA-256 over **every** field, policy included.
    ///
    /// The encoding is the derived `Debug` rendering of the profile's fields under a versioned
    /// domain tag: it is total by construction (a new field is hashed without further code) and
    /// unambiguous for these types (no free-form strings). Formatting changes across toolchains
    /// can only change fingerprints, which costs cache misses, never a false hit.
    ///
    /// The rendering streams into the hash: every compile takes the fingerprint for its key, and a
    /// materialized rendering of a profile with a format table is tens of kilobytes that no
    /// input-scaled ledger charge accounts for.
    pub fn fingerprint(&self) -> Hash256 {
        use sha2::{Digest, Sha256};
        use std::fmt::Write as _;
        struct Hashing(Sha256);
        impl std::fmt::Write for Hashing {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                self.0.update(s.as_bytes());
                Ok(())
            }
        }
        let mut h = Hashing(Sha256::new());
        h.0.update(b"metal2vulkan/device-profile/v1\0");
        // `Hashing` never fails, and `Debug` of these plain-data types only fails when the
        // writer does.
        let _ = write!(h, "{:?}", self.fields);
        h.0.finalize().into()
    }
}

/// Every field of a profile, policy included. The fingerprint hashes all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileFields {
    /// Memory's A/F split.
    pub buffer_access: BufferAccess,
    pub pointer_domain: PointerDomain,
    pub handle_domain: HandleDomain,
    pub containment: Containment,
    pub isolation: GuestIsolation,
    /// Whether a texel's value is the device's conversion or Metal's (the policy's).
    pub texel_conversion: crate::device::policy::TexelConversionPolicy,
    /// Whether a `half4` output of a module without fast math blends into a 16-bit float
    /// attachment in the device's blend or in Metal's, computed by the function (the policy's).
    pub half_blend: crate::device::policy::HalfBlendPolicy,
    /// Whether a `half` or `float` read of a 16-bit float texture answers the device's decode of
    /// the texel or the texel's own bits, read through the texture's unsigned-integer alias (the
    /// policy's).
    pub half_texel_bits: crate::device::policy::HalfTexelBitsPolicy,
    /// Whose `half` rounding a module takes: Metal's (a measured form, or the integer rounding
    /// the policy's `HalfRoundingPolicy::Exact` asks for) or the device's own conversion or
    /// quantize (`lower::arith::narrow::quantize_form`). Derived, so a device measured to round
    /// as Metal does keys the same under both policies.
    pub half_rounding: HalfRoundingRoute,
    pub leaf: LeafCaps,
    pub subgroup: SubgroupModel,
    pub images: ImageCaps,
    pub descriptors: DescriptorCaps,
    pub fragment: FragmentCaps,
    pub float_controls: FloatControls,
    pub vulkan_memory_model: bool,
    pub draw_parameters: bool,
    pub limits: Limits,
    /// Formats usable as vertex attributes (`VertexBuffer`).
    pub vertex_formats: FormatTable,
    /// Formats usable as storage images or storage texel buffers (plain or atomic).
    pub storage_formats: FormatTable,
    /// Every other queried usage: sampled images (and their filtering), uniform texel buffers, and
    /// attachments.
    pub image_formats: FormatTable,
    /// The driver workarounds: leaf encodings, and refusals of what the driver is measured to run
    /// wrong.
    pub workarounds: Vec<DriverWorkaround>,
    /// The device's Vulkan version (the floor is 1.2; every profile emits SPIR-V 1.5).
    pub api_version: ApiVersion,
    /// Vertex-buffer strides, with the other extended dynamic states, are dynamic state: the
    /// description lists `extendedDynamicState` (Vulkan 1.3, or `VK_EXT_extended_dynamic_state`
    /// with its feature enabled). A render pipeline that makes them dynamic requires the feature.
    pub dynamic_vertex_stride: bool,
    /// The enabled capability sets `Requirements` are checked against.
    pub enabled: EnabledCaps,
    /// Which functions the device's `GLSL.std.450` instructions and `OpFDiv` of `float` compute:
    /// Metal's own, so the translation leaves a function to the device where Metal's compile of the
    /// module computes it in the call's math mode, and takes the fast reciprocal from the device's
    /// division, or the device's, so the translation computes the functions whose device answer may
    /// leave Apple's bound itself.
    pub float_functions: FloatFunctions,
    /// Which NaN the device's `float` and `half` arithmetic answers: Metal's canonical NaN, so the
    /// translation leaves it to the device, or the device's own, so the translation selects Metal's
    /// where a result's bits are observed.
    pub nan_encoding: NanEncoding,
    /// Whose NaN a `half` read, sample or gather hands back for a NaN texel: Metal's, by the
    /// class of the texture bound, where the policy's `NanMinMaxPolicy::Exact` asks for it, on
    /// every device, or the device's decode (the default).
    pub texel_nans: TexelNanRoute,
    /// Whether a linear sample the device's sampler answers (`DeviceProfile::device_samples`)
    /// takes Metal's filter of a two-plane YCbCr texture bound at its slot, through the slot's
    /// YCbCr variant (`RuntimeFact::YCbCrTexture`): under the policy's `YCbCrFilterPolicy::Exact`,
    /// and on a device that samples through Metal's own sampler also under `SamplingPolicy::Exact`.
    /// Elsewhere the device filters such a texture as any other. Derived, so a profile that answers
    /// alike under both policies keys the same.
    pub ycbcr_variant: bool,
    /// Whose operand Metal's minimum and maximum functions pick (`lower::arith::native_min_max`):
    /// Metal's, or the device's `NMin`, `NMax` and `NClamp` as they are (the policy's
    /// `NanMinMaxPolicy::Native`). Derived, so a device that keeps Metal's picks keys the same
    /// under both policies.
    pub min_max: MinMaxRoute,
    /// In whose order Metal's `dot`, `mix` and `fast::sqrt` round: Metal's, or the device's
    /// single instructions (the policy's `RoundingOrderPolicy::Native`). Derived, as
    /// [`ProfileFields::min_max`] is.
    pub rounding_order: RoundingOrderRoute,
    /// The Function storage a thread holds correctly (`device::function_storage`): a stage's
    /// private memory, its call stacks included, may take this much a thread
    /// (`VkLimit::FunctionStorageBytesPerThread`), and a vertex or fragment function on a device
    /// not measured more ([`FunctionStorage::Assumed`]).
    pub function_storage: FunctionStorage,
    /// How the driver hands a cube sample's explicit gradients to the GPU, in a graphics and in
    /// a compute pipeline: Vulkan's cube derivative rule unless the driver's release is measured
    /// with another (`DriverDeviation::CubeGradientRoute`, `GraphicsCubeGradientRoute` and
    /// `GraphicsCubeGradientTangents`).
    pub cube_gradients: CubeGradients,
    /// The driver's deviations the profile takes at a release newer than the last one measured
    /// (`device::driver_facts`), each with that release's `driverVersion`: taken, and recorded
    /// by every module compiled for the profile as `PrecisionNote::BeyondMeasuredDeviation`.
    /// Empty on a measured release.
    pub deviations_beyond: BTreeMap<DriverDeviation, u32>,
}

/// How a driver hands a cube sample's explicit gradients to the GPU
/// ([`ProfileFields::cube_gradients`]), per pipeline kind, each decided by the driver's measured
/// deviations (`DriverDeviation::GraphicsCubeGradientRoute`,
/// `DriverDeviation::GraphicsCubeGradientTangents` and `DriverDeviation::CubeGradientRoute`);
/// without them, Vulkan's rule, but [`CubeGradientRoute::Unmeasured`] on a MoltenVK `driver_id`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CubeGradients {
    /// A vertex or fragment function's.
    pub graphics: CubeGradientRoute,
    /// A kernel's.
    pub compute: CubeGradientRoute,
}

/// The route of a cube sample's explicit gradients to the GPU ([`CubeGradients`]). Metal's
/// footprint is the one the Apple M3 Max reads under macOS 27 (`lower::sampler::CubeFootprint`):
/// each gradient's two tangent components on the face, its normal component dropped.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeGradientRoute {
    /// Vulkan's rule ("Cube Map Derivative Transformation") reads the footprint from them: the
    /// function hands an isotropic sample Metal's length of the two tangent components on one
    /// tangent, and an anisotropic one Metal's face gradients without their normal component
    /// (`lower::sampler::CubeFootprint::Length` and `Face`); the level and the anisotropic filter
    /// are the device's (`PrecisionNote::CubeGradientFootprint`).
    Vulkan,
    /// SPIRV-Cross's `spvGradientCube` (its `agx_manual_cube_grad_fixup`) projects each onto the
    /// face as Vulkan's rule does and swizzles the two values into the components Apple's
    /// isotropic reading took under macOS 26.7, then hands them to Metal's `gradientcube`, whose
    /// anisotropic reading on the Apple M3 Max there takes them transposed: that GPU's macOS 26.7
    /// footprint (`lower::sampler::CubeFootprint::Isotropic` and `AnisotropicSwizzled`), measured
    /// on that identity alone.
    AppleSwizzled,
    /// The same fixup under macOS 27, whose Metal reads the face's two tangents of what it is
    /// handed: an `x` or `z` face's gradient without its normal component reaches Metal as its
    /// own, exactly; on a `y` face the fixup writes the face's `x` tangent on both, so the function
    /// hands Metal's length there, which keeps an isotropic sample's footprint and lays an
    /// anisotropic one's ellipse on the diagonal (`lower::sampler::CubeFootprint::Projected`,
    /// `PrecisionNote::CubeGradientFootprint`).
    AppleProjected,
    /// They reach Metal's `gradientcube` as given, which the GPU reads by Metal's own rule, so
    /// the function hands Metal's own (`lower::sampler::CubeFootprint::Metal`).
    Metal,
    /// The driver hands them to Metal by a route its description does not state: a MoltenVK
    /// `driver_id` at a release no measurement covers, or a document of one that states no
    /// route. The function takes Metal's footprint itself where its emulation reaches the sample
    /// (`lower::sampler::anisotropic`): Metal's level, and an anisotropic sampler's taps, each an
    /// explicit level of Metal's own sampler, which no route alters
    /// (`PrecisionNote::AnisotropicFilter` for the taps). Elsewhere it hands the isotropic form's
    /// length, on one tangent, which every route above reads alike (but a `y` face's through
    /// [`CubeGradientRoute::AppleProjected`]'s fixup), through an isotropic sampler
    /// (`metal::sampler::SamplerUse::cube_gradient`, `PrecisionNote::CubeGradientFootprint`).
    Unmeasured,
}

impl CubeGradients {
    /// The route of a function of `stage`: a kernel's is the compute pipeline's.
    pub fn route(self, stage: crate::pipeline::sym::Stage) -> CubeGradientRoute {
        match stage {
            crate::pipeline::sym::Stage::Kernel => self.compute,
            _ => self.graphics,
        }
    }
}

/// The Function storage a thread holds correctly ([`DeviceProfile::function_storage`]), in bytes:
/// how much private memory a stage may keep a thread (its `thread` variables and arrays, its call
/// stacks, and the constants it holds per thread), which a driver backs with scratch memory and
/// past which it can answer wrongly or hang with no error. Measured per device, since no Vulkan
/// property states it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionStorage {
    /// Measured on the device: a thread holds this many bytes correctly.
    Measured(u64),
    /// The device was not measured: it holds this many, the floor, with no note. A vertex or
    /// fragment function may keep more, up to the 192 KiB a thread Metal was measured to hold, and
    /// records `PrecisionNote::FunctionStorageAssumed`; any other stage past the floor is refused.
    Assumed(u64),
}

impl FunctionStorage {
    /// The bytes a thread holds, measured or assumed.
    pub fn bytes(self) -> u64 {
        match self {
            FunctionStorage::Measured(b) | FunctionStorage::Assumed(b) => b,
        }
    }
}

/// Which functions a device's `GLSL.std.450` instructions and `OpFDiv` of `float` compute
/// ([`ProfileFields::float_functions`]), decided by the driver's measured deviation
/// [`DriverDeviation::MetalFloatFunctions`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFunctions {
    /// Metal's own, in the math mode Metal compiles the module with: fast where the entry
    /// declares no `SignedZeroInfNanPreserve`, precise where it declares one, and precise
    /// `tanh` and `atan2` either way. The division likewise: Metal's fast division and
    /// reciprocal in a fast-math module, its correctly rounded ones otherwise.
    Metal,
    /// The device's, within Vulkan's precision table, which the translation does not take
    /// where Metal answers beyond it: its routines of Metal's answers instead (the policy's
    /// `FloatMathsPolicy::Exact`).
    Device,
    /// The device's, within Vulkan's precision table, taken as they are: native device maths
    /// (`FloatMathsPolicy::Native`, the default), each kind recorded with its bound.
    Native,
}

/// Whose rounding a `half` result takes ([`ProfileFields::half_rounding`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HalfRoundingRoute {
    /// Metal's: the device's conversion or quantize where it is measured to round as Metal does
    /// (`DriverDeviation::MetalHalfConversion`, `DriverDeviation::MetalHalfQuantize`), else
    /// integer arithmetic.
    Metal,
    /// The device's own conversion or quantize where neither is measured to be Metal's: native
    /// half rounding (`HalfRoundingPolicy::Native`, the default), each kind recorded with its
    /// bound.
    Native,
}

/// Whose operand Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` on `float` and
/// `half` pick ([`ProfileFields::min_max`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MinMaxRoute {
    /// Metal's: the device's `NMin`, `NMax` and `NClamp` where they are measured to be Metal's
    /// (`DriverDeviation::MetalMinMax`), else the translation's selects of Metal's operand, as on a
    /// device whose functions are Metal's own (`DriverDeviation::MetalFloatFunctions`) or under
    /// `NanMinMaxPolicy::Exact`.
    Metal,
    /// The device's `NMin`, `NMax` and `NClamp` as they are (`NanMinMaxPolicy::Native`, the
    /// default), noted `PrecisionNote::NativeMinMax` with its bound.
    Native,
}

/// In whose order Metal's `dot`, `mix` and `fast::sqrt` on `float` and `half` round
/// ([`ProfileFields::rounding_order`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundingOrderRoute {
    /// Metal's: its fused chain, its two fused multiply-adds, its `x · rsqrt(x)`, or the
    /// device's own instruction where the device's functions are Metal's
    /// (`DriverDeviation::MetalFloatFunctions`); under `RoundingOrderPolicy::Exact` everywhere.
    Metal,
    /// The device's `OpDot`, `FMix` and `Sqrt` (`RoundingOrderPolicy::Native`, the default),
    /// noted `PrecisionNote::NativeDot`, `NativeMix` and `NativeTranscendentals` with their
    /// bounds.
    Native,
}

/// Which NaN a device's `float` and `half` arithmetic answers ([`ProfileFields::nan_encoding`]),
/// decided by the driver's measured deviation [`DriverDeviation::MetalCanonicalNans`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NanEncoding {
    /// Metal's canonical NaN, `0x7FC00000` (`0x7E00` as `half`), for every NaN result of the
    /// arithmetic and of the functions whose NaN Metal canonicalizes.
    MetalCanonical,
    /// The device's own: an operand's bits, quieted or not, or another NaN, which the
    /// translation answers Metal's in place of (the policy's `NanMinMaxPolicy::Exact`).
    Device,
    /// The device's own, taken as it is (`NanMinMaxPolicy::Native`, the default), noted
    /// `PrecisionNote::NativeNanEncoding` with its bound.
    Native,
}

/// Whose NaN a `half` texel access hands back for a NaN texel ([`ProfileFields::texel_nans`]).
/// A texel's NaN is the image's decode, which the access narrows itself, not a result of the
/// device's arithmetic, so [`ProfileFields::nan_encoding`] does not decide it: a device measured
/// to answer Metal's canonical NaN in its arithmetic (`NanEncoding::MetalCanonical`) need not
/// answer it for a texel, and MoltenVK on the Apple M3 Max does not (a 32-bit float texel's NaN
/// read as `half` keeps its payload's top bits).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelNanRoute {
    /// Metal's rule (`NanMinMaxPolicy::Exact`): a 16-bit float texel's own NaN where the access
    /// reads one texel, and `0x7e00` for every other NaN, by the class of the texture bound
    /// (`lower::texture::texel_nans_class`).
    Metal,
    /// The device's decode (`NanMinMaxPolicy::Native`, the default), noted
    /// `PrecisionNote::DeviceNanEncoding`.
    Device,
}

/// What the consumer enabled, as the capability vocabulary `Requirements` uses
/// ([`DeviceProfile::enabled`]). Selection reads the typed caps above; the finalize check
/// (`requirements ⊆ profile`) reads these.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnabledCaps {
    /// The enabled features.
    pub features: BTreeSet<VkFeature>,
    /// The enabled extensions.
    pub extensions: BTreeSet<VkExtension>,
    /// The properties the device reports that a module may rely on.
    pub properties: BTreeSet<VkProperty>,
}

impl EnabledCaps {
    /// Whether `VK_KHR_portability_subset` withholds what its feature `feature` permits: the
    /// extension is enabled (as it is wherever the device lists it) and the feature is not. A
    /// device without the extension has none of the subset's restrictions.
    pub fn portability_withholds(&self, feature: VkFeature) -> bool {
        self.extensions.contains(&VkExtension::KhrPortabilitySubset)
            && !self.features.contains(&feature)
    }
}

/// Memory profile: Profile A (`Addressed`, buffer device address available) or Profile F
/// (`Floor`, descriptors only).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    Addressed,
    Floor,
}

/// How pointer values are resolved to resources. v1 ships `ResidentTable` only.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointerDomain {
    /// The sorted resident set of bound arguments and used resources.
    ResidentTable,
}

/// How resource handles are resolved. v1 ships `Table` only.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandleDomain {
    /// The handle table plus the set-1 heap.
    Table,
}

/// Out-of-bounds containment encoding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Containment {
    /// Containment is on; descriptor views rely on this robustness level when enabled.
    On { robust: Robustness },
    /// No containment. In the fingerprint; never a default.
    UnsafeOff,
}

/// Which buffer robustness the profile enables.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Robustness {
    /// Neither; descriptor views get in-shader checks.
    None,
    /// `robustBufferAccess`.
    Rba,
    /// `robustBufferAccess2`.
    Rba2,
}

/// Whether guest shaders are isolated to the resources the binder supplied.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GuestIsolation {
    On,
    Off,
}

/// Leaf arithmetic, storage, and atomic capabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeafCaps {
    pub int8: bool,
    pub int16: bool,
    pub int64: bool,
    pub float16: bool,
    pub float64: bool,
    pub storage_buffer_8bit: bool,
    pub uniform_buffer_8bit: bool,
    pub push_constant_8bit: bool,
    pub storage_buffer_16bit: bool,
    pub uniform_buffer_16bit: bool,
    pub push_constant_16bit: bool,
    pub input_output_16bit: bool,
    pub workgroup_explicit_layout: bool,
    pub workgroup_explicit_layout_8bit: bool,
    pub workgroup_explicit_layout_16bit: bool,
    pub buffer_int64_atomics: bool,
    pub shared_int64_atomics: bool,
    pub buffer_float32_atomic_add: bool,
    pub buffer_float32_atomic_min_max: bool,
    pub shared_float32_atomic_add: bool,
    pub shared_float32_atomic_min_max: bool,
    pub image_float32_atomic_add: bool,
    pub image_float32_atomic_min_max: bool,
    pub image_int64_atomics: bool,
    pub vertex_stores_and_atomics: bool,
    pub fragment_stores_and_atomics: bool,
    pub zero_initialize_workgroup_memory: bool,
    /// `OpFmaKHR` on 32-bit floats (`shaderFmaFloat32`): a correctly rounded fused multiply-add.
    pub fma32: bool,
    /// `OpFmaKHR` on 16-bit floats (`shaderFmaFloat16`): a correctly rounded fused multiply-add.
    pub fma16: bool,
}

/// Image and sampler capabilities ([`DeviceProfile::images`]). Each field without its own note is
/// the Vulkan feature of that name (`shaderStorageImageMultisample` for `storage_multisample`,
/// ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageCaps {
    pub storage_read_without_format: bool,
    pub storage_write_without_format: bool,
    pub storage_extended_formats: bool,
    pub storage_multisample: bool,
    pub cube_array: bool,
    pub gather_extended: bool,
    /// The `Offset` image operand on `OpImageSample*` (`maintenance8`).
    pub dynamic_sample_offset: bool,
    pub min_lod: bool,
    /// `robustImageAccess` or `robustImageAccess2`: an image access outside the view is
    /// discarded (a write) or reads zero (a read; under the first level alone, a format's missing
    /// alpha reads 0 or 1, and a fetch at a level past the view's is undefined).
    pub robust_image_access: bool,
    /// `robustImageAccess2`: an image read outside the view (coordinate, layer, sample, or level)
    /// returns zero with `(0, 0, 1)` substituted for the format's missing G, B, and A, which is
    /// Metal's out-of-bounds texel.
    pub robust_image_access2: bool,
    pub sampler_anisotropy: bool,
    pub sampler_filter_minmax: bool,
    pub sampler_mirror_clamp_to_edge: bool,
    pub compute_derivative_quads: bool,
    pub compute_derivative_linear: bool,
}

/// Descriptor and addressing capabilities ([`DeviceProfile::descriptors`]). Each field is the
/// Vulkan feature or extension of that name (`bufferDeviceAddress` for `buffer_device_address`,
/// `VK_KHR_push_descriptor` for `push_descriptor`, ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DescriptorCaps {
    pub buffer_device_address: bool,
    pub scalar_block_layout: bool,
    pub uniform_buffer_standard_layout: bool,
    pub runtime_descriptor_array: bool,
    pub partially_bound: bool,
    pub variable_descriptor_count: bool,
    pub update_after_bind_sampled_image: bool,
    pub update_after_bind_storage_image: bool,
    pub update_after_bind_storage_buffer: bool,
    pub update_unused_while_pending: bool,
    pub non_uniform_sampled_image: bool,
    pub non_uniform_storage_image: bool,
    pub non_uniform_storage_buffer: bool,
    pub non_uniform_uniform_texel_buffer: bool,
    pub non_uniform_storage_texel_buffer: bool,
    pub push_descriptor: bool,
    pub null_descriptor: bool,
}

/// Fragment-stage and fixed-function capabilities ([`DeviceProfile::fragment`]). Each field is the
/// Vulkan feature or extension of that name (`shaderStencilExport` for `stencil_export`,
/// `rasterizationOrderColorAttachmentAccess` for `rasterization_order_color_access`, ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FragmentCaps {
    pub demote_to_helper: bool,
    pub terminate_invocation: bool,
    pub pixel_interlock: bool,
    pub sample_interlock: bool,
    pub barycentric: bool,
    pub stencil_export: bool,
    pub sample_rate_shading: bool,
    pub dual_src_blend: bool,
    pub independent_blend: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub output_layer: bool,
    pub output_viewport_index: bool,
    pub clip_distance: bool,
    pub cull_distance: bool,
    pub rasterization_order_color_access: bool,
    pub dynamic_rendering_local_read: bool,
    pub vertex_attribute_divisor: bool,
    pub vertex_attribute_zero_divisor: bool,
}

/// Float-controls support of one width.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FloatWidthControls {
    pub denorm_preserve: bool,
    pub denorm_flush_to_zero: bool,
    pub rounding_rte: bool,
    pub rounding_rtz: bool,
    pub signed_zero_inf_nan_preserve: bool,
}

/// `VkShaderFloatControlsIndependence`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Independence {
    Only32Bit,
    All,
    #[default]
    None,
}

impl Independence {
    /// The contracts from most constrained to fully independent: `None` (every width uses the
    /// same mode, omission included) < `Only32Bit` (32-bit may differ; 16 and 64 agree) < `All`.
    const fn rank(self) -> u8 {
        match self {
            Independence::None => 0,
            Independence::Only32Bit => 1,
            Independence::All => 2,
        }
    }

    /// Whether a device with this independence admits a module that needs `required`
    /// (RuntimeSpirv VUIDs 06289-06292, ported from `src/vulkan/float_controls.rs`).
    pub const fn admits(self, required: Independence) -> bool {
        self.rank() >= required.rank()
    }
}

/// Float-controls properties.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FloatControls {
    pub f16: FloatWidthControls,
    pub f32: FloatWidthControls,
    pub f64: FloatWidthControls,
    pub denorm_independence: Independence,
    pub rounding_independence: Independence,
}

/// The device limits the new core consumes (named after `VkPhysicalDeviceLimits` and extension
/// properties; see `device::requirements::VkLimit`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Limits {
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_push_constants_size: u32,
    pub max_push_descriptors: u32,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_descriptor_set_update_after_bind_samplers: u32,
    pub max_descriptor_set_update_after_bind_sampled_images: u32,
    pub max_descriptor_set_update_after_bind_storage_images: u32,
    pub max_descriptor_set_update_after_bind_storage_buffers: u32,
    pub max_per_stage_descriptor_update_after_bind_samplers: u32,
    pub max_per_stage_descriptor_update_after_bind_sampled_images: u32,
    pub max_per_stage_descriptor_update_after_bind_storage_images: u32,
    pub max_per_stage_descriptor_update_after_bind_storage_buffers: u32,
    pub max_per_stage_update_after_bind_resources: u32,
    pub max_storage_buffer_range: u32,
    pub max_uniform_buffer_range: u32,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_texel_buffer_offset_alignment: u64,
    pub max_texel_buffer_elements: u32,
    /// `maxSamplerLodBias`, as the IEEE-754 bits of the `f32` limit (keeps the struct `Eq + Hash`).
    pub max_sampler_lod_bias_bits: u32,
    /// `maxSamplerAnisotropy`, as the IEEE-754 bits of the `f32` limit.
    pub max_sampler_anisotropy_bits: u32,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    /// `minInterpolationOffset`, as the IEEE-754 bits of the `f32` limit.
    pub min_interpolation_offset_bits: u32,
    /// `maxInterpolationOffset`, as the IEEE-754 bits of the `f32` limit.
    pub max_interpolation_offset_bits: u32,
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    /// `maxTessellationGenerationLevel`: the largest tessellation factor (0 without
    /// `tessellationShader`).
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_color_attachments: u32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub max_vertex_attrib_divisor: u32,
    /// `maxPerStageDescriptorAccelerationStructures` (`VK_KHR_acceleration_structure`; 0 where
    /// the extension is not exposed).
    pub max_per_stage_descriptor_acceleration_structures: u32,
    /// `maxDescriptorSetAccelerationStructures` (0 where `VK_KHR_acceleration_structure` is not
    /// exposed).
    pub max_descriptor_set_acceleration_structures: u32,
    /// Whether a draw with a nonzero `firstInstance` may use a vertex binding divisor other
    /// than 1: `supportsNonZeroFirstInstance` where `VK_KHR_vertex_attribute_divisor` or
    /// Vulkan 1.4 exposes it (VUID-vkCmdDraw-pNext-09461 otherwise requires `firstInstance == 0`),
    /// and `false` otherwise, including a device exposing only `VK_EXT_vertex_attribute_divisor`
    /// (which cannot report the property; its drivers do carry the restriction).
    pub supports_non_zero_first_instance: bool,
    /// `VkPhysicalDevicePortabilitySubsetPropertiesKHR::minVertexInputBindingStrideAlignment` on a
    /// device exposing `VK_KHR_portability_subset` (which must then be enabled): every vertex
    /// binding stride must be a multiple of it and at least as large
    /// (`VUID-VkVertexInputBindingDescription-stride-04456`). `None` on other devices, which
    /// have no such rule; `Some(u32::MAX)` when a portability device does not report it (absence
    /// is "not supported").
    pub min_vertex_input_binding_stride_alignment: Option<u32>,
    pub framebuffer_color_sample_counts: vk::SampleCountFlags,
    pub framebuffer_depth_sample_counts: vk::SampleCountFlags,
    /// `maxMultiviewViewCount` (Vulkan 1.1; at least 6 where `multiview` is supported): the
    /// largest view count of a multiview render pass instance, and so of a vertex amplification.
    pub max_multiview_view_count: u32,
    /// `maxMultiviewInstanceIndex` (Vulkan 1.1; at least 2^27 - 1 where `multiview` is
    /// supported): the largest instance index of a draw in a multiview render pass instance.
    pub max_multiview_instance_index: u32,
    /// `framebufferIntegerColorSampleCounts` (Vulkan 1.2): the sample counts every integer color
    /// attachment supports (`framebufferColorSampleCounts` covers the float and fixed-point ones).
    pub framebuffer_integer_color_sample_counts: vk::SampleCountFlags,
    /// `framebufferStencilSampleCounts`: the sample counts every attachment with stencil supports.
    pub framebuffer_stencil_sample_counts: vk::SampleCountFlags,
    /// `framebufferNoAttachmentsSampleCounts`: the rasterization sample counts a subpass without
    /// attachments supports.
    pub framebuffer_no_attachments_sample_counts: vk::SampleCountFlags,
    /// `storageImageSampleCounts`: the sample counts every 2D optimal-tiling storage image
    /// supports (a multisampled shadow of the stencil).
    pub storage_image_sample_counts: vk::SampleCountFlags,
    /// `maxFramebufferWidth`: the widest framebuffer.
    pub max_framebuffer_width: u32,
    /// `maxFramebufferHeight`: the tallest framebuffer.
    pub max_framebuffer_height: u32,
    /// `maxFramebufferLayers`: the most layers a framebuffer may have.
    pub max_framebuffer_layers: u32,
    /// `maxDrawIndexedIndexValue`: the largest index an indexed draw may fetch (`2^32 - 1` with
    /// `fullDrawIndexUint32`, at least `2^24 - 1` otherwise).
    pub max_draw_indexed_index_value: u32,
    /// `maxTessellationPatchSize`: the most control points a patch may have (0 without
    /// `tessellationShader`).
    pub max_tessellation_patch_size: u32,
    /// `maxGeometryInputComponents`: the scalar components of a geometry stage's user inputs per
    /// vertex (0 without `geometryShader`).
    pub max_geometry_input_components: u32,
    /// `maxGeometryOutputComponents`: the scalar components of a geometry stage's user outputs
    /// per vertex (0 without `geometryShader`).
    pub max_geometry_output_components: u32,
    /// `maxGeometryTotalOutputComponents`: the scalar components of every vertex a geometry stage
    /// emits for one input primitive, together (0 without `geometryShader`).
    pub max_geometry_total_output_components: u32,
    /// `supportedDepthResolveModes` (`VkPhysicalDeviceDepthStencilResolveProperties`, Vulkan
    /// 1.2): the modes a depth attachment may resolve by.
    pub supported_depth_resolve_modes: vk::ResolveModeFlags,
    /// `supportedStencilResolveModes`: the modes a stencil attachment may resolve by.
    pub supported_stencil_resolve_modes: vk::ResolveModeFlags,
    /// `independentResolveNone`: one aspect may resolve while the other does not.
    pub independent_resolve_none: bool,
    /// `independentResolve`: the two aspects may resolve by different modes.
    pub independent_resolve: bool,
    /// `sampleLocationSampleCounts` (`VkPhysicalDeviceSampleLocationsPropertiesEXT`): the sample
    /// counts that take custom sample locations; empty where `VK_EXT_sample_locations` is not
    /// exposed.
    pub sample_location_sample_counts: vk::SampleCountFlags,
    /// `variableSampleLocations`: the sample locations may differ between pipelines of one
    /// subpass; `false` where `VK_EXT_sample_locations` is not exposed.
    pub variable_sample_locations: bool,
    /// `provokingVertexModePerPipeline` (`VkPhysicalDeviceProvokingVertexPropertiesEXT`): the
    /// pipelines of one render pass instance may differ in provoking vertex mode; `false` where
    /// `VK_EXT_provoking_vertex` is not exposed.
    pub provoking_vertex_mode_per_pipeline: bool,
    /// `VkPhysicalDeviceMeshShaderPropertiesEXT`: every limit 0 where `VK_EXT_mesh_shader` is not
    /// exposed.
    pub mesh: MeshLimits,
    /// The upper bound of `pointSizeRange`, in whole pixels, rounded down: the device clamps a
    /// point's size to at most the bound, so a size past it draws smaller than written. Whole
    /// pixels, because a description's JSON states the bound to two decimals (2047.94 for
    /// NVIDIA's 2047.9375, 8191.88 for RADV's 8191.875), where the device's query gives the
    /// `f32` itself: both round down to the same number, and Metal's largest point (511 pixels)
    /// is whole. 1 where a description omits the range: Vulkan's bound without `largePoints`,
    /// `pointSizeRange` (1.0, 1.0).
    pub max_point_size: u32,
}

/// The limits of `VK_EXT_mesh_shader` (`VkPhysicalDeviceMeshShaderPropertiesEXT`) the task and mesh
/// stages are held to; the preference hints and the output granularities (which only the output
/// memory size, not held yet, would read) are not read. Each field is the structure's member of the
/// same name (`max_mesh_output_vertices` is `maxMeshOutputVertices`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshLimits {
    pub max_task_work_group_total_count: u32,
    pub max_task_work_group_count: [u32; 3],
    pub max_task_work_group_invocations: u32,
    pub max_task_work_group_size: [u32; 3],
    pub max_task_payload_size: u32,
    pub max_task_shared_memory_size: u32,
    pub max_task_payload_and_shared_memory_size: u32,
    pub max_mesh_work_group_total_count: u32,
    pub max_mesh_work_group_count: [u32; 3],
    pub max_mesh_work_group_invocations: u32,
    pub max_mesh_work_group_size: [u32; 3],
    pub max_mesh_shared_memory_size: u32,
    pub max_mesh_payload_and_shared_memory_size: u32,
    pub max_mesh_output_memory_size: u32,
    pub max_mesh_payload_and_output_memory_size: u32,
    pub max_mesh_output_components: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_mesh_output_layers: u32,
    pub max_mesh_multiview_view_count: u32,
}

impl Limits {
    /// Whether a vertex binding may have `stride` under this device's portability subset: at
    /// least `min_vertex_input_binding_stride_alignment` and a multiple of it
    /// (`VUID-VkVertexInputBindingDescription-stride-04456`); always true on a device without
    /// the rule.
    pub fn portable_vertex_stride(&self, stride: u32) -> bool {
        portable_vertex_stride(self.min_vertex_input_binding_stride_alignment, stride)
    }
}

/// `Limits::max_point_size` of a `pointSizeRange` whose upper bound is `high`: the bound rounded
/// down to whole pixels, 0 for a bound that is not a number or not above zero, and saturating.
pub(crate) fn whole_point_size(high: f32) -> u32 {
    // `as` rounds toward zero, maps NaN to 0 and saturates.
    high.max(0.0) as u32
}

/// Whether a vertex binding may have `stride` under the portability subset's
/// `minVertexInputBindingStrideAlignment` `alignment`: a multiple of it and at least as large
/// (`VUID-VkVertexInputBindingDescription-stride-04456`); always on a device without the rule
/// (`None`).
pub fn portable_vertex_stride(alignment: Option<u32>, stride: u32) -> bool {
    alignment.is_none_or(|a| stride >= a && stride.is_multiple_of(a))
}

/// Per-format usage support for the formats the translator queries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FormatTable {
    pub formats: BTreeMap<vk::Format, BTreeSet<FormatUsage>>,
}

/// The stage-module word bound for a device whose driver faults compiling a module past it
/// ([`DriverWorkaround::RegisterAllocatorOverflowsLargeModules`], ANV's register allocator):
/// its overflow is a 32-bit triangular pair index, and 400,000 words sits below the smallest
/// module observed to fault with margin. [`DeviceProfile::module_word_bound`] returns it where the
/// workaround is taken.
pub(crate) const REGISTER_ALLOCATOR_MODULE_WORD_BOUND: u32 = 400_000;

/// A typed driver workaround: a defect of the driver measured on it, which changes a leaf
/// encoding or refuses the construct the driver runs wrong ([`Needs::DriverDefect`]), never the
/// pointer model or a semantic; or a measured answer of the driver's that the specifications allow
/// and Metal's differs from, which a module records as the precision note it names; or a measured
/// cost of the driver's that chooses between two exact encodings. Each variant comes with a test.
/// The profile derives them from the description's driver identity (`driver_id`, with the vendor
/// where the defect is the GPU's), never from a shader.
///
/// [`Needs::DriverDefect`]: crate::error::Needs::DriverDefect
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriverWorkaround {
    /// The driver's `robustBufferAccess` does not keep a storage buffer's atomic inside the
    /// buffer: SwiftShader (`VK_DRIVER_ID_GOOGLE_SWIFTSHADER`) masks an atomic by the texel's
    /// bounds only through `OpImageTexelPointer`, and an out-of-bounds atomic read-modify-write or
    /// compare-exchange through a storage buffer descriptor changes the memory past the buffer.
    /// Where containment relies on buffer robustness, every atomic through a storage buffer
    /// descriptor view takes the in-shader check instead (`memory::contain::atomic_encoding`):
    /// skipped out of bounds, yielding 0.
    BufferAtomicsUncontained,
    /// In a pass of more than one sample, the GPU drops every storage image write of a fragment
    /// shader that declares early fragment tests: the shader runs (its buffer writes land), but
    /// none of its image writes does. Measured in Metal itself on the Apple M3 Max (macOS 26.7,
    /// 25G229): at 2 and 4 samples, `[[early_fragment_tests]]` with a 2D or 2D-array texture,
    /// `read_write` or `write`, float or integer, with or without a depth attachment or a raster
    /// order group; at 1 sample, or without early tests, every write lands. MoltenVK
    /// (`VK_DRIVER_ID_MOLTENVK`) on an Apple GPU (`vendor_id` 0x106B) converts `EarlyFragmentTests`
    /// to exactly that attribute, so it has the defect. The per-sample storage form of a
    /// framebuffer fetch writes the attachment's samples through a storage image under early tests
    /// where the pipeline has a depth or stencil attachment (or Metal's function declares them), so
    /// there it is refused (`lower::fragment`'s `storage_form`). An explicit imageblock's storage
    /// image, whose writes Metal keeps under early tests at every sample count, declares none
    /// there: its draws run the tests in the shader before the write, and a function declaring
    /// them that discards, returns depth, stencil or a mask, or writes other memory, which the
    /// tests' place would change, is refused (`lower::fragment::imageblock`).
    EarlyTestsDropMultisampledImageWrites,
    /// The driver rasterizes a triangle it clips as several triangles, so in a pass of more
    /// than one sample a fragment shader that runs once per pixel runs once per piece in each
    /// pixel on the pieces' shared edges, each time with that piece's coverage. Measured on
    /// lavapipe (`VK_DRIVER_ID_MESA_LLVMPIPE`), Mesa 26.0.8, with a bare Vulkan program at 4
    /// samples: the standard fullscreen triangle splits along a diagonal of the viewport, exactly
    /// as two triangles sharing it do. `SampleMask` then holds half the pixel's samples in each
    /// invocation, and a `Centroid` input is read at the first covered sample, where one triangle
    /// whose vertices all lie in the clip volume gives the whole coverage and the pixel center.
    /// Link refuses a multisampled pipeline whose fragment function reads `[[sample_mask]]` or a
    /// centroid varying there (`link::graphics`).
    ClippedTrianglesSplitPixelFragments,
    /// The driver's compiler drops a zero-extending mask (`OpBitwiseAnd` with `2^N - 1`) whose
    /// operand is a signed clamp to exactly `N` bits' range (`SMax` and `SMin` in either order,
    /// or `SClamp`) of a sum or difference of two values sign-extended from `N` bits: the
    /// result comes out sign-extended. Measured on the GeForce RTX 3090 (`deviceID` 0x2204,
    /// Ampere GA102) under NVIDIA's 615.71.09 (`VK_DRIVER_ID_NVIDIA_PROPRIETARY`), by
    /// our bare-Vulkan probes: at `N` = 16 and 8, scalar and three lanes, every such program
    /// answered sign-extended, while `OpBitFieldUExtract(x, 0, N)` in place of the mask answered
    /// right; the RTX 5080 (Blackwell) on the same release answered every program right. Derived
    /// for NVIDIA's driver on the 3090's generation, GA10x
    /// ([`NVIDIA_AMPERE_GA10X`](crate::device::NVIDIA_AMPERE_GA10X)), which its compiler
    /// targets alike. Every narrow integer's mask in a 32-bit container is
    /// emitted as `OpBitFieldUExtract(x, 0, N)` there (`lower::arith::ops::mask`), which is the
    /// same bits on every device.
    NarrowMaskAfterSignedClamp,
    /// The driver's compiler folds a `half` round trip, an `OpFConvert` from 32- to 16-bit floats
    /// and its widening back, each decorated `NoContraction`, into the 32-bit value it started
    /// from, in an entry declaring `RoundingModeRTE`, `DenormPreserve` and
    /// `SignedZeroInfNanPreserve` for 16 bits. Vulkan assumes no value-changing transform of an
    /// operation decorated `NoContraction`, and rounds a conversion between types correctly
    /// ("Precision and Operation of SPIR-V Instructions"). Measured on the GeForce RTX 3090
    /// (`deviceID` 0x2204) under NVIDIA's 615.71.09 (`VK_DRIVER_ID_NVIDIA_PROPRIETARY`): of the
    /// 2^32 `f32` values, the round trip left 2,231,288,832 off Metal's `half(float)`, under each
    /// combination of those modes and with or without the decoration. Keyed on NVIDIA's driver
    /// on every GPU, the others unmeasured. Such a profile rounds a `half` in another form
    /// (`lower::arith::narrow::controlled_modes`).
    HalfConversionPairFolded,
    /// The driver refuses a fragment shader that reads both barycentric builtins,
    /// `BaryCoordKHR` and `BaryCoordNoPerspKHR`, at pipeline creation, where Metal compiles a
    /// fragment function that declares both `[[barycentric_coord, center_perspective]]` and
    /// `[[barycentric_coord, center_no_perspective]]`. MoltenVK (`VK_DRIVER_ID_MOLTENVK`)
    /// 1.4.1 converts the shader with SPIRV-Cross `adec7acb`, which throws "Cannot declare both
    /// BaryCoordNV and BaryCoordNoPerspNV in same shader in MSL." whatever the GPU (measured on
    /// an Apple M4: `corpus-pipeline --fixtures` on `moltenvk-m4`). It is keyed on
    /// the driver's release (`device::driver_facts`): a newer release keeps it until a
    /// measurement says otherwise. Such a fragment function is refused there
    /// (`lower::stage_io`'s input builtins); one form alone compiles.
    BothBarycentricFormsRejected,
    /// In a pass of more than one sample, the driver runs a fragment shader that writes memory
    /// once per covered sample even without sample shading, each run reading the same
    /// `SampleMask`: Metal runs a fragment function that writes memory once per covered sample, and
    /// MoltenVK (`VK_DRIVER_ID_MOLTENVK`) on an Apple GPU (`vendor_id` 0x106B) hands Metal such a
    /// shader as such a function. Measured on the Apple M3 Max (macOS 26.7) and the M4 (27.0.1),
    /// MoltenVK 1.4.1: the loop over the covered samples ran each pass once per covered sample of
    /// each covered sample, at 2 and 4 samples, in both legs. It is keyed on the driver's release
    /// and Apple's vendor ID (`device::driver_facts`): a newer release keeps it until a measurement
    /// says otherwise. There a function that reads `[[sample_mask]]` and writes memory runs as
    /// written, without the loop and without sample shading (`CoverageLoop::DriverRunsPerSample`,
    /// `lower::fragment::coverage`); one that reads `[[position]]` beside its sample index or a
    /// sample-qualified varying is refused as this defect at more than one sample, as SPIRV-Cross
    /// then masks the coverage with `1 << gl_SampleID`.
    MemoryWritingFragmentsRunPerSample,
    /// The driver's `InterpolateAtSample`, and its `InterpolateAtOffset` at an offset other than
    /// (0, 0), read values that are not the input's in control flow that some invocations of the
    /// quad have left, as the helper invocations of a partly covered quad leave a loop over their
    /// coverage, which is empty. Measured on RADV (`VK_DRIVER_ID_MESA_RADV`), Mesa 26.0.8 and
    /// 26.2.4, on the Radeon 780M with a bare Vulkan program: in a 1x1 framebuffer, whose quad
    /// holds three helpers, a computed or a constant sample index, or a nonzero offset, inside such
    /// a loop or branch read garbage, with or without sample shading, while the same reads in
    /// control flow the whole quad runs, or in a quad of four covered pixels, read the sample's
    /// position; an offset of (0, 0) reads the center inside the branch too. It is keyed on the
    /// driver's release (`device::driver_facts`). There the loop over the covered samples reads
    /// each sample-qualified varying at every sample a pipeline of the device may have before the
    /// loop, where the whole quad runs, and each pass reads its sample's value
    /// (`lower::fragment::coverage`).
    InterpolationNeedsTheWholeQuad,
    /// The driver reads a mesh shader's store into one component of a vector output (an
    /// `OpAccessChain` to a component of `PrimitiveLineIndicesEXT` or
    /// `PrimitiveTriangleIndicesEXT`) as a store into a later output slot, so the primitive keeps
    /// none of the index and another output takes it. Measured on lavapipe
    /// (`VK_DRIVER_ID_MESA_LLVMPIPE`), Mesa 26.0.8, x86 (the Linux gate's):
    /// every triangle a mesh function wrote one index at a time (`set_index`) drew nothing, and
    /// RADV and NVIDIA drew every one. Mesa's source names the cause and its fix in 26.2.0
    /// ("lavapipe: lower array-deref-of-vec for mesh shader outputs", f5e42331). It is keyed on
    /// the driver's release (`device::driver_facts`): a newer release keeps it until a
    /// measurement says otherwise. There a line or triangle mesh function keeps its indices in
    /// workgroup words, which its entry wrapper writes to the indices whole, one vector per
    /// primitive, after the body (`lower::mesh`).
    MeshIndexComponentStoresMisplaced,
    /// The driver's shader converter writes an `OpCompositeInsert` into the composite it reads
    /// (`c.x = v;`, and the result names `c`) wherever no later read of the composite in the
    /// text it writes stops it, so in a loop a composite defined before the loop is overwritten
    /// and the next iteration reads the inserted element where the module reads the original.
    /// MoltenVK (`VK_DRIVER_ID_MOLTENVK`) 1.4.1 converts with SPIRV-Cross `adec7acb`, whose MSL
    /// backend does this whatever the GPU (its `OpCompositeInsert` case, `can_modify_in_place`):
    /// measured on the Apple M3 Max and the M4 (the dumped MSL of our probe: `_174.x =
    /// (_332 + _207) + _174.x;` in the loop), where three authored cases whose loop builds each
    /// iteration's coordinate from one vector wrote their second iteration at `32 + 2 * lane`
    /// instead of `32 + lane`. It is keyed on the driver's release (`device::driver_facts`): a
    /// newer release keeps it until a measurement says otherwise. There the serializer inserts into
    /// a copy made where the insert stands when the composite is defined outside the innermost loop
    /// holding the insert (`spirv::serialize::serialize_for`), which SPIRV-Cross declares in the
    /// loop.
    CompositeInsertsWrittenInPlace,
    /// The driver's shader converter writes every function the module defines as one to inline
    /// at each call (`static inline __attribute__((always_inline))`), whatever its function
    /// control, so the GPU's compiler compiles a routine's body once per call. MoltenVK
    /// (`VK_DRIVER_ID_MOLTENVK`) 1.4.1 converts with SPIRV-Cross `adec7acb`, whose MSL backend
    /// does this whatever the GPU: measured on the Apple M3 Max and the M4 (about half of corpus
    /// row `69a35b6f`'s compile; a synthetic routine called 300 times compiled in 0.54 to 0.58 s
    /// against 0.11 to 0.13 s as calls). It is keyed on the driver's release
    /// (`device::driver_facts`). There each routine (`spirv::builder::ModuleBuilder::routine`: the
    /// math routines, and the texel conversions, the filters' sums and the cube's face fold) is
    /// declared `DontInline`, the standard SPIR-V statement that its calls stay calls, for a
    /// converter that honours it; elsewhere a routine takes no function control, since NVIDIA's
    /// 615.71.09 honours `DontInline` and ran a routine-heavy kernel 8.1 times as long with it on
    /// the RTX 3090, and Mesa's drivers inline every function whatever the control. It changes no
    /// answer.
    FunctionsInlinedAtEveryCall,
    /// The driver's shader converter writes a value loaded before a loop into the text of each
    /// instruction in the loop that reads it, so the loop reads that memory again on every
    /// iteration, and after a store the loop makes there the next iteration reads the stored
    /// value where the module reads the value loaded before the loop. MoltenVK
    /// (`VK_DRIVER_ID_MOLTENVK`) 1.4.1 converts with SPIRV-Cross `adec7acb`, whose `OpLoad`
    /// forwards its value without counting its reads (so a read in a deeper loop never forces a
    /// temporary), and an `OpCompositeExtract` or `OpVectorShuffle` of it likewise, whatever the
    /// GPU: measured in the MSL it writes for a synthetic module of our own, through the pinned
    /// converter (`scripts/moltenvk-msl-helper`) and the `spirv-cross` CLI. It is keyed on the
    /// driver's release (`device::driver_facts`): a newer release keeps it until a measurement
    /// says otherwise. There the serializer materializes every such value that a loop not
    /// defining it reads, where it is defined (`spirv::serialize::serialize_for`), which
    /// SPIRV-Cross declares there.
    LoadsForwardedIntoLoops,
    /// The driver's shader converter writes a value loaded from a `Workgroup` or `Private`
    /// variable into the text of the instruction that reads it even past an atomic on that
    /// memory, or a call to a function that writes it, so the reader reads what the atomic or the
    /// call left, where the module reads the value loaded before it. MoltenVK
    /// (`VK_DRIVER_ID_MOLTENVK`) 1.4.1 converts with SPIRV-Cross `adec7acb`, whose MSL backend
    /// declares those variables in the entry point and so leaves them out of the variables whose
    /// forwarded loads an atomic or a call invalidates (a plain store invalidates its own
    /// variable's), whatever the GPU: measured on authored case `8b812530` on the Apple M3 Max
    /// and the M4 (our probe logged its MSL, which reads a threadgroup counter after the
    /// sub-word store's atomics that overwrite it), and in the MSL the `spirv-cross` CLI writes for
    /// synthetic modules of our own, where GLSL and a device buffer are written right. It is keyed
    /// on the driver's release (`device::driver_facts`): a newer release keeps it until a
    /// measurement says otherwise. There the serializer materializes each load from such a variable
    /// in a function holding such a write, where it is defined (`spirv::serialize::serialize_for`),
    /// which SPIRV-Cross declares there.
    LocalizedLoadsForwardedPastAtomicsAndCalls,
    /// The driver groups a draw's vertices into subgroups otherwise than Apple's GPUs group them
    /// into simdgroups: ANV (`VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA`) on Mesa 26.2.4 ran the
    /// vertex fixtures `rvc_simd_vertex` and `rvc_simd_vertex_fill` (forty points) in subgroups
    /// of 8 vertices, interleaved (vertices 0, 2, ..., 14 in one, 1, 3, ..., 15 in another),
    /// where the Apple M4 fills a simdgroup with 32 consecutive vertices and RADV and NVIDIA ran
    /// them as the M4 does. Each such subgroup is a Metal simdgroup of 8 active lanes, which
    /// Metal's API allows, since it does not fix which vertices share one. It is keyed on the
    /// driver's release (`device::driver_facts`): a newer release keeps it until a measurement says
    /// otherwise. Where the vertex stage also takes no required subgroup size, a vertex function
    /// that runs a simdgroup or quad operation records `PrecisionNote::VertexSimdgroupGrouping`
    /// (`lower::simd::finish`); no exact path exists.
    VertexGroupingNotApples,
    /// The driver reads a uniform buffer at its lanes' different addresses faster than a
    /// storage buffer. Measured with a fragment function's sRGB color writes, whose lanes read
    /// the color table at their own channel's byte: ANV (`VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA`) on
    /// the UHD 770 of RPL-S and the Iris Xe of RPL-P drew them in up to 1.5 and 4.0 times the time
    /// from a storage buffer as from a uniform buffer, and lavapipe (`VK_DRIVER_ID_MESA_LLVMPIPE`)
    /// on 128-bit vectors (`subgroupSize` 4, an arm64 host) in up to 1.3 times; NVIDIA's driver
    /// (the RTX 3090, the RTX 5080 Laptop GPU) drew them in 3.5 to 8.6 times the time from the
    /// uniform buffer, lavapipe on 256-bit vectors (`subgroupSize` 8, x86_64) in up to 1.5 times,
    /// and RADV in the same time from either. Keyed on ANV whatever its GPU, and on lavapipe at
    /// `subgroupSize` 4. There the color table lives in a uniform buffer
    /// (`lower::arith::srgb::ColorTable`). It changes no answer.
    UniformReadsOutpaceStorageReads,
    /// The GPU's compiler drops the texture writes of every live thread of a simdgroup that
    /// holds a discarded thread, where the fragment function discards (`discard_fragment()`),
    /// then writes a texture, samples a texture bound through an argument buffer, and calls a
    /// `[[clang::optnone]]` function in a branch only discarded threads enter. Measured in plain
    /// Metal on an Apple M4 (macOS 27.0.1, probes): the MSL MoltenVK wrote for a storage-form
    /// fragment of our own, which discards twice, samples a noise texture and writes its color
    /// attachment's storage image, lost the writes in whole 8-by-4-pixel blocks along its discarded
    /// pixels, three runs alike, with or without the raster order group, the fence, or the helper
    /// guard on the write; without the sample, with the `optnone` call plain, in straight-line code
    /// or in a branch live threads enter too, with the sampled texture bound directly, or with the
    /// discard recorded and taken after the write, every write landed, and color outputs landed in
    /// every form. MoltenVK (`VK_DRIVER_ID_MOLTENVK`) on an Apple GPU (`vendor_id` 0x106B) binds
    /// through argument buffers and writes both a demote and a terminate as `discard_fragment()`,
    /// and an `OpFAdd` decorated `NoContraction` as its `optnone` `spvFAdd`, so it has the defect.
    /// There a fragment function that writes storage after such a discard predicates it and is
    /// killed past those writes (`lower::fragment::discard`).
    ///
    /// The defect needs no discard: the quad helpers past a primitive's right and bottom edges are
    /// threads only of their own branch too. Without any discard, the same fragment lost 1,040 to
    /// 1,063 of its 2,584 texels a run in 98 or 99 blocks where only those helpers took the branch
    /// with the `optnone` call (the probe). There the selections whose arms hold such a
    /// call before a storage write send their helper invocations to another arm where that is exact
    /// (`spirv::helper_branches`).
    DiscardDropsSimdgroupStores,
    /// The driver's compiler keeps a call to a function declared `DontInline` a call and
    /// compiles the function once, where it inlines a function without the control at every
    /// call and compiles each copy. A measured cost, not a defect: NVIDIA's
    /// (`VK_DRIVER_ID_NVIDIA_PROPRIETARY`) 615.71.09 on the RTX 3090 created a fragment of 256
    /// calls of one sample emulation in 0.39 to 0.40 s with the routine `DontInline` and in 10.1
    /// to 10.2 s without (a probe, two rounds, shader caches off), and drew it in 15.2 against 10.3
    /// ms; and its creation of the linear sums kept as calls fell from tens of seconds; the RTX
    /// 5080 Laptop GPU's, its measurement in `device::driver_facts`. Mesa's drivers inline every
    /// function whatever its control, so the control changes nothing there, and MoltenVK states it
    /// on every routine already (`FunctionsInlinedAtEveryCall`). It is keyed on the driver's
    /// release (`device::driver_facts`) and on the GPU measured with it alone, as every cost fact
    /// is: a release or a GPU not measured takes no control, the default, whose code and answers
    /// are the same. There each sampling routine
    /// (`spirv::builder::ModuleBuilder::sampling_routine`) is declared `DontInline`
    /// (`spirv::serialize::serialize_for`). It changes no answer, only the
    /// making's and the draw's time.
    DontInlineKeepsCalls,
    /// The device's rasterizer does not split a primitive it clips as Apple's does: Apple's GPU
    /// draws a triangle that extends past the viewport as pieces whose shared edges cross pixels,
    /// and such a pixel's fragments can run once per piece, each reading that piece's coverage as
    /// `[[sample_mask]]` (on an M3 Max and an M4), where a Vulkan rasterizer hands the pixel's one
    /// fragment its coverage by the whole primitive (and lavapipe splits along its own seams,
    /// `ClippedTrianglesSplitPixelFragments`). Vulkan states no split,
    /// and no Vulkan mechanism follows Apple's clipper, so this is the device's answer, not a
    /// defect. Every driver has it but MoltenVK (`VK_DRIVER_ID_MOLTENVK`) on an Apple GPU
    /// (`vendor_id` 0x106B), which draws through Metal and splits as Metal does. A fragment
    /// function compiled for a multisampled pipeline that reads `[[sample_mask]]` records
    /// `PrecisionNote::ClippedCoverageSplit` there (`lower::fragment::coverage`).
    ClippedCoverageNotMetals,
    /// The device's tessellator does not draw a patch's triangles as Apple's does: Metal's
    /// tessellator draws the D3D11 reference's triangles in its own order, each one's first corner
    /// by its stitch step (`lower::tessellation::tessellator`), where Vulkan leaves a patch's
    /// triangle order, first corners and ring triangulation to the implementation (every patch
    /// differs on the RTX 3090, 55-95% of them on RADV and lavapipe). The device's answer, not a
    /// defect. Every driver has it but MoltenVK (`VK_DRIVER_ID_MOLTENVK`) on an Apple GPU
    /// (`vendor_id` 0x106B), which hands the patches to Metal's tessellator. A pipeline drawn
    /// through the device's tessellation stages records `PrecisionNote::TessellationTriangleOrder`
    /// there (`link::tessellation::note_evaluation_stage`).
    TessellatorNotMetals,
    /// The driver's shader compiler faults creating a pipeline from a stage module past a word
    /// bound, so a module the consumer's exact forms grow beyond it is retranslated with those
    /// forms dropped until its largest stage fits ([`DeviceProfile::module_word_bound`]): the one
    /// workaround that changes no leaf encoding and refuses nothing, but bounds a module's size by
    /// trading exactness for the fast form, device by device.
    ///
    /// ANV (`VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA`), Mesa 26.2.4, faults in its `brw` backend's
    /// register allocator: a bitset indexed by the triangular pair index `n·(n-1)/2` of two live
    /// ranges, computed in 32 bits, which overflows once the function's live-range count is large
    /// and reads outside the allocation (`libvulkan_intel.so`, `SEGV_ACCERR`). Measured on the
    /// Iris Xe of RPL-P with a synthetic compute module whose one function keeps a sliding window
    /// of `W` live `float` values: `vkCreateComputePipelines` compiled `W` = 50,000 (750,107 words)
    /// and faulted `W` = 60,000, 65,000 and 92,000 (900,107 words and up); and with the consumer's
    /// fragments, where a module carrying exact sampling alone (455,402 words) compiled but one
    /// carrying every exact answer faulted. The bound is 400,000 words, below the smallest module
    /// observed to combine enough exact forms to fault, with margin: a module carrying exact
    /// sampling alone above it is reduced too, since the fault point for a module's register
    /// pressure is not known exactly and the fault is unrecoverable, where the reduction is a
    /// logged precision note. Keyed on the driver whatever its GPU, whatever its release, in
    /// `device::derive`; a release that fixes the overflow keeps it until a measurement raises or
    /// removes the bound, and the fast-default form of every module stays below it. The consumer
    /// announces each reduced pipeline (`RuntimeFact::ExactFormsOverCompilerLimit`).
    RegisterAllocatorOverflowsLargeModules,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::requirements::Requirements;
    use crate::pipeline::facts::{Mechanism, RuntimeFact};
    use crate::spirv::ids::{BlockId, DemoteTy, Value, VarId};

    /// Vocabulary shared across threads by the consumer's worker pool.
    #[test]
    fn vocabulary_is_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<DeviceProfile>();
        send_sync::<ProfileFields>();
        send_sync::<Requirements>();
        send_sync::<RuntimeFact>();
        send_sync::<Mechanism>();
        send_sync::<DemoteTy>();
        send_sync::<Value>();
        send_sync::<VarId>();
        send_sync::<BlockId>();
        send_sync::<crate::air::ir::value::ValueId>();
        send_sync::<crate::bind::status::ExecutionStatus>();
    }
}
