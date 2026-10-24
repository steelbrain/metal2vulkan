//! Enable sets: the features and extensions a consumer enables at `vkCreateDevice`.
//!
//! A device **exposes** a feature through the structure that holds its member
//! ([`VkFeature::exposure`], the registry's rule, which `device::khronos` and `DeviceDescription::query` read by): a core
//! structure from its version, an extension structure where the extension is enabled, and a
//! promoted extension structure also from the version that promoted it. A format enum value is
//! exposed the same way ([`vk::Format::exposure`], generated from `vk.xml`). Vulkan lets a
//! consumer enable a feature only through an exposed structure, use a format only where its value
//! is exposed, and enable an extension only together with the extensions it depends on, so every
//! enable set this module returns is **closed** under those three rules.
//!
//! Vulkan also forbids enabling some features without another ([`FEATURE_NEEDS`]) or together
//! ([`BUFFER_UPDATE_AFTER_BIND`] beside `robustBufferAccess` on a device without
//! `robustBufferAccessUpdateAfterBind`), so every enable set this module returns also keeps those
//! **co-enable rules**.
//!
//! - [`DeviceDescription::recommended_enables`]: what the core can use on a device described by
//!   what it supports (`DeviceDescription::query`); the consumer merges it into its own list.
//! - [`DeviceDescription::with_enabled`]: that description restricted to what the consumer
//!   enabled, the input of `Device::new` (which checks the exposure invariant and the feature
//!   dependencies).
//! - [`DeviceDescription::required_enables`]: what a compiled artifact's [`Requirements`] need
//!   enabled on the device.
//! - [`Device::required_enables`]: everything any artifact compiled for a device can need.

use std::collections::BTreeSet;

use crate::device::derive::Device;
use crate::device::description::DeviceDescription;
use crate::device::policy::{ContainmentPolicy, DevicePolicy};
use crate::device::requirements::{
    ApiVersion, Requirements, StageCreateFlag, VkExtension, VkFeature, WITHOUT_FORMAT_CAPABILITIES,
};
use crate::error::{Capability, DescriptorError, Error};
use crate::vk;

/// The Vulkan features and device extensions enabled at `vkCreateDevice`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnableSet {
    /// Feature members set in the `VkPhysicalDeviceFeatures2` chain.
    pub features: BTreeSet<VkFeature>,
    /// `ppEnabledExtensionNames`.
    pub extensions: BTreeSet<VkExtension>,
}

impl EnableSet {
    /// The empty set.
    pub fn new() -> EnableSet {
        EnableSet::default()
    }
}

/// How a device exposes a feature: the structure that holds its member, as the Vulkan registry
/// defines it ([`VkFeature::exposure`]).
///
/// A consumer that creates its own `VkDevice` from an [`EnableSet`] reads it to chain each
/// feature: at the device's effective version (its own, capped by the instance's), a member of a
/// core structure is set there, an extension structure's member through an enabled extension
/// that exposes it ([`Exposure::extensions_at`]), and a feature without a member at that version
/// ([`Exposure::has_member`]) sets nothing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exposure {
    /// A member of a core structure of this version (at most the 1.2 floor, so every device the
    /// core accepts exposes it).
    Core(ApiVersion),
    /// A member of an extension structure promoted to core in `version`: exposed from `version`,
    /// below it through any one of `extensions`.
    Promoted {
        /// The core version that promoted the structure.
        version: ApiVersion,
        /// The extensions whose structure holds the member below `version`.
        extensions: &'static [VkExtension],
    },
    /// A member of an extension structure no core version promotes: exposed through any one of
    /// `extensions` on every device.
    Extension(&'static [VkExtension]),
    /// Core functionality **without** a feature member from `version`; below it, a member of
    /// `extension`'s structure. From `version` there is nothing to enable.
    Implied {
        /// The core version from which the functionality has no member.
        version: ApiVersion,
        /// The extensions whose structure holds the member below `version`.
        extensions: &'static [VkExtension],
    },
}

impl Exposure {
    /// Whether a device of effective version `api` with `extensions` enabled exposes it.
    pub fn exposed(self, api: ApiVersion, extensions: &BTreeSet<VkExtension>) -> bool {
        match self {
            Exposure::Core(v) => api >= v,
            Exposure::Promoted { version, .. } | Exposure::Implied { version, .. }
                if api >= version =>
            {
                true
            }
            _ => self
                .extensions_at(api)
                .iter()
                .any(|e| extensions.contains(e)),
        }
    }

    /// Whether the feature is a member the consumer sets at `api` (`false` only for core
    /// functionality without one).
    pub fn has_member(self, api: ApiVersion) -> bool {
        !matches!(self, Exposure::Implied { version, .. } if api >= version)
    }

    /// The extensions any one of which exposes the feature at `api` (empty where a core
    /// version does).
    pub fn extensions_at(self, api: ApiVersion) -> &'static [VkExtension] {
        match self {
            Exposure::Core(_) => &[],
            Exposure::Promoted {
                version,
                extensions,
            }
            | Exposure::Implied {
                version,
                extensions,
            } => {
                if api >= version {
                    &[]
                } else {
                    extensions
                }
            }
            Exposure::Extension(extensions) => extensions,
        }
    }

    /// The extension of a feature no core version exposes, which every device needs enabled
    /// with it (`None` for a feature some core version exposes).
    pub fn only_extension(self) -> Option<VkExtension> {
        match self {
            Exposure::Extension(&[e]) => Some(e),
            _ => None,
        }
    }
}

const fn v(minor: u32) -> ApiVersion {
    ApiVersion { major: 1, minor }
}

/// Defines [`VkFeature::exposure`] and [`VkFeature::ALL`] from one table: an exhaustive `match`
/// without a wildcard (a new `VkFeature` without an exposure does not compile) whose arms also
/// generate the list.
macro_rules! exposures {
    ($($exposure:expr => [$($feature:ident),* $(,)?];)*) => {
        impl VkFeature {
            /// Every feature of the vocabulary, in exposure-table order. `VkFeature` is
            /// `#[non_exhaustive]`, so a consumer's own table over it (the structure member it
            /// sets for each feature at `vkCreateDevice`) checks its coverage against this list.
            pub const ALL: &'static [VkFeature] = &[$($(VkFeature::$feature),*),*];

            /// How a device exposes the feature (the registry's structures, pinned `vk.xml`).
            /// The `device` readers apply the same rule, and `spirv::requirements` tests check
            /// this table against the generated registry enablement table.
            pub fn exposure(self) -> Exposure {
                match self {
                    $($(VkFeature::$feature)|* => $exposure,)*
                }
            }
        }
    };
}

use Exposure::{Core, Extension, Implied, Promoted};
use VkExtension as E;

const ROBUSTNESS2: &[VkExtension] = &[E::ExtRobustness2];
const ATOMIC_FLOAT: &[VkExtension] = &[E::ExtShaderAtomicFloat];
const ATOMIC_FLOAT2: &[VkExtension] = &[E::ExtShaderAtomicFloat2];
const EXPLICIT_LAYOUT: &[VkExtension] = &[E::KhrWorkgroupMemoryExplicitLayout];
const INTERLOCK: &[VkExtension] = &[E::ExtFragmentShaderInterlock];
const RASTER_ORDER: &[VkExtension] = &[E::ExtRasterizationOrderAttachmentAccess];
const DERIVATIVES: &[VkExtension] = &[E::KhrComputeShaderDerivatives];
const SUBGROUP_SIZE: &[VkExtension] = &[E::ExtSubgroupSizeControl];
const DIVISOR: &[VkExtension] = &[E::KhrVertexAttributeDivisor, E::ExtVertexAttributeDivisor];
const ROTATE: &[VkExtension] = &[E::KhrShaderSubgroupRotate];
const BORDER_SWIZZLE: &[VkExtension] = &[E::ExtBorderColorSwizzle];

const fn promoted(minor: u32, extensions: &'static [VkExtension]) -> Exposure {
    Promoted {
        version: v(minor),
        extensions,
    }
}

exposures! {
    // `VkPhysicalDeviceFeatures` (1.0).
    Core(v(0)) => [
        RobustBufferAccess, ShaderInt16, ShaderInt64, ShaderFloat64,
        VertexPipelineStoresAndAtomics, FragmentStoresAndAtomics, GeometryShader,
        TessellationShader, SampleRateShading, DualSrcBlend, LogicOp, IndependentBlend,
        ShaderClipDistance,
        ShaderCullDistance, DrawIndirectFirstInstance, MultiDrawIndirect, DepthClamp,
        FillModeNonSolid, ImageCubeArray, ShaderImageGatherExtended, ShaderResourceMinLod,
        SamplerAnisotropy, ShaderStorageImageMultisample, ShaderStorageImageExtendedFormats,
        ShaderStorageImageReadWithoutFormat, ShaderStorageImageWriteWithoutFormat,
        ShaderUniformBufferArrayDynamicIndexing, ShaderSampledImageArrayDynamicIndexing,
        ShaderStorageBufferArrayDynamicIndexing, ShaderStorageImageArrayDynamicIndexing,
        MultiViewport, DepthBiasClamp, ShaderResourceResidency, OcclusionQueryPrecise,
        FullDrawIndexUint32, ShaderTessellationAndGeometryPointSize, AlphaToOne,
    ];
    // `VkPhysicalDeviceVulkan11Features`.
    Core(v(1)) => [
        StorageBuffer16BitAccess, UniformAndStorageBuffer16BitAccess, StoragePushConstant16,
        StorageInputOutput16, Multiview, ShaderDrawParameters, MultiviewTessellationShader,
        MultiviewGeometryShader,
    ];
    // `VkPhysicalDeviceVulkan12Features`.
    Core(v(2)) => [
        ShaderInt8, ShaderFloat16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess,
        StoragePushConstant8, ShaderBufferInt64Atomics, ShaderSharedInt64Atomics,
        BufferDeviceAddress, ScalarBlockLayout, UniformBufferStandardLayout, VulkanMemoryModel,
        VulkanMemoryModelDeviceScope, ShaderOutputLayer, ShaderOutputViewportIndex,
        SamplerFilterMinmax, SamplerMirrorClampToEdge, RuntimeDescriptorArray,
        DescriptorBindingPartiallyBound, DescriptorBindingVariableDescriptorCount,
        DescriptorBindingSampledImageUpdateAfterBind, DescriptorBindingStorageImageUpdateAfterBind,
        DescriptorBindingStorageBufferUpdateAfterBind,
        DescriptorBindingUniformTexelBufferUpdateAfterBind,
        DescriptorBindingStorageTexelBufferUpdateAfterBind,
        DescriptorBindingUpdateUnusedWhilePending, DrawIndirectCount,
        ShaderSampledImageArrayNonUniformIndexing, ShaderStorageImageArrayNonUniformIndexing,
        ShaderStorageBufferArrayNonUniformIndexing,
        ShaderUniformTexelBufferArrayNonUniformIndexing,
        ShaderStorageTexelBufferArrayNonUniformIndexing, ShaderSubgroupExtendedTypes,
        ShaderUniformTexelBufferArrayDynamicIndexing, ShaderStorageTexelBufferArrayDynamicIndexing,
        ShaderInputAttachmentArrayDynamicIndexing,
    ];
    // Promoted to 1.3.
    promoted(3, &[E::ExtImageRobustness]) => [RobustImageAccess];
    promoted(3, &[E::KhrZeroInitializeWorkgroupMemory]) => [ShaderZeroInitializeWorkgroupMemory];
    promoted(3, &[E::ExtShaderDemoteToHelperInvocation]) => [ShaderDemoteToHelperInvocation];
    promoted(3, &[E::KhrShaderTerminateInvocation]) => [ShaderTerminateInvocation];
    promoted(3, SUBGROUP_SIZE) => [SubgroupSizeControl, ComputeFullSubgroups];
    promoted(3, &[E::KhrMaintenance4]) => [Maintenance4];
    promoted(3, &[E::KhrSynchronization2]) => [Synchronization2];
    // Core in 1.3 without a feature member (`VK_EXT_extended_dynamic_state`, and
    // `VK_EXT_extended_dynamic_state2`'s first feature).
    Implied { version: v(3), extensions: &[E::ExtExtendedDynamicState] } => [ExtendedDynamicState];
    Implied { version: v(3), extensions: &[E::ExtExtendedDynamicState2] } =>
        [ExtendedDynamicState2];
    // Promoted to 1.4.
    promoted(4, &[E::KhrDynamicRenderingLocalRead]) => [DynamicRenderingLocalRead];
    promoted(4, DIVISOR) => [
        VertexAttributeInstanceRateDivisor, VertexAttributeInstanceRateZeroDivisor,
    ];
    promoted(4, ROTATE) => [ShaderSubgroupRotate, ShaderSubgroupRotateClustered];
    promoted(4, &[E::KhrMaintenance5]) => [Maintenance5];
    // Extension-only.
    Extension(ROBUSTNESS2) => [RobustBufferAccess2, RobustImageAccess2, NullDescriptor];
    Extension(&[E::ExtShaderImageAtomicInt64]) => [ShaderImageInt64Atomics];
    Extension(ATOMIC_FLOAT) => [
        ShaderBufferFloat32Atomics, ShaderBufferFloat32AtomicAdd, ShaderSharedFloat32Atomics,
        ShaderSharedFloat32AtomicAdd, ShaderImageFloat32Atomics, ShaderImageFloat32AtomicAdd,
        ShaderBufferFloat64Atomics, ShaderBufferFloat64AtomicAdd, ShaderSharedFloat64Atomics,
        ShaderSharedFloat64AtomicAdd,
    ];
    Extension(ATOMIC_FLOAT2) => [
        ShaderBufferFloat32AtomicMinMax, ShaderSharedFloat32AtomicMinMax,
        ShaderImageFloat32AtomicMinMax, ShaderBufferFloat16Atomics, ShaderBufferFloat16AtomicAdd,
        ShaderBufferFloat16AtomicMinMax, ShaderSharedFloat16Atomics, ShaderSharedFloat16AtomicAdd,
        ShaderSharedFloat16AtomicMinMax, ShaderBufferFloat64AtomicMinMax,
        ShaderSharedFloat64AtomicMinMax,
    ];
    Extension(EXPLICIT_LAYOUT) => [
        WorkgroupMemoryExplicitLayout, WorkgroupMemoryExplicitLayout8BitAccess,
        WorkgroupMemoryExplicitLayout16BitAccess, WorkgroupMemoryExplicitLayoutScalarBlockLayout,
    ];
    Extension(&[E::KhrFragmentShaderBarycentric]) => [FragmentShaderBarycentric];
    Extension(INTERLOCK) => [FragmentShaderPixelInterlock, FragmentShaderSampleInterlock];
    Extension(RASTER_ORDER) => [
        RasterizationOrderColorAttachmentAccess, RasterizationOrderDepthAttachmentAccess,
        RasterizationOrderStencilAttachmentAccess,
    ];
    Extension(&[E::KhrShaderMaximalReconvergence]) => [ShaderMaximalReconvergence];
    Extension(&[E::KhrShaderQuadControl]) => [ShaderQuadControl];
    Extension(&[E::ExtAttachmentFeedbackLoopLayout]) => [AttachmentFeedbackLoopLayout];
    Extension(&[E::ExtExtendedDynamicState3]) => [ExtendedDynamicState3ColorWriteMask];
    Extension(&[E::KhrShaderSubgroupUniformControlFlow]) => [ShaderSubgroupUniformControlFlow];
    Extension(DERIVATIVES) => [ComputeDerivativeGroupQuads, ComputeDerivativeGroupLinear];
    Extension(&[E::KhrMaintenance8]) => [Maintenance8];
    Extension(&[E::KhrShaderFma]) => [ShaderFmaFloat16, ShaderFmaFloat32];
    Extension(BORDER_SWIZZLE) => [BorderColorSwizzle, BorderColorSwizzleFromImage];
    Extension(&[E::ExtDepthBiasControl]) => [DepthBiasControl, FloatRepresentation, DepthBiasExact];
    Extension(&[E::ExtDepthClipEnable]) => [DepthClipEnable];
    Extension(&[E::ExtDepthClampControl]) => [DepthClampControl];
    Extension(&[E::ExtPrimitiveTopologyListRestart]) => [
        PrimitiveTopologyListRestart, PrimitiveTopologyPatchListRestart,
    ];
    Extension(&[E::ExtProvokingVertex]) => [ProvokingVertexLast];
    Extension(&[E::KhrRayQuery]) => [RayQuery];
    Extension(&[E::ExtMeshShader]) => [TaskShader, MeshShader];
    Extension(&[E::KhrAccelerationStructure]) => [AccelerationStructure];
    Extension(&[E::KhrPortabilitySubset]) => [
        ConstantAlphaColorBlendFactors, VertexAttributeAccessBeyondStride, SamplerMipLodBias,
        MutableComparisonSamplers, ShaderSampleRateInterpolationFunctions, MultisampleArrayImage,
        ImageView2DOn3DImage, SeparateStencilMaskRef, ImageViewFormatSwizzle,
    ];
}

/// What the enable-set rules need to know about one extension.
struct ExtensionFacts {
    /// The core enables it where it is available: it exposes a feature or format the core reads,
    /// or the core reads it directly (stencil export, push descriptors, the runtime's sample
    /// locations), or it is mandatory (the portability subset). Extensions that only name
    /// refusals are not.
    consumed: bool,
    /// The core version that promoted everything the core reads from it (then enabling it
    /// there adds nothing).
    core: Option<ApiVersion>,
    /// Vocabulary extensions it requires, each below the given version (`None`: always). From
    /// the pinned `vk.xml` `depends`; requirements outside the vocabulary are instance
    /// extensions or versions the 1.2 floor satisfies, except for extensions the core does not
    /// consume.
    requires: &'static [(VkExtension, Option<ApiVersion>)],
}

fn facts(ext: VkExtension) -> ExtensionFacts {
    let fact = |consumed, core: Option<u32>, requires| ExtensionFacts {
        consumed,
        core: core.map(v),
        requires,
    };
    const DYNAMIC_RENDERING_BELOW_1_3: &[(VkExtension, Option<ApiVersion>)] =
        &[(E::KhrDynamicRendering, Some(v(3)))];
    const SYNCHRONIZATION2_BELOW_1_3: &[(VkExtension, Option<ApiVersion>)] =
        &[(E::KhrSynchronization2, Some(v(3)))];
    match ext {
        E::ExtRobustness2
        | E::ExtShaderImageAtomicInt64
        | E::ExtShaderStencilExport
        | E::ExtFragmentShaderInterlock
        | E::ExtRasterizationOrderAttachmentAccess
        | E::KhrShaderMaximalReconvergence
        | E::KhrShaderSubgroupUniformControlFlow
        | E::KhrFragmentShaderBarycentric
        | E::KhrComputeShaderDerivatives
        | E::KhrWorkgroupMemoryExplicitLayout
        | E::KhrMaintenance8
        | E::KhrShaderFma
        | E::ExtDepthRangeUnrestricted
        | E::ExtDepthBiasControl
        | E::ExtDepthClipEnable
        | E::ExtDepthClampControl
        | E::ExtSampleLocations
        | E::ExtPrimitiveTopologyListRestart
        | E::ExtProvokingVertex
        | E::ExtAttachmentFeedbackLoopLayout
        | E::ExtExtendedDynamicState3
        | E::ExtPostDepthCoverage
        | E::ImgFormatPvrtc
        | E::KhrPortabilitySubset
        | E::ExtShaderAtomicFloat => fact(true, None, &[]),
        E::ExtShaderAtomicFloat2 => fact(true, None, &[(E::ExtShaderAtomicFloat, None)]),
        E::KhrShaderQuadControl => fact(true, None, &[(E::KhrShaderMaximalReconvergence, None)]),
        E::ExtBorderColorSwizzle => fact(true, None, &[(E::ExtCustomBorderColor, None)]),
        E::ExtShaderViewportIndexLayer => fact(true, Some(2), &[]),
        E::ExtImageRobustness
        | E::ExtShaderDemoteToHelperInvocation
        | E::ExtSubgroupSizeControl
        | E::ExtExtendedDynamicState
        | E::ExtExtendedDynamicState2
        | E::KhrShaderTerminateInvocation
        | E::KhrZeroInitializeWorkgroupMemory
        | E::KhrMaintenance4
        | E::KhrSynchronization2
        | E::ExtTextureCompressionAstcHdr => fact(true, Some(3), &[]),
        // The core reads it directly: a storage access without a format by the format's own bits
        // (`DeviceProfile::without_format_by_format_bits`).
        E::KhrFormatFeatureFlags2 => fact(true, Some(3), &[]),
        E::KhrDynamicRenderingLocalRead | E::KhrMaintenance5 => {
            fact(true, Some(4), DYNAMIC_RENDERING_BELOW_1_3)
        }
        E::ExtVertexAttributeDivisor
        | E::KhrVertexAttributeDivisor
        | E::KhrPushDescriptor
        | E::KhrShaderSubgroupRotate => fact(true, Some(4), &[]),
        // Enabled only as another extension's requirement.
        E::ExtCustomBorderColor => fact(false, None, &[]),
        E::KhrDynamicRendering => fact(false, Some(3), &[]),
        // Object and mesh functions. `VK_KHR_spirv_1_4` is core at the 1.2 floor.
        E::ExtMeshShader => fact(true, None, &[]),
        // Named by refusals; the core never enables them.
        E::ExtExternalMemoryHost
        | E::ExtMutableDescriptorType
        | E::KhrShaderUntypedPointers
        | E::KhrCooperativeMatrix
        | E::KhrShaderBfloat16
        | E::QcomTileShading => fact(false, None, &[]),
        // Ray queries. `VK_KHR_spirv_1_4`, `VK_EXT_descriptor_indexing`, and
        // `VK_KHR_buffer_device_address` are core at the 1.2 floor.
        E::KhrRayQuery => fact(true, None, &[(E::KhrAccelerationStructure, None)]),
        E::KhrAccelerationStructure => fact(true, None, &[(E::KhrDeferredHostOperations, None)]),
        // Enabled only as another extension's requirement.
        E::KhrDeferredHostOperations => fact(false, None, &[]),
        E::ExtDescriptorBuffer => fact(false, None, SYNCHRONIZATION2_BELOW_1_3),
    }
}

impl VkExtension {
    /// The vocabulary extensions this extension requires enabled with it on a device of effective
    /// version `api` (vk.xml `depends`; their own requirements are theirs to name).
    pub fn requires(self, api: ApiVersion) -> Vec<VkExtension> {
        facts(self)
            .requires
            .iter()
            .filter(|(_, below)| below.is_none_or(|b| api < b))
            .map(|&(e, _)| e)
            .collect()
    }
}

/// Whether a device of effective version `api` with `extensions` enabled exposes `format`'s
/// enum value ([`vk::Format::exposure`]). Unnamed formats are never exposed.
pub fn format_exposed(
    format: vk::Format,
    api: ApiVersion,
    extensions: &BTreeSet<VkExtension>,
) -> bool {
    format.exposure().is_some_and(|x| {
        x.exposed(api, |name| {
            crate::device::khronos::extension_from_name(name)
                .is_some_and(|e| extensions.contains(&e))
        })
    })
}

/// The vocabulary extensions any one of which exposes `format` below its core version (empty for
/// a format the version exposes or no vocabulary extension defines).
fn format_extensions(format: vk::Format, api: ApiVersion) -> Vec<VkExtension> {
    match format.exposure() {
        Some(x) if !x.core.is_some_and(|c| api >= c) => x
            .extensions
            .iter()
            .filter_map(|n| crate::device::khronos::extension_from_name(n))
            .collect(),
        _ => Vec::new(),
    }
}

/// The capability a device of effective version `api` with `extensions` enabled lacks to use
/// `format`, or `None` where it is exposed: the first vocabulary extension that defines it, else
/// the core version that does.
pub(crate) fn format_exposure_gap(
    format: vk::Format,
    api: ApiVersion,
    extensions: &BTreeSet<VkExtension>,
) -> Option<Capability> {
    if format_exposed(format, api, extensions) {
        return None;
    }
    let x = format.exposure()?;
    Some(match format_extensions(format, api).first() {
        Some(&e) => Capability::Extension(e),
        None => Capability::ApiVersion(x.core?),
    })
}

/// Adds every vocabulary extension the set's extensions require at `api`, transitively.
fn close_requirements(extensions: &mut BTreeSet<VkExtension>, api: ApiVersion) {
    let mut pending: Vec<VkExtension> = extensions.iter().copied().collect();
    while let Some(e) = pending.pop() {
        for dep in e.requires(api) {
            if extensions.insert(dep) {
                pending.push(dep);
            }
        }
    }
}

/// Features Vulkan lets a consumer enable only together with another one, each with the valid
/// usage rule that says so: every such rule of `VkDeviceCreateInfo` and the feature structures
/// (Vulkan 1.4.357 `validusage.json`) whose two features are both in the vocabulary. The
/// multiview rule is stated on `VkPhysicalDeviceMultiviewFeatures`, whose members
/// `VkPhysicalDeviceVulkan11Features` also holds; it applies whichever structure the consumer
/// chains. No companion needs one itself.
pub const FEATURE_NEEDS: &[(VkFeature, VkFeature, &str)] = &[
    (
        VkFeature::RobustBufferAccess2,
        VkFeature::RobustBufferAccess,
        "VUID-VkPhysicalDeviceRobustness2FeaturesKHR-robustBufferAccess2-04000",
    ),
    (
        VkFeature::MultiviewTessellationShader,
        VkFeature::Multiview,
        "VUID-VkPhysicalDeviceMultiviewFeatures-multiviewTessellationShader-00581",
    ),
    (
        VkFeature::MultiviewGeometryShader,
        VkFeature::Multiview,
        "VUID-VkPhysicalDeviceMultiviewFeatures-multiviewGeometryShader-00580",
    ),
];

/// The update-after-bind features of buffer descriptor types, which a device whose
/// `robustBufferAccessUpdateAfterBind` is false may not enable beside `robustBufferAccess`
/// ([`BUFFER_UPDATE_AFTER_BIND_VUID`]). The rule's fourth,
/// `descriptorBindingUniformBufferUpdateAfterBind`, is not in the vocabulary, so no set the core
/// returns or reads names it.
pub const BUFFER_UPDATE_AFTER_BIND: &[VkFeature] = &[
    VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
    VkFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind,
    VkFeature::DescriptorBindingStorageTexelBufferUpdateAfterBind,
];

/// "If the `robustBufferAccess` feature is enabled, and `robustBufferAccessUpdateAfterBind` is
/// `VK_FALSE`, then `descriptorBindingUniformBufferUpdateAfterBind`,
/// `descriptorBindingStorageBufferUpdateAfterBind`,
/// `descriptorBindingUniformTexelBufferUpdateAfterBind`, and
/// `descriptorBindingStorageTexelBufferUpdateAfterBind` must not be enabled" (Vulkan 1.4.357).
pub const BUFFER_UPDATE_AFTER_BIND_VUID: &str = "VUID-VkDeviceCreateInfo-robustBufferAccess-10247";

/// The first co-enable rule `features` breaks on a device whose
/// `robustBufferAccessUpdateAfterBind` is `rba_update_after_bind`, naming the features and the
/// rule's VUID.
fn co_enable_violation(
    features: &BTreeSet<VkFeature>,
    rba_update_after_bind: bool,
) -> Option<String> {
    if !rba_update_after_bind && features.contains(&VkFeature::RobustBufferAccess) {
        if let Some(f) = BUFFER_UPDATE_AFTER_BIND
            .iter()
            .find(|f| features.contains(f))
        {
            return Some(format!(
                "{f:?} with RobustBufferAccess on a device that does not report \
                 robustBufferAccessUpdateAfterBind ({BUFFER_UPDATE_AFTER_BIND_VUID})"
            ));
        }
    }
    FEATURE_NEEDS
        .iter()
        .find(|(f, needs, _)| features.contains(f) && !features.contains(needs))
        .map(|(f, needs, vuid)| format!("{f:?} without {needs:?} ({vuid})"))
}

/// `features` less what the co-enable rules forbid with the rest, on a device whose
/// `robustBufferAccessUpdateAfterBind` is `rba_update_after_bind`. Where `robustBufferAccess`
/// meets [`BUFFER_UPDATE_AFTER_BIND`] on a device without that property, the update-after-bind
/// features go: the core reads `robustBufferAccess` to choose the containment encoding and reads
/// none of them, since the only update-after-bind set it lays out holds image and sampler classes.
/// Then each feature whose [`FEATURE_NEEDS`] companion is missing goes; no companion is itself
/// dropped, so one pass keeps every rule.
fn co_enabled(
    features: impl Iterator<Item = VkFeature>,
    rba_update_after_bind: bool,
) -> BTreeSet<VkFeature> {
    let mut out: BTreeSet<VkFeature> = features.collect();
    if !rba_update_after_bind && out.contains(&VkFeature::RobustBufferAccess) {
        for f in BUFFER_UPDATE_AFTER_BIND {
            out.remove(f);
        }
    }
    let lonely: Vec<VkFeature> = FEATURE_NEEDS
        .iter()
        .filter(|(f, needs, _)| out.contains(f) && !out.contains(needs))
        .map(|&(f, _, _)| f)
        .collect();
    for f in &lonely {
        out.remove(f);
    }
    out
}

/// The feature a stage create flag needs enabled.
fn stage_flag_feature(flag: StageCreateFlag) -> VkFeature {
    match flag {
        StageCreateFlag::RequireFullSubgroups => VkFeature::ComputeFullSubgroups,
        StageCreateFlag::AllowVaryingSubgroupSize => VkFeature::SubgroupSizeControl,
    }
}

/// The closed enable set of `features` and `extensions` on a device of `api`: each feature with a
/// member, the first of its exposing extensions that `available` lists (a feature none exposes
/// is dropped), every consumed extension of `extensions` that `api` does not make redundant, and
/// their requirements (an extension whose requirement is unavailable is dropped, with the
/// features only it exposed).
fn closed(
    api: ApiVersion,
    features: impl Iterator<Item = VkFeature>,
    extensions: &BTreeSet<VkExtension>,
) -> EnableSet {
    let requirements_available = |e: VkExtension| {
        let mut need = BTreeSet::from([e]);
        close_requirements(&mut need, api);
        need.iter().all(|d| extensions.contains(d))
    };
    let usable: BTreeSet<VkExtension> = extensions
        .iter()
        .copied()
        .filter(|&e| requirements_available(e))
        .collect();
    let mut out = EnableSet::new();
    for f in features {
        let x = f.exposure();
        if !x.exposed(api, &usable) {
            continue;
        }
        let exposing = x.extensions_at(api).iter().find(|e| usable.contains(e));
        // A feature an extension the core does not consume exposes (one that names refusals,
        // `ExtensionFacts::consumed`) stays off with it: `VK_EXT_mesh_shader`'s `taskShader` and
        // `meshShader` until the runtime enables mesh shading.
        if exposing.is_some_and(|&e| !facts(e).consumed) {
            continue;
        }
        if let Some(&e) = exposing {
            out.extensions.insert(e);
        }
        if x.has_member(api) {
            out.features.insert(f);
        }
    }
    for &e in &usable {
        let f = facts(e);
        if f.consumed && !f.core.is_some_and(|c| api >= c) {
            out.extensions.insert(e);
        }
    }
    close_requirements(&mut out.extensions, api);
    out
}

fn invalid(detail: String) -> Error {
    Error::InvalidDescriptor(DescriptorError::DeviceDescription { detail })
}

impl DeviceDescription {
    /// The features and extensions the core can use on this device, for a description of what it
    /// **supports** (`DeviceDescription::query`): every listed feature (with its exposing
    /// extension below its core version), every listed extension the core consumes and the
    /// device's version does not make redundant, and their requirements. The consumer enables
    /// what it wants (normally all of it, merged into its own list) and passes the result to
    /// [`with_enabled`](DeviceDescription::with_enabled).
    ///
    /// The policy removes what it makes useless: with `ContainmentPolicy::UnsafeOff`,
    /// `robustBufferAccess` and `robustBufferAccess2` (the profile reads them only to choose a
    /// containment encoding).
    ///
    /// The set keeps Vulkan's co-enable rules: where `robustBufferAccess` stays and the description
    /// does not report `robustBufferAccessUpdateAfterBind`, the update-after-bind features of the
    /// storage buffer, uniform texel buffer, and storage texel buffer descriptor types, which the
    /// core does not read, are left out (VUID-VkDeviceCreateInfo-robustBufferAccess-10247); and
    /// `robustBufferAccess2` without `robustBufferAccess`, or `multiviewTessellationShader` without
    /// `multiview`, is left out. A consumer that wants those update-after-bind features itself on
    /// such a device enables them without `robustBufferAccess` (and `robustBufferAccess2`), and the
    /// profile then contains buffer accesses in the shader.
    pub fn recommended_enables(&self, policy: &DevicePolicy) -> EnableSet {
        let unsafe_off = policy.containment == ContainmentPolicy::UnsafeOff;
        let features = self.features.iter().copied().filter(|&f| {
            !(unsafe_off
                && matches!(
                    f,
                    VkFeature::RobustBufferAccess | VkFeature::RobustBufferAccess2
                ))
        });
        let features = co_enabled(features, self.robust_buffer_access_update_after_bind);
        closed(self.api_version, features.into_iter(), &self.extensions)
    }

    /// This description restricted to what the consumer enabled at `vkCreateDevice`: features
    /// and extensions outside `enabled` are removed, and so are format entries whose value
    /// `enabled` does not expose. Features without a member at the description's version
    /// (`extendedDynamicState` from 1.3) stay.
    ///
    /// Errors (`InvalidDescriptor(DeviceDescription)`): `enabled` names a feature or extension
    /// this description does not list; enables a feature without exposing it (neither its core
    /// version nor any of its extensions); enables an extension without one it requires; omits
    /// `VK_KHR_portability_subset` where the description lists it
    /// (VUID-VkDeviceCreateInfo-pProperties-04451); or breaks a co-enable rule, naming its VUID:
    /// `robustBufferAccess` with a buffer descriptor type's update-after-bind feature where the
    /// description does not report `robustBufferAccessUpdateAfterBind`
    /// (VUID-VkDeviceCreateInfo-robustBufferAccess-10247), `robustBufferAccess2` without
    /// `robustBufferAccess` (VUID-VkPhysicalDeviceRobustness2FeaturesKHR-robustBufferAccess2-04000),
    /// or `multiviewTessellationShader` without `multiview`
    /// (VUID-VkPhysicalDeviceMultiviewFeatures-multiviewTessellationShader-00581).
    pub fn with_enabled(mut self, enabled: &EnableSet) -> Result<DeviceDescription, Error> {
        let api = self.api_version;
        if let Some(f) = enabled.features.difference(&self.features).next() {
            return Err(invalid(format!(
                "enables {f:?}, which the description does not list"
            )));
        }
        if let Some(e) = enabled.extensions.difference(&self.extensions).next() {
            return Err(invalid(format!(
                "enables {e:?}, which the description does not list"
            )));
        }
        for &f in &enabled.features {
            let x = f.exposure();
            if !x.exposed(api, &enabled.extensions) {
                return Err(invalid(format!(
                    "enables {f:?} without exposing it (enable one of {:?})",
                    x.extensions_at(api)
                )));
            }
        }
        for &e in &enabled.extensions {
            if let Some(d) = e
                .requires(api)
                .into_iter()
                .find(|d| !enabled.extensions.contains(d))
            {
                return Err(invalid(format!("enables {e:?} without {d:?}")));
            }
        }
        let portability = VkExtension::KhrPortabilitySubset;
        if self.extensions.contains(&portability) && !enabled.extensions.contains(&portability) {
            return Err(invalid(format!(
                "does not enable {portability:?}, which the device lists"
            )));
        }
        if let Some(rule) = co_enable_violation(
            &enabled.features,
            self.robust_buffer_access_update_after_bind,
        ) {
            return Err(invalid(format!("enables {rule}")));
        }
        self.features
            .retain(|&f| enabled.features.contains(&f) || !f.exposure().has_member(api));
        self.extensions.retain(|e| enabled.extensions.contains(e));
        let extensions = &self.extensions;
        self.formats
            .formats
            .retain(|&format, _| format_exposed(format, api, extensions));
        Ok(self)
    }

    /// What the consumer must enable on this device to run an artifact with `requirements`
    /// (compiled for a `Device` made from this description): every required feature with a
    /// member (stage create flags add theirs), each one's exposing extension below its core
    /// version, the required extensions, the extension exposing each required format below its
    /// core version, `VK_KHR_portability_subset` where the description lists it, and every
    /// extension those require. A storage capability without a format (`Requirements` lists it
    /// without a feature) takes `shaderStorageImage{Read,Write}WithoutFormat` where the
    /// description lists it, nothing at Vulkan 1.3, else `VK_KHR_format_feature_flags2`.
    ///
    /// Errors: the first capability the description lacks (the API version, a feature, or an
    /// extension, the one a missing exposure or requirement names first). Properties, limits,
    /// and format support are not enables; `DeviceProfile::check` compares them.
    pub fn required_enables(&self, requirements: &Requirements) -> Result<EnableSet, Capability> {
        let api = self.api_version;
        if let Some(v) = requirements.api_version.filter(|&v| v > api) {
            return Err(Capability::ApiVersion(v));
        }
        let pick = |alternatives: &[VkExtension]| {
            alternatives
                .iter()
                .find(|e| self.extensions.contains(e))
                .or(alternatives.first())
                .copied()
        };
        let mut out = EnableSet::new();
        let features = requirements.features.iter().copied().chain(
            requirements
                .stage_flags
                .iter()
                .map(|&f| stage_flag_feature(f)),
        );
        for f in features {
            if !self.has(f) {
                return Err(Capability::Feature(f));
            }
            let x = f.exposure();
            out.extensions.extend(pick(x.extensions_at(api)));
            if x.has_member(api) {
                out.features.insert(f);
            }
        }
        for (capability, feature) in WITHOUT_FORMAT_CAPABILITIES {
            if !requirements.spirv_capabilities.contains(&capability) {
                continue;
            }
            if self.has(feature) {
                let x = feature.exposure();
                out.extensions.extend(pick(x.extensions_at(api)));
                if x.has_member(api) {
                    out.features.insert(feature);
                }
            } else if api < ApiVersion::V1_3 {
                let flags2 = VkExtension::KhrFormatFeatureFlags2;
                if !self.extensions.contains(&flags2) {
                    return Err(Capability::Feature(feature));
                }
                out.extensions.insert(flags2);
            }
        }
        out.extensions
            .extend(requirements.extensions.iter().copied());
        for &(format, _) in &requirements.formats {
            if let Some(gap) = format_exposure_gap(format, api, &self.extensions) {
                return Err(gap);
            }
            out.extensions.extend(pick(&format_extensions(format, api)));
        }
        if self.extensions.contains(&VkExtension::KhrPortabilitySubset) {
            out.extensions.insert(VkExtension::KhrPortabilitySubset);
        }
        close_requirements(&mut out.extensions, api);
        if let Some(&e) = out.extensions.difference(&self.extensions).next() {
            return Err(Capability::Extension(e));
        }
        Ok(out)
    }
}

impl Device {
    /// Everything any artifact compiled for this device can need enabled: the profile's enabled
    /// features, their exposing extensions, the enabled extensions the core consumes, and their
    /// requirements: exactly what the output may declare. A consumer that enabled
    /// [`DeviceDescription::recommended_enables`] (restricted by
    /// [`DeviceDescription::with_enabled`]) enabled at least this.
    ///
    /// It keeps the co-enable rules as the recommendation does, under the description's
    /// `robustBufferAccessUpdateAfterBind`: a description listing `robustBufferAccess` and buffer
    /// update-after-bind features without it (a committed profile read whole, as the tests and the
    /// tier table use them) states a device no consumer can create, and the set leaves out those
    /// update-after-bind features, which no artifact needs.
    pub fn required_enables(&self) -> EnableSet {
        let f = self.profile().fields();
        let features = co_enabled(
            f.enabled.features.iter().copied(),
            self.robust_buffer_access_update_after_bind(),
        );
        closed(f.api_version, features.into_iter(), &f.enabled.extensions)
    }
}

/// `Device::new`'s check of the description invariants: every listed feature is exposed by the
/// description's version or listed extensions, and every listed feature's [`FEATURE_NEEDS`]
/// companion is listed. The rule of [`BUFFER_UPDATE_AFTER_BIND_VUID`] is not checked here: a
/// committed profile read whole lists both sides of it on three profiles, and the profile the core
/// derives from such a description translates exactly as from its creatable restriction.
pub(crate) fn check_invariants(desc: &DeviceDescription) -> Result<(), Error> {
    for &f in &desc.features {
        let x = f.exposure();
        if !x.exposed(desc.api_version, &desc.extensions) {
            return Err(invalid(format!(
                "lists {f:?} without exposing it (list one of {:?})",
                x.extensions_at(desc.api_version)
            )));
        }
    }
    if let Some((f, needs, vuid)) = FEATURE_NEEDS
        .iter()
        .find(|(f, needs, _)| desc.has(*f) && !desc.has(*needs))
    {
        return Err(invalid(format!("lists {f:?} without {needs:?} ({vuid})")));
    }
    Ok(())
}
