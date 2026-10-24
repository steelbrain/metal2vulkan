//! Vulkan capability vocabulary and the `Requirements` type.
//!
//! These enums name the subset of Vulkan the new core consumes: device features, extensions,
//! properties, limits, and format usages. They are the payload of `error::Capability` (what a
//! refusal says would make an input translatable) and of [`Requirements`] (what an emitted module
//! needs enabled). `spirv::requirements` derives `Requirements` values; `device` checks them
//! against a profile. Every enum is `#[non_exhaustive]`; a newly consumed item is a one-line
//! append.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::device::profile::Independence;
use crate::pipeline::sym::Stage;
use crate::vk;

/// A Vulkan device feature the new core can require, named after its `VkPhysicalDevice*Features`
/// member (core 1.0–1.3 structs and the extension structs this crate consumes).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VkFeature {
    // Robustness.
    RobustBufferAccess,
    RobustBufferAccess2,
    RobustImageAccess,
    RobustImageAccess2,
    NullDescriptor,
    // Arithmetic widths.
    ShaderInt8,
    ShaderInt16,
    ShaderInt64,
    ShaderFloat16,
    ShaderFloat64,
    // Narrow storage.
    StorageBuffer8BitAccess,
    UniformAndStorageBuffer8BitAccess,
    StoragePushConstant8,
    StorageBuffer16BitAccess,
    UniformAndStorageBuffer16BitAccess,
    StoragePushConstant16,
    StorageInputOutput16,
    // Atomics.
    ShaderBufferInt64Atomics,
    ShaderSharedInt64Atomics,
    ShaderImageInt64Atomics,
    ShaderBufferFloat32Atomics,
    ShaderBufferFloat32AtomicAdd,
    ShaderBufferFloat32AtomicMinMax,
    ShaderSharedFloat32Atomics,
    ShaderSharedFloat32AtomicAdd,
    ShaderSharedFloat32AtomicMinMax,
    ShaderImageFloat32Atomics,
    ShaderImageFloat32AtomicAdd,
    ShaderImageFloat32AtomicMinMax,
    // Memory model and addressing.
    BufferDeviceAddress,
    ScalarBlockLayout,
    UniformBufferStandardLayout,
    VulkanMemoryModel,
    VulkanMemoryModelDeviceScope,
    WorkgroupMemoryExplicitLayout,
    WorkgroupMemoryExplicitLayout8BitAccess,
    WorkgroupMemoryExplicitLayout16BitAccess,
    ShaderZeroInitializeWorkgroupMemory,
    // Stage stores.
    VertexPipelineStoresAndAtomics,
    FragmentStoresAndAtomics,
    // Stages and stage builtins.
    ShaderDrawParameters,
    GeometryShader,
    TessellationShader,
    SampleRateShading,
    DualSrcBlend,
    IndependentBlend,
    ShaderClipDistance,
    ShaderCullDistance,
    ShaderOutputLayer,
    ShaderOutputViewportIndex,
    Multiview,
    FragmentShaderBarycentric,
    FragmentShaderPixelInterlock,
    FragmentShaderSampleInterlock,
    ShaderDemoteToHelperInvocation,
    ShaderTerminateInvocation,
    DynamicRenderingLocalRead,
    RasterizationOrderColorAttachmentAccess,
    RasterizationOrderDepthAttachmentAccess,
    RasterizationOrderStencilAttachmentAccess,
    VertexAttributeInstanceRateDivisor,
    VertexAttributeInstanceRateZeroDivisor,
    DrawIndirectFirstInstance,
    MultiDrawIndirect,
    /// `drawIndirectCount` (Vulkan 1.2, or `VK_KHR_draw_indirect_count`): an indirect draw whose
    /// draw count is GPU data (an indirect object mesh draw's pieces, `bind::mesh_arguments`).
    DrawIndirectCount,
    DepthClamp,
    FillModeNonSolid,
    // Images and samplers.
    ImageCubeArray,
    ShaderImageGatherExtended,
    ShaderResourceMinLod,
    SamplerAnisotropy,
    SamplerFilterMinmax,
    SamplerMirrorClampToEdge,
    ShaderStorageImageMultisample,
    ShaderStorageImageExtendedFormats,
    ShaderStorageImageReadWithoutFormat,
    ShaderStorageImageWriteWithoutFormat,
    // Descriptor indexing.
    RuntimeDescriptorArray,
    DescriptorBindingPartiallyBound,
    DescriptorBindingVariableDescriptorCount,
    DescriptorBindingSampledImageUpdateAfterBind,
    DescriptorBindingStorageImageUpdateAfterBind,
    DescriptorBindingStorageBufferUpdateAfterBind,
    DescriptorBindingUniformTexelBufferUpdateAfterBind,
    DescriptorBindingStorageTexelBufferUpdateAfterBind,
    ShaderSampledImageArrayNonUniformIndexing,
    ShaderStorageImageArrayNonUniformIndexing,
    ShaderStorageBufferArrayNonUniformIndexing,
    ShaderUniformTexelBufferArrayNonUniformIndexing,
    ShaderStorageTexelBufferArrayNonUniformIndexing,
    // Subgroups and convergence.
    SubgroupSizeControl,
    ComputeFullSubgroups,
    ShaderSubgroupExtendedTypes,
    ShaderMaximalReconvergence,
    ShaderSubgroupUniformControlFlow,
    ComputeDerivativeGroupQuads,
    ComputeDerivativeGroupLinear,
    // Pipeline construction.
    Maintenance4,
    Synchronization2,
    /// `VK_EXT_extended_dynamic_state` (core functionality without a feature bit in 1.3):
    /// dynamic vertex-buffer strides.
    ExtendedDynamicState,
    // Appended (requirement derivation needs them).
    ShaderUniformBufferArrayDynamicIndexing,
    ShaderSampledImageArrayDynamicIndexing,
    ShaderStorageBufferArrayDynamicIndexing,
    ShaderStorageImageArrayDynamicIndexing,
    ShaderUniformTexelBufferArrayDynamicIndexing,
    ShaderStorageTexelBufferArrayDynamicIndexing,
    ShaderInputAttachmentArrayDynamicIndexing,
    ShaderBufferFloat16Atomics,
    ShaderBufferFloat16AtomicAdd,
    ShaderBufferFloat16AtomicMinMax,
    ShaderSharedFloat16Atomics,
    ShaderSharedFloat16AtomicAdd,
    ShaderSharedFloat16AtomicMinMax,
    ShaderBufferFloat64Atomics,
    ShaderBufferFloat64AtomicAdd,
    ShaderBufferFloat64AtomicMinMax,
    ShaderSharedFloat64Atomics,
    ShaderSharedFloat64AtomicAdd,
    ShaderSharedFloat64AtomicMinMax,
    WorkgroupMemoryExplicitLayoutScalarBlockLayout,
    ShaderSubgroupRotate,
    ShaderSubgroupRotateClustered,
    MultiViewport,
    // Appended.
    /// `VK_KHR_maintenance8`'s `maintenance8`: among others, the `Offset` image operand on
    /// `OpImageSample*` and `OpImageFetch` (VUID-RuntimeSpirv-Offset-10213).
    Maintenance8,
    // Appended.
    /// `VK_EXT_border_color_swizzle`'s `borderColorSwizzle`: defined opaque-black and custom
    /// border colors through an image view with a non-identity component mapping.
    BorderColorSwizzle,
    /// `VK_EXT_border_color_swizzle`'s `borderColorSwizzleFromImage`: the implementation applies
    /// the view's component mapping to the border color without a per-sampler mapping.
    BorderColorSwizzleFromImage,
    // Appended.
    /// `VK_KHR_ray_query`'s `rayQuery`: the `RayQueryKHR` capability.
    RayQuery,
    /// `VK_KHR_acceleration_structure`'s `accelerationStructure`: acceleration structures and
    /// their descriptors.
    AccelerationStructure, // Appended.
    /// `multiviewTessellationShader` (Vulkan 1.1): a multiview pipeline with tessellation stages.
    MultiviewTessellationShader,
    // Appended.
    /// `depthBiasClamp` (Vulkan 1.0): a nonzero `depthBiasClamp`, as Metal's `setDepthBias:` can
    /// state (`VUID-vkCmdSetDepthBias-depthBiasClamp-00790`).
    DepthBiasClamp,
    // `VkPhysicalDevicePortabilitySubsetFeaturesKHR` members. Each lifts a restriction
    // `VK_KHR_portability_subset` places on a device that lists it (and so enables it); a device
    // without the extension has none of them, so the core asks for one only on a portability
    // device.
    /// `constantAlphaColorBlendFactors`: `CONSTANT_ALPHA` or `ONE_MINUS_CONSTANT_ALPHA` as a
    /// color blend factor
    /// (`VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04454`, `-04455`).
    ConstantAlphaColorBlendFactors,
    /// `vertexAttributeAccessBeyondStride`: a vertex attribute that ends past its binding's
    /// stride (`VUID-VkVertexInputAttributeDescription-vertexAttributeAccessBeyondStride-04457`).
    VertexAttributeAccessBeyondStride,
    /// `samplerMipLodBias`: a nonzero sampler `mipLodBias`
    /// (`VUID-VkSamplerCreateInfo-samplerMipLodBias-04467`).
    SamplerMipLodBias,
    /// `mutableComparisonSamplers`: a comparison sampler
    /// (`VUID-VkSamplerCreateInfo-mutableComparisonSamplers-04450`).
    MutableComparisonSamplers,
    // Appended.
    /// `shaderSampleRateInterpolationFunctions` (a portability-subset member): the
    /// `InterpolationFunction` capability and `GLSL.std.450`'s `InterpolateAt*`
    /// (`VUID-RuntimeSpirv-shaderSampleRateInterpolationFunctions-06325`).
    ShaderSampleRateInterpolationFunctions,
    // Appended.
    /// `shaderResourceResidency` (Vulkan 1.0): the `SparseResidency` capability, the
    /// `OpImageSparse*` accesses that return a residency code, and
    /// `OpImageSparseTexelsResident`.
    ShaderResourceResidency,
    // Appended.
    /// `VK_KHR_shader_fma`'s `shaderFmaFloat32`: `OpFmaKHR` on 32-bit floats
    /// (`VUID-RuntimeSpirv-shaderFmaFloat32-10978`), which Vulkan's precision table makes correctly
    /// rounded, where `GLSL.std.450` `Fma` may round twice.
    ShaderFmaFloat32,
    // Appended.
    /// `maintenance5` (`VK_KHR_maintenance5`, core in 1.4): among others, a buffer view's usage
    /// narrowed by `VkBufferUsageFlags2CreateInfo` (`VkBufferViewCreateInfo`'s `pNext`). The
    /// vocabulary names it; nothing the translation emits requires it.
    Maintenance5,
    /// `occlusionQueryPrecise` (`VkPhysicalDeviceFeatures`, Vulkan 1.0): an occlusion query
    /// begun with `VK_QUERY_CONTROL_PRECISE_BIT` counts the samples that pass, where one without
    /// it may answer any nonzero count for some. Named for the runtime's counting visibility
    /// results; nothing the translation emits requires it.
    OcclusionQueryPrecise,
    /// `VK_EXT_depth_bias_control`'s `depthBiasControl`: a depth bias may state its
    /// representation (`VkDepthBiasRepresentationInfoEXT`) and is set by `vkCmdSetDepthBias2EXT`.
    /// Named for the runtime's depth bias (Metal's exact bias on unorm depth formats); nothing the
    /// translation emits requires it, nor either of the two that follow.
    DepthBiasControl,
    /// `VK_EXT_depth_bias_control`'s `floatRepresentation`: the representation
    /// `VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT`.
    FloatRepresentation,
    /// `VK_EXT_depth_bias_control`'s `depthBiasExact`: `VkDepthBiasRepresentationInfoEXT`'s
    /// `depthBiasExact`, a depth bias the implementation applies without scaling it to ensure a
    /// minimum resolvable difference.
    DepthBiasExact,
    // The render runtime's vocabulary: nothing the translation emits requires any of these.
    /// `multisampleArrayImage` (a `VkPhysicalDevicePortabilitySubsetFeaturesKHR` member): a
    /// multisampled image with more than one array layer
    /// (`VUID-VkImageCreateInfo-multisampleArrayImage-04460`), as Metal's
    /// `MTLTextureType2DMultisampleArray` is. Like every portability member, a device without
    /// `VK_KHR_portability_subset` has no such restriction.
    MultisampleArrayImage,
    /// `imageView2DOn3DImage` (a portability-subset member): a 3D image created
    /// `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`
    /// (`VUID-VkImageCreateInfo-imageView2DOn3DImage-04459`), so that a 2D view of one slice, as
    /// a render pass attachment on a slice of a Metal 3D texture needs, can be made.
    ImageView2DOn3DImage,
    /// `separateStencilMaskRef` (a portability-subset member): unequal front and back stencil
    /// references with culling off
    /// (`VUID-VkPipelineDepthStencilStateCreateInfo-separateStencilMaskRef-04453`), as Metal's
    /// `setStencilFrontReferenceValue:backReferenceValue:` can set.
    SeparateStencilMaskRef,
    /// `fullDrawIndexUint32` (`VkPhysicalDeviceFeatures`, Vulkan 1.0): an indexed draw may fetch
    /// any 32-bit index (`maxDrawIndexedIndexValue` is `2^32 - 1`); without it, only indices up to
    /// [`VkLimit::MaxDrawIndexedIndexValue`]. Named for the runtime's index clamp.
    FullDrawIndexUint32,
    /// `VK_EXT_depth_clip_enable`'s `depthClipEnable`: a pipeline may clip at the near and far
    /// planes independently of `depthClampEnable`, so a depth clamped as Metal clamps a viewport
    /// depth range past `[0, 1]` is still clipped as Metal clips it.
    DepthClipEnable,
    /// `VK_EXT_depth_clamp_control`'s `depthClampControl`: a pipeline may clamp depth to a range
    /// it states (`VkPipelineViewportDepthClampControlCreateInfoEXT`), `[0, 1]` for Metal, instead
    /// of the viewport's depth range.
    DepthClampControl,
    /// `VK_EXT_primitive_topology_list_restart`'s `primitiveTopologyListRestart`: primitive
    /// restart on a point, line or triangle list. Named for the runtime's primitive restart on
    /// list topologies, once an authored case measures Metal's.
    PrimitiveTopologyListRestart,
    /// `VK_EXT_primitive_topology_list_restart`'s `primitiveTopologyPatchListRestart`: primitive
    /// restart on a patch list.
    PrimitiveTopologyPatchListRestart,
    // Appended.
    /// `imageViewFormatSwizzle` (a portability-subset member): an image view whose component
    /// mapping is not the identity (`VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`),
    /// as a format read through a swizzled view (`metal::format`'s `SwizzledView`) and a Metal
    /// texture made with a swizzle need.
    ImageViewFormatSwizzle,
    // Appended.
    /// `VK_EXT_mesh_shader`'s `taskShader`: an object function (`TaskEXT`).
    TaskShader,
    /// `VK_EXT_mesh_shader`'s `meshShader`: a mesh function (`MeshEXT`), and a fragment function
    /// reading its primitive outputs (`PerPrimitiveEXT`).
    MeshShader,
    // Appended (residency part C).
    /// `descriptorBindingUpdateUnusedWhilePending` (Vulkan 1.2): a descriptor of a binding
    /// created `VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT` may be written while a
    /// command buffer that does not use it is pending, as the set-1 heap's arrays are where the
    /// profile has it (link): the runtime's device-wide heap writes a texture made while other
    /// command buffers run.
    DescriptorBindingUpdateUnusedWhilePending,
    // Appended.
    /// `VK_EXT_extended_dynamic_state2`'s `extendedDynamicState2` (core functionality without a
    /// feature bit in 1.3): among others, a dynamic primitive restart
    /// (`PRIMITIVE_RESTART_ENABLE`), which with [`VkFeature::ExtendedDynamicState`]'s dynamic
    /// topology lets one pipeline serve a class's list and strip topologies.
    ExtendedDynamicState2,
    // Appended.
    /// `VK_KHR_shader_quad_control`'s `shaderQuadControl`: a fragment entry's `RequireFullQuadsKHR`
    /// mode, under which every quad starts with its four invocations, helpers included, as Metal
    /// runs a fragment function's simdgroup and quad operations.
    ShaderQuadControl,
    // Appended.
    /// `shaderTessellationAndGeometryPointSize` (Vulkan 1.0): a geometry stage's `PointSize` (the
    /// `GeometryPointSize` capability). A geometry stage without it draws points of size 1.
    ShaderTessellationAndGeometryPointSize,
    /// `multiviewGeometryShader` (Vulkan 1.1): a multiview pipeline with a geometry stage.
    MultiviewGeometryShader,
    // Appended.
    /// `VK_EXT_attachment_feedback_loop_layout`'s `attachmentFeedbackLoopLayout`: the layout
    /// `VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`, the image usage
    /// `VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT` and the pipeline flag
    /// `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`, under which a draw samples a
    /// color attachment of its own subpass (a feedback loop). A render pipeline that reads a
    /// texture its pass renders through a snapshot requires it.
    AttachmentFeedbackLoopLayout,
    /// `VK_EXT_extended_dynamic_state3`'s `extendedDynamicState3ColorWriteMask`: the dynamic
    /// state `VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT`, set by `vkCmdSetColorWriteMaskEXT`.
    ExtendedDynamicState3ColorWriteMask,
    // Appended.
    /// `VK_KHR_shader_fma`'s `shaderFmaFloat16`: `OpFmaKHR` on 16-bit floats
    /// (`VUID-RuntimeSpirv-shaderFmaFloat16-10977`), which Vulkan's precision table makes correctly
    /// rounded: a `half` fused multiply-add in one instruction under the 16-bit float controls.
    ShaderFmaFloat16,
    // Appended.
    /// `VK_EXT_provoking_vertex`'s `provokingVertexLast`: a pipeline whose rasterization state
    /// chains `VkPipelineRasterizationProvokingVertexStateCreateInfoEXT` with
    /// `VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT`
    /// (`VUID-VkPipelineRasterizationProvokingVertexStateCreateInfoEXT-provokingVertexMode-04883`),
    /// under which a primitive's flat inputs read its last vertex, as Metal's
    /// `setProvokingVertexMode:` of 1 draws.
    ProvokingVertexLast,
    /// `logicOp` (Vulkan 1.0): `VkPipelineColorBlendStateCreateInfo::logicOpEnable`, a logical
    /// operation on the codes of each normalized and integer color attachment. A render pipeline
    /// whose private `logicOperation` the device applies requires it (`GraphicsState::logic_op`).
    LogicOp,
    // Appended.
    /// `alphaToOne` (Vulkan 1.0): `VkPipelineMultisampleStateCreateInfo::alphaToOneEnable`, which
    /// replaces the fragment's alpha with one before the blend reads it
    /// (`VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785`). A render pipeline
    /// whose descriptor sets `alphaToOneEnabled` requires it (`GraphicsState::alpha_to_one`).
    AlphaToOne,
    // Appended.
    /// `VK_EXT_line_rasterization`'s `bresenhamLines`: a pipeline whose rasterization state chains
    /// `VkPipelineRasterizationLineStateCreateInfoEXT` with
    /// `VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT`
    /// (`VUID-VkPipelineRasterizationLineStateCreateInfoKHR-lineRasterizationMode-02768`), under
    /// which a line covers the pixels whose diamond it exits, as Metal rasterizes a line. Without
    /// it a line takes the device's own default mode, which on a device whose `strictLines` is set
    /// covers every pixel the line's rectangle contains and so covers pixels Metal leaves
    /// untouched (`pipeline/graphics.rs`).
    BresenhamLines,
}

/// A Vulkan device extension the new core can require.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VkExtension {
    ExtRobustness2,
    ExtImageRobustness,
    ExtShaderAtomicFloat,
    ExtShaderAtomicFloat2,
    ExtShaderImageAtomicInt64,
    ExtShaderStencilExport,
    ExtShaderDemoteToHelperInvocation,
    ExtShaderViewportIndexLayer,
    ExtFragmentShaderInterlock,
    ExtSubgroupSizeControl,
    ExtRasterizationOrderAttachmentAccess,
    ExtVertexAttributeDivisor,
    ExtExternalMemoryHost,
    ExtExtendedDynamicState,
    ExtMutableDescriptorType,
    ExtDescriptorBuffer,
    ExtMeshShader,
    KhrShaderTerminateInvocation,
    KhrShaderMaximalReconvergence,
    KhrShaderSubgroupUniformControlFlow,
    KhrFragmentShaderBarycentric,
    KhrComputeShaderDerivatives,
    KhrWorkgroupMemoryExplicitLayout,
    KhrZeroInitializeWorkgroupMemory,
    KhrDynamicRenderingLocalRead,
    KhrVertexAttributeDivisor,
    KhrPushDescriptor,
    KhrMaintenance4,
    KhrSynchronization2,
    KhrShaderUntypedPointers,
    KhrCooperativeMatrix,
    KhrShaderBfloat16,
    KhrRayQuery,
    QcomTileShading,
    KhrShaderSubgroupRotate,
    KhrMaintenance8,
    // Appended.
    /// `VK_KHR_maintenance5` (core in 1.4): among others, `VK_FORMAT_A8_UNORM`.
    KhrMaintenance5,
    /// `VK_EXT_texture_compression_astc_hdr` (core in 1.3): the ASTC `SFLOAT` formats.
    ExtTextureCompressionAstcHdr,
    /// `VK_IMG_format_pvrtc` (never promoted; deprecated): the PVRTC formats.
    ImgFormatPvrtc,
    /// `VK_EXT_border_color_swizzle`, which requires `VK_EXT_custom_border_color`.
    ExtBorderColorSwizzle,
    /// `VK_EXT_custom_border_color`.
    ExtCustomBorderColor,
    /// `VK_KHR_portability_subset`: a device that lists it must have it enabled.
    KhrPortabilitySubset,
    /// `VK_KHR_dynamic_rendering` (core in 1.3), which `VK_KHR_maintenance5` and
    /// `VK_KHR_dynamic_rendering_local_read` require below 1.3.
    KhrDynamicRendering,
    // Appended.
    /// `VK_KHR_acceleration_structure`, which `VK_KHR_ray_query` requires.
    KhrAccelerationStructure,
    /// `VK_KHR_deferred_host_operations`, which `VK_KHR_acceleration_structure` requires.
    KhrDeferredHostOperations,
    // Appended.
    /// `VK_KHR_shader_fma` (never promoted): `OpFmaKHR` (`SPV_KHR_fma`).
    KhrShaderFma,
    // Appended.
    /// `VK_EXT_depth_range_unrestricted` (never promoted): among others, a copy into a depth
    /// aspect keeps a value outside `[0, 1]`, which Metal's `Depth32Float` textures hold. No
    /// module requires it; a module whose depth comparisons the device makes on a device without
    /// it enabled records `PrecisionNote::RestrictedDepthRange`, and the consumer announces an
    /// upload into a depth image there (`restricted-depth-upload`).
    ExtDepthRangeUnrestricted,
    // Appended.
    /// `VK_EXT_depth_bias_control` (never promoted): a depth bias's representation and exactness.
    /// No module requires it.
    ExtDepthBiasControl,
    // No module requires any of these.
    /// `VK_EXT_depth_clip_enable` (never promoted): [`VkFeature::DepthClipEnable`].
    ExtDepthClipEnable,
    /// `VK_EXT_depth_clamp_control` (never promoted): [`VkFeature::DepthClampControl`].
    ExtDepthClampControl,
    /// `VK_EXT_sample_locations` (never promoted, no feature member): programmable sample
    /// locations, as Metal's `setSamplePositions:count:` sets
    /// ([`VkProperty::SampleLocationSampleCount`], [`VkProperty::VariableSampleLocations`]).
    ExtSampleLocations,
    /// `VK_EXT_primitive_topology_list_restart` (never promoted):
    /// [`VkFeature::PrimitiveTopologyListRestart`] and
    /// [`VkFeature::PrimitiveTopologyPatchListRestart`].
    ExtPrimitiveTopologyListRestart,
    // Appended.
    /// `VK_EXT_extended_dynamic_state2` (core in 1.3): [`VkFeature::ExtendedDynamicState2`].
    ExtExtendedDynamicState2,
    // Appended.
    /// `VK_KHR_shader_quad_control` (never promoted): [`VkFeature::ShaderQuadControl`].
    KhrShaderQuadControl,
    // Appended.
    /// `VK_EXT_attachment_feedback_loop_layout` (never promoted):
    /// [`VkFeature::AttachmentFeedbackLoopLayout`].
    ExtAttachmentFeedbackLoopLayout,
    /// `VK_EXT_extended_dynamic_state3` (never promoted):
    /// [`VkFeature::ExtendedDynamicState3ColorWriteMask`].
    ExtExtendedDynamicState3,
    // Appended.
    /// `VK_KHR_format_feature_flags2` (core in 1.3, no feature member): with it a storage image or
    /// storage texel buffer is read or written through a declaration without a format where its
    /// format's own `STORAGE_READ_WITHOUT_FORMAT` / `STORAGE_WRITE_WITHOUT_FORMAT` bit says, on a
    /// device without `shaderStorageImageReadWithoutFormat` (or `Write`), as the SPIR-V
    /// capabilities `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat` allow.
    KhrFormatFeatureFlags2,
    // Appended.
    /// `VK_EXT_provoking_vertex` (never promoted): [`VkFeature::ProvokingVertexLast`] and
    /// [`VkProperty::ProvokingVertexModePerPipeline`].
    ExtProvokingVertex,
    /// `VK_EXT_post_depth_coverage` (never promoted, no feature member): a fragment shader's
    /// `SampleMask` input read after the early depth and stencil tests (`PostDepthCoverage`).
    ExtPostDepthCoverage,
    // Appended.
    /// `VK_EXT_line_rasterization` (promoted to `VK_KHR_line_rasterization`, never core):
    /// [`VkFeature::BresenhamLines`].
    ExtLineRasterization,
}

/// One subgroup operation class (`VkSubgroupFeatureFlagBits`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubgroupOp {
    Basic,
    Vote,
    Arithmetic,
    Ballot,
    Shuffle,
    ShuffleRelative,
    Clustered,
    Quad,
}

/// Floating-point width named by a float-controls property.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatWidth {
    F16,
    F32,
    F64,
}

/// A float-controls property class (`VkPhysicalDeviceFloatControlsProperties`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatControl {
    DenormPreserve,
    DenormFlushToZero,
    RoundingModeRte,
    RoundingModeRtz,
    SignedZeroInfNanPreserve,
}

/// A boolean device property the new core can depend on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VkProperty {
    /// `subgroupSupportedStages` contains the stage.
    SubgroupStage(Stage),
    /// `subgroupSupportedOperations` contains the operation class.
    SubgroupOperation(SubgroupOp),
    /// `subgroupQuadOperationsInAllStages`.
    SubgroupQuadOperationsInAllStages,
    /// `requiredSubgroupSizeStages` contains the stage.
    RequiredSubgroupSizeStage(Stage),
    /// A float-controls property for one width.
    FloatControl(FloatControl, FloatWidth),
    /// `filterMinmaxSingleComponentFormats`.
    FilterMinmaxSingleComponentFormats,
    /// `denormBehaviorIndependence` is at least `32_BIT_ONLY`.
    DenormIndependence32BitOnly,
    /// `denormBehaviorIndependence` is `ALL`.
    DenormIndependenceAll,
    /// `roundingModeIndependence` is at least `32_BIT_ONLY`.
    RoundingIndependence32BitOnly,
    /// `roundingModeIndependence` is `ALL`.
    RoundingIndependenceAll,
    /// `rgba4OpaqueBlackSwizzled` (`VK_KHR_maintenance10`): an opaque-black built-in border
    /// color reads as opaque black through `R4G4B4A4_UNORM_PACK16` and `B4G4R4A4_UNORM_PACK16`
    /// views, where otherwise the implementation may swap alpha with red or blue.
    Rgba4OpaqueBlackSwizzled,
    /// `framebufferColorSampleCounts` has this sample count: every float or fixed-point color
    /// attachment supports it.
    FramebufferColorSampleCount(u32),
    /// `framebufferIntegerColorSampleCounts` has this sample count.
    FramebufferIntegerColorSampleCount(u32),
    /// `framebufferDepthSampleCounts` has this sample count.
    FramebufferDepthSampleCount(u32),
    /// `framebufferStencilSampleCounts` has this sample count.
    FramebufferStencilSampleCount(u32),
    /// `framebufferNoAttachmentsSampleCounts` has this sample count.
    FramebufferNoAttachmentsSampleCount(u32),
    /// `GLSL.std.450 InterpolateAtOffset` reads at the pixel center plus its offset, as
    /// GLSL.std.450 states. Every device has it but one whose description reports a MoltenVK
    /// driver, which reads 1/16 pixel off the center at any release. A module that uses the
    /// instruction requires it.
    InterpolateAtOffsetCentered,
    /// The device runs each 32 consecutive invocations of a compute workgroup in lockstep, as an
    /// Apple GPU runs a Metal simdgroup (the driver's measured deviation
    /// [`DriverDeviation::SimdgroupLockstep`](crate::error::DriverDeviation::SimdgroupLockstep)).
    /// No Vulkan query reports it. A kernel whose threadgroup accesses only that lockstep orders,
    /// at a place no subgroup barrier may stand, requires it rather than emulated simdgroups.
    SimdgroupLockstep,
    /// The device samples, filters, and converts texels through Metal's own sampler on Apple's
    /// GPU (the driver's measured deviation
    /// [`DriverDeviation::MetalSampler`](crate::error::DriverDeviation::MetalSampler)), so a
    /// sample's footprint, weights, rounding, and sRGB decode are Metal's. No Vulkan query reports
    /// it. A module that leaves a filtered sample to the device's filter because it is Metal's
    /// (where another device emulates Metal's linear filter by fetches or notes
    /// `PrecisionNote::HalfSampleFilter` or `PrecisionNote::FloatSampleFilter`), or an 8-bit sRGB
    /// texel to the device's decode (where another notes `PrecisionNote::SrgbTexelDecode`),
    /// requires it.
    MetalSampler,
    /// `storageImageSampleCounts` has this sample count: a multisampled shadow of the stencil.
    StorageImageSampleCount(u32),
    /// `supportedDepthResolveModes` (`VkPhysicalDeviceDepthStencilResolveProperties`, Vulkan 1.2)
    /// has this mode: a render pass may resolve a multisampled depth attachment by it, as Metal's
    /// `MTLMultisampleDepthResolveFilter` asks (sample 0, the minimum, or the maximum).
    DepthResolveMode(ResolveMode),
    /// `supportedStencilResolveModes` has this mode (Metal's `Sample0`; its `DepthResolvedSample`
    /// has no resolve mode).
    StencilResolveMode(ResolveMode),
    /// `independentResolveNone`: one aspect of a depth/stencil attachment may resolve while the
    /// other does not.
    IndependentResolveNone,
    /// `independentResolve`: the depth and stencil aspects may resolve by different modes.
    IndependentResolve,
    /// `variableSampleLocations` (`VkPhysicalDeviceSampleLocationsPropertiesEXT`, where
    /// `VK_EXT_sample_locations` is listed): the sample locations may differ between pipelines
    /// in a subpass. Metal's are the render pass's, so it states only what the device can do.
    VariableSampleLocations,
    /// `sampleLocationSampleCounts` (where `VK_EXT_sample_locations` is listed) has this sample
    /// count: an attachment of it may take custom sample locations. Vulkan guarantees what
    /// Metal's positions need of the rest: a 1x1 grid, 4 bits of sub-pixel precision, and the
    /// coordinate range `[0, 0.9375]`, Metal's sixteenths of a pixel.
    SampleLocationSampleCount(u32),
    /// A color attachment of this format takes this sample count
    /// (`FormatProperties::color_attachment_sample_counts`): a 2D, optimal-tiling image of the
    /// format with `COLOR_ATTACHMENT` usage has it among its `sampleCounts`. A device's
    /// `framebuffer*ColorSampleCounts` are the counts every format of a kind takes; one format
    /// may take more.
    ColorAttachmentSampleCount(vk::Format, u32),
    /// The device stores a fragment function's float color output into a normalized, sRGB or
    /// half-float attachment as Metal does (the driver's measured deviation
    /// [`DriverDeviation::MetalColorWrites`](crate::error::DriverDeviation::MetalColorWrites)). No
    /// Vulkan query reports it. Without it, `Device::compile_render_pipeline` has the fragment
    /// function round such outputs by Metal's rules (`LoweringInputs::color_writes`).
    MetalColorWrites,
    /// The device stores the NaN a fragment function writes for a `float` output to a
    /// half-float color channel as Metal does, `0x7e00` (the driver's measured deviation
    /// [`DriverDeviation::MetalFloatColorNans`](crate::error::DriverDeviation::MetalFloatColorNans)).
    /// No Vulkan query reports it. Without it, such a write records
    /// `PrecisionNote::DeviceNanEncoding`.
    MetalFloatColorNans,
    /// The device stores a `half` NaN a fragment function writes to a half-float color channel
    /// as its own bits, as Metal does (the driver's measured deviation
    /// [`DriverDeviation::MetalHalfColorNans`](crate::error::DriverDeviation::MetalHalfColorNans)).
    /// No Vulkan query reports it. Without it, such a write records
    /// `PrecisionNote::DeviceNanEncoding`.
    MetalHalfColorNans,
    /// The device stores a 16-bit float color output to a half-float color channel as its bits,
    /// a NaN's included (the driver's measured deviation
    /// [`DriverDeviation::HalfColorOutputBits`](crate::error::DriverDeviation::HalfColorOutputBits)).
    /// No Vulkan query reports it. Without it, an output the device does not store Metal's NaN of
    /// stays 32-bit and its write records `PrecisionNote::DeviceNanEncoding`.
    HalfColorOutputBits,
    /// The device's `GLSL.std.450 Fma` on 16-bit floats is Metal's own `half` `fma` (the
    /// driver's measured deviation
    /// [`DriverDeviation::MetalHalfFma`](crate::error::DriverDeviation::MetalHalfFma)). No Vulkan
    /// query reports it. A module whose `half` fused multiply-add takes that instruction requires
    /// it; without it, the module rounds the `float` sum to `half` itself.
    MetalHalfFma,
    /// The device's `float` arithmetic, comparisons, `Ldexp`, `Sqrt`, `InverseSqrt`, `Trunc` and
    /// `RoundEven` read an `f32` denormal operand as the zero of its sign and flush a denormal
    /// result, as Metal does
    /// (the driver's measured deviation
    /// [`DriverDeviation::MetalFloatDenormals`](crate::error::DriverDeviation::MetalFloatDenormals)).
    /// No Vulkan query reports it: `DenormFlushToZero` makes only the results' flush a
    /// requirement, the operands' remains the device's choice. A module that leaves such an
    /// operation's denormals to the device requires it; without it, the module flushes them
    /// itself.
    MetalFloatDenormals,
    /// The device's `OpQuantizeToF16` rounds as Metal's conversion to `half` does, to nearest
    /// even below 65520 (the driver's measured deviation
    /// [`DriverDeviation::MetalHalfQuantize`](crate::error::DriverDeviation::MetalHalfQuantize)).
    /// No Vulkan query reports it: the quantize's rounding mode is the device's. A module that
    /// rounds to `half` through the quantize alone requires it; without it, the module clamps
    /// below 65520 first.
    MetalHalfQuantize,
    /// The device's `OpFConvert` from 32-bit to 16-bit floats, of the result of an `OpFAdd`
    /// decorated `NoContraction`, rounds as Metal's conversion to `half` does: to nearest even on
    /// the half grid, half denormals kept, infinity from 65520 up (the driver's measured deviation
    /// [`DriverDeviation::MetalHalfConversion`](crate::error::DriverDeviation::MetalHalfConversion)).
    /// No Vulkan query reports it: without `RoundingModeRTE` and `DenormPreserve` for 16 bits
    /// the conversion's rounding mode and its denormals are the device's. A module that rounds
    /// to `half` through the conversion and its widening requires it; without it, the module
    /// rounds through `OpQuantizeToF16`.
    MetalHalfConversion,
    /// The device reads a center-sampled varying of a multisampled pipeline at the pixel center
    /// in every fragment invocation, a sample-rate one's included, as Metal does (the driver's
    /// measured deviation
    /// [`DriverDeviation::MetalCenterVaryings`](crate::error::DriverDeviation::MetalCenterVaryings)).
    /// No Vulkan query reports it: Vulkan evaluates an undecorated input at an
    /// implementation-defined position in the area the invocation shades. A fragment module
    /// compiled for a multisampled pipeline whose center varyings are read as they are, as their
    /// pixel form, requires it; without it, their pixel form is `InterpolateAtOffset` at offset
    /// 0, which requires [`VkProperty::InterpolateAtOffsetCentered`].
    MetalCenterVaryings,
    /// The device stores a fragment function's `float` color output into a half-float channel,
    /// or an unsigned 11- or 10-bit float one, as Metal does for every value but a NaN (the
    /// driver's measured deviation
    /// [`DriverDeviation::MetalSmallFloatColorWrites`](crate::error::DriverDeviation::MetalSmallFloatColorWrites)).
    /// No Vulkan query reports it: Vulkan leaves the conversion's rounding to the device. A
    /// fragment module that writes such an output as computed requires it; without it, the module
    /// rounds the output by Metal's rule.
    MetalSmallFloatColorWrites,
    /// The device stores a fragment function's float color output into a 16-bit unorm or snorm
    /// channel as Metal does, a blended snorm channel's NaN aside (the driver's measured
    /// deviation
    /// [`DriverDeviation::MetalNorm16ColorWrites`](crate::error::DriverDeviation::MetalNorm16ColorWrites)).
    /// No Vulkan query reports it: Vulkan allows a conversion within one code's width of the exact
    /// product's. A fragment module that writes such an output as computed requires it; without it,
    /// the module rounds the output by Metal's rule.
    MetalNorm16ColorWrites,
    /// The device's `GLSL.std.450` `NMin`, `NMax` and `NClamp` on 32-bit floats are Metal's own
    /// `fmin`, `fmax` and `clamp` (the driver's measured deviation
    /// [`DriverDeviation::MetalMinMax`](crate::error::DriverDeviation::MetalMinMax)). No Vulkan
    /// query reports it. A module whose `float` or `half` minimum, maximum or clamp takes those
    /// instructions requires it; without it, the module picks the operand itself.
    MetalMinMax,
    /// The device's 16-bit `OpFAdd` and `OpFSub` decorated `NoContraction`, and its 16-bit
    /// `Fma(a, b, -0.0)` decorated `NoContraction`, are Metal's own `half` `+`, `-` and `*` (the
    /// driver's measured deviation
    /// [`DriverDeviation::MetalHalfArithmetic`](crate::error::DriverDeviation::MetalHalfArithmetic)).
    /// No Vulkan query reports it. A module whose `half` `+`, `-` or `*` takes those instructions
    /// requires it; without it, the module computes in `float` and rounds to `half` itself.
    MetalHalfArithmetic,
    /// The device stores every `float` but a NaN that a shader writes to a half-float storage
    /// image as Metal stores a kernel's write, rounded toward zero (the driver's measured
    /// deviation
    /// [`DriverDeviation::MetalHalfStorageWrites`](crate::error::DriverDeviation::MetalHalfStorageWrites)).
    /// No Vulkan query reports it: Vulkan leaves the conversion's rounding to the device. A module
    /// that writes such a `float` unrounded requires it; without it, the module rounds the texel by
    /// Metal's rule.
    MetalHalfStorageWrites,
    /// The device stores every NaN a shader writes to a half-float storage image as Metal's
    /// `0x7e00` (the driver's measured deviation
    /// [`DriverDeviation::MetalHalfStorageNans`](crate::error::DriverDeviation::MetalHalfStorageNans)).
    /// No Vulkan query reports it. A module that writes such a NaN as computed requires it; without
    /// it, the module writes it as `0x7fc00000`.
    MetalHalfStorageNans,
    /// The device stores the NaN `0x7fc00000` a shader writes to a half-float storage image as
    /// `0x7e00`, Metal's (the driver's measured deviation
    /// [`DriverDeviation::MetalFloatStorageNans`](crate::error::DriverDeviation::MetalFloatStorageNans)).
    /// No Vulkan query reports it. A module that writes such a NaN as `0x7fc00000` requires it;
    /// without it, the write records `PrecisionNote::DeviceNanEncoding`.
    MetalFloatStorageNans,
    /// The device's fragment subgroups are Metal's simdgroups, and `SubgroupLocalInvocationId` in
    /// a fragment shader is Metal's `thread_index_in_simdgroup` (the driver's measured deviation
    /// [`DriverDeviation::MetalFragmentSimdgroups`](crate::error::DriverDeviation::MetalFragmentSimdgroups)).
    /// No Vulkan query reports it. A fragment module that reads its simdgroup lane requires it;
    /// without it, the lane is the device's own grouping's, with
    /// `PrecisionNote::FragmentSimdgroupGrouping`, where the stage runs 32-lane subgroups, and is
    /// refused elsewhere.
    MetalFragmentSimdgroups,
    /// The device runs a vertex shader's subgroup operations as Metal runs its simdgroup functions
    /// though `subgroupSupportedStages` leaves the vertex stage out (the driver's measured
    /// deviation
    /// [`DriverDeviation::MetalVertexSimdgroups`](crate::error::DriverDeviation::MetalVertexSimdgroups)).
    /// No Vulkan query reports it. A vertex module that uses subgroup operations requires it in
    /// place of `SubgroupStage(Vertex)` on such a device.
    MetalVertexSimdgroups,
    /// In a per-sample fragment invocation of a multisampled pipeline the device gives
    /// `FragCoord.z`, the draw's depth bias included, and a no-perspective varying read as it is
    /// at the pixel centre, and `InterpolateAtSample` at the sample, while its depth test stores
    /// each sample's own depth (the driver's measured deviation
    /// [`DriverDeviation::CenterFragCoordDepth`](crate::error::DriverDeviation::CenterFragCoordDepth)).
    /// No Vulkan query reports it. A fragment module that computes each sample's depth from those
    /// reads for its in-shader depth test requires it.
    CenterFragCoordDepth,
    /// In a per-sample fragment invocation of a multisampled pipeline the device gives
    /// `FragCoord.z` at the invocation's sample, the draw's depth bias included, the depth its
    /// depth test stores there (the driver's measured deviation
    /// [`DriverDeviation::SampleFragCoordDepth`](crate::error::DriverDeviation::SampleFragCoordDepth)).
    /// No Vulkan query reports it. A fragment module whose in-shader depth test reads `FragCoord.z`
    /// at each sample requires it.
    SampleFragCoordDepth,
    /// The device stores the widened `half` a shader writes to a 16-bit float storage image as
    /// the half's own bits, a signalling NaN's unquieted, and the NaN `0x7fc00000` written to a
    /// packed small float one as the format's canonical NaN (the driver's measured deviation
    /// [`DriverDeviation::MetalHalfTexelWrites`](crate::error::DriverDeviation::MetalHalfTexelWrites)).
    /// No Vulkan query reports it. A module whose `half` texel write leaves such a NaN to the
    /// device requires it; without it, the write records `PrecisionNote::DeviceNanEncoding`.
    MetalHalfTexelWrites,
    /// The device stores a `float` texel a shader writes to a unorm or snorm storage image as
    /// Metal's code, the exact product rounded to nearest with a tie away from zero (the driver's
    /// measured deviation
    /// [`DriverDeviation::MetalNormStorageWrites`](crate::error::DriverDeviation::MetalNormStorageWrites)).
    /// No Vulkan query reports it. A module whose `float` texel write leaves a normalized format's
    /// conversion to the device requires it; without it, the shader rounds each channel.
    MetalNormStorageWrites,
    /// The device's linear filter of an 8-bit unsigned normalized texture, at the coordinate of
    /// the centre of Metal's footprint step, lands within one code of Metal's (the driver's
    /// measured deviation
    /// [`DriverDeviation::Unorm8LinearFilter`](crate::error::DriverDeviation::Unorm8LinearFilter)).
    /// No Vulkan query reports it. A module that hands such a sample to the device's filter
    /// requires it; without it, the sample takes Metal's filter, emulated exactly.
    Unorm8LinearFilter,
    /// Native device sampling: the module hands its samples, gathers and comparisons to the
    /// device's own sampler as a device that samples through Metal's
    /// ([`VkProperty::MetalSampler`]) would, on a device that does not, under
    /// [`SamplingPolicy::NativeDevice`](crate::device::SamplingPolicy::NativeDevice), the
    /// default. No Vulkan query reports it: the profile takes it from the policy. A module that
    /// leaves a sample to the device's sampler there requires it and records
    /// [`PrecisionNote::DeviceSampling`](crate::pipeline::PrecisionNote::DeviceSampling).
    DeviceSampling,
    /// The device stores every unbiased depth a draw writes to a `Depth16Unorm` attachment as
    /// Metal's code (the driver's measured deviation
    /// [`DriverDeviation::MetalDepth16Stores`](crate::error::DriverDeviation::MetalDepth16Stores)).
    /// No Vulkan query reports it: Vulkan leaves the conversion's rounding to the device. No module
    /// requires it: where the profile lacks it, the fragment stage writes Metal's code itself.
    MetalDepth16Stores,
    /// The device stores every depth a draw biases on a `Depth16Unorm` attachment as Metal's code
    /// (the driver's measured deviation
    /// [`DriverDeviation::MetalDepth16BiasedStores`](crate::error::DriverDeviation::MetalDepth16BiasedStores)).
    /// No Vulkan query reports it: Vulkan leaves the biased sum's rounding to the device. No module
    /// requires it: where the profile lacks it, a biased draw's fragment stage writes Metal's code
    /// itself.
    MetalDepth16BiasedStores,
    /// The device interpolates a varying as Metal does, bit for bit (the driver's measured
    /// deviation
    /// [`DriverDeviation::MetalInterpolation`](crate::error::DriverDeviation::MetalInterpolation)).
    /// No Vulkan query reports it: Vulkan states no precision for interpolation. A fragment module
    /// that reads a varying the device interpolates and records no
    /// [`PrecisionNote::DeviceInterpolation`](crate::pipeline::PrecisionNote::DeviceInterpolation)
    /// requires it; without it, the module records that note.
    MetalInterpolation,
    /// The device takes a primitive's `Layer` and `ViewportIndex` from its first vertex, whatever
    /// its other vertices store, as Metal takes a primitive's slice and viewport (the driver's
    /// measured deviation
    /// [`DriverDeviation::MetalFirstVertexIndices`](crate::error::DriverDeviation::MetalFirstVertexIndices)).
    /// No Vulkan query reports it: Vulkan takes them only where every vertex of a primitive
    /// stores the same. A vertex module that stores an index differing among a primitive's
    /// vertices to its builtin requires it; without it, the index goes through the pipeline's
    /// synthesized geometry stages.
    MetalFirstVertexIndices,
    /// The device runs a fragment shader's quads whole, its helper invocations taking part in
    /// every group operation, as Metal runs a fragment function's helper threads (the driver's
    /// measured deviation
    /// [`DriverDeviation::MetalFullQuads`](crate::error::DriverDeviation::MetalFullQuads)).
    /// No Vulkan query reports it: Vulkan gives it only under `RequireFullQuadsKHR`. A fragment
    /// module whose group operations observe other lanes requires it in place of
    /// `shaderQuadControl` on a device without that feature.
    MetalFullQuads,
    /// `provokingVertexModePerPipeline` (`VkPhysicalDeviceProvokingVertexPropertiesEXT`, where
    /// `VK_EXT_provoking_vertex` is listed): the pipelines bound within one render pass instance
    /// may have different provoking vertex modes. Without it "all pipelines bound within a render
    /// pass instance must have the same provokingVertexMode" (the Vulkan specification's flat
    /// shading), including "any pipeline already bound when beginning the render pass instance"
    /// (`VUID-vkCmdBindPipeline-pipelineBindPoint-04881`).
    ProvokingVertexModePerPipeline,
    /// The device's `D32_SFLOAT` images keep each value outside `[0, 1]` without
    /// [`VkExtension::ExtDepthRangeUnrestricted`] enabled, as Metal's `Depth32Float` textures do
    /// (the runtime's measured driver fact
    /// [`DriverFact::DepthImageKeepsValuesOutsideUnitRange`](crate::device::DriverFact::DepthImageKeepsValuesOutsideUnitRange)).
    /// No Vulkan query reports it: Vulkan leaves such a value undefined without the extension. A
    /// module whose depth comparisons the device makes requires it on a device without the
    /// extension in place of recording `PrecisionNote::RestrictedDepthRange`.
    DepthImageKeepsValuesOutsideUnitRange,
}

/// One depth or stencil resolve mode (`VkResolveModeFlagBits`), named by
/// [`VkProperty::DepthResolveMode`] and [`VkProperty::StencilResolveMode`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResolveMode {
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT`: the value of sample 0.
    SampleZero,
    /// `VK_RESOLVE_MODE_AVERAGE_BIT`: the average of the samples.
    Average,
    /// `VK_RESOLVE_MODE_MIN_BIT`: the smallest sample.
    Min,
    /// `VK_RESOLVE_MODE_MAX_BIT`: the largest sample.
    Max,
}

impl ResolveMode {
    /// Every mode, in `VkResolveModeFlagBits` order.
    pub const ALL: [ResolveMode; 4] = [
        ResolveMode::SampleZero,
        ResolveMode::Average,
        ResolveMode::Min,
        ResolveMode::Max,
    ];

    /// The mode's `VkResolveModeFlagBits` bit.
    pub fn flag(self) -> vk::ResolveModeFlags {
        match self {
            ResolveMode::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            ResolveMode::Average => vk::ResolveModeFlags::AVERAGE,
            ResolveMode::Min => vk::ResolveModeFlags::MIN,
            ResolveMode::Max => vk::ResolveModeFlags::MAX,
        }
    }
}

/// A numeric device limit the new core checks (`VkPhysicalDeviceLimits` and extension properties).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VkLimit {
    MaxComputeWorkGroupSize(u8),
    MaxComputeWorkGroupCount(u8),
    MaxComputeWorkGroupInvocations,
    MaxComputeSharedMemorySize,
    MaxComputeWorkgroupSubgroups,
    MinSubgroupSize,
    MaxSubgroupSize,
    MaxPushConstantsSize,
    MaxPushDescriptors,
    MaxBoundDescriptorSets,
    MaxPerStageDescriptorSamplers,
    MaxPerStageDescriptorUniformBuffers,
    MaxPerStageDescriptorStorageBuffers,
    MaxPerStageDescriptorSampledImages,
    MaxPerStageDescriptorStorageImages,
    MaxPerStageDescriptorInputAttachments,
    MaxPerStageResources,
    MaxDescriptorSetSamplers,
    MaxDescriptorSetUniformBuffers,
    MaxDescriptorSetStorageBuffers,
    MaxDescriptorSetSampledImages,
    MaxDescriptorSetStorageImages,
    MaxDescriptorSetInputAttachments,
    MaxDescriptorSetUpdateAfterBindSamplers,
    MaxDescriptorSetUpdateAfterBindSampledImages,
    MaxDescriptorSetUpdateAfterBindStorageImages,
    MaxDescriptorSetUpdateAfterBindStorageBuffers,
    MaxPerStageDescriptorUpdateAfterBindSamplers,
    MaxPerStageDescriptorUpdateAfterBindSampledImages,
    MaxPerStageDescriptorUpdateAfterBindStorageImages,
    MaxPerStageDescriptorUpdateAfterBindStorageBuffers,
    MaxPerStageUpdateAfterBindResources,
    MaxStorageBufferRange,
    MaxUniformBufferRange,
    MinStorageBufferOffsetAlignment,
    MinUniformBufferOffsetAlignment,
    MinTexelBufferOffsetAlignment,
    MaxTexelBufferElements,
    MaxSamplerLodBias,
    MaxSamplerAnisotropy,
    MinTexelOffset,
    MaxTexelOffset,
    MinTexelGatherOffset,
    MaxTexelGatherOffset,
    MinInterpolationOffset,
    MaxInterpolationOffset,
    MaxImageDimension1D,
    MaxImageDimension2D,
    MaxImageDimension3D,
    MaxImageDimensionCube,
    MaxImageArrayLayers,
    MaxVertexInputAttributes,
    MaxVertexInputBindings,
    MaxVertexInputAttributeOffset,
    MaxVertexInputBindingStride,
    MaxVertexOutputComponents,
    MaxTessellationGenerationLevel,
    MaxTessellationEvaluationOutputComponents,
    MaxFragmentInputComponents,
    MaxFragmentOutputAttachments,
    MaxFragmentDualSrcAttachments,
    MaxFragmentCombinedOutputResources,
    MaxColorAttachments,
    MaxClipDistances,
    MaxCullDistances,
    MaxCombinedClipAndCullDistances,
    MaxVertexAttribDivisor,
    /// `maxMultiviewViewCount` (Vulkan 1.1): the views of a multiview render pass instance, so
    /// the largest vertex amplification count a render pipeline may declare.
    MaxMultiviewViewCount,
    /// `maxMultiviewInstanceIndex` (Vulkan 1.1): the largest instance index of a draw in a
    /// multiview render pass instance (`VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688`).
    MaxMultiviewInstanceIndex,
    // Appended.
    MaxPerStageDescriptorAccelerationStructures,
    MaxDescriptorSetAccelerationStructures,
    // Appended.
    /// `maxFramebufferWidth` (`VkPhysicalDeviceLimits`): the widest framebuffer a render pass
    /// instance may have. Named for the runtime's refusals; nothing the translation emits
    /// requires it.
    MaxFramebufferWidth,
    /// `maxFramebufferHeight` (`VkPhysicalDeviceLimits`): the tallest framebuffer a render pass
    /// instance may have.
    MaxFramebufferHeight,
    /// `maxFramebufferLayers` (`VkPhysicalDeviceLimits`): the most layers a framebuffer may have.
    MaxFramebufferLayers,
    // Appended.
    /// `maxDrawIndexedIndexValue` (`VkPhysicalDeviceLimits`): the largest index an indexed draw
    /// may fetch, `2^32 - 1` with [`VkFeature::FullDrawIndexUint32`] and at least `2^24 - 1`
    /// otherwise. Named for the runtime's index clamp; nothing the translation emits requires it.
    MaxDrawIndexedIndexValue,
    /// `maxTessellationPatchSize` (`VkPhysicalDeviceLimits`): the most control points a patch
    /// may have (at least 32 with `tessellationShader`, Metal's own maximum). Named for the
    /// runtime's patch draws.
    MaxTessellationPatchSize,
    /// The Function storage a thread may hold, in bytes: measured on each device, since no Vulkan
    /// property states it (a driver backs Function storage with scratch memory, and past it can
    /// answer wrongly or hang with no error). The call stack of a recursion through visible
    /// functions is Function storage, and a stack past the limit is refused
    /// (`Unsupported{Recursion}` needing it). A device not measured has the floor, 16 KiB; a
    /// vertex or fragment function's refusal there names 192 KiB, the most Metal was measured to
    /// hold, which such a function may keep (`device::function_storage`).
    FunctionStorageBytesPerThread,
    /// `maxMeshOutputVertices` (`VkPhysicalDeviceMeshShaderPropertiesEXT`): the most vertices a
    /// mesh entry may declare (`OutputVertices`), a mesh type's `NV`.
    MaxMeshOutputVertices,
    /// `maxMeshOutputPrimitives`: the most primitives a mesh entry may declare
    /// (`OutputPrimitivesEXT`), a mesh type's `NP`.
    MaxMeshOutputPrimitives,
    /// `maxMeshOutputComponents`: the scalar components of a mesh entry's user outputs, per
    /// vertex and per primitive together.
    MaxMeshOutputComponents,
    /// `maxTaskSharedMemorySize`: the bytes of workgroup memory an object function's entry may
    /// declare, its threadgroup memory and its mesh grid.
    MaxTaskSharedMemorySize,
    /// `maxMeshSharedMemorySize`: the same for a mesh function's entry, its threadgroup memory and
    /// its primitive count.
    MaxMeshSharedMemorySize,
    /// `maxMeshWorkGroupCount[axis]`: the mesh threadgroups a mesh draw may launch along an axis.
    /// Named for the binder's mesh draws; nothing the translation emits requires it.
    MaxMeshWorkGroupCount(u8),
    /// `maxMeshWorkGroupTotalCount`: the mesh threadgroups a mesh draw may launch in all.
    MaxMeshWorkGroupTotalCount,
    /// `maxMeshWorkGroupSize[axis]`: the threads of a mesh threadgroup along an axis.
    MaxMeshWorkGroupSize(u8),
    /// `maxMeshWorkGroupInvocations`: the threads of a mesh threadgroup.
    MaxMeshWorkGroupInvocations,
    /// `maxMeshOutputLayers`: one more than the largest `Layer` a mesh entry may write, so the
    /// layers of a pass a mesh function chooses its slice in.
    MaxMeshOutputLayers,
    /// `maxTaskWorkGroupCount[axis]`: the object threadgroups a mesh draw may launch along an
    /// axis. Named for the binder's mesh draws.
    MaxTaskWorkGroupCount(u8),
    /// `maxTaskWorkGroupTotalCount`: the object threadgroups a mesh draw may launch in all.
    MaxTaskWorkGroupTotalCount,
    /// `maxTaskWorkGroupSize[axis]`: the threads of an object threadgroup along an axis.
    MaxTaskWorkGroupSize(u8),
    /// `maxTaskWorkGroupInvocations`: the threads of an object threadgroup.
    MaxTaskWorkGroupInvocations,
    /// `maxTaskPayloadSize`: the bytes of an object function's payload, as an object or a mesh
    /// entry declares it.
    MaxTaskPayloadSize,
    /// `maxTaskPayloadAndSharedMemorySize`: an object entry's payload and workgroup memory
    /// together.
    MaxTaskPayloadAndSharedMemorySize,
    /// `maxMeshPayloadAndSharedMemorySize`: a mesh entry's payload and workgroup memory together.
    MaxMeshPayloadAndSharedMemorySize,
    // Appended.
    /// `maxGeometryInputComponents` (`VkPhysicalDeviceLimits`): the scalar components of a
    /// geometry stage's user inputs, per vertex.
    MaxGeometryInputComponents,
    /// `maxGeometryOutputComponents`: the scalar components of a geometry stage's user outputs,
    /// per vertex.
    MaxGeometryOutputComponents,
    /// `maxGeometryTotalOutputComponents`: the scalar components of the user outputs of every
    /// vertex a geometry stage emits for an input primitive, together.
    MaxGeometryTotalOutputComponents,
}

/// What a format must support for a use (`VkFormatFeatureFlags2`: optimal-tiling features for
/// images, buffer features for vertex and texel buffers).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FormatUsage {
    VertexBuffer,
    UniformTexelBuffer,
    StorageTexelBuffer,
    StorageTexelBufferAtomic,
    SampledImage,
    SampledImageFilterLinear,
    SampledImageFilterMinmax,
    StorageImage,
    StorageImageAtomic,
    ColorAttachment,
    ColorAttachmentBlend,
    DepthStencilAttachment,
    /// A storage image of the format is read through an image declared without a format
    /// (`STORAGE_READ_WITHOUT_FORMAT` in its optimal-tiling features,
    /// `VUID-vkCmdDispatch-OpTypeImage-07028`).
    StorageImageReadWithoutFormat,
    /// A storage image of the format is written through an image declared without a format
    /// (`STORAGE_WRITE_WITHOUT_FORMAT` in its optimal-tiling features, `-07027`).
    StorageImageWriteWithoutFormat,
    /// A storage texel buffer of the format is read through an image declared without a format
    /// (`STORAGE_READ_WITHOUT_FORMAT` in its buffer features, `-07030`).
    StorageTexelBufferReadWithoutFormat,
    /// A storage texel buffer of the format is written through an image declared without a
    /// format (`STORAGE_WRITE_WITHOUT_FORMAT` in its buffer features, `-07029`).
    StorageTexelBufferWriteWithoutFormat,
}

/// A Vulkan API version (`major.minor`); the floor is 1.2 (charter).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
}

impl ApiVersion {
    /// Vulkan 1.2, the floor.
    pub const V1_2: ApiVersion = ApiVersion { major: 1, minor: 2 };
    /// Vulkan 1.3.
    pub const V1_3: ApiVersion = ApiVersion { major: 1, minor: 3 };
}

/// A SPIR-V capability an emitted module declares (`OpCapability`), as its raw SPIR-V grammar
/// value, the way [`vk`] carries Vulkan values: a newer grammar's value stays representable, and
/// no SPIR-V crate's type is part of this API. `Debug` prints the grammar name when this crate's
/// grammar knows the value.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct SpirvCapability(u32);

impl SpirvCapability {
    /// The capability with grammar value `raw`.
    pub const fn from_raw(raw: u32) -> SpirvCapability {
        SpirvCapability(raw)
    }

    /// The grammar value (the `OpCapability` operand).
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    /// The capability of the grammar the builder emits from.
    pub(crate) const fn of(c: crate::spv::Capability) -> SpirvCapability {
        SpirvCapability(c as u32)
    }

    /// The grammar's capability of this value, when the grammar names it.
    pub(crate) fn known(self) -> Option<crate::spv::Capability> {
        crate::spv::Capability::from_u32(self.0)
    }
}

impl fmt::Debug for SpirvCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.known() {
            Some(c) => fmt::Debug::fmt(&c, f),
            None => write!(f, "SpirvCapability({})", self.0),
        }
    }
}

/// A SPIR-V extension an emitted module can declare.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpirvExtension {
    KhrTerminateInvocation,
    ExtDemoteToHelperInvocation,
    ExtFragmentShaderInterlock,
    KhrFragmentShaderBarycentric,
    ExtShaderStencilExport,
    ExtShaderViewportIndexLayer,
    ExtShaderAtomicFloatAdd,
    ExtShaderAtomicFloatMinMax,
    ExtShaderImageInt64,
    KhrWorkgroupMemoryExplicitLayout,
    KhrComputeShaderDerivatives,
    KhrSubgroupUniformControlFlow,
    KhrMaximalReconvergence,
    KhrFloatControls,
    KhrVulkanMemoryModel,
    KhrShaderDrawParameters,
    KhrPhysicalStorageBuffer,
    ExtDescriptorIndexing,
    ExtShaderAtomicFloat16Add,
    KhrSubgroupRotate,
    // Appended.
    KhrRayQuery,
    // Appended.
    KhrFma,
    // Appended.
    ExtMeshShader,
    // Appended.
    KhrQuadControl,
    // Appended.
    KhrPostDepthCoverage,
}

/// The SPIR-V capabilities a module's [`Requirements`] list without the feature that enables them,
/// each with that feature: Vulkan also enables them by Vulkan 1.3 or by
/// `VK_KHR_format_feature_flags2`, which `DeviceProfile::check` and
/// `DeviceDescription::required_enables` resolve on the device
/// (`DeviceProfile::without_format_by_format_bits`).
pub(crate) const WITHOUT_FORMAT_CAPABILITIES: [(SpirvCapability, VkFeature); 2] = [
    (
        SpirvCapability::of(crate::spv::Capability::StorageImageReadWithoutFormat),
        VkFeature::ShaderStorageImageReadWithoutFormat,
    ),
    (
        SpirvCapability::of(crate::spv::Capability::StorageImageWriteWithoutFormat),
        VkFeature::ShaderStorageImageWriteWithoutFormat,
    ),
];

/// A shader-stage create flag a module can require (`VkPipelineShaderStageCreateFlags`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageCreateFlag {
    RequireFullSubgroups,
    AllowVaryingSubgroupSize,
}

/// Everything an emitted module needs from the device.
///
/// A superset of the enable set: SPIR-V declarations, the Vulkan features, extensions, properties,
/// minimum limits, and format usages that enable them, and stage create flags.
///
/// A module's requirements are recorded when it is finalized, from two sources only: what the
/// module contains (its types, variables, builtins, execution modes, and instructions; no
/// lowering writes a capability or extension), and what its stage's subgroup policy needs, which
/// the SPIR-V does not state. A pinned subgroup size adds `subgroupSizeControl`, the stage in
/// `requiredSubgroupSizeStages`, and the size within `minSubgroupSize` and `maxSubgroupSize`; a
/// stage that needs full subgroups adds [`StageCreateFlag::RequireFullSubgroups`]. A pipeline's
/// requirements are the union of its stages', plus, for a render pipeline, what its fixed-function
/// state needs (dynamic states, blending, multisampling, sample counts, multiview).
///
/// The SPIR-V capabilities `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat`
/// are listed without a feature: Vulkan enables each by `shaderStorageImage{Read,Write}WithoutFormat`,
/// by Vulkan 1.3, or by `VK_KHR_format_feature_flags2` (under which each format's own
/// `STORAGE_{READ,WRITE}_WITHOUT_FORMAT` bit governs its access), and
/// [`DeviceProfile::check`](crate::device::DeviceProfile::check) and
/// [`DeviceDescription::required_enables`](crate::device::DeviceDescription::required_enables)
/// resolve it on the device: the feature where the device has it, else nothing at Vulkan 1.3, else
/// the extension.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Requirements {
    pub api_version: Option<ApiVersion>,
    pub spirv_capabilities: BTreeSet<SpirvCapability>,
    pub spirv_extensions: BTreeSet<SpirvExtension>,
    pub features: BTreeSet<VkFeature>,
    pub extensions: BTreeSet<VkExtension>,
    pub properties: BTreeSet<VkProperty>,
    /// The value each limit must accommodate: a `Max*` limit must be at least this, and a `Min*`
    /// limit (an alignment or a lower bound) at most this.
    pub limits: BTreeMap<VkLimit, u64>,
    pub formats: BTreeSet<(vk::Format, FormatUsage)>,
    pub stage_flags: BTreeSet<StageCreateFlag>,
    /// The denormal-mode independence the entry points' float-controls modes need
    /// (`denormBehaviorIndependence`, RuntimeSpirv VUIDs 06289-06292).
    pub denorm_independence: Independence,
    /// The rounding-mode independence needed (`roundingModeIndependence`).
    pub rounding_independence: Independence,
}
