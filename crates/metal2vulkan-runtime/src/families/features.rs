//! The feature rows of Apple's tables, and how each is decided on a device.
//!
//! [`FeatureRow::state`] has one arm per row. Its comment names what provides the feature (a
//! runtime method or a compile-layer capability). Its code reads the state from an exact source:
//! - a feature, property or limit the device was created with (its description);
//! - the compile layer's Metal answers for the device (`MetalDeviceProperties`);
//! - one of the runtime's own queries;
//! - the runtime's per-format answers (`formats`);
//! - or nothing, where the state is the same on every device.
//!
//! Where a row has several blockers, the state names the first in this order:
//! 1. what the device lacks;
//! 2. the runtime piece not built yet;
//! 3. what the compile layer implements on no device.
//!
//! The first is the one no later work on the runtime lifts.
//!
//! **Measured for these rows** on an Apple M3 Max (macOS 26.7, 2026-10-05), with Metal's API
//! validation on:
//! - Metal makes 3D textures of the ASTC (LDR and HDR) and BC formats. It refuses them for the
//!   PVRTC and EAC/ETC formats ("cannot be used with MTLTextureType3D").
//! - Apple's compiler takes these in vertex and fragment functions, as `air.simd_*` and
//!   `air.quad_*` calls of those names: `simd_product`, `simd_prefix_inclusive_product`,
//!   `simd_shuffle_rotate_up`, `quad_and`, `quad_or`, `quad_xor`, `quad_product`,
//!   `quad_prefix_exclusive_sum`, `quad_broadcast_first`, `quad_shuffle_rotate_up` and
//!   `quad_shuffle_and_fill_up`. The compile layer decodes and lowers them, and their siblings
//!   (the other prefix sums and products, the fills down), in every stage.
//! - It takes simdgroup matrices in vertex and fragment functions too. The compile layer lowers
//!   them in kernels only.

use super::formats::{FormatRow, Missing};
use super::{Column, Facts, FormatCapability, Framework, Refusal, RowState};
use crate::capabilities::Query;
use crate::error::RuntimeGap;
use crate::framebuffer_fetch::{renders_through_words, through_storage};
use crate::rate_mapped_pass::RegionRefusal;
use metal2vulkan::device::{
    DriverWorkaround, OpBits, StageSimd, SubgroupOp, VkExtension, VkFeature, VkProperty,
};
use metal2vulkan::error::{Capability, Feature, LowerFamily, Unsupported};
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::{ArgumentBuffersTier, PixelFormat, ReadWriteTextureTier};
use metal2vulkan::pipeline::Stage;
use metal2vulkan::vk::{ResolveModeFlags, ShaderStageFlags};

/// A feature entry of Apple's tables.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FeatureRow {
    /// "MetalKit".
    MetalKit,
    /// "Metal performance shaders" (2023: "Metal Performance Shaders").
    MetalPerformanceShaders,
    /// "Programmable blending".
    ProgrammableBlending,
    /// "Separate depth and stencil attachments".
    SeparateDepthStencilAttachments,
    /// "PVRTC pixel formats".
    PvrtcPixelFormats,
    /// "EAC/ETC pixel formats".
    EacEtcPixelFormats,
    /// "ASTC pixel formats".
    AstcPixelFormats,
    /// "BC pixel formats".
    BcPixelFormats,
    /// "Compressed volume texture formats".
    CompressedVolumeTextureFormats,
    /// "Extended range pixel formats".
    ExtendedRangePixelFormats,
    /// "Wide color pixel format".
    WideColorPixelFormat,
    /// "Depth-16 pixel format".
    Depth16PixelFormat,
    /// "Linear textures".
    LinearTextures,
    /// "MSAA depth resolve".
    MsaaDepthResolve,
    /// "Array of textures (read)".
    ArrayOfTexturesRead,
    /// "Array of textures (write)".
    ArrayOfTexturesWrite,
    /// "Cube map texture arrays".
    CubeMapTextureArrays,
    /// "Stencil texture views".
    StencilTextureViews,
    /// "Array of samplers".
    ArrayOfSamplers,
    /// "Sampler maximum anisotropy" (2023: "Sampler max anisotropy").
    SamplerMaxAnisotropy,
    /// "Sampler LOD clamp".
    SamplerLodClamp,
    /// "MTLSamplerState support for comparison functions".
    SamplerCompareFunctions,
    /// "16-bit unsigned integer coordinates".
    UnsignedShortCoordinates,
    /// "Border color".
    BorderColor,
    /// "Counting occlusion query".
    CountingOcclusionQuery,
    /// "Base vertex/instance drawing".
    BaseVertexInstanceDrawing,
    /// "Layered rendering".
    LayeredRendering,
    /// "Layered rendering to multisample textures".
    LayeredRenderingToMultisampleTextures,
    /// "Memoryless render targets".
    MemorylessRenderTargets,
    /// "Dual-source blending".
    DualSourceBlending,
    /// "Combined MSAA store and resolve action".
    StoreAndMultisampleResolve,
    /// "MSAA blits".
    MsaaBlits,
    /// "Programmable sample positions".
    ProgrammableSamplePositions,
    /// "Deferred store action".
    DeferredStoreAction,
    /// "Memory barriers".
    MemoryBarriers,
    /// "Indirect command buffer support for memory barriers (compute)" (2023: "Memory barriers in
    /// indirect command buffers (compute)").
    IcbComputeMemoryBarriers,
    /// "Indirect command buffer support for memory barriers (rendering)" (2023: "Memory barriers in
    /// indirect command buffers (rendering)").
    IcbRenderMemoryBarriers,
    /// "Tessellation".
    Tessellation,
    /// "Indirect tessellation arguments".
    IndirectTessellationArguments,
    /// "Tessellation in indirect command buffers".
    IcbTessellation,
    /// "Resource heaps".
    ResourceHeaps,
    /// "Function specialization".
    FunctionSpecialization,
    /// "Read/write buffers in functions".
    ReadWriteBuffers,
    /// "Read/write textures in functions".
    ReadWriteTextures,
    /// "Extract, insert, and reverse bits".
    ExtractInsertReverseBits,
    /// "SIMD barrier".
    SimdBarrier,
    /// "Indirect draw and dispatch arguments" (2023: "Indirect draw & dispatch arguments").
    IndirectDrawDispatchArguments,
    /// "Argument buffers tier 1" (2023: "Argument buffers tier").
    ArgumentBuffersTier1,
    /// "Argument buffers tier 2" (2023: "Argument buffers tier").
    ArgumentBuffersTier2,
    /// "Indirect command buffers (rendering)".
    IcbRendering,
    /// "Indirect command buffers (compute)".
    IcbCompute,
    /// "Uniform type".
    UniformType,
    /// "Imageblocks".
    Imageblocks,
    /// "Tile shaders".
    TileShaders,
    /// "Imageblock sample coverage control".
    ImageblockSampleCoverageControl,
    /// "Post-depth coverage".
    PostDepthCoverage,
    /// "Quad-scoped permute operations".
    QuadPermute,
    /// "Quad-scoped ballot operation".
    QuadBallot,
    /// "Quad-scoped reduction operations".
    QuadReduction,
    /// "SIMD-scoped permute operations".
    SimdPermute,
    /// "SIMD-scoped reduction operations".
    SimdReduction,
    /// "SIMD-scoped matrix multiply operations".
    SimdMatrixMultiply,
    /// "Raster order groups".
    RasterOrderGroups,
    /// "Nonuniform threadgroup size" (2023: "Non-uniform threadgroup size").
    NonuniformThreadgroupSize,
    /// "Multiple viewports".
    MultipleViewports,
    /// "Stencil feedback".
    StencilFeedback,
    /// "Stencil resolve".
    StencilResolve,
    /// "Nonsquare tile dispatch" (2023: "Non-square tile dispatch").
    NonsquareTileDispatch,
    /// "Texture swizzle".
    TextureSwizzle,
    /// "Placement heap".
    PlacementHeap,
    /// "Primitive ID".
    PrimitiveId,
    /// "Barycentric coordinates".
    BarycentricCoordinates,
    /// "Read/write cube map textures in functions".
    ReadWriteCubeMapTextures,
    /// "Sparse textures".
    SparseTextures,
    /// "Sparse depth and stencil textures".
    SparseDepthStencilTextures,
    /// "Variable rasterization rate".
    VariableRasterizationRate,
    /// "Vertex amplification".
    VertexAmplification,
    /// "64-bit integer math".
    Int64Math,
    /// "Lossy texture compression".
    LossyTextureCompression,
    /// "SIMD shift and fill".
    SimdShiftAndFill,
    /// "Render dynamic libraries".
    RenderDynamicLibraries,
    /// "Compute dynamic libraries".
    ComputeDynamicLibraries,
    /// "Mesh shading". A device that runs mesh functions natively (`VK_EXT_mesh_shader` and
    /// `meshShader`) needs `taskShader`; one that runs them as compute work
    /// (`DeviceProfile::emulates_mesh_shading`) needs nothing more.
    MeshShading,
    /// "Indirect mesh draw arguments": as [`MeshShading`](Self::MeshShading).
    IndirectMeshDrawArguments,
    /// "Indirect command buffers containing mesh draws": as [`MeshShading`](Self::MeshShading),
    /// after the indirect command buffers' own gap.
    IcbMeshDraws,
    /// "MetalFX spatial upscaling".
    MetalFxSpatialUpscaling,
    /// "MetalFX temporal upscaling".
    MetalFxTemporalUpscaling,
    /// "MetalFX frame interpolation".
    MetalFxFrameInterpolation,
    /// "MetalFX denoised upscaling".
    MetalFxDenoisedUpscaling,
    /// "Fast resource loading".
    FastResourceLoading,
    /// "Ray tracing in compute pipelines".
    RayTracingCompute,
    /// "Ray tracing in render pipelines".
    RayTracingRender,
    /// "Floating-point atomics" (2023: "Floating point atomics").
    FloatAtomics,
    /// "Texture atomics".
    TextureAtomics,
    /// "64-bit atomics".
    Int64Atomics,
    /// "Query texture LOD".
    QueryTextureLod,
    /// "Binary archives".
    BinaryArchives,
    /// "Function pointers in compute pipelines".
    FunctionPointersCompute,
    /// "Function pointers in render pipelines".
    FunctionPointersRender,
    /// "Depth sample compare bias and gradient".
    DepthCompareBiasAndGradient,
    /// "Nonprivate depth stencil textures" (2023: "Non-private depth stencil textures").
    NonprivateDepthStencilTextures,
    /// "Dynamic stride for attribute buffers".
    DynamicAttributeStride,
    /// "MTLAttributeFormat.floatRGB9E5 and .floatRG11B10".
    PackedFloatAttributeFormats,
    /// "MTLDataType.bfloat (brain float) scalar and vector cases".
    BFloat,
    /// "Relaxed math".
    RelaxedMath,
    /// "Global built-ins and bindings".
    GlobalBuiltinsAndBindings,
    /// "Memory coherence for textures and buffers in shaders".
    CoherentMemory,
    /// "Per-pipeline shader validation".
    ShaderValidation,
    /// "Shader logging".
    ShaderLogging,
    /// "Residency sets".
    ResidencySets,
    /// "Acceleration structures containing row-major matrices".
    RowMajorAccelerationStructures,
    /// "Ray tracing with per-component motion interpolation".
    PerComponentMotionInterpolation,
    /// "Direct access to on-chip ray-intersection result storage".
    IntersectionResultStorage,
    /// "Fragment visibility count accumulation".
    VisibilityCountAccumulation,
    /// "Argument tables".
    ArgumentTables,
    /// "Command allocators".
    CommandAllocators,
    /// "Decoupled command queues and command buffers".
    DecoupledQueuesAndCommandBuffers,
    /// "Texture view pools".
    TextureViewPools,
    /// "Command barriers".
    CommandBarriers,
    /// "Placement sparse buffers".
    PlacementSparseBuffers,
    /// "Placement sparse textures".
    PlacementSparseTextures,
    /// "Dedicated compilation contexts".
    CompilationContexts,
    /// "Pipeline dataset serialization".
    PipelineDatasetSerialization,
    /// "Flexible render pipeline state".
    FlexibleRenderPipelineState,
    /// "Color attachment mapping".
    ColorAttachmentMapping,
    /// "Machine learning encoding".
    MachineLearningEncoding,
    /// "Tensors".
    Tensors,
    /// "Performance counter heaps".
    PerformanceCounterHeaps,
    /// "Address-driven acceleration structure builds".
    AddressDrivenAccelerationStructureBuilds,
    /// "Acceleration structure build options".
    AccelerationStructureBuildOptions,
    /// "Intersection function buffers".
    IntersectionFunctionBuffers,
    /// "Sampler minimum and maximum reduction".
    SamplerMinMaxReduction,
    /// "Sampler LOD bias".
    SamplerLodBias,
    /// "Access to pre-raster per-vertex values".
    PreRasterVertexValues,
    /// "Depth bounds testing".
    DepthBoundsTesting,
    /// "Indirect command buffer support for raster and depth/stencil state".
    IcbRasterAndDepthStencilState,
    /// "Atomics on cube map and cube map array textures".
    CubeMapAtomics,
    /// "Universal texture compression".
    UniversalTextureCompression,
    /// "Acquire and release memory ordering atomics".
    AcquireReleaseAtomics,
    /// "2 x 1 and 1 x 2 block texture reads".
    BlockTextureReads,
    /// "Clamp to edge behavior for texture reads".
    ClampToEdgeTextureReads,
    /// "Integer coordinate offsets for texture reads and atomics".
    IntegerCoordinateOffsets,
    /// "Texture barriers", in the 2023 edition only.
    TextureBarriers,
    /// "Device notifications", in the 2023 edition only.
    DeviceNotifications,
}

impl FeatureRow {
    /// Every row: the 2026 edition's, in its order, then the 2023 edition's others.
    pub const ALL: [FeatureRow; 144] = [
        FeatureRow::MetalKit,
        FeatureRow::MetalPerformanceShaders,
        FeatureRow::ProgrammableBlending,
        FeatureRow::SeparateDepthStencilAttachments,
        FeatureRow::PvrtcPixelFormats,
        FeatureRow::EacEtcPixelFormats,
        FeatureRow::AstcPixelFormats,
        FeatureRow::BcPixelFormats,
        FeatureRow::CompressedVolumeTextureFormats,
        FeatureRow::ExtendedRangePixelFormats,
        FeatureRow::WideColorPixelFormat,
        FeatureRow::Depth16PixelFormat,
        FeatureRow::LinearTextures,
        FeatureRow::MsaaDepthResolve,
        FeatureRow::ArrayOfTexturesRead,
        FeatureRow::ArrayOfTexturesWrite,
        FeatureRow::CubeMapTextureArrays,
        FeatureRow::StencilTextureViews,
        FeatureRow::ArrayOfSamplers,
        FeatureRow::SamplerMaxAnisotropy,
        FeatureRow::SamplerLodClamp,
        FeatureRow::SamplerCompareFunctions,
        FeatureRow::UnsignedShortCoordinates,
        FeatureRow::BorderColor,
        FeatureRow::CountingOcclusionQuery,
        FeatureRow::BaseVertexInstanceDrawing,
        FeatureRow::LayeredRendering,
        FeatureRow::LayeredRenderingToMultisampleTextures,
        FeatureRow::MemorylessRenderTargets,
        FeatureRow::DualSourceBlending,
        FeatureRow::StoreAndMultisampleResolve,
        FeatureRow::MsaaBlits,
        FeatureRow::ProgrammableSamplePositions,
        FeatureRow::DeferredStoreAction,
        FeatureRow::MemoryBarriers,
        FeatureRow::IcbComputeMemoryBarriers,
        FeatureRow::IcbRenderMemoryBarriers,
        FeatureRow::Tessellation,
        FeatureRow::IndirectTessellationArguments,
        FeatureRow::IcbTessellation,
        FeatureRow::ResourceHeaps,
        FeatureRow::FunctionSpecialization,
        FeatureRow::ReadWriteBuffers,
        FeatureRow::ReadWriteTextures,
        FeatureRow::ExtractInsertReverseBits,
        FeatureRow::SimdBarrier,
        FeatureRow::IndirectDrawDispatchArguments,
        FeatureRow::ArgumentBuffersTier1,
        FeatureRow::ArgumentBuffersTier2,
        FeatureRow::IcbRendering,
        FeatureRow::IcbCompute,
        FeatureRow::UniformType,
        FeatureRow::Imageblocks,
        FeatureRow::TileShaders,
        FeatureRow::ImageblockSampleCoverageControl,
        FeatureRow::PostDepthCoverage,
        FeatureRow::QuadPermute,
        FeatureRow::QuadBallot,
        FeatureRow::QuadReduction,
        FeatureRow::SimdPermute,
        FeatureRow::SimdReduction,
        FeatureRow::SimdMatrixMultiply,
        FeatureRow::RasterOrderGroups,
        FeatureRow::NonuniformThreadgroupSize,
        FeatureRow::MultipleViewports,
        FeatureRow::StencilFeedback,
        FeatureRow::StencilResolve,
        FeatureRow::NonsquareTileDispatch,
        FeatureRow::TextureSwizzle,
        FeatureRow::PlacementHeap,
        FeatureRow::PrimitiveId,
        FeatureRow::BarycentricCoordinates,
        FeatureRow::ReadWriteCubeMapTextures,
        FeatureRow::SparseTextures,
        FeatureRow::SparseDepthStencilTextures,
        FeatureRow::VariableRasterizationRate,
        FeatureRow::VertexAmplification,
        FeatureRow::Int64Math,
        FeatureRow::LossyTextureCompression,
        FeatureRow::SimdShiftAndFill,
        FeatureRow::RenderDynamicLibraries,
        FeatureRow::ComputeDynamicLibraries,
        FeatureRow::MeshShading,
        FeatureRow::IndirectMeshDrawArguments,
        FeatureRow::IcbMeshDraws,
        FeatureRow::MetalFxSpatialUpscaling,
        FeatureRow::MetalFxTemporalUpscaling,
        FeatureRow::MetalFxFrameInterpolation,
        FeatureRow::MetalFxDenoisedUpscaling,
        FeatureRow::FastResourceLoading,
        FeatureRow::RayTracingCompute,
        FeatureRow::RayTracingRender,
        FeatureRow::FloatAtomics,
        FeatureRow::TextureAtomics,
        FeatureRow::Int64Atomics,
        FeatureRow::QueryTextureLod,
        FeatureRow::BinaryArchives,
        FeatureRow::FunctionPointersCompute,
        FeatureRow::FunctionPointersRender,
        FeatureRow::DepthCompareBiasAndGradient,
        FeatureRow::NonprivateDepthStencilTextures,
        FeatureRow::DynamicAttributeStride,
        FeatureRow::PackedFloatAttributeFormats,
        FeatureRow::BFloat,
        FeatureRow::RelaxedMath,
        FeatureRow::GlobalBuiltinsAndBindings,
        FeatureRow::CoherentMemory,
        FeatureRow::ShaderValidation,
        FeatureRow::ShaderLogging,
        FeatureRow::ResidencySets,
        FeatureRow::RowMajorAccelerationStructures,
        FeatureRow::PerComponentMotionInterpolation,
        FeatureRow::IntersectionResultStorage,
        FeatureRow::VisibilityCountAccumulation,
        FeatureRow::ArgumentTables,
        FeatureRow::CommandAllocators,
        FeatureRow::DecoupledQueuesAndCommandBuffers,
        FeatureRow::TextureViewPools,
        FeatureRow::CommandBarriers,
        FeatureRow::PlacementSparseBuffers,
        FeatureRow::PlacementSparseTextures,
        FeatureRow::CompilationContexts,
        FeatureRow::PipelineDatasetSerialization,
        FeatureRow::FlexibleRenderPipelineState,
        FeatureRow::ColorAttachmentMapping,
        FeatureRow::MachineLearningEncoding,
        FeatureRow::Tensors,
        FeatureRow::PerformanceCounterHeaps,
        FeatureRow::AddressDrivenAccelerationStructureBuilds,
        FeatureRow::AccelerationStructureBuildOptions,
        FeatureRow::IntersectionFunctionBuffers,
        FeatureRow::SamplerMinMaxReduction,
        FeatureRow::SamplerLodBias,
        FeatureRow::PreRasterVertexValues,
        FeatureRow::DepthBoundsTesting,
        FeatureRow::IcbRasterAndDepthStencilState,
        FeatureRow::CubeMapAtomics,
        FeatureRow::UniversalTextureCompression,
        FeatureRow::AcquireReleaseAtomics,
        FeatureRow::BlockTextureReads,
        FeatureRow::ClampToEdgeTextureReads,
        FeatureRow::IntegerCoordinateOffsets,
        FeatureRow::TextureBarriers,
        FeatureRow::DeviceNotifications,
    ];

    /// The row's name as Apple's tables print it (the 2026 edition's, where it has the row).
    pub fn label(self) -> &'static str {
        match self {
            FeatureRow::MetalKit => "MetalKit",
            FeatureRow::MetalPerformanceShaders => "Metal performance shaders",
            FeatureRow::ProgrammableBlending => "Programmable blending",
            FeatureRow::SeparateDepthStencilAttachments => "Separate depth and stencil attachments",
            FeatureRow::PvrtcPixelFormats => "PVRTC pixel formats",
            FeatureRow::EacEtcPixelFormats => "EAC/ETC pixel formats",
            FeatureRow::AstcPixelFormats => "ASTC pixel formats",
            FeatureRow::BcPixelFormats => "BC pixel formats",
            FeatureRow::CompressedVolumeTextureFormats => "Compressed volume texture formats",
            FeatureRow::ExtendedRangePixelFormats => "Extended range pixel formats",
            FeatureRow::WideColorPixelFormat => "Wide color pixel format",
            FeatureRow::Depth16PixelFormat => "Depth-16 pixel format",
            FeatureRow::LinearTextures => "Linear textures",
            FeatureRow::MsaaDepthResolve => "MSAA depth resolve",
            FeatureRow::ArrayOfTexturesRead => "Array of textures (read)",
            FeatureRow::ArrayOfTexturesWrite => "Array of textures (write)",
            FeatureRow::CubeMapTextureArrays => "Cube map texture arrays",
            FeatureRow::StencilTextureViews => "Stencil texture views",
            FeatureRow::ArrayOfSamplers => "Array of samplers",
            FeatureRow::SamplerMaxAnisotropy => "Sampler maximum anisotropy",
            FeatureRow::SamplerLodClamp => "Sampler LOD clamp",
            FeatureRow::SamplerCompareFunctions => {
                "MTLSamplerState support for comparison functions"
            }
            FeatureRow::UnsignedShortCoordinates => "16-bit unsigned integer coordinates",
            FeatureRow::BorderColor => "Border color",
            FeatureRow::CountingOcclusionQuery => "Counting occlusion query",
            FeatureRow::BaseVertexInstanceDrawing => "Base vertex/instance drawing",
            FeatureRow::LayeredRendering => "Layered rendering",
            FeatureRow::LayeredRenderingToMultisampleTextures => {
                "Layered rendering to multisample textures"
            }
            FeatureRow::MemorylessRenderTargets => "Memoryless render targets",
            FeatureRow::DualSourceBlending => "Dual-source blending",
            FeatureRow::StoreAndMultisampleResolve => "Combined MSAA store and resolve action",
            FeatureRow::MsaaBlits => "MSAA blits",
            FeatureRow::ProgrammableSamplePositions => "Programmable sample positions",
            FeatureRow::DeferredStoreAction => "Deferred store action",
            FeatureRow::MemoryBarriers => "Memory barriers",
            FeatureRow::IcbComputeMemoryBarriers => {
                "Indirect command buffer support for memory barriers (compute)"
            }
            FeatureRow::IcbRenderMemoryBarriers => {
                "Indirect command buffer support for memory barriers (rendering)"
            }
            FeatureRow::Tessellation => "Tessellation",
            FeatureRow::IndirectTessellationArguments => "Indirect tessellation arguments",
            FeatureRow::IcbTessellation => "Tessellation in indirect command buffers",
            FeatureRow::ResourceHeaps => "Resource heaps",
            FeatureRow::FunctionSpecialization => "Function specialization",
            FeatureRow::ReadWriteBuffers => "Read/write buffers in functions",
            FeatureRow::ReadWriteTextures => "Read/write textures in functions",
            FeatureRow::ExtractInsertReverseBits => "Extract, insert, and reverse bits",
            FeatureRow::SimdBarrier => "SIMD barrier",
            FeatureRow::IndirectDrawDispatchArguments => "Indirect draw and dispatch arguments",
            FeatureRow::ArgumentBuffersTier1 => "Argument buffers tier 1",
            FeatureRow::ArgumentBuffersTier2 => "Argument buffers tier 2",
            FeatureRow::IcbRendering => "Indirect command buffers (rendering)",
            FeatureRow::IcbCompute => "Indirect command buffers (compute)",
            FeatureRow::UniformType => "Uniform type",
            FeatureRow::Imageblocks => "Imageblocks",
            FeatureRow::TileShaders => "Tile shaders",
            FeatureRow::ImageblockSampleCoverageControl => "Imageblock sample coverage control",
            FeatureRow::PostDepthCoverage => "Post-depth coverage",
            FeatureRow::QuadPermute => "Quad-scoped permute operations",
            FeatureRow::QuadBallot => "Quad-scoped ballot operation",
            FeatureRow::QuadReduction => "Quad-scoped reduction operations",
            FeatureRow::SimdPermute => "SIMD-scoped permute operations",
            FeatureRow::SimdReduction => "SIMD-scoped reduction operations",
            FeatureRow::SimdMatrixMultiply => "SIMD-scoped matrix multiply operations",
            FeatureRow::RasterOrderGroups => "Raster order groups",
            FeatureRow::NonuniformThreadgroupSize => "Nonuniform threadgroup size",
            FeatureRow::MultipleViewports => "Multiple viewports",
            FeatureRow::StencilFeedback => "Stencil feedback",
            FeatureRow::StencilResolve => "Stencil resolve",
            FeatureRow::NonsquareTileDispatch => "Nonsquare tile dispatch",
            FeatureRow::TextureSwizzle => "Texture swizzle",
            FeatureRow::PlacementHeap => "Placement heap",
            FeatureRow::PrimitiveId => "Primitive ID",
            FeatureRow::BarycentricCoordinates => "Barycentric coordinates",
            FeatureRow::ReadWriteCubeMapTextures => "Read/write cube map textures in functions",
            FeatureRow::SparseTextures => "Sparse textures",
            FeatureRow::SparseDepthStencilTextures => "Sparse depth and stencil textures",
            FeatureRow::VariableRasterizationRate => "Variable rasterization rate",
            FeatureRow::VertexAmplification => "Vertex amplification",
            FeatureRow::Int64Math => "64-bit integer math",
            FeatureRow::LossyTextureCompression => "Lossy texture compression",
            FeatureRow::SimdShiftAndFill => "SIMD shift and fill",
            FeatureRow::RenderDynamicLibraries => "Render dynamic libraries",
            FeatureRow::ComputeDynamicLibraries => "Compute dynamic libraries",
            FeatureRow::MeshShading => "Mesh shading",
            FeatureRow::IndirectMeshDrawArguments => "Indirect mesh draw arguments",
            FeatureRow::IcbMeshDraws => "Indirect command buffers containing mesh draws",
            FeatureRow::MetalFxSpatialUpscaling => "MetalFX spatial upscaling",
            FeatureRow::MetalFxTemporalUpscaling => "MetalFX temporal upscaling",
            FeatureRow::MetalFxFrameInterpolation => "MetalFX frame interpolation",
            FeatureRow::MetalFxDenoisedUpscaling => "MetalFX denoised upscaling",
            FeatureRow::FastResourceLoading => "Fast resource loading",
            FeatureRow::RayTracingCompute => "Ray tracing in compute pipelines",
            FeatureRow::RayTracingRender => "Ray tracing in render pipelines",
            FeatureRow::FloatAtomics => "Floating-point atomics",
            FeatureRow::TextureAtomics => "Texture atomics",
            FeatureRow::Int64Atomics => "64-bit atomics",
            FeatureRow::QueryTextureLod => "Query texture LOD",
            FeatureRow::BinaryArchives => "Binary archives",
            FeatureRow::FunctionPointersCompute => "Function pointers in compute pipelines",
            FeatureRow::FunctionPointersRender => "Function pointers in render pipelines",
            FeatureRow::DepthCompareBiasAndGradient => "Depth sample compare bias and gradient",
            FeatureRow::NonprivateDepthStencilTextures => "Nonprivate depth stencil textures",
            FeatureRow::DynamicAttributeStride => "Dynamic stride for attribute buffers",
            FeatureRow::PackedFloatAttributeFormats => {
                "MTLAttributeFormat.floatRGB9E5 and .floatRG11B10"
            }
            FeatureRow::BFloat => "MTLDataType.bfloat (brain float) scalar and vector cases",
            FeatureRow::RelaxedMath => "Relaxed math",
            FeatureRow::GlobalBuiltinsAndBindings => "Global built-ins and bindings",
            FeatureRow::CoherentMemory => "Memory coherence for textures and buffers in shaders",
            FeatureRow::ShaderValidation => "Per-pipeline shader validation",
            FeatureRow::ShaderLogging => "Shader logging",
            FeatureRow::ResidencySets => "Residency sets",
            FeatureRow::RowMajorAccelerationStructures => {
                "Acceleration structures containing row-major matrices"
            }
            FeatureRow::PerComponentMotionInterpolation => {
                "Ray tracing with per-component motion interpolation"
            }
            FeatureRow::IntersectionResultStorage => {
                "Direct access to on-chip ray-intersection result storage"
            }
            FeatureRow::VisibilityCountAccumulation => "Fragment visibility count accumulation",
            FeatureRow::ArgumentTables => "Argument tables",
            FeatureRow::CommandAllocators => "Command allocators",
            FeatureRow::DecoupledQueuesAndCommandBuffers => {
                "Decoupled command queues and command buffers"
            }
            FeatureRow::TextureViewPools => "Texture view pools",
            FeatureRow::CommandBarriers => "Command barriers",
            FeatureRow::PlacementSparseBuffers => "Placement sparse buffers",
            FeatureRow::PlacementSparseTextures => "Placement sparse textures",
            FeatureRow::CompilationContexts => "Dedicated compilation contexts",
            FeatureRow::PipelineDatasetSerialization => "Pipeline dataset serialization",
            FeatureRow::FlexibleRenderPipelineState => "Flexible render pipeline state",
            FeatureRow::ColorAttachmentMapping => "Color attachment mapping",
            FeatureRow::MachineLearningEncoding => "Machine learning encoding",
            FeatureRow::Tensors => "Tensors",
            FeatureRow::PerformanceCounterHeaps => "Performance counter heaps",
            FeatureRow::AddressDrivenAccelerationStructureBuilds => {
                "Address-driven acceleration structure builds"
            }
            FeatureRow::AccelerationStructureBuildOptions => "Acceleration structure build options",
            FeatureRow::IntersectionFunctionBuffers => "Intersection function buffers",
            FeatureRow::SamplerMinMaxReduction => "Sampler minimum and maximum reduction",
            FeatureRow::SamplerLodBias => "Sampler LOD bias",
            FeatureRow::PreRasterVertexValues => "Access to pre-raster per-vertex values",
            FeatureRow::DepthBoundsTesting => "Depth bounds testing",
            FeatureRow::IcbRasterAndDepthStencilState => {
                "Indirect command buffer support for raster and depth/stencil state"
            }
            FeatureRow::CubeMapAtomics => "Atomics on cube map and cube map array textures",
            FeatureRow::UniversalTextureCompression => "Universal texture compression",
            FeatureRow::AcquireReleaseAtomics => "Acquire and release memory ordering atomics",
            FeatureRow::BlockTextureReads => "2 x 1 and 1 x 2 block texture reads",
            FeatureRow::ClampToEdgeTextureReads => "Clamp to edge behavior for texture reads",
            FeatureRow::IntegerCoordinateOffsets => {
                "Integer coordinate offsets for texture reads and atomics"
            }
            FeatureRow::TextureBarriers => "Texture barriers",
            FeatureRow::DeviceNotifications => "Device notifications",
        }
    }

    /// The columns that list the row (the module docs of `families`): the first Apple family of
    /// the 2026 edition's row of the Metal 3 programming model (0 for none, as for a row of Metal
    /// 4's model only), the 2023 edition's Common1 to Common3 cells (bit 0 to bit 2), its Mac2
    /// cell, and the Metal3 column's answer.
    fn listing(self) -> Listing {
        use FeatureRow as F;
        let l = |apple, common, mac2, metal3| Listing {
            apple,
            common,
            mac2,
            metal3,
        };
        match self {
            F::MetalKit => l(2, 0b111, true, true),
            F::MetalPerformanceShaders => l(2, 0b110, true, true),
            F::ProgrammableBlending => l(2, 0b000, false, false),
            F::SeparateDepthStencilAttachments => l(2, 0b000, false, true),
            F::PvrtcPixelFormats => l(2, 0b000, false, false),
            F::EacEtcPixelFormats => l(2, 0b000, false, false),
            F::AstcPixelFormats => l(2, 0b000, false, false),
            F::BcPixelFormats => l(9, 0b000, true, false),
            F::CompressedVolumeTextureFormats => l(3, 0b110, true, true),
            F::ExtendedRangePixelFormats => l(3, 0b000, false, false),
            F::WideColorPixelFormat => l(2, 0b111, true, true),
            F::Depth16PixelFormat => l(2, 0b111, true, true),
            F::LinearTextures => l(2, 0b111, true, true),
            F::MsaaDepthResolve => l(3, 0b100, true, true),
            F::ArrayOfTexturesRead => l(3, 0b110, true, true),
            F::ArrayOfTexturesWrite => l(6, 0b110, true, true),
            F::CubeMapTextureArrays => l(4, 0b110, true, true),
            F::StencilTextureViews => l(2, 0b111, true, true),
            F::ArrayOfSamplers => l(3, 0b110, true, true),
            F::SamplerMaxAnisotropy => l(2, 0b111, true, true),
            F::SamplerLodClamp => l(2, 0b111, true, true),
            F::SamplerCompareFunctions => l(3, 0b110, true, true),
            F::UnsignedShortCoordinates => l(2, 0b111, true, true),
            F::BorderColor => l(7, 0b000, true, true),
            F::CountingOcclusionQuery => l(3, 0b110, true, true),
            F::BaseVertexInstanceDrawing => l(3, 0b110, true, true),
            F::LayeredRendering => l(5, 0b100, true, true),
            F::LayeredRenderingToMultisampleTextures => l(7, 0b100, true, true),
            F::MemorylessRenderTargets => l(2, 0b000, false, false),
            F::DualSourceBlending => l(2, 0b111, true, true),
            F::StoreAndMultisampleResolve => l(3, 0b110, true, true),
            F::MsaaBlits => l(2, 0b111, true, true),
            F::ProgrammableSamplePositions => l(2, 0b111, true, true),
            F::DeferredStoreAction => l(2, 0b111, true, true),
            F::MemoryBarriers => l(3, 0b110, true, true),
            F::IcbComputeMemoryBarriers => l(3, 0b110, true, true),
            F::IcbRenderMemoryBarriers => l(9, 0b000, false, false),
            F::Tessellation => l(3, 0b110, true, true),
            F::IndirectTessellationArguments => l(5, 0b000, true, true),
            F::IcbTessellation => l(5, 0b000, true, true),
            F::ResourceHeaps => l(2, 0b111, true, true),
            F::FunctionSpecialization => l(2, 0b111, true, true),
            F::ReadWriteBuffers => l(3, 0b110, true, true),
            F::ReadWriteTextures => l(4, 0b100, true, true),
            F::ExtractInsertReverseBits => l(2, 0b111, true, true),
            F::SimdBarrier => l(2, 0b111, true, true),
            F::IndirectDrawDispatchArguments => l(3, 0b110, true, true),
            F::ArgumentBuffersTier1 => l(2, 0b000, true, true),
            F::ArgumentBuffersTier2 => l(6, 0b000, true, true),
            F::IcbRendering => l(3, 0b110, true, true),
            F::IcbCompute => l(3, 0b110, true, true),
            F::UniformType => l(2, 0b111, true, true),
            F::Imageblocks => l(4, 0b000, false, false),
            F::TileShaders => l(4, 0b000, false, false),
            F::ImageblockSampleCoverageControl => l(4, 0b000, false, false),
            F::PostDepthCoverage => l(4, 0b000, false, false),
            F::QuadPermute => l(4, 0b100, true, true),
            F::QuadBallot => l(6, 0b000, false, true),
            F::QuadReduction => l(7, 0b000, false, true),
            F::SimdPermute => l(6, 0b000, true, true),
            F::SimdReduction => l(7, 0b000, true, true),
            F::SimdMatrixMultiply => l(7, 0b000, false, false),
            F::RasterOrderGroups => l(4, 0b100, false, true),
            F::NonuniformThreadgroupSize => l(4, 0b100, true, true),
            F::MultipleViewports => l(5, 0b100, true, true),
            F::StencilFeedback => l(5, 0b100, true, true),
            F::StencilResolve => l(5, 0b100, true, true),
            F::NonsquareTileDispatch => l(5, 0b000, false, false),
            F::TextureSwizzle => l(2, 0b000, true, true),
            F::PlacementHeap => l(2, 0b000, true, true),
            F::PrimitiveId => l(7, 0b000, true, true),
            F::BarycentricCoordinates => l(7, 0b000, false, false),
            F::ReadWriteCubeMapTextures => l(4, 0b000, true, true),
            F::SparseTextures => l(6, 0b000, false, false),
            F::SparseDepthStencilTextures => l(8, 0b000, false, false),
            F::VariableRasterizationRate => l(6, 0b000, false, false),
            F::VertexAmplification => l(6, 0b000, false, false),
            F::Int64Math => l(3, 0b000, false, true),
            F::LossyTextureCompression => l(8, 0b000, false, false),
            F::SimdShiftAndFill => l(8, 0b000, false, false),
            F::RenderDynamicLibraries => l(6, 0b000, false, false),
            F::ComputeDynamicLibraries => l(6, 0b000, true, true),
            F::MeshShading => l(7, 0b000, true, true),
            F::IndirectMeshDrawArguments => l(9, 0b000, false, false),
            F::IcbMeshDraws => l(9, 0b000, false, false),
            F::MetalFxSpatialUpscaling => l(3, 0b110, true, true),
            F::MetalFxTemporalUpscaling => l(7, 0b000, false, false),
            F::MetalFxFrameInterpolation => l(5, 0b000, false, true),
            F::MetalFxDenoisedUpscaling => l(9, 0b000, false, false),
            F::FastResourceLoading => l(2, 0b111, true, true),
            F::RayTracingCompute => l(6, 0b000, false, true),
            F::RayTracingRender => l(6, 0b000, false, false),
            F::FloatAtomics => l(7, 0b000, true, false),
            F::TextureAtomics => l(6, 0b000, true, true),
            F::Int64Atomics => l(9, 0b000, false, false),
            F::QueryTextureLod => l(8, 0b000, false, false),
            F::BinaryArchives => l(3, 0b110, true, true),
            F::FunctionPointersCompute => l(6, 0b000, false, true),
            F::FunctionPointersRender => l(6, 0b000, false, false),
            F::DepthCompareBiasAndGradient => l(2, 0b000, false, false),
            F::NonprivateDepthStencilTextures => l(2, 0b000, false, false),
            F::DynamicAttributeStride => l(4, 0b100, true, true),
            F::PackedFloatAttributeFormats => l(5, 0b100, true, true),
            F::BFloat => l(6, 0b000, true, true),
            F::RelaxedMath => l(4, 0b000, false, true),
            F::GlobalBuiltinsAndBindings => l(6, 0b000, false, true),
            F::CoherentMemory => l(6, 0b000, false, true),
            F::ShaderValidation => l(6, 0b000, false, true),
            F::ShaderLogging => l(6, 0b000, false, true),
            F::ResidencySets => l(6, 0b000, false, true),
            F::RowMajorAccelerationStructures => l(9, 0b000, false, false),
            F::PerComponentMotionInterpolation => l(9, 0b000, false, false),
            F::IntersectionResultStorage => l(9, 0b000, false, false),
            F::VisibilityCountAccumulation => l(7, 0b000, false, true),
            F::ArgumentTables => l(0, 0b000, false, false),
            F::CommandAllocators => l(0, 0b000, false, false),
            F::DecoupledQueuesAndCommandBuffers => l(0, 0b000, false, false),
            F::TextureViewPools => l(7, 0b000, false, true),
            F::CommandBarriers => l(0, 0b000, false, false),
            F::PlacementSparseBuffers => l(0, 0b000, false, false),
            F::PlacementSparseTextures => l(0, 0b000, false, false),
            F::CompilationContexts => l(0, 0b000, false, false),
            F::PipelineDatasetSerialization => l(0, 0b000, false, false),
            F::FlexibleRenderPipelineState => l(0, 0b000, false, false),
            F::ColorAttachmentMapping => l(7, 0b000, false, true),
            F::MachineLearningEncoding => l(0, 0b000, false, false),
            F::Tensors => l(0, 0b000, false, false),
            F::PerformanceCounterHeaps => l(0, 0b000, false, false),
            F::AddressDrivenAccelerationStructureBuilds => l(0, 0b000, false, false),
            F::AccelerationStructureBuildOptions => l(9, 0b000, false, false),
            F::IntersectionFunctionBuffers => l(9, 0b000, false, false),
            F::SamplerMinMaxReduction => l(10, 0b000, false, false),
            F::SamplerLodBias => l(10, 0b000, false, false),
            F::PreRasterVertexValues => l(10, 0b000, false, false),
            F::DepthBoundsTesting => l(10, 0b000, false, false),
            F::IcbRasterAndDepthStencilState => l(10, 0b000, false, false),
            F::CubeMapAtomics => l(10, 0b000, false, false),
            F::UniversalTextureCompression => l(10, 0b000, false, false),
            F::AcquireReleaseAtomics => l(7, 0b000, false, true),
            F::BlockTextureReads => l(7, 0b000, false, true),
            F::ClampToEdgeTextureReads => l(7, 0b000, false, true),
            F::IntegerCoordinateOffsets => l(7, 0b000, false, true),
            F::TextureBarriers => l(0, 0b000, true, false),
            F::DeviceNotifications => l(0, 0b000, true, false),
        }
    }
}

/// The columns that list a feature row ([`FeatureRow::listing`]).
#[derive(Clone, Copy, Debug)]
struct Listing {
    /// The first Apple family that lists the row, or 0 for none.
    apple: u8,
    /// The Common families that list the row: bit 0 for Common1 to bit 2 for Common3.
    common: u8,
    mac2: bool,
    metal3: bool,
}

/// One check of a row: the state that blocks it, or `None` where the row passes the check.
pub(super) type Check = Option<RowState>;

impl FeatureRow {
    /// Whether the family read from `column` lists the row.
    pub(super) fn listed(self, column: Column) -> bool {
        let l = self.listing();
        match column {
            Column::Apple(n) => l.apple != 0 && n >= l.apple,
            Column::Common(n) => l.common & (1 << (n - 1)) != 0,
            Column::Mac2 => l.mac2,
            Column::Metal3 => l.metal3,
        }
    }

    /// The row's state on a device of `f`, for a family read from `column` (the module docs).
    pub(super) fn state(self, column: Column, f: &Facts<'_>) -> RowState {
        use FeatureRow as F;
        use FormatCapability::{Atomic, Filter};
        use RuntimeGap as G;
        let d = &f.properties.description;
        match self {
            // ---- Frameworks layered on Metal: shown, never blocking (the families' docs) ----
            F::MetalKit => RowState::NotADeviceFeature(Framework::MetalKit),
            F::MetalPerformanceShaders => {
                RowState::NotADeviceFeature(Framework::MetalPerformanceShaders)
            }
            F::MetalFxSpatialUpscaling
            | F::MetalFxTemporalUpscaling
            | F::MetalFxFrameInterpolation
            | F::MetalFxDenoisedUpscaling => RowState::NotADeviceFeature(Framework::MetalFx),

            // ---- Pixel formats: the runtime's per-format answers (`formats`) ----
            // Each format of the row, sampled through a linear filter.
            F::PvrtcPixelFormats => first([formats(f, &[FormatRow::PvrtcFormats], Filter)]),
            F::EacEtcPixelFormats => first([formats(f, &[FormatRow::EacEtcFormats], Filter)]),
            F::AstcPixelFormats => first([formats(f, &[FormatRow::AstcFormats], Filter)]),
            F::BcPixelFormats => first([formats(f, &[FormatRow::BcFormats], Filter)]),
            F::ExtendedRangePixelFormats => first([formats(
                f,
                &[
                    FormatRow::BGRA10XR,
                    FormatRow::BGRA10XRSRGB,
                    FormatRow::BGR10XR,
                    FormatRow::BGR10XRSRGB,
                ],
                Filter,
            )]),
            F::WideColorPixelFormat => first([formats(f, &[FormatRow::BGR10A2Unorm], Filter)]),
            F::Depth16PixelFormat => first([formats(f, &[FormatRow::Depth16Unorm], Filter)]),
            // A 3D texture of each ASTC (LDR and HDR) and BC format whose row the family lists
            // with Filter. Metal makes no 3D texture of the other compressed formats (measured,
            // the module docs).
            F::CompressedVolumeTextureFormats => first([listed_formats(
                &[
                    FormatRow::AstcFormats,
                    FormatRow::HdrAstcFormats,
                    FormatRow::BcFormats,
                ],
                |row| row.listed(column).contains(Filter),
                |pf| (f.format)(pf).volume(),
            )]),
            // `MTLSamplerBorderColor`: Metal's opaque black border, read through the sampled view
            // of each format whose row the family lists (`metal::sampler::check_border_view`).
            F::BorderColor => first([listed_formats(
                &FormatRow::ALL,
                |row| row.listed(column) != super::FormatCapabilities::NONE,
                |pf| (f.format)(pf).border(),
            )]),
            // Texture atomics on `R32Uint` and `R32Sint` textures. Metal's `ulong` texture atomics
            // are the 64-bit atomics row's.
            F::TextureAtomics => first([formats(
                f,
                &[FormatRow::R32Uint, FormatRow::R32Sint],
                Atomic,
            )]),
            // Apple10's lossless compression of every texture reads back the bits written. Its
            // opt-out (`allow_gpu_optimized_contents`) and `optimize_contents_for_gpu_access` are
            // taken.
            F::UniversalTextureCompression => RowState::Exact,

            // ---- Textures and samplers ----
            // A texture over a buffer (`Buffer::new_texture`): made, sampled, written, blitted
            // and rendered to. Still refused (`NotImplemented(Textures)`): a texture buffer's view
            // in a format that serves one kind of texel buffer, on a device without `maintenance5`
            // (`buffer_texture::view_usage`), and a texture over a buffer whose image no copy can
            // write (`texture.rs`).
            F::LinearTextures => first([gap(G::Textures)]),
            // The stencil view of a combined depth/stencil texture (`X32Stencil8` of
            // `Depth32FloatStencil8`, `Texture::new_texture_view`), the one depth or stencil view
            // of another format Metal makes: sampled, read, blitted and copied by the CPU as
            // plain Metal does (probes).
            F::StencilTextureViews => RowState::Exact,
            // Arrays of textures and samplers indexed by a value the function computes: the
            // compile layer's descriptor arrays, which `shader*ArrayDynamicIndexing` lets it index.
            F::ArrayOfTexturesRead => first([needs_features(
                f,
                Feature::Lowering(LowerFamily::Texture),
                &[VkFeature::ShaderSampledImageArrayDynamicIndexing],
            )]),
            F::ArrayOfTexturesWrite => first([needs_features(
                f,
                Feature::Lowering(LowerFamily::Texture),
                &[VkFeature::ShaderStorageImageArrayDynamicIndexing],
            )]),
            F::ArrayOfSamplers => first([needs_features(
                f,
                Feature::Lowering(LowerFamily::Sampler),
                &[VkFeature::ShaderSampledImageArrayDynamicIndexing],
            )]),
            // `TextureType::TypeCubeArray`: `imageCubeArray`.
            F::CubeMapTextureArrays => first([needs_features(
                f,
                Feature::CubeArray,
                &[VkFeature::ImageCubeArray],
            )]),
            // `SamplerDescriptor::max_anisotropy`: `samplerAnisotropy`.
            F::SamplerMaxAnisotropy => first([needs_features(
                f,
                Feature::SamplerAnisotropy,
                &[VkFeature::SamplerAnisotropy],
            )]),
            // `SamplerDescriptor::lod_min_clamp` and `lod_max_clamp`: core Vulkan.
            F::SamplerLodClamp => RowState::Exact,
            // A sampler's compare function, and a depth sample's compare with a bias or gradient:
            // a comparison sampler, which a portability subset may withhold
            // (`mutableComparisonSamplers`).
            F::SamplerCompareFunctions | F::DepthCompareBiasAndGradient => first([withheld(
                f,
                Feature::SamplerCompare,
                VkFeature::MutableComparisonSamplers,
            )]),
            // `SamplerDescriptor::reduction_mode` (Apple10): `supportsSamplerReductionMode`, which
            // answers no on every device, as on an Apple M4: a sampler's minimum or maximum
            // reduction reads the weighted average.
            F::SamplerMinMaxReduction => first([query(
                f.properties.supports_sampler_reduction_mode(),
                Query::SamplerReductionMode,
            )]),
            // `SamplerDescriptor::lod_bias` (Apple10): not on the runtime's sampler descriptor.
            F::SamplerLodBias => first([gap(G::Textures)]),
            // 16-bit coordinates in a texture read, a texture's swizzle, `Shared` and `Managed`
            // depth/stencil textures, and the clamp-to-edge read: the compile layer and the
            // runtime on every device.
            F::UnsignedShortCoordinates
            | F::NonprivateDepthStencilTextures
            | F::ClampToEdgeTextureReads
            | F::QueryTextureLod => RowState::Exact,
            // `TextureDescriptor::swizzle`: a swizzled texture's sampled view composes it, and a
            // function applies a texture buffer's, but for a texture buffer read as `ulong` texels,
            // whose swizzled texel Metal is not measured to give (`bind::plan::resources`,
            // `NotImplemented(TextureSwizzle)`).
            F::TextureSwizzle => first([not_implemented(Feature::TextureSwizzle)]),
            // `access::read_write` textures in kernels need read-write tier 1 (a storage form of
            // `R32Float`, `R32Uint` and `R32Sint`), and in vertex and fragment functions the
            // stages' store features.
            F::ReadWriteTextures | F::ReadWriteCubeMapTextures => {
                first([read_write_tier(f), stage_stores(f)])
            }
            // Atomics on cube textures, 2 x 1 and 1 x 2 block reads, and integer coordinate
            // offsets of reads and atomics (Apple7 and Apple10): not lowered.
            F::CubeMapAtomics | F::BlockTextureReads | F::IntegerCoordinateOffsets => {
                first([not_implemented(Feature::Lowering(LowerFamily::Texture))])
            }

            // ---- Render passes and attachments ----
            // Framebuffer fetch (the compile layer's color inputs, the runtime's
            // `framebuffer_fetch`): exact through input attachments in rasterization order. A
            // device with the ordered pixel interlock but not that access (MoltenVK, whose
            // interlock is Metal's raster order groups, and RADV, ANV and NVIDIA) fetches through
            // the interlock's storage form (`lower::fragment::fetch`), exact for a fetch and for
            // an unblended write, and for Metal's blend of an extended-range, a 16-bit float, an
            // 8-bit unorm, an 8-bit sRGB or a packed 16-bit attachment; another float-class
            // format's blend is the shader's generic one, logged
            // (`PrecisionNote::UnmeasuredFetchBlend`), as are a blend of the second source and an
            // sRGB blend the 8-bit unorm model does not build. A device that renders some color
            // format only through the view of its word (`metal::format::WordAttachment`: the one-
            // and two-channel 8-bit sRGB formats on NVIDIA and ANV, and on NVIDIA the packed
            // 16-bit ones and `RGB9E5Float`) does not fetch an attachment
            // of it (`NotImplemented(Format)` per pipeline): the row's gap there. A device with
            // neither form fetches through input attachments ordered between draws, the
            // overlapping fragments of one draw unordered, logged
            // (`PrecisionNote::UnorderedFetchWithinDraw`).
            F::ProgrammableBlending => first([
                match through_storage(f.profile) && renders_through_words(f.profile) {
                    true => not_implemented(Feature::FramebufferFetch),
                    false => None,
                },
            ]),
            // Separate depth and stencil textures of one pass: their resolve of a stencil drawn
            // after the pass (`DepthResolvedSample` after `Min` or `Max`) is not built.
            F::SeparateDepthStencilAttachments => first([gap(G::DepthStencilResolve)]),
            // `MultisampleDepthResolveFilter` `Min` and `Max`: the device's
            // `supportedDepthResolveModes`.
            F::MsaaDepthResolve => first([(!d
                .limits
                .supported_depth_resolve_modes
                .contains(ResolveModeFlags::MIN.union(ResolveModeFlags::MAX)))
            .then_some(RowState::Gap(G::DepthStencilResolve))]),
            // `MultisampleStencilResolveFilter::DepthResolvedSample` after a `Min` or `Max` depth
            // resolve: no Vulkan resolve mode states it. The runtime draws it after the pass
            // (`depth_stencil_resolve.rs`) for a pass of one layer whose attachment is sampled, but
            // still refuses a layered pass and an attachment it cannot sample.
            F::StencilResolve => first([gap(G::DepthStencilResolve)]),
            // `StoreAction::StoreAndMultisampleResolve`, memoryless render targets, and the store
            // actions set after a pass's draws (`set_*_store_action`).
            F::StoreAndMultisampleResolve | F::MemorylessRenderTargets | F::DeferredStoreAction => {
                RowState::Exact
            }
            // `copy_from_texture` between multisampled textures: `vkCmdCopyImage` of equal sample
            // counts.
            F::MsaaBlits => RowState::Exact,
            // `VisibilityResultMode::Counting`: `occlusionQueryPrecise`.
            F::CountingOcclusionQuery => first([needs_features(
                f,
                Feature::CountingVisibility,
                &[VkFeature::OcclusionQueryPrecise],
            )]),
            // `set_sample_positions`: the runtime's query, which waits on render passes.
            F::ProgrammableSamplePositions => first([query(
                f.properties.are_programmable_sample_positions_supported(),
                Query::ProgrammableSamplePositions,
            )]),
            // `render_target_array_index`, stored from the vertex function (`shaderOutputLayer`) or
            // by the generated geometry stage (`geometryShader`), and passes of more than one layer
            // (`render_target_array_length`, `render_pass.rs`): built, an index at or past the
            // length included, and resolves, separate depth and stencil textures and stand-ins over
            // every layer; a tile dispatch in one is the tile shading row's.
            F::LayeredRendering => first([needs_any(
                f,
                Feature::LayeredRendering,
                &[
                    &[VkFeature::ShaderOutputLayer],
                    &[VkFeature::GeometryShader],
                ],
            )]),
            // The same, into multisampled array textures, which a portability subset may withhold
            // (`multisampleArrayImage`).
            F::LayeredRenderingToMultisampleTextures => first([
                needs_any(
                    f,
                    Feature::LayeredRendering,
                    &[
                        &[VkFeature::ShaderOutputLayer],
                        &[VkFeature::GeometryShader],
                    ],
                ),
                withheld(
                    f,
                    Feature::TextureSampleCount,
                    VkFeature::MultisampleArrayImage,
                ),
            ]),
            // `viewport_array_index` and `set_viewports`: `multiViewport`, and the index stored
            // from the vertex function (`shaderOutputViewportIndex`) or by the generated geometry
            // stage (`geometryShader`).
            F::MultipleViewports => first([
                needs_features(f, Feature::LayeredRendering, &[VkFeature::MultiViewport]),
                needs_any(
                    f,
                    Feature::LayeredRendering,
                    &[
                        &[VkFeature::ShaderOutputViewportIndex],
                        &[VkFeature::GeometryShader],
                    ],
                ),
            ]),
            // Dual-source blending: `dualSrcBlend`.
            F::DualSourceBlending => first([needs_features(
                f,
                Feature::DualSourceBlending,
                &[VkFeature::DualSrcBlend],
            )]),
            // `[[stencil]]` output: `VK_EXT_shader_stencil_export`. Without it the runtime draws
            // through the shader-side stencil, which takes only states whose stencil write the
            // depth test does not decide (the test runs after the function) and no alpha to
            // coverage: the binder and the link refuse the others as needing the extension, so the
            // row is the extension.
            F::StencilFeedback => first([(!f.profile.fragment().stencil_export).then(|| {
                refused(Unsupported::needs(
                    Feature::StencilExport,
                    Capability::Extension(VkExtension::ExtShaderStencilExport),
                ))
            })]),
            // `[[primitive_id]]` in a fragment function: the builtin through the geometry or
            // tessellation capability; on a device with neither, a varying the vertex function
            // writes, which an indexed draw's shared vertices cannot carry: an indexed list or
            // point draw runs de-indexed where the device addresses buffers, and an indexed
            // strip or indirect indexed draw is refused (`NotImplemented(PrimitiveId)`).
            F::PrimitiveId => {
                let d = f.profile.fragment();
                first([(!d.geometry_shader && !d.tessellation_shader)
                    .then(|| refused(Unsupported::not_implemented(Feature::PrimitiveId)))])
            }
            // `[[barycentric_coord]]`: `fragmentShaderBarycentric`, or the generated geometry
            // stage that writes the weights as varyings, which the compile layer lists where the
            // device lacks both.
            F::BarycentricCoordinates => first([listed(f, Feature::Barycentrics)]),
            // Raster order groups: the pixel interlock, which the compile layer lists where the
            // device lacks both interlocks, then the runtime's query, which answers yes where the
            // pixel interlock is enabled.
            F::RasterOrderGroups => first([
                listed(f, Feature::RasterOrderGroups),
                query(
                    f.properties.are_raster_order_groups_supported(),
                    Query::RasterOrderGroups,
                ),
            ]),
            // Vertex amplification: multiview, and for what multiview does not draw as Metal does
            // the variant that draws one amplification at a time, which stores the slice and the
            // viewport from the vertex function (the layer and viewport index outputs) or by the
            // generated geometry stage, the viewport among more than one (`multiViewport`); then
            // the runtime's query of two views; then a tessellation pipeline drawn so, which the
            // binder does not implement.
            F::VertexAmplification => first([
                needs_features(
                    f,
                    Feature::VertexAmplification,
                    &[VkFeature::Multiview, VkFeature::MultiViewport],
                ),
                needs_any(
                    f,
                    Feature::VertexAmplification,
                    &[
                        &[VkFeature::ShaderOutputLayer],
                        &[VkFeature::GeometryShader],
                    ],
                ),
                needs_any(
                    f,
                    Feature::VertexAmplification,
                    &[
                        &[VkFeature::ShaderOutputViewportIndex],
                        &[VkFeature::GeometryShader],
                    ],
                ),
                query(
                    f.properties.supports_vertex_amplification_count(2),
                    Query::VertexAmplificationCount,
                ),
                not_implemented(Feature::VertexAmplification),
            ]),
            // A render pass with a map draws region by region, each slice by its own
            // layer's regions: a pipeline whose vertex function chooses its slice over a map whose
            // layers differ is refused, and so is a scissor that cuts into the viewport over a map
            // that is not the identity.
            F::VariableRasterizationRate => first([
                gap(G::RateMapRegions(RegionRefusal::LayeredPipeline)),
                gap(G::RateMapRegions(RegionRefusal::Scissor)),
            ]),
            // Base vertex and instance of a direct draw: the compile layer's draw parameters, or
            // its emulation.
            F::BaseVertexInstanceDrawing => RowState::Exact,
            // The encoders' memory barriers (`memory_barrier_with_scope` and
            // `memory_barrier_with_resources`, `fence.rs`), and a buffer's attribute stride set
            // per bind, by the render and compute encoders and by indirect commands
            // (`render/vertex_strides.rs`, `compute.rs`, `indirect_command_buffer.rs`): built.
            F::MemoryBarriers | F::DynamicAttributeStride => RowState::Exact,
            // The render encoder's `textureBarrier`, a visibility result accumulated across
            // passes, and depth bounds: not built.
            F::TextureBarriers | F::VisibilityCountAccumulation | F::DepthBoundsTesting => {
                first([gap(G::Render)])
            }
            // Patch draws: the runtime's patch draws, direct and indirect, are built
            // (`tessellation.rs`), through the device's tessellation stages where it enables
            // `tessellationShader` and emulated elsewhere (`render/tessellation_scan.rs`). The
            // compile layer still refuses a post-tessellation vertex function whose patch states
            // no control-point count (`[[patch(kind)]]`), and with the device's stages a fragment
            // function behind one that reads `[[primitive_id]]` (`NotImplemented(Tessellation)`).
            F::Tessellation | F::IndirectTessellationArguments => {
                first([not_implemented(Feature::Tessellation)])
            }
            // Coverage after the depth test (`[[sample_mask, post_depth_coverage]]`): `SampleMask`
            // under the `PostDepthCoverage` mode, which `VK_EXT_post_depth_coverage` provides. A
            // driver that splits a clipped triangle's pixel-rate fragments refuses every coverage
            // read of a multisampled pipeline (`pipeline::link`), the only one where it differs.
            F::PostDepthCoverage => {
                let split = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
                first([
                    (!f.profile
                        .enabled()
                        .extensions
                        .contains(&VkExtension::ExtPostDepthCoverage))
                    .then(|| {
                        refused(Unsupported::needs(
                            Feature::PostDepthCoverage,
                            Capability::Extension(VkExtension::ExtPostDepthCoverage),
                        ))
                    }),
                    f.profile.workarounds().contains(&split).then(|| {
                        refused(Unsupported::driver_defect(Feature::PixelCoverage, split))
                    }),
                ])
            }
            // Pre-raster per-vertex values, and global built-ins and bindings: not lowered.
            F::PreRasterVertexValues | F::GlobalBuiltinsAndBindings => {
                first([not_implemented(Feature::Lowering(LowerFamily::StageIo))])
            }

            // ---- Shaders and compute ----
            // Function constants (`Library::new_function_with_name_constant_values`), bit
            // extraction, uniform types, nonuniform threadgroups (`dispatch_threads` by regions),
            // 64-bit integers (two words), floating-point atomics (a compare-and-swap loop where
            // the device has none), acquire and release orderings, `bfloat` (held widened), the
            // packed float vertex formats (fetched in the shader), and relaxed math.
            F::FunctionSpecialization
            | F::ExtractInsertReverseBits
            | F::UniformType
            | F::NonuniformThreadgroupSize
            | F::Int64Math
            | F::FloatAtomics
            | F::AcquireReleaseAtomics
            | F::BFloat
            | F::PackedFloatAttributeFormats
            | F::RelaxedMath => RowState::Exact,
            // Device buffers written in vertex and fragment functions: the stages' store features.
            F::ReadWriteBuffers => first([stage_stores(f)]),
            // `ulong` atomics on buffers (`shaderBufferInt64Atomics`, through device addresses) and
            // on `RG32Uint` textures (the format's atomic answer).
            F::Int64Atomics => first([
                needs_features(
                    f,
                    Feature::Int64Atomics,
                    &[VkFeature::ShaderBufferInt64Atomics],
                ),
                addressed(f, Feature::Int64Atomics),
                formats(f, &[FormatRow::RG32Uint], Atomic),
            ]),
            // Coherent device memory (`coherent(device)`): the Vulkan memory model at device
            // scope, then coherent textures, which are not lowered.
            F::CoherentMemory => first([
                needs_features(
                    f,
                    Feature::CoherentMemory,
                    &[
                        VkFeature::VulkanMemoryModel,
                        VkFeature::VulkanMemoryModelDeviceScope,
                    ],
                ),
                not_implemented(Feature::CoherentMemory),
            ]),
            // Simdgroup and quad operations in vertex and fragment functions (kernels emulate
            // every class the device's subgroups lack), each row asking the classes its members
            // lower to: the relative shuffles, rotations and fills are absolute shuffles, and a
            // quad's first active lane, prefix sums and reductions the ballot and shuffles. Each
            // row asks the classes its members lower to in the vertex and fragment stages
            // (`lower::simd::direct_ops`): a quad's reductions, prefix sums and votes read its
            // lanes by the ballot and shuffles, and a constant lane's broadcast or exchange
            // takes Vulkan's quad class only where the stage has it, a shuffle elsewhere, so no
            // row asks the quad class.
            F::SimdBarrier => first([simd(f, OpBits::BASIC, Feature::SubgroupOperations)]),
            F::QuadBallot => first([simd(f, OpBits::BALLOT, Feature::QuadOperations)]),
            F::QuadPermute => first([simd(
                f,
                OpBits::SHUFFLE.union(OpBits::BALLOT),
                Feature::QuadOperations,
            )]),
            F::QuadReduction => first([simd(
                f,
                OpBits::SHUFFLE.union(OpBits::BALLOT),
                Feature::QuadOperations,
            )]),
            F::SimdPermute => first([simd(
                f,
                OpBits::SHUFFLE.union(OpBits::BALLOT),
                Feature::SubgroupOperations,
            )]),
            F::SimdReduction => first([simd(f, OpBits::ARITHMETIC, Feature::SubgroupOperations)]),
            F::SimdShiftAndFill => first([simd(f, OpBits::SHUFFLE, Feature::SubgroupOperations)]),
            // Simdgroup matrices: lowered in kernels only.
            F::SimdMatrixMultiply => first([not_implemented(Feature::SimdgroupMatrix)]),
            // Indirect dispatches run exactly through their host step; indirect draws keep
            // their gap: an index offset at or past the index buffer's length on every device, and
            // more where the device lacks what the GPU-side draw needs.
            F::IndirectDrawDispatchArguments => first([gap(G::IndirectDrawArguments)]),
            // Argument buffers: buffer pointers, which the compile layer reaches by device
            // address, then the runtime's resident and handle tables, and set 1, the heap's set,
            // for dispatches, draws and tile dispatches alike. A program writes them directly, with
            // addresses and resource IDs, or through `MTLArgumentEncoder` (`ArgumentEncoder`),
            // which writes the same bytes.
            F::ArgumentBuffersTier1 => first([listed(f, Feature::ArgumentBufferPointers)]),
            // Tier 2 also holds textures and samplers by handle: the compile layer's answer
            // (`argument_buffers_tier`), from device addresses and descriptor indexing.
            F::ArgumentBuffersTier2 => first([tier2(f)]),
            // Indirect command buffers. The object, its compute commands encoded on the CPU or by a
            // kernel (barriers, resets and threadgroup lengths among them), their executions,
            // direct and indirect, and the blit encoder's copy, reset and optimize are built; a
            // kernel writes commands through device addresses (`lower::access::encode`), which a
            // device without them refuses (`GpuEncodedIndirectCommands`). Render commands are built
            // too, mesh draws among them, but for what `IndirectRenderCommands` names: an
            // emulated mesh draw whose compute work touches what the pass's earlier draws do, an
            // execution while a visibility result mode is set or of a range an earlier draw may
            // write, commands that need an instance of their own, and a depth/stencil state's
            // ID.
            F::IcbCompute | F::IcbComputeMemoryBarriers => {
                first([addressed(f, Feature::GpuEncodedIndirectCommands)])
            }
            F::IcbRendering | F::IcbRenderMemoryBarriers | F::IcbRasterAndDepthStencilState => {
                first([
                    addressed(f, Feature::GpuEncodedIndirectCommands),
                    gap(G::IndirectRenderCommands),
                ])
            }
            F::IcbTessellation => first([
                addressed(f, Feature::GpuEncodedIndirectCommands),
                gap(G::IndirectRenderCommands),
            ]),
            // Imageblocks. A kernel's run (`set_imageblock_size`, step 9's first piece), and a tile
            // function's in a tile dispatch, whose refusals are the tile shaders row's. A fragment
            // function's needs the ordered pixel interlock, and fragment stores to write it
            // (`lower::fragment::imageblock`), and draws through the runtime's render encoder,
            // sharing the pass's storage with its tile functions' explicit imageblocks. The runtime
            // refuses a pass whose draws view imageblocks of different masters or sizes
            // (`ImageblockMasters`), and one with a tile function's explicit imageblock aliased
            // onto the attachments beside them (`TileAndFragmentImageblocks`). A multisampled
            // pass's function reaches each sample's block, once per sample, and a write the tests
            // may drop runs the tests first, early or in the shader (`TestedStorageWrites`); the
            // compile layer and the binder refuse, as not implemented, a writing function under
            // alpha to coverage, and where its tests act in a counted draw, over a depth format
            // other than 32-bit float or a stencil attachment alone; and, on MoltenVK at more
            // than one sample, whose early tests drop image writes
            // (`EarlyTestsDropMultisampledImageWrites`), one declaring early tests that discards,
            // returns depth, stencil or a mask, or writes other memory.
            F::Imageblocks => first([
                needs_features(
                    f,
                    Feature::Imageblocks,
                    &[
                        VkFeature::FragmentShaderPixelInterlock,
                        VkFeature::FragmentStoresAndAtomics,
                    ],
                ),
                gap(G::ImageblockMasters),
                not_implemented(Feature::Imageblocks),
            ]),
            // Tile dispatches run in the render encoder: a tile function over the pass's tiles, in
            // a single-sampled pass of one layer, in threadgroups of even sides to 32x32 (and the
            // single thread), each thread at Metal's place. The compile layer and the binder
            // refuse the rest of what Metal takes (`NotImplemented(TileShading)`): a multisampled
            // pass's tile function, and so its sample coverage, a threadgroup with an odd side,
            // whose quads leave lanes empty, and a layered pass. Threadgroup memory that persists
            // through the pass is built.
            F::TileShaders | F::ImageblockSampleCoverageControl | F::NonsquareTileDispatch => {
                first([not_implemented(Feature::TileShading)])
            }
            // Mesh pipelines (`new_mesh_render_pipeline_state`, the encoder's mesh draws): the
            // stages ([`mesh_stages`]). Indirect mesh draws read the counts an arguments pass
            // writes (`render/mesh_arguments.rs`); an indirect command buffer's mesh draws are
            // render commands first.
            F::MeshShading | F::IndirectMeshDrawArguments => first([mesh_stages(f)]),
            F::IcbMeshDraws => first([mesh_stages(f), gap(G::IndirectRenderCommands)]),
            // Ray tracing: ray queries and acceleration structures on device addresses, then the
            // runtime's acceleration structures.
            F::RayTracingCompute
            | F::RayTracingRender
            | F::RowMajorAccelerationStructures
            | F::PerComponentMotionInterpolation
            | F::IntersectionResultStorage
            | F::AccelerationStructureBuildOptions
            | F::IntersectionFunctionBuffers => {
                first([ray_tracing(f), gap(G::AccelerationStructures)])
            }
            // Function pointers in kernels: `new_visible_function_table`, `function_handle` and
            // `set_visible_function_table` (step 15's tables). The compile layer calls through a
            // table's entry by a switch over the pipeline's linked functions. A table an argument
            // buffer holds is read through the stage's function-table directory and the set-1 table
            // array, which needs the four features below (`lower::access::function_table`).
            // Recursion through tables runs on a call stack of the pipeline's `maxCallStackDepth`,
            // and so does recursion through a function pointer passed as a value. A pointer chosen
            // at run time among functions' addresses, not a table's entry alone, is a word, and a
            // call through it a switch over the functions it can hold: selected, merged, carried
            // around a loop, kept in thread or threadgroup memory, compared and converted to an
            // integer and back, as Metal runs them.
            F::FunctionPointersCompute => first([function_tables(f)]),
            // Function pointers in render stages: the render encoder's tables
            // (`set_vertex_visible_function_table`, `set_fragment_visible_function_table`,
            // `render/arguments.rs`) are built, and the compile layer's needs above hold there
            // too.
            F::FunctionPointersRender => first([function_tables(f)]),
            // `BinaryArchive`: compute and render adds, files, and failing on a miss. A tile add
            // comes with tile pipelines, whose rows block their families. The vertex
            // stage's archive key differs from Metal's where two adds' stages combine, which blocks
            // no family (orchestrator, 2026-10-06): a creation failing on a miss is a cache
            // contract on Metal too, which misses whenever an OS or driver update invalidates an
            // archive, so a program using it already handles the refusal; and a hit where Metal
            // misses makes only a pipeline Metal makes without the option. Neither leaves a program
            // Metal runs unrunnable.
            F::BinaryArchives => RowState::Exact,
            // Dynamic libraries: made from a library or a file, serialized, preloaded by compute,
            // tile and each render stage, and loaded by install name at creation; the compile layer
            // links a stage's imports to their exports, which needs nothing of the device.
            F::RenderDynamicLibraries | F::ComputeDynamicLibraries => RowState::Exact,

            // ---- Resources and heaps ----
            // Heaps and their textures: a texture made in a heap after a command reading texture
            // handles was encoded is reached through the device's texture table. Its
            // near-equivalents are announced.
            F::ResourceHeaps | F::PlacementHeap => RowState::Exact,
            // Sparse textures: a sparse texture of the row's format as `Heap::new_texture` makes it
            // (the format's Sparse answer, the device's support), then what every sparse texture
            // still lacks (a view of a format whose tile is not its texture's, a cube or
            // multisampled texture's tile map), then access counters, which are not scheduled.
            F::SparseTextures => first([
                formats(f, &[FormatRow::RGBA8Unorm], FormatCapability::Sparse),
                gap(G::SparseHeaps),
                gap(G::TextureAccessCounters),
            ]),
            F::SparseDepthStencilTextures => first([
                formats(f, &[FormatRow::Depth32Float], FormatCapability::Sparse),
                gap(G::SparseHeaps),
                gap(G::TextureAccessCounters),
            ]),

            // ---- Pieces not scheduled ----
            F::LossyTextureCompression => first([gap(G::LossyTextureCompression)]),
            F::FastResourceLoading => first([gap(G::IoCommandQueues)]),
            F::ShaderValidation => first([gap(G::ShaderValidation)]),
            F::ShaderLogging => first([gap(G::ShaderLogging)]),
            F::ResidencySets => first([gap(G::ResidencySets)]),
            F::TextureViewPools => first([gap(G::TextureViewPools)]),
            F::ColorAttachmentMapping => first([gap(G::ColorAttachmentMapping)]),
            F::DeviceNotifications => first([gap(G::DeviceNotifications)]),

            // ---- Metal 4's programming model only: the `MTL4` API, no family's row here ----
            F::ArgumentTables
            | F::CommandAllocators
            | F::DecoupledQueuesAndCommandBuffers
            | F::CommandBarriers
            | F::PlacementSparseBuffers
            | F::PlacementSparseTextures
            | F::CompilationContexts
            | F::PipelineDatasetSerialization
            | F::FlexibleRenderPipelineState
            | F::MachineLearningEncoding
            | F::Tensors
            | F::PerformanceCounterHeaps
            | F::AddressDrivenAccelerationStructureBuilds => first([gap(G::Metal4Api)]),
        }
    }
}

/// The state of a row whose checks are `checks`: the first that blocks, in their order, or
/// exact.
pub(super) fn first<const N: usize>(checks: [Check; N]) -> RowState {
    checks
        .into_iter()
        .flatten()
        .next()
        .unwrap_or(RowState::Exact)
}

/// A refusal's state.
pub(super) fn refused(unsupported: Unsupported) -> RowState {
    RowState::Refused(Refusal::Unsupported(unsupported))
}

/// The check of a runtime piece not built yet.
pub(super) fn gap(gap: RuntimeGap) -> Check {
    Some(RowState::Gap(gap))
}

/// The check of what the compile layer implements on no device.
pub(super) fn not_implemented(feature: Feature) -> Check {
    Some(refused(Unsupported::not_implemented(feature)))
}

/// The check of `capability` of every format of `rows`.
fn formats(f: &Facts<'_>, rows: &[FormatRow], capability: FormatCapability) -> Check {
    rows.iter()
        .flat_map(|row| row.formats())
        .find_map(|&pf| (f.format)(pf).get(capability).err())
        .map(Missing::state)
}

/// The check of `answer` of every format of the rows of `rows` that `listed` keeps.
fn listed_formats(
    rows: &[FormatRow],
    listed: impl Fn(FormatRow) -> bool,
    answer: impl Fn(PixelFormat) -> Result<(), Missing>,
) -> Check {
    rows.iter()
        .filter(|&&row| listed(row))
        .flat_map(|row| row.formats())
        .find_map(|&pf| answer(pf).err())
        .map(Missing::state)
}

/// `feature`, refused where the device was created without any of `features`.
pub(super) fn needs_features(f: &Facts<'_>, feature: Feature, features: &[VkFeature]) -> Check {
    let missing: Vec<Capability> = features
        .iter()
        .filter(|&&x| !f.properties.description.has(x))
        .map(|&x| Capability::Feature(x))
        .collect();
    (!missing.is_empty()).then(|| refused(Unsupported::needs_any_of(feature, vec![missing])))
}

/// A mesh pipeline's object and mesh stages: `meshShader` and `taskShader` where the device runs
/// them natively; nothing where it runs them as compute work
/// (`DeviceProfile::emulates_mesh_shading`, which every device can).
fn mesh_stages(f: &Facts<'_>) -> Check {
    if f.profile.emulates_mesh_shading() {
        return None;
    }
    needs_features(
        f,
        Feature::MeshShading,
        &[VkFeature::MeshShader, VkFeature::TaskShader],
    )
}

/// `feature`, refused where the device was created without every feature of each of
/// `alternatives`.
pub(super) fn needs_any(f: &Facts<'_>, feature: Feature, alternatives: &[&[VkFeature]]) -> Check {
    let has = |x: &VkFeature| f.properties.description.has(*x);
    if alternatives.iter().any(|a| a.iter().all(has)) {
        return None;
    }
    let needs = alternatives
        .iter()
        .map(|a| a.iter().map(|&x| Capability::Feature(x)).collect())
        .collect();
    Some(refused(Unsupported::needs_any_of(feature, needs)))
}

/// `feature`, refused where the device's `VK_KHR_portability_subset` withholds `withheld`.
fn withheld(f: &Facts<'_>, feature: Feature, withheld: VkFeature) -> Check {
    f.profile
        .enabled()
        .portability_withholds(withheld)
        .then(|| refused(Unsupported::needs(feature, Capability::Feature(withheld))))
}

/// The compile layer's refusal of `feature` on every pipeline of the device, where it lists
/// one (`MetalDeviceProperties::unsupported_features`).
pub(super) fn listed(f: &Facts<'_>, feature: Feature) -> Check {
    f.metal
        .unsupported_features
        .iter()
        .find(|u| u.feature == feature)
        .map(|u| refused(u.clone()))
}

/// `feature`, refused where the device does not reach buffers by device address.
pub(super) fn addressed(f: &Facts<'_>, feature: Feature) -> Check {
    let needs = f.profile.addressed_needs();
    (!needs.is_empty()).then(|| refused(Unsupported::needs_any_of(feature, vec![needs])))
}

/// The runtime's query `answer`: exact where it answers yes, refused naming `query` where it
/// answers no, and the gap it waits on while its runtime piece is not built.
pub(super) fn query(answer: Result<bool, RuntimeGap>, query: Query) -> Check {
    match answer {
        Ok(true) => None,
        Ok(false) => Some(RowState::Refused(Refusal::Query(query))),
        Err(gap) => Some(RowState::Gap(gap)),
    }
}

/// Writes from vertex and fragment functions: `vertexPipelineStoresAndAtomics` and
/// `fragmentStoresAndAtomics`.
fn stage_stores(f: &Facts<'_>) -> Check {
    needs_features(
        f,
        Feature::StageStoresAndAtomics,
        &[
            VkFeature::VertexPipelineStoresAndAtomics,
            VkFeature::FragmentStoresAndAtomics,
        ],
    )
}

/// Read-write texture tier 1: the compile layer's answer
/// (`MetalDeviceProperties::read_write_texture_tier`), refused naming what the first tier-1
/// format's storage form lacks.
fn read_write_tier(f: &Facts<'_>) -> Check {
    if f.metal.read_write_texture_tier != ReadWriteTextureTier::None {
        return None;
    }
    [
        PixelFormat::R32Float,
        PixelFormat::R32Uint,
        PixelFormat::R32Sint,
    ]
    .into_iter()
    .find_map(|pf| storage_image_choice(pf, f.profile).err())
    .map(|error| match error {
        metal2vulkan::error::Error::Unsupported(unsupported) => refused(unsupported),
        _ => refused(Unsupported::not_representable(Feature::Format)),
    })
}

/// Argument buffers tier 2: the compile layer's answer
/// (`MetalDeviceProperties::argument_buffers_tier`), refused naming the device addresses
/// and the descriptor indexing it lacks.
fn tier2(f: &Facts<'_>) -> Check {
    if f.metal.argument_buffers_tier == ArgumentBuffersTier::Tier2 {
        return None;
    }
    let mut needs = f.profile.addressed_needs();
    needs.extend(
        [
            VkFeature::RuntimeDescriptorArray,
            VkFeature::DescriptorBindingPartiallyBound,
            VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
            VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
        ]
        .into_iter()
        .filter(|&x| !f.properties.description.has(x))
        .map(Capability::Feature),
    );
    Some(refused(Unsupported::needs_any_of(
        Feature::ArgumentBufferTextures,
        vec![needs],
    )))
}

/// A visible-function table an argument buffer holds, read by its resource id through the set-1
/// table array: the features the compile layer refuses such a read without, in its order
/// (`lower::access::function_table`).
fn function_tables(f: &Facts<'_>) -> Check {
    needs_features(
        f,
        Feature::DynamicCall,
        &[
            VkFeature::RuntimeDescriptorArray,
            VkFeature::DescriptorBindingPartiallyBound,
            VkFeature::ShaderStorageBufferArrayNonUniformIndexing,
            VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
        ],
    )
}

/// Ray queries and acceleration structures, on device addresses (`rayQuery`,
/// `accelerationStructure`, `bufferDeviceAddress`).
pub(super) fn ray_tracing(f: &Facts<'_>) -> Check {
    needs_features(
        f,
        Feature::RayTracing,
        &[
            VkFeature::RayQuery,
            VkFeature::AccelerationStructure,
            VkFeature::BufferDeviceAddress,
        ],
    )
}

/// Simdgroup or quad operations of classes `ops` in vertex and fragment functions, refused as
/// `feature`: the compile layer's per-stage answer (`MetalDeviceProperties::simd`). A stage that
/// has no subgroup operations at all is refused as the compile layer lists it; one that lacks
/// some is refused naming the classes it lacks (`subgroupSupportedOperations`), or the stage
/// where the device has them but runs the stage unlisted, measured for other classes only.
fn simd(f: &Facts<'_>, ops: OpBits, feature: Feature) -> Check {
    let subgroup = &f.properties.description.subgroup;
    let stages = [
        (Stage::Vertex, f.metal.simd.vertex, ShaderStageFlags::VERTEX),
        (
            Stage::Fragment,
            f.metal.simd.fragment,
            ShaderStageFlags::FRAGMENT,
        ),
    ];
    for (stage, simd, flag) in stages {
        let StageSimd { cross_lane, .. } = simd;
        if cross_lane.contains(ops) {
            continue;
        }
        if !cross_lane.contains(OpBits::BASIC) {
            let listed =
                f.metal.unsupported_features.iter().find(|u| {
                    matches!(u.feature, Feature::SubgroupOperations | Feature::SimdWidth)
                });
            if let Some(u) = listed {
                return Some(refused(u.clone()));
            }
            if !subgroup.supported_stages.contains(flag) {
                return Some(refused(Unsupported::needs(
                    Feature::SubgroupOperations,
                    Capability::Property(VkProperty::SubgroupStage(stage)),
                )));
            }
        }
        // A class the device has but the stage lacks is one measured nowhere in a stage the
        // device does not list (`DriverDeviation::MetalVertexSimdgroups`), which listing it would
        // give, as the compile layer names it.
        let device = OpBits(subgroup.supported_operations.as_raw());
        let mut missing: Vec<Capability> = Vec::new();
        for op in SUBGROUP_OPS {
            let bit = OpBits::of(op);
            if !ops.contains(bit) || cross_lane.contains(bit) {
                continue;
            }
            let need = if device.contains(bit) && !subgroup.supported_stages.contains(flag) {
                Capability::Property(VkProperty::SubgroupStage(stage))
            } else {
                Capability::Property(VkProperty::SubgroupOperation(op))
            };
            if !missing.contains(&need) {
                missing.push(need);
            }
        }
        return Some(refused(Unsupported::needs_any_of(feature, vec![missing])));
    }
    None
}

/// Vulkan's subgroup operation classes.
const SUBGROUP_OPS: [SubgroupOp; 8] = [
    SubgroupOp::Basic,
    SubgroupOp::Vote,
    SubgroupOp::Arithmetic,
    SubgroupOp::Ballot,
    SubgroupOp::Shuffle,
    SubgroupOp::ShuffleRelative,
    SubgroupOp::Clustered,
    SubgroupOp::Quad,
];
