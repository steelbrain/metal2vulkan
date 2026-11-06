//! The limit rows of Apple's tables, and how each is decided on a device.
//!
//! A cell is a [`Need`]: at least a value, no limit, an alignment the runtime takes, or a tile
//! size. [`LimitRow::state`] has one arm per row, whose comment names what provides it and whose
//! code reads it from the device's description, the compile layer's Metal answers, or one of the
//! runtime's queries (the feature rows' order of blockers, `features`).
//!
//! A limit of a feature (an imageblock's memory, a tessellation factor) is listed only where the
//! family lists the feature's row. Metal3 takes the Apple7 column's limits, and its Mac devices
//! have neither imageblocks nor tile shaders.

use super::features::{
    addressed, first, gap, listed, needs_any, needs_features, query, ray_tracing, refused, Check,
};
use super::{Column, Facts, FeatureRow, Refusal, RowState};
use crate::capabilities::Query;
use crate::error::RuntimeGap;
use metal2vulkan::device::{VkFeature, VkLimit, VkProperty};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::vk::SampleCountFlags;

/// What a cell asks of a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Need {
    /// At least this value: a count, or a size in bytes or texels.
    AtLeast(u64),
    /// No limit: as many as the device's memory holds.
    Unbounded,
    /// Every value aligned to this many bytes is taken.
    AlignedTo(u64),
    /// A tile of this width and height in pixels.
    Tile(u32, u32),
}

/// A limit row's cells ([`LimitRow::cells`]).
struct Cells {
    /// From each Apple family on, until the next run, the column's need (`None`: not listed).
    apple: &'static [(u8, Option<Need>)],
    mac2: Option<Need>,
}

impl Cells {
    /// The Apple`n` column's need.
    fn apple(&self, n: u8) -> Option<Need> {
        self.apple
            .iter()
            .rev()
            .find(|&&(first, _)| first <= n)
            .and_then(|&(_, need)| need)
    }
}

/// A limit entry of Apple's tables.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LimitRow {
    /// "Maximum number of vertex attributes, per vertex descriptor".
    VertexAttributes,
    /// "Maximum number of entries in the buffer argument table, per graphics or kernel function".
    BufferArgumentEntries,
    /// "Maximum number of entries in the texture argument table, per graphics or kernel function".
    TextureArgumentEntries,
    /// "Maximum number of entries in the sampler state argument table, per graphics or kernel
    /// functions" (2023: "Maximum number of entries in the sampler state argument table, per
    /// graphics or kernel function").
    SamplerArgumentEntries,
    /// "Maximum number of entries in the threadgroup memory argument table, per kernel function".
    ThreadgroupMemoryArgumentEntries,
    /// "Maximum number of constant buffer arguments in vertex, fragment, tile, or kernel functions"
    /// (2023: "Maximum number of constant buffer arguments in vertex, fragment, tile, or kernel
    /// function").
    ConstantBufferArguments,
    /// "Maximum length of inlined buffer contents using setBytes" (2023: "Maximum length of
    /// constant buffer arguments in vertex, fragment, tile, or kernel function").
    InlineBytesLength,
    /// "Maximum threads per threadgroup".
    ThreadsPerThreadgroup,
    /// "Maximum total threadgroup memory allocation".
    ThreadgroupMemory,
    /// "Maximum explicit image block allocation".
    ExplicitImageblockMemory,
    /// "Maximum implicit image block allocation".
    ImplicitImageblockMemory,
    /// "Maximum total tile memory allocation".
    TileMemory,
    /// "Threadgroup memory length alignment".
    ThreadgroupMemoryAlignment,
    /// "Maximum function memory allocation for a buffer in the constant address space".
    ConstantAddressSpaceBuffer,
    /// "Maximum scalar or vector inputs to a fragment function. (Declare with the `[[stage_in]]`
    /// qualifier.)" (2023: "Maximum scalars or vectors inputs to a fragment function. (Declare with
    /// the `[[stage_in]]` qualifier.)").
    FragmentInputs,
    /// "Maximum number of input components to a fragment function. (Declare with the `[[stage_in]]`
    /// qualifier.)".
    FragmentInputComponents,
    /// "Maximum number of function constants".
    FunctionConstants,
    /// "Maximum tessellation factor".
    TessellationFactor,
    /// "Maximum number of viewports and scissor rectangles, per vertex function".
    ViewportsAndScissors,
    /// "Maximum number of raster order groups, per fragment function".
    RasterOrderGroupsPerFunction,
    /// "Minimum alignment of buffer layout descriptor stride and attribute descriptor offset"
    /// (2023: "Minimum alignment of buffer layout descriptor stride").
    VertexBufferStrideAlignment,
    /// "Maximum size of buffer layout descriptor stride".
    VertexBufferStride,
    /// "Maximum number of buffers you can access, per stage, from an argument buffer".
    ArgumentBufferBuffers,
    /// "Maximum number of textures you can access, per stage, from an argument buffer".
    ArgumentBufferTextures,
    /// "Maximum number of samplers you can access, per stage, from an argument buffer".
    ArgumentBufferSamplers,
    /// "Minimum constant buffer offset alignment".
    ConstantBufferOffsetAlignment,
    /// "Maximum 1D texture width".
    TextureWidth1D,
    /// "Maximum 2D texture width and height".
    TextureSize2D,
    /// "Maximum cube map texture width and height".
    CubeTextureSize,
    /// "Maximum 3D texture width, height, and depth".
    TextureSize3D,
    /// "Maximum texture buffer width".
    TextureBufferWidth,
    /// "Maximum number of layers per 1D texture array, 2D texture array, or 3D texture array"
    /// (2023: "Maximum number of layers per 1D texture array, 2D texture array, or 3D texture").
    TextureArrayLayers,
    /// "Buffer alignment for copying an existing texture to a buffer".
    TextureToBufferCopyAlignment,
    /// "Maximum counter sample buffer length".
    CounterSampleBufferLength,
    /// "Maximum number of sample buffers".
    CounterSampleBuffers,
    /// "Maximum number of residency sets per queue".
    ResidencySetsPerQueue,
    /// "Maximum number of residency sets per buffer".
    ResidencySetsPerBuffer,
    /// "Maximum indirect command buffer length".
    IndirectCommandBufferLength,
    /// "Maximum number of color render targets per render pass descriptor".
    ColorRenderTargets,
    /// "Maximum size of a point primitive".
    PointSize,
    /// "Maximum explicit image block size, per pixel, per sample, when using multiple color render
    /// targets".
    ExplicitImageblockSizePerSample,
    /// "Maximum implicit image block size, per pixel, per sample, when using multiple color render
    /// targets".
    ImplicitImageblockSizePerSample,
    /// "Maximum total render target size, per pixel, when using multiple color render targets".
    RenderTargetSizePerPixel,
    /// "Maximum visibility query offset".
    VisibilityQueryOffset,
    /// "Maximum sample count in render passes with MSAA".
    RenderPassSampleCount,
    /// "Maximum tile size in render passes without MSAA".
    TileSizeWithoutMsaa,
    /// "Maximum tile size in render passes with 2x MSAA".
    TileSize2xMsaa,
    /// "Maximum tile size in render passes with 4x MSAA".
    TileSize4xMsaa,
    /// "Maximum tile size in render passes with 8x MSAA".
    TileSize8xMsaa,
    /// "Maximum number of fences".
    Fences,
    /// "Maximum number of I/O commands per buffer".
    IoCommandsPerBuffer,
    /// "Maximum vertex count for vertex amplification".
    VertexAmplificationCount,
    /// "Maximum threadgroups per object shader grid".
    ObjectThreadgroupsPerGrid,
    /// "Maximum threadgroups per mesh shader grid".
    MeshThreadgroupsPerGrid,
    /// "Maximum payload in mesh shader pipeline".
    MeshPayload,
    /// "Largest number of levels a ray-tracing intersector can traverse in an acceleration
    /// structure".
    IntersectorLevels,
    /// "Largest number of levels a ray-tracing intersection query can traverse in an acceleration
    /// structure".
    IntersectionQueryLevels,
    /// "Maximum texture view pool entries".
    TextureViewPoolEntries,
    /// "Maximum performance counter heaps (per process)".
    PerformanceCounterHeaps,
    /// "Minimum alignment of intersection function buffer".
    IntersectionFunctionBufferAlignment,
    /// "Minimum alignment of intersection function buffer stride".
    IntersectionFunctionBufferStrideAlignment,
    /// "Maximum size of intersection function buffer stride".
    IntersectionFunctionBufferStride,
}

impl LimitRow {
    /// Every row: the 2026 edition's, in its order, with the 2023 edition's others where it
    /// prints them.
    pub const ALL: [LimitRow; 62] = [
        LimitRow::VertexAttributes,
        LimitRow::BufferArgumentEntries,
        LimitRow::TextureArgumentEntries,
        LimitRow::SamplerArgumentEntries,
        LimitRow::ThreadgroupMemoryArgumentEntries,
        LimitRow::ConstantBufferArguments,
        LimitRow::InlineBytesLength,
        LimitRow::ThreadsPerThreadgroup,
        LimitRow::ThreadgroupMemory,
        LimitRow::ExplicitImageblockMemory,
        LimitRow::ImplicitImageblockMemory,
        LimitRow::TileMemory,
        LimitRow::ThreadgroupMemoryAlignment,
        LimitRow::ConstantAddressSpaceBuffer,
        LimitRow::FragmentInputs,
        LimitRow::FragmentInputComponents,
        LimitRow::FunctionConstants,
        LimitRow::TessellationFactor,
        LimitRow::ViewportsAndScissors,
        LimitRow::RasterOrderGroupsPerFunction,
        LimitRow::VertexBufferStrideAlignment,
        LimitRow::VertexBufferStride,
        LimitRow::ArgumentBufferBuffers,
        LimitRow::ArgumentBufferTextures,
        LimitRow::ArgumentBufferSamplers,
        LimitRow::ConstantBufferOffsetAlignment,
        LimitRow::TextureWidth1D,
        LimitRow::TextureSize2D,
        LimitRow::CubeTextureSize,
        LimitRow::TextureSize3D,
        LimitRow::TextureBufferWidth,
        LimitRow::TextureArrayLayers,
        LimitRow::TextureToBufferCopyAlignment,
        LimitRow::CounterSampleBufferLength,
        LimitRow::CounterSampleBuffers,
        LimitRow::ResidencySetsPerQueue,
        LimitRow::ResidencySetsPerBuffer,
        LimitRow::IndirectCommandBufferLength,
        LimitRow::ColorRenderTargets,
        LimitRow::PointSize,
        LimitRow::ExplicitImageblockSizePerSample,
        LimitRow::ImplicitImageblockSizePerSample,
        LimitRow::RenderTargetSizePerPixel,
        LimitRow::VisibilityQueryOffset,
        LimitRow::RenderPassSampleCount,
        LimitRow::TileSizeWithoutMsaa,
        LimitRow::TileSize2xMsaa,
        LimitRow::TileSize4xMsaa,
        LimitRow::TileSize8xMsaa,
        LimitRow::Fences,
        LimitRow::IoCommandsPerBuffer,
        LimitRow::VertexAmplificationCount,
        LimitRow::ObjectThreadgroupsPerGrid,
        LimitRow::MeshThreadgroupsPerGrid,
        LimitRow::MeshPayload,
        LimitRow::IntersectorLevels,
        LimitRow::IntersectionQueryLevels,
        LimitRow::TextureViewPoolEntries,
        LimitRow::PerformanceCounterHeaps,
        LimitRow::IntersectionFunctionBufferAlignment,
        LimitRow::IntersectionFunctionBufferStrideAlignment,
        LimitRow::IntersectionFunctionBufferStride,
    ];

    /// The row's name as Apple's tables print it (the 2026 edition's, where it has the row).
    pub fn label(self) -> &'static str {
        match self {
            LimitRow::VertexAttributes => "Maximum number of vertex attributes, per vertex descriptor",
            LimitRow::BufferArgumentEntries => "Maximum number of entries in the buffer argument table, per graphics or kernel function",
            LimitRow::TextureArgumentEntries => "Maximum number of entries in the texture argument table, per graphics or kernel function",
            LimitRow::SamplerArgumentEntries => "Maximum number of entries in the sampler state argument table, per graphics or kernel functions",
            LimitRow::ThreadgroupMemoryArgumentEntries => "Maximum number of entries in the threadgroup memory argument table, per kernel function",
            LimitRow::ConstantBufferArguments => "Maximum number of constant buffer arguments in vertex, fragment, tile, or kernel functions",
            LimitRow::InlineBytesLength => "Maximum length of inlined buffer contents using setBytes",
            LimitRow::ThreadsPerThreadgroup => "Maximum threads per threadgroup",
            LimitRow::ThreadgroupMemory => "Maximum total threadgroup memory allocation",
            LimitRow::ExplicitImageblockMemory => "Maximum explicit image block allocation",
            LimitRow::ImplicitImageblockMemory => "Maximum implicit image block allocation",
            LimitRow::TileMemory => "Maximum total tile memory allocation",
            LimitRow::ThreadgroupMemoryAlignment => "Threadgroup memory length alignment",
            LimitRow::ConstantAddressSpaceBuffer => "Maximum function memory allocation for a buffer in the constant address space",
            LimitRow::FragmentInputs => "Maximum scalar or vector inputs to a fragment function. (Declare with the [[stage_in]] qualifier.)",
            LimitRow::FragmentInputComponents => "Maximum number of input components to a fragment function. (Declare with the [[stage_in]] qualifier.)",
            LimitRow::FunctionConstants => "Maximum number of function constants",
            LimitRow::TessellationFactor => "Maximum tessellation factor",
            LimitRow::ViewportsAndScissors => "Maximum number of viewports and scissor rectangles, per vertex function",
            LimitRow::RasterOrderGroupsPerFunction => "Maximum number of raster order groups, per fragment function",
            LimitRow::VertexBufferStrideAlignment => "Minimum alignment of buffer layout descriptor stride and attribute descriptor offset",
            LimitRow::VertexBufferStride => "Maximum size of buffer layout descriptor stride",
            LimitRow::ArgumentBufferBuffers => "Maximum number of buffers you can access, per stage, from an argument buffer",
            LimitRow::ArgumentBufferTextures => "Maximum number of textures you can access, per stage, from an argument buffer",
            LimitRow::ArgumentBufferSamplers => "Maximum number of samplers you can access, per stage, from an argument buffer",
            LimitRow::ConstantBufferOffsetAlignment => "Minimum constant buffer offset alignment",
            LimitRow::TextureWidth1D => "Maximum 1D texture width",
            LimitRow::TextureSize2D => "Maximum 2D texture width and height",
            LimitRow::CubeTextureSize => "Maximum cube map texture width and height",
            LimitRow::TextureSize3D => "Maximum 3D texture width, height, and depth",
            LimitRow::TextureBufferWidth => "Maximum texture buffer width",
            LimitRow::TextureArrayLayers => "Maximum number of layers per 1D texture array, 2D texture array, or 3D texture array",
            LimitRow::TextureToBufferCopyAlignment => "Buffer alignment for copying an existing texture to a buffer",
            LimitRow::CounterSampleBufferLength => "Maximum counter sample buffer length",
            LimitRow::CounterSampleBuffers => "Maximum number of sample buffers",
            LimitRow::ResidencySetsPerQueue => "Maximum number of residency sets per queue",
            LimitRow::ResidencySetsPerBuffer => "Maximum number of residency sets per buffer",
            LimitRow::IndirectCommandBufferLength => "Maximum indirect command buffer length",
            LimitRow::ColorRenderTargets => "Maximum number of color render targets per render pass descriptor",
            LimitRow::PointSize => "Maximum size of a point primitive",
            LimitRow::ExplicitImageblockSizePerSample => "Maximum explicit image block size, per pixel, per sample, when using multiple color render targets",
            LimitRow::ImplicitImageblockSizePerSample => "Maximum implicit image block size, per pixel, per sample, when using multiple color render targets",
            LimitRow::RenderTargetSizePerPixel => "Maximum total render target size, per pixel, when using multiple color render targets",
            LimitRow::VisibilityQueryOffset => "Maximum visibility query offset",
            LimitRow::RenderPassSampleCount => "Maximum sample count in render passes with MSAA",
            LimitRow::TileSizeWithoutMsaa => "Maximum tile size in render passes without MSAA",
            LimitRow::TileSize2xMsaa => "Maximum tile size in render passes with 2x MSAA",
            LimitRow::TileSize4xMsaa => "Maximum tile size in render passes with 4x MSAA",
            LimitRow::TileSize8xMsaa => "Maximum tile size in render passes with 8x MSAA",
            LimitRow::Fences => "Maximum number of fences",
            LimitRow::IoCommandsPerBuffer => "Maximum number of I/O commands per buffer",
            LimitRow::VertexAmplificationCount => "Maximum vertex count for vertex amplification",
            LimitRow::ObjectThreadgroupsPerGrid => "Maximum threadgroups per object shader grid",
            LimitRow::MeshThreadgroupsPerGrid => "Maximum threadgroups per mesh shader grid",
            LimitRow::MeshPayload => "Maximum payload in mesh shader pipeline",
            LimitRow::IntersectorLevels => "Largest number of levels a ray-tracing intersector can traverse in an acceleration structure",
            LimitRow::IntersectionQueryLevels => "Largest number of levels a ray-tracing intersection query can traverse in an acceleration structure",
            LimitRow::TextureViewPoolEntries => "Maximum texture view pool entries",
            LimitRow::PerformanceCounterHeaps => "Maximum performance counter heaps (per process)",
            LimitRow::IntersectionFunctionBufferAlignment => "Minimum alignment of intersection function buffer",
            LimitRow::IntersectionFunctionBufferStrideAlignment => "Minimum alignment of intersection function buffer stride",
            LimitRow::IntersectionFunctionBufferStride => "Maximum size of intersection function buffer stride",
        }
    }

    /// The row's cells: the 2026 edition's Apple columns as runs of equal cells, each from the
    /// first Apple family it holds for, then the 2023 edition's Mac2 cell. Sizes are in bytes
    /// ("KB" is 1024 B), texel counts in texels ("M px" is 2^20), and other counts as printed
    /// ("K", "M" and "million" are 10^3 and 10^6).
    fn cells(self) -> Cells {
        use LimitRow as L;
        use Need::*;
        match self {
            L::VertexAttributes => Cells {
                apple: &[(2, Some(AtLeast(31)))],
                mac2: Some(AtLeast(31)),
            },
            L::BufferArgumentEntries => Cells {
                apple: &[(2, Some(AtLeast(31)))],
                mac2: Some(AtLeast(31)),
            },
            L::TextureArgumentEntries => Cells {
                apple: &[
                    (2, Some(AtLeast(31))),
                    (4, Some(AtLeast(96))),
                    (6, Some(AtLeast(128))),
                ],
                mac2: Some(AtLeast(128)),
            },
            L::SamplerArgumentEntries => Cells {
                apple: &[(2, Some(AtLeast(16)))],
                mac2: Some(AtLeast(16)),
            },
            L::ThreadgroupMemoryArgumentEntries => Cells {
                apple: &[(2, Some(AtLeast(31)))],
                mac2: Some(AtLeast(31)),
            },
            L::ConstantBufferArguments => Cells {
                apple: &[(2, Some(AtLeast(31)))],
                mac2: Some(AtLeast(14)),
            },
            L::InlineBytesLength => Cells {
                apple: &[(2, Some(AtLeast(4_096)))],
                mac2: Some(AtLeast(4_096)),
            },
            L::ThreadsPerThreadgroup => Cells {
                apple: &[(2, Some(AtLeast(512))), (4, Some(AtLeast(1_024)))],
                mac2: Some(AtLeast(1_024)),
            },
            L::ThreadgroupMemory => Cells {
                apple: &[
                    (2, Some(AtLeast(16_352))),
                    (3, Some(AtLeast(16_384))),
                    (4, Some(AtLeast(32_768))),
                ],
                mac2: Some(AtLeast(32_768)),
            },
            L::ExplicitImageblockMemory => Cells {
                apple: &[(4, Some(AtLeast(32_768)))],
                mac2: None,
            },
            L::ImplicitImageblockMemory => Cells {
                apple: &[(4, Some(AtLeast(131_072))), (9, Some(AtLeast(262_144)))],
                mac2: None,
            },
            L::TileMemory => Cells {
                apple: &[],
                mac2: None,
            },
            L::ThreadgroupMemoryAlignment => Cells {
                apple: &[(2, Some(AlignedTo(16)))],
                mac2: Some(AlignedTo(16)),
            },
            L::ConstantAddressSpaceBuffer => Cells {
                apple: &[],
                mac2: Some(Unbounded),
            },
            L::FragmentInputs => Cells {
                apple: &[(2, Some(AtLeast(60))), (4, Some(AtLeast(124)))],
                mac2: Some(AtLeast(32)),
            },
            L::FragmentInputComponents => Cells {
                apple: &[(2, Some(AtLeast(60))), (4, Some(AtLeast(124)))],
                mac2: Some(AtLeast(124)),
            },
            L::FunctionConstants => Cells {
                apple: &[(2, Some(AtLeast(65_536)))],
                mac2: Some(AtLeast(65_536)),
            },
            L::TessellationFactor => Cells {
                apple: &[(3, Some(AtLeast(16))), (5, Some(AtLeast(64)))],
                mac2: Some(AtLeast(64)),
            },
            L::ViewportsAndScissors => Cells {
                apple: &[(2, Some(AtLeast(1))), (5, Some(AtLeast(16)))],
                mac2: Some(AtLeast(16)),
            },
            L::RasterOrderGroupsPerFunction => Cells {
                apple: &[(4, Some(AtLeast(8)))],
                mac2: Some(AtLeast(8)),
            },
            L::VertexBufferStrideAlignment => Cells {
                apple: &[(2, Some(AlignedTo(4))), (5, Some(AlignedTo(1)))],
                mac2: Some(AlignedTo(4)),
            },
            L::VertexBufferStride => Cells {
                apple: &[(2, Some(Unbounded))],
                mac2: Some(AtLeast(4_096)),
            },
            L::ArgumentBufferBuffers => Cells {
                apple: &[
                    (2, Some(AtLeast(31))),
                    (4, Some(AtLeast(96))),
                    (6, Some(Unbounded)),
                ],
                mac2: Some(Unbounded),
            },
            L::ArgumentBufferTextures => Cells {
                apple: &[
                    (2, Some(AtLeast(31))),
                    (4, Some(AtLeast(96))),
                    (6, Some(AtLeast(1_000_000))),
                ],
                mac2: Some(AtLeast(1_000_000)),
            },
            L::ArgumentBufferSamplers => Cells {
                apple: &[
                    (2, Some(AtLeast(16))),
                    (6, Some(AtLeast(128))),
                    (7, Some(AtLeast(996))),
                    (9, Some(AtLeast(500_000))),
                ],
                mac2: Some(AtLeast(1_024)),
            },
            L::ConstantBufferOffsetAlignment => Cells {
                apple: &[(2, Some(AlignedTo(4)))],
                mac2: Some(AlignedTo(32)),
            },
            L::TextureWidth1D => Cells {
                apple: &[
                    (2, Some(AtLeast(8_192))),
                    (3, Some(AtLeast(16_384))),
                    (10, Some(AtLeast(32_768))),
                ],
                mac2: Some(AtLeast(16_384)),
            },
            L::TextureSize2D => Cells {
                apple: &[
                    (2, Some(AtLeast(8_192))),
                    (3, Some(AtLeast(16_384))),
                    (10, Some(AtLeast(32_768))),
                ],
                mac2: Some(AtLeast(16_384)),
            },
            L::CubeTextureSize => Cells {
                apple: &[
                    (2, Some(AtLeast(8_192))),
                    (3, Some(AtLeast(16_384))),
                    (10, Some(AtLeast(32_768))),
                ],
                mac2: Some(AtLeast(16_384)),
            },
            L::TextureSize3D => Cells {
                apple: &[(2, Some(AtLeast(2_048)))],
                mac2: Some(AtLeast(2_048)),
            },
            L::TextureBufferWidth => Cells {
                apple: &[
                    (2, Some(AtLeast(67_108_864))),
                    (3, Some(AtLeast(268_435_456))),
                ],
                mac2: Some(AtLeast(268_435_456)),
            },
            L::TextureArrayLayers => Cells {
                apple: &[(2, Some(AtLeast(2_048)))],
                mac2: Some(AtLeast(2_048)),
            },
            L::TextureToBufferCopyAlignment => Cells {
                apple: &[(2, Some(AlignedTo(64))), (3, Some(AlignedTo(16)))],
                mac2: Some(AlignedTo(256)),
            },
            L::CounterSampleBufferLength => Cells {
                apple: &[(2, Some(AtLeast(32_768)))],
                mac2: Some(Unbounded),
            },
            L::CounterSampleBuffers => Cells {
                apple: &[(2, Some(AtLeast(32)))],
                mac2: Some(Unbounded),
            },
            L::ResidencySetsPerQueue => Cells {
                apple: &[(6, Some(AtLeast(32)))],
                mac2: None,
            },
            L::ResidencySetsPerBuffer => Cells {
                apple: &[(6, Some(AtLeast(32)))],
                mac2: None,
            },
            L::IndirectCommandBufferLength => Cells {
                apple: &[(3, Some(Unbounded))],
                mac2: None,
            },
            L::ColorRenderTargets => Cells {
                apple: &[(2, Some(AtLeast(8)))],
                mac2: Some(AtLeast(8)),
            },
            L::PointSize => Cells {
                apple: &[(2, Some(AtLeast(511)))],
                mac2: Some(AtLeast(511)),
            },
            L::ExplicitImageblockSizePerSample => Cells {
                apple: &[(4, Some(AtLeast(64)))],
                mac2: None,
            },
            L::ImplicitImageblockSizePerSample => Cells {
                apple: &[
                    (2, Some(AtLeast(32))),
                    (4, Some(AtLeast(64))),
                    (7, Some(AtLeast(128))),
                ],
                mac2: None,
            },
            L::RenderTargetSizePerPixel => Cells {
                apple: &[],
                mac2: Some(Unbounded),
            },
            L::VisibilityQueryOffset => Cells {
                apple: &[(2, Some(AtLeast(65_528))), (7, Some(AtLeast(262_144)))],
                mac2: Some(AtLeast(262_144)),
            },
            L::RenderPassSampleCount => Cells {
                apple: &[(2, Some(AtLeast(4))), (10, Some(AtLeast(8)))],
                mac2: None,
            },
            L::TileSizeWithoutMsaa => Cells {
                apple: &[(2, Some(Tile(32, 32)))],
                mac2: None,
            },
            L::TileSize2xMsaa => Cells {
                apple: &[(2, Some(Tile(32, 32)))],
                mac2: None,
            },
            L::TileSize4xMsaa => Cells {
                apple: &[(2, Some(Tile(32, 16)))],
                mac2: None,
            },
            L::TileSize8xMsaa => Cells {
                apple: &[(10, Some(Tile(16, 16)))],
                mac2: None,
            },
            L::Fences => Cells {
                apple: &[(2, Some(AtLeast(32_768)))],
                mac2: Some(AtLeast(32_768)),
            },
            L::IoCommandsPerBuffer => Cells {
                apple: &[(2, Some(AtLeast(8_192)))],
                mac2: Some(AtLeast(8_192)),
            },
            L::VertexAmplificationCount => Cells {
                apple: &[(6, Some(AtLeast(2))), (7, Some(AtLeast(8)))],
                mac2: None,
            },
            L::ObjectThreadgroupsPerGrid => Cells {
                apple: &[(7, Some(Unbounded))],
                mac2: Some(AtLeast(1_024)),
            },
            L::MeshThreadgroupsPerGrid => Cells {
                apple: &[
                    (7, Some(AtLeast(1_024))),
                    (9, Some(AtLeast(1_048_575))),
                    (10, Some(AtLeast(4_194_303))),
                ],
                mac2: Some(AtLeast(1_024)),
            },
            L::MeshPayload => Cells {
                apple: &[(7, Some(AtLeast(16_384)))],
                mac2: Some(AtLeast(16_384)),
            },
            L::IntersectorLevels => Cells {
                apple: &[(6, Some(AtLeast(32)))],
                mac2: Some(AtLeast(32)),
            },
            L::IntersectionQueryLevels => Cells {
                apple: &[(6, Some(AtLeast(16)))],
                mac2: Some(AtLeast(16)),
            },
            L::TextureViewPoolEntries => Cells {
                apple: &[
                    (7, Some(AtLeast(128_000_000))),
                    (9, Some(AtLeast(256_000_000))),
                ],
                mac2: None,
            },
            L::PerformanceCounterHeaps => Cells {
                apple: &[(7, Some(AtLeast(32)))],
                mac2: None,
            },
            L::IntersectionFunctionBufferAlignment => Cells {
                apple: &[(7, Some(AlignedTo(64)))],
                mac2: None,
            },
            L::IntersectionFunctionBufferStrideAlignment => Cells {
                apple: &[(7, Some(AlignedTo(8)))],
                mac2: None,
            },
            L::IntersectionFunctionBufferStride => Cells {
                apple: &[(7, Some(AtLeast(4_096)))],
                mac2: None,
            },
        }
    }
}

impl LimitRow {
    /// The feature row the limit belongs to, where it has one.
    fn feature(self) -> Option<FeatureRow> {
        use FeatureRow as F;
        use LimitRow as L;
        Some(match self {
            L::ExplicitImageblockMemory
            | L::ImplicitImageblockMemory
            | L::TileMemory
            | L::ExplicitImageblockSizePerSample => F::Imageblocks,
            L::TileSizeWithoutMsaa | L::TileSize2xMsaa | L::TileSize4xMsaa | L::TileSize8xMsaa => {
                F::TileShaders
            }
            L::TessellationFactor => F::Tessellation,
            L::RasterOrderGroupsPerFunction => F::RasterOrderGroups,
            L::VertexAmplificationCount => F::VertexAmplification,
            L::ObjectThreadgroupsPerGrid | L::MeshThreadgroupsPerGrid | L::MeshPayload => {
                F::MeshShading
            }
            L::IntersectorLevels | L::IntersectionQueryLevels => F::RayTracingCompute,
            L::IntersectionFunctionBufferAlignment
            | L::IntersectionFunctionBufferStrideAlignment
            | L::IntersectionFunctionBufferStride => F::IntersectionFunctionBuffers,
            L::TextureViewPoolEntries => F::TextureViewPools,
            L::PerformanceCounterHeaps => F::PerformanceCounterHeaps,
            L::ResidencySetsPerQueue | L::ResidencySetsPerBuffer => F::ResidencySets,
            L::IndirectCommandBufferLength => F::IcbRendering,
            L::IoCommandsPerBuffer => F::FastResourceLoading,
            L::ArgumentBufferBuffers | L::ArgumentBufferTextures | L::ArgumentBufferSamplers => {
                F::ArgumentBuffersTier1
            }
            _ => return None,
        })
    }

    /// What the family read from `column` asks of the row, where it lists the row.
    pub(super) fn need(self, column: Column) -> Option<Need> {
        if self.feature().is_some_and(|row| !row.listed(column)) {
            return None;
        }
        let cells = self.cells();
        match column {
            Column::Apple(n) => cells.apple(n),
            Column::Metal3 => cells.apple(7),
            Column::Mac2 => cells.mac2,
            Column::Common(_) => None,
        }
    }

    /// The row's state on a device of `f`, for what its family's column asks (`need`).
    pub(super) fn state(self, need: Need, f: &Facts<'_>) -> RowState {
        use LimitRow as L;
        use RuntimeGap as G;
        let l = &f.properties.description.limits;
        let at_least = |feature: Feature, limit: VkLimit, available: u32| {
            at_least(need, feature, limit, u64::from(available))
        };
        match self {
            // A vertex descriptor's attributes: `maxVertexInputAttributes`.
            L::VertexAttributes => first([at_least(
                Feature::InterfaceComponents,
                VkLimit::MaxVertexInputAttributes,
                l.max_vertex_input_attributes,
            )]),
            // A function's buffers, constant ones among them: by device address where the device
            // has them, else storage buffer descriptors.
            L::BufferArgumentEntries | L::ConstantBufferArguments => {
                if f.profile.addressed_needs().is_empty() {
                    RowState::Exact
                } else {
                    first([at_least(
                        Feature::DescriptorCount,
                        VkLimit::MaxPerStageDescriptorStorageBuffers,
                        l.max_per_stage_descriptor_storage_buffers,
                    )])
                }
            }
            // A function's textures, read or written: sampled and storage image descriptors.
            L::TextureArgumentEntries => first([
                at_least(
                    Feature::DescriptorCount,
                    VkLimit::MaxPerStageDescriptorSampledImages,
                    l.max_per_stage_descriptor_sampled_images,
                ),
                at_least(
                    Feature::DescriptorCount,
                    VkLimit::MaxPerStageDescriptorStorageImages,
                    l.max_per_stage_descriptor_storage_images,
                ),
            ]),
            // A function's samplers, constexpr ones among them: sampler descriptors.
            L::SamplerArgumentEntries => first([at_least(
                Feature::DescriptorCount,
                VkLimit::MaxPerStageDescriptorSamplers,
                l.max_per_stage_descriptor_samplers,
            )]),
            // A kernel's threadgroup memory arguments are arenas of one workgroup array, of any
            // number; function constants and fences have no count of their own; a visibility
            // result is written at any offset of its buffer.
            L::ThreadgroupMemoryArgumentEntries
            | L::FunctionConstants
            | L::Fences
            | L::VisibilityQueryOffset => RowState::Exact,
            // `set_bytes` and its stage forms: a copy of the bytes in the command buffer's upload
            // arena, bound as a buffer descriptor (`argument_table.rs`).
            L::InlineBytesLength => RowState::Exact,
            // `maxTotalThreadsPerThreadgroup`: the compile layer's answer
            // (`maxComputeWorkGroupInvocations`, an Apple GPU's 1024 at most).
            L::ThreadsPerThreadgroup => first([at_least(
                Feature::LocalSize { axis: None },
                VkLimit::MaxComputeWorkGroupInvocations,
                f.metal.max_total_threads_per_threadgroup,
            )]),
            // `maxThreadgroupMemoryLength`: the compile layer's answer
            // (`maxComputeSharedMemorySize`, an Apple GPU's 32 KiB at most).
            L::ThreadgroupMemory => first([at_least(
                Feature::ThreadgroupMemorySize,
                VkLimit::MaxComputeSharedMemorySize,
                f.metal.max_threadgroup_memory_length,
            )]),
            // An explicit imageblock: a kernel's is threadgroup memory, within the compile layer's
            // `maxThreadgroupMemoryLength`.
            L::ExplicitImageblockMemory => first([at_least(
                Feature::ThreadgroupMemorySize,
                VkLimit::MaxComputeSharedMemorySize,
                f.metal.max_threadgroup_memory_length,
            )]),
            // A render pass's tile memory: its per-sample storage, tile memory and tile sizes are
            // checked as Metal checks them (`tile_pass.rs`, measured on an M3 Max and an M4), on
            // every device, and its attachments' packing as Vulkan bounds it. The feature rows say
            // what runs in such a pass. Apple's measured GPUs have no 8-sample pass, so a pass of 8
            // samples that sets its tile memory is not measured.
            L::ImplicitImageblockMemory
            | L::TileMemory
            | L::ExplicitImageblockSizePerSample
            | L::TileSizeWithoutMsaa
            | L::TileSize2xMsaa
            | L::TileSize4xMsaa => RowState::Exact,
            L::TileSize8xMsaa => first([gap(G::TileAtSampleCount)]),
            // The bytes per pixel of a pass's color attachments: Vulkan bounds their number only
            // (the color render targets row).
            L::ImplicitImageblockSizePerSample | L::RenderTargetSizePerPixel => RowState::Exact,
            // Threadgroup memory lengths in multiples of 16 bytes, which the compute encoder asks;
            // constant buffers at any offset the runtime binds; a texture copied into a buffer at
            // an offset that is a multiple of every texel block's size (16 bytes and more).
            L::ThreadgroupMemoryAlignment
            | L::ConstantBufferOffsetAlignment
            | L::TextureToBufferCopyAlignment => RowState::Exact,
            // A constant buffer is bounded by its buffer only (`maxBufferLength`).
            L::ConstantAddressSpaceBuffer => RowState::Exact,
            // A fragment function's inputs: the components the link packs them into, within
            // `maxFragmentInputComponents` and the vertex function's `maxVertexOutputComponents`.
            L::FragmentInputs | L::FragmentInputComponents => first([
                at_least(
                    Feature::InterfaceComponents,
                    VkLimit::MaxFragmentInputComponents,
                    l.max_fragment_input_components,
                ),
                at_least(
                    Feature::InterfaceComponents,
                    VkLimit::MaxVertexOutputComponents,
                    l.max_vertex_output_components,
                ),
            ]),
            // Tessellation factors: `maxTessellationGenerationLevel` on a device that enables
            // `tessellationShader`; elsewhere the emulation's tessellator takes Metal's every
            // factor (`render/tessellation_scan.rs`). The patch draws that read them are built
            // (`tessellation.rs`); what the compile layer still refuses of tessellation is the
            // feature row's.
            L::TessellationFactor => first([
                match f.properties.description.has(VkFeature::TessellationShader) {
                    true => at_least(
                        Feature::Tessellation,
                        VkLimit::MaxTessellationGenerationLevel,
                        l.max_tessellation_generation_level,
                    ),
                    false => None,
                },
            ]),
            // One viewport on every device; more need `multiViewport`, whose devices take at least
            // 16 (Vulkan's required `maxViewports`).
            L::ViewportsAndScissors => first([match need {
                Need::AtLeast(n) if n > 1 => {
                    needs_features(f, Feature::LayeredRendering, &[VkFeature::MultiViewport])
                }
                _ => None,
            }]),
            // Raster order groups: as the feature row.
            L::RasterOrderGroupsPerFunction => first([
                listed(f, Feature::RasterOrderGroups),
                query(
                    f.properties.are_raster_order_groups_supported(),
                    Query::RasterOrderGroups,
                ),
            ]),
            // A vertex buffer's stride and an attribute's offset: the compile layer fetches in the
            // shader what the device's vertex input cannot take.
            L::VertexBufferStrideAlignment | L::VertexBufferStride => RowState::Exact,
            // Argument buffers' buffers, reached by device address (set 1 for dispatches, draws
            // and tile dispatches alike).
            L::ArgumentBufferBuffers => first([addressed(f, Feature::ArgumentBufferPointers)]),
            // Argument buffers' textures: the handle table's descriptors.
            L::ArgumentBufferTextures => first([at_least(
                Feature::ArgumentBufferTextures,
                VkLimit::MaxDescriptorSetUpdateAfterBindSampledImages,
                l.max_descriptor_set_update_after_bind_sampled_images,
            )]),
            // Argument buffers' samplers: the live states the runtime holds for them,
            // `maxArgumentBufferSamplerCount` as it answers it
            // (`DeviceProperties::max_argument_buffer_sampler_count`), from the set-1 sampler
            // array link makes (`pipeline::heap_sampler_capacity`) and the device's live samplers.
            L::ArgumentBufferSamplers => {
                let available = f.properties.max_argument_buffer_sampler_count;
                match need {
                    Need::AtLeast(required) if available < required => {
                        RowState::Refused(Refusal::ArgumentBufferSamplerCount {
                            required,
                            available,
                        })
                    }
                    _ => RowState::Exact,
                }
            }
            // Texture sizes: the device's image limits.
            L::TextureWidth1D => first([at_least(
                Feature::TextureSize,
                VkLimit::MaxImageDimension1D,
                l.max_image_dimension_1d,
            )]),
            L::TextureSize2D => first([at_least(
                Feature::TextureSize,
                VkLimit::MaxImageDimension2D,
                l.max_image_dimension_2d,
            )]),
            L::CubeTextureSize => first([at_least(
                Feature::TextureSize,
                VkLimit::MaxImageDimensionCube,
                l.max_image_dimension_cube,
            )]),
            L::TextureSize3D => first([at_least(
                Feature::TextureSize,
                VkLimit::MaxImageDimension3D,
                l.max_image_dimension_3d,
            )]),
            L::TextureArrayLayers => first([at_least(
                Feature::TextureSize,
                VkLimit::MaxImageArrayLayers,
                l.max_image_array_layers,
            )]),
            // A texture buffer's texels: `maxTexelBufferElements`.
            L::TextureBufferWidth => first([at_least(
                Feature::TexelBuffer,
                VkLimit::MaxTexelBufferElements,
                l.max_texel_buffer_elements,
            )]),
            // A pass's color attachments: `maxColorAttachments` and
            // `maxFragmentOutputAttachments`. Past the lesser, a pass draws its colors in groups
            // of that many (`render/color_groups.rs`), whose refusals keep the row a gap.
            L::ColorRenderTargets => {
                let width = l
                    .max_color_attachments
                    .min(l.max_fragment_output_attachments);
                first([match need {
                    Need::AtLeast(n) if u64::from(width) < n => gap(G::ColorAttachmentGroups),
                    _ => None,
                }])
            }
            // `[[point_size]]`: the device clamps a point to its `pointSizeRange`, whose upper
            // bound (in whole pixels, `Limits::max_point_size`) must reach Metal's largest point;
            // the translation stores a larger size as Metal's.
            L::PointSize => match need {
                Need::AtLeast(n) if u64::from(l.max_point_size) < n => {
                    RowState::Refused(Refusal::PointSizeRange)
                }
                _ => RowState::Exact,
            },
            // A multisampled pass: the sample count in `framebufferColorSampleCounts` and
            // `framebufferDepthSampleCounts`.
            L::RenderPassSampleCount => first([sample_count(f, need)]),
            // Vertex amplification: multiview, whose views past `maxMultiviewViewCount` draw one
            // amplification at a time, storing the slice through the layer output or the
            // generated geometry stage, then the runtime's query (the feature row holds the
            // tessellation pipelines not implemented).
            L::VertexAmplificationCount => first([
                needs_features(f, Feature::VertexAmplification, &[VkFeature::Multiview]),
                match need {
                    Need::AtLeast(n) if n > u64::from(l.max_multiview_view_count) => needs_any(
                        f,
                        Feature::VertexAmplification,
                        &[
                            &[VkFeature::ShaderOutputLayer],
                            &[VkFeature::GeometryShader],
                        ],
                    ),
                    _ => None,
                },
                match need {
                    Need::AtLeast(n) => query(
                        f.properties.supports_vertex_amplification_count(n),
                        Query::VertexAmplificationCount,
                    ),
                    _ => None,
                },
            ]),
            // Mesh grids: `meshShader`, then the device's group counts, along each axis (a grid
            // may lie along one, up to the 65,535 threadgroups an axis past which Metal draws
            // nothing: `bind::METAL_MESH_GRID_AXIS`, measured on the M3 Max) and in all. Object
            // grids: `taskShader` too, then the device's task group counts likewise. Payloads:
            // `taskShader`, then `maxTaskPayloadSize`. Where the device runs both stages as compute
            // work (`DeviceProfile::emulates_mesh_shading`): a mesh grid is folded into rows of
            // `maxComputeWorkGroupCount[0]`, at most `maxComputeWorkGroupCount[1]` of them, which
            // Vulkan's minimums (65,535 each) put past every count the tables list; an object grid
            // is a compute dispatch, within `maxComputeWorkGroupCount` along each axis; and the
            // payload is compute workgroup memory beside the object wrapper's 12 bytes.
            L::MeshThreadgroupsPerGrid if f.profile.emulates_mesh_shading() => RowState::Exact,
            L::ObjectThreadgroupsPerGrid if f.profile.emulates_mesh_shading() => {
                let counts = l.max_compute_work_group_count;
                let along = |i: u8| {
                    self::at_least(
                        need,
                        Feature::MeshShading,
                        VkLimit::MaxComputeWorkGroupCount(i),
                        u64::from(counts[usize::from(i)]),
                    )
                };
                first([along(0), along(1), along(2)])
            }
            L::MeshPayload if f.profile.emulates_mesh_shading() => first([self::at_least(
                need,
                Feature::MeshShading,
                VkLimit::MaxComputeSharedMemorySize,
                u64::from(l.max_compute_shared_memory_size).saturating_sub(12),
            )]),
            L::MeshThreadgroupsPerGrid => {
                let counts = l.mesh.max_mesh_work_group_count;
                let axis = match need {
                    Need::AtLeast(n) => {
                        Need::AtLeast(n.min(u64::from(metal2vulkan::bind::METAL_MESH_GRID_AXIS)))
                    }
                    other => other,
                };
                let along = |i: u8| {
                    self::at_least(
                        axis,
                        Feature::MeshShading,
                        VkLimit::MaxMeshWorkGroupCount(i),
                        u64::from(counts[usize::from(i)]),
                    )
                };
                first([
                    needs_features(f, Feature::MeshShading, &[VkFeature::MeshShader]),
                    along(0),
                    along(1),
                    along(2),
                    at_least(
                        Feature::MeshShading,
                        VkLimit::MaxMeshWorkGroupTotalCount,
                        l.mesh.max_mesh_work_group_total_count,
                    ),
                ])
            }
            L::ObjectThreadgroupsPerGrid => {
                let counts = l.mesh.max_task_work_group_count;
                first([
                    needs_features(
                        f,
                        Feature::MeshShading,
                        &[VkFeature::MeshShader, VkFeature::TaskShader],
                    ),
                    at_least(
                        Feature::MeshShading,
                        VkLimit::MaxTaskWorkGroupCount(0),
                        counts[0],
                    ),
                    at_least(
                        Feature::MeshShading,
                        VkLimit::MaxTaskWorkGroupCount(1),
                        counts[1],
                    ),
                    at_least(
                        Feature::MeshShading,
                        VkLimit::MaxTaskWorkGroupCount(2),
                        counts[2],
                    ),
                    at_least(
                        Feature::MeshShading,
                        VkLimit::MaxTaskWorkGroupTotalCount,
                        l.mesh.max_task_work_group_total_count,
                    ),
                ])
            }
            L::MeshPayload => first([
                needs_features(
                    f,
                    Feature::MeshShading,
                    &[VkFeature::MeshShader, VkFeature::TaskShader],
                ),
                at_least(
                    Feature::MeshShading,
                    VkLimit::MaxTaskPayloadSize,
                    l.mesh.max_task_payload_size,
                ),
            ]),
            // Ray tracing: as the feature rows.
            L::IntersectorLevels
            | L::IntersectionQueryLevels
            | L::IntersectionFunctionBufferAlignment
            | L::IntersectionFunctionBufferStrideAlignment
            | L::IntersectionFunctionBufferStride => {
                first([ray_tracing(f), gap(G::AccelerationStructures)])
            }
            L::CounterSampleBufferLength | L::CounterSampleBuffers => {
                first([gap(G::CounterSampleBuffers)])
            }
            L::ResidencySetsPerQueue | L::ResidencySetsPerBuffer => first([gap(G::ResidencySets)]),
            // An indirect command buffer's commands are records in one buffer of the device's
            // `max_buffer_length` (`IndirectCommandBufferPastMaxBufferLength` past it), as Metal's
            // are bounded by its memory.
            L::IndirectCommandBufferLength => RowState::Exact,
            L::IoCommandsPerBuffer => first([gap(G::IoCommandQueues)]),
            L::TextureViewPoolEntries => first([gap(G::TextureViewPools)]),
            L::PerformanceCounterHeaps => first([gap(G::Metal4Api)]),
        }
    }
}

/// `feature`, refused where the device's `limit`, `available`, is below what `need` asks. The
/// rows it serves ask a bound in every column.
fn at_least(need: Need, feature: Feature, limit: VkLimit, available: u64) -> Check {
    match need {
        Need::AtLeast(required) if available < required => Some(refused(Unsupported::needs(
            feature,
            Capability::Limit {
                limit,
                required,
                available,
            },
        ))),
        _ => None,
    }
}

/// A multisampled pass of the sample count `need` asks: `framebufferColorSampleCounts` and
/// `framebufferDepthSampleCounts` have it.
fn sample_count(f: &Facts<'_>, need: Need) -> Check {
    let Need::AtLeast(count) = need else {
        return None;
    };
    let l = &f.properties.description.limits;
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    let bit = SampleCountFlags::from_raw(count);
    let missing = if !l.framebuffer_color_sample_counts.contains(bit) {
        VkProperty::FramebufferColorSampleCount
    } else if !l.framebuffer_depth_sample_counts.contains(bit) {
        VkProperty::FramebufferDepthSampleCount
    } else {
        return None;
    };
    Some(refused(Unsupported::needs(
        Feature::RasterSampleCount,
        Capability::Property(missing(count)),
    )))
}
