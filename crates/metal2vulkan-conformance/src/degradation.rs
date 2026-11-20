//! The degrade-notice codes the suite's cases name: `metal2vulkan-degrade-<kind>`, one per kind,
//! and the kind each of the runtime's precision notes announces as.
//!
//! A case loosens a field by naming a code ([`crate::allowance`]), and the comparison holds it
//! there only where the run announced that code. The runtime's side announces the codes of
//! the precision notes its states, textures and sampler states carry
//! ([`Degradation::from_precision_note`]); a code no note announces justifies nothing on its run.

use metal2vulkan::pipeline::PrecisionNote;

/// A kind of degrade notice, by its code ([`Degradation::code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Degradation {
    /// `metal2vulkan-degrade-shadowed-texture`.
    ShadowedTexture,
    /// `metal2vulkan-degrade-unordered-draw-bytes`.
    UnorderedDrawBytes,
    /// `metal2vulkan-degrade-unorm-depth-rounding`.
    UnormDepthRounding,
    /// `metal2vulkan-degrade-biased-depth-past-unit-range`.
    BiasedDepthPastUnitRange,
    /// `metal2vulkan-degrade-fragments-before-geometry`.
    FragmentsBeforeGeometry,
    /// `metal2vulkan-degrade-indirect-index-bounds-pass`.
    IndirectIndexBoundsPass,
    /// `metal2vulkan-degrade-emulated-mesh-draws`.
    EmulatedMeshDraws,
    /// `metal2vulkan-degrade-emulated-tessellation-draws`.
    EmulatedTessellationDraws,
    /// `metal2vulkan-degrade-overlapping-buffer-copy`.
    OverlappingBufferCopy,
    /// `metal2vulkan-degrade-split-buffer-fill`.
    SplitBufferFill,
    /// `metal2vulkan-degrade-staged-texture-copy`.
    StagedTextureCopy,
    /// `metal2vulkan-degrade-overlapping-texel-copy`.
    OverlappingTexelCopy,
    /// `metal2vulkan-degrade-split-dispatch`.
    SplitDispatch,
    /// `metal2vulkan-degrade-separate-depth-stencil`.
    SeparateDepthStencil,
    /// `metal2vulkan-degrade-one-aspect-resolve`.
    OneAspectResolve,
    /// `metal2vulkan-degrade-shader-resolve`.
    ShaderResolve,
    /// `metal2vulkan-degrade-shader-color-conversion`.
    ShaderColorConversion,
    /// `metal2vulkan-degrade-render-pass-split`.
    RenderPassSplit,
    /// `metal2vulkan-degrade-memoryless-texture-buffer`.
    MemorylessTextureBuffer,
    /// `metal2vulkan-degrade-fetch-through-scratch`.
    FetchThroughScratch,
    /// `metal2vulkan-degrade-sample-planes`.
    SamplePlanes,
    /// `metal2vulkan-degrade-shader-side-stencil`.
    ShaderSideStencil,
    /// `metal2vulkan-degrade-tile-dispatch-scratch`.
    TileDispatchScratch,
    /// `metal2vulkan-degrade-host-replayed-indirect-commands`.
    HostReplayedIndirectCommands,
    /// `metal2vulkan-degrade-staged-indirect-records`.
    StagedIndirectRecords,
    /// `metal2vulkan-degrade-uncalibrated-gpu-times`.
    UncalibratedGpuTimes,
    /// `metal2vulkan-degrade-loosely-calibrated-gpu-times`.
    LooselyCalibratedGpuTimes,
    /// `metal2vulkan-degrade-gpu-times-from-host`.
    GpuTimesFromHost,
    /// `metal2vulkan-degrade-device-rounded-resolve`.
    DeviceRoundedResolve,
    /// `metal2vulkan-degrade-extended-range-resolve`.
    ExtendedRangeResolve,
    /// `metal2vulkan-degrade-dont-care-store`.
    DontCareStore,
    /// `metal2vulkan-degrade-dont-care-load-tiles`.
    DontCareLoadTiles,
    /// `metal2vulkan-degrade-fresh-pattern-zeroed`.
    FreshPatternZeroed,
    /// `metal2vulkan-degrade-implementation-defined-reconvergence`.
    ImplementationDefinedReconvergence,
    /// `metal2vulkan-degrade-restricted-depth-upload`.
    RestrictedDepthUpload,
    /// `metal2vulkan-degrade-host-read-dispatch-arguments`.
    HostReadDispatchArguments,
    /// `metal2vulkan-degrade-non-strict-residency`.
    NonStrictResidency,
    /// `metal2vulkan-degrade-host-read-mapping-arguments`.
    HostReadMappingArguments,
    /// `metal2vulkan-degrade-emulated-sparse-texture`.
    EmulatedSparseTexture,
    /// `metal2vulkan-degrade-shared-in-system-memory`.
    SharedInSystemMemory,
    /// `metal2vulkan-degrade-device-filtered-mipmaps`.
    DeviceFilteredMipmaps,
    /// `metal2vulkan-degrade-shadowed-texture-view`.
    ShadowedTextureView,
    /// `metal2vulkan-degrade-capped-lod-bias`.
    CappedLodBias,
    /// `metal2vulkan-degrade-unnamed-texture-unviewable`.
    UnnamedTextureUnviewable,
    /// `metal2vulkan-degrade-unnamed-shadowed-texture`.
    UnnamedShadowedTexture,
    /// `metal2vulkan-degrade-drawn-depth-copy`.
    DrawnDepthCopy,
    /// `metal2vulkan-degrade-heap-two-plane-layout`.
    HeapTwoPlaneLayout,
    /// `metal2vulkan-degrade-two-plane-texture-copy`.
    TwoPlaneTextureCopy,
    /// `metal2vulkan-degrade-extended-range-bits-kept`.
    ExtendedRangeBitsKept,
    /// `metal2vulkan-degrade-snorm-minimum-kept`.
    SnormMinimumKept,
    /// `metal2vulkan-degrade-shared-exponent-kept`.
    SharedExponentKept,
    /// `metal2vulkan-degrade-two-plane-buffer-blit`.
    TwoPlaneBufferBlit,
    /// `metal2vulkan-degrade-two-plane-buffer-blit-order`.
    TwoPlaneBufferBlitOrder,
    /// `metal2vulkan-degrade-set-past-layout-support`.
    SetPastLayoutSupport,
    /// `metal2vulkan-degrade-ycbcr-filter-not-exact`.
    YCbCrFilterNotExact,
    /// `metal2vulkan-degrade-rotated-load-tiles`.
    RotatedLoadTiles,
    /// `metal2vulkan-degrade-heap-texture-lossless-layout`.
    HeapTextureLosslessLayout,
    /// `metal2vulkan-degrade-heap-texture-untiled-layout`.
    HeapTextureUntiledLayout,
    /// `metal2vulkan-degrade-extended-range-blends-stored-codes`.
    ExtendedRangeBlendsStoredCodes,
    /// `metal2vulkan-degrade-compressed-texel-decode`.
    CompressedTexelDecode,
    /// `metal2vulkan-degrade-cpu-block-decode`.
    CpuBlockDecode,
    /// `metal2vulkan-degrade-carried-decoded-twin`.
    CarriedDecodedTwin,
    /// `metal2vulkan-degrade-decoded-texel-precision`.
    DecodedTexelPrecision,
    /// `metal2vulkan-degrade-heap-texture-larger-than-metal`.
    HeapTextureLargerThanMetal,
    /// `metal2vulkan-degrade-dont-care-stand-in`.
    DontCareStandIn,
    /// `metal2vulkan-degrade-copied-host-pages`.
    CopiedHostPages,
    /// `metal2vulkan-degrade-depth-denormals`.
    DepthDenormals,
    /// `metal2vulkan-degrade-signed-depth-test`.
    SignedDepthTest,
    /// `metal2vulkan-degrade-first-sparse-resource-wait`.
    FirstSparseResourceWait,
    /// `metal2vulkan-degrade-unpacked-fill-color`.
    UnpackedFillColor,
    /// `metal2vulkan-degrade-texture-buffer-fill-leftover`.
    TextureBufferFillLeftover,
    /// `metal2vulkan-degrade-device-line-ties`.
    DeviceLineTies,
    /// `metal2vulkan-degrade-device-subpixel-snap`.
    DeviceSubpixelSnap,
    /// `metal2vulkan-degrade-device-boolean-visibility`.
    DeviceBooleanVisibility,
    /// `metal2vulkan-degrade-float-tininess-after-rounding`.
    FloatTininessAfterRounding,
    /// `metal2vulkan-degrade-f16-fma-double-rounding`.
    F16FmaDoubleRounding,
    /// `metal2vulkan-degrade-f16-int-conv-double-rounding`.
    F16IntConvDoubleRounding,
    /// `metal2vulkan-degrade-grad-from-bias`.
    GradFromBias,
    /// `metal2vulkan-degrade-device-transcendentals`.
    DeviceTranscendentals,
    /// `metal2vulkan-degrade-reduction-order`.
    ReductionOrder,
    /// `metal2vulkan-degrade-device-denormals`.
    DeviceDenormals,
    /// `metal2vulkan-degrade-pixel-sampler-scaling`.
    PixelSamplerScaling,
    /// `metal2vulkan-degrade-device-arithmetic`.
    DeviceArithmetic,
    /// `metal2vulkan-degrade-device-signed-zero-inf-nan`.
    DeviceSignedZeroInfNan,
    /// `metal2vulkan-degrade-device-ray-intersection`.
    DeviceRayIntersection,
    /// `metal2vulkan-degrade-fast-math-rewrites`.
    FastMathRewrites,
    /// `metal2vulkan-degrade-tessellation-winding`.
    TessellationWinding,
    /// `metal2vulkan-degrade-tessellation-fractional-spacing`.
    TessellationFractionalSpacing,
    /// `metal2vulkan-degrade-tessellation-factor-edge-cases`.
    TessellationFactorEdgeCases,
    /// `metal2vulkan-degrade-tessellation-factor-scale`.
    TessellationFactorScale,
    /// `metal2vulkan-degrade-tessellation-factor-instance`.
    TessellationFactorInstance,
    /// `metal2vulkan-degrade-fp8-value`.
    Fp8Value,
    /// `metal2vulkan-degrade-multisample-position`.
    MultisamplePosition,
    /// `metal2vulkan-degrade-rasterization-rate-map`.
    RasterizationRateMap,
    /// `metal2vulkan-degrade-bicubic-filter-arithmetic`.
    BicubicFilterArithmetic,
    /// `metal2vulkan-degrade-sparse-residency`.
    SparseResidency,
    /// `metal2vulkan-degrade-device-nan-encoding`.
    DeviceNanEncoding,
    /// `metal2vulkan-degrade-restricted-depth-range`.
    RestrictedDepthRange,
    /// `metal2vulkan-degrade-cube-gradient-footprint`.
    CubeGradientFootprint,
    /// `metal2vulkan-degrade-half-texel-conversion`.
    HalfTexelConversion,
    /// `metal2vulkan-degrade-half-sample-filter`.
    HalfSampleFilter,
    /// `metal2vulkan-degrade-srgb-texel-decode`.
    SrgbTexelDecode,
    /// `metal2vulkan-degrade-float-sample-filter`.
    FloatSampleFilter,
    /// `metal2vulkan-degrade-function-storage-assumed`.
    FunctionStorageAssumed,
    /// `metal2vulkan-degrade-quads-counted-before-discard`.
    QuadsCountedBeforeDiscard,
    /// `metal2vulkan-degrade-view-base-texel-offset`.
    ViewBaseTexelOffset,
    /// `metal2vulkan-degrade-device-color-conversion`.
    DeviceColorConversion,
    /// `metal2vulkan-degrade-device-interpolation`.
    DeviceInterpolation,
    /// `metal2vulkan-degrade-device-interpolation-offset`.
    DeviceInterpolationOffset,
    /// `metal2vulkan-degrade-extrapolated-interpolation`.
    ExtrapolatedInterpolation,
    /// `metal2vulkan-degrade-quads-split-by-clipping`.
    QuadsSplitByClipping,
    /// `metal2vulkan-degrade-vertex-simdgroup-grouping`.
    VertexSimdgroupGrouping,
    /// `metal2vulkan-degrade-fragment-simdgroup-grouping`.
    FragmentSimdgroupGrouping,
    /// `metal2vulkan-degrade-blended-write-rounding`.
    BlendedWriteRounding,
    /// `metal2vulkan-degrade-rotated-blend-rounding`.
    RotatedBlendRounding,
    /// `metal2vulkan-degrade-linked-sampler-numbering`.
    LinkedSamplerNumbering,
    /// `metal2vulkan-degrade-computed-sample-depth`.
    ComputedSampleDepth,
    /// `metal2vulkan-degrade-argument-pixel-sampler-scaling`.
    ArgumentPixelSamplerScaling,
    /// `metal2vulkan-degrade-half-output-precision`.
    HalfOutputPrecision,
    /// `metal2vulkan-degrade-extended-range-blend-source`.
    ExtendedRangeBlendSource,
    /// `metal2vulkan-degrade-discard-drops-simdgroup-stores`.
    DiscardDropsSimdgroupStores,
    /// `metal2vulkan-degrade-helper-branch-drops-simdgroup-stores`.
    HelperBranchDropsSimdgroupStores,
    /// `metal2vulkan-degrade-extended-range-float-blend-source`.
    ExtendedRangeFloatBlendSource,
    /// `metal2vulkan-degrade-anisotropic-filter`.
    AnisotropicFilter,
    /// `metal2vulkan-degrade-device-anisotropic-filter`.
    DeviceAnisotropicFilter,
    /// `metal2vulkan-degrade-device-point-size`.
    DevicePointSize,
    /// `metal2vulkan-degrade-clipped-coverage-split`.
    ClippedCoverageSplit,
    /// `metal2vulkan-degrade-looped-sample-outputs`.
    LoopedSampleOutputs,
    /// `metal2vulkan-degrade-half-output-fast-math-blend`.
    HalfOutputFastMathBlend,
    /// `metal2vulkan-degrade-device-sampling`.
    DeviceSampling,
    /// `metal2vulkan-degrade-native-transcendentals`.
    NativeTranscendentals,
    /// `metal2vulkan-degrade-native-division`.
    NativeDivision,
    /// `metal2vulkan-degrade-native-fma`.
    NativeFma,
    /// `metal2vulkan-degrade-device-texel-conversion`.
    DeviceTexelConversion,
    /// `metal2vulkan-degrade-device-vertex-conversion`.
    DeviceVertexConversion,
    /// `metal2vulkan-degrade-native-half-conversion`.
    NativeHalfConversion,
    /// `metal2vulkan-degrade-native-half-quantize`.
    NativeHalfQuantize,
    /// `metal2vulkan-degrade-native-nan-encoding`.
    NativeNanEncoding,
    /// `metal2vulkan-degrade-native-min-max`.
    NativeMinMax,
    /// `metal2vulkan-degrade-native-dot`.
    NativeDot,
    /// `metal2vulkan-degrade-native-mix`.
    NativeMix,
    /// `metal2vulkan-degrade-device-pack-conversion`.
    DevicePackConversion,
    /// `metal2vulkan-degrade-color-attachment-groups`.
    ColorAttachmentGroups,
    /// `metal2vulkan-degrade-tessellation-triangle-order`.
    TessellationTriangleOrder,
    /// `metal2vulkan-degrade-unmeasured-group-speculation`.
    UnmeasuredGroupSpeculation,
    /// `metal2vulkan-degrade-not-mipmapped-anisotropic-filter`.
    NotMipmappedAnisotropicFilter,
    /// `metal2vulkan-degrade-device-sample-level`.
    DeviceSampleLevel,
    /// `metal2vulkan-degrade-coverage-reads-rounded-alpha`.
    CoverageReadsRoundedAlpha,
    /// `metal2vulkan-degrade-device-alpha-coverage`.
    DeviceAlphaCoverage,
    /// `metal2vulkan-degrade-unmeasured-fetch-blend`.
    UnmeasuredFetchBlend,
    /// `metal2vulkan-degrade-unordered-fetch-within-draw`.
    UnorderedFetchWithinDraw,
    /// `metal2vulkan-degrade-native-logic-operation`.
    NativeLogicOperation,
    /// `metal2vulkan-degrade-native-float8`.
    NativeFloat8,
    /// `metal2vulkan-degrade-device-depth-bias`.
    DeviceDepthBias,
    /// `metal2vulkan-degrade-swizzled-opaque-black-border`.
    SwizzledOpaqueBlackBorder,
}

impl Degradation {
    /// Every kind, in declaration order.
    pub const ALL: &'static [Degradation] = &[
        Degradation::ShadowedTexture,
        Degradation::UnorderedDrawBytes,
        Degradation::UnormDepthRounding,
        Degradation::BiasedDepthPastUnitRange,
        Degradation::FragmentsBeforeGeometry,
        Degradation::IndirectIndexBoundsPass,
        Degradation::EmulatedMeshDraws,
        Degradation::EmulatedTessellationDraws,
        Degradation::OverlappingBufferCopy,
        Degradation::SplitBufferFill,
        Degradation::StagedTextureCopy,
        Degradation::OverlappingTexelCopy,
        Degradation::SplitDispatch,
        Degradation::SeparateDepthStencil,
        Degradation::OneAspectResolve,
        Degradation::ShaderResolve,
        Degradation::ShaderColorConversion,
        Degradation::RenderPassSplit,
        Degradation::MemorylessTextureBuffer,
        Degradation::FetchThroughScratch,
        Degradation::SamplePlanes,
        Degradation::ShaderSideStencil,
        Degradation::TileDispatchScratch,
        Degradation::HostReplayedIndirectCommands,
        Degradation::StagedIndirectRecords,
        Degradation::UncalibratedGpuTimes,
        Degradation::LooselyCalibratedGpuTimes,
        Degradation::GpuTimesFromHost,
        Degradation::DeviceRoundedResolve,
        Degradation::ExtendedRangeResolve,
        Degradation::DontCareStore,
        Degradation::DontCareLoadTiles,
        Degradation::FreshPatternZeroed,
        Degradation::ImplementationDefinedReconvergence,
        Degradation::RestrictedDepthUpload,
        Degradation::HostReadDispatchArguments,
        Degradation::NonStrictResidency,
        Degradation::HostReadMappingArguments,
        Degradation::EmulatedSparseTexture,
        Degradation::SharedInSystemMemory,
        Degradation::DeviceFilteredMipmaps,
        Degradation::ShadowedTextureView,
        Degradation::CappedLodBias,
        Degradation::UnnamedTextureUnviewable,
        Degradation::UnnamedShadowedTexture,
        Degradation::DrawnDepthCopy,
        Degradation::HeapTwoPlaneLayout,
        Degradation::TwoPlaneTextureCopy,
        Degradation::ExtendedRangeBitsKept,
        Degradation::SnormMinimumKept,
        Degradation::SharedExponentKept,
        Degradation::TwoPlaneBufferBlit,
        Degradation::TwoPlaneBufferBlitOrder,
        Degradation::SetPastLayoutSupport,
        Degradation::YCbCrFilterNotExact,
        Degradation::RotatedLoadTiles,
        Degradation::HeapTextureLosslessLayout,
        Degradation::HeapTextureUntiledLayout,
        Degradation::ExtendedRangeBlendsStoredCodes,
        Degradation::CompressedTexelDecode,
        Degradation::CpuBlockDecode,
        Degradation::CarriedDecodedTwin,
        Degradation::DecodedTexelPrecision,
        Degradation::HeapTextureLargerThanMetal,
        Degradation::DontCareStandIn,
        Degradation::CopiedHostPages,
        Degradation::DepthDenormals,
        Degradation::SignedDepthTest,
        Degradation::FirstSparseResourceWait,
        Degradation::UnpackedFillColor,
        Degradation::TextureBufferFillLeftover,
        Degradation::DeviceLineTies,
        Degradation::DeviceSubpixelSnap,
        Degradation::DeviceBooleanVisibility,
        Degradation::FloatTininessAfterRounding,
        Degradation::F16FmaDoubleRounding,
        Degradation::F16IntConvDoubleRounding,
        Degradation::GradFromBias,
        Degradation::DeviceTranscendentals,
        Degradation::ReductionOrder,
        Degradation::DeviceDenormals,
        Degradation::PixelSamplerScaling,
        Degradation::DeviceArithmetic,
        Degradation::DeviceSignedZeroInfNan,
        Degradation::DeviceRayIntersection,
        Degradation::FastMathRewrites,
        Degradation::TessellationWinding,
        Degradation::TessellationFractionalSpacing,
        Degradation::TessellationFactorEdgeCases,
        Degradation::TessellationFactorScale,
        Degradation::TessellationFactorInstance,
        Degradation::Fp8Value,
        Degradation::MultisamplePosition,
        Degradation::RasterizationRateMap,
        Degradation::BicubicFilterArithmetic,
        Degradation::SparseResidency,
        Degradation::DeviceNanEncoding,
        Degradation::RestrictedDepthRange,
        Degradation::CubeGradientFootprint,
        Degradation::HalfTexelConversion,
        Degradation::HalfSampleFilter,
        Degradation::SrgbTexelDecode,
        Degradation::FloatSampleFilter,
        Degradation::FunctionStorageAssumed,
        Degradation::QuadsCountedBeforeDiscard,
        Degradation::ViewBaseTexelOffset,
        Degradation::DeviceColorConversion,
        Degradation::DeviceInterpolation,
        Degradation::DeviceInterpolationOffset,
        Degradation::ExtrapolatedInterpolation,
        Degradation::QuadsSplitByClipping,
        Degradation::VertexSimdgroupGrouping,
        Degradation::FragmentSimdgroupGrouping,
        Degradation::BlendedWriteRounding,
        Degradation::RotatedBlendRounding,
        Degradation::LinkedSamplerNumbering,
        Degradation::ComputedSampleDepth,
        Degradation::ArgumentPixelSamplerScaling,
        Degradation::HalfOutputPrecision,
        Degradation::ExtendedRangeBlendSource,
        Degradation::DiscardDropsSimdgroupStores,
        Degradation::HelperBranchDropsSimdgroupStores,
        Degradation::ExtendedRangeFloatBlendSource,
        Degradation::AnisotropicFilter,
        Degradation::DeviceAnisotropicFilter,
        Degradation::DevicePointSize,
        Degradation::ClippedCoverageSplit,
        Degradation::LoopedSampleOutputs,
        Degradation::HalfOutputFastMathBlend,
        Degradation::DeviceSampling,
        Degradation::NativeTranscendentals,
        Degradation::NativeDivision,
        Degradation::NativeFma,
        Degradation::DeviceTexelConversion,
        Degradation::DeviceVertexConversion,
        Degradation::NativeHalfConversion,
        Degradation::NativeHalfQuantize,
        Degradation::NativeNanEncoding,
        Degradation::NativeMinMax,
        Degradation::NativeDot,
        Degradation::NativeMix,
        Degradation::DevicePackConversion,
        Degradation::ColorAttachmentGroups,
        Degradation::TessellationTriangleOrder,
        Degradation::UnmeasuredGroupSpeculation,
        Degradation::NotMipmappedAnisotropicFilter,
        Degradation::DeviceSampleLevel,
        Degradation::CoverageReadsRoundedAlpha,
        Degradation::DeviceAlphaCoverage,
        Degradation::UnmeasuredFetchBlend,
        Degradation::UnorderedFetchWithinDraw,
        Degradation::NativeLogicOperation,
        Degradation::NativeFloat8,
        Degradation::DeviceDepthBias,
        Degradation::SwizzledOpaqueBlackBorder,
    ];

    /// The kind's code, `metal2vulkan-degrade-<kind>`.
    pub const fn code(self) -> &'static str {
        match self {
            Degradation::ShadowedTexture => "metal2vulkan-degrade-shadowed-texture",
            Degradation::UnorderedDrawBytes => "metal2vulkan-degrade-unordered-draw-bytes",
            Degradation::UnormDepthRounding => "metal2vulkan-degrade-unorm-depth-rounding",
            Degradation::BiasedDepthPastUnitRange => {
                "metal2vulkan-degrade-biased-depth-past-unit-range"
            }
            Degradation::FragmentsBeforeGeometry => {
                "metal2vulkan-degrade-fragments-before-geometry"
            }
            Degradation::IndirectIndexBoundsPass => {
                "metal2vulkan-degrade-indirect-index-bounds-pass"
            }
            Degradation::EmulatedMeshDraws => "metal2vulkan-degrade-emulated-mesh-draws",
            Degradation::EmulatedTessellationDraws => {
                "metal2vulkan-degrade-emulated-tessellation-draws"
            }
            Degradation::OverlappingBufferCopy => "metal2vulkan-degrade-overlapping-buffer-copy",
            Degradation::SplitBufferFill => "metal2vulkan-degrade-split-buffer-fill",
            Degradation::StagedTextureCopy => "metal2vulkan-degrade-staged-texture-copy",
            Degradation::OverlappingTexelCopy => "metal2vulkan-degrade-overlapping-texel-copy",
            Degradation::SplitDispatch => "metal2vulkan-degrade-split-dispatch",
            Degradation::SeparateDepthStencil => "metal2vulkan-degrade-separate-depth-stencil",
            Degradation::OneAspectResolve => "metal2vulkan-degrade-one-aspect-resolve",
            Degradation::ShaderResolve => "metal2vulkan-degrade-shader-resolve",
            Degradation::ShaderColorConversion => "metal2vulkan-degrade-shader-color-conversion",
            Degradation::RenderPassSplit => "metal2vulkan-degrade-render-pass-split",
            Degradation::MemorylessTextureBuffer => {
                "metal2vulkan-degrade-memoryless-texture-buffer"
            }
            Degradation::FetchThroughScratch => "metal2vulkan-degrade-fetch-through-scratch",
            Degradation::SamplePlanes => "metal2vulkan-degrade-sample-planes",
            Degradation::ShaderSideStencil => "metal2vulkan-degrade-shader-side-stencil",
            Degradation::TileDispatchScratch => "metal2vulkan-degrade-tile-dispatch-scratch",
            Degradation::HostReplayedIndirectCommands => {
                "metal2vulkan-degrade-host-replayed-indirect-commands"
            }
            Degradation::StagedIndirectRecords => "metal2vulkan-degrade-staged-indirect-records",
            Degradation::UncalibratedGpuTimes => "metal2vulkan-degrade-uncalibrated-gpu-times",
            Degradation::LooselyCalibratedGpuTimes => {
                "metal2vulkan-degrade-loosely-calibrated-gpu-times"
            }
            Degradation::GpuTimesFromHost => "metal2vulkan-degrade-gpu-times-from-host",
            Degradation::DeviceRoundedResolve => "metal2vulkan-degrade-device-rounded-resolve",
            Degradation::ExtendedRangeResolve => "metal2vulkan-degrade-extended-range-resolve",
            Degradation::DontCareStore => "metal2vulkan-degrade-dont-care-store",
            Degradation::DontCareLoadTiles => "metal2vulkan-degrade-dont-care-load-tiles",
            Degradation::FreshPatternZeroed => "metal2vulkan-degrade-fresh-pattern-zeroed",
            Degradation::ImplementationDefinedReconvergence => {
                "metal2vulkan-degrade-implementation-defined-reconvergence"
            }
            Degradation::RestrictedDepthUpload => "metal2vulkan-degrade-restricted-depth-upload",
            Degradation::HostReadDispatchArguments => {
                "metal2vulkan-degrade-host-read-dispatch-arguments"
            }
            Degradation::NonStrictResidency => "metal2vulkan-degrade-non-strict-residency",
            Degradation::HostReadMappingArguments => {
                "metal2vulkan-degrade-host-read-mapping-arguments"
            }
            Degradation::EmulatedSparseTexture => "metal2vulkan-degrade-emulated-sparse-texture",
            Degradation::SharedInSystemMemory => "metal2vulkan-degrade-shared-in-system-memory",
            Degradation::DeviceFilteredMipmaps => "metal2vulkan-degrade-device-filtered-mipmaps",
            Degradation::ShadowedTextureView => "metal2vulkan-degrade-shadowed-texture-view",
            Degradation::CappedLodBias => "metal2vulkan-degrade-capped-lod-bias",
            Degradation::UnnamedTextureUnviewable => {
                "metal2vulkan-degrade-unnamed-texture-unviewable"
            }
            Degradation::UnnamedShadowedTexture => "metal2vulkan-degrade-unnamed-shadowed-texture",
            Degradation::DrawnDepthCopy => "metal2vulkan-degrade-drawn-depth-copy",
            Degradation::HeapTwoPlaneLayout => "metal2vulkan-degrade-heap-two-plane-layout",
            Degradation::TwoPlaneTextureCopy => "metal2vulkan-degrade-two-plane-texture-copy",
            Degradation::ExtendedRangeBitsKept => "metal2vulkan-degrade-extended-range-bits-kept",
            Degradation::SnormMinimumKept => "metal2vulkan-degrade-snorm-minimum-kept",
            Degradation::SharedExponentKept => "metal2vulkan-degrade-shared-exponent-kept",
            Degradation::TwoPlaneBufferBlit => "metal2vulkan-degrade-two-plane-buffer-blit",
            Degradation::TwoPlaneBufferBlitOrder => {
                "metal2vulkan-degrade-two-plane-buffer-blit-order"
            }
            Degradation::SetPastLayoutSupport => "metal2vulkan-degrade-set-past-layout-support",
            Degradation::YCbCrFilterNotExact => "metal2vulkan-degrade-ycbcr-filter-not-exact",
            Degradation::RotatedLoadTiles => "metal2vulkan-degrade-rotated-load-tiles",
            Degradation::HeapTextureLosslessLayout => {
                "metal2vulkan-degrade-heap-texture-lossless-layout"
            }
            Degradation::HeapTextureUntiledLayout => {
                "metal2vulkan-degrade-heap-texture-untiled-layout"
            }
            Degradation::ExtendedRangeBlendsStoredCodes => {
                "metal2vulkan-degrade-extended-range-blends-stored-codes"
            }
            Degradation::CompressedTexelDecode => "metal2vulkan-degrade-compressed-texel-decode",
            Degradation::CpuBlockDecode => "metal2vulkan-degrade-cpu-block-decode",
            Degradation::CarriedDecodedTwin => "metal2vulkan-degrade-carried-decoded-twin",
            Degradation::DecodedTexelPrecision => "metal2vulkan-degrade-decoded-texel-precision",
            Degradation::HeapTextureLargerThanMetal => {
                "metal2vulkan-degrade-heap-texture-larger-than-metal"
            }
            Degradation::DontCareStandIn => "metal2vulkan-degrade-dont-care-stand-in",
            Degradation::CopiedHostPages => "metal2vulkan-degrade-copied-host-pages",
            Degradation::DepthDenormals => "metal2vulkan-degrade-depth-denormals",
            Degradation::SignedDepthTest => "metal2vulkan-degrade-signed-depth-test",
            Degradation::FirstSparseResourceWait => {
                "metal2vulkan-degrade-first-sparse-resource-wait"
            }
            Degradation::UnpackedFillColor => "metal2vulkan-degrade-unpacked-fill-color",
            Degradation::TextureBufferFillLeftover => {
                "metal2vulkan-degrade-texture-buffer-fill-leftover"
            }
            Degradation::DeviceLineTies => "metal2vulkan-degrade-device-line-ties",
            Degradation::DeviceSubpixelSnap => "metal2vulkan-degrade-device-subpixel-snap",
            Degradation::DeviceBooleanVisibility => {
                "metal2vulkan-degrade-device-boolean-visibility"
            }
            Degradation::FloatTininessAfterRounding => {
                "metal2vulkan-degrade-float-tininess-after-rounding"
            }
            Degradation::F16FmaDoubleRounding => "metal2vulkan-degrade-f16-fma-double-rounding",
            Degradation::F16IntConvDoubleRounding => {
                "metal2vulkan-degrade-f16-int-conv-double-rounding"
            }
            Degradation::GradFromBias => "metal2vulkan-degrade-grad-from-bias",
            Degradation::DeviceTranscendentals => "metal2vulkan-degrade-device-transcendentals",
            Degradation::ReductionOrder => "metal2vulkan-degrade-reduction-order",
            Degradation::DeviceDenormals => "metal2vulkan-degrade-device-denormals",
            Degradation::PixelSamplerScaling => "metal2vulkan-degrade-pixel-sampler-scaling",
            Degradation::DeviceArithmetic => "metal2vulkan-degrade-device-arithmetic",
            Degradation::DeviceSignedZeroInfNan => {
                "metal2vulkan-degrade-device-signed-zero-inf-nan"
            }
            Degradation::DeviceRayIntersection => "metal2vulkan-degrade-device-ray-intersection",
            Degradation::FastMathRewrites => "metal2vulkan-degrade-fast-math-rewrites",
            Degradation::TessellationWinding => "metal2vulkan-degrade-tessellation-winding",
            Degradation::TessellationFractionalSpacing => {
                "metal2vulkan-degrade-tessellation-fractional-spacing"
            }
            Degradation::TessellationFactorEdgeCases => {
                "metal2vulkan-degrade-tessellation-factor-edge-cases"
            }
            Degradation::TessellationFactorScale => {
                "metal2vulkan-degrade-tessellation-factor-scale"
            }
            Degradation::TessellationFactorInstance => {
                "metal2vulkan-degrade-tessellation-factor-instance"
            }
            Degradation::Fp8Value => "metal2vulkan-degrade-fp8-value",
            Degradation::MultisamplePosition => "metal2vulkan-degrade-multisample-position",
            Degradation::RasterizationRateMap => "metal2vulkan-degrade-rasterization-rate-map",
            Degradation::BicubicFilterArithmetic => {
                "metal2vulkan-degrade-bicubic-filter-arithmetic"
            }
            Degradation::SparseResidency => "metal2vulkan-degrade-sparse-residency",
            Degradation::DeviceNanEncoding => "metal2vulkan-degrade-device-nan-encoding",
            Degradation::RestrictedDepthRange => "metal2vulkan-degrade-restricted-depth-range",
            Degradation::CubeGradientFootprint => "metal2vulkan-degrade-cube-gradient-footprint",
            Degradation::HalfTexelConversion => "metal2vulkan-degrade-half-texel-conversion",
            Degradation::HalfSampleFilter => "metal2vulkan-degrade-half-sample-filter",
            Degradation::SrgbTexelDecode => "metal2vulkan-degrade-srgb-texel-decode",
            Degradation::FloatSampleFilter => "metal2vulkan-degrade-float-sample-filter",
            Degradation::FunctionStorageAssumed => "metal2vulkan-degrade-function-storage-assumed",
            Degradation::QuadsCountedBeforeDiscard => {
                "metal2vulkan-degrade-quads-counted-before-discard"
            }
            Degradation::ViewBaseTexelOffset => "metal2vulkan-degrade-view-base-texel-offset",
            Degradation::DeviceColorConversion => "metal2vulkan-degrade-device-color-conversion",
            Degradation::DeviceInterpolation => "metal2vulkan-degrade-device-interpolation",
            Degradation::DeviceInterpolationOffset => {
                "metal2vulkan-degrade-device-interpolation-offset"
            }
            Degradation::ExtrapolatedInterpolation => {
                "metal2vulkan-degrade-extrapolated-interpolation"
            }
            Degradation::QuadsSplitByClipping => "metal2vulkan-degrade-quads-split-by-clipping",
            Degradation::VertexSimdgroupGrouping => {
                "metal2vulkan-degrade-vertex-simdgroup-grouping"
            }
            Degradation::FragmentSimdgroupGrouping => {
                "metal2vulkan-degrade-fragment-simdgroup-grouping"
            }
            Degradation::BlendedWriteRounding => "metal2vulkan-degrade-blended-write-rounding",
            Degradation::RotatedBlendRounding => "metal2vulkan-degrade-rotated-blend-rounding",
            Degradation::LinkedSamplerNumbering => "metal2vulkan-degrade-linked-sampler-numbering",
            Degradation::ComputedSampleDepth => "metal2vulkan-degrade-computed-sample-depth",
            Degradation::ArgumentPixelSamplerScaling => {
                "metal2vulkan-degrade-argument-pixel-sampler-scaling"
            }
            Degradation::HalfOutputPrecision => "metal2vulkan-degrade-half-output-precision",
            Degradation::ExtendedRangeBlendSource => {
                "metal2vulkan-degrade-extended-range-blend-source"
            }
            Degradation::DiscardDropsSimdgroupStores => {
                "metal2vulkan-degrade-discard-drops-simdgroup-stores"
            }
            Degradation::HelperBranchDropsSimdgroupStores => {
                "metal2vulkan-degrade-helper-branch-drops-simdgroup-stores"
            }
            Degradation::ExtendedRangeFloatBlendSource => {
                "metal2vulkan-degrade-extended-range-float-blend-source"
            }
            Degradation::AnisotropicFilter => "metal2vulkan-degrade-anisotropic-filter",
            Degradation::DeviceAnisotropicFilter => {
                "metal2vulkan-degrade-device-anisotropic-filter"
            }
            Degradation::DevicePointSize => "metal2vulkan-degrade-device-point-size",
            Degradation::ClippedCoverageSplit => "metal2vulkan-degrade-clipped-coverage-split",
            Degradation::LoopedSampleOutputs => "metal2vulkan-degrade-looped-sample-outputs",
            Degradation::HalfOutputFastMathBlend => {
                "metal2vulkan-degrade-half-output-fast-math-blend"
            }
            Degradation::DeviceSampling => "metal2vulkan-degrade-device-sampling",
            Degradation::NativeTranscendentals => "metal2vulkan-degrade-native-transcendentals",
            Degradation::NativeDivision => "metal2vulkan-degrade-native-division",
            Degradation::NativeFma => "metal2vulkan-degrade-native-fma",
            Degradation::DeviceTexelConversion => "metal2vulkan-degrade-device-texel-conversion",
            Degradation::DeviceVertexConversion => "metal2vulkan-degrade-device-vertex-conversion",
            Degradation::NativeHalfConversion => "metal2vulkan-degrade-native-half-conversion",
            Degradation::NativeHalfQuantize => "metal2vulkan-degrade-native-half-quantize",
            Degradation::NativeNanEncoding => "metal2vulkan-degrade-native-nan-encoding",
            Degradation::NativeMinMax => "metal2vulkan-degrade-native-min-max",
            Degradation::NativeDot => "metal2vulkan-degrade-native-dot",
            Degradation::NativeMix => "metal2vulkan-degrade-native-mix",
            Degradation::DevicePackConversion => "metal2vulkan-degrade-device-pack-conversion",
            Degradation::ColorAttachmentGroups => "metal2vulkan-degrade-color-attachment-groups",
            Degradation::TessellationTriangleOrder => {
                "metal2vulkan-degrade-tessellation-triangle-order"
            }
            Degradation::UnmeasuredGroupSpeculation => {
                "metal2vulkan-degrade-unmeasured-group-speculation"
            }
            Degradation::NotMipmappedAnisotropicFilter => {
                "metal2vulkan-degrade-not-mipmapped-anisotropic-filter"
            }
            Degradation::DeviceSampleLevel => "metal2vulkan-degrade-device-sample-level",
            Degradation::CoverageReadsRoundedAlpha => {
                "metal2vulkan-degrade-coverage-reads-rounded-alpha"
            }
            Degradation::DeviceAlphaCoverage => "metal2vulkan-degrade-device-alpha-coverage",
            Degradation::UnmeasuredFetchBlend => "metal2vulkan-degrade-unmeasured-fetch-blend",
            Degradation::UnorderedFetchWithinDraw => {
                "metal2vulkan-degrade-unordered-fetch-within-draw"
            }
            Degradation::NativeLogicOperation => "metal2vulkan-degrade-native-logic-operation",
            Degradation::NativeFloat8 => "metal2vulkan-degrade-native-float8",
            Degradation::DeviceDepthBias => "metal2vulkan-degrade-device-depth-bias",
            Degradation::SwizzledOpaqueBlackBorder => {
                "metal2vulkan-degrade-swizzled-opaque-black-border"
            }
        }
    }

    /// The kind a precision note announces as.
    pub const fn from_precision_note(note: PrecisionNote) -> Degradation {
        match note {
            PrecisionNote::F16FmaDoubleRounding => Degradation::F16FmaDoubleRounding,
            PrecisionNote::F16IntConvDoubleRounding => Degradation::F16IntConvDoubleRounding,
            PrecisionNote::GradFromBias => Degradation::GradFromBias,
            PrecisionNote::DeviceTranscendentals => Degradation::DeviceTranscendentals,
            PrecisionNote::ReductionOrder => Degradation::ReductionOrder,
            PrecisionNote::DeviceDenormals => Degradation::DeviceDenormals,
            PrecisionNote::PixelSamplerScaling => Degradation::PixelSamplerScaling,
            PrecisionNote::DeviceArithmetic => Degradation::DeviceArithmetic,
            PrecisionNote::DeviceSignedZeroInfNan => Degradation::DeviceSignedZeroInfNan,
            PrecisionNote::DeviceRayIntersection => Degradation::DeviceRayIntersection,
            PrecisionNote::FastMathRewrites => Degradation::FastMathRewrites,
            PrecisionNote::TessellationWinding => Degradation::TessellationWinding,
            PrecisionNote::TessellationFractionalSpacing => {
                Degradation::TessellationFractionalSpacing
            }
            PrecisionNote::TessellationFactorEdgeCases => Degradation::TessellationFactorEdgeCases,
            PrecisionNote::TessellationFactorScale => Degradation::TessellationFactorScale,
            PrecisionNote::TessellationFactorInstance => Degradation::TessellationFactorInstance,
            PrecisionNote::Fp8Value => Degradation::Fp8Value,
            PrecisionNote::MultisamplePosition => Degradation::MultisamplePosition,
            PrecisionNote::RasterizationRateMap => Degradation::RasterizationRateMap,
            PrecisionNote::BicubicFilterArithmetic => Degradation::BicubicFilterArithmetic,
            PrecisionNote::SparseResidency => Degradation::SparseResidency,
            PrecisionNote::DeviceNanEncoding => Degradation::DeviceNanEncoding,
            PrecisionNote::RestrictedDepthRange => Degradation::RestrictedDepthRange,
            PrecisionNote::CubeGradientFootprint => Degradation::CubeGradientFootprint,
            PrecisionNote::HalfTexelConversion => Degradation::HalfTexelConversion,
            PrecisionNote::HalfSampleFilter => Degradation::HalfSampleFilter,
            PrecisionNote::SrgbTexelDecode => Degradation::SrgbTexelDecode,
            PrecisionNote::FloatSampleFilter => Degradation::FloatSampleFilter,
            PrecisionNote::FunctionStorageAssumed => Degradation::FunctionStorageAssumed,
            PrecisionNote::QuadsCountedBeforeDiscard => Degradation::QuadsCountedBeforeDiscard,
            PrecisionNote::ViewBaseTexelOffset => Degradation::ViewBaseTexelOffset,
            PrecisionNote::DeviceColorConversion => Degradation::DeviceColorConversion,
            PrecisionNote::DeviceInterpolation => Degradation::DeviceInterpolation,
            PrecisionNote::DeviceInterpolationOffset => Degradation::DeviceInterpolationOffset,
            PrecisionNote::ExtrapolatedInterpolation => Degradation::ExtrapolatedInterpolation,
            PrecisionNote::QuadsSplitByClipping => Degradation::QuadsSplitByClipping,
            PrecisionNote::VertexSimdgroupGrouping => Degradation::VertexSimdgroupGrouping,
            PrecisionNote::FragmentSimdgroupGrouping => Degradation::FragmentSimdgroupGrouping,
            PrecisionNote::BlendedWriteRounding => Degradation::BlendedWriteRounding,
            PrecisionNote::RotatedBlendRounding => Degradation::RotatedBlendRounding,
            PrecisionNote::LinkedSamplerNumbering => Degradation::LinkedSamplerNumbering,
            PrecisionNote::ComputedSampleDepth => Degradation::ComputedSampleDepth,
            PrecisionNote::ArgumentPixelSamplerScaling => Degradation::ArgumentPixelSamplerScaling,
            PrecisionNote::HalfOutputPrecision => Degradation::HalfOutputPrecision,
            PrecisionNote::ExtendedRangeBlendSource => Degradation::ExtendedRangeBlendSource,
            PrecisionNote::DiscardDropsSimdgroupStores => Degradation::DiscardDropsSimdgroupStores,
            PrecisionNote::HelperBranchDropsSimdgroupStores => {
                Degradation::HelperBranchDropsSimdgroupStores
            }
            PrecisionNote::ExtendedRangeFloatBlendSource => {
                Degradation::ExtendedRangeFloatBlendSource
            }
            PrecisionNote::AnisotropicFilter => Degradation::AnisotropicFilter,
            PrecisionNote::DeviceAnisotropicFilter => Degradation::DeviceAnisotropicFilter,
            PrecisionNote::DevicePointSize => Degradation::DevicePointSize,
            PrecisionNote::ClippedCoverageSplit => Degradation::ClippedCoverageSplit,
            PrecisionNote::LoopedSampleOutputs => Degradation::LoopedSampleOutputs,
            PrecisionNote::HalfOutputFastMathBlend => Degradation::HalfOutputFastMathBlend,
            PrecisionNote::DeviceSampling => Degradation::DeviceSampling,
            PrecisionNote::NativeTranscendentals => Degradation::NativeTranscendentals,
            PrecisionNote::NativeDivision => Degradation::NativeDivision,
            PrecisionNote::NativeFma => Degradation::NativeFma,
            PrecisionNote::DeviceTexelConversion => Degradation::DeviceTexelConversion,
            PrecisionNote::DeviceVertexConversion => Degradation::DeviceVertexConversion,
            PrecisionNote::NativeHalfConversion => Degradation::NativeHalfConversion,
            PrecisionNote::NativeHalfQuantize => Degradation::NativeHalfQuantize,
            PrecisionNote::NativeNanEncoding => Degradation::NativeNanEncoding,
            PrecisionNote::NativeMinMax => Degradation::NativeMinMax,
            PrecisionNote::NativeDot => Degradation::NativeDot,
            PrecisionNote::NativeMix => Degradation::NativeMix,
            PrecisionNote::DevicePackConversion => Degradation::DevicePackConversion,
            PrecisionNote::ColorAttachmentGroups => Degradation::ColorAttachmentGroups,
            PrecisionNote::TessellationTriangleOrder => Degradation::TessellationTriangleOrder,
            PrecisionNote::UnmeasuredGroupSpeculation => Degradation::UnmeasuredGroupSpeculation,
            PrecisionNote::NotMipmappedAnisotropicFilter => {
                Degradation::NotMipmappedAnisotropicFilter
            }
            PrecisionNote::DeviceSampleLevel => Degradation::DeviceSampleLevel,
            PrecisionNote::CoverageReadsRoundedAlpha => Degradation::CoverageReadsRoundedAlpha,
            PrecisionNote::DeviceAlphaCoverage => Degradation::DeviceAlphaCoverage,
            PrecisionNote::UnmeasuredFetchBlend => Degradation::UnmeasuredFetchBlend,
            PrecisionNote::UnorderedFetchWithinDraw => Degradation::UnorderedFetchWithinDraw,
            PrecisionNote::NativeLogicOperation => Degradation::NativeLogicOperation,
            PrecisionNote::NativeFloat8 => Degradation::NativeFloat8,
            PrecisionNote::DeviceDepthBias => Degradation::DeviceDepthBias,
            PrecisionNote::SwizzledOpaqueBlackBorder => Degradation::SwizzledOpaqueBlackBorder,
            PrecisionNote::CappedLodBias => Degradation::CappedLodBias,
            PrecisionNote::SnormMinimumKept => Degradation::SnormMinimumKept,
            PrecisionNote::DepthDenormals => Degradation::DepthDenormals,
            PrecisionNote::DeviceFilteredMipmaps => Degradation::DeviceFilteredMipmaps,
            PrecisionNote::DeviceRoundedResolve => Degradation::DeviceRoundedResolve,
            PrecisionNote::CompressedTexelDecode => Degradation::CompressedTexelDecode,
            PrecisionNote::SharedExponentKept => Degradation::SharedExponentKept,
            PrecisionNote::DeviceReconvergence => Degradation::ImplementationDefinedReconvergence,
        }
    }
}
