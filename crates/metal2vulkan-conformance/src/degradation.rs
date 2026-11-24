//! The degrade-notice codes the suite's cases name: `metal2vulkan-degrade-<kind>`, one per kind,
//! and the kind each of the runtime's precision notes announces as.
//!
//! A case loosens a field by naming a code ([`crate::allowance`]), and the comparison holds it
//! there only where the run announced that code. The runtime's side announces the codes of
//! the precision notes its states, textures and sampler states carry
//! ([`Degradation::from_precision_note`]), and every kind here is one of those notes'.

use metal2vulkan::pipeline::PrecisionNote;

/// A kind of degrade notice, by its code ([`Degradation::code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Degradation {
    /// `metal2vulkan-degrade-device-rounded-resolve`.
    DeviceRoundedResolve,
    /// `metal2vulkan-degrade-device-reconvergence`.
    DeviceReconvergence,
    /// `metal2vulkan-degrade-simdgroup-access-order`.
    SimdgroupAccessOrder,
    /// `metal2vulkan-degrade-device-filtered-mipmaps`.
    DeviceFilteredMipmaps,
    /// `metal2vulkan-degrade-capped-lod-bias`.
    CappedLodBias,
    /// `metal2vulkan-degrade-snorm-minimum-kept`.
    SnormMinimumKept,
    /// `metal2vulkan-degrade-shared-exponent-kept`.
    SharedExponentKept,
    /// `metal2vulkan-degrade-compressed-texel-decode`.
    CompressedTexelDecode,
    /// `metal2vulkan-degrade-depth-denormals`.
    DepthDenormals,
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
    /// `metal2vulkan-degrade-srgb-texel-decode`.
    SrgbTexelDecode,
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
    /// `metal2vulkan-degrade-fragment-simdgroup-grouping`.
    FragmentSimdgroupGrouping,
    /// `metal2vulkan-degrade-blended-write-rounding`.
    BlendedWriteRounding,
    /// `metal2vulkan-degrade-rotated-blend-rounding`.
    RotatedBlendRounding,
    /// `metal2vulkan-degrade-linked-sampler-numbering`.
    LinkedSamplerNumbering,
    /// `metal2vulkan-degrade-argument-pixel-sampler-scaling`.
    ArgumentPixelSamplerScaling,
    /// `metal2vulkan-degrade-half-output-precision`.
    HalfOutputPrecision,
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
    /// `metal2vulkan-degrade-tessellation-triangle-order`.
    TessellationTriangleOrder,
    /// `metal2vulkan-degrade-unmeasured-group-speculation`.
    UnmeasuredGroupSpeculation,
    /// `metal2vulkan-degrade-device-sample-level`.
    DeviceSampleLevel,
    /// `metal2vulkan-degrade-coverage-reads-rounded-alpha`.
    CoverageReadsRoundedAlpha,
    /// `metal2vulkan-degrade-device-alpha-coverage`.
    DeviceAlphaCoverage,
    /// `metal2vulkan-degrade-unordered-fetch-within-draw`.
    UnorderedFetchWithinDraw,
    /// `metal2vulkan-degrade-native-logic-operation`.
    NativeLogicOperation,
    /// `metal2vulkan-degrade-device-depth-bias`.
    DeviceDepthBias,
    /// `metal2vulkan-degrade-swizzled-opaque-black-border`.
    SwizzledOpaqueBlackBorder,
    /// `metal2vulkan-degrade-device-line-rasterization`.
    DeviceLineRasterization,
    /// `metal2vulkan-degrade-device-fragment-runs`.
    DeviceFragmentRuns,
    /// `metal2vulkan-degrade-device-occlusion-count`.
    DeviceOcclusionCount,
    /// `metal2vulkan-degrade-vertex-simdgroup-grouping`.
    VertexSimdgroupGrouping,
    /// `metal2vulkan-degrade-restricted-depth-upload`.
    RestrictedDepthUpload,
    /// `metal2vulkan-degrade-stencil-output-overlap`.
    StencilOutputOverlap,
}

impl Degradation {
    /// Every kind, in declaration order.
    pub const ALL: &'static [Degradation] = &[
        Degradation::DeviceRoundedResolve,
        Degradation::DeviceReconvergence,
        Degradation::SimdgroupAccessOrder,
        Degradation::DeviceFilteredMipmaps,
        Degradation::CappedLodBias,
        Degradation::SnormMinimumKept,
        Degradation::SharedExponentKept,
        Degradation::CompressedTexelDecode,
        Degradation::DepthDenormals,
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
        Degradation::BicubicFilterArithmetic,
        Degradation::SparseResidency,
        Degradation::DeviceNanEncoding,
        Degradation::RestrictedDepthRange,
        Degradation::CubeGradientFootprint,
        Degradation::HalfTexelConversion,
        Degradation::SrgbTexelDecode,
        Degradation::FunctionStorageAssumed,
        Degradation::QuadsCountedBeforeDiscard,
        Degradation::ViewBaseTexelOffset,
        Degradation::DeviceColorConversion,
        Degradation::DeviceInterpolation,
        Degradation::DeviceInterpolationOffset,
        Degradation::ExtrapolatedInterpolation,
        Degradation::QuadsSplitByClipping,
        Degradation::FragmentSimdgroupGrouping,
        Degradation::BlendedWriteRounding,
        Degradation::RotatedBlendRounding,
        Degradation::LinkedSamplerNumbering,
        Degradation::ArgumentPixelSamplerScaling,
        Degradation::HalfOutputPrecision,
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
        Degradation::TessellationTriangleOrder,
        Degradation::UnmeasuredGroupSpeculation,
        Degradation::DeviceSampleLevel,
        Degradation::CoverageReadsRoundedAlpha,
        Degradation::DeviceAlphaCoverage,
        Degradation::UnorderedFetchWithinDraw,
        Degradation::NativeLogicOperation,
        Degradation::DeviceDepthBias,
        Degradation::SwizzledOpaqueBlackBorder,
        Degradation::DeviceLineRasterization,
        Degradation::DeviceFragmentRuns,
        Degradation::DeviceOcclusionCount,
        Degradation::VertexSimdgroupGrouping,
        Degradation::RestrictedDepthUpload,
        Degradation::StencilOutputOverlap,
    ];

    /// The kind's code, `metal2vulkan-degrade-<kind>`.
    pub const fn code(self) -> &'static str {
        match self {
            Degradation::DeviceRoundedResolve => "metal2vulkan-degrade-device-rounded-resolve",
            Degradation::DeviceReconvergence => "metal2vulkan-degrade-device-reconvergence",
            Degradation::SimdgroupAccessOrder => "metal2vulkan-degrade-simdgroup-access-order",
            Degradation::DeviceFilteredMipmaps => "metal2vulkan-degrade-device-filtered-mipmaps",
            Degradation::CappedLodBias => "metal2vulkan-degrade-capped-lod-bias",
            Degradation::SnormMinimumKept => "metal2vulkan-degrade-snorm-minimum-kept",
            Degradation::SharedExponentKept => "metal2vulkan-degrade-shared-exponent-kept",
            Degradation::CompressedTexelDecode => "metal2vulkan-degrade-compressed-texel-decode",
            Degradation::DepthDenormals => "metal2vulkan-degrade-depth-denormals",
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
            Degradation::BicubicFilterArithmetic => {
                "metal2vulkan-degrade-bicubic-filter-arithmetic"
            }
            Degradation::SparseResidency => "metal2vulkan-degrade-sparse-residency",
            Degradation::DeviceNanEncoding => "metal2vulkan-degrade-device-nan-encoding",
            Degradation::RestrictedDepthRange => "metal2vulkan-degrade-restricted-depth-range",
            Degradation::CubeGradientFootprint => "metal2vulkan-degrade-cube-gradient-footprint",
            Degradation::HalfTexelConversion => "metal2vulkan-degrade-half-texel-conversion",
            Degradation::SrgbTexelDecode => "metal2vulkan-degrade-srgb-texel-decode",
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
            Degradation::FragmentSimdgroupGrouping => {
                "metal2vulkan-degrade-fragment-simdgroup-grouping"
            }
            Degradation::BlendedWriteRounding => "metal2vulkan-degrade-blended-write-rounding",
            Degradation::RotatedBlendRounding => "metal2vulkan-degrade-rotated-blend-rounding",
            Degradation::LinkedSamplerNumbering => "metal2vulkan-degrade-linked-sampler-numbering",
            Degradation::ArgumentPixelSamplerScaling => {
                "metal2vulkan-degrade-argument-pixel-sampler-scaling"
            }
            Degradation::HalfOutputPrecision => "metal2vulkan-degrade-half-output-precision",
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
            Degradation::TessellationTriangleOrder => {
                "metal2vulkan-degrade-tessellation-triangle-order"
            }
            Degradation::UnmeasuredGroupSpeculation => {
                "metal2vulkan-degrade-unmeasured-group-speculation"
            }
            Degradation::DeviceSampleLevel => "metal2vulkan-degrade-device-sample-level",
            Degradation::CoverageReadsRoundedAlpha => {
                "metal2vulkan-degrade-coverage-reads-rounded-alpha"
            }
            Degradation::DeviceAlphaCoverage => "metal2vulkan-degrade-device-alpha-coverage",
            Degradation::UnorderedFetchWithinDraw => {
                "metal2vulkan-degrade-unordered-fetch-within-draw"
            }
            Degradation::NativeLogicOperation => "metal2vulkan-degrade-native-logic-operation",
            Degradation::DeviceDepthBias => "metal2vulkan-degrade-device-depth-bias",
            Degradation::SwizzledOpaqueBlackBorder => {
                "metal2vulkan-degrade-swizzled-opaque-black-border"
            }
            Degradation::DeviceLineRasterization => {
                "metal2vulkan-degrade-device-line-rasterization"
            }
            Degradation::DeviceFragmentRuns => "metal2vulkan-degrade-device-fragment-runs",
            Degradation::DeviceOcclusionCount => "metal2vulkan-degrade-device-occlusion-count",
            Degradation::VertexSimdgroupGrouping => {
                "metal2vulkan-degrade-vertex-simdgroup-grouping"
            }
            Degradation::RestrictedDepthUpload => "metal2vulkan-degrade-restricted-depth-upload",
            Degradation::StencilOutputOverlap => "metal2vulkan-degrade-stencil-output-overlap",
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
            PrecisionNote::BicubicFilterArithmetic => Degradation::BicubicFilterArithmetic,
            PrecisionNote::SparseResidency => Degradation::SparseResidency,
            PrecisionNote::DeviceNanEncoding => Degradation::DeviceNanEncoding,
            PrecisionNote::RestrictedDepthRange => Degradation::RestrictedDepthRange,
            PrecisionNote::CubeGradientFootprint => Degradation::CubeGradientFootprint,
            PrecisionNote::HalfTexelConversion => Degradation::HalfTexelConversion,
            PrecisionNote::SrgbTexelDecode => Degradation::SrgbTexelDecode,
            PrecisionNote::FunctionStorageAssumed => Degradation::FunctionStorageAssumed,
            PrecisionNote::QuadsCountedBeforeDiscard => Degradation::QuadsCountedBeforeDiscard,
            PrecisionNote::ViewBaseTexelOffset => Degradation::ViewBaseTexelOffset,
            PrecisionNote::DeviceColorConversion => Degradation::DeviceColorConversion,
            PrecisionNote::DeviceInterpolation => Degradation::DeviceInterpolation,
            PrecisionNote::DeviceInterpolationOffset => Degradation::DeviceInterpolationOffset,
            PrecisionNote::ExtrapolatedInterpolation => Degradation::ExtrapolatedInterpolation,
            PrecisionNote::QuadsSplitByClipping => Degradation::QuadsSplitByClipping,
            PrecisionNote::FragmentSimdgroupGrouping => Degradation::FragmentSimdgroupGrouping,
            PrecisionNote::BlendedWriteRounding => Degradation::BlendedWriteRounding,
            PrecisionNote::RotatedBlendRounding => Degradation::RotatedBlendRounding,
            PrecisionNote::LinkedSamplerNumbering => Degradation::LinkedSamplerNumbering,
            PrecisionNote::ArgumentPixelSamplerScaling => Degradation::ArgumentPixelSamplerScaling,
            PrecisionNote::HalfOutputPrecision => Degradation::HalfOutputPrecision,
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
            PrecisionNote::TessellationTriangleOrder => Degradation::TessellationTriangleOrder,
            PrecisionNote::UnmeasuredGroupSpeculation => Degradation::UnmeasuredGroupSpeculation,
            PrecisionNote::DeviceSampleLevel => Degradation::DeviceSampleLevel,
            PrecisionNote::CoverageReadsRoundedAlpha => Degradation::CoverageReadsRoundedAlpha,
            PrecisionNote::DeviceAlphaCoverage => Degradation::DeviceAlphaCoverage,
            PrecisionNote::UnorderedFetchWithinDraw => Degradation::UnorderedFetchWithinDraw,
            PrecisionNote::NativeLogicOperation => Degradation::NativeLogicOperation,
            PrecisionNote::DeviceDepthBias => Degradation::DeviceDepthBias,
            PrecisionNote::SwizzledOpaqueBlackBorder => Degradation::SwizzledOpaqueBlackBorder,
            PrecisionNote::CappedLodBias => Degradation::CappedLodBias,
            PrecisionNote::SnormMinimumKept => Degradation::SnormMinimumKept,
            PrecisionNote::DepthDenormals => Degradation::DepthDenormals,
            PrecisionNote::DeviceFilteredMipmaps => Degradation::DeviceFilteredMipmaps,
            PrecisionNote::DeviceRoundedResolve => Degradation::DeviceRoundedResolve,
            PrecisionNote::CompressedTexelDecode => Degradation::CompressedTexelDecode,
            PrecisionNote::SharedExponentKept => Degradation::SharedExponentKept,
            PrecisionNote::DeviceReconvergence => Degradation::DeviceReconvergence,
            PrecisionNote::SimdgroupAccessOrder => Degradation::SimdgroupAccessOrder,
            PrecisionNote::DeviceLineRasterization => Degradation::DeviceLineRasterization,
            PrecisionNote::DeviceFragmentRuns => Degradation::DeviceFragmentRuns,
            PrecisionNote::DeviceOcclusionCount => Degradation::DeviceOcclusionCount,
            PrecisionNote::VertexSimdgroupGrouping => Degradation::VertexSimdgroupGrouping,
            PrecisionNote::RestrictedDepthUpload => Degradation::RestrictedDepthUpload,
            PrecisionNote::StencilOutputOverlap => Degradation::StencilOutputOverlap,
        }
    }
}
