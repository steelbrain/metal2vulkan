//! Comparison classes of authored observations.
//!
//! A candidate observation is compared with its Metal observation under one
//! [`ComparisonClass`], derived mechanically from what the executed modules record: their
//! `precision_notes` and `StrategyCensus` ([`ComparisonClass::derive`]). Nothing authors or
//! chooses a class. Byte equality is a match under every class; a class only ever widens what
//! counts as a match on lanes of a structurally known float format ([`ObservationLanes`]), and
//! only by a bound that follows from the note ([`allowance`]). A note whose effect on a stored
//! value no table bounds makes the class [`ComparisonClass::Unbounded`]: a difference stays a
//! mismatch, recorded with the notes that may explain it, and is never accepted.
//!
//! The class is per observation, not per lane: every lane of the declared format gets the
//! tolerance, including lanes the noted operation did not produce, and a tolerance says nothing
//! about values computed further from the noted result. Both make a class looser than the note
//! for some lanes; neither is used to absorb a difference the table does not state.

use crate::case::{AuthoredCase, OutputSelection, TextureFormat};
use metal2vulkan::pipeline::{PrecisionNote, StrategyCensus};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;

/// How a candidate observation compares with its Metal observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonClass {
    /// Byte for byte: no executed module records a precision note.
    Exact,
    /// Float lanes may differ as the table states; every other byte is exact.
    UlpBounded(UlpTable),
    /// A float reduction or prefix sum runs in the device's association order
    /// (`PrecisionNote::ReductionOrder`): lanes of each float format may differ by the
    /// reassociation bound of the widest sum of that format the census records
    /// ([`reduction_table`]), together with the table of the other notes.
    ReductionOrderTolerant(UlpTable),
    /// These notes accept a difference that no table bounds on the stored value: only byte
    /// equality is a match.
    Unbounded(BTreeSet<PrecisionNote>),
}

/// Per float format, how far a candidate lane may be from the Metal lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UlpTable {
    pub half: LaneTolerance,
    pub float: LaneTolerance,
}

/// The tolerance of one float lane format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneTolerance {
    /// The largest distance, in representable values, between two finite lanes of the same sign.
    pub ulps: u32,
}

/// What one precision note allows on an observation: one row of the class derivation table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allowance {
    /// Lanes may differ as the table states.
    Lanes(UlpTable),
    /// A float reduction reassociates; the bound follows from the census ([`reduction_table`]).
    Reduction,
    /// No bound on the stored value follows from the note.
    Unbounded,
    /// The note records a device behavior every computation of its format is exposed to: it
    /// neither widens a lane nor explains a difference, and the class is derived as if it were
    /// absent. Were it unbounded, every `float` observation would be explained by it and lose the
    /// tolerance its other notes derive.
    Recorded,
}

/// A little-endian IEEE lane format an observation can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatLane {
    Half,
    Float,
}

/// How an observation's bytes split into lanes, as far as the case states it structurally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationLanes {
    /// Bytes of no declared element type (buffer ranges, imageblock planes, the empty output):
    /// only byte equality compares them.
    Untyped,
    /// Integer or normalized texels, and stencil bytes: only byte equality compares them.
    Integer,
    /// Tightly packed little-endian lanes of one float format.
    Float(FloatLane),
}

impl LaneTolerance {
    /// Nothing but equal bits.
    pub const EXACT: LaneTolerance = LaneTolerance { ulps: 0 };

    /// The looser of `self` and `other`.
    fn widen(self, other: LaneTolerance) -> LaneTolerance {
        LaneTolerance {
            ulps: self.ulps.max(other.ulps),
        }
    }
}

impl UlpTable {
    /// Equal bits in every format.
    pub const EXACT: UlpTable = UlpTable {
        half: LaneTolerance::EXACT,
        float: LaneTolerance::EXACT,
    };

    fn half(tolerance: LaneTolerance) -> UlpTable {
        UlpTable {
            half: tolerance,
            float: LaneTolerance::EXACT,
        }
    }

    fn widen(self, other: UlpTable) -> UlpTable {
        UlpTable {
            half: self.half.widen(other.half),
            float: self.float.widen(other.float),
        }
    }

    fn lane(&self, format: FloatLane) -> LaneTolerance {
        match format {
            FloatLane::Half => self.half,
            FloatLane::Float => self.float,
        }
    }
}

/// The class derivation table: what each precision note allows.
///
/// - `F16FmaDoubleRounding`, `F16IntConvDoubleRounding`: rounding the exact result to `f32` and
///   then to `half` lands on one of the two `half` values around it, as correct rounding does, so
///   the two differ by at most one `half` ulp.
/// - `ReductionOrder`: [`Allowance::Reduction`].
/// - `DeviceRayIntersection`: unbounded until Metal readings bound the device-computed hit
///   distance and barycentrics and settle the translation's conventions; a miss, a culled ray, or a
///   facing convention that differs changes whole fields, not lanes.
/// - `FastMathRewrites`: unbounded. The AIR lets a compiler regroup or fuse the arithmetic
///   (LLVM `reassoc`, `contract`, `llvm.fmuladd`), and two groupings of a sum of mixed signs, or
///   a fused and an unfused multiply-add, can differ by any amount relative to the result.
/// - `TessellationWinding`, `TessellationFactorEdgeCases`, `TessellationFactorScale`,
///   `TessellationFactorInstance`: unbounded until Metal readings settle a tessellation
///   pipeline's winding and factor rules: another vertex order, level, or culled patch changes
///   whole primitives, not lanes.
/// - `TessellationTriangleOrder`: unbounded: the device's tessellator's triangle order, first
///   corners and ring triangulation change flat varyings, interpolated values and blend order by
///   whole primitives, not lanes.
/// - `UnmeasuredGroupSpeculation`: unbounded: a quad or simdgroup operation answering over other
///   lanes reads another lane's value, which differs from Metal's by any amount.
/// - `NotMipmappedAnisotropicFilter`: unbounded: another anisotropic footprint reads other texels,
///   which differ from Metal's by up to the texture's range.
/// - `DeviceSampleLevel`: unbounded: another level or filter near a boundary reads other texels,
///   which differ from Metal's by up to the texture's range, and an LOD query answers the device's
///   level.
/// - `UnmeasuredFetchBlend`: unbounded until Metal readings bound its blend arithmetic for the
///   format: the shader's `float` blend is not Metal's, whose arithmetic for it is not measured.
/// - `Fp8Value`: unbounded until a Metal device that implements 8-bit float conversions reads
///   them: Metal on Apple M3 returns the carrier byte's integer extension, which differs from the
///   format's value by any amount.
/// - `MultisamplePosition`: unbounded: a value interpolated at another position in the pixel
///   differs by the attribute's slope over that distance, which the note does not bound.
/// - `RasterizationRateMap`: unbounded: a rate-mapped pass drawn by regions over a map that is not
///   the identity records it: the regions draw each tile's exact line where Apple's decoder reads
///   its fixed-point one, a fragment's screen point differs from Metal's by up to about 0.025
///   pixels where the device keeps the viewport's fractions, and an axis's last physical line may
///   be shaded where Metal leaves it, so a coordinate the function rounds or compares can land on
///   another pixel, and what it reads there differs by any amount. A shader's decoder records
///   nothing: it is Apple's GPU decoder over Metal's own parameter data.
/// - `BicubicFilterArithmetic`: unbounded in ulps. An emulated filter sums its taps in the
///   device's arithmetic under the Catmull-Rom weights computed from the fraction, where Metal sums
///   its fixed-point weights exactly onto the format's grid: a lane may differ by a few grid steps
///   and by the weights' distance times the taps, which cancellation makes any multiple of the
///   result's ulp. The comparator cannot see a lane's footprint, so it cannot check the
///   near-tie conditions; it hands back the narrowest bound it can check.
/// - `SparseResidency`: unbounded: the sparse form's residency and unmapped texels are the
///   device's, which no Metal reading has compared; another residency answer flips a whole status
///   word, and an unmapped texel can read anything where the device is not strict.
/// - `RestrictedDepthRange`: unbounded: a depth texel outside `[0, 1]` uploaded without
///   `VK_EXT_depth_range_unrestricted` is undefined, read back clamped (RADV) or kept.
/// - `CubeGradientFootprint`: unbounded: Apple's approximate length and truncated level against the
///   device's can select another level or filter near a boundary, which changes whole texels, not
///   lanes.
/// - `HalfTexelConversion`: unbounded: a format whose conversion to `half` is not measured keeps
///   the device's quantize of its `f32` texel, so a lane can differ from Metal's by any amount (an
///   sRGB texel's decode).
/// - `HalfSampleFilter`: unbounded: a filtered `half` sample of a two-plane YCbCr texture through
///   Metal's own sampler (MoltenVK) filters the converted texels as an `RGBA16Float` or
///   `RGBA32Float` texture's, off the format's grid; one step of the grid was measured on one chip
///   and format, which no tolerance is derived from.
/// - `SrgbTexelDecode`: unbounded: an 8-bit sRGB texel the device decodes where Metal decodes its
///   raw code through a twelve-bit table; Vulkan bounds the device's decode only by its precision
///   rules, and Metal's `float` gathers of sRGB texels are not measured.
/// - `FloatSampleFilter`: unbounded: a filtered `float` sample of a two-plane YCbCr texture through
///   Metal's own sampler (MoltenVK) filters the converted texels as an `RGBA16Float` or
///   `RGBA32Float` texture's, off the format's grid; one step of the grid was measured on one chip
///   and format, which no tolerance is derived from.
/// - `QuadsCountedBeforeDiscard`: unbounded: a non-quad simdgroup operation before a discard, in a
///   shape whose placement by Metal's compiler is not known, takes, among its lanes, the quads the
///   discard drops, which moves a count, a ballot, or a reduction by whole lanes.
/// - `DeviceInterpolation`: unbounded: the device's interpolation of a varying is measured up to
///   about `3e-4` off Metal's on the probes (more where the varying's range is wider), and the
///   interpolated value is ordinary data later words use, so no word's difference is bounded by
///   it.
/// - `QuadsSplitByClipping`: unbounded: a quad a clipping seam crosses runs once per piece, which
///   moves a count, a ballot, or a reduction by whole lanes.
/// - `VertexSimdgroupGrouping`: unbounded: a vertex function's simdgroup holds the device's group
///   of vertices (ANV's 8, interleaved, where Apple's GPUs group 32 consecutive ones), which moves
///   a sum, a ballot, or a shuffle by whole vertices.
/// - `FragmentSimdgroupGrouping`: unbounded: a fragment function's simdgroup holds the pixels the
///   device's rasterizer packs together, not Apple's, which moves a sum, a ballot, or a shuffle by
///   whole pixels.
/// - `BlendedWriteRounding`: unbounded: a blended half-float channel under
///   `textureWriteRoundingMode` 2 stores the device's rounding of the blend's result, which no
///   Vulkan blend states, where Metal rounds it to nearest even.
/// - `CoverageReadsRoundedAlpha`: unbounded: a fragment whose alpha lies near a coverage
///   threshold may cover one sample more or fewer than Metal's, which moves every channel a
///   resolve or a later blend reads there.
/// - `DeviceAlphaCoverage`: unbounded: the device's alpha to coverage may cover other samples
///   than Metal's, a whole fragment stored or dropped at one sample, which moves every channel a
///   resolve or a later blend reads there.
/// - `RotatedBlendRounding`: unbounded: an `ARGB8Unorm` attachment blended in none of the measured
///   forms may store a channel one code apart from Metal's, which moves every later word that reads
///   it.
/// - `DeviceDenormals`, `DeviceNanEncoding`: [`Allowance::Recorded`]. Metal flushes `float`
///   denormals and canonicalizes NaNs, a device need not, and every `float` operation the device
///   performs is exposed to both; a `half` texel's NaN is the device's decode where Metal hands
///   back the texel's own: a difference they cause stays a mismatch for review, never accepted or
///   explained by the class. C3 tells which differing values they cause ([`value_difference`]).
/// - `FunctionStorageAssumed`: [`Allowance::Recorded`]. A vertex or fragment function's private
///   memory passes the floor of a device not measured, resting on no measurement; a difference is
///   a mismatch for review, never accepted or explained by the note.
/// - `ViewBaseTexelOffset`: [`Allowance::Recorded`]. Only a view whose first level lies above its
///   parent's base level reads otherwise than Metal, which no authored case binds: a difference is
///   a mismatch for review.
/// - `DeviceColorConversion`: [`Allowance::Recorded`]. A color channel left to the device's
///   conversion may store the neighbouring code near a tie; a difference is a mismatch for review,
///   never accepted or explained by the note.
/// - `LinkedSamplerNumbering`: [`Allowance::Recorded`]. Only sampler resource IDs may differ, no
///   shader value; a difference is a mismatch for review.
/// - `ColorAttachmentGroups`: [`Allowance::Recorded`]. Only a group past the first may differ, at
///   a discard or depth edge where the driver compiled the groups' variants apart; a difference is
///   a mismatch for review.
/// - Every other note, and any note this table does not know, bounds nothing on the stored
///   value: Vulkan's transcendental and arithmetic precision is stated per operation and input
///   range (absolute error for `sin`/`cos`, inherited bounds for `pow` and an unfused `Fma`, where
///   cancellation makes the relative error unbounded), a device may drop denormals, signed
///   zeros, infinities, and NaNs on inputs, and the sampling, footprint, and ordering notes have
///   no numeric bound at all.
pub fn allowance(note: PrecisionNote) -> Allowance {
    use PrecisionNote as N;
    match note {
        N::F16FmaDoubleRounding | N::F16IntConvDoubleRounding => {
            Allowance::Lanes(UlpTable::half(LaneTolerance { ulps: 1 }))
        }
        N::ReductionOrder => Allowance::Reduction,
        N::DeviceDenormals
        | N::DeviceNanEncoding
        | N::FunctionStorageAssumed
        | N::ViewBaseTexelOffset => Allowance::Recorded,
        N::DeviceColorConversion => Allowance::Recorded,
        N::LinkedSamplerNumbering | N::ColorAttachmentGroups => Allowance::Recorded,
        N::GradFromBias
        | N::DeviceTranscendentals
        | N::PixelSamplerScaling
        | N::ArgumentPixelSamplerScaling
        | N::DeviceArithmetic
        | N::DeviceSignedZeroInfNan
        | N::DeviceRayIntersection
        | N::FastMathRewrites
        | N::TessellationWinding
        | N::TessellationFactorEdgeCases
        | N::TessellationFactorScale
        | N::TessellationFactorInstance
        | N::TessellationTriangleOrder
        | N::UnmeasuredGroupSpeculation
        | N::NotMipmappedAnisotropicFilter
        | N::DeviceSampleLevel
        | N::UnmeasuredFetchBlend
        | N::Fp8Value
        | N::MultisamplePosition
        | N::RasterizationRateMap
        | N::BicubicFilterArithmetic
        | N::SparseResidency
        | N::RestrictedDepthRange
        | N::CubeGradientFootprint
        | N::HalfTexelConversion
        | N::HalfSampleFilter
        | N::SrgbTexelDecode
        | N::FloatSampleFilter
        | N::QuadsCountedBeforeDiscard
        | N::DeviceInterpolation
        | N::ComputedSampleDepth
        | N::QuadsSplitByClipping
        | N::VertexSimdgroupGrouping
        | N::FragmentSimdgroupGrouping
        | N::BlendedWriteRounding
        | N::CoverageReadsRoundedAlpha
        | N::DeviceAlphaCoverage
        | N::RotatedBlendRounding
        | N::HalfOutputPrecision
        | N::ExtendedRangeBlendSource
        | N::ExtendedRangeFloatBlendSource
        | N::AnisotropicFilter
        | N::DeviceAnisotropicFilter
        | N::DevicePointSize
        | N::ClippedCoverageSplit
        | N::LoopedSampleOutputs
        | N::HalfOutputFastMathBlend
        | N::DeviceSampling
        | N::NativeTranscendentals
        | N::NativeDivision
        | N::NativeFma
        | N::DeviceTexelConversion
        | N::DeviceVertexConversion
        | N::NativeHalfConversion
        | N::NativeHalfQuantize
        | N::NativeNanEncoding
        | N::NativeMinMax
        | N::NativeDot
        | N::NativeMix
        | N::DevicePackConversion
        | N::DiscardDropsSimdgroupStores
        | N::HelperBranchDropsSimdgroupStores => Allowance::Unbounded,
    }
}

/// How one differing value shows a precision note's difference from Metal, for the notes whose
/// difference can be told value by value. C3 judges each value a reading differs in against these
/// beside Apple's bound (`c3::bound`): a value that shows the difference of a note its row records
/// is a near-equivalent of Metal's answer, open until an exact path exists (the user's decision of
/// 2026-10-03). The comparison class does not read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDifference {
    /// Both answers are NaNs, so only their encodings differ: Metal answers `0x7fc00000`, a
    /// device need not.
    NanEncoding,
    /// A subnormal of the value's format is an operand, an operation's exact value, or an
    /// answer: Metal flushes `float` denormals, a device need not.
    Denormal,
}

/// The [`ValueDifference`] of `note`: `DeviceNanEncoding` and `DeviceDenormals` each have one.
/// Every other note has none: C3 judges it for the whole row
/// ([`crate::c3::DEFERRED_NEAR_EQUIVALENTS`]), per operation (`DeviceArithmetic`,
/// [`crate::c3::bound::near_equivalent`]), against Apple's bound alone, or not at all.
pub fn value_difference(note: PrecisionNote) -> Option<ValueDifference> {
    match note {
        PrecisionNote::DeviceNanEncoding => Some(ValueDifference::NanEncoding),
        PrecisionNote::DeviceDenormals => Some(ValueDifference::Denormal),
        _ => None,
    }
}

/// The float format a census reduction sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumFormat {
    Half,
    BFloat,
    Float,
    Double,
}

/// The census strategies that sum or multiply float lanes in the device's association order
/// (`PrecisionNote::ReductionOrder`), the format each combines, and the number of terms its
/// widest sum or product combines: a simdgroup is 32 lanes. A simdgroup float sum or product and a
/// simdgroup float prefix sum or product (whose last lane combines every lane) are the strategies
/// that record the note alone, and the census names each by its format; a quad's sums, products and
/// prefix scans take Metal's order and record none. The matrix and AGX multiply-adds that also
/// record it record `DeviceArithmetic` with it.
pub const REDUCTIONS: &[(&str, SumFormat, u32)] = &[
    ("simd.reduce.fadd.half", SumFormat::Half, 32),
    ("simd.reduce.fadd.bfloat", SumFormat::BFloat, 32),
    ("simd.reduce.fadd.float", SumFormat::Float, 32),
    ("simd.reduce.fadd.double", SumFormat::Double, 32),
    ("simd.scan.fadd.half", SumFormat::Half, 32),
    ("simd.scan.fadd.bfloat", SumFormat::BFloat, 32),
    ("simd.scan.fadd.float", SumFormat::Float, 32),
    ("simd.scan.fadd.double", SumFormat::Double, 32),
    ("simd.reduce.fmul.half", SumFormat::Half, 32),
    ("simd.reduce.fmul.bfloat", SumFormat::BFloat, 32),
    ("simd.reduce.fmul.float", SumFormat::Float, 32),
    ("simd.reduce.fmul.double", SumFormat::Double, 32),
    ("simd.scan.fmul.half", SumFormat::Half, 32),
    ("simd.scan.fmul.bfloat", SumFormat::BFloat, 32),
    ("simd.scan.fmul.float", SumFormat::Float, 32),
    ("simd.scan.fmul.double", SumFormat::Double, 32),
];

/// The reassociation table of the float sums among `strategies` (census names): per lane
/// format, the bound of the widest sum of that format ([`reorder_ulps`]), and equal bits in a
/// format no sum produces. `None` when no strategy is a sum [`REDUCTIONS`] lists, or when one
/// sums `bfloat` or `double`, which no observation holds as lanes: the note then bounds nothing.
pub fn reduction_table<'a>(strategies: impl IntoIterator<Item = &'a str>) -> Option<UlpTable> {
    let mut half = 0;
    let mut float = 0;
    let mut any = false;
    for name in strategies {
        let Some(&(_, format, terms)) = REDUCTIONS.iter().find(|(known, ..)| *known == name) else {
            continue;
        };
        any = true;
        match format {
            SumFormat::Half => half = half.max(terms),
            SumFormat::Float => float = float.max(terms),
            SumFormat::BFloat | SumFormat::Double => return None,
        }
    }
    let lane = |terms, format| LaneTolerance {
        ulps: reorder_ulps(terms, format),
    };
    any.then(|| UlpTable {
        half: lane(half, FloatLane::Half),
        float: lane(float, FloatLane::Float),
    })
}

/// How many representable values of `format` two association orders of a `terms`-term float
/// sum of same-sign values, or of a `terms`-term float product of any signs, may lie apart.
///
/// Each order is within `γ(n−1)·|S|` of the exact sum or product `S` (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, §4.2 and §3.1: a product of `n` terms rounds `n − 1`
/// times, each within `u` relatively), `γ(k) = k·u / (1 − k·u)` with unit roundoff `u`
/// (`2⁻²⁴` for `float`, `2⁻¹¹` for `half`), so the two are within `2·γ·|S|` of each other, and
/// neither is smaller in magnitude than `(1 − γ)·|S|`. Adjacent values of magnitude at least `m`
/// are at least `u·m` apart, so at most `⌊2·γ / (u·(1 − γ))⌋` steps separate the two. Sums of
/// mixed signs can cancel and are not bounded: the class then reports a difference beyond this as
/// a mismatch. A product does not cancel.
///
/// The bound assumes both devices round every addition and multiplication to nearest (ties to
/// even) and keep subnormal results. Vulkan fixes neither without the `RoundingModeRTE` and
/// `DenormPreserve` execution modes, which the product does not declare: a device that rounds
/// toward zero or flushes subnormal sums or products can differ by more, and that difference is
/// reported as a mismatch, never absorbed.
pub fn reorder_ulps(terms: u32, format: FloatLane) -> u32 {
    if terms <= 1 {
        return 0;
    }
    let u = match format {
        FloatLane::Half => 2f64.powi(-11),
        FloatLane::Float => 2f64.powi(-24),
    };
    let k = f64::from(terms - 1);
    let gamma = k * u / (1.0 - k * u);
    (2.0 * gamma / (u * (1.0 - gamma))).floor() as u32
}

impl ComparisonClass {
    /// The class of observations of modules that record `notes` and `strategies` (census
    /// names): the union over every executed stage.
    pub fn derive<'a>(
        notes: impl IntoIterator<Item = PrecisionNote>,
        strategies: impl IntoIterator<Item = &'a str>,
    ) -> ComparisonClass {
        let notes = notes
            .into_iter()
            .filter(|note| allowance(*note) != Allowance::Recorded)
            .collect::<BTreeSet<_>>();
        let mut table = UlpTable::EXACT;
        let mut unbounded = BTreeSet::new();
        let mut reduction = false;
        for &note in &notes {
            match allowance(note) {
                Allowance::Lanes(lanes) => table = table.widen(lanes),
                Allowance::Reduction => reduction = true,
                Allowance::Unbounded => {
                    unbounded.insert(note);
                }
                Allowance::Recorded => {}
            }
        }
        if reduction {
            match reduction_table(strategies) {
                Some(reordered) => table = table.widen(reordered),
                // The note names no sum this table bounds: nothing bounds it.
                None => {
                    unbounded.insert(PrecisionNote::ReductionOrder);
                }
            }
        }
        if !unbounded.is_empty() {
            ComparisonClass::Unbounded(unbounded)
        } else if notes.is_empty() {
            ComparisonClass::Exact
        } else if reduction {
            ComparisonClass::ReductionOrderTolerant(table)
        } else {
            ComparisonClass::UlpBounded(table)
        }
    }

    /// The lane table of a class that has one.
    fn table(&self) -> Option<&UlpTable> {
        match self {
            ComparisonClass::UlpBounded(table) | ComparisonClass::ReductionOrderTolerant(table) => {
                Some(table)
            }
            ComparisonClass::Exact | ComparisonClass::Unbounded(_) => None,
        }
    }

    /// Whether a class can accept bytes that differ (on typed float lanes).
    pub fn tolerates_differences(&self) -> bool {
        self.table().is_some()
    }

    /// Whether `candidate` matches `golden` under this class for an observation of `lanes`:
    /// equal bytes always do; otherwise every lane must be of a float format this class has a
    /// table for, and within it.
    pub fn matches(&self, lanes: ObservationLanes, golden: &[u8], candidate: &[u8]) -> bool {
        if golden == candidate {
            return true;
        }
        let (Some(table), ObservationLanes::Float(format)) = (self.table(), lanes) else {
            return false;
        };
        let width = match format {
            FloatLane::Half => 2,
            FloatLane::Float => 4,
        };
        if golden.len() != candidate.len() || !golden.len().is_multiple_of(width) {
            return false;
        }
        let tolerance = table.lane(format);
        golden
            .chunks_exact(width)
            .zip(candidate.chunks_exact(width))
            .all(|(g, c)| lane_within(format, tolerance, bits(g), bits(c)))
    }
}

/// The names of a census's strategies.
pub fn census_names(census: &StrategyCensus) -> impl Iterator<Item = &str> {
    census.entries.keys().map(String::as_str)
}

fn bits(lane: &[u8]) -> u32 {
    lane.iter()
        .rev()
        .fold(0, |bits, &byte| (bits << 8) | u32::from(byte))
}

/// Whether candidate lane `c` is within `tolerance` of Metal lane `g`, both raw bits of
/// `format`. Infinities and NaNs match only bit for bit, and a zero differs from the other signed
/// zero.
pub fn lane_within(format: FloatLane, tolerance: LaneTolerance, g: u32, c: u32) -> bool {
    if g == c {
        return true;
    }
    let (sign, exponent, magnitude) = match format {
        FloatLane::Half => (0x8000, 0x7c00, 0x7fff),
        FloatLane::Float => (0x8000_0000, 0x7f80_0000, 0x7fff_ffff),
    };
    let finite = |x: u32| x & exponent != exponent;
    if !finite(g) || !finite(c) {
        return false;
    }
    g & sign == c & sign && (g & magnitude).abs_diff(c & magnitude) <= tolerance.ulps
}

impl ObservationLanes {
    /// The lanes of `case`'s selected output: the format of the texture, render target, or depth
    /// attachment it reads, or [`ObservationLanes::Untyped`] for every output whose bytes carry
    /// no declared element type.
    pub fn of(case: &AuthoredCase) -> ObservationLanes {
        let format = match &case.output {
            OutputSelection::Texture { binding, .. } => case
                .textures
                .iter()
                .find(|texture| texture.binding == *binding)
                .map(|texture| texture.format),
            OutputSelection::TextureArrayElement { binding, .. } => case
                .texture_arrays
                .iter()
                .find(|array| array.binding == *binding)
                .map(|array| array.format),
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                ..
            } => case
                .argument_buffer_textures
                .iter()
                .find(|texture| {
                    texture.buffer_binding == *buffer_binding
                        && texture.field_offset == *field_offset
                })
                .map(|texture| texture.format),
            OutputSelection::RenderTarget { index, .. } => case
                .render_targets
                .iter()
                .find(|target| target.index == *index)
                .map(|target| target.format),
            // Depth is read back as tightly packed little-endian `f32` values.
            OutputSelection::Depth { .. } => Some(TextureFormat::Depth32Float),
            OutputSelection::Stencil { .. } => return ObservationLanes::Integer,
            OutputSelection::None
            | OutputSelection::Buffer { .. }
            | OutputSelection::ArgumentBufferBuffer { .. }
            | OutputSelection::DeviceBufferArrayElement { .. }
            | OutputSelection::FragmentImageblock { .. } => None,
        };
        format.map_or(ObservationLanes::Untyped, ObservationLanes::of_format)
    }

    /// The lanes of texels of `format`.
    pub fn of_format(format: TextureFormat) -> ObservationLanes {
        use TextureFormat as F;
        match format {
            F::R16Float | F::Rg16Float | F::Rgba16Float => ObservationLanes::Float(FloatLane::Half),
            F::R32Float | F::Rg32Float | F::Rgba32Float | F::Depth32Float => {
                ObservationLanes::Float(FloatLane::Float)
            }
            F::R8Unorm
            | F::Rgba8Unorm
            | F::Rgba8Uint
            | F::Rgba8Sint
            | F::R16Uint
            | F::Rgba16Uint
            | F::Rgba16Sint
            | F::R32Uint
            | F::R32Sint
            | F::Rgba32Uint
            | F::Rgba32Sint
            // A `ulong` texel is its two 32-bit words.
            | F::Rg32Uint
            // Normalized and sRGB texels are stored as integer codes.
            | F::R8Snorm
            | F::Rgba8Snorm
            | F::R16Unorm
            | F::Rgba16Unorm
            | F::R16Snorm
            | F::Rg16Snorm
            | F::Rgba16Snorm
            | F::Rgba8UnormSrgb
            | F::Bgra8Unorm
            | F::Bgra8UnormSrgb
            | F::R8UnormSrgb
            | F::Rg8Unorm
            | F::Rg8UnormSrgb
            | F::Rg8Snorm
            | F::Rg16Unorm
            | F::A8Unorm
            | F::B5g6r5Unorm
            | F::A1bgr5Unorm
            | F::Abgr4Unorm
            | F::Bgr5a1Unorm
            | F::Rgb10a2Unorm
            | F::Bgr10a2Unorm
            | F::Rg11b10Float
            | F::Rgb9e5Float => ObservationLanes::Integer,
        }
    }
}
