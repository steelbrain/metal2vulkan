//! What the device-sampled filters share with the emulated ones the device has no form for
//! (`extended`, `heap_raw`, `bicubic`): the census words of a filter's footprint and
//! of a runtime sampler's word, the rule of an axis whose two taps read one texel ([`merged`]), and
//! the samplers that read one texel of one level ([`unfiltered`]).
//!
//! Every filtered sample of a format the device samples is the device's own
//! (`VkProperty::DeviceSampling`, recorded as `PrecisionNote::DeviceSampling`).
//!
//! **Two-plane YCbCr textures** (`YCBCR8_420_2P` and its kin). A format's view holds each channel
//! of the converted color as Metal reads it, a step `k / D` (alpha 1) on its field's grid: `2048`
//! (`PackedTexels::YCbCr2048`), `8192` (`PackedTexels::YCbCr8192`), or `8192` signed within
//! `[-2, 2]` (`PackedTexels::YCbCrExtended8192`). Metal's 2D linear sample sums the steps on the
//! grid, each term truncated; the device's sampler filters the view's texels its own way
//! (`PrecisionNote::DeviceSampling`).

use crate::air::abi::{Filter, MipFilter, SamplerState};
use crate::lower::select::{NoteOn, NoteTest};
use crate::metal::sampler::SamplerWord;
use crate::metal::SamplerMipFilter;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::bicubic::{scalar, value, Tap, Tys};

/// The census words a runtime sampler's emulation adds once per sample: the word's filter test and
/// each axis's mode test (`reaches_word`), each axis's mode and the coordinates' extraction, the
/// border color's selection, and a 3D sample's reflected-axis test, with their constants.
pub(super) const WORD_WORDS: u32 = 448;

/// The census words a runtime sampler's tap adds to a constexpr sampler's: the tap's index under
/// each of its modes (`MirrorClampToEdge`'s included), the mode's tests and selects, and its border
/// test.
pub(super) const WORD_TAP_WORDS: u32 = 160;

/// The census words of a linear 2D sample's emulation.
pub(super) const PLANE_WORDS: u32 = 576;

/// The tests of the word of the sampler bound at runtime sampler slot `n` (`NoteOn::BoundSampler`,
/// `SamplerWord`) of which one holds exactly where a sample through it filters, [`unfiltered`]'s
/// rule read from the word: a linear minification or magnification filter, anisotropy or a cubic
/// filter (any of their bits set), or a linear mip filter (the second test).
pub(super) fn filtering_word(n: u32) -> [NoteTest; 2] {
    let on = NoteOn::BoundSampler(n);
    let filters = SamplerWord::MIN_LINEAR
        | SamplerWord::MAG_LINEAR
        | SamplerWord::ANISOTROPIC
        | SamplerWord::CUBIC;
    let linear_mip = SamplerMipFilter::Linear.as_raw() << SamplerWord::MIP_FILTER_SHIFT;
    [
        NoteTest::differ(on, filters, 0),
        NoteTest::equal(on, SamplerWord::MIP_FILTER, linear_mip),
    ]
}

/// Whether a sample through sampler `s` reads one texel of one level, its value unfiltered:
/// nearest minification and magnification, no mip filter or the nearest level's, and no
/// anisotropy (a lowered sampler's reduction is the weighted average). Any other sampler filters:
/// linear minification or magnification, a linear mip filter (a blend of two levels), or
/// anisotropy.
pub(crate) fn unfiltered(s: &SamplerState) -> bool {
    s.min_filter == Filter::Nearest
        && s.mag_filter == Filter::Nearest
        && matches!(s.mip_filter, MipFilter::None | MipFilter::Nearest)
        && s.max_anisotropy <= 1
}

/// The fraction's steps `k` of an axis whose taps are `a` and `b`, or `0` where both are one
/// texel (an index clamped or reflected onto its partner's, a level one texel wide, or both taps
/// outside a border axis): Metal weighs such a pair `(256, 0)` before it rounds a 3D sample's
/// weights and before it truncates a 2D sample's terms.
pub(super) fn merged(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    k: SValue,
    [a, b]: [Tap; 2],
) -> BResult<SValue> {
    let mut same = fb.binary(BinOp::IEqual, a.0, b.0)?;
    if let (Some(x), Some(y)) = (a.1, b.1) {
        let both = fb.binary(BinOp::LogicalEqual, x, y)?;
        same = fb.binary(BinOp::LogicalAnd, same, both)?;
    }
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    fb.select(same, zero, k)
}
