//! What the device-sampled filters share with the emulated ones the device has no form for
//! (`extended`, `heap_raw`, `in_place`, `bicubic`): the census words of a filter's footprint and
//! of a runtime sampler's word, the rule of an axis whose two taps read one texel ([`merged`]), the samplers
//! that read one texel of one level ([`unfiltered`]), and the note of a filtered
//! sample that Metal's own sampler answers otherwise than Metal filters the format bound there
//! ([`filter_note`]).
//!
//! Every filtered sample of a format the device samples is the device's: Metal's own filter on a
//! device that samples through Metal's own sampler (`VkProperty::MetalSampler`, MoltenVK), and the
//! device's own elsewhere (`VkProperty::DeviceSampling`, recorded as
//! `PrecisionNote::DeviceSampling`).
//!
//! **Two-plane YCbCr textures** (`YCBCR8_420_2P` and its kin). A format's view holds each channel
//! of the converted color as Metal reads it, a step `k / D` (alpha 1) on its field's grid: `2048`
//! (`PackedTexels::YCbCr2048`), `8192` (`PackedTexels::YCbCr8192`), or `8192` signed within
//! `[-2, 2]` (`PackedTexels::YCbCrExtended8192`). Metal's 2D linear sample sums the steps on the
//! grid, each term truncated; Metal's own sampler filters the view's texels as an `RGBA16Float` or
//! `RGBA32Float` texture's, within one step of the grid, so a filtered sample of such a slot there
//! records `PrecisionNote::HalfSampleFilter` or `PrecisionNote::FloatSampleFilter` where the bound
//! format is two-plane YCbCr ([`metal_filtered_ycbcr`]), through a runtime sampler slot only where
//! the bound sampler filters ([`filtering_word`]). A device whose memory is the host's reads the
//! planes in place, Metal's filter exactly (`in_place`).

use crate::air::abi::{Filter, MipFilter, SamplerState};
use crate::error::Error;
use crate::lower::select::{any_of, NoteClause, NoteOn, NoteTest, SelectCx};
use crate::lower::texture::{ImageRef, TexelConv, TexelForm};
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::{PackedTexels, TexelFormat};
use crate::metal::SamplerMipFilter;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::bicubic::{scalar, value, Tap, Tys};
use super::SampleOp;

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

/// The note of sample `op` of `image` through `sampler` (`None`: an argument buffer's) with
/// texel form `form` where Metal's own sampler (`VkProperty::MetalSampler`) filters it otherwise
/// than Metal filters the format bound there, with the clauses under which it holds
/// ([`metal_filtered_ycbcr`]): `PrecisionNote::HalfSampleFilter` for a `half` result,
/// `PrecisionNote::FloatSampleFilter` for a `float` one (a depth texture's and a comparison's
/// included); `None` for a sample that is not filtered (a constexpr sampler that reads one texel
/// of one level, [`unfiltered`]), one of another result, a heap texture's, and every sample on a
/// device that samples natively, whose module records `PrecisionNote::DeviceSampling` instead.
/// Through a runtime sampler slot each clause also tests the word of the sampler bound there
/// ([`filtering_word`]), which the runtime reads at each draw or dispatch: a bound sampler that
/// reads one texel of one level takes the texel the view holds, Metal's read, and holds none.
/// An argument buffer's sampler states its word in a table no slot's test reads, so its clauses
/// test the bound format alone.
pub(super) fn filter_note(
    cx: &SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (sampler, image): (Option<SamplerRef>, ImageRef),
) -> Result<Option<(PrecisionNote, Vec<NoteClause>)>, Error> {
    if !matches!(op, SampleOp::Sample { .. }) {
        return Ok(None);
    }
    let note = match form.map(|f| f.conv) {
        Some(TexelConv::F16) => PrecisionNote::HalfSampleFilter,
        Some(TexelConv::F32) => PrecisionNote::FloatSampleFilter,
        _ => return Ok(None),
    };
    let filters = match sampler {
        Some(SamplerRef::Constexpr(words)) => match unfiltered(&super::state_of(words)?) {
            true => return Ok(None),
            false => None,
        },
        Some(SamplerRef::Slot(n)) => Some(filtering_word(n)),
        None => None,
    };
    let Some(formats) = metal_filtered_ycbcr(cx, image) else {
        return Ok(None);
    };
    let when = match filters {
        None => formats,
        Some(words) => formats
            .iter()
            .flat_map(|format| {
                words.iter().map(move |&word| {
                    let mut clause = format.clone();
                    clause.insert(word);
                    clause
                })
            })
            .collect(),
    };
    Ok(Some((note, when)))
}

/// The tests of the word of the sampler bound at runtime sampler slot `n` (`NoteOn::BoundSampler`,
/// `SamplerWord`) of which one holds exactly where a sample through it filters, [`unfiltered`]'s
/// rule read from the word: a linear minification or magnification filter, anisotropy or a cubic
/// filter (any of their bits set), or a linear mip filter.
fn filtering_word(n: u32) -> [NoteTest; 2] {
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

/// On a device that samples through Metal's own sampler (`VkProperty::MetalSampler`), the clauses
/// under which a
/// filtered sample of texture slot `image` is not Metal's filter of the format bound there: where
/// that format is two-plane YCbCr (its class's `TexelFormat::PACKED` field one of
/// [`PackedTexels::YCbCr2048`], [`PackedTexels::YCbCr8192`] and
/// [`PackedTexels::YCbCrExtended8192`]), whose view holds Metal's reads exactly but which Metal's
/// sampler then filters as an `RGBA16Float` or `RGBA32Float` texture's texels, not on the format's
/// own grid. The runtime reads the bound format (`SpecKey::BoundTexelFormat`) and announces it as
/// `ycbcr-filter-not-exact` ([`filter_note`] adds a runtime sampler slot's word). `None` on any
/// other device, and for a heap's texture.
pub(super) fn metal_filtered_ycbcr(cx: &SelectCx<'_>, image: ImageRef) -> Option<Vec<NoteClause>> {
    let ImageRef::Slot(index) = image else {
        return None;
    };
    let metal = cx
        .profile
        .fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler);
    metal.then(|| {
        any_of(
            [
                PackedTexels::YCbCr2048,
                PackedTexels::YCbCr8192,
                PackedTexels::YCbCrExtended8192,
            ]
            .map(|p| NoteTest::equal(NoteOn::Bound(index), TexelFormat::PACKED_MASK, p.class().0)),
        )
    })
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
