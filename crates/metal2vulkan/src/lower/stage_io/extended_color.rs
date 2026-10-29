//! A fragment function's color outputs to Apple's extended-range attachments
//! (`LoweringInputs::extended_range_colors`).
//!
//! Such an attachment holds the format's raw texels
//! (`metal::format::ExtendedRange::raw_format`), so the function writes each `float` or `half`
//! output to one as the raw texel Metal stores for it, through an output variable of the raw
//! format's unsigned components ([`plan`], [`encode`]): Metal rounds a `float` output to `half`
//! (to nearest, ties to even) and stores the texel a kernel's write of that value stores
//! (`arith::extended_range::encode`), and a `half` output's texel is the kernel write's of the
//! `half` itself (every texel of 8,192 random outputs to each of the six formats,
//! `float` and `half`, on an Apple M4 under macOS 27.0.1, plain and under the validation layer
//! alike). The interface variable keeps the Metal output's type (`StageIo::outputs`), which link
//! checks against the attachment's format as for any other color.
//!
//! A bfloat attachment (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) holds `half` values in Metal's
//! tile, as an extended-range one does, and stores each as the bfloat code nearest it, ties to
//! even, a NaN `0x7fc0` ([`bfloat_output`]): a `float` output rounds to `half` toward zero, a
//! finite value past 65504 saturating there, or to nearest even under the pipeline's
//! `textureWriteRoundingMode` `NearestEven` (`ExtendedRangeColor::nearest_even`); a `half` output
//! is taken as it is (an Apple M4 under macOS 27.0.1: every code's `float` and
//! `half` output and the `float`s at and around each code's tie, into each bfloat format, at the
//! default mode and at `NearestEven`).
//!
//! An integer output to such an attachment is Metal's pipeline error: it is not encoded, and link
//! refuses it (`ColorAttachment`). Refused `NotImplemented(Format)` until measured and modelled:
//! an attachment the pipeline blends (`ExtendedRangeColor::blended`; Metal's validation refuses a
//! bfloat one's blend) and a dual-source second output to one. A `bfloat` output is refused
//! before, as Metal's compiler rejects it.
//!
//! On a device that orders framebuffer fetch through rasterization-order input attachments, a blend
//! the function computes over the attachment's input attachment (`lower::fragment`'s `input_blend`)
//! is built: the output's write stores the raw texel of Metal's blend over the texel the input
//! attachment reads, which `stage_io` writes through the output variable as it writes [`encode`]'s.

use crate::error::{Error, Feature, Unsupported};
use crate::lower::arith::extended_range;
use crate::lower::arith::texel_round::{self, HalfRounding};
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::select::SelectCx;
use crate::metal::format::ExtendedRange;
use crate::pipeline::lower_inputs::ExtendedRangeColor;
use crate::pipeline::stage_io::{IoNumeric, IoType};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{ConstBits, Constant, Type};

/// An upper estimate of the SPIR-V words of one output's [`encode`]: the lanes completed to four,
/// the rounding to `half`, the texel's encode, and the raw components' extract.
pub(crate) const WORDS: u64 = 60 * 6 + extended_range::ENCODE_WORDS as u64;

/// Plan output `f` (dual-source `index`) of a fragment function to color attachment `color`: where
/// `LoweringInputs::extended_range_colors` lists the attachment and the output is `float` or
/// `half`, record the attachment on the interface (`StageIo::extended_range_colors`) and the
/// encode each return stores (`StageIoPlan::extended`), and give the type of the raw texel the
/// output variable is declared with (`StageIo::declared_type`). `None` where the output is stored
/// as it is: another attachment's, or an integer output (see the module docs).
pub(super) fn plan(
    cx: &mut SelectCx<'_>,
    color: u32,
    index: u32,
    f: Form,
) -> Result<Option<IoType>, Error> {
    if cx.stage != Stage::Fragment {
        return Ok(None);
    }
    let Some(e) = cx.inputs.extended_range_colors.get(&color).copied() else {
        return Ok(None);
    };
    let Some(fmt) = f.float_fmt() else {
        return Ok(None);
    };
    let blended = e.blended && !cx.families.fragment.input_blends.contains_key(&color);
    if index != 0 || blended || !matches!(fmt, FloatFmt::F32 | FloatFmt::Half) {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    cx.families
        .stage_io
        .interface
        .extended_range_colors
        .insert(color, e);
    cx.families.arith.converts(e.form);
    cx.families.stage_io.extended.insert(color, e);
    Ok(Some(IoType {
        numeric: IoNumeric::Uint,
        components: e.form.raw_components() as u8,
        source_bits: 32,
    }))
}

/// The raw texel Metal stores for output `x` (form `f`, `float` or `half` lanes) to the
/// attachment `e`: [`texel`] (or, a bfloat one's [`bfloat_output`]), as the raw format's
/// components.
pub(super) fn encode(
    fb: &mut FunctionBuilder<'_>,
    e: ExtendedRangeColor,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let form = e.form;
    let texel = match form.bfloat() {
        true => bfloat_output(fb, e, f, x)?,
        false => texel(fb, form, f, x)?,
    };
    raw_components(fb, form, texel)
}

/// The raw format's components of `texel`, a `uvec4` raw texel of an attachment held in `form`:
/// what the output variable holds.
pub(super) fn raw_components(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    texel: Value,
) -> BResult<Value> {
    match form.raw_components() {
        1 => fb.composite_extract(texel, &[0]),
        2 => fb.vector_shuffle(texel, texel, &[0, 1]),
        _ => Ok(texel),
    }
}

/// The raw texel Metal stores for output `x` (form `f`, `float` or `half` lanes) to the bfloat
/// attachment `e` (the module docs): the lanes completed to four as [`held`] completes them, a
/// `float` output rounded to `half` toward zero (saturating at 65504) or, where
/// `e.nearest_even`, to nearest even, then each lane's bfloat code nearest, ties to even, a NaN
/// `0x7fc0` (`arith::extended_range::bfloat_stored`). Integer operations on the bits alone.
pub(crate) fn bfloat_output(
    fb: &mut FunctionBuilder<'_>,
    e: ExtendedRangeColor,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let rounding = match e.nearest_even {
        true => HalfRounding::NearestEven,
        false => HalfRounding::TowardZeroSaturating,
    };
    let v = four_lanes(fb, f, x)?;
    let v = match f.float_fmt() {
        Some(FloatFmt::F32) => texel_round::f32_to_half(fb, FOUR, v, rounding)?,
        _ => v,
    };
    extended_range::bfloat_stored(fb, e.form, v)
}

/// A `vec4` of `f32` lanes.
const FOUR: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 4,
};

/// The raw texel Metal stores for output `x` (form `f`, `float` or `half` lanes) to an attachment
/// held in `form`, unblended, as a `uvec4` (`arith::extended_range::encode`): the lanes completed
/// to four (a color lane the output lacks 0, a missing alpha 1; an output with fewer lanes than
/// the format's channels is refused before), a `float` output rounded to `half` to nearest even,
/// then the kernel write's encode. The storage form's unblended write stores it too
/// (`lower::fragment::fetch`).
pub(crate) fn texel(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let v = held(fb, form, f, x)?;
    extended_range::encode(fb, form, v)
}

/// The value [`texel`] encodes for output `x` (form `f`) to an attachment held in `form`, a
/// `vec4` of `half` values: the lanes completed to four, a `float` output rounded to `half`.
/// Metal's tile holds it until the store, which the storage form's shadow of the attachment keeps
/// (`lower::fragment::fetch`). Not a bfloat form's, whose rounding is the pipeline's
/// ([`bfloat_output`]) and which takes no storage form.
pub(crate) fn held(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let v = four_lanes(fb, f, x)?;
    // A padded format's store rounds a `float` output to `half` toward zero (every value swept
    // around each code's midpoint through three draw shapes, plain and
    // blended under one and zero), an extended-range format's to nearest even.
    let rounding = match form.padded() {
        true => HalfRounding::TowardZeroSaturating,
        false => HalfRounding::NearestEven,
    };
    match f.float_fmt() {
        Some(FloatFmt::F32) => texel_round::f32_to_half(fb, FOUR, v, rounding),
        _ => Ok(v),
    }
}

/// Output `x` (form `f`) as a `vec4` of `f32` lanes: a color lane the output lacks 0, a missing
/// alpha 1.
fn four_lanes(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    Ok(match f.lanes {
        4 => x,
        lanes => {
            let f32_ = fb.module().ty(Type::Float { width: 32 })?;
            let v4 = fb.module().ty(Type::Vector { elem: f32_, len: 4 })?;
            let mut parts = Vec::with_capacity(4);
            for lane in 0..4u8 {
                parts.push(match lane < lanes {
                    true if lanes == 1 => x,
                    true => fb.composite_extract(x, &[u32::from(lane)])?,
                    false => {
                        let fill: f32 = if lane == 3 { 1.0 } else { 0.0 };
                        let c = fb.module().constant(Constant::Scalar {
                            ty: f32_,
                            bits: ConstBits(u64::from(fill.to_bits())),
                        })?;
                        fb.module().const_value(c)?
                    }
                });
            }
            fb.composite_construct(v4, &parts)?
        }
    })
}
