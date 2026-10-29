//! A fragment function's color outputs to attachments the device renders through a word view
//! (`LoweringInputs::word_colors`, `metal::format::WordAttachment`).
//!
//! Such an attachment is a view of the texture's bits as the unsigned integer format of its word,
//! so the function writes each `float` or `half` output to one as the word Metal stores for it,
//! through an output variable of one unsigned 32-bit component ([`plan`], [`encode`]): each
//! channel's value rounded to `half` to nearest even, clamped to `[0, 1]` (a NaN 0), times the
//! channel's maximum code, rounded to nearest even, and the codes packed into their fields (the
//! measurement is `metal::format::WordAttachment`'s). The interface variable keeps the Metal
//! output's type (`StageIo::outputs`), which link checks against the attachment's format as for
//! any other color.
//!
//! The product of a `half` in `[0, 1]` and a maximum code of at most 6 bits is exact in `f32`
//! (eleven significant bits times six), so `RoundEven` of it is Metal's code: its only ties are
//! the products at one half, which Metal stores even.
//!
//! An `RGB9E5Float` word takes a `float` output's word: the mantissas truncated under the largest
//! channel's exponent (`texture::rgb9e5_word`, the rule of its kernel writes). A `half` output to
//! one is not built (Metal rounds the `float` the `half` came from, `WordAttachment`'s docs).
//!
//! An integer output to such an attachment is Metal's pipeline error: it is not encoded, and link
//! refuses it (`ColorAttachment`).
//!
//! The interlocked storage form writes such an attachment through the same encode where it blends
//! or keeps some channels (`lower::fragment::fetch`), and a blend there reads the word read first
//! as Metal's blend reads it ([`decode`]).

use crate::air::intrinsics::{PackDir, PackFormat};
use crate::error::{Error, Feature, Unsupported};
use crate::lower::arith::color_write::half_rounded;
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::lower::arith::{pack, FloatFmt, Form, Held};
use crate::lower::select::SelectCx;
use crate::metal::format::WordAttachment;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::stage_io::{IoNumeric, IoType};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

/// An upper estimate of the SPIR-V words of one output's [`encode`]: per channel the rounding to
/// `half`, the clamp, the product's rounding, the conversion, shift and merge.
pub(crate) const WORDS: u64 = 4 * (40 * 6 + 12 * 6);

/// An upper estimate of the SPIR-V words of one output's [`encode`] into `word`: [`WORDS`], and
/// for an 8-bit sRGB word `arith::pack::pack`'s encode.
pub(crate) fn encode_words(word: WordAttachment) -> u64 {
    match word {
        WordAttachment::Packed16 { .. } => WORDS,
        WordAttachment::Srgb8 { .. } => {
            let f4 = Form {
                held: Held::Float(FloatFmt::F32),
                lanes: 4,
            };
            WORDS + u64::from(pack::words(PackFormat::Unorm4x8Srgb, PackDir::Pack, f4))
        }
        WordAttachment::Rgb9e5 => crate::lower::texture::RGB9E5_WORDS + 6 * 8,
    }
}

/// Plan output `f` (dual-source `index`) of a fragment function to color attachment `color`:
/// where `LoweringInputs::word_colors` lists the attachment and the output is `float` or `half` of
/// dual-source index 0, record the attachment on the interface (`StageIo::word_colors`) and the
/// encode each return stores (`StageIoPlan::words_written`), and give the word's type the output
/// variable is declared with (`StageIo::declared_type`). `None` where the output is stored as it
/// is: another attachment's, or an integer output (link refuses it). A `half` output to an
/// `RGB9E5Float` word is `NotImplemented(Format)`.
pub(super) fn plan(
    cx: &mut SelectCx<'_>,
    color: u32,
    index: u32,
    f: Form,
) -> Result<Option<IoType>, Error> {
    if cx.stage != Stage::Fragment || index != 0 {
        return Ok(None);
    }
    let Some(word) = cx.inputs.word_colors.get(&color).copied() else {
        return Ok(None);
    };
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(None);
    }
    if word == WordAttachment::Rgb9e5 && f.float_fmt() != Some(FloatFmt::F32) {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    cx.families
        .stage_io
        .interface
        .word_colors
        .insert(color, word);
    cx.families.stage_io.words_written.insert(color, word);
    // An 8-bit sRGB word encodes by the device's conversion.
    if let WordAttachment::Srgb8 { .. } = word {
        cx.precision.insert(PrecisionNote::DevicePackConversion);
    }
    Ok(Some(IoType {
        numeric: IoNumeric::Uint,
        components: 1,
        source_bits: 32,
    }))
}

/// The word Metal stores for output `x` (form `f`, `float` or `half` lanes held in 32-bit
/// containers) to a word attachment `word` (the module docs), as a `u32`. A channel the output
/// lacks is 0, a missing alpha 1. An 8-bit sRGB word encodes by the device's conversion
/// (`arith::pack::pack` of `unorm4x8_srgb`, `PrecisionNote::DevicePackConversion`). An
/// `RGB9E5Float` word takes a `float` output's red, green and blue (`texture::rgb9e5_word`).
pub(crate) fn encode(
    fb: &mut FunctionBuilder<'_>,
    word: WordAttachment,
    (f, x): (Form, Value),
) -> BResult<Value> {
    let s = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    if let WordAttachment::Rgb9e5 = word {
        let mut lanes = Vec::with_capacity(3);
        for lane in 0..3u8 {
            lanes.push(match lane < f.lanes {
                true if f.lanes == 1 => x,
                true => fb.composite_extract(x, &[u32::from(lane)])?,
                false => fsplat(fb, s, 0.0)?,
            });
        }
        let f3 = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 3,
        };
        let t = ty(fb, f3)?;
        let x3 = fb.composite_construct(t, &lanes)?;
        return crate::lower::texture::rgb9e5_word(fb, x3);
    }
    if let WordAttachment::Srgb8 { .. } = word {
        // The output's four lanes (a lane it lacks 0), packed by the device's sRGB encode: red's byte
        // first, then green's; the word keeps the format's.
        let mut lanes = Vec::with_capacity(4);
        for lane in 0..4u8 {
            lanes.push(match lane < f.lanes {
                true if f.lanes == 1 => x,
                true => fb.composite_extract(x, &[u32::from(lane)])?,
                false => fsplat(fb, s, 0.0)?,
            });
        }
        let f4 = Form {
            held: Held::Float(FloatFmt::F32),
            lanes: 4,
        };
        let t = ty(fb, f4)?;
        let x4 = fb.composite_construct(t, &lanes)?;
        let packed = pack::pack(fb, PackFormat::Unorm4x8Srgb, f4, x4)?;
        let u = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        };
        let mask = match word.word_format() {
            crate::vk::Format::R8_UINT => 0xff,
            _ => 0xffff,
        };
        let mask = splat(fb, u, mask)?;
        return fb.binary(BinOp::BitwiseAnd, packed, mask);
    }
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    let ut = ty(fb, u)?;
    let mut out = splat(fb, u, 0)?;
    for (lane, field) in word.fields().iter().enumerate() {
        let Some(field) = field else {
            continue;
        };
        let v = match u8::try_from(lane).is_ok_and(|l| l < f.lanes) {
            true if f.lanes == 1 => x,
            true => fb.composite_extract(x, &[lane as u32])?,
            false => fsplat(fb, s, if lane == 3 { 1.0 } else { 0.0 })?,
        };
        // The `half` nearest the value (a `half` output is one already), a NaN 0, in `[0, 1]`.
        let h = half_rounded(fb, s, v)?;
        let is_nan = fb.unary(UnOp::IsNan, h)?;
        let zero = fsplat(fb, s, 0.0)?;
        let h = fb.select(is_nan, zero, h)?;
        let one = fsplat(fb, s, 1.0)?;
        let h = fb.glsl(GlslOp::NClamp, &[h, zero, one])?;
        let m = fsplat(fb, s, f64::from((1u32 << field.bits) - 1))?;
        let p = fb.binary_no_contraction(BinOp::FMul, h, m)?;
        let c = fb.glsl(GlslOp::RoundEven, &[p])?;
        let c = fb.convert(ConvOp::ConvertFToU, c, ut)?;
        let c = match field.shift {
            0 => c,
            shift => {
                let shift = splat(fb, u, u64::from(shift))?;
                fb.binary(BinOp::ShiftLeftLogical, c, shift)?
            }
        };
        out = fb.binary(BinOp::BitwiseOr, out, c)?;
    }
    Ok(out)
}

/// An upper estimate of the SPIR-V words of one [`decode`]: per channel the shift, mask,
/// conversion, product and rounding to `half`, and the vector's construction.
pub(crate) const DECODE_WORDS: u64 = 4 * (12 * 6 + 40 * 6) + 12;

/// The four channels Metal's blend reads from word `x` (a `u32`) of packed 16-bit word attachment
/// `word` (`WordAttachment::Packed16`), as a
/// `vec4` of `f32`, red first: each channel's code over its maximum code rounded to `half`, nearest
/// even, a channel the format lacks 0, alpha 1 (measured: the packed 16-bit blend sweep,
/// `extended_range::Blend::packed16`). The code times the `f32` nearest the maximum's reciprocal
/// lies within two `f32` steps of the quotient, whose `half` it rounds to: the quotient of a code
/// by `2^k - 1` repeats `k` bits without end (or is 0 or 1), which keeps it far from a `half`
/// midpoint, more than `2^-(k+1)` of a `half` step for `k` up to 6.
pub(crate) fn decode(
    fb: &mut FunctionBuilder<'_>,
    word: WordAttachment,
    x: Value,
) -> BResult<Value> {
    let s = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    let st = ty(fb, s)?;
    let mut lanes = Vec::with_capacity(4);
    for (lane, field) in word.fields().iter().enumerate() {
        let Some(field) = field else {
            lanes.push(fsplat(fb, s, if lane == 3 { 1.0 } else { 0.0 })?);
            continue;
        };
        let max = (1u32 << field.bits) - 1;
        let code = match field.shift {
            0 => x,
            shift => {
                let shift = splat(fb, u, u64::from(shift))?;
                fb.binary(BinOp::ShiftRightLogical, x, shift)?
            }
        };
        let mask = splat(fb, u, u64::from(max))?;
        let code = fb.binary(BinOp::BitwiseAnd, code, mask)?;
        let c = fb.convert(ConvOp::ConvertUToF, code, st)?;
        let r = fsplat(fb, s, f64::from(1.0f32 / max as f32))?;
        let v = fb.binary_no_contraction(BinOp::FMul, c, r)?;
        lanes.push(half_rounded(fb, s, v)?);
    }
    let v4 = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    };
    let t = ty(fb, v4)?;
    fb.composite_construct(t, &lanes)
}
