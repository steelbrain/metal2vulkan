//! Apple's extended-range texels (`metal::format::ExtendedRange`): the value of each raw code a
//! slot reads, and the code of each value it writes, by the device's arithmetic.
//!
//! A channel's 10-bit code `c` holds `(c - 384) / 510`: a read answers `float(c - 384)` times the
//! `f32` nearest `1 / 510` (alpha's code clamped to `[384, 894]` first, a two-plane texel's alpha
//! byte `b` times the `f32` nearest `1 / 255`); a write takes the value, a NaN as 0, clamped to the
//! codes' range (alpha to `[0, 1]`), times 510, `RoundEven`, plus 384 ([`encode`]), as
//! `arith::pack` encodes `rgb10a2`. A padded format's code `c` reads `float(c)` times the `f32`
//! nearest `1 / 1023` (`RGB10A2Unorm_sRGB`'s alpha `1 / 3`) and writes `RoundEven(clamp(v, 0, 1) ·
//! 1023)`. An `_sRGB` color channel goes through the MSL transfer by the device's `Pow`
//! (`arith::pack`'s): a read decodes the magnitude of the linear value of its code, signed, and a
//! write encodes the magnitude of the value, signed, before the code's rounding;
//! `RG10Unorm_X12_sRGB` reads its codes linearly and writes the 8-bit sRGB code of its linear code
//! `m` over 4092 (`RoundEven(255 · srgb(m / 4092))`), as Metal does. A `half` result is the
//! device's narrowing of the `float` value (`narrow::round_texel`).
//!
//! Metal answers the correctly rounded quotient, rounds its `half` with ties away from zero,
//! encodes by the exact product, and converts the `_sRGB` forms through its own tables, which
//! the transfer meets: within half a `2^-13` step of each extended-range decode step, a code of
//! each encode (32 of the 16,385 values on the `2^-13` grid up to 2 a code apart), within 1.64
//! steps of `R10Unorm_X6_sRGB`'s decode steps and a code of its encode (381 of 8,193), and a code
//! of `RG10Unorm_X12_sRGB`'s write (36 of the 1,024 linear codes). Every function that converts
//! such a texel records `PrecisionNote::DevicePackConversion` (`ArithPlan::converts`).
//!
//! A bfloat form's texel is each code shifted up 16 bits as a `float`, a NaN as it is; a write
//! keeps a value's top 16 bits, a NaN kept a NaN by its quiet bit.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, splat};
use crate::lower::arith::pack::{srgb_decode, srgb_encode};
use crate::metal::format::ExtendedRange;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type, UnOp};

pub mod blend;
pub(crate) mod blend_color;
pub use blend::Blend;

/// An upper estimate of the SPIR-V words of one texel's [`decode`]: the codes, the conversion
/// and product, an `_sRGB` form's transfer (its steps and constants) and sign, and the `half`
/// narrowing.
pub const DECODE_WORDS: u32 = 40 * 6 + 9 * 6 + 5 * 8 + 64 + 16;

/// An upper estimate of the SPIR-V words of one texel's [`encode`]: the clean value, an `_sRGB`
/// form's transfer and sign, the clamp, product, rounding and conversion, and the packing.
pub const ENCODE_WORDS: u32 = 48 * 6 + 9 * 6 + 5 * 8;

/// Four `u32` lanes.
fn words() -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 4,
    }
}

/// Four `f32` lanes.
fn floats() -> Form {
    words().with(Held::Float(FloatFmt::F32))
}

/// Four `half` lanes, held as `f32`.
fn halves() -> Form {
    words().with(Held::Float(FloatFmt::Half))
}

/// Four `bfloat` lanes, held as `f32`.
fn bfloats() -> Form {
    words().with(Held::Float(FloatFmt::BFloat))
}

/// The four codes of raw texel `raw` (the raw format's `uvec4` as a read gives it) in Metal's
/// channel order, red first; `BGR10_XR`'s alpha is the code of 1, and `BGRA10_XR`'s is clamped
/// to `[384, 894]`, the codes of `[0, 1]`. A two-plane texel's alpha is its alpha byte (the
/// second component), an 8-bit unorm code.
pub(crate) fn codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    match form {
        ExtendedRange::Bgr10 { .. } | ExtendedRange::Planes { .. } => {
            let w = fb.composite_extract(raw, &[0])?;
            let mut lanes = Vec::with_capacity(4);
            for shift in [20u32, 10, 0] {
                let s = fb.u32_const(shift)?;
                let s = fb.module().const_value(s)?;
                let x = fb.binary(BinOp::ShiftRightLogical, w, s)?;
                let m = fb.u32_const(0x3FF)?;
                let m = fb.module().const_value(m)?;
                lanes.push(fb.binary(BinOp::BitwiseAnd, x, m)?);
            }
            if let ExtendedRange::Planes { .. } = form {
                let a = fb.composite_extract(raw, &[1])?;
                let m = fb.u32_const(0xFF)?;
                let m = fb.module().const_value(m)?;
                lanes.push(fb.binary(BinOp::BitwiseAnd, a, m)?);
            } else {
                let one = fb.u32_const(894)?;
                lanes.push(fb.module().const_value(one)?);
            }
            fb.composite_construct(ut, &lanes)
        }
        ExtendedRange::Bgra10 { .. } => {
            let six = splat(fb, u, 6)?;
            let c = fb.binary(BinOp::ShiftRightLogical, raw, six)?;
            // Red is the third field, blue the first.
            let c = fb.vector_shuffle(c, c, &[2, 1, 0, 3])?;
            let lo = per_lane(fb, [0, 0, 0, 384])?;
            let hi = per_lane(fb, [1023, 1023, 1023, 894])?;
            fb.glsl(GlslOp::UClamp, &[c, lo, hi])
        }
        // A padded format's words hold each code in their top 10 bits; blue is 0 and alpha the
        // code of 1 (`value_of_codes` scales every lane by `1 / 1023`).
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb => {
            let six = splat(fb, u, 6)?;
            let c = fb.binary(BinOp::ShiftRightLogical, raw, six)?;
            let fill = per_lane(fb, [0, 0, 0, 1023])?;
            fb.vector_shuffle(c, fill, padded_lanes(form))
        }
        // `RGB10A2Unorm_sRGB`'s word: red, green and blue in 10-bit fields from bit 0, alpha's
        // 2-bit code in bits 30 and 31.
        ExtendedRange::Rgb10A2Srgb => {
            let w = fb.composite_extract(raw, &[0])?;
            let w = fb.composite_construct(ut, &[w, w, w, w])?;
            let shifts = per_lane(fb, [0, 10, 20, 30])?;
            let c = fb.binary(BinOp::ShiftRightLogical, w, shifts)?;
            let m = per_lane(fb, [0x3FF, 0x3FF, 0x3FF, 0x3])?;
            fb.binary(BinOp::BitwiseAnd, c, m)
        }
        // A bfloat form has no codes of this kind: its texels convert by [`bfloat_value`] and
        // [`bfloat_texel`], and no attachment of one is held.
        ExtendedRange::R16Bfloat | ExtendedRange::Rg16Bfloat | ExtendedRange::Rgba16Bfloat => Err(
            crate::spirv::error::BuildError::Type(crate::spirv::error::TypeRule::FunctionSignature),
        ),
    }
}

/// The shuffle that keeps a padded form's channels of a first vector and takes the rest from a
/// second: red alone, or red and green.
fn padded_lanes(form: ExtendedRange) -> &'static [u32] {
    match form.raw_components() {
        1 => &[0, 5, 6, 7],
        _ => &[0, 1, 6, 7],
    }
}

/// A `uvec4` constant of the four lanes `v`.
fn per_lane(fb: &mut FunctionBuilder<'_>, v: [u32; 4]) -> BResult<Value> {
    let ut = crate::lower::arith::ops::ty(fb, words())?;
    let e = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let mut parts = Vec::with_capacity(4);
    for x in v {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(u64::from(x)),
        })?);
    }
    let c = fb.module().constant(Constant::Composite {
        ty: ut,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// The value of raw texel `raw` of a slot bound in `form` (the raw format's `uvec4`), a `vec4` of
/// `f32`, red first (the module docs). For a `half` result (`half`) each lane is the device's
/// narrowing of that value to `half`, held as the `f32` of it.
pub fn decode(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    half: bool,
) -> BResult<Value> {
    if form.bfloat() {
        return bfloat_value(fb, form, raw, half);
    }
    let c = codes(fb, form, raw)?;
    value_of_codes(fb, form, c, half)
}

/// The tile value of raw texel `raw` of a color attachment of `form` (a framebuffer fetch's
/// read, a `float` and a `half` read alike): a `vec4` of `f32`, red first, each lane a `half`
/// value. Metal's tile holds no color code above 1022: a linear form's code and
/// `BGRA10_XR_sRGB`'s are clamped to 1022 and decoded ([`decode`]), `BGR10_XR_sRGB`'s and the
/// two-plane twin's decoded and then clamped to the linear values of codes 0 and 1022,
/// `[-384 / 510, 638 / 510]`; each lane is the device's narrowing of that value to `half`.
/// Alpha is [`decode`]'s.
pub fn tile_value(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, raw: Value) -> BResult<Value> {
    let c = codes(fb, form, raw)?;
    let by_code = !form.srgb() || matches!(form, ExtendedRange::Bgra10 { .. });
    let c = match by_code {
        true => {
            let top = per_lane(fb, [1022, 1022, 1022, u32::MAX])?;
            fb.glsl(GlslOp::UMin, &[c, top])?
        }
        false => c,
    };
    let v = value_of_codes(fb, form, c, false)?;
    let v = match by_code {
        true => v,
        false => {
            let lo = (-384.0f64 / 510.0) as f32;
            let hi = (638.0f64 / 510.0) as f32;
            let lo = per_lane_f(fb, [lo, lo, lo, f32::NEG_INFINITY])?;
            let hi = per_lane_f(fb, [hi, hi, hi, f32::INFINITY])?;
            fb.glsl(GlslOp::FClamp, &[v, lo, hi])?
        }
    };
    narrow::round(fb, halves(), v)
}

/// Metal's read of the tile memory of an attachment of `form` that holds `held` (a `vec4` of
/// `f32`, red first, each lane a `half`: what the pass's clear, load or last write left there,
/// the `half` shadow's texel): a fetch's value, and a blend's destination. Its color lanes are
/// clamped to the blend's source range (`BGRA10_XR_sRGB`'s [`SRGB_SOURCE_RANGE`], every other
/// form's [`SOURCE_RANGE`]), alpha to `[0, 1]`, a NaN to the low end (`blend::clamped`, the
/// blend's own clamp of its source), and `BGR10_XR`'s alpha is one: the tile holds what a write
/// wrote, unclamped, and stores its encode, but every read clamps (on an Apple
/// M4: unblended `half` writes past the range, NaNs and the infinities, each read back by a
/// fetch and blended over, for every form; their codes are the writes' own encodes, code 1023
/// for 2.0 where the read is 1022's value). [`tile_value`] of a texel's codes is such a read
/// already.
pub fn tile_read(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, held: Value) -> BResult<Value> {
    let read = blend::clamped(fb, held, source_range(form))?;
    match form {
        ExtendedRange::Bgr10 { .. } => {
            let one = fsplat(fb, floats(), 1.0)?;
            fb.vector_shuffle(read, one, &[0, 1, 2, 7])
        }
        _ => Ok(read),
    }
}

pub(crate) use blend_color::{SOURCE_RANGE, SRGB_SOURCE_RANGE};

/// The range Metal clamps a blend's `half` source color to before blending into an attachment
/// of `form` ([`SOURCE_RANGE`], [`SRGB_SOURCE_RANGE`]).
fn source_range(form: ExtendedRange) -> (f32, f32) {
    match form {
        ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
        _ => SOURCE_RANGE,
    }
}

/// The `ivec4` type.
fn ivec4(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    let i32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    fb.module().ty(Type::Vector { elem: i32_, len: 4 })
}

/// The lanes of `x` where `colour` holds and of `t` elsewhere (a shuffle of the two).
fn colour_lanes(
    fb: &mut FunctionBuilder<'_>,
    x: Value,
    t: Value,
    colour: [bool; 4],
) -> BResult<Value> {
    let pick: Vec<u32> = (0u32..)
        .zip(colour)
        .map(|(i, c)| if c { i } else { 4 + i })
        .collect();
    fb.vector_shuffle(x, t, &pick)
}

/// The lanes, red first, an `_sRGB` form's transfer converts: its color channels.
/// `RG10Unorm_X12_sRGB`'s reads are linear and its writes take their own transfer
/// ([`padded_codes`]).
fn srgb_lanes(form: ExtendedRange) -> [bool; 4] {
    match form {
        ExtendedRange::Bgr10 { srgb: true }
        | ExtendedRange::Bgra10 { srgb: true }
        | ExtendedRange::Planes { srgb: true }
        | ExtendedRange::Rgb10A2Srgb => [true, true, true, false],
        ExtendedRange::R10X6Srgb => [true, false, false, false],
        _ => [false; 4],
    }
}

/// [`decode`]'s value of the codes `c` ([`codes`]) of `form`.
fn value_of_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    c: Value,
    half: bool,
) -> BResult<Value> {
    let f = floats();
    let ft = crate::lower::arith::ops::ty(fb, f)?;
    let (n, scale) = match form.padded() {
        // A padded format's lane: its code; `RGB10A2Unorm_sRGB`'s alpha over 3.
        true => {
            let n = fb.convert(ConvOp::ConvertUToF, c, ft)?;
            let a = match form {
                ExtendedRange::Rgb10A2Srgb => 3.0,
                _ => 1023.0,
            };
            (n, [1023.0, 1023.0, 1023.0, a])
        }
        // A code less the zero code, signed; a two-plane texel's alpha byte over 255.
        false => {
            let planes = matches!(form, ExtendedRange::Planes { .. });
            let (zero, a) = match planes {
                true => (per_lane(fb, [384, 384, 384, 0])?, 255.0),
                false => (per_lane(fb, [384; 4])?, 510.0),
            };
            let d = fb.binary(BinOp::ISub, c, zero)?;
            let it = ivec4(fb)?;
            let d = fb.convert(ConvOp::Bitcast, d, it)?;
            (
                fb.convert(ConvOp::ConvertSToF, d, ft)?,
                [510.0, 510.0, 510.0, a],
            )
        }
    };
    let r = per_lane_f(fb, scale.map(|d: f32| 1.0 / d))?;
    let v = fb.binary(BinOp::FMul, n, r)?;
    let colour = srgb_lanes(form);
    let v = match colour.contains(&true) {
        // The transfer of the magnitude, signed.
        true => {
            let m = fb.glsl(GlslOp::FAbs, &[v])?;
            let t = srgb_decode(fb, f, m)?;
            let minus = fb.unary(UnOp::FNegate, t)?;
            let zero = fsplat(fb, f, 0.0)?;
            let neg = fb.binary(BinOp::FOrdLessThan, v, zero)?;
            let t = fb.select(neg, minus, t)?;
            colour_lanes(fb, t, v, colour)?
        }
        false => v,
    };
    match half {
        true => narrow::round_texel(fb, halves(), v),
        false => Ok(v),
    }
}

/// The raw texel (the raw format's `uvec4`) a write of `v` (a `vec4` of `f32`, red first) to a
/// slot bound in `form` stores (the module docs).
pub fn encode(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, v: Value) -> BResult<Value> {
    // A padded format's code in its word's top 10 bits, the low 6 bits zero.
    if form.padded() {
        let u = words();
        let k = padded_codes(fb, form, v)?;
        if form == ExtendedRange::Rgb10A2Srgb {
            return pack(fb, form, k);
        }
        let six = splat(fb, u, 6)?;
        let w = fb.binary(BinOp::ShiftLeftLogical, k, six)?;
        let z = splat(fb, u, 0)?;
        return fb.vector_shuffle(w, z, padded_lanes(form));
    }
    if form.bfloat() {
        return bfloat_texel(fb, form, v);
    }
    let code = linear_codes(fb, form, v)?;
    pack(fb, form, code)
}

/// `v` (a `vec4` of `f32`) with a NaN lane 0.
fn clean(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let zero = fsplat(fb, floats(), 0.0)?;
    let nan = fb.unary(UnOp::IsNan, v)?;
    fb.select(nan, zero, v)
}

/// `v` (a `vec4` of `f32` within the codes' range) times `scale` per lane, `RoundEven`, plus
/// `bias` per lane, as `u32` codes.
fn rounded_codes(
    fb: &mut FunctionBuilder<'_>,
    v: Value,
    scale: [f32; 4],
    bias: [f32; 4],
) -> BResult<Value> {
    let scale = per_lane_f(fb, scale)?;
    let x = fb.binary(BinOp::FMul, v, scale)?;
    let x = fb.glsl(GlslOp::RoundEven, &[x])?;
    let x = match bias.iter().any(|&b| b != 0.0) {
        true => {
            let bias = per_lane_f(fb, bias)?;
            fb.binary(BinOp::FAdd, x, bias)?
        }
        false => x,
    };
    let ut = crate::lower::arith::ops::ty(fb, words())?;
    fb.convert(ConvOp::ConvertFToU, x, ut)
}

/// The codes a write of `v` (a `vec4` of `f32`, red first) to a slot of a padded form stores
/// ([`encode`]): the value, a NaN 0, clamped to `[0, 1]`, an `_sRGB` color channel through the
/// transfer, times 1023 (`RGB10A2Unorm_sRGB`'s alpha 3), `RoundEven`. `RG10Unorm_X12_sRGB`'s
/// red and green are the 8-bit sRGB codes of their linear codes `m` over 4092,
/// `RoundEven(255 · srgb(m / 4092))`, as Metal writes them.
fn padded_codes(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, v: Value) -> BResult<Value> {
    let f = floats();
    let v = clean(fb, v)?;
    let zero = fsplat(fb, f, 0.0)?;
    let one = fsplat(fb, f, 1.0)?;
    let v = fb.glsl(GlslOp::FClamp, &[v, zero, one])?;
    let colour = srgb_lanes(form);
    let v = match colour.contains(&true) {
        true => {
            let t = srgb_encode(fb, f, v)?;
            colour_lanes(fb, t, v, colour)?
        }
        false => v,
    };
    let alpha = match form {
        ExtendedRange::Rgb10A2Srgb => 3.0,
        _ => 1023.0,
    };
    let k = rounded_codes(fb, v, [1023.0, 1023.0, 1023.0, alpha], [0.0; 4])?;
    if form != ExtendedRange::Rg10X12Srgb {
        return Ok(k);
    }
    let ft = crate::lower::arith::ops::ty(fb, f)?;
    let m = fb.convert(ConvOp::ConvertUToF, k, ft)?;
    let r = fsplat(fb, f, f64::from(1.0f32 / 4092.0))?;
    let m = fb.binary(BinOp::FMul, m, r)?;
    let t = srgb_encode(fb, f, m)?;
    rounded_codes(fb, t, [255.0; 4], [0.0; 4])
}

/// The codes (red first) a write of `v` (a `vec4` of `f32`, red first) to a slot of `form` stores,
/// as [`encode`] packs them: a linear or `_sRGB` form's color codes and alpha's, or a padded
/// form's.
pub(crate) fn write_codes(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
) -> BResult<Value> {
    match form.padded() {
        true => padded_codes(fb, form, v),
        false => linear_codes(fb, form, v),
    }
}

/// The lanes a bfloat form keeps of a first vector, the rest taken from a second: red; red and
/// green; or all four.
fn bfloat_lanes(form: ExtendedRange) -> &'static [u32] {
    match form.raw_components() {
        1 => &[0, 5, 6, 7],
        2 => &[0, 1, 6, 7],
        _ => &[0, 1, 2, 3],
    }
}

/// The value of raw texel `raw` of a bfloat form ([`decode`]; `metal::format::ExtendedRange`'s
/// `R16Bfloat`): each channel's code shifted up 16 bits as a `float`, a NaN code's NaN as it is
/// (Metal answers `0x7fc00000`); a channel the form lacks 0 and alpha 1; a `half` result the
/// device's narrowing of each lane.
fn bfloat_value(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    raw: Value,
    half: bool,
) -> BResult<Value> {
    let (u, f) = (words(), floats());
    let ft = crate::lower::arith::ops::ty(fb, f)?;
    let sixteen = splat(fb, u, 16)?;
    let bits = fb.binary(BinOp::ShiftLeftLogical, raw, sixteen)?;
    let fill = per_lane(fb, [0, 0, 0, 1.0f32.to_bits()])?;
    let bits = fb.vector_shuffle(bits, fill, bfloat_lanes(form))?;
    let v = fb.convert(ConvOp::Bitcast, bits, ft)?;
    match half {
        true => narrow::round_texel(fb, halves(), v),
        false => Ok(v),
    }
}

/// `v`, a bfloat sample's `vec4` of `float` values ([`decode`] without `half`, or the border
/// texel), as the sample's `half` result: the device's narrowing of each lane, where Metal's
/// rounds toward zero and saturates a finite value past `65504` there.
pub(crate) fn bfloat_sample_half(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    narrow::round_texel(fb, halves(), v)
}

/// The raw texel a kernel's write of `v` (a `vec4` of `f32`; a `half` value held widened)
/// stores to a bfloat form ([`encode`]): each lane's top 16 bits (toward zero, as Metal writes
/// them), a NaN's with its quiet bit set so that it stays a NaN (Metal writes `0x7fc0`), the
/// components past the form's zero.
fn bfloat_texel(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, v: Value) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let abs = splat(fb, u, 0x7FFF_FFFF)?;
    let mag = fb.binary(BinOp::BitwiseAnd, bits, abs)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let nan = fb.binary(BinOp::UGreaterThan, mag, inf)?;
    let quiet = splat(fb, u, 0x0040_0000)?;
    let q = fb.binary(BinOp::BitwiseOr, bits, quiet)?;
    let bits = fb.select(nan, q, bits)?;
    let sixteen = splat(fb, u, 16)?;
    let code = fb.binary(BinOp::ShiftRightLogical, bits, sixteen)?;
    let zero = splat(fb, u, 0)?;
    fb.vector_shuffle(code, zero, bfloat_lanes(form))
}

/// The raw texel a render pass stores to an attachment of the bfloat form `form` whose tile holds
/// `v` (a `vec4` of `half` values held as `f32`; `stage_io::extended_color::bfloat_output`): each
/// lane rounded to `bfloat` to nearest even (`narrow::round`, a NaN kept a NaN by its quiet bit
/// where Metal stores `0x7fc0`), its top 16 bits, the components past the form's zero. Not a
/// kernel write's code, which drops the low 16 bits ([`bfloat_texel`]).
pub(crate) fn bfloat_stored(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let r = narrow::round(fb, bfloats(), v)?;
    let bits = fb.convert(ConvOp::Bitcast, r, ut)?;
    let sixteen = splat(fb, u, 16)?;
    let code = fb.binary(BinOp::ShiftRightLogical, bits, sixteen)?;
    let zero = splat(fb, u, 0)?;
    fb.vector_shuffle(code, zero, bfloat_lanes(form))
}

/// [`encode`] of `v` into a texel of a linear or `_sRGB` form that keeps the channels `kept`
/// (red, green, blue, alpha) of `old`, the raw texel the write overwrites, as Metal's blended
/// write under a write mask keeps them (every masked configuration, each texel): each kept
/// channel's code as [`decode`] reads it (a color code as it is, 1023 too; `BGRA10_XR`'s alpha
/// clamped to `[384, 894]`; a two-plane texel's alpha byte), packed as a write packs it: plane 0's
/// and `BGR10_XR`'s top two bits as the write sets them (zeros and ones), `BGRA10_XR`'s six low
/// bits zero.
pub fn encode_keeping(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    v: Value,
    (old, kept): (Value, [bool; 4]),
) -> BResult<Value> {
    let new = linear_codes(fb, form, v)?;
    let old = codes(fb, form, old)?;
    let lanes: Vec<u32> = kept
        .iter()
        .zip(0..4u32)
        .map(|(&kept, k)| match kept {
            true => 4 + k,
            false => k,
        })
        .collect();
    let code = fb.vector_shuffle(new, old, &lanes)?;
    pack(fb, form, code)
}

/// The codes a write of `v` (a `vec4` of `f32`, red first) to a slot of a linear or `_sRGB` form
/// stores ([`encode`]): the value, a NaN 0, an `_sRGB` color channel's magnitude through the
/// transfer, signed, clamped to the linear values of codes 0 and 1023 (alpha to `[0, 1]`), times
/// 510, `RoundEven`, plus 384; a two-plane form's alpha byte `RoundEven(255 a)`.
fn linear_codes(fb: &mut FunctionBuilder<'_>, form: ExtendedRange, v: Value) -> BResult<Value> {
    let f = floats();
    let v = clean(fb, v)?;
    let colour = srgb_lanes(form);
    let v = match colour.contains(&true) {
        true => {
            let m = fb.glsl(GlslOp::FAbs, &[v])?;
            let t = srgb_encode(fb, f, m)?;
            let minus = fb.unary(UnOp::FNegate, t)?;
            let zero = fsplat(fb, f, 0.0)?;
            let neg = fb.binary(BinOp::FOrdLessThan, v, zero)?;
            let t = fb.select(neg, minus, t)?;
            colour_lanes(fb, t, v, colour)?
        }
        false => v,
    };
    let lo = (-384.0f64 / 510.0) as f32;
    let hi = (639.0f64 / 510.0) as f32;
    let lo = per_lane_f(fb, [lo, lo, lo, 0.0])?;
    let hi = per_lane_f(fb, [hi, hi, hi, 1.0])?;
    let v = fb.glsl(GlslOp::FClamp, &[v, lo, hi])?;
    match matches!(form, ExtendedRange::Planes { .. }) {
        true => rounded_codes(
            fb,
            v,
            [510.0, 510.0, 510.0, 255.0],
            [384.0, 384.0, 384.0, 0.0],
        ),
        false => rounded_codes(fb, v, [510.0; 4], [384.0; 4]),
    }
}

/// The raw texel (the raw format's `uvec4`) of the codes `code` (red first; a two-plane form's
/// alpha its byte) of a linear or `_sRGB` form, or of `RGB10A2Unorm_sRGB`.
pub(crate) fn pack(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    code: Value,
) -> BResult<Value> {
    let u = words();
    let ut = crate::lower::arith::ops::ty(fb, u)?;
    let planes = matches!(form, ExtendedRange::Planes { .. });
    match form {
        ExtendedRange::Bgr10 { .. } | ExtendedRange::Planes { .. } => {
            // Bits 30 and 31 are written as ones by `BGR10_XR` and as zeros by plane 0 of the
            // two-plane format, as Metal writes them.
            let top = match planes {
                true => fb.u32_const(0)?,
                false => fb.u32_const(0xC000_0000)?,
            };
            let mut w = fb.module().const_value(top)?;
            for (lane, shift) in [(0u32, 20u32), (1, 10), (2, 0)] {
                let c = fb.composite_extract(code, &[lane])?;
                let s = fb.u32_const(shift)?;
                let s = fb.module().const_value(s)?;
                let c = fb.binary(BinOp::ShiftLeftLogical, c, s)?;
                w = fb.binary(BinOp::BitwiseOr, w, c)?;
            }
            let z = fb.u32_const(0)?;
            let z = fb.module().const_value(z)?;
            let alpha = match planes {
                true => fb.composite_extract(code, &[3])?,
                false => z,
            };
            fb.composite_construct(ut, &[w, alpha, z, z])
        }
        ExtendedRange::Bgra10 { .. } => {
            // Blue's field first; each code in its field's top 10 bits.
            let c = fb.vector_shuffle(code, code, &[2, 1, 0, 3])?;
            let six = splat(fb, u, 6)?;
            fb.binary(BinOp::ShiftLeftLogical, c, six)
        }
        // `RGB10A2Unorm_sRGB`'s fields from bit 0, alpha in bits 30 and 31.
        ExtendedRange::Rgb10A2Srgb => {
            let shifts = per_lane(fb, [0, 10, 20, 30])?;
            let k = fb.binary(BinOp::ShiftLeftLogical, code, shifts)?;
            let mut w = fb.composite_extract(k, &[0])?;
            for lane in 1..4u32 {
                let x = fb.composite_extract(k, &[lane])?;
                w = fb.binary(BinOp::BitwiseOr, w, x)?;
            }
            let z = fb.u32_const(0)?;
            let z = fb.module().const_value(z)?;
            fb.composite_construct(ut, &[w, z, z, z])
        }
        // Another padded form's words are [`encode`]'s own, and a bfloat form's.
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb
        | ExtendedRange::R16Bfloat
        | ExtendedRange::Rg16Bfloat
        | ExtendedRange::Rgba16Bfloat => Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        )),
    }
}

/// A `vec4` constant of the four `f32` lanes `v`.
fn per_lane_f(fb: &mut FunctionBuilder<'_>, v: [f32; 4]) -> BResult<Value> {
    let ft = crate::lower::arith::ops::ty(fb, floats())?;
    let e = fb.module().ty(Type::Float { width: 32 })?;
    let mut parts = Vec::with_capacity(4);
    for x in v {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(u64::from(x.to_bits())),
        })?);
    }
    let c = fb.module().constant(Constant::Composite {
        ty: ft,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

#[cfg(test)]
pub(crate) mod tests;
