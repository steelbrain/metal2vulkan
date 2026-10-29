//! `air.pack` and `air.unpack` of the normalized formats by the device's arithmetic
//! (`PrecisionNote::DevicePackConversion`).
//!
//! A format `GLSL.std.450` names is its instruction: `unorm4x8`, `snorm4x8`, `unorm2x16` and
//! `snorm2x16` are `Pack`/`Unpack` `Unorm4x8`, `Snorm4x8`, `Unorm2x16` and `Snorm2x16`. The others
//! take the instruction's definition on their own fields: a pack clamps each channel by the
//! device's `NClamp` to `[0, 1]` (`[-1, 1]` signed), multiplies it by `2^N - 1` (`2^(N-1) - 1`),
//! rounds it by `RoundEven` and converts it (`unorm.rgb10a2`, `snorm.rgb10a2`, `unorm.rgb565`), and
//! an unpack multiplies each field's code by the scale's reciprocal, rounded to `float` (signed:
//! the code sign-extended, the product at least `-1`), as Metal's `snorm.rgb10a2` does. `unorm4x8_srgb`'s colour channels take the MSL transfer functions
//! around the 8-bit instruction, by the device's `Pow`: an encode `12.92 c` below `0.0031308` and
//! `1.055 c^(1/2.4) - 0.055` above, a decode `c / 12.92` up to `0.04045` and
//! `((c + 0.055) / 1.055)^2.4` above; alpha is the plain 8-bit channel. A NaN channel is the
//! device's instruction's answer. The packed float formats' unpack moves the fields' bits
//! (`math::unpack_float`, exact: every field is a `half` value).

use crate::air::intrinsics::{PackDir, PackFormat};
use crate::lower::arith::exact_division::per_lane;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::math;
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, ones, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp};

/// The `GLSL.std.450` instruction that is `format`'s conversion in direction `dir`, where one is.
pub fn instruction(format: PackFormat, dir: PackDir) -> Option<GlslOp> {
    Some(match (format, dir) {
        (PackFormat::Unorm4x8, PackDir::Pack) => GlslOp::PackUnorm4x8,
        (PackFormat::Snorm4x8, PackDir::Pack) => GlslOp::PackSnorm4x8,
        (PackFormat::Unorm2x16, PackDir::Pack) => GlslOp::PackUnorm2x16,
        (PackFormat::Snorm2x16, PackDir::Pack) => GlslOp::PackSnorm2x16,
        (PackFormat::Unorm4x8, PackDir::Unpack) => GlslOp::UnpackUnorm4x8,
        (PackFormat::Snorm4x8, PackDir::Unpack) => GlslOp::UnpackSnorm4x8,
        (PackFormat::Unorm2x16, PackDir::Unpack) => GlslOp::UnpackUnorm2x16,
        (PackFormat::Snorm2x16, PackDir::Unpack) => GlslOp::UnpackSnorm2x16,
        _ => return None,
    })
}

/// The error of a conversion whose format or lanes the emission cannot take (a packed float
/// format to pack to, a lane count not the format's): selection rules both out.
fn unpackable() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ExtInst,
        rule: OperandRule::Arity,
    }
}

/// The `float` container form of float lanes `f`, where the conversions compute.
fn container(f: Form) -> Form {
    f.with(Held::Float(FloatFmt::F32))
}

/// `air.pack.<format>`: float lanes `x` (form `f`, held in `float`) to the packed integer (a `u32`;
/// a `ushort` for `rgb565`, zero-extended).
pub fn pack(fb: &mut FunctionBuilder<'_>, format: PackFormat, f: Form, x: Value) -> BResult<Value> {
    let Some((widths, snorm)) = math::pack_layout(format) else {
        return Err(unpackable());
    };
    if usize::from(f.lanes) != widths.len() {
        return Err(unpackable());
    }
    let s = container(f);
    if format == PackFormat::Unorm4x8Srgb {
        let encoded = srgb_colour(fb, s, x, srgb_encode)?;
        return fb.glsl(GlslOp::PackUnorm4x8, &[encoded]);
    }
    if let Some(op) = instruction(format, PackDir::Pack) {
        return fb.glsl(op, &[x]);
    }
    // Each field: the clamped channel times its scale, rounded to nearest even, converted, and
    // (signed) masked to its width, then placed at its offset.
    let u = narrow::u32_form(s);
    let lo = fsplat(fb, s, if snorm { -1.0 } else { 0.0 })?;
    let one = fsplat(fb, s, 1.0)?;
    let v = fb.glsl(GlslOp::NClamp, &[x, lo, one])?;
    let scales: Vec<u64> = widths
        .iter()
        .map(|w| s.const_bits(f64::from(scale(*w, snorm))))
        .collect();
    let m = per_lane(fb, s, &scales)?;
    let p = fb.binary(BinOp::FMul, v, m)?;
    let p = fb.glsl(GlslOp::RoundEven, &[p])?;
    let ut = ty(fb, u)?;
    let op = if snorm {
        ConvOp::ConvertFToS
    } else {
        ConvOp::ConvertFToU
    };
    let r = fb.convert(op, p, ut)?;
    let r = if snorm {
        let m: Vec<u64> = widths.iter().map(|w| ones(*w as u8)).collect();
        let m = per_lane(fb, u, &m)?;
        fb.binary(BinOp::BitwiseAnd, r, m)?
    } else {
        r
    };
    let at = offsets(widths);
    let sh = per_lane(fb, u, &at)?;
    let r = fb.binary(BinOp::ShiftLeftLogical, r, sh)?;
    let mut acc = fb.composite_extract(r, &[0])?;
    for i in 1..widths.len() {
        let field = fb.composite_extract(r, &[i as u32])?;
        acc = fb.binary(BinOp::BitwiseOr, acc, field)?;
    }
    Ok(acc)
}

/// `air.unpack.<format>`: the packed integer `w` (a `u32`; a `ushort` for `rgb565`) to float
/// lanes of form `f`, a narrow form rounding each lane once.
pub fn unpack(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    w: Value,
) -> BResult<Value> {
    if math::float_unpack_lanes(format).is_some() {
        return math::unpack_float(fb, format, f, w);
    }
    let Some((widths, snorm)) = math::pack_layout(format) else {
        return Err(unpackable());
    };
    if usize::from(f.lanes) != widths.len() {
        return Err(unpackable());
    }
    let s = container(f);
    let q = if format == PackFormat::Unorm4x8Srgb {
        let v = fb.glsl(GlslOp::UnpackUnorm4x8, &[w])?;
        srgb_colour(fb, s, v, srgb_decode)?
    } else if let Some(op) = instruction(format, PackDir::Unpack) {
        fb.glsl(op, &[w])?
    } else {
        fields(fb, s, widths, snorm, w)?
    };
    narrow::round(fb, f, q)
}

/// The scale of a channel of `w` bits: `2^w - 1`, or `2^(w-1) - 1` signed.
fn scale(w: u32, snorm: bool) -> u32 {
    if snorm {
        (1 << (w - 1)) - 1
    } else {
        (1 << w) - 1
    }
}

/// The bit offset of each field of `widths`, low field first.
fn offsets(widths: &[u32]) -> Vec<u64> {
    let mut at = Vec::with_capacity(widths.len());
    let mut next = 0u64;
    for w in widths {
        at.push(next);
        next += u64::from(*w);
    }
    at
}

/// The fields of word `w` (formats `widths`, signed where `snorm`) as `float` lanes of form `s`:
/// each code times its scale's reciprocal, the device's multiply, and a signed one at least `-1`.
fn fields(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    widths: &[u32],
    snorm: bool,
    w: Value,
) -> BResult<Value> {
    let u = narrow::u32_form(s);
    let ut = ty(fb, u)?;
    let spread = fb.composite_construct(ut, &vec![w; widths.len()])?;
    let at = offsets(widths);
    let sh = per_lane(fb, u, &at)?;
    let v = fb.binary(BinOp::ShiftRightLogical, spread, sh)?;
    let m: Vec<u64> = widths.iter().map(|n| ones(*n as u8)).collect();
    let m = per_lane(fb, u, &m)?;
    let v = fb.binary(BinOp::BitwiseAnd, v, m)?;
    let st = ty(fb, s)?;
    let c = if snorm {
        let up: Vec<u64> = widths.iter().map(|n| u64::from(32 - n)).collect();
        let up = per_lane(fb, u, &up)?;
        let v = fb.binary(BinOp::ShiftLeftLogical, v, up)?;
        let v = fb.binary(BinOp::ShiftRightArithmetic, v, up)?;
        fb.convert(ConvOp::ConvertSToF, v, st)?
    } else {
        fb.convert(ConvOp::ConvertUToF, v, st)?
    };
    let d: Vec<u64> = widths
        .iter()
        .map(|n| s.const_bits(f64::from(1.0 / scale(*n, snorm) as f32)))
        .collect();
    let d = per_lane(fb, s, &d)?;
    let q = fb.binary(BinOp::FMul, c, d)?;
    if !snorm {
        return Ok(q);
    }
    let m1 = fsplat(fb, s, -1.0)?;
    fb.glsl(GlslOp::NMax, &[q, m1])
}

/// Texel `x` (four `f32` lanes) of 8-bit sRGB codes as a UNORM view reads them (`c / 255`), its
/// lanes where `colour` holds decoded by the MSL transfer ([`srgb_decode`], the device's division
/// and `Pow`) and the others (alpha) kept: an 8-bit sRGB format's decode where the device has no
/// sRGB view of it (a storage image's UNORM alias, a UNORM twin).
pub fn srgb_decode_texel(
    fb: &mut FunctionBuilder<'_>,
    x: Value,
    colour: [bool; 4],
) -> BResult<Value> {
    if !colour.contains(&true) {
        return Ok(x);
    }
    let s = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    };
    let t = srgb_decode(fb, s, x)?;
    let pick: Vec<u32> = (0u32..)
        .zip(colour)
        .map(|(i, c)| if c { i } else { 4 + i })
        .collect();
    fb.vector_shuffle(t, x, &pick)
}

/// The SPIR-V words of [`srgb_decode_texel`], at most: the transfer's steps and their constants,
/// and the shuffle.
pub const SRGB_DECODE_TEXEL_WORDS: u32 = 9 * 6 + 5 * 8 + 12;

/// A transfer function on `float` lanes of form `s`.
type Transfer = fn(&mut FunctionBuilder<'_>, Form, Value) -> BResult<Value>;

/// `x` (four `float` lanes of form `s`) with its colour lanes (red, green, blue) taken through
/// `transfer` and alpha kept.
fn srgb_colour(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    x: Value,
    transfer: Transfer,
) -> BResult<Value> {
    let t = transfer(fb, s, x)?;
    fb.vector_shuffle(t, x, &[0, 1, 2, 7])
}

/// The MSL linear-to-sRGB transfer of `c` (`float` lanes of form `s`): `12.92 c` below
/// `0.0031308`, `1.055 c^(1/2.4) - 0.055` elsewhere, by the device's `Pow`. A lane outside
/// `[0, 1]` is the 8-bit instruction's clamp's.
pub(crate) fn srgb_encode(fb: &mut FunctionBuilder<'_>, s: Form, c: Value) -> BResult<Value> {
    let k = |fb: &mut FunctionBuilder<'_>, v: f64| fsplat(fb, s, v);
    let slope = k(fb, 12.92)?;
    let low = fb.binary(BinOp::FMul, c, slope)?;
    let exponent = k(fb, 1.0 / 2.4)?;
    let p = fb.glsl(GlslOp::Pow, &[c, exponent])?;
    let gain = k(fb, 1.055)?;
    let p = fb.binary(BinOp::FMul, p, gain)?;
    let offset = k(fb, 0.055)?;
    let high = fb.binary(BinOp::FSub, p, offset)?;
    let knee = k(fb, 0.003_130_8)?;
    let below = fb.binary(BinOp::FOrdLessThan, c, knee)?;
    fb.select(below, low, high)
}

/// The MSL sRGB-to-linear transfer of `c` (`float` lanes of form `s`, in `[0, 1]`): `c / 12.92`
/// up to `0.04045`, `((c + 0.055) / 1.055)^2.4` above, by the device's division and `Pow`.
pub(crate) fn srgb_decode(fb: &mut FunctionBuilder<'_>, s: Form, c: Value) -> BResult<Value> {
    let k = |fb: &mut FunctionBuilder<'_>, v: f64| fsplat(fb, s, v);
    let slope = k(fb, 12.92)?;
    let low = fb.binary(BinOp::FDiv, c, slope)?;
    let offset = k(fb, 0.055)?;
    let t = fb.binary(BinOp::FAdd, c, offset)?;
    let gain = k(fb, 1.055)?;
    let t = fb.binary(BinOp::FDiv, t, gain)?;
    let exponent = k(fb, 2.4)?;
    let high = fb.glsl(GlslOp::Pow, &[t, exponent])?;
    let knee = k(fb, 0.040_45)?;
    let below = fb.binary(BinOp::FOrdLessThanEqual, c, knee)?;
    fb.select(below, low, high)
}

/// An upper estimate of the SPIR-V words of [`pack`] or [`unpack`] of `format` on `floats`: the
/// instruction, or the fields' clamp, product, rounding and placements (an unpack's shifts, masks,
/// conversion and division), the sRGB transfer's steps, and a narrow form's rounding.
pub fn words(format: PackFormat, dir: PackDir, floats: Form) -> u32 {
    let lanes = u32::from(floats.lanes);
    let body = if math::float_unpack_lanes(format).is_some() {
        // The packed float formats' bit moves (`math::unpack_float`).
        400
    } else if instruction(format, dir).is_some() {
        16
    } else if format == PackFormat::Unorm4x8Srgb {
        // The 8-bit instruction, the transfer's nine steps and their constants, the shuffle.
        16 + 9 * 6 + 5 * 8 + 12
    } else {
        // The per-lane constants, each step, and each field's extract and placement.
        96 + 24 * lanes
    };
    body + narrow::round_words(floats)
}
