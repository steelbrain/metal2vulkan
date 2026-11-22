//! Exact conversions of texel values, shared by the sampling lowerings: integer sequences only, so
//! every device and every fast-math mode answers alike.
//!
//! - [`f32_to_half`]: an `f32` rounded to `half` to nearest even, or toward zero saturating at the
//!   largest finite `half`; half denormals are kept.
//!   Metal converts a 32-bit float texel to a `half` result to nearest even when a shader reads
//!   it and toward zero, saturating at `±65504`, when it samples it (authored case
//!   `half-conversion-of-wider-texels-by-read-and-sample`).
//!
//! Metal's texture reads and samples decode a raw sRGB code through its twelve-bit table, as its
//! `air.unpack` does: the step `k` of [`srgb`](crate::lower::arith::srgb), then `k / 4095`
//! correctly rounded (every code of the authored case
//! `half-conversion-of-wider-texels-by-read-and-sample`, and every bicubic sum of
//! `bicubic-rounding-of-random-float-and-srgb-texels`, read those steps).
//!
//! The functions work lane-wise on any number of lanes (`Form`s of 32-bit lanes).

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp};

/// How [`f32_to_half`] rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HalfRounding {
    /// To nearest, ties to even; past the largest finite `half` to the infinity of the sign.
    NearestEven,
    /// Toward zero; a finite value past the largest finite `half` becomes `±65504`.
    TowardZeroSaturating,
}

/// The unsigned 32-bit lanes of `f`.
fn words(f: Form) -> Form {
    f.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

/// The `f32` lanes of `f`.
fn floats(f: Form) -> Form {
    f.with(Held::Float(FloatFmt::F32))
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// The `f32` bits of `h · 2^-24` for an integer `h` (`u32` lanes `u`) at most `2^11`: zero, or
/// the power of two of `h`'s leading bit with the rest of `h` as the significand.
fn grid_bits(fb: &mut FunctionBuilder<'_>, u: Form, h: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let zero = bin(fb, BinOp::IEqual, h, z)?;
    let some = fb.glsl(GlslOp::UMax, &[h, one])?;
    let m = fb.glsl(GlslOp::FindUMsb, &[some])?;
    // The exponent `m - 24`, biased: `m + 103`.
    let k103 = c(fb, 103)?;
    let e = bin(fb, BinOp::IAdd, m, k103)?;
    let k23 = c(fb, 23)?;
    let e = bin(fb, BinOp::ShiftLeftLogical, e, k23)?;
    let up = bin(fb, BinOp::ISub, k23, m)?;
    let s = bin(fb, BinOp::ShiftLeftLogical, some, up)?;
    let frac = c(fb, 0x007F_FFFF)?;
    let s = bin(fb, BinOp::BitwiseAnd, s, frac)?;
    let bits = bin(fb, BinOp::BitwiseOr, e, s)?;
    fb.select(zero, z, bits)
}

/// `x` (`f32` lanes of form `f`) rounded to `half` by `mode`, held as the `f32` of that value.
/// Integer operations on the bits only:
///
/// - An infinity or a NaN is returned as it is.
/// - A magnitude of at least `2^-14` (a normal `half`) rounds its `f32` significand to 10 bits by
///   adding the rounding bias below bit 13 and clearing those bits, so a carry runs into the
///   exponent. A result past `65504` is the infinity, or `65504` when saturating.
/// - A smaller magnitude rounds on the `half` denormal grid `2^-24`: its significand (with the
///   leading one of a normal `f32`) shifted right by `126 - E` for biased exponent `E` (`125` for
///   an `f32` denormal), at most 25, below which every value rounds to zero in every mode. The
///   result keeps the sign, so a negative value that rounds to zero is `-0`.
pub fn f32_to_half(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    mode: HalfRounding,
) -> BResult<Value> {
    to_half(fb, f, x, mode)
}

/// [`f32_to_half`]: the sequence of its docs.
fn to_half(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, rule: HalfRounding) -> BResult<Value> {
    let u = words(f);
    let ut = ty(fb, u)?;
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let sign_bit = c(fb, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, sign_bit)?;
    let abs = c(fb, 0x7FFF_FFFF)?;
    let mag = bin(fb, BinOp::BitwiseAnd, b, abs)?;
    let inf = c(fb, 0x7F80_0000)?;
    let special = bin(fb, BinOp::UGreaterThanEqual, mag, inf)?;
    // A normal half: round at bit 13 of the f32 bits.
    let k13 = c(fb, 13)?;
    let keep = c(fb, 0xFFFF_E000)?;
    let even_bias = |fb: &mut FunctionBuilder<'_>| -> BResult<Value> {
        let lsb = bin(fb, BinOp::ShiftRightLogical, mag, k13)?;
        let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
        let half = c(fb, 0x0FFF)?;
        bin(fb, BinOp::IAdd, half, lsb)
    };
    let biased = match rule {
        HalfRounding::NearestEven => {
            let bias = even_bias(fb)?;
            bin(fb, BinOp::IAdd, mag, bias)?
        }
        HalfRounding::TowardZeroSaturating => mag,
    };
    let normal = bin(fb, BinOp::BitwiseAnd, biased, keep)?;
    let max = c(fb, 0x477F_E000)?;
    let over = bin(fb, BinOp::UGreaterThan, normal, max)?;
    let past = match rule {
        HalfRounding::TowardZeroSaturating => max,
        HalfRounding::NearestEven => inf,
    };
    let normal = fb.select(over, past, normal)?;
    // A denormal half (or zero): round on the grid 2^-24.
    let k23 = c(fb, 23)?;
    let e = bin(fb, BinOp::ShiftRightLogical, mag, k23)?;
    let frac = c(fb, 0x007F_FFFF)?;
    let sig = bin(fb, BinOp::BitwiseAnd, mag, frac)?;
    let denormal_f32 = bin(fb, BinOp::IEqual, e, z)?;
    let lead = c(fb, 0x0080_0000)?;
    let full = bin(fb, BinOp::BitwiseOr, sig, lead)?;
    let sig = fb.select(denormal_f32, sig, full)?;
    // The shift `126 - E` (`125` for E = 0), clamped into [14, 25]; E <= 112 here.
    let k126 = c(fb, 126)?;
    let e1 = fb.glsl(GlslOp::UMax, &[e, one])?;
    let sh = bin(fb, BinOp::ISub, k126, e1)?;
    let k14 = c(fb, 14)?;
    let k25 = c(fb, 25)?;
    let sh = fb.glsl(GlslOp::UClamp, &[sh, k14, k25])?;
    let q = bin(fb, BinOp::ShiftRightLogical, sig, sh)?;
    let unit = bin(fb, BinOp::ShiftLeftLogical, one, sh)?;
    let below = bin(fb, BinOp::ISub, unit, one)?;
    let rem = bin(fb, BinOp::BitwiseAnd, sig, below)?;
    let sh1 = bin(fb, BinOp::ISub, sh, one)?;
    let tie = bin(fb, BinOp::ShiftLeftLogical, one, sh1)?;
    let even_up = |fb: &mut FunctionBuilder<'_>| -> BResult<Value> {
        let gt = bin(fb, BinOp::UGreaterThan, rem, tie)?;
        let eq = bin(fb, BinOp::IEqual, rem, tie)?;
        let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
        let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
        let tie_up = bin(fb, BinOp::LogicalAnd, eq, odd)?;
        bin(fb, BinOp::LogicalOr, gt, tie_up)
    };
    let up = match rule {
        HalfRounding::NearestEven => Some(even_up(fb)?),
        HalfRounding::TowardZeroSaturating => None,
    };
    let h = match up {
        Some(up) => {
            let inc = fb.select(up, one, z)?;
            bin(fb, BinOp::IAdd, q, inc)?
        }
        None => q,
    };
    let denormal = grid_bits(fb, u, h)?;
    let smallest = c(fb, 0x3880_0000)?;
    let small = bin(fb, BinOp::ULessThan, mag, smallest)?;
    let r = fb.select(small, denormal, normal)?;
    let r = bin(fb, BinOp::BitwiseOr, r, sign)?;
    let r = fb.select(special, b, r)?;
    let ft = ty(fb, floats(f))?;
    fb.convert(ConvOp::Bitcast, r, ft)
}
