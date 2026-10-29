//! Exact conversions of texel values, shared by the sampling lowerings: integer sequences only, so
//! every device and every fast-math mode answers alike.
//!
//! - [`f32_to_half`]: an `f32` rounded to `half` to nearest even, to nearest with ties away from
//!   zero, or toward zero saturating at the largest finite `half`; half denormals are kept.
//!   Metal converts a 32-bit float texel to a `half` result to nearest even when a shader reads
//!   it and toward zero, saturating at `±65504`, when it samples it (authored case
//!   `half-conversion-of-wider-texels-by-read-and-sample`).
//! - [`ratio_to_f32`] and [`ratio_to_half`]: `k / d` correctly rounded to `f32` or to `half`
//!   (nearest even) for a divisor known only at run time, as a normalized channel's value `n / D`
//!   on a format's grid is (the 8- and 16-bit normalized formats' bicubic sums). [`ratio_to_half`]
//!   rounds once: rounding the `f32` quotient to `half` instead differs from it at `±65455 /
//!   65535`, `±65519 / 65535`, and `±32759 / 32767`.
//! - [`ratio_by_255_to_f32`]: `m / 255` correctly rounded to `f32` by one multiplication and an
//!   exact integer correction, the extended-range grids' quotient (`255 · 2^k`).
//! - [`normal_half_bits`]: a non-negative `f32` rounded to a normal `half`, to nearest even or
//!   with ties away, by an addition and a mask.
//! - [`f32_to_half_from_even`]: toward zero saturating or ties away, derived from an exact
//!   rounding to nearest even.
//! - [`grid_value`]: a signed step's value `n / D` on a normalized format's grid exactly, the
//!   bicubic sums'.
//!
//! Metal's texture reads and samples decode a raw sRGB code through its twelve-bit table, as
//! `air.unpack` does: the step `k` of [`srgb`](crate::lower::arith::srgb), then `k / 4095`
//! ([`div_exact`](crate::lower::arith::exact_division::div_exact); every code of the authored case
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
    /// To nearest, ties away from zero; past the largest finite `half` to the infinity.
    NearestAway,
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

/// `x` (`f32` lanes of form `f`) rounded to `half` toward zero, saturating at `±65504`, where
/// `toward_zero` (a `bool` scalar), else to nearest with ties away from zero, both derived from
/// `even`, `x` rounded to nearest even. `even` must be exact: the nearest-even `half` of every
/// finite `x` (either sign of a zero), an infinity's infinity, and a NaN `x` itself, as
/// [`round_texel`](crate::lower::arith::narrow::round_texel) gives where the device's rounding is
/// measured to be Metal's. The result is exact by construction:
///
/// - Each directed result is `even` or `o`, `even`'s neighbour on `x`'s side (its predecessor in
///   magnitude where `|even| > |x|`, else its successor). Toward zero takes `o` where `even` lies
///   past `x`, saturated at `65504` (a finite overflow's infinity steps back to the largest
///   `half`); ties away take the larger of `even` and `o` where `x` is their midpoint.
/// - Above the least normal `half` `2^-14`, `o` and the midpoint are `even`'s `f32` bits
///   `± 0x2000` and `± 0x1000`: a carry or borrow through the exponent lands on the next binade's
///   `half`, or its midpoint, exactly. At or below it they are `|even| ± 2^-24` and `± 2^-25` by
///   one `f32` addition each, whose operands and result lie on the `2^-25` grid below `2^-13`, so
///   exact in every rounding mode, and normal `f32`s or zero, so no mode flushes them. No
///   multiplication feeds them, so nothing contracts.
/// - Every test is on integer bits and the sign of `x` is set last, so no float mode changes the
///   answer: an infinity or a NaN is `even`, which is `x`.
///
/// About thirty instructions.
pub fn f32_to_half_from_even(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    (x, even): (Value, Value),
    toward_zero: Value,
) -> BResult<Value> {
    let u = words(f);
    let ut = ty(fb, u)?;
    let ft = ty(fb, floats(f))?;
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let fc = |fb: &mut FunctionBuilder<'_>, v: f32| splat(fb, floats(f), u64::from(v.to_bits()));
    let abs = c(fb, 0x7FFF_FFFF)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let mag = bin(fb, BinOp::BitwiseAnd, b, abs)?;
    let sign_bit = c(fb, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, sign_bit)?;
    let e = fb.convert(ConvOp::Bitcast, even, ut)?;
    let e = bin(fb, BinOp::BitwiseAnd, e, abs)?;
    // `even` past `x`: its neighbour toward zero, else away from it.
    let past = bin(fb, BinOp::UGreaterThan, e, mag)?;
    // Normal halves: on the bits.
    let (down, up) = (c(fb, 0xFFFF_E000)?, c(fb, 0x2000)?);
    let step = fb.select(past, down, up)?;
    let o_normal = bin(fb, BinOp::IAdd, e, step)?;
    let (half_down, half_up) = (c(fb, 0xFFFF_F000)?, c(fb, 0x1000)?);
    let half_step = fb.select(past, half_down, half_up)?;
    let mid_normal = bin(fb, BinOp::IAdd, e, half_step)?;
    // Half denormals and the least normal half: on the `2^-24` grid, by exact `f32` additions.
    let ef = fb.convert(ConvOp::Bitcast, e, ft)?;
    let (down, up) = (fc(fb, -(2f32.powi(-24)))?, fc(fb, 2f32.powi(-24))?);
    let step = fb.select(past, down, up)?;
    let o_low = bin(fb, BinOp::FAdd, ef, step)?;
    let o_low = fb.convert(ConvOp::Bitcast, o_low, ut)?;
    // `2^-24 - 2^-24` is `+0` in every rounding mode; the mask keeps it so wherever a mode might
    // not.
    let o_low = bin(fb, BinOp::BitwiseAnd, o_low, abs)?;
    let (half_down, half_up) = (fc(fb, -(2f32.powi(-25)))?, fc(fb, 2f32.powi(-25))?);
    let half_step = fb.select(past, half_down, half_up)?;
    let mid_low = bin(fb, BinOp::FAdd, ef, half_step)?;
    let mid_low = fb.convert(ConvOp::Bitcast, mid_low, ut)?;
    let least_normal = c(fb, 0x3880_0000)?;
    let low = bin(fb, BinOp::ULessThanEqual, e, least_normal)?;
    let o = fb.select(low, o_low, o_normal)?;
    let mid = fb.select(low, mid_low, mid_normal)?;
    // Toward zero: `o` where `even` is past `x`, saturated. Ties away: the larger where `x` is
    // the midpoint.
    let tie = bin(fb, BinOp::IEqual, mid, mag)?;
    let moves = fb.select(toward_zero, past, tie)?;
    let max = c(fb, 0x477F_E000)?;
    let saturated = fb.glsl(GlslOp::UMin, &[o, max])?;
    let larger = fb.glsl(GlslOp::UMax, &[o, e])?;
    let moved = fb.select(toward_zero, saturated, larger)?;
    let r = fb.select(moves, moved, e)?;
    let r = bin(fb, BinOp::BitwiseOr, r, sign)?;
    fb.convert(ConvOp::Bitcast, r, ft)
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
        HalfRounding::NearestAway => {
            let half = c(fb, 0x1000)?;
            bin(fb, BinOp::IAdd, mag, half)?
        }
        HalfRounding::TowardZeroSaturating => mag,
    };
    let normal = bin(fb, BinOp::BitwiseAnd, biased, keep)?;
    let max = c(fb, 0x477F_E000)?;
    let over = bin(fb, BinOp::UGreaterThan, normal, max)?;
    let past = match rule {
        HalfRounding::TowardZeroSaturating => max,
        HalfRounding::NearestEven | HalfRounding::NearestAway => inf,
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
        HalfRounding::NearestAway => Some(bin(fb, BinOp::UGreaterThanEqual, rem, tie)?),
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

/// The leading part of [`ratio_to_f32`] and [`ratio_to_half`]: for `1 <= k <= d < 2^16`, the
/// shift `e >= 0` that puts `a = k · 2^e` in `[d, 2d)`, and `a`.
fn normalize(fb: &mut FunctionBuilder<'_>, u: Form, k: Value, d: Value) -> BResult<(Value, Value)> {
    let one = splat(fb, u, 1)?;
    let m = fb.glsl(GlslOp::FindUMsb, &[k])?;
    let top = fb.glsl(GlslOp::FindUMsb, &[d])?;
    let e = bin(fb, BinOp::ISub, top, m)?;
    let a = bin(fb, BinOp::ShiftLeftLogical, k, e)?;
    let short = bin(fb, BinOp::ULessThan, a, d)?;
    let a2 = bin(fb, BinOp::ShiftLeftLogical, a, one)?;
    let a = fb.select(short, a2, a)?;
    let e1 = bin(fb, BinOp::IAdd, e, one)?;
    let e = fb.select(short, e1, e)?;
    Ok((e, a))
}

/// Whether quotient `q` with remainder `r` of a division by `d` rounds up to nearest even.
fn round_up(fb: &mut FunctionBuilder<'_>, u: Form, q: Value, r: Value, d: Value) -> BResult<Value> {
    let (z, one) = (splat(fb, u, 0)?, splat(fb, u, 1)?);
    let twice = bin(fb, BinOp::ShiftLeftLogical, r, one)?;
    let above = bin(fb, BinOp::UGreaterThan, twice, d)?;
    let tie = bin(fb, BinOp::IEqual, twice, d)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
    let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, tie, odd)?;
    bin(fb, BinOp::LogicalOr, above, tie_up)
}

/// `k / d` correctly rounded to `f32` (nearest even), for `u32` lanes (form `u`) with
/// `0 <= k <= d` and `1 <= d < 2^16`, the divisor a run-time value:
/// [`div_exact`](crate::lower::arith::exact_division::div_exact)'s sequence, the divisor's
/// leading bit found at run time. `k = 0` is `+0`.
pub fn ratio_to_f32(fb: &mut FunctionBuilder<'_>, u: Form, k: Value, d: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let zero = bin(fb, BinOp::IEqual, k, z)?;
    let k = fb.glsl(GlslOp::UMax, &[k, one])?;
    let (e, a) = normalize(fb, u, k, d)?;
    // q = floor(a · 2^23 / d) in two exact divisions (a · 2^12 < 2^29).
    let (k11, k12) = (c(fb, 11)?, c(fb, 12)?);
    let hi = bin(fb, BinOp::ShiftLeftLogical, a, k12)?;
    let q1 = bin(fb, BinOp::UDiv, hi, d)?;
    let t = bin(fb, BinOp::IMul, q1, d)?;
    let r1 = bin(fb, BinOp::ISub, hi, t)?;
    let lo = bin(fb, BinOp::ShiftLeftLogical, r1, k11)?;
    let q2 = bin(fb, BinOp::UDiv, lo, d)?;
    let t = bin(fb, BinOp::IMul, q2, d)?;
    let r2 = bin(fb, BinOp::ISub, lo, t)?;
    let q = bin(fb, BinOp::ShiftLeftLogical, q1, k11)?;
    let q = bin(fb, BinOp::BitwiseOr, q, q2)?;
    let up = round_up(fb, u, q, r2, d)?;
    let inc = fb.select(up, one, z)?;
    let k126 = c(fb, 126)?;
    let be = bin(fb, BinOp::ISub, k126, e)?;
    let k23 = c(fb, 23)?;
    let be = bin(fb, BinOp::ShiftLeftLogical, be, k23)?;
    let bits = bin(fb, BinOp::IAdd, be, q)?;
    let bits = bin(fb, BinOp::IAdd, bits, inc)?;
    let bits = fb.select(zero, z, bits)?;
    let ft = ty(fb, floats(u))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}

/// `m / 255` correctly rounded to `f32` (nearest even), for `u32` lanes (form `u`) with
/// `m < 2^24`, in a third of [`ratio_to_f32`]'s operations: the product `q₀` of `m` (exact as an
/// `f32`) and `1/255` rounded to `f32`, each of its two roundings within `2^-24` of the value, so
/// `q₀` lies within two units in the last place of the quotient, then corrected by the exact
/// residual. With `q₀ = Q · 2^-s` (`Q` its significand, `2^23 <= Q < 2^24`), `m · 2^s` is
/// within `2 · 255` of `255 Q`, below `2^32`, so the residual `R = m · 2^s - 255 Q` is exact in
/// one word, and the quotient's significand is `Q + R / 255`: it rounds to `Q + k`, `k` the
/// integer nearest `R / 255` (never a tie: `m / 255` is dyadic only where it is an integer), the
/// `f32` whose bits are `q₀`'s plus `k`. The one quotient that crosses a binade, a power of two
/// (`m = 255 · 2^j`), does so from just below it by `k = 1`, which the bits' carry gives. `m = 0`
/// is `+0`. The product takes `NoContraction`.
pub fn ratio_by_255_to_f32(fb: &mut FunctionBuilder<'_>, u: Form, m: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let f = floats(u);
    let ft = ty(fb, f)?;
    let x = fb.convert(ConvOp::ConvertUToF, m, ft)?;
    let r = splat(fb, f, u64::from((1.0f32 / 255.0).to_bits()))?;
    let q0 = fb.binary_no_contraction(BinOp::FMul, x, r)?;
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, q0, ut)?;
    // q₀ = Q · 2^(e - 150) for its biased exponent e.
    let (k23, k150, k31) = (c(fb, 23)?, c(fb, 150)?, c(fb, 31)?);
    let e = bin(fb, BinOp::ShiftRightLogical, bits, k23)?;
    let (low, hidden) = (c(fb, 0x7F_FFFF)?, c(fb, 0x80_0000)?);
    let q = bin(fb, BinOp::BitwiseAnd, bits, low)?;
    let q = bin(fb, BinOp::BitwiseOr, q, hidden)?;
    let s = bin(fb, BinOp::ISub, k150, e)?;
    // Zero's shift is past the word; its lane is +0 below.
    let s = fb.glsl(GlslOp::UMin, &[s, k31])?;
    let scaled = bin(fb, BinOp::ShiftLeftLogical, m, s)?;
    let k255 = c(fb, 255)?;
    let q255 = bin(fb, BinOp::IMul, q, k255)?;
    let residual = bin(fb, BinOp::ISub, scaled, q255)?;
    // k = [R >= 128] + [R >= 383] - [R <= -128] - [R <= -383], R signed.
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let mut bits = bits;
    for (op, bound, add) in [
        (BinOp::SGreaterThanEqual, 128i32, true),
        (BinOp::SGreaterThanEqual, 383, true),
        (BinOp::SLessThanEqual, -128, false),
        (BinOp::SLessThanEqual, -383, false),
    ] {
        let b = c(fb, u64::from(bound as u32))?;
        let past = bin(fb, op, residual, b)?;
        let step = fb.select(past, one, z)?;
        let op = if add { BinOp::IAdd } else { BinOp::ISub };
        bits = bin(fb, op, bits, step)?;
    }
    let zero = bin(fb, BinOp::IEqual, m, z)?;
    let bits = fb.select(zero, z, bits)?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}

/// The `half` nearest the non-negative `f32` whose bits are `bits` (`u32` lanes of form `u`),
/// held as its `f32` bits, for a value that rounds to a normal `half` or is zero: to nearest even,
/// or with ties away from zero on the lanes where `away` (`bool` lanes) holds. The significand
/// keeps its top ten fraction bits: half a `half` unit (`2^12` in the `f32` bits), less one for an
/// even kept bit, is added before the low thirteen bits are cleared (only half a unit, for ties
/// away), so a carry rounds into the exponent. Not for a value whose `half` is a denormal or past
/// the largest finite `half` ([`f32_to_half`]).
pub fn normal_half_bits(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    bits: Value,
    away: Value,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let (k13, one) = (c(fb, 13)?, c(fb, 1)?);
    let kept = bin(fb, BinOp::ShiftRightLogical, bits, k13)?;
    let odd = bin(fb, BinOp::BitwiseAnd, kept, one)?;
    let (below, tie) = (c(fb, 0xFFF)?, c(fb, 0x1000)?);
    let even = bin(fb, BinOp::IAdd, below, odd)?;
    let bias = fb.select(away, tie, even)?;
    let up = bin(fb, BinOp::IAdd, bits, bias)?;
    let mask = c(fb, 0xFFFF_E000)?;
    bin(fb, BinOp::BitwiseAnd, up, mask)
}

/// `k / d` correctly rounded to `half` (nearest even), held as its `f32`, for `u32` lanes (form
/// `u`) with `0 <= k <= d` and `1 <= d < 2^16`, the divisor a run-time value. `k = 0` is `+0`.
///
/// A quotient of at least `2^-14` takes the 11-bit significand `floor(a · 2^10 / d)` of the
/// normalized `a` (`normalize`, one exact division: `a · 2^10 < 2^27`) and rounds it on the
/// remainder; the rounding carries into the exponent. A smaller one (then `k <= 3`) is
/// `k · 2^24 / d` rounded to an integer, its value on the denormal grid `2^-24`: at most `2^10`,
/// to which the step is clamped, so a lane that takes the normal quotient keeps the denormal
/// one's shifts in range (a grid `d <= 3` puts `3 · 2^24 / d` at `2^24` or more).
pub fn ratio_to_half(fb: &mut FunctionBuilder<'_>, u: Form, k: Value, d: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let zero = bin(fb, BinOp::IEqual, k, z)?;
    let kk = fb.glsl(GlslOp::UMax, &[k, one])?;
    let (e, a) = normalize(fb, u, kk, d)?;
    let k10 = c(fb, 10)?;
    let hi = bin(fb, BinOp::ShiftLeftLogical, a, k10)?;
    let q = bin(fb, BinOp::UDiv, hi, d)?;
    let t = bin(fb, BinOp::IMul, q, d)?;
    let r = bin(fb, BinOp::ISub, hi, t)?;
    let up = round_up(fb, u, q, r, d)?;
    let inc = fb.select(up, one, z)?;
    // Bits: exponent 127 - e, significand (q - 2^10) << 13, the rounding added below it.
    let k126 = c(fb, 126)?;
    let be = bin(fb, BinOp::ISub, k126, e)?;
    let k23 = c(fb, 23)?;
    let be = bin(fb, BinOp::ShiftLeftLogical, be, k23)?;
    let sig = bin(fb, BinOp::IAdd, q, inc)?;
    let k13 = c(fb, 13)?;
    let sig = bin(fb, BinOp::ShiftLeftLogical, sig, k13)?;
    let normal = bin(fb, BinOp::IAdd, be, sig)?;
    // Below 2^-14 (e >= 15): k <= 3, so k · 2^24 < 2^26.
    let k3 = c(fb, 3)?;
    let small_k = fb.glsl(GlslOp::UMin, &[kk, k3])?;
    let k24 = c(fb, 24)?;
    let n = bin(fb, BinOp::ShiftLeftLogical, small_k, k24)?;
    let g = bin(fb, BinOp::UDiv, n, d)?;
    let t = bin(fb, BinOp::IMul, g, d)?;
    let gr = bin(fb, BinOp::ISub, n, t)?;
    let gup = round_up(fb, u, g, gr, d)?;
    let ginc = fb.select(gup, one, z)?;
    let g = bin(fb, BinOp::IAdd, g, ginc)?;
    let k1024 = c(fb, 1024)?;
    let g = fb.glsl(GlslOp::UMin, &[g, k1024])?;
    let denormal = grid_bits(fb, u, g)?;
    let k15 = c(fb, 15)?;
    let small = bin(fb, BinOp::UGreaterThanEqual, e, k15)?;
    let bits = fb.select(small, denormal, normal)?;
    let bits = fb.select(zero, z, bits)?;
    let ft = ty(fb, floats(u))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}

/// The value `n / D` of signed grid step `n` (`i32` bits in `u32` lanes of form `u`, within
/// `[-D, D]`; `d` the grid `D` in `u32` lanes, `1 <= D < 2^16`): `+0` for zero, else the
/// magnitude `|n| / D` correctly rounded by `round` ([`ratio_to_f32`], [`ratio_to_half`], or a
/// choice of the two) with the sign of `n`.
pub fn grid_value(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    n: Value,
    d: Value,
    round: impl FnOnce(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>,
) -> BResult<Value> {
    let z = splat(fb, u, 0)?;
    let neg = bin(fb, BinOp::SLessThan, n, z)?;
    let m = fb.glsl(GlslOp::SAbs, &[n])?;
    let r = round(fb, m, d)?;
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, r, ut)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let sign = fb.select(neg, sign, z)?;
    let bits = bin(fb, BinOp::BitwiseOr, bits, sign)?;
    let ft = ty(fb, floats(u))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}
