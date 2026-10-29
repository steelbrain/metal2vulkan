//! A float rounded to the code Metal stores in a color channel, where a translation writes the
//! code itself rather than leaving the conversion to the device.
//!
//! A fragment function's color output is the device's to convert (`LoweringInputs::color_writes`),
//! a unorm channel's too, also where the shader computes a blend in the storage form of framebuffer
//! fetch (`PrecisionNote::DeviceColorConversion`): these rules serve the writes whose value the
//! translation computes, Metal's `textureWriteRoundingMode` 2 (a half-float channel rounded to
//! nearest even), and the private alpha test's 8-bit code ([`unorm8_code`]):
//!
//! - a small float ([`ChannelWrite::FloatTowardZero`]: a half, or an unsigned 11- or 10-bit
//!   float): the value rounded toward zero, by integer operations on the bits, to the `f32` that
//!   holds it exactly: a finite value past the largest is the largest, an infinity stays, a NaN
//!   is the quiet NaN (written as the `f32` NaN `0x7FC00000`), and where unsigned a negative
//!   value is 0;
//! - a small float rounded to nearest even ([`ChannelWrite::FloatNearestEven`], a render
//!   pipeline's half-float channel under Apple's `textureWriteRoundingMode` 2): the same, but the
//!   dropped bits rounded to nearest with ties to even, by integer operations on the bits, and a
//!   finite value from the midpoint past the largest infinity.
//!
//! A channel without a rule ([`ChannelWrite::Device`] or `None`) is written as computed; another
//! rule (a unorm, snorm or sRGB one) is the device's conversion's, which [`round`] does not take.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

/// An upper estimate of the SPIR-V words of one channel's rounding by `rule`.
pub fn channel_words(rule: ChannelWrite) -> u64 {
    match rule {
        ChannelWrite::FloatTowardZero { .. } => 32 * 6,
        // The rounding's bias and the denormal's shift clamped beside the toward-zero rule's.
        ChannelWrite::FloatNearestEven { .. } => 36 * 6,
        // Written as computed, or no rule [`round`] takes.
        _ => 0,
    }
}

/// An upper estimate of the SPIR-V words of rounding an output by `rules`, its extracts and
/// construct included.
pub fn words(rules: &[Option<ChannelWrite>; 4]) -> u64 {
    let written = rules.iter().flatten().count() as u64;
    rules
        .iter()
        .flatten()
        .map(|&r| channel_words(r))
        .sum::<u64>()
        + written.saturating_add(1).saturating_mul(6)
}

fn malformed() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ExtInst,
        rule: OperandRule::Arity,
    }
}

/// Output `x` (float lanes of form `f`, a 32-bit container) with each lane `i` rounded by
/// `rules[i]` (see the module docs): a small float channel's rule, or as computed for
/// [`ChannelWrite::Device`], [`ChannelWrite::BlendedNearestEven`] and `None`. Another rule is
/// malformed.
pub fn round(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    rules: &[Option<ChannelWrite>; 4],
) -> BResult<Value> {
    if f.float_fmt().map(FloatFmt::container) != Some(32) || f.lanes == 0 || f.lanes > 4 {
        return Err(malformed());
    }
    let s = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let mut written = Vec::with_capacity(usize::from(f.lanes));
    for (i, rule) in rules.iter().enumerate().take(usize::from(f.lanes)) {
        let v = if f.lanes == 1 {
            x
        } else {
            fb.composite_extract(x, &[i as u32])?
        };
        written.push(match *rule {
            None | Some(ChannelWrite::Device | ChannelWrite::BlendedNearestEven { .. }) => v,
            Some(
                rule @ (ChannelWrite::FloatTowardZero { .. }
                | ChannelWrite::FloatNearestEven { .. }),
            ) => small_float(fb, s, v, rule)?,
            Some(_) => return Err(malformed()),
        });
    }
    if let [one] = written.as_slice() {
        return Ok(*one);
    }
    let t = ty(fb, f.with(Held::Float(FloatFmt::F32)))?;
    fb.composite_construct(t, &written)
}

/// The `k` of a unorm channel's maximum code `2^k - 1`.
fn unorm_bits(bits: u8) -> BResult<u32> {
    match bits {
        1..=16 => Ok(u32::from(bits)),
        _ => Err(malformed()),
    }
}

/// `a` (`f32` lanes of form `s`, each in `[0, 1]`) times `2^k - 1` (`k` in `1..=16`), rounded to
/// the nearest integer from the exact product, ties to even, as `f32` lanes holding the integer.
/// The product's exact split is `p - w`: `p = fl(2^k a - a)` with `2^k a` exact, and `w = a + (p -
/// 2^k a)` (Fast2Sum: both steps are exact). `p` rounds to nearest even as `r`, `d = p - r` is
/// exact in `[-1/2, 1/2]`, and `r` moves by one where the product's distance `d - w` from `r` is
/// past a half: up where `w < d - 1/2`, down where `w > d + 1/2`, each threshold exact wherever the
/// comparison can go either way (`|w|` is below `2^(k-23)`). The only tie (`a = 1/2`, `w = 0` and
/// `d = ±1/2`) stays at `RoundEven`'s even integer: `2^(k-1)` for `k >= 2`, which is Metal's
/// rounding away from zero there.
///
/// Every step is decorated `NoContraction`, so no device fuses or reassociates the split, and
/// none relies on the rounding mode: an exhaustive check of every `f32` in `[0, 1]` for every
/// `k` matched the exact product rounded to nearest even under rounding to nearest and toward zero,
/// with and without denormals flushed. About fourteen instructions, where
/// [`round_exact_product_lanes`](crate::lower::arith::math::round_exact_product_lanes) takes about
/// forty and a conversion back.
fn round_mersenne_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    a: Value,
    k: u32,
) -> BResult<Value> {
    if !(1..=16).contains(&k) {
        return Err(malformed());
    }
    let scale = fsplat(fb, s, f64::from(1u32 << k))?;
    round_scaled_product(fb, s, a, scale)
}

/// [`round_mersenne_product`] by a scale `scale` (`f32` lanes of form `s`, each `2^k` for a `k`
/// in `1..=16`, a value the shader may compute): `a` times `scale - 1`, rounded to nearest from
/// the exact product, ties to even, by the same instructions, so each lane is exact as the
/// constant scale's is (the split's steps do not depend on `k`'s being constant).
fn round_scaled_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    a: Value,
    scale: Value,
) -> BResult<Value> {
    let big = fb.binary_no_contraction(BinOp::FMul, a, scale)?;
    let p = fb.binary_no_contraction(BinOp::FSub, big, a)?;
    let z = fb.binary_no_contraction(BinOp::FSub, p, big)?;
    // The exact product is `p - w`.
    let w = fb.binary_no_contraction(BinOp::FAdd, a, z)?;
    let r = fb.glsl(GlslOp::RoundEven, &[p])?;
    let d = fb.binary_no_contraction(BinOp::FSub, p, r)?;
    let half = fsplat(fb, s, 0.5)?;
    let below = fb.binary_no_contraction(BinOp::FSub, d, half)?;
    let above = fb.binary_no_contraction(BinOp::FAdd, d, half)?;
    let up = fb.binary(BinOp::FOrdLessThan, w, below)?;
    let down = fb.binary(BinOp::FOrdGreaterThan, w, above)?;
    let one = fsplat(fb, s, 1.0)?;
    let r_up = fb.binary_no_contraction(BinOp::FAdd, r, one)?;
    let r_down = fb.binary_no_contraction(BinOp::FSub, r, one)?;
    let c = fb.select(up, r_up, r)?;
    fb.select(down, r_down, c)
}

/// The 8-bit unorm code of `v` (`f32` lanes of form `s`) as Metal converts it to an 8-bit unorm
/// channel, as `f32` lanes holding the integer: NaN is 0, the value is clamped to `[0, 1]`, and
/// the code is the exact product with 255 rounded to nearest ([`round_mersenne_product`]). The
/// private alpha test compares its alpha so (`stage_io::alpha_test`).
pub(crate) fn unorm8_code(fb: &mut FunctionBuilder<'_>, s: Form, v: Value) -> BResult<Value> {
    let c = clamped(fb, s, v)?;
    round_mersenne_product(fb, s, c, unorm_bits(8)?)
}

/// `v` clamped to `[0, 1]` with NaN taken as 0.
fn clamped(fb: &mut FunctionBuilder<'_>, s: Form, v: Value) -> BResult<Value> {
    let is_nan = fb.unary(UnOp::IsNan, v)?;
    let zero = fsplat(fb, s, 0.0)?;
    let v = fb.select(is_nan, zero, v)?;
    let one = fsplat(fb, s, 1.0)?;
    fb.glsl(GlslOp::NClamp, &[v, zero, one])
}

/// `x` (`f32` lanes of form `f`) each rounded by the small-float rule `rule`
/// ([`ChannelWrite::FloatTowardZero`] or [`ChannelWrite::FloatNearestEven`]), every lane by the
/// same instructions on the whole vector; another rule is malformed.
pub fn small_float(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    rule: ChannelWrite,
) -> BResult<Value> {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return Err(malformed());
    }
    match rule {
        ChannelWrite::FloatTowardZero { mantissa, signed } => {
            float_toward_zero(fb, f, x, mantissa, signed)
        }
        ChannelWrite::FloatNearestEven { mantissa, signed } => {
            float_nearest_even(fb, f, x, mantissa, signed)
        }
        _ => Err(malformed()),
    }
}

/// `v` (`f32` lanes of form `s`) rounded to the nearest `half`, ties to even, held in an `f32`: a
/// NaN stays a NaN and a value past the largest finite `half` an infinity (a packed 16-bit word's
/// channel reads it so, `ChannelWrite::HalfUnorm`).
pub(crate) fn half_rounded(fb: &mut FunctionBuilder<'_>, s: Form, v: Value) -> BResult<Value> {
    float_nearest_even(fb, s, v, 10, true)
}

/// `v` (`f32` lanes of form `s`) rounded to nearest even to a float of a five-bit exponent (bias
/// 15) and `mantissa` fraction bits, signed or not, held in an `f32` (see the module docs): `x`'s
/// bits plus `2^(k-1) - 1` and the lowest kept bit, shifted right by the `k` bits dropped, is the
/// kept value rounded to nearest with ties to even.
fn float_nearest_even(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    mantissa: u8,
    signed: bool,
) -> BResult<Value> {
    let m = match mantissa {
        1..=10 => u32::from(mantissa),
        _ => return Err(malformed()),
    };
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: s.lanes,
    };
    let f = u.with(Held::Float(FloatFmt::F32));
    let k = |fb: &mut FunctionBuilder<'_>, bits: u32| splat(fb, u, u64::from(bits));
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let one = k(fb, 1)?;
    // `x` rounded to nearest even at bit `shift`: plus `2^(shift-1) - 1` and the lowest kept bit.
    let round = |fb: &mut FunctionBuilder<'_>, x: Value, shift: Value| -> BResult<Value> {
        let kept = fb.binary(BinOp::ShiftRightLogical, x, shift)?;
        let lsb = fb.binary(BinOp::BitwiseAnd, kept, one)?;
        let below = fb.binary(BinOp::ISub, shift, one)?;
        let half = fb.binary(BinOp::ShiftLeftLogical, one, below)?;
        let bias = fb.binary(BinOp::ISub, half, one)?;
        let x = fb.binary(BinOp::IAdd, x, bias)?;
        let x = fb.binary(BinOp::IAdd, x, lsb)?;
        fb.binary(BinOp::ShiftRightLogical, x, shift)
    };
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let sign_bit = k(fb, 0x8000_0000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_bit)?;
    let abs_mask = k(fb, 0x7FFF_FFFF)?;
    let a = fb.binary(BinOp::BitwiseAnd, bits, abs_mask)?;
    // A normal value: its bits rounded at the format's last fraction bit, a carry into the
    // exponent included.
    let dropped = k(fb, 23 - m)?;
    let normal = round(fb, a, dropped)?;
    let normal = fb.binary(BinOp::ShiftLeftLogical, normal, dropped)?;
    // A denormal one: its multiples of `2^-(14 + m)`, `mant * 2^(e - 150)` for the biased
    // exponent `e < 113` (an `f32` denormal, `e = 0`, is below 2^-126 and rounds to 0), so
    // `mant` rounded at bit `136 - m - e`, that shift held in [1, 31] (a value of 31 rounds to
    // 0 there, and a larger exponent's lane is the normal one's).
    let k23 = k(fb, 23)?;
    let e = fb.binary(BinOp::ShiftRightLogical, a, k23)?;
    let man_mask = k(fb, 0x7F_FFFF)?;
    let man = fb.binary(BinOp::BitwiseAnd, a, man_mask)?;
    let lead = k(fb, 0x80_0000)?;
    let man = fb.binary(BinOp::BitwiseOr, man, lead)?;
    let base = k(fb, 136 - m)?;
    let shift = fb.binary(BinOp::ISub, base, e)?;
    let k31 = k(fb, 31)?;
    let shift = fb.glsl(GlslOp::UClamp, &[shift, one, k31])?;
    let units = round(fb, man, shift)?;
    let units = fb.convert(ConvOp::ConvertUToF, units, ft)?;
    let step = fsplat(fb, f, 2f64.powi(-14 - i32::from(mantissa)))?;
    let denormal = fb.binary(BinOp::FMul, units, step)?;
    let denormal = fb.convert(ConvOp::Bitcast, denormal, ut)?;
    let smallest_normal = k(fb, 0x3880_0000)?;
    let is_normal = fb.binary(BinOp::UGreaterThanEqual, a, smallest_normal)?;
    let mag = fb.select(is_normal, normal, denormal)?;
    // From 2^16, past the largest finite value rounded up (an infinity included): infinity.
    let past = k(fb, 0x4780_0000)?;
    let infinity = k(fb, 0x7F80_0000)?;
    let is_past = fb.binary(BinOp::UGreaterThanEqual, mag, past)?;
    let mag = fb.select(is_past, infinity, mag)?;
    let out = if signed {
        fb.binary(BinOp::BitwiseOr, sign, mag)?
    } else {
        let zero = k(fb, 0)?;
        let negative = fb.binary(BinOp::INotEqual, sign, zero)?;
        fb.select(negative, zero, mag)?
    };
    let is_nan = fb.binary(BinOp::UGreaterThan, a, infinity)?;
    let quiet = k(fb, 0x7FC0_0000)?;
    let out = fb.select(is_nan, quiet, out)?;
    fb.convert(ConvOp::Bitcast, out, ft)
}

/// `v` (`f32` lanes of form `s`) rounded toward zero to a float of a five-bit exponent (bias 15)
/// and `mantissa` fraction bits, signed or not, held in an `f32` (see the module docs).
fn float_toward_zero(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    mantissa: u8,
    signed: bool,
) -> BResult<Value> {
    let m = match mantissa {
        1..=10 => u32::from(mantissa),
        _ => return Err(malformed()),
    };
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: s.lanes,
    };
    let f = u.with(Held::Float(FloatFmt::F32));
    let k = |fb: &mut FunctionBuilder<'_>, bits: u32| splat(fb, u, u64::from(bits));
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let sign_bit = k(fb, 0x8000_0000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_bit)?;
    let abs_mask = k(fb, 0x7FFF_FFFF)?;
    let a = fb.binary(BinOp::BitwiseAnd, bits, abs_mask)?;
    // A normal value: the significand bits below the format's `m` dropped.
    let keep = k(fb, u32::MAX << (23 - m))?;
    let normal = fb.binary(BinOp::BitwiseAnd, a, keep)?;
    // A denormal one: the whole multiples of `2^-(14 + m)` below the value. The value is `mant *
    // 2^(e - 150)` for the biased exponent `e < 113` (an `f32` denormal, `e = 0`, is below
    // 2^-126 and has none), so `mant >> (136 - m - e)` of them.
    let k23 = k(fb, 23)?;
    let e = fb.binary(BinOp::ShiftRightLogical, a, k23)?;
    let man_mask = k(fb, 0x7F_FFFF)?;
    let man = fb.binary(BinOp::BitwiseAnd, a, man_mask)?;
    let lead = k(fb, 0x80_0000)?;
    let man = fb.binary(BinOp::BitwiseOr, man, lead)?;
    let base = k(fb, 136 - m)?;
    let shift = fb.binary(BinOp::ISub, base, e)?;
    let k31 = k(fb, 31)?;
    let shift = fb.glsl(GlslOp::UMin, &[shift, k31])?;
    let units = fb.binary(BinOp::ShiftRightLogical, man, shift)?;
    let units = fb.convert(ConvOp::ConvertUToF, units, ft)?;
    let step = fsplat(fb, f, 2f64.powi(-14 - i32::from(mantissa)))?;
    let denormal = fb.binary(BinOp::FMul, units, step)?;
    let denormal = fb.convert(ConvOp::Bitcast, denormal, ut)?;
    let smallest_normal = k(fb, 0x3880_0000)?;
    let is_normal = fb.binary(BinOp::UGreaterThanEqual, a, smallest_normal)?;
    let mag = fb.select(is_normal, normal, denormal)?;
    // Past the largest finite value (and infinity and NaN, chosen below): the largest.
    let largest = k(fb, 0x4700_0000 | (((1 << m) - 1) << (23 - m)))?;
    let past = fb.binary(BinOp::UGreaterThanEqual, a, largest)?;
    let mag = fb.select(past, largest, mag)?;
    let infinity = k(fb, 0x7F80_0000)?;
    let is_infinite = fb.binary(BinOp::IEqual, a, infinity)?;
    let mag = fb.select(is_infinite, infinity, mag)?;
    let out = if signed {
        fb.binary(BinOp::BitwiseOr, sign, mag)?
    } else {
        let zero = k(fb, 0)?;
        let negative = fb.binary(BinOp::INotEqual, sign, zero)?;
        fb.select(negative, zero, mag)?
    };
    let is_nan = fb.binary(BinOp::UGreaterThan, a, infinity)?;
    let quiet = k(fb, 0x7FC0_0000)?;
    let out = fb.select(is_nan, quiet, out)?;
    fb.convert(ConvOp::Bitcast, out, ft)
}
