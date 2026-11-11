//! `air.convert` with a rounding after its types (`.rte`, `.rtz`, `.rtp`, `.rtn`) and `.sat` after
//! it: the forms Apple's OpenCL compiler emits for every `convert_<type>[_sat][_<mode>]`.
//!
//! - **To an integer** from a float: the float rounded to an integral value first ([`Integral`]:
//!   the conversion's own truncation, or `RoundEven`, `Ceil` or `Floor`, each exact), then the
//!   saturating conversion every `air.convert` to an integer takes (`math::ConvertKind::ToInt`).
//!   `.sat` adds nothing to it.
//! - **To an integer** from an integer: a rounding is nothing to an integer, and `.sat` clamps to
//!   the range both types share ([`saturate`], [`sat_bounds`]).
//! - **To a float**, rounding to nearest even (`.rte`): the conversion every `air.convert` takes,
//!   which rounds that way where the entry declares `RoundingModeRTE` for the result's width
//!   (`narrow::rte_modes`) and else is the device's choice of the two neighbours
//!   (`PrecisionNote::NativeConversionRounding`). A conversion that cannot round needs neither.
//! - **To a float**, rounding toward zero or an infinity ([`Directed`]): Vulkan has no such
//!   conversion (its `FPRoundingMode` decoration is for 16-bit stores only, and an entry's
//!   `RoundingModeRTZ` would round its arithmetic too), so the result is computed exactly by
//!   integer arithmetic on the operand's bits ([`int_to_float`], [`narrow_float`]); a conversion
//!   that cannot round is the plain one. `air.convert` names no `double`.

use crate::air::intrinsics::ConvertRounding;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{self, fsplat, mask, sext, splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

/// How a float is rounded to an integral value before `air.convert`'s saturating conversion to an
/// integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Integral {
    /// The conversion's own truncation: `rtz`, and the conversion without a rounding.
    Truncate,
    /// `RoundEven` first: `rte`.
    Even,
    /// `Ceil` first: `rtp`.
    Up,
    /// `Floor` first: `rtn`.
    Down,
}

impl Integral {
    /// The rounding a conversion to an integer names.
    pub fn of(rounding: Option<ConvertRounding>) -> Integral {
        match rounding {
            None | Some(ConvertRounding::Rtz) => Integral::Truncate,
            Some(ConvertRounding::Rte) => Integral::Even,
            Some(ConvertRounding::Rtp) => Integral::Up,
            Some(ConvertRounding::Rtn) => Integral::Down,
        }
    }
}

/// `x` (a float of any form) rounded to an integral value by `k`: the device's `RoundEven`,
/// `Ceil` or `Floor`, each exact; `x` itself for [`Integral::Truncate`].
pub fn integral(fb: &mut FunctionBuilder<'_>, k: Integral, x: Value) -> BResult<Value> {
    let op = match k {
        Integral::Truncate => return Ok(x),
        Integral::Even => GlslOp::RoundEven,
        Integral::Up => GlslOp::Ceil,
        Integral::Down => GlslOp::Floor,
    };
    fb.glsl(op, &[x])
}

/// A rounding of a conversion to a float toward zero or an infinity, which Vulkan gives no
/// conversion of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Directed {
    /// `rtz`: toward zero.
    Zero,
    /// `rtp`: toward positive infinity.
    Up,
    /// `rtn`: toward negative infinity.
    Down,
}

impl Directed {
    /// The directed rounding a conversion to a float names, or `None` for nearest even and for
    /// the conversion without a rounding.
    pub fn of(rounding: Option<ConvertRounding>) -> Option<Directed> {
        match rounding {
            None | Some(ConvertRounding::Rte) => None,
            Some(ConvertRounding::Rtz) => Some(Directed::Zero),
            Some(ConvertRounding::Rtp) => Some(Directed::Up),
            Some(ConvertRounding::Rtn) => Some(Directed::Down),
        }
    }
}

/// Where a magnitude rounds away from zero: never, always, or in the lanes a `bool` value holds.
#[derive(Clone, Copy)]
enum Away {
    Never,
    Always,
    Where(Value),
}

/// Where a lane rounded `dir` moves its magnitude away from zero, given which lanes are negative
/// (`None` for an unsigned source, never negative).
fn away(fb: &mut FunctionBuilder<'_>, dir: Directed, neg: Option<Value>) -> BResult<Away> {
    Ok(match (dir, neg) {
        (Directed::Zero, _) | (Directed::Down, None) => Away::Never,
        (Directed::Up, None) => Away::Always,
        (Directed::Up, Some(n)) => Away::Where(fb.unary(UnOp::LogicalNot, n)?),
        (Directed::Down, Some(n)) => Away::Where(n),
    })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// A pair of forms selection never hands a directed conversion.
fn unconverted() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ConvertUToF,
        rule: OperandRule::Width(0),
    }
}

/// `t` (`m >> s`, lanes of form `f`) plus one in the lanes that round away and lost a set bit of
/// `m` to the shift.
fn bump(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    (m, t, s): (Value, Value, Value),
    away: Away,
) -> BResult<Value> {
    let lost = match away {
        Away::Never => return Ok(t),
        _ => {
            let kept = bin(fb, BinOp::ShiftLeftLogical, t, s)?;
            bin(fb, BinOp::INotEqual, kept, m)?
        }
    };
    let up = match away {
        Away::Where(c) => bin(fb, BinOp::LogicalAnd, c, lost)?,
        _ => lost,
    };
    let (one, zero) = (splat(fb, f, 1)?, splat(fb, f, 0)?);
    let d = fb.select(up, one, zero)?;
    bin(fb, BinOp::IAdd, t, d)
}

/// The significand bits of float format `fmt` (the leading one counted).
fn significand(fmt: FloatFmt) -> u8 {
    match fmt {
        FloatFmt::Half => 11,
        FloatFmt::BFloat => 8,
        FloatFmt::F32 => 24,
        FloatFmt::F64 => 53,
    }
}

/// Whether every integer of form `from` (read as signed when `signed`) is a value of float format
/// `to`, so no conversion to it rounds: its magnitude has at most the format's significand bits
/// (`half`'s largest such integer, 2048, is within its range).
pub fn converts_exactly(from: Form, signed: bool, to: FloatFmt) -> bool {
    let magnitude = if signed {
        from.bits().saturating_sub(1)
    } else {
        from.bits()
    };
    magnitude <= significand(to)
}

/// Unsigned magnitude `m` (`u32` lanes of form `u`) as an `f32` (lanes of form `to`) rounded
/// toward zero, or away from it where `away`: the bits below the 24 the significand keeps are
/// dropped, and one is added to what is kept where any of them was set and the lane rounds away.
/// The kept part (at most `2^24`) converts exactly and scales back exactly by `2^s`.
fn magnitude32(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    to: Form,
    m: Value,
    away: Away,
) -> BResult<Value> {
    let msb = fb.glsl(GlslOp::FindUMsb, &[m])?;
    let k23 = splat(fb, u, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let z = splat(fb, u, 0)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s)?;
    let t = bump(fb, u, (m, t, s), away)?;
    let ft = ty(fb, to.with(Held::Float(FloatFmt::F32)))?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// Unsigned magnitude `m` (`u32` lanes of form `u`) of an integer that is negative in the lanes
/// `neg` holds (`None`: none) as the `f32` (lanes of form `to`) of its magnitude rounded `dir`, its
/// sign not applied ([`magnitude32`]).
pub fn magnitude_f32(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    to: Form,
    (m, neg): (Value, Option<Value>),
    dir: Directed,
) -> BResult<Value> {
    let a = away(fb, dir, neg)?;
    magnitude32(fb, u, to, m, a)
}

/// Unsigned 64-bit magnitude `m` (`u64` lanes of form `w`) as an `f32` (lanes of form `to`)
/// rounded toward zero, or away from it where `away`, as [`magnitude32`] does for 32 bits: the most
/// significant bit is found from the two halves, the bits below the 24 the significand keeps are
/// dropped (one added where any was set and the lane rounds away), and the kept part converts
/// exactly and scales back exactly.
fn magnitude64(
    fb: &mut FunctionBuilder<'_>,
    w: Form,
    to: Form,
    m: Value,
    away: Away,
) -> BResult<Value> {
    if to.float_fmt() != Some(FloatFmt::F32) {
        return Err(unconverted());
    }
    let keep = u64::from(significand(FloatFmt::F32));
    let u = u32_form(w);
    let ut = ty(fb, u)?;
    let wt = ty(fb, w)?;
    let lo = fb.convert(ConvOp::UConvert, m, ut)?;
    let k32w = splat(fb, w, 32)?;
    let hi = bin(fb, BinOp::ShiftRightLogical, m, k32w)?;
    let hi = fb.convert(ConvOp::UConvert, hi, ut)?;
    let mlo = fb.glsl(GlslOp::FindUMsb, &[lo])?;
    let mhi = fb.glsl(GlslOp::FindUMsb, &[hi])?;
    let k32 = splat(fb, u, 32)?;
    let mhi = bin(fb, BinOp::IAdd, mhi, k32)?;
    let z = splat(fb, u, 0)?;
    let hz = bin(fb, BinOp::IEqual, hi, z)?;
    let msb = fb.select(hz, mlo, mhi)?;
    let below = splat(fb, u, keep - 1)?;
    let s = bin(fb, BinOp::ISub, msb, below)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let s64 = fb.convert(ConvOp::UConvert, s, wt)?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s64)?;
    let t = bump(fb, w, (m, t, s64), away)?;
    let ft = ty(fb, to)?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// Integer `x` of form `from` (read as signed when `signed`) converted to float form `to` (`half`
/// or `float`), rounded `dir`, exactly on every device.
///
/// A `half` is the integer's `f32` (exact below `2^24`, and at least `2^24` elsewhere, past
/// `half`'s range whichever way the device rounds it) rounded to `half` by [`to_half`]. A `float`
/// is the magnitude rounded by integer arithmetic ([`magnitude32`], [`magnitude64`]), given its
/// sign.
pub fn int_to_float(
    fb: &mut FunctionBuilder<'_>,
    (signed, dir): (bool, Directed),
    from: Form,
    to: Form,
    x: Value,
) -> BResult<Value> {
    if to.float_fmt() == Some(FloatFmt::Half) {
        let f32f = to.with(Held::Float(FloatFmt::F32));
        let v = ops::int_to_float(fb, signed, from, f32f, x)?;
        return to_half(fb, f32f, v, dir);
    }
    let (m, neg) = if signed {
        let s = sext(fb, from, x)?;
        let z = splat(fb, from, 0)?;
        let neg = bin(fb, BinOp::SLessThan, s, z)?;
        (fb.glsl(GlslOp::SAbs, &[s])?, Some(neg))
    } else {
        (x, None)
    };
    let a = away(fb, dir, neg)?;
    let r = if from.container() == 64 {
        let w = from.with(Held::Int {
            bits: 64,
            container: 64,
        });
        magnitude64(fb, w, to, m, a)?
    } else if to.float_fmt() == Some(FloatFmt::F32) {
        magnitude32(fb, u32_form(from), to, m, a)?
    } else {
        return Err(unconverted());
    };
    match neg {
        Some(n) => {
            let nr = fb.unary(UnOp::FNegate, r)?;
            fb.select(n, nr, r)
        }
        None => Ok(r),
    }
}

/// The widened `half` NaN of a conversion that rounds a NaN `float` to `half` here: the canonical
/// `0x7e00`, Metal's answer to a `float` narrowed to `half` (`nan::convert_answer`).
const HALF_NAN: u64 = 0x7FC0_0000;

/// `x` (`f32` lanes of form `f`) rounded to `half` `dir`, held widened in `f32`, by integer
/// arithmetic on its bits, exactly on every device: a normal `half` keeps the top 11 of the 24
/// significand bits (adding one unit below them first where the lane rounds away), a result past
/// 65504 is the infinity when the lane rounds away and 65504 when it does not, and below `2^-14`
/// the value is the multiple of `2^-24` the exact significand gives, shifted right and rounded the
/// same way (an `f32` denormal operand counted by its value). A zero's sign and an infinity are
/// kept; a NaN is [`HALF_NAN`].
pub fn to_half(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, dir: Directed) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let sign_mask = splat(fb, u, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, sign_mask)?;
    let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
    let a = bin(fb, BinOp::BitwiseAnd, b, abs_mask)?;
    let z = splat(fb, u, 0)?;
    let neg = bin(fb, BinOp::INotEqual, sign, z)?;
    let away = away(fb, dir, Some(neg))?;
    // A normal half: the top 11 significand bits, one unit below them added first to round away.
    let r = match away {
        Away::Never => a,
        Away::Always | Away::Where(_) => {
            let unit = splat(fb, u, 0x1FFF)?;
            let up = bin(fb, BinOp::IAdd, a, unit)?;
            match away {
                Away::Where(c) => fb.select(c, up, a)?,
                _ => up,
            }
        }
    };
    let keep = splat(fb, u, 0xFFFF_E000)?;
    let r = bin(fb, BinOp::BitwiseAnd, r, keep)?;
    // Past the largest half: its infinity rounding away, 65504 otherwise.
    let largest = splat(fb, u, 0x477F_E000)?;
    let over = bin(fb, BinOp::UGreaterThan, r, largest)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let cap = match away {
        Away::Never => largest,
        Away::Always => inf,
        Away::Where(c) => fb.select(c, inf, largest)?,
    };
    let r = fb.select(over, cap, r)?;
    // Below the smallest normal half: the significand shifted to units of 2^-24.
    let k23 = splat(fb, u, 23)?;
    let e = bin(fb, BinOp::ShiftRightLogical, a, k23)?;
    let frac_mask = splat(fb, u, 0x007F_FFFF)?;
    let frac = bin(fb, BinOp::BitwiseAnd, a, frac_mask)?;
    let has_one = bin(fb, BinOp::INotEqual, e, z)?;
    let one_bit = splat(fb, u, 0x0080_0000)?;
    let lead = fb.select(has_one, one_bit, z)?;
    let sig = bin(fb, BinOp::BitwiseOr, frac, lead)?;
    let one = splat(fb, u, 1)?;
    let e1 = fb.glsl(GlslOp::UMax, &[e, one])?;
    let k126 = splat(fb, u, 126)?;
    let sh = bin(fb, BinOp::ISub, k126, e1)?;
    let k31 = splat(fb, u, 31)?;
    let sh = fb.glsl(GlslOp::UMin, &[sh, k31])?;
    let q = bin(fb, BinOp::ShiftRightLogical, sig, sh)?;
    let q = bump(fb, u, (sig, q, sh), away)?;
    let ft = ty(fb, f)?;
    let qf = fb.convert(ConvOp::ConvertUToF, q, ft)?;
    let unit = fsplat(fb, f, 2f64.powi(-24))?;
    let small_value = bin(fb, BinOp::FMul, qf, unit)?;
    let small_bits = fb.convert(ConvOp::Bitcast, small_value, ut)?;
    let min_normal = splat(fb, u, 0x3880_0000)?;
    let small = bin(fb, BinOp::ULessThan, a, min_normal)?;
    let r = fb.select(small, small_bits, r)?;
    // An infinity keeps its bits; a NaN is the canonical half NaN.
    let is_inf = bin(fb, BinOp::IEqual, a, inf)?;
    let r = fb.select(is_inf, a, r)?;
    let r = bin(fb, BinOp::BitwiseOr, r, sign)?;
    let is_nan = bin(fb, BinOp::UGreaterThan, a, inf)?;
    let nan = splat(fb, u, HALF_NAN)?;
    let r = fb.select(is_nan, nan, r)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// Float `x` of form `from` narrowed to float form `to`, rounded `dir`, exactly: `float` to `half`
/// ([`to_half`]), the one narrowing `air.convert` names a directed rounding for.
pub fn narrow_float(
    fb: &mut FunctionBuilder<'_>,
    from: Form,
    to: Form,
    x: Value,
    dir: Directed,
) -> BResult<Value> {
    if !narrows(from, to) {
        return Err(unconverted());
    }
    to_half(fb, from, x, dir)
}

/// Whether [`narrow_float`] narrows from `from` to `to`: `float` to `half`.
pub fn narrows(from: Form, to: Form) -> bool {
    (from.float_fmt(), to.float_fmt()) == (Some(FloatFmt::F32), Some(FloatFmt::Half))
}

/// The range of an integer of `bits` bits, signed or not.
fn range(bits: u8, signed: bool) -> (i128, i128) {
    let bits = u32::from(bits.clamp(1, 64));
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

/// The bounds a saturating conversion from an integer of `from` bits (signed when `src_signed`) to
/// one of `to` bits (signed when `dst_signed`) clamps its operand to, each `None` where the
/// operand's own range already lies within it: the larger of the two lower bounds and the smaller
/// of the two upper ones.
pub fn sat_bounds(
    (from, src_signed): (u8, bool),
    (to, dst_signed): (u8, bool),
) -> (Option<i128>, Option<i128>) {
    let (smin, smax) = range(from, src_signed);
    let (dmin, dmax) = range(to, dst_signed);
    let lo = smin.max(dmin);
    let hi = smax.min(dmax);
    ((lo > smin).then_some(lo), (hi < smax).then_some(hi))
}

/// Integer `x` of form `from` (read as signed when `src_signed`) clamped, in its own container,
/// to the range it shares with an integer of form `to` (signed when `dst_signed`) ([`sat_bounds`]):
/// `SClamp`, `SMax` or `SMin` of its sign extension, masked back to its width, or `UMin`. The
/// conversion to `to` that follows it is then exact.
pub fn saturate(
    fb: &mut FunctionBuilder<'_>,
    (src_signed, dst_signed): (bool, bool),
    from: Form,
    to: Form,
    x: Value,
) -> BResult<Value> {
    let (lo, hi) = sat_bounds((from.bits(), src_signed), (to.bits(), dst_signed));
    if lo.is_none() && hi.is_none() {
        return Ok(x);
    }
    // Two's complement in the container (`splat` masks to it).
    let k = |fb: &mut FunctionBuilder<'_>, v: i128| splat(fb, from, v as i64 as u64);
    if !src_signed {
        return match hi {
            Some(h) => {
                let h = k(fb, h)?;
                fb.glsl(GlslOp::UMin, &[x, h])
            }
            None => Ok(x),
        };
    }
    let s = sext(fb, from, x)?;
    let r = match (lo, hi) {
        (Some(l), Some(h)) => {
            let (l, h) = (k(fb, l)?, k(fb, h)?);
            fb.glsl(GlslOp::SClamp, &[s, l, h])?
        }
        (Some(l), None) => {
            let l = k(fb, l)?;
            fb.glsl(GlslOp::SMax, &[s, l])?
        }
        (None, Some(h)) => {
            let h = k(fb, h)?;
            fb.glsl(GlslOp::SMin, &[s, h])?
        }
        (None, None) => s,
    };
    mask(fb, from, r)
}

/// An upper estimate of the SPIR-V words of a directed conversion to `to` beyond the conversion it
/// replaces: [`to_half`]'s sixty instructions and twenty constants (a vector's constants carry
/// their lanes), or a magnitude's rounding.
pub fn directed_words(to: Form) -> u32 {
    let lanes = u32::from(to.lanes);
    match to.float_fmt() {
        Some(FloatFmt::Half) => 392 + 20 * lanes,
        _ => 200 + 10 * lanes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturating_bounds_clamp_only_where_the_ranges_differ() {
        // `int` to `uchar`: both bounds.
        assert_eq!(sat_bounds((32, true), (8, false)), (Some(0), Some(255)));
        // `uint` to `char`: the unsigned operand has no lower bound to clamp.
        assert_eq!(sat_bounds((32, false), (8, true)), (None, Some(127)));
        // `uchar` to `int`: nothing.
        assert_eq!(sat_bounds((8, false), (32, true)), (None, None));
        // `char` to `uint`: the negatives to zero.
        assert_eq!(sat_bounds((8, true), (32, false)), (Some(0), None));
        // `int` to `uint` and back.
        assert_eq!(sat_bounds((32, true), (32, false)), (Some(0), None));
        assert_eq!(
            sat_bounds((32, false), (32, true)),
            (None, Some(i128::from(i32::MAX)))
        );
        // `ulong` to `long`.
        assert_eq!(
            sat_bounds((64, false), (64, true)),
            (None, Some(i128::from(i64::MAX)))
        );
        // `long` to `int`.
        assert_eq!(
            sat_bounds((64, true), (32, true)),
            (Some(i128::from(i32::MIN)), Some(i128::from(i32::MAX)))
        );
    }

    #[test]
    fn exact_conversions_are_the_ones_whose_magnitude_fits_the_significand() {
        let int = |bits: u8| Form {
            held: Held::Int {
                bits,
                container: if bits > 32 { 64 } else { 32 },
            },
            lanes: 1,
        };
        assert!(converts_exactly(int(8), false, FloatFmt::Half));
        assert!(converts_exactly(int(8), true, FloatFmt::Half));
        assert!(!converts_exactly(int(16), true, FloatFmt::Half));
        assert!(converts_exactly(int(16), false, FloatFmt::F32));
        assert!(!converts_exactly(int(32), true, FloatFmt::F32));
        assert!(!converts_exactly(int(32), false, FloatFmt::F32));
        assert!(converts_exactly(int(32), false, FloatFmt::F64));
        assert!(!converts_exactly(int(64), true, FloatFmt::F64));
    }

    #[test]
    fn directions_follow_the_conversion_rounding() {
        assert_eq!(
            Directed::of(Some(ConvertRounding::Rtz)),
            Some(Directed::Zero)
        );
        assert_eq!(Directed::of(Some(ConvertRounding::Rtp)), Some(Directed::Up));
        assert_eq!(
            Directed::of(Some(ConvertRounding::Rtn)),
            Some(Directed::Down)
        );
        assert_eq!(Directed::of(Some(ConvertRounding::Rte)), None);
        assert_eq!(Directed::of(None), None);
        assert_eq!(Integral::of(None), Integral::Truncate);
        assert_eq!(Integral::of(Some(ConvertRounding::Rtz)), Integral::Truncate);
        assert_eq!(Integral::of(Some(ConvertRounding::Rte)), Integral::Even);
        assert_eq!(Integral::of(Some(ConvertRounding::Rtp)), Integral::Up);
        assert_eq!(Integral::of(Some(ConvertRounding::Rtn)), Integral::Down);
    }
}
