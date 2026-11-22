//! IEEE-754 floats of width 16, 32, and 64 as bits, with round-to-nearest-even per operation.
//!
//! Every operation widens its operands to `f64` exactly, computes there, and rounds once to the
//! operand width. For `+ - * /` and `sqrt` the double rounding through `f64` is innocuous
//! (53 ≥ 2p + 2 for p = 11 and 24), so each result is the correctly rounded one.

use super::{malformed, IResult};

/// `bits` of a `width`-bit float as `f64` (exact).
pub(super) fn to_f64(bits: u64, width: u8) -> IResult<f64> {
    match width {
        16 => Ok(f16_to_f64(bits as u16)),
        32 => Ok(f64::from(f32::from_bits(bits as u32))),
        64 => Ok(f64::from_bits(bits)),
        _ => Err(malformed(format!("float width {width}"))),
    }
}

/// `x` rounded to nearest-even in a `width`-bit float, as bits.
pub(super) fn from_f64(x: f64, width: u8) -> IResult<u64> {
    match width {
        16 => Ok(u64::from(f64_to_f16(x))),
        32 => Ok(u64::from((x as f32).to_bits())),
        64 => Ok(x.to_bits()),
        _ => Err(malformed(format!("float width {width}"))),
    }
}
/// Apply `f` to two operands and round.
pub(super) fn bin(width: u8, a: u64, b: u64, f: impl Fn(f64, f64) -> f64) -> IResult<u64> {
    from_f64(f(to_f64(a, width)?, to_f64(b, width)?), width)
}

pub(super) fn add(w: u8, a: u64, b: u64) -> IResult<u64> {
    bin(w, a, b, |x, y| x + y)
}

pub(super) fn sub(w: u8, a: u64, b: u64) -> IResult<u64> {
    bin(w, a, b, |x, y| x - y)
}

pub(super) fn mul(w: u8, a: u64, b: u64) -> IResult<u64> {
    bin(w, a, b, |x, y| x * y)
}

/// `a * b + c` with one rounding. On `f16` the product of two halves (22 significand bits) is
/// exact in `f64`, and the `f64` sum rounds only where the product is below a quarter of the
/// sum's `half` step and the addend a multiple of that step (a product any larger keeps the
/// sum's bits within 53), so the sum lies at least a quarter step from every `half` midpoint,
/// far more than its rounding moves it: the `f64` fused result rounded to `f16` is the
/// correctly rounded one.
pub(super) fn fma(w: u8, a: u64, b: u64, c: u64) -> IResult<u64> {
    match w {
        16 => {
            let [a, b, c] = [a, b, c].map(|x| f16_to_f64(x as u16));
            Ok(u64::from(f64_to_f16(a.mul_add(b, c))))
        }
        32 => {
            let r = f32::from_bits(a as u32)
                .mul_add(f32::from_bits(b as u32), f32::from_bits(c as u32));
            Ok(u64::from(r.to_bits()))
        }
        64 => Ok(f64::from_bits(a)
            .mul_add(f64::from_bits(b), f64::from_bits(c))
            .to_bits()),
        _ => Err(malformed(format!("float width {w}"))),
    }
}

/// The value of a binary16.
pub(super) fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f64::from(h & 0x3ff);
    match exp {
        0 => sign * man * 2f64.powi(-24),
        0x1f if man == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        e => sign * (1.0 + man / 1024.0) * 2f64.powi(e - 15),
    }
}

/// `x` rounded to nearest-even binary16.
pub(super) fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    if x.is_nan() {
        return 0x7e00;
    }
    let a = x.abs();
    // 65520 is halfway between the largest finite half (65504) and 2^16; ties go to even (inf).
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    let min_normal = 2f64.powi(-14);
    let (exp, quantum) = if a < min_normal {
        (0i32, 2f64.powi(-24))
    } else {
        let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        (e, 2f64.powi(e - 10))
    };
    // `a / quantum` is exact (a power-of-two scaling) and below 2^11.
    let m = a / quantum;
    let fl = m.floor();
    let d = m - fl;
    let r = if d > 0.5 || (d == 0.5 && fl % 2.0 != 0.0) {
        fl + 1.0
    } else {
        fl
    };
    let r = r as u32;
    if a < min_normal {
        // r == 1024 is the smallest normal, which the same bits encode.
        return sign | r as u16;
    }
    let (exp, r) = if r == 2048 { (exp + 1, 1024) } else { (exp, r) };
    if exp > 15 {
        return sign | 0x7c00;
    }
    sign | (((exp + 15) as u16) << 10) | ((r - 1024) as u16)
}
