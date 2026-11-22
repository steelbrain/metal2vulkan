//! Interpreter differential tests of the exact texel conversions (`texel_round`).
//!
//! References compute on exact values in plain Rust: `f64` holds every `f32` exactly, and its
//! own correctly rounded division or rounding to `half` is followed by at most one more rounding
//! that `f64`'s 53 bits make innocuous (`2p + 2` bits for both `f32` and `half`).

use super::*;
use crate::lower::arith::texel_round::{self, HalfRounding};

/// The `f32` bits of the `half` nearest `x` by `mode` (exact `f64` arithmetic on the denormal
/// grid and the significand), the reference of [`texel_round::f32_to_half`].
pub(super) fn half_of(x: f32, mode: HalfRounding) -> u32 {
    if !x.is_finite() {
        return x.to_bits();
    }
    let v = f64::from(x);
    let a = v.abs();
    // The spacing of halves at |v|: 2^-24 below 2^-14, else 2^(e - 10).
    let ulp = if a < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(a.log2().floor() as i32 - 10)
    };
    let q = a / ulp;
    let r = match mode {
        HalfRounding::NearestEven => q.round_ties_even(),
        HalfRounding::TowardZeroSaturating => q.trunc(),
    } * ulp;
    let r = if r > 65504.0 {
        match mode {
            HalfRounding::TowardZeroSaturating => 65504.0,
            _ => f64::INFINITY,
        }
    } else {
        r
    };
    (r.copysign(v) as f32).to_bits()
}

/// The `f32` patterns the rounding is checked on, both signs: each half's neighbourhood (the half
/// itself, one `f32` ulp around it, its tie with the next half and one ulp around that) for every
/// half denormal, every 13th normal half, and the top binade; `f32` denormals, the tie below the
/// smallest half denormal, the overflow edge, and the specials.
pub(super) fn half_probe_bits() -> Vec<u64> {
    let mut out = Vec::new();
    let halves = (0u16..0x400)
        .chain((0x400..0x7B00).step_by(13))
        .chain(0x7B00..0x7C00);
    for h in halves {
        let v = f16_to_f64(h) as f32;
        let next = f16_to_f64(h + 1) as f32;
        let tie = ((f64::from(v) + f64::from(next)) / 2.0) as f32;
        for x in [v, tie] {
            let b = x.to_bits();
            out.extend([b, b.wrapping_sub(1), b + 1]);
        }
    }
    out.extend([
        0x0000_0001,
        0x0040_0000,
        0x007F_FFFF,
        0x0080_0000,
        0x3300_0000, // 2^-25: the tie between 0 and the smallest half denormal
        0x32FF_FFFF,
        0x3300_0001,
        0x3380_0000,
        0x477F_E000, // 65504
        0x477F_EFFF,
        0x477F_F000, // 65520: the tie past the largest half
        0x477F_F001,
        0x4780_0000,
        0x7F7F_FFFF,
        0x7F80_0000,
        0x7FC0_0000,
        0x7F80_0001,
    ]);
    let negated: Vec<u32> = out.iter().map(|b| b | 0x8000_0000).collect();
    out.extend(negated);
    out.into_iter().map(u64::from).collect()
}

/// `f32_to_half` rounds every probe like the exact reference in each mode, half denormals kept,
/// the sign of a zero kept, and `±65504` where saturation stops a toward-zero overflow.
#[test]
fn f32_to_half_rounds_exactly_in_every_mode() {
    let s = float(32);
    let cases: Vec<Case> = half_probe_bits()
        .into_iter()
        .map(|b| vec![vec![b]])
        .collect();
    for mode in [
        HalfRounding::NearestEven,
        HalfRounding::TowardZeroSaturating,
    ] {
        check(
            &format!("f32_to_half {mode:?}"),
            &[s],
            s,
            &|fb, x| texel_round::f32_to_half(fb, s, x[0], mode),
            &cases,
            &|c| {
                Some(vec![u64::from(half_of(
                    f32::from_bits(arg(c, 0) as u32),
                    mode,
                ))])
            },
            false,
        );
    }
}
