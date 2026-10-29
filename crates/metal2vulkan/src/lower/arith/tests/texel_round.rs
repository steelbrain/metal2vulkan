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
        HalfRounding::NearestAway => q.round(),
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
        HalfRounding::NearestAway,
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

/// Every half magnitude's neighbourhood that decides a directed rounding, both signs: the half
/// itself, its tie with the next half, and one `f32` ulp on either side of the tie (one below a
/// half is one above the tie before it), for every finite half; with the `f32` denormals, the
/// overflow edge, and the specials of [`half_probe_bits`].
fn every_tie_bits() -> Vec<u64> {
    let mut out: Vec<u32> = Vec::new();
    for h in 0u16..0x7C00 {
        let v = f16_to_f64(h) as f32;
        let next = f16_to_f64(h + 1);
        let tie = ((f64::from(v) + next) / 2.0) as f32;
        let t = tie.to_bits();
        out.extend([v.to_bits(), t - 1, t, t + 1]);
    }
    let negated: Vec<u32> = out.iter().map(|b| b | 0x8000_0000).collect();
    out.extend(negated);
    let mut out: Vec<u64> = out.into_iter().map(u64::from).collect();
    out.extend(
        half_probe_bits()
            .into_iter()
            .filter(|b| *b as u32 & 0x7FFF_FFFF >= 0x477F_E000),
    );
    out.extend([0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF]);
    out
}

/// `f32_to_half_from_even` rounds every half's tie and its neighbours, and every half, toward
/// zero saturating and with ties away exactly, from the module's own rounding to nearest even
/// (`narrow::round_texel`) in every [`QuantizeForm`] (from the device's quantize, a half denormal
/// may be its signed zero): four lanes at a time, as the texel routine calls it.
#[test]
fn f32_to_half_from_even_is_exact_on_every_tie() {
    use crate::lower::arith::narrow;
    use crate::spirv::builder::QuantizeForm;
    let v = vec(float(32), 4);
    let h = vec(float(16), 4);
    let b = Form {
        held: Held::Bool,
        lanes: 1,
    };
    let quads: Vec<Vec<u64>> = every_tie_bits()
        .chunks(4)
        .map(|c| {
            let mut q = c.to_vec();
            q.resize(4, 0);
            q
        })
        .collect();
    for q in [
        QuantizeForm::Device,
        QuantizeForm::Converted,
        QuantizeForm::Controlled,
    ] {
        for (tz, mode) in [
            (true, HalfRounding::TowardZeroSaturating),
            (false, HalfRounding::NearestAway),
        ] {
            let cases: Vec<Case> = quads
                .iter()
                .map(|x| vec![x.clone(), vec![u64::from(tz)]])
                .collect();
            check_accept(
                &format!("f32_to_half_from_even {mode:?} {q:?}"),
                &[v, b],
                v,
                &|fb, x| {
                    fb.module().set_quantize_form(q);
                    let even = narrow::round_texel(fb, h, x[0])?;
                    texel_round::f32_to_half_from_even(fb, v, (x[0], even), x[1])
                },
                &cases,
                &|c| {
                    Some(
                        c[0].iter()
                            .map(|x| u64::from(half_of(f32::from_bits(*x as u32), mode)))
                            .collect(),
                    )
                },
                &|g, w| {
                    g == w
                        || (is_nan(v, g) && is_nan(v, w))
                        || (q == QuantizeForm::Device && quantize_flush_of(g, w))
                },
            );
        }
    }
}

/// The normalized grids the bicubic filter's exact sums land on: `16 · 255`, `16 · 127`, `65535`,
/// `32767`, and the sRGB step scale 4095. The grids: the bicubic sums', the sRGB table's, and the
/// normalized formats' fields' that a tile function's `half` member reads, down to a 2-bit alpha's
/// 3 and 1, where `3 · 2^24 / d` reaches `2^24`.
const GRIDS: [u32; 10] = [4080, 2032, 65535, 32767, 4095, 1023, 255, 127, 3, 1];

fn ratio_cases(d: u32) -> Vec<Case> {
    let step = if d > 4095 { 13 } else { 1 };
    (0..=u64::from(d))
        .step_by(step)
        .chain([1, 2, 3, 4, u64::from(d) - 1, u64::from(d)])
        // The quotients that round differently through `f32` (module docs).
        .chain([65455, 65519, 32759])
        .filter(|k| *k <= u64::from(d))
        .map(|k| vec![vec![k], vec![u64::from(d)]])
        .collect()
}

/// `ratio_to_f32` is correctly rounded with a run-time divisor on every grid.
#[test]
fn ratio_to_f32_is_correctly_rounded() {
    let u = int(32);
    for d in GRIDS {
        check(
            &format!("k / {d} to f32"),
            &[u, u],
            float(32),
            &|fb, a| texel_round::ratio_to_f32(fb, u, a[0], a[1]),
            &ratio_cases(d),
            &|c| {
                Some(vec![u64::from(
                    ((arg(c, 0) as f64 / d as f64) as f32).to_bits(),
                )])
            },
            false,
        );
    }
}

/// `ratio_to_half` rounds `k / d` to `half` once, half denormals included, where rounding the
/// `f32` quotient would round twice.
#[test]
fn ratio_to_half_rounds_once() {
    let u = int(32);
    for d in GRIDS {
        check(
            &format!("k / {d} to half"),
            &[u, u],
            float(32),
            &|fb, a| texel_round::ratio_to_half(fb, u, a[0], a[1]),
            &ratio_cases(d),
            &|c| {
                let h = f64_to_f16(arg(c, 0) as f64 / d as f64);
                Some(vec![u64::from((f16_to_f64(h) as f32).to_bits())])
            },
            false,
        );
    }
    // The quotients where the f32 detour differs exist: the test above covers them.
    let twice = |k: u32, d: u32| {
        let f = (f64::from(k) / f64::from(d)) as f32;
        f64_to_f16(f64::from(f)) != f64_to_f16(f64::from(k) / f64::from(d))
    };
    assert!(twice(65455, 65535) && twice(65519, 65535) && twice(32759, 32767));
}
