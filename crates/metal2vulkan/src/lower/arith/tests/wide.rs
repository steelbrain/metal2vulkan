//! Interpreter differential tests of wide integers held as two words.
//!
//! Operands and results cross the kernel boundary as their two words per lane (the parent's
//! `load`/`store` go through `wide::from_words`/`wide::to_words`), so the word composition is
//! checked by every test. Every kernel also passes L3 on `vk12-floor`, which has no
//! `shaderInt64` (a float64 kernel only on the parent's 64-bit profile), built in the profile's
//! `half` rounding form (it has no `shaderFloat16`): the sequences need no 64-bit integer.
//! References are the parent's LLVM and Metal definitions.

use super::super::wide::{self as w, WideOp};
use super::*;

fn wide(bits: u8) -> Form {
    Form {
        held: Held::Wide { bits },
        lanes: 1,
    }
}

/// The committed profile without `shaderInt64` the sequences are validated on.
fn floor() -> DeviceProfile {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(n, _)| n == "vk12-floor")
        .map(|(_, p)| p)
        .expect("vk12-floor")
}

/// [`check_accept`] after L3 on `vk12-floor` (unless a form needs 64-bit floats), the module there
/// built in the profile's `half` rounding form.
fn check_wide(
    what: &str,
    ins: &[Form],
    out: Form,
    body: Body<'_>,
    cases: &[Case],
    reference: &dyn Fn(&Case) -> Option<Vec<u64>>,
) {
    let f64_used = ins
        .iter()
        .chain(std::iter::once(&out))
        .any(|f| f.float_fmt() == Some(FloatFmt::F64));
    if !f64_used {
        let p = floor();
        let q = crate::lower::arith::narrow::quantize_form(&p);
        let m = kernel_defs(
            ins,
            out,
            &|b| {
                b.set_quantize_form(q);
                Ok(Vec::new())
            },
            &|fb, xs, _| body(fb, xs),
            false,
            false,
        )
        .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
        assert_valid(m, &p);
    }
    check(what, ins, out, body, cases, reference, false);
}

const WIDE_WIDTHS: [u8; 4] = [33, 40, 63, 64];

/// Wide binary operator `op` on form `f`: a division through its routine, defined in the
/// kernel's module as `arith::declare` defines it.
fn wide_bin(
    fb: &mut FunctionBuilder<'_>,
    op: AirBin,
    f: Form,
    a: Value,
    b: Value,
) -> BResult<Value> {
    match op {
        AirBin::UDiv | AirBin::URem | AirBin::SDiv | AirBin::SRem => {
            let r = w::define_divide(fb.module(), w::Divide::of(f))?;
            w::divide(fb, op, f, a, b, r)
        }
        _ => w::binary(fb, op, f, a, b),
    }
}

#[test]
fn wide_operators_match_llvm() {
    let mut rng = Rng(0xD95);
    for bits in WIDE_WIDTHS {
        let vals = int_values(bits, &mut rng, 5);
        let mut shifts = vals.clone();
        shifts.extend([0, 1, 5, 31, 32, 33, 47, u64::from(bits) - 1]);
        let f = wide(bits);
        for op in INT_OPS {
            let b: &[u64] = if matches!(op, AirBin::Shl | AirBin::LShr | AirBin::AShr) {
                &shifts
            } else {
                &vals
            };
            check_wide(
                &format!("{op:?} wide i{bits}"),
                &[f, f],
                f,
                &|fb, x| wide_bin(fb, op, f, x[0], x[1]),
                &pairs(&vals, b),
                &|c| Some(vec![llvm_int(op, bits, arg(c, 0), arg(c, 1))?]),
            );
        }
    }
}

#[test]
fn wide_operators_on_vectors_work_per_lane() {
    let f = vec(wide(64), 3);
    let big = 0x8000_0000_0000_0000u64;
    let cases: Vec<Case> = vec![
        vec![vec![u64::MAX, big, 7], vec![3, u64::MAX, 2]],
        vec![
            vec![0x1_0000_0000, 0xFFFF_FFFF, big + 5],
            vec![0x1_0000_0001, 33, 63],
        ],
        vec![vec![12345678901234, 1, 0], vec![1000000007, 64 - 1, 5]],
        vec![
            vec![big + 12345, u64::MAX - 3, 0x1234_5678_9ABC],
            vec![1, 32, 45],
        ],
    ];
    for op in [
        AirBin::Add,
        AirBin::Sub,
        AirBin::Mul,
        AirBin::UDiv,
        AirBin::SRem,
        AirBin::Shl,
        AirBin::AShr,
    ] {
        check_wide(
            &format!("{op:?} <3 x i64> wide"),
            &[f, f],
            f,
            &|fb, x| wide_bin(fb, op, f, x[0], x[1]),
            &cases,
            &|c| {
                (0..3)
                    .map(|l| llvm_int(op, 64, c[0][l], c[1][l]))
                    .collect::<Option<Vec<_>>>()
            },
        );
    }
}

/// The divisors of the division tests: 0, 1, the maximum, every power of two and its neighbours
/// (every normalization shift, with the divisor's high word zero or not), and words where Knuth's
/// digit estimate needs one or two corrections (a normalized divisor whose low digit is large, a
/// dividend whose top digits equal the divisor's).
fn division_edges(bits: u8) -> Vec<u64> {
    let m = mask(bits);
    let mut v = vec![0, 1, 2, 3, 5, 7, 10, m, m - 1, m >> 1];
    for k in 0..u32::from(bits) {
        let p = 1u64 << k;
        v.extend([p, p - 1]);
    }
    v.extend(
        division_hard_words()
            .into_iter()
            .flat_map(|w| [w, w << 16, w << 32]),
    );
    let mut v: Vec<u64> = v.into_iter().map(|x| x & m).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Words that exercise the digit corrections and the normalization of the compact division.
fn division_hard_words() -> Vec<u64> {
    [
        0x7FFFu64,
        0x8000,
        0xFFFF,
        0x8000_0001,
        0xFFFF_0001,
        0xFFFF_8000,
        0x7FFF_8000,
        0x8000_7FFF,
        0xFFFE_FFFF,
        0x1_FFFF_FFFF,
        0x7FFF_8000_0000_0000,
        0x8000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0001,
        0x0123_4567_89AB_CDEF,
    ]
    .into_iter()
    .collect()
}

/// The dividends of the division tests: fewer than the divisors (every case is an interpreter
/// invocation), still 0, 1, the maximum, the powers of two around each word and digit boundary,
/// and the hard words.
fn division_dividends(bits: u8) -> Vec<u64> {
    let m = mask(bits);
    let mut v = vec![0, 1, 2, 3, 10, m, m - 1, m >> 1];
    for k in [16u32, 32, 33, 48, u32::from(bits) - 1] {
        let p = 1u64 << k;
        v.extend([p, p - 1, p + 1]);
    }
    v.extend(division_hard_words().into_iter().flat_map(|w| [w, w << 32]));
    let mut v: Vec<u64> = v.into_iter().map(|x| x & m).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Dividend and divisor pairs whose quotient digit estimate is two too large, one for each
/// divisor class (below `2^32` or not, and the leading zeros of its top word, 0 to 31 in the
/// classes that have one): without the second correction of a quotient digit they divide
/// wrongly (found by a model of the sequence over random operands).
const SECOND_CORRECTION: [(u64, u64); 32] = [
    (0x44a_d6a2_6348_b40a, 0x83b2_fc96),
    (0x4_7092_9a68_38f9, 0x47b4_6deb),
    (0x1_16fc_e100_959f, 0x2f0e_3da4),
    (0x53a0_9b7a_d155, 0x151a_7ad8),
    (0x2855_6768_36dc_97af, 0xa2c_aeb2),
    (0x86a_7d74_5774, 0x483_afaf),
    (0x187_05fd_e4db, 0x22d_ffd2),
    (0x2087_20f0_6ab6_3bc7, 0x116_77fe),
    (0x3be2_4fb9_7c35_2142, 0x82_89ca),
    (0x14d_1d32_c8fa, 0x43_91fc),
    (0x38_753b_3b1a, 0x20_aa77),
    (0xaea_0c49_d010_1ff3, 0x10_74df),
    (0x878a_c9d3_586e, 0x8_0f6a),
    (0x47d_5b0a_a11b, 0x5_33e7),
    (0x75_e312_3009_0b83, 0x2_1f3f),
    (0xfaab_611a_5b45_4850, 0x1_3049_fd29_9750),
    (0xf6fa_1c66_ab7d_1bb7, 0xa33a_ee6b_aebb),
    (0xae6b_aa32_743c_1428, 0x4579_ffdf_dedb),
    (0xf4cc_857c_5c9d_a040, 0x218b_7b57_a4da),
    (0xf70a_a94d_ed97_d7ef, 0x131a_bbbf_b13b),
    (0xf002_67b0_851f_2f1e, 0x82d_da0e_2912),
    (0xcf4f_4587_30bd_f3e7, 0x403_e7fc_1967),
    (0xe975_f96d_cc4c_7c78, 0x233_5f8b_61c7),
    (0xe98f_203b_76a0_eddb, 0x103_4fd0_3252),
    (0xba8f_2665_5944_ae5c, 0x80_1afa_f4fa),
    (0xcfcb_e30c_130b_abd7, 0x41_28ef_d9e6),
    (0xe744_c3ca_9abb_defd, 0x28_d23d_6e77),
    (0xfae4_1bf8_0e0a_fb36, 0x10_6edc_2fc2),
    (0xcaf4_6482_656d_ef1e, 0x8_004a_a235),
    (0xf303_008c_5c66_c891, 0x4_8327_6e76),
    (0xd650_48cd_ba4b_8bd7, 0x2_41ef_84f7),
    (0xd3ce_0def_5105_de3c, 0x1_05bd_e8d7),
];

/// The compact division (Hacker's Delight `divdu` on 32-bit `OpUDiv`) is Rust's `u64` division (the
/// LLVM reference) on every pair of edge values, at 64 bits and at narrower wide widths, quotient
/// and remainder, signed and unsigned; and a zero divisor, undefined in LLVM, traps nowhere (the
/// interpreter traps a division by zero and an out-of-range shift) and divides by one.
#[test]
fn wide_division_matches_u64_division_on_edge_values() {
    for bits in [64u8, 40, 33] {
        let f = wide(bits);
        let dividends = division_dividends(bits);
        let (every, fewer) = (
            pairs(&dividends, &division_edges(bits)),
            pairs(&dividends, &dividends),
        );
        for op in [AirBin::UDiv, AirBin::URem, AirBin::SDiv, AirBin::SRem] {
            // Every divisor against the unsigned 64-bit routine; the sign handling and the
            // narrower widths (zero-extended into the same routine) on fewer.
            let cases = match (bits, op) {
                (64, AirBin::UDiv | AirBin::URem) => &every,
                (64, _) | (_, AirBin::UDiv | AirBin::SRem) => &fewer,
                _ => continue,
            };
            let reference = |c: &Case| {
                let (a, b) = (arg(c, 0), arg(c, 1));
                if b == 0 {
                    // A zero divisor divides by one.
                    let q = matches!(op, AirBin::UDiv | AirBin::SDiv);
                    return Some(vec![if q { a } else { 0 }]);
                }
                Some(vec![llvm_int(op, bits, a, b)?])
            };
            let cases: Vec<Case> = if bits == 64 {
                let corrections = SECOND_CORRECTION
                    .iter()
                    .map(|(a, b)| vec![vec![*a], vec![*b]]);
                cases.iter().cloned().chain(corrections).collect()
            } else {
                cases.clone()
            };
            // The interpreter bounds the steps of one run: at most a few thousand cases each.
            for (i, part) in cases.chunks(3000).enumerate() {
                let what = format!("{op:?} wide i{bits} on edge values, part {i}");
                let body: Body<'_> = &|fb, x| wide_bin(fb, op, f, x[0], x[1]);
                if i == 0 {
                    check_wide(&what, &[f, f], f, body, part, &reference);
                } else {
                    check(&what, &[f, f], f, body, part, &reference, false);
                }
            }
        }
    }
}

/// On vectors each lane divides on its own, whichever of the division's two cases (a divisor below
/// `2^32` or not) its neighbours take.
#[test]
fn wide_division_mixes_its_cases_across_lanes() {
    let edges = division_edges(64);
    for lanes in [2u8, 4] {
        let f = vec(wide(64), lanes);
        let n = usize::from(lanes);
        // Neighbouring lanes alternate between divisors below `2^32` (zero among them) and
        // wider ones, over every dividend.
        let (narrow, wider): (Vec<u64>, Vec<u64>) = edges.iter().partition(|d| *d >> 32 == 0);
        let mut cases: Vec<Case> = Vec::new();
        for i in 0..edges.len() {
            for flip in 0..2 {
                let dividend = |l: usize| edges[(i * n + l * 37 + flip) % edges.len()];
                let divisor = |l: usize| {
                    let from = if (l + flip).is_multiple_of(2) {
                        &narrow
                    } else {
                        &wider
                    };
                    from[(i * n + l) % from.len()]
                };
                cases.push(vec![
                    (0..n).map(dividend).collect(),
                    (0..n).map(divisor).collect(),
                ]);
            }
        }
        for op in [AirBin::UDiv, AirBin::URem] {
            check_wide(
                &format!("{op:?} <{lanes} x i64> wide on edge values"),
                &[f, f],
                f,
                &|fb, x| wide_bin(fb, op, f, x[0], x[1]),
                &cases,
                &|c| {
                    (0..n)
                        .map(|l| match c[1][l] {
                            0 if op == AirBin::UDiv => Some(c[0][l]),
                            0 => Some(0),
                            b => llvm_int(op, 64, c[0][l], b),
                        })
                        .collect::<Option<Vec<_>>>()
                },
            );
        }
    }
}

#[test]
fn wide_comparisons_match_llvm() {
    let mut rng = Rng(95);
    for bits in [40, 64] {
        let vals = int_values(bits, &mut rng, 4);
        let f = wide(bits);
        for pred in IntPredicate::ALL {
            check_wide(
                &format!("icmp {pred:?} wide i{bits}"),
                &[f, f],
                boolean(),
                &|fb, x| w::int_compare(fb, pred, f, x[0], x[1]),
                &pairs(&vals, &vals),
                &|c| Some(vec![u64::from(llvm_icmp(pred, bits, arg(c, 0), arg(c, 1)))]),
            );
        }
    }
    // Lane-wise on a vector.
    let f = vec(wide(64), 2);
    check_wide(
        "icmp slt <2 x i64> wide",
        &[f, f],
        vec(boolean(), 2),
        &|fb, x| w::int_compare(fb, IntPredicate::Slt, f, x[0], x[1]),
        &[vec![vec![u64::MAX, 5], vec![0, 0x1_0000_0000]]],
        &|_| Some(vec![1, 1]),
    );
}

#[test]
fn integer_casts_to_and_from_wide_match_llvm() {
    let mut rng = Rng(0xCA57);
    let tr = |x: u64, to: Form| {
        if to.is_bool() {
            x & 1
        } else {
            x & mask(to.bits())
        }
    };
    // Truncations.
    for (from, to) in [
        (wide(64), int(32)),
        (wide(64), int(8)),
        (wide(64), boolean()),
        (wide(64), wide(40)),
        (wide(40), int(16)),
        (wide(33), int(32)),
    ] {
        let vals = int_values(from.bits(), &mut rng, 6);
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        check_wide(
            &format!("trunc {from:?} -> {to:?}"),
            &[from],
            to,
            &|fb, x| w::cast(fb, CastKind::Trunc, from, to, x[0]),
            &cases,
            &|c| Some(vec![tr(arg(c, 0), to)]),
        );
    }
    // Extensions.
    for (from, to) in [
        (int(32), wide(64)),
        (int(8), wide(40)),
        (int(31), wide(33)),
        (boolean(), wide(64)),
        (wide(40), wide(64)),
        (wide(33), wide(63)),
    ] {
        let fb_bits = from.bits();
        let vals = if from.is_bool() {
            vec![0, 1]
        } else {
            int_values(fb_bits, &mut rng, 6)
        };
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        for kind in [CastKind::ZExt, CastKind::SExt] {
            check_wide(
                &format!("{kind:?} {from:?} -> {to:?}"),
                &[from],
                to,
                &|fb, x| w::cast(fb, kind, from, to, x[0]),
                &cases,
                &|c| {
                    let x = arg(c, 0);
                    Some(vec![if kind == CastKind::SExt {
                        (sext(x, fb_bits) as u64) & mask(to.bits())
                    } else {
                        x
                    }])
                },
            );
        }
    }
}

/// `x` (an exact integer magnitude below 2^64) rounded to nearest even at `keep` significant
/// bits, as an `f64` (exact: at most 53 bits are kept).
fn round_bits(x: u64, keep: u32) -> f64 {
    if x == 0 {
        return 0.0;
    }
    let msb = 63 - x.leading_zeros();
    if msb < keep {
        return x as f64;
    }
    let shift = msb + 1 - keep;
    let q = x >> shift;
    let rem = x & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let q = if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    };
    (q as f64) * 2f64.powi(shift as i32)
}

#[test]
fn wide_to_float_rounds_once() {
    let mut rng = Rng(0xF10A7);
    for bits in [64, 40] {
        let mut vals = int_values(bits, &mut rng, 12);
        // Ties and near-ties of f32 and bfloat at several magnitudes.
        for base in [1u64 << 24, 1 << 40, 1 << 62] {
            for d in [0u64, 1, 2, 3, 0x7F, 0x80, 0x81, 0x8000, 0x8001] {
                vals.push((base | d) & mask(bits));
                vals.push((base + (base >> 24) + d) & mask(bits));
            }
        }
        vals.sort_unstable();
        vals.dedup();
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        let f = wide(bits);
        for signed in [false, true] {
            for to in [float(32), float(16), bfloat(), float(64)] {
                let kind = if signed {
                    CastKind::SiToFp
                } else {
                    CastKind::UiToFp
                };
                check_wide(
                    &format!("{kind:?} wide i{bits} -> {to:?}"),
                    &[f],
                    to,
                    &|fb, x| w::cast(fb, kind, f, to, x[0]),
                    &cases,
                    &|c| {
                        let x = arg(c, 0);
                        let (neg, m) = if signed && sext(x, bits) < 0 {
                            (true, sext(x, bits).unsigned_abs())
                        } else {
                            (false, x)
                        };
                        let keep = match to.float_fmt() {
                            Some(FloatFmt::F64) => 53,
                            Some(FloatFmt::F32) => 24,
                            Some(FloatFmt::BFloat) => 8,
                            _ => 11,
                        };
                        // A half keeps 11 bits only below 2^16; above, it is infinite.
                        let v = round_bits(m, keep);
                        let v = if neg { -v } else { v };
                        // A narrow result crosses the kernel as its format bits.
                        let c = to.const_bits(v);
                        Some(vec![to.float_fmt().map_or(c, |f| f.narrow_bits(c))])
                    },
                );
            }
        }
    }
}

/// A wide integer converts to `half` by integer arithmetic on its `f32`, with no `OpQuantizeToF16`
/// even where the module rounds by the device's quantize ([`QuantizeForm::Device`]), on the values
/// `integers_for_half` gives a 64-bit integer, plus `2^32` and `2^39` and their neighbours.
#[test]
fn wide_integers_round_to_half_without_quantize() {
    for bits in [64u8, 40] {
        let f = wide(bits);
        let m = mask(bits);
        let mut vals: Vec<u64> = super::narrow::integers_for_half(64)
            .into_iter()
            .map(|v| v & m)
            .collect();
        for base in [1i64 << 32, 1 << 39] {
            for d in -3i64..=3 {
                vals.push((base + d) as u64 & m);
                vals.push((-(base + d)) as u64 & m);
            }
        }
        vals.sort_unstable();
        vals.dedup();
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        for (kind, signed) in [(CastKind::UiToFp, false), (CastKind::SiToFp, true)] {
            let what = format!("{kind:?} wide i{bits} -> half");
            let body =
                |fb: &mut FunctionBuilder<'_>, x: &[Value]| w::cast(fb, kind, f, float(16), x[0]);
            let module = kernel_defs(
                &[f],
                float(16),
                &|b| {
                    b.set_quantize_form(QuantizeForm::Device);
                    Ok(Vec::new())
                },
                &|fb, xs, _| body(fb, xs),
                false,
                false,
            )
            .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            assert!(
                !super::super::narrow::quantizes(&module),
                "{what}: the module holds OpQuantizeToF16"
            );
            check_wide(&what, &[f], float(16), &body, &cases, &|c| {
                let x = arg(c, 0);
                let v = if signed {
                    i128::from(sext(x, bits))
                } else {
                    i128::from(x)
                };
                Some(vec![super::narrow::half_of_integer(v)])
            });
        }
    }
}

#[test]
fn float_to_wide_truncates_and_saturates() {
    let vals: Vec<u64> = f32_values();
    let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
    for bits in [64, 40] {
        let to = wide(bits);
        for signed in [false, true] {
            // LLVM `fptoui`/`fptosi`: out of range (and NaN) is poison.
            let kind = if signed {
                CastKind::FpToSi
            } else {
                CastKind::FpToUi
            };
            check_wide(
                &format!("{kind:?} f32 -> wide i{bits}"),
                &[float(32)],
                to,
                &|fb, x| w::cast(fb, kind, float(32), to, x[0]),
                &cases,
                &|c| {
                    let x = f64::from(f32v(arg(c, 0))).trunc();
                    let (lo, hi) = if signed {
                        (
                            -(2f64.powi(i32::from(bits) - 1)),
                            2f64.powi(i32::from(bits) - 1),
                        )
                    } else {
                        (0.0, 2f64.powi(i32::from(bits)))
                    };
                    if !(x >= lo && x < hi) {
                        return None;
                    }
                    Some(vec![(x as i128 as u64) & mask(bits)])
                },
            );
            // `air.convert` saturates: NaN to 0, out of range to the bound.
            check_wide(
                &format!("convert saturating f32 -> wide i{bits} signed={signed}"),
                &[float(32)],
                to,
                &|fb, x| {
                    w::convert(
                        fb,
                        ConvertKind::ToInt {
                            signed,
                            round: None,
                        },
                        float(32),
                        to,
                        x[0],
                    )
                },
                &cases,
                &|c| {
                    let x = f64::from(f32v(arg(c, 0)));
                    if x.is_nan() {
                        return Some(vec![0]);
                    }
                    let (lo, hi) = if signed {
                        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
                    } else {
                        (0, (1i128 << bits) - 1)
                    };
                    let t = x.trunc();
                    let v = if t <= lo as f64 {
                        lo
                    } else if t >= hi as f64 {
                        hi
                    } else {
                        t as i128
                    };
                    Some(vec![(v as u64) & mask(bits)])
                },
            );
        }
    }
    // A half and a double source (the double on the 64-bit profile only).
    let halves: Vec<Case> = [0x3C00u64, 0xBC00, 0x7BFF, 0xFBFF, 0x3800, 0x0001]
        .iter()
        .map(|h| vec![vec![*h]])
        .collect();
    check_wide(
        "fptosi half -> wide i64",
        &[float(16)],
        wide(64),
        &|fb, x| w::cast(fb, CastKind::FpToSi, float(16), wide(64), x[0]),
        &halves,
        &|c| {
            let x = f16_to_f64(arg(c, 0) as u16).trunc();
            Some(vec![x as i64 as u64])
        },
    );
    let doubles: Vec<Case> = [1e18f64, -1e18, 12345.75, -0.5, 9.2e18, 1.8e19]
        .iter()
        .map(|d| vec![vec![d.to_bits()]])
        .collect();
    check_wide(
        "fptoui f64 -> wide i64",
        &[float(64)],
        wide(64),
        &|fb, x| w::cast(fb, CastKind::FpToUi, float(64), wide(64), x[0]),
        &doubles,
        &|c| {
            let x = f64v(arg(c, 0)).trunc();
            (x >= 0.0 && x < 2f64.powi(64)).then(|| vec![x as u64])
        },
    );
}

/// Integer `median3` on wide values (two 32-bit words) answers the middle one of its operands,
/// signed and unsigned, on scalars and 4-lane vectors of 64 and 40 bits: every ordered triple of
/// the width's edges, ties and the extremes among them.
#[test]
fn wide_integer_median3_answers_the_middle_one_at_ties_and_extremes() {
    for bits in [64u8, 40] {
        let edges = width_edges(bits);
        for s in [Sign::Signed, Sign::Unsigned] {
            let op = IntFn::Median3(s);
            for lanes in [1u8, 4] {
                let f = vec(wide(bits), lanes);
                check_wide(
                    &format!("{op:?} wide i{bits} x{lanes}"),
                    &[f; 3],
                    f,
                    &|fb, x| w::int_fn(fb, op, f, x),
                    &triples(&edges, usize::from(lanes)),
                    &|c| {
                        (0..usize::from(lanes))
                            .map(|l| int_ref(op, bits, &[c[0][l], c[1][l], c[2][l]]))
                            .collect::<Option<Vec<u64>>>()
                    },
                );
            }
        }
    }
}

#[test]
fn wide_integer_functions_match_metal() {
    let mut rng = Rng(0x1F);
    for bits in [64, 33, 40, 48, 63] {
        let f = wide(bits);
        let vals = int_values(bits, &mut rng, 3);
        let one: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        let two = pairs(&vals, &vals);
        // Every pair of values with a third drawn across the range (saturating results, large
        // funnel amounts), plus the small cube.
        let mut three: Vec<Case> = Vec::new();
        for (i, a) in vals.iter().enumerate() {
            for (j, b) in vals.iter().enumerate() {
                let c = vals[(i * 3 + j * 5) % vals.len()];
                three.push(vec![vec![*a], vec![*b], vec![c]]);
            }
        }
        for a in vals.iter().take(4) {
            for b in vals.iter().take(4) {
                for c in vals.iter().take(4) {
                    three.push(vec![vec![*a], vec![*b], vec![*c]]);
                }
            }
        }
        let mut ops: Vec<IntFn> = vec![
            IntFn::Abs,
            IntFn::Popcount,
            IntFn::ReverseBits,
            IntFn::Clz,
            IntFn::Ctz,
        ];
        if bits % 16 == 0 {
            ops.push(IntFn::Bswap);
        }
        for s in [Sign::Signed, Sign::Unsigned] {
            ops.extend([
                IntFn::AbsDiff(s),
                IntFn::Min(s),
                IntFn::Max(s),
                IntFn::Min3(s),
                IntFn::Max3(s),
                IntFn::Median3(s),
                IntFn::Clamp(s),
                IntFn::AddSat(s),
                IntFn::SubSat(s),
                IntFn::HAdd(s),
                IntFn::RHAdd(s),
                IntFn::MulHi(s),
                IntFn::MadHi(s),
                IntFn::MadSat(s),
            ]);
        }
        ops.extend([IntFn::Rotate, IntFn::Fshl, IntFn::Fshr]);
        for op in ops {
            assert!(w::int_fn_lowered(op, f), "{op:?} i{bits}");
            let cases = match op.arity() {
                1 => &one,
                2 => &two,
                _ => &three,
            };
            let ins = vec![f; op.arity()];
            check_wide(
                &format!("{op:?} wide i{bits}"),
                &ins,
                f,
                &|fb, x| w::int_fn(fb, op, f, x),
                cases,
                &|c| {
                    let a: Vec<u64> = c.iter().map(|l| l[0]).collect();
                    // Clamp with lo > hi is undefined in Metal.
                    if let IntFn::Clamp(sg) = op {
                        let v = |x: u64| {
                            if sg == Sign::Signed {
                                i128::from(sext(x, bits))
                            } else {
                                i128::from(x)
                            }
                        };
                        if v(a[1]) > v(a[2]) {
                            return None;
                        }
                    }
                    Some(vec![int_ref(op, bits, &a)?])
                },
            );
        }
        // Bit fields with 32-bit offset and count.
        let (x, y) = (
            0xF0E1_D2C3_B4A5_9687u64 & mask(bits),
            0x0123_4567_89AB_CDEFu64 & mask(bits),
        );
        let mut ext = Vec::new();
        let mut ins = Vec::new();
        for (off, cnt) in [
            (0u64, u64::from(bits)),
            (bits as u64 - 4, 4),
            (4, 30),
            (31, 2),
            (17, 0),
            (33, 5),
        ] {
            ext.push(vec![vec![x], vec![off], vec![cnt]]);
            ins.push(vec![vec![x], vec![y], vec![off], vec![cnt]]);
        }
        for s in [Sign::Signed, Sign::Unsigned] {
            let op = IntFn::ExtractBits(s);
            check_wide(
                &format!("{op:?} wide i{bits}"),
                &[f, int(32), int(32)],
                f,
                &|fb, v| w::int_fn(fb, op, f, v),
                &ext,
                &|c| Some(vec![int_ref(op, bits, &[c[0][0], c[1][0], c[2][0]])?]),
            );
        }
        check_wide(
            &format!("InsertBits wide i{bits}"),
            &[f, f, int(32), int(32)],
            f,
            &|fb, v| w::int_fn(fb, IntFn::InsertBits, f, v),
            &ins,
            &|c| {
                Some(vec![int_ref(
                    IntFn::InsertBits,
                    bits,
                    &[c[0][0], c[1][0], c[2][0], c[3][0]],
                )?])
            },
        );
    }
    // Lane-wise on a vector.
    let f = vec(wide(64), 2);
    check_wide(
        "Popcount <2 x i64> wide",
        &[f],
        f,
        &|fb, v| w::int_fn(fb, IntFn::Popcount, f, v),
        &[vec![vec![u64::MAX, 0x8000_0000_0000_0001]]],
        &|_| Some(vec![64, 2]),
    );
}

#[test]
fn wide_bitcasts_and_integer_converts_move_bits() {
    let x = 0xC3B4_A596_8778_6958u64;
    let y = 0x0123_4567_89AB_CDEFu64;
    let lanes = |v: u64, bits: u8, n: u8| -> Vec<u64> {
        (0..n)
            .map(|i| (v >> (u32::from(i) * u32::from(bits))) & mask(bits))
            .collect()
    };
    for other in [
        vec(int(32), 2),
        vec(float(32), 2),
        vec(int(16), 4),
        vec(int(8), 4),
    ] {
        let n = other.lanes;
        let b = other.bits();
        if u32::from(b) * u32::from(n) != 64 {
            continue;
        }
        check_wide(
            &format!("bitcast wide i64 -> {other:?}"),
            &[wide(64)],
            other,
            &|fb, v| w::bitcast(fb, wide(64), other, v[0]),
            &[vec![vec![x]], vec![vec![y]]],
            &|c| Some(lanes(arg(c, 0), b, n)),
        );
        check_wide(
            &format!("bitcast {other:?} -> wide i64"),
            &[other],
            wide(64),
            &|fb, v| w::bitcast(fb, other, wide(64), v[0]),
            &[vec![lanes(x, b, n)], vec![lanes(y, b, n)]],
            &|c| {
                Some(vec![c[0].iter().enumerate().fold(0u64, |acc, (i, l)| {
                    acc | (l << (i as u32 * u32::from(b)))
                })])
            },
        );
    }
    // Two wide lanes to four words, and a double.
    check_wide(
        "bitcast <2 x i64> wide -> <4 x i32>",
        &[vec(wide(64), 2)],
        vec(int(32), 4),
        &|fb, v| w::bitcast(fb, vec(wide(64), 2), vec(int(32), 4), v[0]),
        &[vec![vec![x, y]]],
        &|_| Some(vec![x & 0xFFFF_FFFF, x >> 32, y & 0xFFFF_FFFF, y >> 32]),
    );
    check_wide(
        "bitcast f64 -> wide i64",
        &[float(64)],
        wide(64),
        &|fb, v| w::bitcast(fb, float(64), wide(64), v[0]),
        &[vec![vec![x]]],
        &|_| Some(vec![x]),
    );
    // `air.convert` between integers: truncating or extending by the source's sign.
    for (from, to, signed) in [
        (wide(64), int(32), true),
        (int(32), wide(64), true),
        (int(32), wide(64), false),
        (int(16), wide(64), true),
        (wide(40), wide(64), true),
    ] {
        let vals = int_values(from.bits(), &mut Rng(3), 4);
        let cases: Vec<Case> = vals.iter().map(|v| vec![vec![*v]]).collect();
        check_wide(
            &format!("convert {from:?} -> {to:?} signed={signed}"),
            &[from],
            to,
            &|fb, v| w::convert(fb, ConvertKind::Int { signed }, from, to, v[0]),
            &cases,
            &|c| {
                let x = arg(c, 0);
                Some(vec![if to.bits() < from.bits() {
                    x & mask(to.bits())
                } else if signed {
                    (sext(x, from.bits()) as u64) & mask(to.bits())
                } else {
                    x
                }])
            },
        );
    }
}

#[test]
fn a_wide_integer_to_bool_convert_is_nonzero() {
    // `x != 0`, so a value with only high-word bits is true.
    for from in [wide(64), wide(40), vec(wide(64), 2)] {
        let top = 1u64 << (from.bits() - 1);
        let vals = [
            0,
            1,
            2,
            1 << 32,
            top,
            mask(from.bits()),
            0x1_0000_0000 | 2,
            0,
        ];
        let cases: Vec<Case> = vals
            .chunks(usize::from(from.lanes))
            .map(|c| vec![c.to_vec()])
            .collect();
        let to = vec(boolean(), from.lanes);
        check_wide(
            &format!("convert {from:?} -> bool"),
            &[from],
            to,
            &|fb, v| w::convert(fb, ConvertKind::IntToBool, from, to, v[0]),
            &cases,
            &|c| Some(c[0].iter().map(|v| u64::from(*v != 0)).collect()),
        );
    }
}

#[test]
fn wide_element_moves_and_selects_work_per_half() {
    let f = vec(wide(64), 3);
    let v = vec![u64::MAX, 0x1_0000_0002, 0x8000_0000_0000_0003];
    for i in 0..3u32 {
        check_wide(
            &format!("extract {i}"),
            &[f],
            wide(64),
            &|fb, x| {
                w::emit(
                    fb,
                    &WideOp::Extract {
                        vec: f,
                        at: Some(i),
                        wide_index: false,
                    },
                    x,
                    None,
                )
            },
            &[vec![v.clone()]],
            &|_| Some(vec![v[i as usize]]),
        );
    }
    // A dynamic index, narrow and wide.
    for (index, wide_index) in [(int(32), false), (wide(64), true)] {
        check_wide(
            &format!("extract dynamic wide_index={wide_index}"),
            &[f, index],
            wide(64),
            &|fb, x| {
                w::emit(
                    fb,
                    &WideOp::Extract {
                        vec: f,
                        at: None,
                        wide_index,
                    },
                    x,
                    None,
                )
            },
            &[vec![v.clone(), vec![2]], vec![v.clone(), vec![0]]],
            &|c| Some(vec![v[c[1][0] as usize]]),
        );
        check_wide(
            &format!("insert dynamic wide_index={wide_index}"),
            &[f, wide(64), index],
            f,
            &|fb, x| {
                w::emit(
                    fb,
                    &WideOp::Insert {
                        vec: f,
                        at: None,
                        wide_index,
                    },
                    x,
                    None,
                )
            },
            &[vec![v.clone(), vec![42], vec![1]]],
            &|_| Some(vec![v[0], 42, v[2]]),
        );
    }
    check_wide(
        "shuffle",
        &[f, f],
        vec(wide(64), 2),
        &|fb, x| {
            w::emit(
                fb,
                &WideOp::Shuffle {
                    mask: vec![5, 0].into(),
                    out: vec(wide(64), 2),
                },
                x,
                None,
            )
        },
        &[vec![v.clone(), vec![7, 8, 9]]],
        &|_| Some(vec![9, v[0]]),
    );
    // Narrower wide lanes keep their width.
    let g = vec(wide(40), 2);
    let u = vec![0xFF_FFFF_FFFF, 0x80_0000_0001];
    check_wide(
        "shuffle i40",
        &[g, g],
        vec(wide(40), 4),
        &|fb, x| {
            w::emit(
                fb,
                &WideOp::Shuffle {
                    mask: vec![3, 0, 1, 2].into(),
                    out: vec(wide(40), 4),
                },
                x,
                None,
            )
        },
        &[vec![u.clone(), vec![7, 0x12_3456_789A]]],
        &|_| Some(vec![0x12_3456_789A, u[0], u[1], 7]),
    );
    let c2 = vec(boolean(), 3);
    check_wide(
        "select per lane",
        &[c2, f, f],
        f,
        &|fb, x| w::emit(fb, &WideOp::Select { form: f }, x, None),
        &[vec![vec![1, 0, 1], v.clone(), vec![7, 8, 9]]],
        &|_| Some(vec![v[0], 8, v[2]]),
    );
}

/// The per-use word bound of a wide integer function covers what it emits (the kernel around it,
/// which also loads every operand, subtracted), at every wide width and on vectors.
#[test]
fn wide_integer_function_words_bound_their_emission() {
    use crate::spirv::serialize::serialize;
    let size = |ins: &[Form], out: Form, body: Body<'_>| -> usize {
        let m = kernel(ins, out, body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let mut ops: Vec<IntFn> = vec![
        IntFn::Abs,
        IntFn::Popcount,
        IntFn::ReverseBits,
        IntFn::Clz,
        IntFn::Ctz,
        IntFn::Bswap,
        IntFn::Rotate,
        IntFn::Fshl,
        IntFn::Fshr,
        IntFn::InsertBits,
    ];
    for s in [Sign::Signed, Sign::Unsigned] {
        ops.extend([
            IntFn::AbsDiff(s),
            IntFn::Min3(s),
            IntFn::Median3(s),
            IntFn::Clamp(s),
            IntFn::AddSat(s),
            IntFn::SubSat(s),
            IntFn::RHAdd(s),
            IntFn::MulHi(s),
            IntFn::MadHi(s),
            IntFn::MadSat(s),
            IntFn::ExtractBits(s),
        ]);
    }
    for (bits, lanes) in [(64u8, 1u8), (40, 1), (63, 3), (48, 4)] {
        let f = if lanes == 1 {
            wide(bits)
        } else {
            vec(wide(bits), lanes)
        };
        for op in &ops {
            let op = *op;
            if !w::int_fn_lowered(op, f) {
                continue;
            }
            let ins: Vec<Form> = match op {
                IntFn::ExtractBits(_) => vec![f, int(32), int(32)],
                IntFn::InsertBits => vec![f, f, int(32), int(32)],
                _ => vec![f; op.arity()],
            };
            let with = size(&ins, f, &|fb, x| w::int_fn(fb, op, f, x));
            let without = size(&ins, f, &|_, x| Ok(x[0]));
            let bound = WideOp::IntFn { f: op, form: f }.words() as usize;
            assert!(
                with - without <= bound,
                "{op:?} {f:?}: {} words above the kernel, bound {bound}",
                with - without
            );
        }
    }
}

/// A wide division's per-use bound covers its call, and its routine's bound (once per module)
/// covers the routine, at every wide width and on vectors: the second division of a form adds only
/// its call.
#[test]
fn wide_division_words_bound_their_emission() {
    use crate::spirv::serialize::serialize;
    let size = |ins: &[Form], out: Form, body: Body<'_>| -> usize {
        let m = kernel(ins, out, body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let (mut most_call, mut most_routine) = (0, 0);
    for (bits, lanes) in [(64u8, 1u8), (40, 1), (33, 1), (63, 3), (64, 4)] {
        let f = if lanes == 1 {
            wide(bits)
        } else {
            vec(wide(bits), lanes)
        };
        let d = w::Divide::of(f);
        for op in [AirBin::UDiv, AirBin::URem, AirBin::SDiv, AirBin::SRem] {
            let without = size(&[f, f], f, &|_, x| Ok(x[0]));
            let one = size(&[f, f], f, &|fb, x| {
                let r = w::define_divide(fb.module(), d)?;
                w::divide(fb, op, f, x[0], x[1], r)
            });
            let two = size(&[f, f], f, &|fb, x| {
                let r = w::define_divide(fb.module(), d)?;
                let t = w::divide(fb, op, f, x[0], x[1], r)?;
                w::divide(fb, op, f, t, x[1], r)
            });
            let call = two - one;
            let routine = one - without - call;
            let (call_bound, routine_bound) = (
                WideOp::Bin { op, form: f }.words() as usize,
                d.words() as usize,
            );
            assert!(
                call <= call_bound,
                "{op:?} {f:?}: a call of {call} words, bound {call_bound}"
            );
            assert!(
                routine <= routine_bound,
                "{op:?} {f:?}: a routine of {routine} words, bound {routine_bound}"
            );
            most_call = most_call.max(call);
            most_routine = most_routine.max(routine);
        }
    }
    eprintln!("wide division: at most {most_call} words per call, {most_routine} per routine");
}

/// The per-use word bound of each wide element move and select covers what it emits: a select of a
/// scalar (one select of the pair) and of a vector (word by word, on a vector or a scalar
/// condition), shuffles, and constant and dynamic (32-bit and two-word index) extracts and inserts.
#[test]
fn wide_element_moves_and_selects_bound_their_emission() {
    use crate::spirv::serialize::serialize;
    let size = |ins: &[Form], out: Form, body: Body<'_>| -> usize {
        let m = kernel(ins, out, body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let check = |what: &str, op: WideOp, ins: &[Form], out: Form, keep: usize| {
        let with = size(ins, out, &|fb, x| w::emit(fb, &op, x, None));
        let without = size(ins, out, &|_, x| Ok(x[keep]));
        let bound = op.words() as usize;
        assert!(
            with - without <= bound,
            "{what}: {} words above the kernel, bound {bound}",
            with - without
        );
    };
    for bits in [64u8, 40] {
        let s = wide(bits);
        check(
            "select",
            WideOp::Select { form: s },
            &[boolean(), s, s],
            s,
            1,
        );
        for lanes in [2u8, 3, 4] {
            let v = vec(s, lanes);
            let c = vec(boolean(), lanes);
            check(
                "select per lane",
                WideOp::Select { form: v },
                &[c, v, v],
                v,
                1,
            );
            check(
                "select on a scalar",
                WideOp::Select { form: v },
                &[boolean(), v, v],
                v,
                1,
            );
            for out in 2..=4u8 {
                let mask: Box<[u32]> = (0..u32::from(out)).map(|i| (i * 3) % 8).collect();
                let mask: Box<[u32]> = mask.iter().map(|m| m % (2 * u32::from(lanes))).collect();
                let o = vec(s, out);
                // The kernel's output is `o`: return the result, or a third operand of `o`.
                let op = WideOp::Shuffle { mask, out: o };
                let with = size(&[v, v, o], o, &|fb, x| w::emit(fb, &op, &x[..2], None));
                let without = size(&[v, v, o], o, &|_, x| Ok(x[2]));
                assert!(
                    with - without <= op.words() as usize,
                    "shuffle {lanes} -> {out}: {} words, bound {}",
                    with - without,
                    op.words()
                );
            }
            let at = Some(u32::from(lanes) - 1);
            check(
                "extract",
                WideOp::Extract {
                    vec: v,
                    at,
                    wide_index: false,
                },
                &[v, s],
                s,
                1,
            );
            check(
                "insert",
                WideOp::Insert {
                    vec: v,
                    at,
                    wide_index: false,
                },
                &[v, s],
                v,
                0,
            );
            for (index, wide_index) in [(int(32), false), (wide(64), true)] {
                check(
                    "dynamic extract",
                    WideOp::Extract {
                        vec: v,
                        at: None,
                        wide_index,
                    },
                    &[v, index, s],
                    s,
                    2,
                );
                check(
                    "dynamic insert",
                    WideOp::Insert {
                        vec: v,
                        at: None,
                        wide_index,
                    },
                    &[v, s, index],
                    v,
                    0,
                );
            }
        }
    }
}

/// The word bound of each lane move on a chunked vector covers what it emits, for two-word lanes
/// (64 and 40 bits) and, where the flat bounds stay, 32-bit lanes: constant and dynamic (32-bit and
/// two-word index) extracts and inserts, shuffles, and word bitcasts to 32-bit lanes and back, at
/// 5, 6, and 8 lanes (chunks of 4 and 1, 4 and 2, and two of 4).
#[test]
fn chunked_lane_moves_bound_their_emission() {
    use super::super::chunk::{self, Shape};
    use super::super::ArithStrategy as S;
    use crate::lower::strategy::FamilyStrategy;
    use crate::spirv::serialize::serialize;
    let size = |ins: &[Form], out: Form, body: Body<'_>| -> usize {
        let m = kernel(ins, out, body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let check = |what: &str, s: &S, with: usize, without: usize| {
        let bound = s.words() as usize;
        assert!(
            with - without <= bound,
            "{what}: {} words above the kernel, bound {bound}",
            with - without
        );
    };
    for lane in [wide(64), wide(40), int(32)] {
        let four = Shape {
            held: lane.held,
            lanes: 4,
            chunked: false,
        };
        let v4 = vec(lane, 4);
        for n in [5u16, 6, 8] {
            let s = Shape {
                held: lane.held,
                lanes: n,
                chunked: true,
            };
            let what = |m: &str| format!("{m} {lane:?} x {n}");
            let all: Vec<u32> = (0..u32::from(n)).collect();
            // The chunked operand, from the first two (4-lane) operands.
            let make = |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
                chunk::shuffle(fb, four, four, s, &all, &x[..2])
            };
            // Lane 0 of a chunked value, as the kernel's output.
            let first =
                |fb: &mut FunctionBuilder<'_>, v: Value| chunk::extract(fb, s, Some(0), None, &[v]);
            let unmoved = |ins: &[Form]| {
                size(ins, lane, &|fb, x| {
                    let v = make(fb, x)?;
                    first(fb, v)
                })
            };
            // Constant moves of lane 1, which lives in a 4-lane chunk.
            let ins = [v4, v4, lane];
            let kept = size(&ins, lane, &|fb, x| {
                make(fb, x)?;
                Ok(x[2])
            });
            let with = size(&ins, lane, &|fb, x| {
                let v = make(fb, x)?;
                chunk::extract(fb, s, Some(1), None, &[v])
            });
            let extract = S::LaneExtract {
                src: s,
                at: Some(1),
                index: None,
            };
            check(&what("extract"), &extract, with, kept);
            let with = size(&ins, lane, &|fb, x| {
                let v = make(fb, x)?;
                let r = chunk::insert(fb, s, Some(1), None, &[v, x[2]])?;
                first(fb, r)
            });
            let insert = S::LaneInsert {
                dst: s,
                at: Some(1),
                index: None,
            };
            check(&what("insert"), &insert, with, unmoved(&ins));
            // Dynamic moves.
            for index in [int(32), wide(64)] {
                let ins = [v4, v4, lane, index];
                let kept = size(&ins, lane, &|fb, x| {
                    make(fb, x)?;
                    Ok(x[2])
                });
                let with = size(&ins, lane, &|fb, x| {
                    let v = make(fb, x)?;
                    chunk::extract(fb, s, None, Some(index), &[v, x[3]])
                });
                let extract = S::LaneExtract {
                    src: s,
                    at: None,
                    index: Some(index),
                };
                check(&what("dynamic extract"), &extract, with, kept);
                let with = size(&ins, lane, &|fb, x| {
                    let v = make(fb, x)?;
                    let r = chunk::insert(fb, s, None, Some(index), &[v, x[2], x[3]])?;
                    first(fb, r)
                });
                let insert = S::LaneInsert {
                    dst: s,
                    at: None,
                    index: Some(index),
                };
                check(&what("dynamic insert"), &insert, with, unmoved(&ins));
            }
            // A shuffle of every lane from both operands.
            let ins = [v4, v4];
            let mask: Box<[u32]> = (0..u32::from(n))
                .map(|i| (i * 5 + 3) % (2 * u32::from(n)))
                .collect();
            let with = size(&ins, lane, &|fb, x| {
                let v = make(fb, x)?;
                let r = chunk::shuffle(fb, s, s, s, &mask, &[v, v])?;
                first(fb, r)
            });
            let shuffle = S::LaneShuffle {
                a: s,
                b: s,
                out: s,
                mask: mask.clone(),
                fragment: false,
            };
            check(&what("shuffle"), &shuffle, with, unmoved(&ins));
            // Word bitcasts of 64-bit lanes to twice as many 32-bit lanes and back.
            if lane.bits() == 64 {
                let words = Shape {
                    held: int(32).held,
                    lanes: 2 * n,
                    chunked: true,
                };
                let with = size(&ins, lane, &|fb, x| {
                    let v = make(fb, x)?;
                    let u = chunk::bitcast(fb, s, words, v)?;
                    let r = chunk::bitcast(fb, words, s, u)?;
                    first(fb, r)
                });
                let there = S::WordBitcast { from: s, to: words };
                let back = S::WordBitcast { from: words, to: s };
                let bound = (there.words() + back.words()) as usize;
                let emitted = with - unmoved(&ins);
                assert!(
                    emitted <= bound,
                    "{}: {emitted} words above the kernel, bound {bound}",
                    what("bitcast and back")
                );
            }
        }
    }
}

/// A shuffle into a simdgroup matrix's fragment (64 lanes held whole) emits no more than its bound,
/// `chunk::fragment_shuffle_words`, for a merge taking lanes 0 and 1 from one operand and the rest
/// from the other, of 32-bit integer and of `float` lanes.
#[test]
fn a_fragment_shuffle_bounds_its_emission() {
    use super::super::chunk::{self, Shape};
    use super::super::ArithStrategy as S;
    use crate::lower::strategy::FamilyStrategy;
    use crate::spirv::serialize::serialize;
    let size = |ins: &[Form], out: Form, body: Body<'_>| -> usize {
        let m = kernel(ins, out, body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    for lane in [int(32), float(32)] {
        let four = Shape {
            held: lane.held,
            lanes: 4,
            chunked: false,
        };
        let s = Shape {
            held: lane.held,
            lanes: 64,
            chunked: true,
        };
        let v4 = vec(lane, 4);
        let ins = [v4, v4];
        let spread: Vec<u32> = (0..64).map(|i| i % 8).collect();
        let make = |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
            chunk::shuffle(fb, four, four, s, &spread, &x[..2])
        };
        let first =
            |fb: &mut FunctionBuilder<'_>, v: Value| chunk::extract(fb, s, Some(0), None, &[v]);
        let without = size(&ins, lane, &|fb, x| {
            let v = make(fb, x)?;
            first(fb, v)
        });
        let mask: Box<[u32]> = [0, 1].into_iter().chain(66..128).collect();
        let with = size(&ins, lane, &|fb, x| {
            let v = make(fb, x)?;
            let r = chunk::shuffle_fragment(fb, s, s, s, &mask, &[v, v])?;
            first(fb, r)
        });
        let strategy = S::LaneShuffle {
            a: s,
            b: s,
            out: s,
            mask,
            fragment: true,
        };
        let bound = strategy.words() as usize;
        assert!(
            with - without <= bound,
            "{lane:?}: {} words above the kernel, bound {bound}",
            with - without
        );
    }
}

/// The roundings a pipeline of OpenCL mode honors, with a wide side: a wide integer to `f32`
/// toward zero, up and down; and an `f32` to a wide integer to nearest even, up and down,
/// saturating (`convert_suffix.rs` holds the native forms').
#[test]
fn wide_conversions_honor_their_suffixes() {
    use super::convert_suffix::{
        clamped, f32_of, integral_of, rounding_floats, rounding_ints, value_of, DIRECTED, INTEGRAL,
    };
    let f32f = float(32);
    for bits in [40u8, 64] {
        let from = wide(bits);
        let cases: Vec<Case> = rounding_ints(bits).iter().map(|v| vec![vec![*v]]).collect();
        for signed in [true, false] {
            for mode in DIRECTED {
                check_wide(
                    &format!("wide i{bits} signed={signed} to f32 {mode:?}"),
                    &[from],
                    f32f,
                    &|fb, x| {
                        let kind = ConvertKind::ToFloat {
                            signed,
                            directed: Some(mode),
                        };
                        w::convert(fb, kind, from, f32f, x[0])
                    },
                    &cases,
                    &|c| Some(vec![f32_of(value_of(arg(c, 0), bits, signed), mode)]),
                );
            }
        }
    }
    let floats: Vec<Case> = rounding_floats().iter().map(|v| vec![vec![*v]]).collect();
    for signed in [true, false] {
        for mode in INTEGRAL {
            check_wide(
                &format!("f32 to wide i64 signed={signed} {mode:?}"),
                &[f32f],
                wide(64),
                &|fb, x| {
                    let kind = ConvertKind::ToInt {
                        signed,
                        round: Some(mode),
                    };
                    w::convert(fb, kind, f32f, wide(64), x[0])
                },
                &floats,
                &|c| {
                    let x = f64::from(f32v(arg(c, 0)));
                    Some(vec![saturated(integral_of(x, mode), 64, signed)])
                },
            );
        }
    }
}
