//! `air.convert`'s rounding and `.sat` where a compute pipeline of OpenCL mode honors them
//! ([`super::super::honored`]), on the CPU interpreter against exact references: an integer to
//! `f32` toward zero, up and down; and an `f32` to an integer to nearest even, up and down, with
//! and without saturation. The wide path's (no `shaderInt64`) are in `wide.rs`.

use super::super::honored;
use super::*;
use crate::air::intrinsics::{AirIntrinsic, Convert, Rounding};

/// The three roundings a conversion to `f32` honors beside the device's own.
pub(super) const DIRECTED: [Rounding; 3] = [Rounding::Zero, Rounding::Up, Rounding::Down];

/// The three roundings a conversion to an integer honors beside its truncation.
pub(super) const INTEGRAL: [Rounding; 3] = [Rounding::Even, Rounding::Up, Rounding::Down];

/// The `f32` bits of integer `n` rounded as `mode` says, exactly: the magnitude's top 24 bits,
/// moved up one where `mode` rounds the magnitude up and a bit below them is set.
pub(super) fn f32_of(n: i128, mode: Rounding) -> u64 {
    let sign = if n < 0 { 0x8000_0000u64 } else { 0 };
    let a = n.unsigned_abs();
    if a == 0 {
        return 0;
    }
    let e = 127 - a.leading_zeros();
    if e < 24 {
        return sign | u64::from((a as f32).to_bits());
    }
    let shift = e - 23;
    let (q, rest) = (a >> shift, a & ((1u128 << shift) - 1));
    let half = 1u128 << (shift - 1);
    let up = match (mode, n < 0) {
        (Rounding::Zero, _) | (Rounding::Up, true) | (Rounding::Down, false) => false,
        (Rounding::Up, false) | (Rounding::Down, true) => rest != 0,
        (Rounding::Even, _) => rest > half || (rest == half && q & 1 == 1),
    };
    let v = (q + u128::from(up)) << shift;
    sign | u64::from((v as f32).to_bits())
}

/// `x` rounded to an integral value as `mode` says.
pub(super) fn integral_of(x: f64, mode: Rounding) -> f64 {
    match mode {
        Rounding::Even => x.round_ties_even(),
        Rounding::Zero => x.trunc(),
        Rounding::Up => x.ceil(),
        Rounding::Down => x.floor(),
    }
}

/// The value of held integer bits `b` of `bits` bits read as `signed`.
pub(super) fn value_of(b: u64, bits: u8, signed: bool) -> i128 {
    if signed {
        i128::from(sext(b, bits))
    } else {
        i128::from(b & mask(bits))
    }
}

/// Integers around every place a conversion to `f32` rounds: `2^24` and the values beside it and
/// beside `2^25`, each with its negation, the bounds of 32 and 64 bits and the values that round to
/// them, as `bits`-bit held integers, and random ones.
pub(super) fn rounding_ints(bits: u8) -> Vec<u64> {
    let mut v: Vec<i128> = vec![0, 1, -1, 123_456_789, -123_456_789, 0x7FFF_FF80];
    for b in [24u32, 25, 26, 31, 32, 53, 62, 63, 64] {
        let p = 1i128 << b;
        for d in [-65, -64, -63, -3, -2, -1, 0, 1, 2, 3, 63, 64, 65] {
            v.extend([p + d, -(p + d)]);
        }
    }
    v.extend([
        0x7FFF_FF80_0000_0000,
        0x7FFF_FFC0_0000_0000,
        0xFFFF_FF80_0000_0000,
        0xFFFF_FFFF_FFFF_FF00,
        0x0000_0100_0001_0001,
        0x0020_0000_2000_0001,
    ]);
    let mut rng = Rng(0xC0_4E7);
    let mut out: Vec<u64> = v.iter().map(|n| (*n as u64) & mask(bits)).collect();
    out.extend(int_values(bits, &mut rng, 24));
    out.sort_unstable();
    out.dedup();
    out
}

/// `f32` values around every place a conversion to an integer rounds or saturates: ties and the
/// values beside them, each sign, the bounds of 8 to 64 bits and the values beside them, the
/// infinities and NaN.
pub(super) fn rounding_floats() -> Vec<u64> {
    let mut v: Vec<f32> = vec![
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        3.7,
        0.75,
        0.499_999_97,
        1.0,
        32767.6,
        32767.4,
        -32768.6,
        255.5,
        254.5,
        127.5,
        65535.5,
        1e10,
        f32::INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        1e-30,
    ];
    for k in [7, 8, 15, 16, 31, 32, 63, 64] {
        let p = 2f32.powi(k);
        v.extend([p, p - 0.5, p + 0.5, p - 1.0, p + 1.0, p - 1.5]);
        let b = p.to_bits();
        v.extend([f32::from_bits(b - 1), f32::from_bits(b + 1)]);
    }
    let mut out: Vec<u64> = v
        .iter()
        .flat_map(|x| [u64::from(x.to_bits()), u64::from((-x).to_bits())])
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn cases(values: &[u64]) -> Vec<Case> {
    values.iter().map(|v| vec![vec![*v]]).collect()
}

/// An integer of 32 or 64 bits converts to `f32` toward zero, up and down exactly, read as
/// signed and as unsigned; a 16-bit one is exact. The device's conversion answers to nearest even
/// on the interpreter, so every value between two `f32`s takes the step.
#[test]
fn an_integer_converts_to_f32_by_each_directed_rounding() {
    let f32f = float(32);
    for ibits in [16u8, 32, 64] {
        let from = int(ibits);
        let cs = cases(&rounding_ints(ibits));
        for signed in [true, false] {
            for mode in DIRECTED {
                check(
                    &format!("{}{ibits} to f32 {mode:?}", if signed { "s" } else { "u" }),
                    &[from],
                    f32f,
                    &|fb, x| {
                        let kind = ConvertKind::ToFloat {
                            signed,
                            directed: Some(mode),
                        };
                        math::convert(fb, kind, from, f32f, x[0], NanTest::IsNan)
                    },
                    &cs,
                    &|c| Some(vec![f32_of(value_of(arg(c, 0), ibits, signed), mode)]),
                    false,
                );
            }
        }
    }
}

/// The words reims measured in OpenCL mode on a Radeon Pro 5300M and a UHD Graphics 630, which
/// the two answer alike, and those of the probe of every directed form on both.
#[test]
fn directed_conversions_to_f32_answer_metal_s_words() {
    let f32f = float(32);
    for (signed, ibits, mode, x, want) in [
        (true, 32u8, Rounding::Zero, 16_777_219u64, 0x4b80_0001u64),
        (true, 32, Rounding::Up, 16_777_217, 0x4b80_0001),
        (true, 32, Rounding::Down, 16_777_219, 0x4b80_0001),
        (false, 32, Rounding::Zero, 0xFFFF_FFFF, 0x4f7f_ffff),
        (true, 32, Rounding::Up, 0x7FFF_FFFF, 0x4f00_0000),
        (true, 32, Rounding::Zero, 0x7FFF_FFFF, 0x4eff_ffff),
        (true, 32, Rounding::Down, 0x8000_0001, 0xcf00_0000),
        (true, 32, Rounding::Zero, 0x8000_0001, 0xceff_ffff),
        (true, 64, Rounding::Zero, 0x7FFF_FFFF_FFFF_FFFF, 0x5eff_ffff),
        (false, 64, Rounding::Down, u64::MAX, 0x5f7f_ffff),
        (false, 64, Rounding::Up, u64::MAX, 0x5f80_0000),
    ] {
        let from = int(ibits);
        check(
            &format!("{x:#x} to f32 {mode:?} signed={signed}"),
            &[from],
            f32f,
            &|fb, v| {
                let kind = ConvertKind::ToFloat {
                    signed,
                    directed: Some(mode),
                };
                math::convert(fb, kind, from, f32f, v[0], NanTest::IsNan)
            },
            &cases(&[x]),
            &|_| Some(vec![want]),
            false,
        );
    }
}

/// Four lanes convert each by itself.
#[test]
fn a_vector_converts_to_f32_lane_by_lane() {
    let (from, to) = (vec(int(32), 4), vec(float(32), 4));
    let lanes = [16_777_217u64, 0xFFFF_FFFF, 33_554_435, 0x8000_0001];
    for mode in DIRECTED {
        check(
            &format!("int4 to float4 {mode:?}"),
            &[from],
            to,
            &|fb, x| {
                let kind = ConvertKind::ToFloat {
                    signed: true,
                    directed: Some(mode),
                };
                math::convert(fb, kind, from, to, x[0], NanTest::IsNan)
            },
            &[vec![lanes.to_vec()]],
            &|_| {
                Some(
                    lanes
                        .iter()
                        .map(|l| f32_of(value_of(*l, 32, true), mode))
                        .collect(),
                )
            },
            false,
        );
    }
}

/// An `f32` converts to every integer width to nearest even, up and down, then saturates as the
/// conversion without a rounding does (NaN to 0, out of range to the bound), whichever way the
/// entry tells a NaN.
#[test]
fn an_f32_converts_to_an_integer_by_each_rounding() {
    let f32f = float(32);
    let cs = cases(&rounding_floats());
    for ibits in [8u8, 16, 32, 64] {
        let to = int(ibits);
        for signed in [true, false] {
            for mode in INTEGRAL {
                for test in [NanTest::IsNan, NanTest::Bits] {
                    check(
                        &format!(
                            "f32 to {}{ibits} {mode:?} ({test:?})",
                            if signed { "s" } else { "u" }
                        ),
                        &[f32f],
                        to,
                        &|fb, x| {
                            let kind = ConvertKind::ToInt {
                                signed,
                                round: Some(mode),
                            };
                            math::convert(fb, kind, f32f, to, x[0], test)
                        },
                        &cs,
                        &|c| {
                            let x = f64::from(f32::from_bits(arg(c, 0) as u32));
                            Some(vec![saturated(integral_of(x, mode), ibits, signed)])
                        },
                        false,
                    );
                }
            }
        }
    }
}

/// The words of the brief's table and of the probe, which both oracles answer alike.
#[test]
fn rounded_conversions_to_an_integer_answer_metal_s_words() {
    let f32f = float(32);
    for (ibits, signed, mode, x, want) in [
        (32u8, true, Rounding::Even, 1.5f32, 2u64),
        (32, true, Rounding::Up, 0.5, 1),
        (32, true, Rounding::Down, -0.5, 0xFFFF_FFFF),
        (32, true, Rounding::Even, 32767.6, 32768),
        (16, true, Rounding::Even, 32767.6, 32767),
        (8, false, Rounding::Even, 0.5, 0),
        (8, false, Rounding::Even, 1.5, 2),
        (8, false, Rounding::Even, 2.5, 2),
        (8, false, Rounding::Even, -0.5, 0),
        (8, false, Rounding::Even, 254.5, 254),
        (8, false, Rounding::Up, 254.5, 255),
        (8, false, Rounding::Even, 127.5, 128),
        (32, false, Rounding::Up, 4_294_967_040.0, 0xFFFF_FF00),
    ] {
        let to = int(ibits);
        check(
            &format!("{x} to {ibits} {mode:?}"),
            &[f32f],
            to,
            &|fb, v| {
                let kind = ConvertKind::ToInt {
                    signed,
                    round: Some(mode),
                };
                math::convert(fb, kind, f32f, to, v[0], NanTest::IsNan)
            },
            &cases(&[u64::from(x.to_bits())]),
            &|_| Some(vec![want]),
            false,
        );
    }
}

/// The conversion of `name` with types `from` and `to` as the arith family selects it in a
/// pipeline of OpenCL mode or not.
fn selected(name: &str, open_cl_mode: bool, from: Form, to: Form) -> ConvertKind {
    let Some(AirIntrinsic::Convert(c)) = AirIntrinsic::parse(name) else {
        panic!("{name}");
    };
    let plain = plain_kind(&c);
    honored(open_cl_mode, &c, plain, from, to)
}

/// The kind the conversion's types name, as selection builds it.
fn plain_kind(c: &Convert) -> ConvertKind {
    use crate::air::intrinsics::NumKind;
    match (c.dst_kind, c.src_kind) {
        (NumKind::Float, NumKind::Float) => ConvertKind::Float,
        (NumKind::Float, s) => ConvertKind::ToFloat {
            signed: s == NumKind::Signed,
            directed: None,
        },
        (d, NumKind::Float) => ConvertKind::ToInt {
            signed: d == NumKind::Signed,
            round: None,
        },
        (_, s) => ConvertKind::Int {
            signed: s == NumKind::Signed,
        },
    }
}

/// OpenCL mode honors a directed rounding to `f32` and a rounding of an `f32` to an integer other
/// than truncation; it leaves the device's rounding to nearest even, truncation, a `half` side and
/// a suffix between integers as the types name them, and outside OpenCL mode every suffix is the
/// conversion its types name.
#[test]
fn only_a_pipeline_of_opencl_mode_honors_a_suffix() {
    let (i32f, f32f, h) = (int(32), float(32), float(16));
    let honored_kinds = [
        (
            "air.convert.f.f32.s.i32.rtz",
            i32f,
            f32f,
            ConvertKind::ToFloat {
                signed: true,
                directed: Some(Rounding::Zero),
            },
        ),
        (
            "air.convert.f.f32.u.i32.rtp",
            i32f,
            f32f,
            ConvertKind::ToFloat {
                signed: false,
                directed: Some(Rounding::Up),
            },
        ),
        (
            "air.convert.s.i32.f.f32.rte",
            f32f,
            i32f,
            ConvertKind::ToInt {
                signed: true,
                round: Some(Rounding::Even),
            },
        ),
        (
            "air.convert.u.i32.f.f32.rtn.sat",
            f32f,
            i32f,
            ConvertKind::ToInt {
                signed: false,
                round: Some(Rounding::Down),
            },
        ),
    ];
    for (name, from, to, want) in honored_kinds {
        assert_eq!(
            selected(name, true, from, to),
            want,
            "{name} in OpenCL mode"
        );
        let Some(AirIntrinsic::Convert(c)) = AirIntrinsic::parse(name) else {
            panic!("{name}");
        };
        assert_eq!(
            selected(name, false, from, to),
            plain_kind(&c),
            "{name} outside OpenCL mode"
        );
    }
    for (name, from, to) in [
        ("air.convert.f.f32.s.i32.rte", i32f, f32f),
        ("air.convert.s.i32.f.f32.rtz", f32f, i32f),
        ("air.convert.s.i32.f.f32.rtz.sat", f32f, i32f),
        ("air.convert.f.f16.s.i32.rtz", i32f, h),
        ("air.convert.f.f16.f.f32.rtp", f32f, h),
        ("air.convert.s.i16.f.f16.rtp.sat", h, int(16)),
        ("air.convert.s.i16.s.i32.rtp", i32f, int(16)),
        ("air.convert.u.i8.s.i32.rtz.sat", i32f, int(8)),
    ] {
        let Some(AirIntrinsic::Convert(c)) = AirIntrinsic::parse(name) else {
            panic!("{name}");
        };
        for mode in [true, false] {
            assert_eq!(
                selected(name, mode, from, to),
                plain_kind(&c),
                "{name} (OpenCL mode {mode})"
            );
        }
    }
}
