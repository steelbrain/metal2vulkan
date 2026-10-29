//! Apple's accuracy rows and the per-word check over synthetic operands (the tests own their
//! inputs): each bound admits what MSL states and, one representable value further, no more.

use super::msl::{pow2, Admits, Expr, Format, Math};
use super::*;
use crate::case::ResourceRole;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::pipeline::PrecisionNote;
use std::sync::OnceLock;

const F32: Format = Format::F32;
const F16: Format = Format::F16;

/// The profile the committed venue profile `stem` derives (`validation/profiles/<stem>.json`),
/// as the list reads a venue's.
fn derived(stem: &str) -> DeviceProfile {
    let venue = crate::c3::VenueProfile::committed(stem).unwrap();
    crate::acceptance::derived_profile(venue.description).unwrap()
}

/// lavapipe's profile: its `float` functions are the device's own.
fn lavapipe() -> &'static DeviceProfile {
    static PROFILE: OnceLock<DeviceProfile> = OnceLock::new();
    PROFILE.get_or_init(|| derived("gate-lavapipe"))
}

/// The M3 Max's MoltenVK profile: its `float` functions are Metal's own.
fn moltenvk() -> &'static DeviceProfile {
    static PROFILE: OnceLock<DeviceProfile> = OnceLock::new();
    PROFILE.get_or_init(|| derived("moltenvk-m3"))
}

/// What `computes` admits in `format` under `math` at point operands `values`.
fn admits(computes: &str, format: Format, math: Math, values: &[f64]) -> Admits {
    let expr = Expr::parse(computes).unwrap();
    let bindings = values
        .iter()
        .map(|value| msl::operand(*value, format, math))
        .collect::<Vec<_>>();
    msl::admits(&expr, &bindings, format, math).unwrap()
}

/// The `f32` `steps` representable values away from `x`.
fn step(x: f32, steps: i32) -> f64 {
    f64::from(f32::from_bits((x.to_bits() as i32 + steps) as u32))
}

fn unstated(admits: &Admits) -> &str {
    match admits {
        Admits::Unstated(why) => why,
        Admits::Values { .. } => panic!("stated: {admits:?}"),
    }
}

#[test]
fn ulps_and_formats_follow_msl_section_8_4() {
    assert_eq!(F32.ulp(1.5), pow2(-23));
    // At a power of two the nearest other value is below it.
    assert_eq!(F32.ulp(1.0), pow2(-24));
    assert_eq!(F32.ulp(-1.0), pow2(-24));
    // Between two values, their distance.
    assert_eq!(F32.ulp(1.0 + 1e-12), pow2(-23));
    assert_eq!(F32.ulp(0.0), pow2(-149));
    assert_eq!(F32.ulp(1e-40), pow2(-149));
    assert_eq!(F32.ulp(pow2(-126)), pow2(-149));
    assert_eq!(F32.ulp(1e39), pow2(104));
    assert_eq!(F16.ulp(1.0), pow2(-11));
    assert_eq!(F16.ulp(1.5), pow2(-10));
    assert_eq!(F16.ulp(1e-6), pow2(-24));
    assert_eq!(F16.decode(0x3c00), 1.0);
    assert_eq!(F16.decode(0x8000).to_bits(), (-0.0f64).to_bits());
    assert_eq!(F16.decode(0x0001), pow2(-24));
    assert_eq!(F16.decode(0x7bff), 65504.0);
    assert_eq!(F16.decode(0xfc00), f64::NEG_INFINITY);
    assert!(F16.decode(0x7e00).is_nan());
    assert_eq!(F32.decode(0x3f80_0000), 1.0);
    assert!(F32.represents(1.5) && !F32.represents(0.1));
    assert!(F16.represents(65504.0) && !F16.represents(65505.0));
    assert!(F32.is_subnormal(1e-40) && !F32.is_subnormal(0.0) && !F32.is_subnormal(pow2(-126)));
}

#[test]
fn expressions_name_msl_operations_over_lettered_operands() {
    let expr = Expr::parse("sub(a, mul(b, add(mul(sign(a), c), d)))").unwrap();
    assert_eq!(expr.operands(), 4);
    let mut calls = vec![];
    expr.calls(&mut calls);
    assert_eq!(
        calls.iter().map(|prim| prim.name()).collect::<Vec<_>>(),
        ["sub", "mul", "add", "sign"]
    );
    assert_eq!(expr.reference(&[2.0, 0.5, 1.0, 0.0]), 1.5);
    assert_eq!(
        Expr::parse(" powr( a ,2.5 )").unwrap().reference(&[4.0]),
        32.0
    );
    for (bad, why) in [
        ("cbrt(a)", "no MSL operation"),
        ("atan(a, b)", "takes 1 arguments"),
        ("atan(a", "not closed"),
        ("atan", "needs its arguments"),
        ("atan(a) b", "unexpected"),
        ("", "expected an operand"),
    ] {
        let error = Expr::parse(bad).unwrap_err();
        assert!(error.contains(why), "{bad:?}: {error}");
    }
}

/// Table 8.2's `atan` (5 ulp) at 1: pi/4, whose nearest `f32` is 0.37 ulp above it.
#[test]
fn a_stated_ulp_bound_admits_its_ulps_and_one_value_more_is_outside() {
    let quarter = std::f32::consts::FRAC_PI_4;
    let fast = admits("atan(a)", F32, Math::Fast, &[1.0]);
    for inside in [-5, -1, 0, 1, 4] {
        assert!(fast.admits(step(quarter, inside)), "{inside}");
    }
    for outside in [-6, 5, -55] {
        assert!(!fast.admits(step(quarter, outside)), "{outside}");
    }
    // Table 8.1's atan2 (6 ulp) at C99's infinite diagonal, pi/4: llvmpipe's 0x3f490fa4 is 55
    // values below Metal's.
    let precise = admits(
        "atan2(a, b)",
        F32,
        Math::Precise,
        &[f64::INFINITY, f64::INFINITY],
    );
    assert!(precise.admits(step(quarter, -6)));
    assert!(!precise.admits(f64::from(f32::from_bits(0x3f49_0fa4))));
    // An ulp bound does not pin a zero's sign: atan2(-0, +1) admits +0.
    let zero = admits("atan2(a, b)", F32, Math::Precise, &[-0.0, 1.0]);
    assert!(zero.admits(0.0) && zero.admits(-0.0) && zero.admits(pow2(-147)));
    assert!(!zero.admits(pow2(-146)));
}

/// Tables 8.1 and 8.2's `sinpi`, `cospi`, and `tanpi`. The exact value reduces by half turns, so
/// its zeros are exact and `tanpi`'s poles are C23's (and Metal's) signed infinities; under fast
/// math `sinpi` and `cospi` are within `2^-13` on `[-1, 1]` and unstated past it, and `tanpi` is
/// `tan(x * pi)` of `M_PI_F`, whose `sin` and `cos` take the rounded product's range.
#[test]
fn pi_functions_follow_tables_8_1_and_8_2() {
    // Exact zeros, poles and units.
    for (computes, x, want) in [
        ("sinpi(a)", 1.0, 0.0),
        ("sinpi(a)", 0.5, 1.0),
        ("cospi(a)", 0.5, 0.0),
        ("cospi(a)", 3.0, -1.0),
        ("tanpi(a)", 0.25, 1.0),
        ("tanpi(a)", 0.5, f64::INFINITY),
        ("tanpi(a)", -0.5, f64::NEG_INFINITY),
        ("tanpi(a)", 1.5, f64::NEG_INFINITY),
        ("tanpi(a)", 100.5, f64::INFINITY),
    ] {
        let got = Expr::parse(computes).unwrap().reference(&[x]);
        assert!(
            got == want || (got - want).abs() < 1e-15,
            "{computes} at {x}: {got}"
        );
    }
    // Table 8.1: 4 ulp, 6 for tanpi.
    let x = 0.3f64;
    let exact = (std::f64::consts::PI * x).sin();
    let precise = admits("sinpi(a)", F32, Math::Precise, &[x]);
    assert!(precise.admits(exact + 3.5 * F32.ulp(exact)));
    assert!(!precise.admits(exact + 4.5 * F32.ulp(exact)));
    let exact = (std::f64::consts::PI * x).tan();
    let precise = admits("tanpi(a)", F32, Math::Precise, &[x]);
    assert!(precise.admits(exact + 5.5 * F32.ulp(exact)));
    assert!(!precise.admits(exact + 6.5 * F32.ulp(exact)));
    // Table 8.2: 2^-13 absolute on [-1, 1], nothing stated past it.
    let exact = (std::f64::consts::PI * x).cos();
    let fast = admits("cospi(a)", F32, Math::Fast, &[x]);
    assert!(fast.admits(exact + pow2(-13) * 0.99) && !fast.admits(exact + pow2(-13) * 1.01));
    assert!(unstated(&admits("sinpi(a)", F32, Math::Fast, &[2.5])).contains("larger otherwise"));
    // tan(x * pi): the sine and cosine of the product's range, each within 2^-13.
    let fast = admits("tanpi(a)", F32, Math::Fast, &[x]);
    let exact = (std::f64::consts::PI * x).tan();
    assert!(fast.admits(exact) && fast.admits(exact * (1.0 + 1e-4)));
    assert!(!fast.admits(exact * 1.01));
    // Past [-pi, pi] for the product (M_PI_F is above pi), Apple states nothing.
    assert!(unstated(&admits("tanpi(a)", F32, Math::Fast, &[1.0])).contains("larger otherwise"));
}

#[test]
fn absolute_bounds_hold_on_their_domain_and_apple_states_nothing_past_it() {
    let reference = 0.5f64.sin();
    let sin = admits("sin(a)", F32, Math::Fast, &[0.5]);
    assert!(sin.admits(reference + pow2(-13) * 0.99));
    assert!(sin.admits(reference - pow2(-13) * 0.99));
    assert!(!sin.admits(reference + pow2(-13) * 1.01));
    // Outside [-pi, pi] Table 8.2 says only "larger otherwise".
    let far = admits("sin(a)", F32, Math::Fast, &[1e6]);
    assert!(unstated(&far).contains("larger otherwise"));
    // Without fast math, Table 8.1's 4 ulp holds everywhere.
    let precise = admits("sin(a)", F32, Math::Precise, &[1e6]);
    let exact = 1e6f64.sin();
    assert!(precise.admits(exact + 3.5 * F32.ulp(exact)));
    assert!(!precise.admits(exact + 4.5 * F32.ulp(exact)));
    // log2: 2^-22 absolute on [0.5, 2], else 2 ulp for x > 0, else undefined.
    let near = admits("log2(a)", F32, Math::Fast, &[1.5]);
    let exact = 1.5f64.log2();
    assert!(near.admits(exact + pow2(-22) * 0.99) && !near.admits(exact + pow2(-22) * 1.01));
    let eight = admits("log2(a)", F32, Math::Fast, &[8.0]);
    assert!(eight.admits(3.0 + pow2(-21)) && !eight.admits(3.0 + 3.0 * pow2(-22)));
    assert!(unstated(&admits("log2(a)", F32, Math::Fast, &[-1.0])).contains("undefined"));
    // exp2's bound grows with its argument: 3 + floor(2 * 10) = 23 ulp of 1024 (2^-14 each).
    let exp2 = admits("exp2(a)", F32, Math::Fast, &[10.0]);
    assert!(exp2.admits(1024.0 + 22.0 * pow2(-14)) && exp2.admits(1024.0 - 23.0 * pow2(-14)));
    assert!(!exp2.admits(1024.0 + 24.0 * pow2(-14)));
    // fast fmod is "Undefined".
    assert!(unstated(&admits("fmod(a, b)", F32, Math::Fast, &[1e10, 3.0])).contains("Undefined"));
    assert!(!admits("fmod(a, b)", F32, Math::Precise, &[1e10, 3.0]).admits(-512.0));
}

/// "Correctly rounded" admits both rounding modes section 8.2 allows, and no third value.
#[test]
fn correct_rounding_admits_nearest_even_and_toward_zero_only() {
    let third = admits("div(a, b)", F32, Math::Precise, &[1.0, 3.0]);
    let (nearest, toward_zero) = (
        f64::from(f32::from_bits(0x3eaa_aaab)),
        f64::from(f32::from_bits(0x3eaa_aaaa)),
    );
    assert!(third.admits(nearest) && third.admits(toward_zero));
    assert!(!third.admits(step(1.0 / 3.0, 2)) && !third.admits(step(1.0 / 3.0, -2)));
    // A representable result admits itself alone.
    let half = admits("div(a, b)", F32, Math::Precise, &[1.0, 2.0]);
    assert!(half.admits(0.5) && !half.admits(step(0.5, 1)) && !half.admits(step(0.5, -1)));
    // Table 8.2's division: 2.5 ulp for a divisor in [2^-126, 2^126], nothing past it. The ulp of
    // 0.5 is 2^-25, the spacing below it; above it the values are 2^-24 apart.
    let fast = admits("div(a, b)", F32, Math::Fast, &[1.0, 2.0]);
    assert!(fast.admits(step(0.5, 1)) && fast.admits(step(0.5, -2)));
    assert!(!fast.admits(step(0.5, 2)) && !fast.admits(step(0.5, -3)));
    let past = admits("div(a, b)", F32, Math::Fast, &[pow2(-125), pow2(127)]);
    assert!(unstated(&past).contains("x / y") || unstated(&past).contains("div"));
}

/// Section 8.1 lets a subnormal operand or result be flushed; section 8.5 leaves the flushed
/// zero's sign undefined. Fast math leaves a NaN or INF undefined.
#[test]
fn subnormals_may_flush_and_fast_math_leaves_nan_and_inf_undefined() {
    // A subnormal product may read as either zero, or as itself; nothing else.
    let product = admits("mul(a, b)", F32, Math::Precise, &[pow2(-140), 1.0]);
    assert!(product.admits(pow2(-140)) && product.admits(0.0) && product.admits(-0.0));
    assert!(!product.admits(pow2(-139)));
    // A subnormal divisor flushed gives an infinity of either sign; both flushed, a NaN.
    let quotient = admits("div(a, b)", F32, Math::Precise, &[pow2(-148), pow2(-149)]);
    for value in [2.0, f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN] {
        assert!(quotient.admits(value), "{value}");
    }
    assert!(!quotient.admits(1.0));
    // Under fast math a NaN or INF operand or result is undefined.
    assert!(unstated(&admits("atan(a)", F32, Math::Fast, &[f64::INFINITY])).contains("8.1"));
    assert!(unstated(&admits("atan(a)", F32, Math::Fast, &[f64::NAN])).contains("8.1"));
    let precise = admits("atan(a)", F32, Math::Precise, &[f64::INFINITY]);
    assert!(precise.admits(std::f64::consts::FRAC_PI_2));
    // Without fast math a NaN result is any NaN (MSL states no encoding), and only a NaN.
    let nan = admits("div(a, b)", F32, Math::Precise, &[0.0, 0.0]);
    assert!(nan.admits(f64::NAN) && !nan.admits(0.0));
    // sign of a NaN is 0.0 (Table 6.1).
    let sign = admits("sign(a)", F32, Math::Precise, &[f64::NAN]);
    assert!(sign.admits(0.0) && !sign.admits(1.0) && !sign.admits(f64::NAN));
}

/// Every MSL operation the bound reads is spelled as Metal compiles it (`metal` 32023.921, both
/// formats, both modes), and the product lowers each such AIR operation as a float operation on
/// every `float` fused multiply-add route: the bound asks the product, never a copy of it.
#[test]
fn every_operation_is_metals_air_and_the_product_lowers_it() {
    use msl::Prim;
    let spelled = |prim, format, math| AirOp::of(prim, format, math).to_string();
    assert_eq!(spelled(Prim::Tanh, F32, Math::Fast), "air.fast_tanh.f32");
    assert_eq!(spelled(Prim::Tanh, F32, Math::Precise), "air.tanh.f32");
    assert_eq!(spelled(Prim::Tanh, F16, Math::Fast), "air.tanh.f16");
    assert_eq!(spelled(Prim::Sign, F32, Math::Fast), "air.sign.f32");
    assert_eq!(spelled(Prim::Atan2, F32, Math::Fast), "air.fast_atan2.f32");
    assert_eq!(spelled(Prim::Recip, F16, Math::Fast), "fdiv");
    assert_eq!(spelled(Prim::Sub, F32, Math::Precise), "fsub");
    for prim in Prim::ALL {
        for format in [F32, F16] {
            for math in [Math::Fast, Math::Precise] {
                let op = AirOp::of(prim, format, math);
                if let AirOp::Call(name) = &op {
                    assert!(
                        matches!(AirIntrinsic::parse(name), Some(AirIntrinsic::Math(_))),
                        "{name}"
                    );
                }
                for profile in [lavapipe(), moltenvk()] {
                    if let Err(error) = op.product_notes(format, math, profile) {
                        panic!("{prim:?} {format:?} {math:?}: {error}");
                    }
                }
            }
        }
    }
}

/// The near-equivalents are the product's own, as its lowering records them: its fast `float`
/// division (Metal's precise division and every `half` division are a shared routine's correctly
/// rounded quotient, which records nothing), fast `fmod`, and a function it computes with a
/// division of its own (fast `float` `atan2`). Its transcendentals are the device's: `tanh`, fast
/// or precise, `float` or `half`, is the device's `Tanh`, a transcendental, not a near-equivalent;
/// fast `float` `sqrt` is Metal's `x * rsqrt(x)` on the device's `rsqrt`, and precise `sqrt` and
/// `fmod`, `float` and `half`, are the one right value, which records nothing. On a device whose
/// `float` functions are Metal's own, a fast division and those functions are Metal's own under
/// fast math and record nothing there either.
#[test]
fn the_products_own_approximations_are_its_near_equivalents() {
    use msl::Prim;
    let note_on = |prim, format, math, profile| {
        near_equivalent(prim, format, math, profile)
            .unwrap()
            .map(|(note, _)| note)
    };
    let note = |prim, format, math| note_on(prim, format, math, lavapipe());
    let arithmetic = Some(PrecisionNote::DeviceArithmetic);
    for prim in [Prim::Div, Prim::Recip] {
        for format in [F32, F16] {
            for math in [Math::Precise, Math::Fast] {
                let fast_float = format == F32 && math == Math::Fast;
                let want = if fast_float { arithmetic } else { None };
                assert_eq!(
                    note(prim, format, math),
                    want,
                    "{prim:?} {format:?} {math:?}"
                );
                assert_eq!(
                    note_on(prim, format, math, moltenvk()),
                    None,
                    "{prim:?} {format:?} {math:?} on MoltenVK"
                );
            }
        }
    }
    for format in [F32, F16] {
        for math in [Math::Precise, Math::Fast] {
            let fast_float = format == F32 && math == Math::Fast;
            let want = if fast_float { arithmetic } else { None };
            assert_eq!(note(Prim::Fmod, format, math), want, "{format:?} {math:?}");
            assert_eq!(note(Prim::Sqrt, format, math), None, "{format:?} {math:?}");
        }
    }
    assert_eq!(note(Prim::Atan2, F32, Math::Fast), arithmetic);
    assert_eq!(note(Prim::Atan2, F32, Math::Precise), None);
    assert_eq!(note(Prim::Atan2, F16, Math::Fast), None);
    assert_eq!(note_on(Prim::Atan2, F32, Math::Fast, moltenvk()), None);
    for format in [F32, F16] {
        for math in [Math::Precise, Math::Fast] {
            for profile in [lavapipe(), moltenvk()] {
                assert_eq!(
                    note_on(Prim::Tanh, format, math, profile),
                    None,
                    "{format:?} {math:?}"
                );
            }
        }
    }
    for prim in [
        Prim::Sin,
        Prim::Exp,
        Prim::Pow,
        Prim::Powr,
        Prim::Sign,
        Prim::Add,
        Prim::Mul,
    ] {
        assert_eq!(note(prim, F32, Math::Fast), None, "{prim:?}");
    }

    // Fast tanh(44): t + 1 is past 2^126, where Table 8.2 states no accuracy for the division.
    // Metal answers 0, the device 1. The product's fast tanh is the device's `Tanh`, which records
    // no division of its own, so the word stays Apple's unstated one even where the row records
    // the device's arithmetic for another operation.
    let lanes = vec![BoundedLane {
        words: words(0, 1, 1),
        result: Part::F32,
        math: Math::Fast,
        computes: "tanh(a)".into(),
        operands: vec![Operand::Bits {
            format: Format::F32,
            first: 44.0f32.to_bits(),
            stride: 1,
        }],
    }];
    let judge = |recorded: &[PrecisionNote]| {
        check(
            &lanes,
            &[],
            &recorded.iter().copied().collect(),
            lavapipe(),
            &0.0f32.to_le_bytes(),
            &1.0f32.to_le_bytes(),
        )
        .words[0]
            .verdict
            .clone()
    };
    let WordVerdict::Unstated(why) = judge(&[PrecisionNote::DeviceTranscendentals]) else {
        panic!("tanh(44) is stated");
    };
    assert!(why.contains("div"), "{why}");
    let device = judge(&[
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceTranscendentals,
    ]);
    assert_eq!(device.label(), "unstated", "{device:?}");
    assert!(!device.keeps_open(UnstatedWords::WithinPromise));
}

/// Fast `atan2` with a divisor past `2^126` (case f6524207's lanes 351 and 26): Table 8.2's formula
/// `atan(y / x)` takes the fast division, whose accuracy it states only for a divisor in `[2^-126,
/// 2^126]`, so Apple states none for the word. On MoltenVK the product's fast `atan2` in a
/// fast-math entry is Metal's own and records no note, so such a word is within Apple's promise
/// even where the row records the device's arithmetic for another operation; where the product
/// computes it with a division of its own, it stays open.
#[test]
fn fast_atan2_past_the_divisors_domain_is_unstated() {
    for (y, x) in [
        (1308.72f32, f32::from_bits(0x7ec9_0fdb)),
        (6_588_397.5, f32::MAX),
    ] {
        let stated = admits(
            "atan2(a, b)",
            F32,
            Math::Fast,
            &[f64::from(y), f64::from(x)],
        );
        assert!(matches!(stated, Admits::Unstated(_)), "{y} {x}: {stated:?}");
        let lanes = vec![BoundedLane {
            words: words(0, 1, 1),
            result: Part::F32,
            math: Math::Fast,
            computes: "atan2(a, b)".into(),
            operands: [y, x]
                .iter()
                .map(|value| Operand::Bits {
                    format: Format::F32,
                    first: value.to_bits(),
                    stride: 1,
                })
                .collect(),
        }];
        let recorded = BTreeSet::from([PrecisionNote::DeviceArithmetic]);
        // Metal answers +0; MoltenVK `0x055049fa` at the first pair.
        let verdict = |profile| {
            check(
                &lanes,
                &[],
                &recorded,
                profile,
                &0u32.to_le_bytes(),
                &0x0550_49fau32.to_le_bytes(),
            )
            .words[0]
                .verdict
                .clone()
        };
        let own = verdict(moltenvk());
        assert_eq!(own.label(), "unstated", "{y} {x}: {own:?}");
        assert!(!own.keeps_open(UnstatedWords::WithinPromise));
        let near = verdict(lavapipe());
        assert_eq!(near.label(), "near_equivalent", "{y} {x}: {near:?}");
    }
}

/// The verdict on one `float` word computing `computes` of the `float` bit patterns `operands`
/// under `math`, where Metal answers `metal` and the device `device`, in a row recording
/// `recorded`, on lavapipe.
fn verdict_of(
    computes: &str,
    math: Math,
    operands: &[u32],
    answers: (u32, u32),
    recorded: &[PrecisionNote],
) -> WordVerdict {
    verdict_on(lavapipe(), computes, math, operands, answers, recorded)
}

/// [`verdict_of`] on `profile`'s device.
fn verdict_on(
    profile: &DeviceProfile,
    computes: &str,
    math: Math,
    operands: &[u32],
    (metal, device): (u32, u32),
    recorded: &[PrecisionNote],
) -> WordVerdict {
    let lanes = vec![BoundedLane {
        words: words(0, 1, 1),
        result: Part::F32,
        math,
        computes: computes.into(),
        operands: operands
            .iter()
            .map(|bits| Operand::Bits {
                format: Format::F32,
                first: *bits,
                stride: 1,
            })
            .collect(),
    }];
    check(
        &lanes,
        &[],
        &recorded.iter().copied().collect(),
        profile,
        &metal.to_le_bytes(),
        &device.to_le_bytes(),
    )
    .words[0]
        .verdict
        .clone()
}

/// The verdict on the low `half` of one word computing `computes` of the `half` bit pattern
/// `operand` under `math`, where Metal answers `metal` and the device `device`.
fn half_verdict_of(
    computes: &str,
    math: Math,
    operand: u32,
    (metal, device): (u16, u16),
    recorded: &[PrecisionNote],
) -> WordVerdict {
    let lanes = vec![BoundedLane {
        words: words(0, 1, 1),
        result: Part::F16Low,
        math,
        computes: computes.into(),
        operands: vec![Operand::Bits {
            format: Format::F16,
            first: operand,
            stride: 1,
        }],
    }];
    check(
        &lanes,
        &[],
        &recorded.iter().copied().collect(),
        lavapipe(),
        &u32::from(metal).to_le_bytes(),
        &u32::from(device).to_le_bytes(),
    )
    .words[0]
        .verdict
        .clone()
}

/// Every note the product records on a `float` or `half` arithmetic row.
const EVERY_ARITHMETIC_NOTE: [PrecisionNote; 4] = [
    PrecisionNote::DeviceTranscendentals,
    PrecisionNote::DeviceArithmetic,
    PrecisionNote::DeviceDenormals,
    PrecisionNote::DeviceNanEncoding,
];

/// Metal answers the canonical NaN reproducibly, so where both answers are NaNs of an operation
/// whose lowering leaves the encoding to the device, and the row records `DeviceNanEncoding`, the
/// value stays open with the exact path. On lavapipe, which is not measured to answer Metal's NaN,
/// the product's `float` lowerings that leave a NaN's encoding to the device record the note:
/// precise `pow`, and fast `sin` and `cos` (the device's instructions, whose NaN over an infinity
/// is the device's). Every other operation this check reads answers a NaN whose encoding the
/// product states (`NativeNanEncoding`) or none, so a differing NaN of `div` or `mul` is not the
/// note's even where the row records it (another operation of the module may), and the verdict is
/// Apple's bound's; Metal's precise and fast divisions answer `0x7fc00000` themselves. A differing
/// NaN of fast `sin` is the note's; of precise `sin`, which does not record it, it is not. Two
/// different answers, not both NaNs, are not the encoding's.
#[test]
fn a_nan_encoding_the_product_answers_is_not_a_near_equivalent() {
    let (metal, device) = (0x7fc0_0000, 0xffc0_0000);
    let zero = 0.0f32.to_bits();
    let infinity = f32::INFINITY.to_bits();
    let nan = [PrecisionNote::DeviceNanEncoding];
    for math in [Math::Precise, Math::Fast] {
        let device_nans: &[msl::Prim] = match math {
            Math::Precise => &[msl::Prim::Pow],
            Math::Fast => &[msl::Prim::Sin, msl::Prim::Cos],
        };
        for prim in msl::Prim::ALL {
            let (op, notes) = lowering_notes(prim, F32, math, lavapipe()).unwrap();
            assert_eq!(
                notes.contains(&PrecisionNote::DeviceNanEncoding),
                device_nans.contains(&prim),
                "{op} under {math:?}: {notes:?}"
            );
        }
        let product = verdict_of("mul(a, b)", math, &[infinity, zero], (metal, device), &nan);
        assert_ne!(product.label(), "near_equivalent", "{math:?}: {product:?}");
        // Apple's verdict: within for the precise division, unstated for the fast one (a zero
        // divisor is outside its domain); never the encoding's.
        let division = verdict_of("div(a, b)", math, &[zero, zero], (metal, device), &nan);
        assert_ne!(
            division.label(),
            "near_equivalent",
            "{math:?}: {division:?}"
        );
    }
    // A product of a NaN operand.
    let quiet = 0x7fc0_0001;
    let mul = verdict_of(
        "mul(a, b)",
        Math::Precise,
        &[quiet, zero],
        (metal, device),
        &nan,
    );
    assert_ne!(mul.label(), "near_equivalent", "{mul:?}");
    // Fast `sin` is the device's `Sin`, whose NaN of an infinity is the device's: the note's, open.
    let sin = verdict_of("sin(a)", Math::Fast, &[infinity], (metal, device), &nan);
    assert_eq!(sin.label(), "near_equivalent", "{sin:?}");
    assert!(sin.keeps_open(UnstatedWords::WithinPromise), "{sin:?}");
    // Precise `sin` records no `DeviceNanEncoding`: a row recording the note for another
    // operation does not make this value the note's.
    let sin = verdict_of("sin(a)", Math::Precise, &[infinity], (metal, device), &nan);
    assert_ne!(sin.label(), "near_equivalent", "{sin:?}");
    // Precise `pow` of a negative base and a non-integer exponent: a NaN the device encodes.
    let pow = verdict_of(
        "pow(a, b)",
        Math::Precise,
        &[(-1.0f32).to_bits(), 0.5f32.to_bits()],
        (metal, device),
        &nan,
    );
    assert_eq!(pow.label(), "near_equivalent", "{pow:?}");
    let one = 1.0f32.to_bits();
    let finite = verdict_of("div(a, b)", Math::Fast, &[one, one], (one, one + 1), &nan);
    assert_ne!(finite.label(), "near_equivalent", "{finite:?}");
}

/// Metal flushes a `float` denormal operand, intermediate, and result; where the device keeps one
/// in an operation whose lowering records `DeviceDenormals` and the row records the note, the value
/// stays open even though section 8.5 lets a denormal be flushed or not. Fast `sinpi` is such an
/// operation: its lowering is not among those that answer Metal at `float` denormals themselves
/// (`FloatFn::metal_denormals`; precise `fmod` is one). The product's `float` `+ - *` and
/// divisions are the device's: on lavapipe, which is not measured to read denormals as Metal does,
/// each records the note, so a product's or quotient's subnormal stays open too; where the device
/// is measured to (MoltenVK), a product records none and its subnormal keeps Apple's verdict, even
/// where the row records the note for another operation. A value no subnormal touches keeps its
/// verdict.
#[test]
fn a_denormal_the_device_keeps_stays_open() {
    let tiny = 1u32; // the smallest positive `float` subnormal
    let one = 1.0f32.to_bits();
    let denormals = [PrecisionNote::DeviceDenormals];
    // The premise: the product records the note for fast `sinpi` and no division of its own.
    let (_, notes) = lowering_notes(msl::Prim::SinPi, F32, Math::Fast, lavapipe()).unwrap();
    assert!(
        notes.contains(&PrecisionNote::DeviceDenormals)
            && !notes.contains(&PrecisionNote::DeviceArithmetic),
        "{notes:?}"
    );
    // A word whose Metal answer to `fast::sinpi(tiny)` is +0 (the operand read as zero) and whose
    // device keeps the subnormal: pi * 2^-149 rounds to 3 * 2^-149. Both
    // are within Table 8.2's absolute 2^-13.
    let sinpi =
        |recorded: &[PrecisionNote]| verdict_of("sinpi(a)", Math::Fast, &[tiny], (0, 3), recorded);
    let open = sinpi(&denormals);
    assert_eq!(open.label(), "near_equivalent", "{open:?}");
    for part in [
        "within Apple's stated accuracy",
        "DeviceDenormals",
        "a flushing select after each operation",
        "air.fast_sinpi.f32",
    ] {
        assert!(open.detail().contains(part), "{part}: {open:?}");
    }
    assert_eq!(sinpi(&[]), WordVerdict::Within);
    // `tiny * 1`: lavapipe's multiply keeps the subnormal, so the row's note is its; MoltenVK's
    // reads it as Metal does, so the note is not its.
    let product = |profile| {
        verdict_on(
            profile,
            "mul(a, b)",
            Math::Precise,
            &[tiny, one],
            (0, tiny),
            &denormals,
        )
    };
    let kept = product(lavapipe());
    assert_eq!(kept.label(), "near_equivalent", "{kept:?}");
    assert!(kept.detail().contains("(Metal's mul)"), "{kept:?}");
    assert_eq!(product(moltenvk()), WordVerdict::Within);
    // A quotient below the smallest normal (2^-100 / 2^30): Metal's +0, the device's 2^-130.
    let bits = |e| (pow2(e) as f32).to_bits();
    let (num, den) = (bits(-100), bits(30));
    let quotient = bits(-130);
    for math in [Math::Precise, Math::Fast] {
        let division = verdict_of("div(a, b)", math, &[num, den], (0, quotient), &denormals);
        assert_eq!(
            division.label(),
            "near_equivalent",
            "{math:?}: {division:?}"
        );
    }
    // A subnormal intermediate the next operation scales back up: fast::sinpi(2^-140) * 2^30,
    // +0 where the operand is read as zero, and the subnormal pi * 2^-140 scaled by 2^30 where
    // the device keeps it.
    let (small, big) = (bits(-140), bits(30));
    let kept = ((std::f64::consts::PI * pow2(-140)) as f32 * pow2(30) as f32).to_bits();
    let scaled = verdict_of(
        "mul(sinpi(a), b)",
        Math::Fast,
        &[small, big],
        (0, kept),
        &denormals,
    );
    assert_eq!(scaled.label(), "near_equivalent", "{scaled:?}");
    // The same through two multiplies (2^-140 * 2^-1 * 2^30): lavapipe keeps the intermediate,
    // MoltenVK reads it as Metal does.
    let multiplied = |profile| {
        verdict_on(
            profile,
            "mul(mul(a, b), c)",
            Math::Precise,
            &[small, bits(-1), big],
            (0, bits(-111)),
            &denormals,
        )
    };
    let kept = multiplied(lavapipe());
    assert_eq!(kept.label(), "near_equivalent", "{kept:?}");
    assert_eq!(multiplied(moltenvk()), WordVerdict::Within);
    // A product of normals one ulp off is outside the correctly rounded bound, as before.
    let two = 2.0f32.to_bits();
    let normal = verdict_of(
        "mul(a, b)",
        Math::Precise,
        &[two, one],
        (two, two + 1),
        &denormals,
    );
    assert_eq!(normal.label(), "outside", "{normal:?}");
    // `half` arithmetic records no `float` environment note: a `half` subnormal (2^-8 squared,
    // Metal's 2^-16 against the device's +0) is not the note's.
    let half_tiny = half_verdict_of("mul(a, a)", Math::Precise, 0x1c00, (0x0100, 0), &denormals);
    assert_ne!(half_tiny.label(), "near_equivalent", "{half_tiny:?}");
}

/// The signed-off readings' kind of value keeps its verdict whatever arithmetic notes the row
/// records: a fast transcendental a few ulps from Metal's within Table 8.2's bound (`da0ab893`,
/// `59103b58`, `a919193f`), and one where Apple states none (`179b6aa6`'s fast `sin` past `[-pi,
/// pi]`), touch no NaN, subnormal, or `half`.
#[test]
fn a_transcendental_within_apples_bound_keeps_its_verdict() {
    let x = 0.75f32;
    let log2 = x.log2().to_bits();
    let within = verdict_of(
        "log2(a)",
        Math::Fast,
        &[x.to_bits()],
        (log2, log2 + 2),
        &EVERY_ARITHMETIC_NOTE,
    );
    assert_eq!(within, WordVerdict::Within);
    let far = 1e6f32;
    let sin = far.sin().to_bits();
    let unstated = verdict_of(
        "sin(a)",
        Math::Fast,
        &[far.to_bits()],
        (sin, sin + 1),
        &EVERY_ARITHMETIC_NOTE,
    );
    assert_eq!(unstated.label(), "unstated", "{unstated:?}");
}

/// A zero's sign is a result like any other: an exact or correctly rounded zero admits its own
/// sign alone (Table 6.1's `sign`, IEEE 754's rounding of an exact zero), unless a flush (section
/// 8.5) or fast math (section 1.6.3) makes the sign insignificant; a bound that widens a zero by
/// ulps admits both.
#[test]
fn an_exact_zero_keeps_its_sign_unless_a_flush_or_fast_math_frees_it() {
    let sign = |math, x| admits("sign(a)", F32, math, &[x]);
    // Table 6.1: sign(-0) is -0, and sign(+0) is +0.
    let negative = sign(Math::Precise, -0.0);
    assert!(negative.admits(-0.0) && !negative.admits(0.0));
    let positive = sign(Math::Precise, 0.0);
    assert!(positive.admits(0.0) && !positive.admits(-0.0));
    // Exact arithmetic: -0 * 3 is -0, and 1 - 1 is +0 in both rounding modes section 8.2 allows.
    let product = admits("mul(a, b)", F32, Math::Precise, &[-0.0, 3.0]);
    assert!(product.admits(-0.0) && !product.admits(0.0));
    let difference = admits("sub(a, b)", F32, Math::Precise, &[1.0, 1.0]);
    assert!(difference.admits(0.0) && !difference.admits(-0.0));
    // Table 8.1's fmod is exact: fmod(-0, 3) is -0.
    let remainder = admits("fmod(a, b)", F32, Math::Precise, &[-0.0, 3.0]);
    assert!(remainder.admits(-0.0) && !remainder.admits(0.0));
    // The sign of a zero computed through an expression: sign(-0) * 2 is -0.
    let scaled = admits("mul(sign(a), b)", F32, Math::Precise, &[-0.0, 2.0]);
    assert!(scaled.admits(-0.0) && !scaled.admits(0.0));
    // A subnormal operand may flush to either zero (section 8.5), so its sign is either.
    let flushed = sign(Math::Precise, -pow2(-149));
    assert!(flushed.admits(-1.0) && flushed.admits(0.0) && flushed.admits(-0.0));
    assert!(!flushed.admits(1.0));
    // Under fast math the sign of a zero argument or result is insignificant (section 1.6.3).
    let fast = sign(Math::Fast, -0.0);
    assert!(fast.admits(0.0) && fast.admits(-0.0) && !fast.admits(1.0));
    // An ulp bound around an exact zero admits both zeros (atan(-0) within 4 ulp).
    let atan = admits("atan(a)", F32, Math::Precise, &[-0.0]);
    assert!(atan.admits(0.0) && atan.admits(-0.0));
}

/// Table 8.2's "Implemented as" formulas: the bound is derived from their parts' rows.
#[test]
fn formulas_take_the_bounds_of_their_parts() {
    // tan(0.5) = sin * (1 / cos): sin and cos within 2^-13 give about 2.2e-4 here.
    let exact = 0.5f64.tan();
    let tan = admits("tan(a)", F32, Math::Fast, &[0.5]);
    assert!(tan.admits(exact + 2.0e-4) && tan.admits(exact - 2.0e-4));
    assert!(!tan.admits(exact + 3.0e-4) && !tan.admits(exact - 3.0e-4));
    // sqrt(2) = 2 * rsqrt(2): rsqrt within 2 ulp, then a correctly rounded product.
    let root = std::f32::consts::SQRT_2;
    let sqrt = admits("sqrt(a)", F32, Math::Fast, &[2.0]);
    assert!(sqrt.admits(f64::from(root)) && sqrt.admits(step(root, 2)));
    assert!(!sqrt.admits(step(root, 5)) && !sqrt.admits(step(root, -5)));
    // "Special cases handled correctly": a zero is its own root, either zero under fast math.
    let zero = admits("sqrt(a)", F32, Math::Fast, &[0.0]);
    assert!(zero.admits(0.0) && zero.admits(-0.0) && !zero.admits(pow2(-149)));
    let negative_zero = admits("sqrt(a)", F32, Math::Precise, &[-0.0]);
    assert!(negative_zero.admits(-0.0) && !negative_zero.admits(0.0));
    // powr(2, 10) = exp2(10 * log2(2)): log2 within 2^-22 absolute, exp2 within 23 ulp.
    let powr = admits("powr(a, b)", F32, Math::Fast, &[2.0, 10.0]);
    assert!(powr.admits(1024.0) && powr.admits(1024.002) && !powr.admits(1024.006));
    assert!(unstated(&admits("powr(a, b)", F32, Math::Fast, &[0.0, 0.0])).contains("x = 0"));
    // tanh = (t - 1) / (t + 1), t = exp(2x): exact enough at 0.5; past 2^126 the division
    // states nothing (x = 44); a subnormal t (x = -44) may flush, and -1 conforms.
    let exact = 0.5f64.tanh();
    let tanh = admits("tanh(a)", F32, Math::Fast, &[0.5]);
    assert!(tanh.admits(exact) && !tanh.admits(exact + 1e-5));
    assert!(matches!(
        admits("tanh(a)", F32, Math::Fast, &[44.0]),
        Admits::Unstated(_)
    ));
    assert!(admits("tanh(a)", F32, Math::Fast, &[-44.0]).admits(-1.0));
    // atan2 under fast math is atan(y / x) with M_PI_F, undefined on an axis.
    let quadrant = admits("atan2(a, b)", F32, Math::Fast, &[1.0, -1.0]);
    assert!(quadrant.admits(3.0 * std::f64::consts::FRAC_PI_4));
    assert!(matches!(
        admits("atan2(a, b)", F32, Math::Fast, &[0.0, 1.0]),
        Admits::Unstated(_)
    ));
    // Half-precision rows are Table 8.3's, under fast math too: tanh within 1 ulp.
    let half = admits("tanh(a)", F16, Math::Fast, &[0.5]);
    let exact = 0.5f64.tanh();
    assert!(
        half.admits(exact + 0.9 * F16.ulp(exact)) && !half.admits(exact + 1.1 * F16.ulp(exact))
    );
}

fn buffer(binding: u32, bytes: &[u8]) -> BufferResource {
    BufferResource {
        binding,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    }
}

fn words(first: usize, count: usize, stride: usize) -> Words {
    Words {
        first,
        count,
        stride,
    }
}

/// The check reads each operand from the case's inputs and judges every differing value; the
/// one flag ([`UnstatedWords`]) decides whether a word Apple states nothing for keeps a reading
/// open.
#[test]
fn the_check_recomputes_every_differing_word_from_the_cases_inputs() {
    // in = [1.0, 0.5, 1e6, 2.0]; out[k] = fast atan(in[k]) for k in 0..3, and word 3 holds
    // two halves: low = half atan2(in16[0], in16[1]), high = the same of in16[2], in16[3].
    let input = [1.0f32, 0.5, 1e6, 2.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let halves = [0x3c00u16, 0x3c00, 0x8000, 0x3c00]
        .iter()
        .flat_map(|half| half.to_le_bytes())
        .collect::<Vec<_>>();
    let buffers = [buffer(1, &input), buffer(2, &halves)];
    let lanes = vec![
        BoundedLane {
            words: words(0, 3, 1),
            result: Part::F32,
            math: Math::Fast,
            computes: "atan(a)".into(),
            operands: vec![Operand::Buffer {
                binding: 1,
                format: Format::F32,
                first: 0,
                stride: 1,
            }],
        },
        BoundedLane {
            words: words(3, 1, 1),
            result: Part::F16Low,
            math: Math::Precise,
            computes: "atan2(a, b)".into(),
            operands: vec![
                Operand::Buffer {
                    binding: 2,
                    format: Format::F16,
                    first: 0,
                    stride: 1,
                },
                Operand::Buffer {
                    binding: 2,
                    format: Format::F16,
                    first: 1,
                    stride: 1,
                },
            ],
        },
    ];
    check_lanes(&lanes).unwrap();
    let atan = |x: f32| x.atan();
    let quarter_half = 0x3a48u32; // half(pi/4)
    let metal = [
        atan(1.0).to_bits(),
        atan(0.5).to_bits(),
        atan(1e6).to_bits(),
        quarter_half | 0x8000 << 16,
    ];
    let device = [
        // 1 value off: within 5 ulp.
        atan(1.0).to_bits() + 1,
        // 6 values off at 0.46: outside.
        atan(0.5).to_bits() + 6,
        // Bit-identical.
        atan(1e6).to_bits(),
        // The low half is Metal's; the high half (no lane) is +0 against -0.
        quarter_half,
    ];
    let bytes = |words: [u32; 4]| {
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let none = BTreeSet::new();
    let check = check(
        &lanes,
        &buffers,
        &none,
        lavapipe(),
        &bytes(metal),
        &bytes(device),
    );
    let verdicts = check
        .words
        .iter()
        .map(|word| (word.word, word.part, word.verdict.label()))
        .collect::<Vec<_>>();
    assert_eq!(
        verdicts,
        [
            (0, Part::F32, "within"),
            (1, Part::F32, "outside"),
            (3, Part::F16High, "uncovered"),
        ]
    );
    let outside = check.words[1].verdict.detail();
    assert!(
        outside.contains("ulp from") && outside.contains("<= 5 ulp"),
        "{outside}"
    );
    assert!(!check.clean(UnstatedWords::WithinPromise));

    // A word Apple states nothing for: within the promise under the user's rule, open under
    // the other setting of the flag.
    let lanes = vec![BoundedLane {
        words: words(0, 1, 1),
        result: Part::F32,
        math: Math::Fast,
        computes: "fmod(a, b)".into(),
        operands: vec![
            Operand::Buffer {
                binding: 1,
                format: Format::F32,
                first: 2,
                stride: 1,
            },
            Operand::Buffer {
                binding: 1,
                format: Format::F32,
                first: 0,
                stride: 1,
            },
        ],
    }];
    let check = super::check(
        &lanes,
        &buffers,
        &none,
        lavapipe(),
        &(-512.0f32).to_le_bytes(),
        &0.0f32.to_le_bytes(),
    );
    assert_eq!(check.count("unstated"), 1);
    assert!(check.clean(UnstatedWords::WithinPromise));
    assert!(!check.clean(UnstatedWords::Open));
    assert_eq!(UNSTATED_WORDS, UnstatedWords::WithinPromise);

    // The same word, where the row records the product's remainder as a near-equivalent of
    // Metal's (`DeviceArithmetic`): the difference is the product's approximation, so it stays
    // open under either setting of the flag. The device's transcendentals are not one.
    let arithmetic = BTreeSet::from([PrecisionNote::DeviceArithmetic]);
    let check = super::check(
        &lanes,
        &buffers,
        &arithmetic,
        lavapipe(),
        &(-512.0f32).to_le_bytes(),
        &0.0f32.to_le_bytes(),
    );
    assert_eq!(check.count("near_equivalent"), 1);
    assert!(check.words[0].verdict.detail().contains("DeviceArithmetic"));
    assert!(!check.clean(UnstatedWords::WithinPromise));
    let transcendentals = BTreeSet::from([PrecisionNote::DeviceTranscendentals]);
    let check = super::check(
        &lanes,
        &buffers,
        &transcendentals,
        lavapipe(),
        &(-512.0f32).to_le_bytes(),
        &0.0f32.to_le_bytes(),
    );
    assert_eq!(check.count("unstated"), 1);

    // An operand past the buffer cannot be checked.
    let mut past = lanes.clone();
    past[0].operands[0] = Operand::Buffer {
        binding: 1,
        format: Format::F32,
        first: 9,
        stride: 1,
    };
    let check = super::check(
        &past,
        &buffers,
        &none,
        lavapipe(),
        &(-512.0f32).to_le_bytes(),
        &0.0f32.to_le_bytes(),
    );
    assert_eq!(check.count("not_checkable"), 1);
    assert!(check.words[0].verdict.detail().contains("past buffer 1"));

    // A thread-derived operand: lane value k reads the half bit pattern 16k.
    let lanes = vec![BoundedLane {
        words: words(0, 2, 1),
        result: Part::F32,
        math: Math::Fast,
        computes: "atan(a)".into(),
        operands: vec![Operand::Bits {
            format: Format::F16,
            first: 0x3c00 - 16,
            stride: 16,
        }],
    }];
    let below_one = Format::F16.decode(0x3c00 - 16) as f32;
    let metal = [below_one.atan().to_bits(), atan(1.0).to_bits()];
    let device = [below_one.atan().to_bits() + 1, atan(1.0).to_bits() + 9];
    let check = super::check(
        &lanes,
        &[],
        &none,
        lavapipe(),
        &metal
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
        &device
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        check
            .words
            .iter()
            .map(|word| word.verdict.label())
            .collect::<Vec<_>>(),
        ["within", "outside"]
    );
}
