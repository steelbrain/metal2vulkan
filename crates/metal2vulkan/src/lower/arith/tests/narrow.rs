//! Interpreter differential tests of narrow floats held widened.
//!
//! Operands and results cross the kernel boundary as their 16-bit format bits (the parent's
//! `load`/`store` go through `narrow::from_bits`/`narrow::to_bits`), so every test also checks
//! the exact bit conversions. References compute on exact `f64` values and round once to the
//! narrow format (to nearest even), half denormals included (the rounding keeps them, as Metal
//! does, whichever signed zero the device's `OpQuantizeToF16` gives below the smallest normal
//! half).

use super::*;
use crate::spirv::ir::UnOp;

fn half() -> Form {
    float(16)
}

/// The value of narrow format bits `b`.
fn nval(f: Form, b: u64) -> f64 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => f16_to_f64(b as u16),
        Some(FloatFmt::BFloat) => f64::from(f32::from_bits((b as u32) << 16)),
        Some(FloatFmt::F32) => f64::from(f32v(b)),
        _ => f64v(b),
    }
}

/// `x` rounded to the nearest bfloat16 (ties to even), computed on the exact value: an
/// independent reference for the round-to-odd paths.
fn bf16_of(x: f64) -> u64 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let m = x.abs();
    if m == 0.0 {
        return sign;
    }
    if m.is_infinite() {
        return sign | 0x7F80;
    }
    // The unit in the last place at m's binade (8 significand bits; denormals below 2^-126).
    let e = m.log2().floor() as i32;
    let e = if 2f64.powi(e) > m { e - 1 } else { e };
    let e = if 2f64.powi(e + 1) <= m { e + 1 } else { e };
    let ulp = 2f64.powi(e.max(-126) - 7);
    let q = (m / ulp).round_ties_even() * ulp;
    let b = (q as f32).to_bits();
    if q >= 2f64.powi(128) {
        return sign | 0x7F80;
    }
    sign | u64::from(b >> 16)
}

/// `x` rounded to format `f`, as the format bits.
fn nbits(f: Form, x: f64) -> u64 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => u64::from(f64_to_f16(x)),
        Some(FloatFmt::BFloat) => bf16_of(x),
        Some(FloatFmt::F32) => u64::from((x as f32).to_bits()),
        _ => x.to_bits(),
    }
}

/// Equal bits, or both NaN.
fn exact_ok(out: Form) -> impl Fn(u64, u64) -> bool {
    move |g, w| g == w || (is_nan(out, g) && is_nan(out, w))
}

/// Interesting half values: zeros, denormals, the normal extremes, halfway points, integers,
/// infinities, a NaN, and negative non-integral values whose `x - floor(x)` needs more bits
/// than the format has (`0xB4CD` = -0.300048828125, `0xA3D7`, `0xC248`, `0x8001`).
fn half_values() -> Vec<u64> {
    vec![
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x3BFF, 0x3800,
        0x4000, 0x4100, 0xC100, 0x3555, 0x4B00, 0x6400, 0x7BFF, 0xFBFF, 0x7800, 0x0200, 0x1400,
        0x7C00, 0xFC00, 0x7E00, 0xB4CD, 0xA3D7, 0xC248, 0xBBFF,
    ]
}

/// Interesting bfloat values, with negative non-integral values whose `x - floor(x)` needs
/// more bits than the format has (`0xBE9B` = -0.302734375, `0xBDCD`, `0xBF7F`).
fn bf16_values() -> Vec<u64> {
    vec![
        0x0000, 0x8000, 0x0001, 0x0080, 0x3F80, 0xBF80, 0x3F81, 0x3FC0, 0x4049, 0xC049, 0x7F7F,
        0xFF7F, 0x0040, 0x4B80, 0x7F80, 0xFF80, 0x7FC0, 0x3DCC, 0xBE9B, 0xBDCD, 0xBF7F,
    ]
}

/// Every half and bfloat bit pattern round-trips through its widened `f32` bit for bit (denormals,
/// infinities, and NaN payloads and signs included) where the value is stored unchanged, and the
/// widened value is the pattern's.
///
/// The store of a loaded value reuses the bits it was decoded from, so the round trip runs twice:
/// once through that reuse, and once through a copy of the value, which has no noted bits, so
/// `to_bits` encodes it: there a half NaN is the device's widening's and narrowing's (here the
/// interpreter's canonical NaN, `PrecisionNote::NativeNanEncoding`), and every other half and
/// every bfloat pattern comes back exactly.
#[test]
fn narrow_bits_round_trip_every_pattern() {
    for f in [half(), bfloat()] {
        let cases: Vec<Case> = (0..=0xFFFFu64).map(|b| vec![vec![b]]).collect();
        check_accept(
            &format!("{f:?} round trip"),
            &[f],
            f,
            &|_, x| Ok(x[0]),
            &cases,
            &|c| Some(vec![arg(c, 0)]),
            &|g, w| g == w,
        );
        check_accept(
            &format!("{f:?} round trip through the encode"),
            &[f],
            f,
            &|fb, x| fb.unary(UnOp::CopyObject, x[0]),
            &cases,
            &|c| Some(vec![arg(c, 0)]),
            &|g, w| g == w || (f == half() && is_nan(f, g) && is_nan(f, w)),
        );
        check(
            &format!("{f:?} widened value"),
            &[f],
            float(32),
            &|_, x| Ok(x[0]),
            &cases,
            &|c| Some(vec![u64::from((nval(f, arg(c, 0)) as f32).to_bits())]),
            false,
        );
    }
}

/// One half round trip through its format bits, `to_bits` then `from_bits` (their types and
/// constants shared with the kernel's own load and store), is the device's narrowing and widening
/// alone: at most 40 words. A load also decodes
/// without masking the lane's low 16 bits first (the round trip test above feeds `from_bits` junk
/// above them).
#[test]
fn a_half_round_trip_through_its_bits_is_compact() {
    use crate::spirv::serialize::serialize;
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&[half()], half(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    // Copies keep the decoded bits from being reused, so both conversions run.
    let without = size(&|fb, x| fb.unary(UnOp::CopyObject, x[0]));
    let with = size(&|fb, x| {
        let c = fb.unary(UnOp::CopyObject, x[0])?;
        let b = super::super::narrow::to_bits(fb, half(), c)?;
        let y = super::super::narrow::from_bits(fb, half(), b)?;
        fb.unary(UnOp::CopyObject, y)
    });
    assert!(with - without <= 40, "{} words", with - without);
}

/// Under native half rounding a `half` may reach its store unrounded (the device folds the
/// conversion pair), and the store's bits are then the device's narrowing of it, in either form
/// (its conversion to 16 bits, `PackHalf2x16` without `shaderFloat16`): here the interpreter's, to
/// nearest even on the half grid, half denormals kept, a finite magnitude from `65520` up the
/// infinity of its sign, never a NaN's code or a code past 16 bits (whose bit 17 a pair's store
/// would carry into the next lane), and a NaN a NaN; a half value is its own bits. The narrowing is
/// the same instructions whether the module's rounding is native or not.
#[test]
fn an_unrounded_half_is_the_devices_narrowing_at_its_store() {
    use super::super::narrow::f32_of_half_bits;
    use crate::spirv::builder::QuantizeForm;
    let mut bits: Vec<u32> = vec![
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x4780_0000,
        0x4780_0001,
        0x477F_FFFF,
        0x477F_F000,
        0x477F_EFFF,
        0x477F_E000,
        0x47C3_5000,
        0x7149_F2CA,
        0xC780_0000,
        0x3F80_0001,
        0x3F80_1000,
        0x3F80_1FFF,
        0x3F80_3000,
        0x3649_539C,
        0x3380_0001,
        0x0000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x7F80_0001,
        0x7FC1_2345,
        0xFFFF_FFFF,
    ];
    bits.extend((0..=0xFFFFu32).step_by(7).map(f32_of_half_bits));
    let cases: Vec<Case> = bits.iter().map(|&b| vec![vec![u64::from(b)]]).collect();
    for q in [QuantizeForm::Converted, QuantizeForm::Device] {
        let words = |native: bool| {
            let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                fb.module().set_quantize_form(q);
                fb.module().set_native_half(native);
                Ok(x[0])
            };
            let m = kernel_with(&[float(32)], half(), &body, true, false).expect("build");
            crate::spirv::serialize::serialize(m)
                .expect("serialize")
                .words
                .len()
        };
        let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
            fb.module().set_quantize_form(q);
            fb.module().set_native_half(true);
            Ok(x[0])
        };
        check_accept(
            &format!("native {q:?}"),
            &[float(32)],
            half(),
            &body,
            &cases,
            &|c| Some(vec![metal_half(arg(c, 0))]),
            &|g, w| g == w || (is_nan(half(), g) && is_nan(half(), w)),
        );
        assert_eq!(
            words(false),
            words(true),
            "{q:?}: no cap under native rounding"
        );
    }
}

/// A `half`'s bits from its value, where none are known, are the device's narrowing on every form,
/// with no integer encode: wherever the module can hold a 16-bit float (`QuantizeForm::Converted`
/// and `Controlled`, a profile with `shaderFloat16`), one conversion of the lanes to 16 bits, read
/// as words beside a 16-bit zero, decorated `NoContraction` under the float controls; in
/// `QuantizeForm::Device` (no `shaderFloat16`), `GLSL.std.450 PackHalf2x16` of each lane beside a
/// zero (one a lane, and no 16-bit type). Every half value, on one to four lanes, encodes to its
/// own bits zero-extended but a NaN, which encodes to a NaN (`PrecisionNote::NativeNanEncoding`),
/// and `narrow::encodes_bits` finds the narrowing.
#[test]
fn a_half_encode_is_the_devices_narrowing_in_every_form() {
    use super::super::narrow::{encodes_bits, f32_of_half_bits, half_bits_of_f32, to_bits};
    use crate::spirv::builder::QuantizeForm;
    use crate::spirv::ir::Contraction;
    let lanes_of = |n: u32| -> Vec<Case> {
        (0..0x1_0000u32)
            .step_by(n as usize)
            .map(|h| {
                vec![(h..h + n)
                    .map(|x| u64::from(f32_of_half_bits(x & 0xFFFF)))
                    .collect()]
            })
            .collect()
    };
    let want = |c: &Case| {
        Some(
            c[0].iter()
                .map(|b| u64::from(half_bits_of_f32(*b as u32)))
                .collect(),
        )
    };
    let half_nan = |b: u64| b >> 16 == 0 && is_nan(half(), b);
    for q in [
        QuantizeForm::Device,
        QuantizeForm::Converted,
        QuantizeForm::Controlled,
    ] {
        let float16 = q != QuantizeForm::Device;
        for lanes in 1u8..=4 {
            let cases = lanes_of(u32::from(lanes));
            let cases = &cases;
            let what = format!("{lanes} lanes, {q:?}");
            let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                fb.module().set_quantize_form(q);
                to_bits(fb, vec(half(), lanes), x[0])
            };
            let (ins, out) = ([vec(float(32), lanes)], vec(int(32), lanes));
            let m = kernel_with(&ins, out, &body, true, false)
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            let budget = crate::budget::Budget::default();
            let mut meter = crate::budget::Meter::new(&budget);
            assert!(
                encodes_bits(&m, &mut meter).expect("encodes_bits"),
                "{what}: encodes_bits"
            );
            let insts = || m.blocks().iter().flat_map(|b| &b.insts);
            let integer_encode = insts().any(|n| {
                matches!(
                    n.inst,
                    Inst::Convert {
                        op: ConvOp::ConvertFToU,
                        ..
                    } | Inst::Select { .. }
                )
            });
            let conversions: Vec<Contraction> = insts()
                .filter_map(|n| match n.inst {
                    Inst::Convert {
                        op: ConvOp::FConvert,
                        contract,
                        ..
                    } => Some(contract),
                    _ => None,
                })
                .collect();
            let packs = insts()
                .filter(|n| {
                    matches!(
                        n.inst,
                        Inst::Glsl {
                            op: GlslOp::PackHalf2x16,
                            ..
                        }
                    )
                })
                .count();
            let halves = m
                .types()
                .iter()
                .any(|(_, t)| matches!(t, Type::Float { width: 16 }));
            let held = match q {
                QuantizeForm::Controlled => Contraction::Forbidden,
                _ => Contraction::Allowed,
            };
            assert_eq!(
                conversions,
                vec![held; usize::from(float16)],
                "{what}: the conversions"
            );
            let want_packs = if float16 { 0 } else { usize::from(lanes) };
            assert_eq!(packs, want_packs, "{what}: the packs");
            assert_eq!(halves, float16, "{what}: a 16-bit float type");
            assert!(!integer_encode, "{what}: an integer encode");
            check_accept(&what, &ins, out, &body, cases, &want, &|g, w| {
                g == w || (half_nan(g) && half_nan(w))
            });
        }
    }
}

/// The host's bits of a half value and value of half bits, which fold a constant's bits at
/// translation time, agree on every half pattern (the widened value is the pattern's, and back is
/// the pattern itself), and a constant that is no half value (a lost NaN payload, an `f32` between
/// two halves) is not folded.
#[test]
fn narrow_constant_bits_fold_exactly() {
    use super::super::narrow::{f32_of_half_bits, half_bits_of_f32};
    for h in 0..=0xFFFFu32 {
        let x = f32_of_half_bits(h);
        let want = nval(half(), u64::from(h)) as f32;
        assert!(
            x == want.to_bits() || (want.is_nan() && f32::from_bits(x).is_nan()),
            "{h:#06x}: {x:#010x}"
        );
        assert_eq!(half_bits_of_f32(x), h, "{h:#06x}");
    }
    for (bits, folds) in [
        (0x3F80_0000u32, true),
        (0xBF80_0000, true),
        (0x7F80_0000, true),
        (0x3380_0000, true),
        (0x3F80_1000, false),
        (0x7F80_0001, false),
        (0x0000_0001, false),
    ] {
        let x = f32_of_half_bits(half_bits_of_f32(bits));
        assert_eq!(x == bits, folds, "{bits:#010x}");
    }
    // A store of a constant takes its bits at translation time: the kernel's store of every
    // tested constant equals the pattern (the device's narrowing would compute the same but for a
    // NaN's bits).
    for f in [half(), bfloat()] {
        let vals = if f == half() {
            half_values()
        } else {
            bf16_values()
        };
        for v in vals {
            let k = nval(f, v);
            let cases: Vec<Case> = vec![vec![vec![0]]];
            check_accept(
                &format!("{f:?} constant {v:#06x}"),
                &[int(32)],
                f,
                &|fb, _| {
                    let ty = ops::ty(fb, float(32))?;
                    let c = fb.module().constant(crate::spirv::ir::Constant::Scalar {
                        ty,
                        bits: ConstBits(u64::from((k as f32).to_bits())),
                    })?;
                    fb.module().const_value(c)
                },
                &cases,
                &|_| Some(vec![v]),
                &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
            );
        }
    }
}

/// A store of a vector built by lane inserts repeats the inserts on the lanes' bits, and a stored
/// lane takes its bits from its vector's when those are known: every half and bfloat pattern, each
/// in one lane of a 4-lane vector built from `undef`, is stored exactly as it was loaded (the junk
/// the harness loads above each lane's 16 bits cleared); so is a vector that keeps two lanes of a
/// loaded vector and replaces the other two; so is a chain one lane of which has no known bits (a
/// copy of a loaded lane), which the device's narrowing encodes; so is each lane stored alone, with
/// and without known vector bits; and the vector's widened lanes are the patterns' values. Where
/// the narrowing encodes, a `half` NaN's bits are the device's
/// (`PrecisionNote::NativeNanEncoding`): so in a copy's chain, and in the loaded vector's, which
/// the harness builds from its lanes' decodes (a construct, whose bits are not noted).
#[test]
fn narrow_lanes_move_as_bits_through_vectors() {
    use super::super::narrow::note_insert;
    // Lane `copy` (if any) is a copy, which has no noted bits.
    let assemble = |fb: &mut FunctionBuilder<'_>,
                    f: Form,
                    x: &[Value],
                    copy: Option<usize>|
     -> BResult<Value> {
        let t = ops::ty(fb, vec(f, 4))?;
        let mut v = fb.undef(t)?;
        for (i, e) in x.iter().enumerate().take(4) {
            let e = if copy == Some(i) {
                fb.unary(UnOp::CopyObject, *e)?
            } else {
                *e
            };
            let y = fb.composite_insert(e, v, &[i as u32])?;
            note_insert(fb, y, v, e, i as u32)?;
            v = y;
        }
        Ok(v)
    };
    for f in [half(), bfloat()] {
        let cases: Vec<Case> = (0..0x4000u64)
            .map(|b| (0..4).map(|l| vec![(b + 0x4000 * l) & 0xFFFF]).collect())
            .collect();
        let ins = [f, f, f, f];
        let lanes = |c: &Case| Some((0..4).map(|i| arg(c, i)).collect());
        // Bits the device's narrowing computes: a `half` NaN is the device's.
        let sequenced = |g: u64, w: u64| g == w || (f == half() && is_nan(f, g) && is_nan(f, w));
        for copy in [None, Some(2)] {
            check_accept(
                &format!("{f:?} lanes through a vector (copied lane: {copy:?})"),
                &ins,
                vec(f, 4),
                &|fb, x| assemble(fb, f, x, copy),
                &cases,
                &lanes,
                &|g, w| match copy {
                    Some(_) => sequenced(g, w),
                    None => g == w,
                },
            );
        }
        check(
            &format!("{f:?} widened lanes of the vector"),
            &ins,
            vec(float(32), 4),
            &|fb, x| {
                let v = assemble(fb, f, x, None)?;
                fb.unary(UnOp::CopyObject, v)
            },
            &cases,
            &|c| {
                Some(
                    (0..4)
                        .map(|i| u64::from((nval(f, arg(c, i)) as f32).to_bits()))
                        .collect(),
                )
            },
            false,
        );
        // Lanes 0 and 2 of a loaded vector replaced by two loaded lanes: the kept lanes' junk
        // above their 16 bits is cleared.
        let mixed: Vec<Case> = (0..0x4000u64)
            .map(|b| {
                vec![
                    vec![b],
                    vec![b + 0x4000],
                    (0..4).map(|l| (b + 0x8000 + 0x1111 * l) & 0xFFFF).collect(),
                ]
            })
            .collect();
        check_accept(
            &format!("{f:?} lanes into a loaded vector"),
            &[f, f, vec(f, 4)],
            vec(f, 4),
            &|fb, x| {
                let y = fb.composite_insert(x[0], x[2], &[0])?;
                note_insert(fb, y, x[2], x[0], 0)?;
                let z = fb.composite_insert(x[1], y, &[2])?;
                note_insert(fb, z, y, x[1], 2)?;
                Ok(z)
            },
            &mixed,
            &|c| Some(vec![c[0][0], c[2][1], c[1][0], c[2][3]]),
            &sequenced,
        );
        // The first and the last lane (the other lanes extract the same way).
        for lane in [0u32, 3] {
            for copied in [false, true] {
                check_accept(
                    &format!("{f:?} lane {lane} stored from the vector (copied: {copied})"),
                    &ins,
                    f,
                    &|fb, x| {
                        let v = assemble(fb, f, x, None)?;
                        let v = if copied {
                            fb.unary(UnOp::CopyObject, v)?
                        } else {
                            v
                        };
                        let y = fb.composite_extract(v, &[lane])?;
                        fb.note_lane(y, v, lane);
                        Ok(y)
                    },
                    &cases,
                    &|c| Some(vec![arg(c, lane as usize)]),
                    &|g, w| match copied {
                        true => sequenced(g, w),
                        false => g == w,
                    },
                );
            }
        }
    }
}

/// Repeating a vector's lane inserts on bits never costs words. Four loaded halves stored as a
/// vector need neither their decode nor the vector's encode once unread instructions are removed; a
/// chain one lane of which has no known bits costs what it did without the notes (the device's
/// narrowing encodes the vector); and so does a lane stored from a vector whose bits are unknown
/// (the lane is encoded alone).
#[test]
fn narrow_lane_inserts_never_cost_words() {
    use super::super::narrow::note_insert;
    use crate::spirv::serialize::serialize;
    let size = |out: Form, body: Body<'_>| -> usize {
        let m = kernel_with(&[half(); 4], out, body, true, false).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let chain = |noted: bool, copy: Option<usize>| {
        move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
            let t = ops::ty(fb, vec(half(), 4))?;
            let mut v = fb.undef(t)?;
            for (i, e) in x.iter().enumerate() {
                let e = if copy == Some(i) {
                    fb.unary(UnOp::CopyObject, *e)?
                } else {
                    *e
                };
                let y = fb.composite_insert(e, v, &[i as u32])?;
                if noted {
                    note_insert(fb, y, v, e, i as u32)?;
                }
                v = y;
            }
            Ok(v)
        }
    };
    let v4 = vec(half(), 4);
    let (with, without) = (size(v4, &chain(true, None)), size(v4, &chain(false, None)));
    assert!(
        with + 64 < without,
        "{with} words with notes, {without} without"
    );
    let (with, without) = (
        size(v4, &chain(true, Some(1))),
        size(v4, &chain(false, Some(1))),
    );
    assert!(
        with <= without,
        "{with} words with notes, {without} without"
    );
    let lane = |noted: bool| {
        move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
            let v = chain(false, None)(fb, x)?;
            let v = fb.unary(UnOp::CopyObject, v)?;
            let y = fb.composite_extract(v, &[1])?;
            if noted {
                fb.note_lane(y, v, 1);
            }
            Ok(y)
        }
    };
    let (with, without) = (size(half(), &lane(true)), size(half(), &lane(false)));
    assert!(
        with <= without,
        "{with} words with notes, {without} without"
    );
}

/// A vector of narrow loads is decoded once when the module is smaller for it, and the lanes of a
/// vector stored one by one share one encode of the vector when that is smaller: every half and
/// bfloat pattern, each in one lane of a vector built by `narrow::insert`, widens to the `f32` bits
/// its own decode gives (`f32_of_half_bits` is checked against that decode in
/// `narrow_constant_bits_fold_exactly`), and every pattern, each in one lane of a vector of
/// unknown bits stored lane by lane, is stored exactly; the first is smaller than the lane
/// decodes, and the second costs no more than one vector encode and a copy per lane.
#[test]
fn narrow_vector_decodes_and_encodes_are_offered() {
    use super::super::narrow::insert;
    use crate::spirv::serialize::serialize;
    let words = |m: TypedModule| serialize(m).expect("serialize").words.len();
    for f in [half(), bfloat()] {
        let cases: Vec<Case> = (0..0x4000u64)
            .map(|b| (0..4).map(|l| vec![(b + 0x4000 * l) & 0xFFFF]).collect())
            .collect();
        let ins = [f, f, f, f];
        let built = |offered: bool| {
            move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let t = ops::ty(fb, vec(f, 4))?;
                let mut v = fb.undef(t)?;
                for (i, e) in x.iter().enumerate() {
                    v = if offered {
                        insert(fb, v, *e, i as u32)?
                    } else {
                        fb.composite_insert(*e, v, &[i as u32])?
                    };
                }
                Ok(v)
            }
        };
        let out = vec(float(32), 4);
        let m = kernel_with(&ins, out, &built(true), true, false).expect("build");
        let plain = kernel_with(&ins, out, &built(false), true, false).expect("build");
        let (with, without) = (words(m.clone()), words(plain));
        // A bfloat decode is one shift, a half decode the device's widening of a few.
        assert!(with < without, "{f:?}: {with} words offered, {without} not");
        // Bit for bit what each lane's own decode gives: a bfloat's every pattern (a signaling NaN
        // stays signaling), a half's but a NaN, which is the device's widening's NaN.
        let widened = |h: u64| {
            let h = h as u32;
            u64::from(if f == half() {
                super::super::narrow::f32_of_half_bits(h)
            } else {
                h << 16
            })
        };
        let nan32 = |b: u64| f32::from_bits(b as u32).is_nan();
        check_module(
            &format!("{f:?} lanes decoded once"),
            m,
            &ins,
            out,
            &cases,
            &|c| Some((0..4).map(|i| widened(arg(c, i))).collect()),
            &|g, w| g == w || (f == half() && nan32(g) && nan32(w)),
        );
        // A copy of a loaded vector has no known bits; its lanes are stored one by one.
        let v4 = vec(f, 4);
        let cases: Vec<Case> = (0..0x4000u64)
            .map(|b| vec![(0..4).map(|l| (b + 0x4000 * l) & 0xFFFF).collect()])
            .collect();
        let copy = |fb: &mut FunctionBuilder<'_>, x: &[Value]| fb.unary(UnOp::CopyObject, x[0]);
        let lanes = kernel_with(&[v4], v4, &copy, true, true).expect("build");
        let whole = kernel_with(&[v4], v4, &copy, true, false).expect("build");
        let (by_lane, at_once) = (words(lanes.clone()), words(whole));
        assert!(
            by_lane <= at_once + 16,
            "{f:?}: {by_lane} words by lane, {at_once} at once"
        );
        check_module(
            &format!("{f:?} lanes stored from one encode"),
            lanes,
            &[v4],
            v4,
            &cases,
            &|c| Some(c[0].clone()),
            // A copy's half NaN is the device's widening's (`NativeNanEncoding`).
            &|g, w| g == w || (f == half() && is_nan(f, g) && is_nan(f, w)),
        );
    }
}

/// The census names half and bfloat arithmetic apart, so the output-size gate's exception for exact
/// half rounding (`arith.float.half`) excuses no bfloat word; a `float` operator stays
/// `arith.float`.
#[test]
fn narrow_float_operators_are_named_by_format() {
    use crate::lower::select::FamilyPlans;
    use crate::lower::strategy::FamilyStrategy;
    let name = |form| {
        super::super::ArithStrategy::Float {
            op: AirBin::FAdd,
            form,
            contract: Contraction::Forbidden,
        }
        .name(&FamilyPlans::default())
    };
    assert_eq!(name(half()), "arith.float.half");
    assert_eq!(name(bfloat()), "arith.float.bfloat");
    assert_eq!(name(float(32)), "arith.float");
}

/// A bitcast from or to narrow float vectors moves narrow patterns bit for bit (NaN payloads
/// included, but for a `half` NaN whose loaded bits are not reused: its bits are the device's
/// widening's, `PrecisionNote::NativeNanEncoding`) under each plan: through one word (`Packed`),
/// through words (`Words`), and lane by lane (`Lanes`), where a narrow vector converts to and from
/// its format bits once as a vector. The word plan between `<4 x half>` and `<2 x i32>` moves every
/// half pattern; the others move the interesting values and pseudo-random patterns.
#[test]
fn narrow_vector_bitcasts_move_narrow_patterns() {
    let mut rng = Rng(11);
    for (f, vals) in [(half(), half_values()), (bfloat(), bf16_values())] {
        let mut some = vals.clone();
        some.extend((0..128).map(|_| rng.next() & 0xFFFF));
        for (from, to) in [
            (vec(f, 2), int(32)),
            (int(32), vec(f, 2)),
            (vec(f, 4), vec(int(32), 2)),
            (vec(int(32), 2), vec(f, 4)),
            (vec(f, 4), vec(int(16), 4)),
            (vec(f, 3), vec(int(16), 3)),
            (vec(int(16), 3), vec(f, 3)),
        ] {
            let plan = ops::bitcast_plan(from, to).expect("a plan");
            let every = f == half() && [from, to].contains(&vec(int(32), 2));
            let patterns: Vec<u64> = if every {
                (0..0x10000).collect()
            } else {
                some.clone()
            };
            // The patterns in order, low lanes first.
            let per = (u32::from(from.bits()) * u32::from(from.lanes) / 16) as usize;
            let cases: Vec<Case> = patterns
                .chunks(per)
                .map(|chunk| {
                    let all: u128 = (0..per)
                        .map(|j| u128::from(chunk.get(j).copied().unwrap_or(0)) << (16 * j))
                        .fold(0, |a, b| a | b);
                    vec![(0..from.lanes)
                        .map(|i| {
                            (all >> (u32::from(i) * u32::from(from.bits()))) as u64
                                & mask(from.bits())
                        })
                        .collect()]
                })
                .collect();
            check_accept(
                &format!("bitcast {from:?} to {to:?} ({plan:?})"),
                &[from],
                to,
                &|fb, x| ops::bitcast(fb, plan, from, to, x[0]),
                &cases,
                &|c| {
                    let mut all: u128 = 0;
                    for (i, l) in c[0].iter().enumerate() {
                        all |= u128::from(*l) << (i as u32 * u32::from(from.bits()));
                    }
                    Some(
                        (0..to.lanes)
                            .map(|i| {
                                (all >> (u32::from(i) * u32::from(to.bits()))) as u64
                                    & mask(to.bits())
                            })
                            .collect(),
                    )
                },
                &|g, w| {
                    // Each 16-bit pattern alike, or a `half` NaN for a `half` NaN.
                    (0..4).all(|k| {
                        let (g, w) = ((g >> (16 * k)) & 0xFFFF, (w >> (16 * k)) & 0xFFFF);
                        g == w || (f == half() && is_nan(f, g) && is_nan(f, w))
                    })
                },
            );
        }
    }
}

/// `+ - * / %` on widened narrow floats round once to the format (double rounding through `f32` is
/// innocuous for these), on scalars and vectors.
#[test]
fn narrow_float_operators_round_once() {
    for (f, vals) in [(half(), half_values()), (bfloat(), bf16_values())] {
        for op in [
            AirBin::FAdd,
            AirBin::FSub,
            AirBin::FMul,
            AirBin::FDiv,
            AirBin::FRem,
        ] {
            let accept = exact_ok(f);
            check_accept(
                &format!("{op:?} {f:?}"),
                &[f, f],
                f,
                &|fb, x| ops::float_binary(fb, op, f, Contraction::Forbidden, x[0], x[1]),
                &pairs(&vals, &vals),
                &|c| {
                    let (a, b) = (nval(f, arg(c, 0)), nval(f, arg(c, 1)));
                    Some(vec![match op {
                        AirBin::FAdd => nbits(f, a + b),
                        AirBin::FSub => nbits(f, a - b),
                        AirBin::FMul => nbits(f, a * b),
                        AirBin::FDiv => nbits(f, a / b),
                        _ => nbits(f, a % b),
                    }])
                },
                &accept,
            );
        }
    }
    let v = vec(half(), 3);
    let vals = half_values();
    let cases: Vec<Case> = (0..vals.len())
        .map(|i| {
            let l = |k: usize| vals[(i + k) % vals.len()];
            vec![vec![l(0), l(1), l(2)], vec![l(3), l(5), l(7)]]
        })
        .collect();
    let accept = exact_ok(v);
    check_accept(
        "fmul <3 x half>",
        &[v, v],
        v,
        &|fb, x| ops::float_binary(fb, AirBin::FMul, v, Contraction::Forbidden, x[0], x[1]),
        &cases,
        &|c| {
            Some(
                (0..3)
                    .map(|i| nbits(v, nval(v, c[0][i]) * nval(v, c[1][i])))
                    .collect(),
            )
        },
        &accept,
    );
}

/// The float functions on half and bfloat values: closed functions return values of the format,
/// `fract` rounds its `f32` difference once (it is not closed: a negative non-integral input needs
/// more bits), `mix` and `llvm.fmuladd` round per step (the unfused choice), `sqrt` rounds once.
#[test]
fn narrow_float_functions_match_their_definitions() {
    for (f, vals) in [(half(), half_values()), (bfloat(), bf16_values())] {
        narrow_float_functions_on(f, &vals);
    }
}

fn narrow_float_functions_on(f: Form, vals: &[u64]) {
    let is_half = f.float_fmt() == Some(FloatFmt::Half);
    let few: Vec<u64> = vals.iter().copied().step_by(3).collect();
    for op in [
        FloatFn::Abs,
        FloatFn::Floor,
        FloatFn::Ceil,
        FloatFn::Trunc,
        FloatFn::Round,
        FloatFn::Rint,
        FloatFn::Fract,
        FloatFn::Sign,
        FloatFn::Saturate,
        FloatFn::Min,
        FloatFn::Max,
        FloatFn::Clamp,
        FloatFn::Min3,
        FloatFn::Max3,
        FloatFn::Median3,
        FloatFn::Mix,
        FloatFn::MulAdd,
        FloatFn::Sqrt,
    ] {
        let arity = op.arity();
        let cases: Vec<Case> = if arity == 1 {
            vals.iter().map(|v| vec![vec![*v]]).collect()
        } else {
            let mut c = Vec::new();
            for (i, p) in few.iter().enumerate() {
                for (j, q) in few.iter().enumerate() {
                    let t = few[(i + 2 * j) % few.len()];
                    c.push(
                        vec![vec![*p], vec![*q], vec![t]]
                            .into_iter()
                            .take(arity)
                            .collect(),
                    );
                }
            }
            c
        };
        // The device's `NMin`, `NMax` and `NClamp`, which may give either zero.
        let loose = matches!(
            op,
            FloatFn::Min
                | FloatFn::Max
                | FloatFn::Clamp
                | FloatFn::Min3
                | FloatFn::Max3
                | FloatFn::Saturate
        );
        let flush = exact_ok(f);
        let accept = |g: u64, w: u64| flush(g, w) || (loose && is_zero(f, g) && is_zero(f, w));
        check_accept(
            &format!("{op:?} {f:?}"),
            &vec![f; arity],
            f,
            &|fb, x| math::float_fn(fb, op, f, Contraction::Forbidden, x, &Lib::instructions()),
            &cases,
            &|c| {
                let x = |i: usize| nval(f, c[i][0]);
                // An intermediate step's rounding.
                let q = |v: f64| nval(f, nbits(f, v));
                let r = match op {
                    FloatFn::Abs => x(0).abs(),
                    FloatFn::Floor => x(0).floor(),
                    FloatFn::Ceil => x(0).ceil(),
                    FloatFn::Trunc => x(0).trunc(),
                    FloatFn::Round => c99_round(x(0)),
                    FloatFn::Rint => x(0).round_ties_even(),
                    // The device's `Fract`: a NaN at a NaN and an infinity.
                    FloatFn::Fract if !x(0).is_finite() => f64::NAN,
                    FloatFn::Fract => {
                        let below = 1.0 - 2f64.powi(if is_half { -11 } else { -8 });
                        q(x(0) - x(0).floor()).min(below)
                    }
                    // The device's `FSign`, undefined at a NaN; a zero keeps its sign.
                    FloatFn::Sign if x(0).is_nan() => return None,
                    FloatFn::Sign if x(0) > 0.0 => 1.0,
                    FloatFn::Sign if x(0) < 0.0 => -1.0,
                    FloatFn::Sign => x(0),
                    FloatFn::Saturate if x(0) > 0.0 => x(0).min(1.0),
                    FloatFn::Saturate => 0.0,
                    FloatFn::Min => x(0).min(x(1)),
                    FloatFn::Max => x(0).max(x(1)),
                    FloatFn::Clamp => x(0).max(x(1)).min(x(2)),
                    FloatFn::Min3 => x(0).min(x(1)).min(x(2)),
                    FloatFn::Max3 => x(0).max(x(1)).max(x(2)),
                    FloatFn::Median3 => {
                        let b = [c[0][0], c[1][0], c[2][0]];
                        return Some(vec![median3_ref(b, &|v| nval(f, v))]);
                    }
                    // A `half` one the device's `FMix` in `float`, rounded once.
                    FloatFn::Mix if is_half => {
                        let (p, y, t) = (x(0) as f32, x(1) as f32, x(2) as f32);
                        f64::from(p * (1.0 - t) + y * t)
                    }
                    FloatFn::Mix => q(x(0) * q(1.0 - x(2))) + q(x(1) * x(2)),
                    FloatFn::MulAdd => q(q(x(0) * x(1)) + x(2)),
                    _ => x(0).sqrt(),
                };
                Some(vec![nbits(f, r)])
            },
            &accept,
        );
    }
}

/// The `f32` operands [`converted_half_rounding_is_the_exactly_rounded_half`] rounds to `half`:
/// every 17th half pattern and every denormal one, each exactly and at its exact ties with both
/// neighbours, just inside and outside them, and the `f32` edge values (the ties at the smallest
/// normal half, `65504..65520`, signed zeros, `f32` denormals, NaN).
fn half_rounding_operands() -> Vec<f32> {
    let mut xs: Vec<f32> = Vec::new();
    for h in (0..0x7C00u16).step_by(17).chain(0..0x400) {
        let v = f16_to_f64(h);
        let ulp = if h < 0x400 {
            2f64.powi(-24)
        } else {
            2f64.powi(((h >> 10) as i32) - 25)
        };
        for d in [0.0, 0.5, -0.5, 0.25, -0.25, 0.75] {
            for s in [1.0, -1.0] {
                let x = (s * (v + d * ulp)) as f32;
                xs.push(x);
                // Just inside and outside the ties.
                xs.push(f32::from_bits(x.to_bits().wrapping_add(1)));
                xs.push(f32::from_bits(x.to_bits().wrapping_sub(1)));
            }
        }
    }
    xs.extend(
        [
            6.103_515_6e-5,
            6.100_463_9e-5,
            6.1004_64e-5,
            65504.0,
            65519.996,
            65520.0,
            1e10,
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::from_bits(0x007F_FFFF),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            2f32.powi(-25),
            -(2f32.powi(-25)),
            3.0 * 2f32.powi(-25),
            f32::from_bits(0x3300_0001),
        ]
        .iter(),
    );
    xs
}

/// In `QuantizeForm::Converted` and `Controlled` a `half` rounding is the conversion to 16 bits
/// and its widening alone, which the interpreter rounds to nearest even with half denormals kept,
/// as the Vulkan spec makes it under the controlled form's float controls: over
/// [`half_rounding_operands`], `round` gives the exactly rounded half on scalars and vectors, with
/// no `OpQuantizeToF16` and no `RoundEven` in the module. In `Controlled` each conversion is
/// decorated `NoContraction`, and none in `Converted`. Metal's readings of A1's
/// `half-denormal-results-of-arithmetic-and-conversions` (dd2cb644) and
/// `half-tanh-under-fast-math-at-every-sixteenth-pattern` (27353187) keep every half denormal a
/// result rounds to, as this rounding does.
#[test]
fn converted_half_rounding_is_the_exactly_rounded_half() {
    let xs = half_rounding_operands();
    let (f32f, h) = (float(32), half());
    let v = vec(h, 3);
    let cases: Vec<Case> = xs
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits())]])
        .collect();
    let vc: Vec<Case> = xs
        .chunks(3)
        .filter(|c| c.len() == 3)
        .map(|c| vec![c.iter().map(|x| u64::from(x.to_bits())).collect()])
        .collect();
    let want = |c: &Case| {
        Some(
            c[0].iter()
                .map(|b| u64::from(f64_to_f16(f64::from(f32v(*b)))))
                .collect::<Vec<u64>>(),
        )
    };
    use crate::spirv::builder::QuantizeForm;
    use crate::spirv::ir::Contraction;
    let name = "round";
    for q in [QuantizeForm::Converted, QuantizeForm::Controlled] {
        for (form, input, cs) in [(h, f32f, &cases), (v, vec(f32f, 3), &vc)] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
                fb.module().set_quantize_form(q);
                crate::lower::arith::narrow::round(fb, form, x[0])
            };
            let m = kernel_with(&[input], form, &body, true, false)
                .unwrap_or_else(|e| panic!("{name} {q:?}: build: {e:?}"));
            let conversions: Vec<Contraction> = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|n| match n.inst {
                    Inst::Convert {
                        op: crate::spirv::ir::ConvOp::FConvert,
                        contract,
                        ..
                    } => Some(contract),
                    _ => None,
                })
                .collect();
            let held = match q {
                QuantizeForm::Controlled => Contraction::Forbidden,
                _ => Contraction::Allowed,
            };
            assert!(
                conversions.len() >= 2 && conversions.iter().all(|c| *c == held),
                "{name} {q:?}: {conversions:?}"
            );
            let quantizes = crate::lower::arith::narrow::quantizes(&m);
            let rounds_even = m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
                matches!(
                    n.inst,
                    Inst::Glsl {
                        op: GlslOp::RoundEven,
                        ..
                    }
                )
            });
            assert!(
                !quantizes && !rounds_even,
                "{name}: a quantize or RoundEven"
            );
            check_module(
                &format!("{q:?} {name} on {form:?}"),
                m,
                &[input],
                form,
                cs,
                &want,
                &exact_ok(form),
            );
        }
    }
}

/// Metal's conversion of the `f32` of bits `b` to `half` (on every `f32`, and the M3 Max's probe of
/// every NaN): to nearest even on the half grid, half denormals and the sign of a zero kept,
/// infinity from 65520 up, and `0x7E00` for every NaN.
fn metal_half(b: u64) -> u64 {
    let x = f32v(b);
    if x.is_nan() {
        0x7E00
    } else {
        u64::from(f64_to_f16(f64::from(x)))
    }
}

/// Operands for [`metal_half`]: finite half patterns of both signs (every denormal; in every
/// binade the smallest, largest, middle and their neighbours' significands; and every 13th
/// pattern), exactly, at the ties with its upper neighbour, and one `f32` ulp either side of
/// those ties; every `f32` near 65504, near the tie 65520, and near `2^16`; each `f32`
/// exponent's smallest, middle and largest significand; `f32` denormals; the infinities; and
/// NaNs of both signs, quiet and signalling, with their payload in the bits a half keeps, in the
/// 13 it drops, or in both. (Every half pattern would be 8 times as many; the interpreter makes
/// that minutes.)
fn metal_half_operands() -> Vec<u32> {
    let mut xs = Vec::new();
    let picked = |h: u16| {
        h < 0x400
            || h.is_multiple_of(13)
            || matches!(
                h & 0x3FF,
                0 | 1 | 2 | 0x1FF | 0x200 | 0x201 | 0x3FD | 0x3FE | 0x3FF
            )
    };
    for h in (0..0x7C00u16).filter(|&h| picked(h)) {
        let v = f16_to_f64(h);
        let ulp = if h < 0x400 {
            2f64.powi(-24)
        } else {
            2f64.powi(((h >> 10) as i32) - 25)
        };
        for s in [1.0, -1.0] {
            let x = (s * v) as f32;
            xs.push(x.to_bits());
            let tie = (s * (v + 0.5 * ulp)) as f32;
            xs.extend([tie.to_bits(), tie.to_bits() + 1, tie.to_bits() - 1]);
        }
    }
    for edge in [0x477F_E000u32, 0x477F_F000, 0x4780_0000] {
        for b in edge - 0x40..=edge + 0x40 {
            xs.extend([b, b | 0x8000_0000]);
        }
    }
    for e in 0..=0xFEu32 {
        for m in [0, 1, 0x40_0000, 0x7F_E000, 0x7F_F000, 0x7F_FFFF] {
            let b = (e << 23) | m;
            xs.extend([b, b | 0x8000_0000]);
        }
    }
    for m in [
        1, 0x1FFF, 0x2000, 0x3F_FFFF, 0x40_0000, 0x40_0001, 0x40_2000, 0x7F_E000, 0x7F_FFFF,
    ] {
        xs.extend([0x7F80_0000 | m, 0xFF80_0000 | m]);
    }
    xs.extend([0x7F80_0000, 0xFF80_0000]);
    xs
}

/// In `QuantizeForm::Device`, the default, a `half` rounding is the device's `OpQuantizeToF16`
/// alone, with no denormal arm (no `RoundEven`, no test of the quantize for a zero): through the
/// interpreter's quantize (to
/// nearest even, flushing what is too small for a normal half to the zero of its sign), `round`
/// over [`metal_half_operands`] (signed zeros included, a NaN a NaN), on scalars and vectors, is
/// Metal's conversion of a `float` to `half` ([`metal_half`]) from the smallest normal half up and
/// the quantize's zero of the operand's sign below it, where Metal answers the half denormal.
#[test]
fn device_half_rounding_is_the_quantize_alone() {
    use crate::spirv::builder::QuantizeForm;
    let xs = metal_half_operands();
    let (f32f, h) = (float(32), half());
    let v = vec(h, 3);
    // The quantize alone: Metal's conversion where it is a normal half, an infinity or a NaN, and
    // the zero of the operand's sign below the smallest normal half.
    let flushed = |b: u64| match metal_half(b) {
        h if h & 0x7C00 == 0 => h & 0x8000,
        h => h,
    };
    let want = |c: &Case| Some(c[0].iter().map(|b| flushed(*b)).collect::<Vec<u64>>());
    // The quantize is the interpreter's: a quarter of the operands shows the module around it.
    let ops: Vec<u32> = xs.iter().copied().step_by(4).collect();
    let cases: Vec<Case> = ops.iter().map(|b| vec![vec![u64::from(*b)]]).collect();
    let vc: Vec<Case> = ops
        .chunks(3)
        .step_by(5)
        .filter(|c| c.len() == 3)
        .map(|c| vec![c.iter().map(|b| u64::from(*b)).collect()])
        .collect();
    for (lanes, input, cs) in [(h, f32f, &cases), (v, vec(f32f, 3), &vc)] {
        let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
            fb.module().set_quantize_form(QuantizeForm::Device);
            crate::lower::arith::narrow::round(fb, lanes, x[0])
        };
        let what = format!("device round on {lanes:?}");
        let m = kernel_with(&[input], lanes, &body, true, false)
            .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
        assert!(
            crate::lower::arith::narrow::quantizes(&m),
            "{what}: OpQuantizeToF16"
        );
        let arm = m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
            matches!(
                n.inst,
                Inst::Glsl {
                    op: GlslOp::RoundEven,
                    ..
                } | Inst::Binary {
                    op: BinOp::FOrdEqual,
                    ..
                }
            )
        });
        assert!(!arm, "{what}: a denormal arm");
        // The interpreter's quantize keeps a NaN a NaN, its bits the device's.
        check_module(&what, m, &[input], lanes, cs, &want, &exact_ok(lanes));
    }
}

/// A `half` sum or difference takes the device's rounding (`narrow::round`): over
/// every eleventh half pattern below `2^-13` and its negation, the patterns around the smallest
/// normal half, and normal, infinite, and NaN operands, on scalars and vectors, every sum and
/// difference is the exactly rounded half where that is a normal half, a zero, an infinity or a
/// NaN; below the smallest normal half it is the device's: the interpreter's quantize's zero of its
/// sign in `QuantizeForm::Device`, and the half denormal its conversion keeps in
/// `QuantizeForm::Converted`. The addends include the ties at the half's last bit (`2048 + 1`
/// rounds to 2048 and `2050 + 1` to 2052) and the overflow threshold (`65504 + 8` stays 65504,
/// `65504 + 16` is the tie at 65520 and rounds to the infinity).
#[test]
fn half_sums_take_the_devices_rounding_below_the_smallest_normal() {
    use crate::spirv::builder::QuantizeForm;
    let mut vals: Vec<u64> = (0..0x0800u64).step_by(11).collect();
    vals.extend(0x03F0..0x0410u64);
    vals.extend([
        0x3C00, 0x3555, 0x7BFF, 0x7BFE, 0x7C00, 0x7E00, 0x1400, 0x0C01, 0x6800, 0x6801,
    ]);
    let signed: Vec<u64> = vals.iter().flat_map(|v| [*v, *v | 0x8000]).collect();
    let mut few: Vec<u64> = signed.iter().copied().step_by(9).collect();
    few.extend([0x3C00, 0x4800, 0x4C00, 0x0001, 0x8001, 0xBC00]);
    let h = half();
    let v = vec(h, 4);
    let cases: Vec<Case> = (0..signed.len())
        .map(|i| {
            let l = |k: usize| signed[(i + k) % signed.len()];
            vec![vec![l(0), l(1), l(2), l(3)], vec![l(7), l(5), l(3), l(1)]]
        })
        .collect();
    for q in [QuantizeForm::Device, QuantizeForm::Converted] {
        // The exactly rounded half, or below the smallest normal half the zero of its sign where
        // the interpreter's quantize flushes it.
        let rounded = move |r: f64| {
            let b = nbits(h, r);
            match q {
                QuantizeForm::Device if b & 0x7C00 == 0 => b & 0x8000,
                _ => b,
            }
        };
        for op in [AirBin::FAdd, AirBin::FSub] {
            check_accept(
                &format!("{op:?} below the smallest normal, {q:?}"),
                &[h, h],
                h,
                &|fb, x| {
                    fb.module().set_quantize_form(q);
                    ops::float_binary(fb, op, h, Contraction::Forbidden, x[0], x[1])
                },
                &pairs(&signed, &few),
                &|c| {
                    let (a, b) = (nval(h, arg(c, 0)), nval(h, arg(c, 1)));
                    let r = if op == AirBin::FAdd { a + b } else { a - b };
                    Some(vec![rounded(r)])
                },
                &exact_ok(h),
            );
        }
        check_accept(
            &format!("fsub <4 x half> below the smallest normal, {q:?}"),
            &[v, v],
            v,
            &|fb, x| {
                fb.module().set_quantize_form(q);
                ops::float_binary(fb, AirBin::FSub, v, Contraction::Forbidden, x[0], x[1])
            },
            &cases,
            &|c| {
                Some(
                    (0..4)
                        .map(|i| rounded(nval(v, c[0][i]) - nval(v, c[1][i])))
                        .collect(),
                )
            },
            &exact_ok(v),
        );
    }
}

/// Conversions into and out of the narrow formats round once, including from `double` through a
/// round-to-odd `f32` (values chosen where plain double rounding is wrong) and from integers into
/// `bfloat`.
#[test]
fn narrow_conversions_round_once() {
    let (f32f, f64f) = (float(32), float(64));
    // f32 -> narrow, narrow -> f32/f64.
    let f32s: Vec<Case> = f32_values()
        .into_iter()
        .chain(
            [
                1.0f32 + 2f32.powi(-11),
                1.0 + 2f32.powi(-11) + 2f32.powi(-23),
                65519.0,
                65520.0,
                6.0e-8,
                3.0e-8,
                1.0 + 2f32.powi(-8) + 2f32.powi(-20),
            ]
            .iter()
            .map(|x| u64::from(x.to_bits())),
        )
        .map(|v| vec![vec![v]])
        .collect();
    let f64s: Vec<Case> = f64_values()
        .into_iter()
        .chain(
            [
                1.0f64 + 2f64.powi(-11) + 2f64.powi(-40),
                1.0 + 2f64.powi(-11) - 2f64.powi(-40),
                1.0 + 2f64.powi(-8) + 2f64.powi(-40),
                -(1.0 + 2f64.powi(-8) + 2f64.powi(-45)),
                65519.99999,
                1e300,
                3.0e-8,
            ]
            .iter()
            .map(|x| x.to_bits()),
        )
        .map(|v| vec![vec![v]])
        .collect();
    for n in [half(), bfloat()] {
        for (from, cases) in [(f32f, &f32s), (f64f, &f64s)] {
            let accept = exact_ok(n);
            check_accept(
                &format!("fptrunc {from:?} to {n:?}"),
                &[from],
                n,
                &|fb, x| ops::cast(fb, CastKind::FpResize, from, n, x[0]),
                cases,
                &|c| Some(vec![nbits(n, nval(from, arg(c, 0)))]),
                &accept,
            );
        }
        let ncases: Vec<Case> = (0..=0xFFFFu64).step_by(7).map(|b| vec![vec![b]]).collect();
        for to in [f32f, f64f] {
            check(
                &format!("fpext {n:?} to {to:?}"),
                &[n],
                to,
                &|fb, x| ops::cast(fb, CastKind::FpResize, n, to, x[0]),
                &ncases,
                &|c| Some(vec![nbits(to, nval(n, arg(c, 0)))]),
                false,
            );
        }
    }
    // half <-> bfloat.
    for (from, to) in [(half(), bfloat()), (bfloat(), half())] {
        let cases: Vec<Case> = (0..=0xFFFFu64).step_by(5).map(|b| vec![vec![b]]).collect();
        let accept = exact_ok(to);
        check_accept(
            &format!("convert {from:?} to {to:?}"),
            &[from],
            to,
            &|fb, x| math::convert(fb, ConvertKind::Float, from, to, x[0], NanTest::IsNan),
            &cases,
            &|c| Some(vec![nbits(to, nval(from, arg(c, 0)))]),
            &accept,
        );
    }
    // Integers -> narrow (round once) and narrow -> integers (exact, saturating).
    let mut rng = Rng(21);
    for ibits in [8u8, 16, 32] {
        let fi = int(ibits);
        let mut ivals = int_values(ibits, &mut rng, 12);
        if ibits == 32 {
            ivals.extend([
                0x0100_0001,
                0x0101_8000,
                0x0100_8000,
                0x7FFF_FFFF,
                0xFF80_8001,
            ]);
        }
        let cases: Vec<Case> = ivals.iter().map(|v| vec![vec![*v]]).collect();
        for n in [half(), bfloat()] {
            for (k, signed) in [(CastKind::UiToFp, false), (CastKind::SiToFp, true)] {
                check(
                    &format!("{k:?} i{ibits} to {n:?}"),
                    &[fi],
                    n,
                    &|fb, x| ops::cast(fb, k, fi, n, x[0]),
                    &cases,
                    &|c| {
                        let v = arg(c, 0);
                        let x = if signed {
                            sext(v, ibits) as f64
                        } else {
                            v as f64
                        };
                        Some(vec![nbits(n, x)])
                    },
                    false,
                );
            }
        }
        let hcases: Vec<Case> = half_values().iter().map(|v| vec![vec![*v]]).collect();
        for signed in [true, false] {
            check(
                &format!("convert half to {}{ibits}", if signed { "s" } else { "u" }),
                &[half()],
                fi,
                &|fb, x| {
                    math::convert(
                        fb,
                        ConvertKind::ToInt { signed },
                        half(),
                        fi,
                        x[0],
                        NanTest::IsNan,
                    )
                },
                &hcases,
                &|c| {
                    let x = nval(half(), arg(c, 0));
                    let v: i128 = if x.is_nan() {
                        0
                    } else if signed {
                        x.trunc().clamp(smin(ibits) as f64, smax(ibits) as f64) as i128
                    } else {
                        x.trunc().clamp(0.0, umax(ibits) as f64) as i128
                    };
                    Some(vec![(v as u64) & mask(ibits)])
                },
                false,
            );
        }
    }
}

/// The bfloat16 bits of the exact value `(-1)^neg * mag * 2^exp`, rounded to nearest even
/// (denormals below `2^-126`, infinity past the largest finite value).
fn bf16_exact(neg: bool, mag: u128, exp: i32) -> u64 {
    let sign = if neg { 0x8000 } else { 0 };
    if mag == 0 {
        return sign;
    }
    let msb = 127 - mag.leading_zeros() as i32;
    // The unit in the last place: 8 significand bits, never below the denormal unit 2^-133.
    let ulp = (msb + exp - 7).max(-133);
    let shift = ulp - exp;
    let q = if shift <= 0 {
        mag << (-shift)
    } else if shift > 120 {
        // mag < 2^117 is below half a unit.
        0
    } else {
        let (q, rem) = (mag >> shift, mag & ((1u128 << shift) - 1));
        let half = 1u128 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    let v = q as f64 * 2f64.powi(ulp);
    if v >= 2f64.powi(128) {
        return sign | 0x7F80;
    }
    sign | u64::from((v as f32).to_bits() >> 16)
}

/// A finite bfloat's `(negative, significand, exponent)`: its value is `significand * 2^exponent`.
fn bf16_parts(b: u64) -> (bool, u128, i32) {
    let e = ((b >> 7) & 0xFF) as i32;
    let m = (b & 0x7F) as u128;
    if e == 0 {
        (b & 0x8000 != 0, m, -133)
    } else {
        (b & 0x8000 != 0, m | 0x80, e - 134)
    }
}

/// The correctly rounded `fma(a, b, c)` of bfloat bits, from the exact sum.
fn bf16_fma_ref(a: u64, b: u64, c: u64) -> u64 {
    let (x, y, z) = (nval(bfloat(), a), nval(bfloat(), b), nval(bfloat(), c));
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return bf16_of(x * y + z);
    }
    let (na, ma, ea) = bf16_parts(a);
    let (nb, mb, eb) = bf16_parts(b);
    let (nc, mc, ec) = bf16_parts(c);
    let (np, mp, ep) = (na != nb, ma * mb, ea + eb);
    if mp == 0 && mc == 0 {
        return if np && nc { 0x8000 } else { 0 };
    }
    if mp == 0 {
        return c;
    }
    if mc == 0 {
        return bf16_exact(np, mp, ep);
    }
    // Align on the smaller exponent; a term more than 100 bits below the other only acts as
    // a sticky bit, so it is replaced by one unit 100 bits below.
    let (mut mp, mut ep, mut mc, mut ec) = (mp as i128, ep, mc as i128, ec);
    if ep - ec > 100 {
        (mc, ec) = (1, ep - 100);
    } else if ec - ep > 100 {
        (mp, ep) = (1, ec - 100);
    }
    let lo = ep.min(ec);
    let p = if np { -mp } else { mp } << (ep - lo);
    let q = if nc { -mc } else { mc } << (ec - lo);
    let n = p + q;
    if n == 0 {
        return 0;
    }
    bf16_exact(n < 0, n.unsigned_abs(), lo)
}

/// A `bfloat` fused multiply-add rounds once from the exact sum: `Fma` in `f32`, corrected where
/// the `f32` result is a `bfloat` tie the exact sum is not (`17/16 * 17/16 = 289/256` is such a
/// tie, and a tiny addend decides it), on scalars and vectors.
#[test]
fn bfloat_fma_rounds_once() {
    let vals = bf16_values();
    let mut cases: Vec<Case> = Vec::new();
    for (i, a) in vals.iter().enumerate() {
        for (j, b) in vals.iter().enumerate() {
            let c = vals[(i + 3 * j) % vals.len()];
            cases.push(vec![vec![*a], vec![*b], vec![c]]);
        }
    }
    for (a, b, c) in [
        (0x3F88, 0x3F88, 0x2B80),
        (0x3F88, 0x3F88, 0xAB80),
        (0x3F88, 0x3F88, 0x0000),
        (0xBF88, 0x3F88, 0x2B80),
        (0xBF88, 0x3F88, 0xAB80),
        (0x3F88, 0x3F88, 0x3380),
        (0x4388, 0x3F88, 0x0001),
        (0x3F88, 0x3F88, 0xBF90),
        (0x3F81, 0x3F81, 0xBF80),
        (0x7F7F, 0x3FC0, 0xFF7F),
        // `a * b` overflows `f32` while the exact sum `2^122 + 2^120 + 2^114` is a finite
        // `bfloat` tie: the correction must not act on an infinite product.
        (0x5F81, 0x5F81, 0xFF7F),
        (0xDF81, 0x5F81, 0x7F7F),
        (0x5F81, 0xDF81, 0x7F7F),
        (0xDF81, 0xDF81, 0xFF7F),
        // A denormal product (`2^-136 * 129^2`) beside a normal addend.
        (0x1B81, 0x2381, 0x0081),
    ] {
        cases.push(vec![vec![a], vec![b], vec![c]]);
    }
    let mut rng = Rng(5);
    for _ in 0..400 {
        // Exponents near each other, so the sums cancel and round in interesting places.
        let mut r = || (rng.next() & 0x87FF) | 0x3800;
        cases.push(vec![vec![r()], vec![r()], vec![r()]]);
    }
    for f in [bfloat(), vec(bfloat(), 2)] {
        let lanes = usize::from(f.lanes);
        let lcases: Vec<Case> = cases
            .iter()
            .map(|c| c.iter().map(|x| vec![x[0]; lanes]).collect())
            .collect();
        check(
            &format!("fma {f:?}"),
            &[f, f, f],
            f,
            &|fb, x| {
                math::float_fn(
                    fb,
                    FloatFn::Fma,
                    f,
                    Contraction::Forbidden,
                    x,
                    &Lib::instructions(),
                )
            },
            &lcases,
            &|c| Some(vec![bf16_fma_ref(c[0][0], c[1][0], c[2][0]); lanes]),
            false,
        );
    }
}

/// 64-bit integers convert to `bfloat` through a round-to-odd `f32` (values just above and below
/// `bfloat` ties that a plain `f32` step would round onto).
#[test]
fn int64_to_bfloat_rounds_once() {
    let fi = int(64);
    let mut rng = Rng(9);
    let mut ivals = int_values(64, &mut rng, 16);
    ivals.extend([
        (1u64 << 56) + (1 << 48) + 1,
        (1u64 << 56) + (1 << 48),
        (1u64 << 56) + (1 << 48) - 1,
        (1u64 << 63) + (1 << 55) + 1,
        u64::MAX,
        i64::MAX as u64,
        i64::MIN as u64,
        0x0000_0001_0080_0001,
        0x0000_0000_FF80_0000,
        0xFFFF_FFFF_0080_0000,
    ]);
    let cases: Vec<Case> = ivals.iter().map(|v| vec![vec![*v]]).collect();
    for (k, signed) in [(CastKind::UiToFp, false), (CastKind::SiToFp, true)] {
        check(
            &format!("{k:?} i64 to bfloat"),
            &[fi],
            bfloat(),
            &|fb, x| ops::cast(fb, k, fi, bfloat(), x[0]),
            &cases,
            &|c| {
                let v = arg(c, 0);
                let (neg, mag) = if signed && (v as i64) < 0 {
                    (true, u128::from((v as i64).unsigned_abs()))
                } else {
                    (false, u128::from(v))
                };
                Some(vec![bf16_exact(neg, mag, 0)])
            },
            false,
        );
    }
}

/// The half bits of integer `v`, rounded once to nearest even on its exact value by integer
/// arithmetic only (an independent reference for `narrow::round_integer`): infinite from 65520
/// up, as Metal answers (65519 is 65504; 65520 rounds to the even `2^16`).
pub(super) fn half_of_integer(v: i128) -> u64 {
    let sign = if v < 0 { 0x8000 } else { 0 };
    let m = v.unsigned_abs();
    if m >= 65520 {
        return sign | 0x7C00;
    }
    if m == 0 {
        return sign;
    }
    let msb = 127 - m.leading_zeros();
    let r = if msb <= 10 {
        m
    } else {
        let shift = msb - 10;
        let q = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        q << shift
    };
    // `r` is a half's value: at most 65504, with at most 11 significant bits.
    let e = 127 - r.leading_zeros();
    let frac = if e >= 10 {
        (r >> (e - 10)) & 0x3FF
    } else {
        (r << (10 - e)) & 0x3FF
    };
    sign | (u64::from(e + 15) << 10) | frac as u64
}

/// The values of an `ibits`-bit integer whose conversion to `half` is worth checking, as
/// two's-complement bit patterns: every one at 8 bits; above, of either sign, every integer below
/// 300 and from 2030 to 2100 (the last exact binade's end and the first that rounds), the first
/// and last 40 of each binade from `2^11` to `2^16` (its ties, and the carry into the next), every
/// integer from 65480 to 65560 (the overflow at 65520), every 17th up to 66000, the `f32` step's
/// own roundings around `2^24` and `2^31`, and the extremes.
pub(super) fn integers_for_half(ibits: u8) -> Vec<u64> {
    let m = mask(ibits);
    if ibits <= 8 {
        return (0..=m).collect();
    }
    let mut mags: Vec<i64> = (0..300).chain(2030..=2100).chain(65_480..=65_560).collect();
    for e in 11..=16 {
        for j in 0..=40 {
            mags.push((1 << e) + j);
            mags.push((1 << e) - j);
        }
    }
    mags.extend((0..=66_000).step_by(17));
    for base in [1i64 << 24, 1 << 31] {
        mags.extend((-3..=3).map(|d| base + d));
    }
    let mut v: Vec<u64> = mags
        .iter()
        .flat_map(|x| [*x as u64 & m, x.wrapping_neg() as u64 & m])
        .collect();
    v.extend([m, m >> 1, (m >> 1) + 1]);
    if ibits == 64 {
        v.extend([1u64 << 32, (1u64 << 53) + 1]);
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// An integer converts to `half` rounded once to nearest even and infinite from 65520 up, by
/// integer arithmetic with no `OpQuantizeToF16`, at every width, on the values of
/// [`integers_for_half`]. NVIDIA's `OpQuantizeToF16` keeps an integer conversion's value (4097
/// stays 4097), so the module must not hold one.
#[test]
fn integers_round_to_half_once_by_integer_arithmetic() {
    for ibits in [8u8, 16, 32, 64] {
        let fi = int(ibits);
        let cases: Vec<Case> = integers_for_half(ibits)
            .iter()
            .map(|v| vec![vec![*v]])
            .collect();
        for (k, signed) in [(CastKind::UiToFp, false), (CastKind::SiToFp, true)] {
            let what = format!("{k:?} i{ibits} to half");
            let body =
                |fb: &mut FunctionBuilder<'_>, x: &[Value]| ops::cast(fb, k, fi, half(), x[0]);
            let m = kernel_with(&[fi], half(), &body, true, false)
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            assert!(
                !super::super::narrow::quantizes(&m),
                "{what}: the module holds OpQuantizeToF16"
            );
            check(
                &what,
                &[fi],
                half(),
                &body,
                &cases,
                &|c| {
                    let v = arg(c, 0);
                    let v = if signed {
                        i128::from(sext(v, ibits))
                    } else {
                        i128::from(v)
                    };
                    Some(vec![half_of_integer(v)])
                },
                false,
            );
        }
    }
    // Per lane, through vector constants.
    let fi = vec(int(32), 4);
    let to = vec(half(), 4);
    let lanes = [2051u64, 4097, 65_519, 65_520];
    let cases: Vec<Case> = vec![
        vec![lanes.to_vec()],
        vec![lanes.iter().map(|v| v.wrapping_neg() & mask(32)).collect()],
    ];
    let body =
        |fb: &mut FunctionBuilder<'_>, x: &[Value]| ops::cast(fb, CastKind::SiToFp, fi, to, x[0]);
    let m = kernel_with(&[fi], to, &body, true, false).expect("build");
    assert!(
        !super::super::narrow::quantizes(&m),
        "int4 to half4: the module holds OpQuantizeToF16"
    );
    check(
        "SiToFp int4 to half4",
        &[fi],
        to,
        &body,
        &cases,
        &|c| {
            Some(
                c[0].iter()
                    .map(|v| half_of_integer(i128::from(sext(*v, 32))))
                    .collect(),
            )
        },
        false,
    );
}

/// 8-bit float sources of `air.convert` decode exactly (every pattern of each format, to every
/// float destination), with LLVM's semantics of the format names.
#[test]
fn fp8_converts_decode_every_pattern_exactly() {
    use crate::air::intrinsics::Fp8;
    use crate::lower::arith::narrow::Fp8Layout;
    let from = int(8);
    let cases: Vec<Case> = (0..=0xFFu64).map(|b| vec![vec![b]]).collect();
    for fmt in [Fp8::E4M3, Fp8::E4M3Fn, Fp8::E5M2] {
        let l = Fp8Layout::of(fmt);
        for to in [float(32), half(), bfloat(), float(64)] {
            check(
                &format!("{fmt:?} to {to:?}"),
                &[from],
                to,
                &|fb, x| math::convert(fb, ConvertKind::FromFp8(l), from, to, x[0], NanTest::IsNan),
                &cases,
                &|c| Some(vec![nbits(to, l.value(arg(c, 0) as u8))]),
                false,
            );
        }
    }
    // Spot values: the largest finite and the smallest denormal of each format.
    assert_eq!(Fp8Layout::of(Fp8::E4M3Fn).value(0x7E), 448.0);
    assert!(Fp8Layout::of(Fp8::E4M3Fn).value(0x7F).is_nan());
    assert_eq!(Fp8Layout::of(Fp8::E4M3).value(0x77), 240.0);
    assert_eq!(Fp8Layout::of(Fp8::E4M3).value(0x78), f64::INFINITY);
    assert_eq!(Fp8Layout::of(Fp8::E5M2).value(0x7B), 57344.0);
    assert_eq!(Fp8Layout::of(Fp8::E5M2).value(0x01), 2f64.powi(-16));
}

/// An 8-bit float converts to every integer type as its value (the reading) does under Metal's
/// saturating float-to-integer rule: toward zero, NaN to 0, beyond the range (infinities included)
/// to its bound. Every pattern of every format, to `i8` through `i64`, signed and unsigned.
#[test]
fn fp8_to_integer_converts_every_pattern_as_its_value() {
    use crate::air::intrinsics::Fp8;
    use crate::lower::arith::narrow::Fp8Layout;
    let from = int(8);
    let cases: Vec<Case> = (0..=0xFFu64).map(|b| vec![vec![b]]).collect();
    for fmt in [Fp8::E4M3, Fp8::E4M3Fn, Fp8::E5M2] {
        let fp8 = Fp8Layout::of(fmt);
        for bits in [8u8, 16, 32, 64] {
            for signed in [true, false] {
                let to = int(bits);
                let (lo, hi): (i128, i128) = if signed {
                    (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
                } else {
                    (0, (1i128 << bits) - 1)
                };
                check(
                    &format!("{fmt:?} to {}{bits}", if signed { 's' } else { 'u' }),
                    &[from],
                    to,
                    &|fb, x| {
                        math::convert(
                            fb,
                            ConvertKind::Fp8ToInt { fp8, signed },
                            from,
                            to,
                            x[0],
                            NanTest::IsNan,
                        )
                    },
                    &cases,
                    &|c| {
                        let v = fp8.value(arg(c, 0) as u8);
                        // Every finite 8-bit float value is below 2^16 in magnitude.
                        let r = if v.is_nan() {
                            0
                        } else if v.is_infinite() {
                            if v > 0.0 {
                                hi
                            } else {
                                lo
                            }
                        } else {
                            (v.trunc() as i128).clamp(lo, hi)
                        };
                        Some(vec![(r as u64) & mask(bits)])
                    },
                    false,
                );
            }
        }
    }
}

/// `k / d` is correctly rounded (the host's IEEE `f32` division) for every `k` in `[0, d]` of every
/// normalized scale and the sRGB step scale 4095, and for every 97th `k` and both ends of the
/// 16-bit scales.
#[test]
fn normalized_division_is_correctly_rounded() {
    let u = int(32);
    let s = float(32);
    for d in [1u32, 3, 31, 63, 127, 255, 1023, 4095, 32767, 65535] {
        let step = if d > 4095 { 97 } else { 1 };
        let cases: Vec<Case> = (0..=u64::from(d))
            .step_by(step)
            .chain([u64::from(d) - 1, u64::from(d)])
            .map(|k| vec![vec![k]])
            .collect();
        check(
            &format!("k / {d}"),
            &[u],
            s,
            &|fb, k| crate::lower::arith::exact_division::div_exact(fb, u, k[0], d),
            &cases,
            &|c| Some(vec![u64::from((arg(c, 0) as f32 / d as f32).to_bits())]),
            false,
        );
    }
}

/// The value of a packed small-float field with a 5-bit exponent (bias 15) above `man` bits.
fn small_float(field: u64, man: u32) -> f64 {
    let e = (field >> man) & 0x1F;
    let m = (field & ((1 << man) - 1)) as f64 / f64::from(1u32 << man);
    match e {
        0 => m * 2f64.powi(-14),
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + m) * 2f64.powi(e as i32 - 15),
    }
}

/// `air.unpack` of `rg11b10f` and `rgb9e5` decodes every channel exactly (every value is a `half`
/// value, so a `half` result does not round or flush), from random words and the edges (denormal,
/// largest, infinite, and NaN fields; the smallest and largest shared exponents).
#[test]
fn packed_float_formats_unpack_exactly() {
    use crate::air::intrinsics::PackFormat as P;
    let word = int(32);
    let mut rng = Rng(11);
    let mut words: Vec<u64> = (0..32).map(|_| rng.next() & 0xFFFF_FFFF).collect();
    words.extend([
        0,
        0xFFFF_FFFF,
        0x0000_0001,
        0x7C0 | (0x7C1 << 11) | (0x3E0 << 22),
        0x7BF | (0x001 << 11) | (0x3DF << 22),
        0x03F | (0x040 << 11) | (0x01F << 22),
        (31 << 27) | 0x1FF | (0x1FF << 9) | (0x1FF << 18),
        1 | (2 << 9) | (3 << 18),
    ]);
    let cases: Vec<Case> = words.iter().map(|w| vec![vec![*w]]).collect();
    for format in [P::UnormRg11b10f, P::UnormRgb9e5] {
        for lane in [float(32), half()] {
            let f = vec(lane, 3);
            check(
                &format!("unpack {format:?} to {lane:?}"),
                &[word],
                f,
                &|fb, x| math::unpack_float(fb, format, f, x[0]),
                &cases,
                &|c| {
                    let w = arg(c, 0);
                    let vals: Vec<f64> = if format == P::UnormRgb9e5 {
                        let e = (w >> 27) as i32;
                        (0..3)
                            .map(|k| ((w >> (9 * k)) & 0x1FF) as f64 * 2f64.powi(e - 24))
                            .collect()
                    } else {
                        vec![
                            small_float(w & 0x7FF, 6),
                            small_float((w >> 11) & 0x7FF, 6),
                            small_float((w >> 22) & 0x3FF, 5),
                        ]
                    };
                    Some(vals.into_iter().map(|v| nbits(lane, v)).collect())
                },
                false,
            );
        }
    }
}

/// `air.pack` and `air.unpack` of `unorm4x8_srgb` read no table: the census names them as the
/// other normalized formats, and a plan of no division, `fmod` routine or extended-range table
/// charges no routine.
#[test]
fn srgb_packs_read_no_table() {
    use crate::air::intrinsics::{PackDir, PackFormat as P};
    use crate::lower::arith::{routine_words, ArithPlan, ArithStrategy};
    use crate::lower::select::FamilyPlans;
    use crate::lower::strategy::FamilyStrategy;
    let name = |dir, format| {
        ArithStrategy::Pack {
            dir,
            format,
            floats: vec(float(32), 4),
        }
        .name(&FamilyPlans::default())
        .into_owned()
    };
    assert_eq!(name(PackDir::Pack, P::Unorm4x8Srgb), "arith.pack");
    assert_eq!(name(PackDir::Unpack, P::Unorm4x8Srgb), "arith.unpack");
    assert_eq!(name(PackDir::Pack, P::Unorm4x8), "arith.pack");
    assert_eq!(name(PackDir::Unpack, P::UnormRgb10a2), "arith.unpack");
    assert_eq!(routine_words(&ArithPlan::default()).count(), 0);
}

/// A `half` `fadd`, `fsub` or `fmul` takes the device's 16-bit arithmetic, on one to four lanes,
/// exactly where the profile has `shaderFloat16` and takes `DriverDeviation::MetalHalfArithmetic`
/// (the MoltenVK captures alone, measured) or rounds to `half` under the 16-bit float controls (by
/// the Vulkan spec, with no fact); `fdiv`, `frem`, other formats, and a chunk wider than four lanes
/// never do.
#[test]
fn half_arithmetic_on_the_device_follows_the_profile() {
    use super::super::F16Basis;
    use crate::error::DriverDeviation;
    use crate::spirv::builder::QuantizeForm;
    use crate::spirv::testing::is_moltenvk_capture;
    let (mut measured, mut controlled) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let float16 = p.fields().leaf.float16;
        let fact = p.takes_driver_deviation(DriverDeviation::MetalHalfArithmetic) && float16;
        assert_eq!(fact, is_moltenvk_capture(&stem), "{stem}");
        let controls = super::super::narrow::quantize_form(&p) == QuantizeForm::Controlled;
        let basis = match (fact, controls) {
            (true, _) => Some(F16Basis::Measured),
            (false, true) => Some(F16Basis::Controlled),
            (false, false) => None,
        };
        for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul] {
            for lanes in 1..=4 {
                assert_eq!(
                    super::super::f16_arithmetic(&p, op, vec(half(), lanes)),
                    basis,
                    "{stem}: {op:?} on {lanes} lanes"
                );
            }
            for f in [vec(half(), 8), float(32), bfloat(), float(64)] {
                assert_eq!(
                    super::super::f16_arithmetic(&p, op, f),
                    None,
                    "{stem}: {op:?} on {f:?}"
                );
            }
        }
        for op in [AirBin::FDiv, AirBin::FRem] {
            assert_eq!(
                super::super::f16_arithmetic(&p, op, half()),
                None,
                "{stem}: {op:?}"
            );
        }
        measured += usize::from(basis == Some(F16Basis::Measured));
        controlled += usize::from(basis == Some(F16Basis::Controlled));
    }
    assert!(measured >= 1, "the MoltenVK capture takes it, measured");
    assert!(
        controlled >= 2,
        "the float-controls profiles take it, by the spec"
    );
}

/// The device's `half` `+ - *` ([`ops::f16_binary`]) is the exact result rounded once to `half`, on
/// the interpreter's correctly rounded 16-bit arithmetic: the operands narrow and the result widens
/// exactly (a NaN stays a NaN), on scalars and vectors, and the product kept apart as `Fma(a, b,
/// -0.0)` keeps a `-0` product.
#[test]
fn half_arithmetic_on_the_device_is_the_rounded_result() {
    let vals: Vec<u64> = half_values();
    let few: Vec<u64> = vals
        .iter()
        .copied()
        .step_by(2)
        .chain([0x8000, 0x8001])
        .collect();
    let h = half();
    let exact = |op: AirBin, a: f64, b: f64| match op {
        AirBin::FAdd => a + b,
        AirBin::FSub => a - b,
        _ => a * b,
    };
    for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul] {
        check_accept(
            &format!("{op:?} on the device's half"),
            &[h, h],
            h,
            &|fb, x| ops::f16_binary(fb, op, 1, x[0], x[1]),
            &pairs(&vals, &few),
            &|c| {
                Some(vec![nbits(
                    h,
                    exact(op, nval(h, arg(c, 0)), nval(h, arg(c, 1))),
                )])
            },
            &exact_ok(h),
        );
    }
    for lanes in [2u8, 3, 4] {
        let v = vec(h, lanes);
        let n = usize::from(lanes);
        let cases: Vec<Case> = (0..vals.len())
            .map(|i| {
                let l = |k: usize| vals[(i + k) % vals.len()];
                vec![
                    (0..n).map(l).collect(),
                    (0..n).map(|k| l(7 + 2 * k)).collect(),
                ]
            })
            .collect();
        for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul] {
            check_accept(
                &format!("{op:?} on {lanes} lanes of the device's half"),
                &[v, v],
                v,
                &|fb, x| ops::f16_binary(fb, op, lanes, x[0], x[1]),
                &cases,
                &|c| {
                    Some(
                        (0..n)
                            .map(|i| nbits(v, exact(op, nval(v, c[0][i]), nval(v, c[1][i]))))
                            .collect(),
                    )
                },
                &exact_ok(v),
            );
        }
    }
}

/// The device's `half` arithmetic is the instruction on 16-bit floats and a widening, with no
/// rounding of the module's own: no conversion barrier (`narrow::rounds_by_conversion` finds none),
/// and the only instructions computing 16-bit values are the operation and at most two narrowings
/// of its operands: an operand the module widened from a 16-bit float is that float, narrowed by
/// nothing (`FunctionBuilder::convert`); the operation is always kept apart: a sum or difference
/// decorated `NoContraction`, a product `Fma(a, b, -0.0)` decorated `NoContraction`, never an
/// `OpFMul` (undecorated, Metal's fast compile answers the widened operands' `float` product;
/// decorated, SPIRV-Cross's `spvFMul` answers `+0` for a `-0` product).
#[test]
fn half_arithmetic_on_the_device_is_one_16_bit_instruction() {
    use crate::spirv::serialize::serialize;
    use std::collections::BTreeSet;
    // SPIR-V opcodes, `GLSL.std.450 Fma` and the `NoContraction` decoration.
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_EXT_INST: u32 = 12;
    const OP_DECORATE: u32 = 71;
    const OP_FCONVERT: u32 = 115;
    const OP_FADD: u32 = 129;
    const OP_FSUB: u32 = 131;
    const OP_FMUL: u32 = 133;
    const OP_FDIV: u32 = 136;
    const NO_CONTRACTION: u32 = 42;
    const FMA: u32 = 50;
    let h = half();
    for lanes in [1u8, 4] {
        let v = vec(h, lanes);
        for op in [AirBin::FAdd, AirBin::FSub, AirBin::FMul] {
            let what = format!("{op:?} on {lanes} lanes");
            let m = kernel(&[v, v], v, &|fb, x| {
                ops::f16_binary(fb, op, lanes, x[0], x[1])
            })
            .expect("kernel");
            let mut meter = crate::budget::Meter::new(&crate::budget::Budget::default());
            assert!(
                !crate::lower::arith::narrow::rounds_by_conversion(&m, &mut meter).expect("scan"),
                "{what}: a conversion barrier"
            );
            let words = serialize(m).expect("serialize").words;
            let insts = || {
                let mut out = Vec::new();
                let mut i = 5;
                while i < words.len() {
                    let n = ((words[i] >> 16) as usize).max(1);
                    out.push(&words[i..(i + n).min(words.len())]);
                    i += n;
                }
                out
            };
            // The 16-bit float types, scalar and vector.
            let mut f16 = BTreeSet::new();
            for w in insts() {
                match (w[0] & 0xFFFF, w) {
                    (OP_TYPE_FLOAT, [_, id, 16, ..]) => {
                        f16.insert(*id);
                    }
                    (OP_TYPE_VECTOR, [_, id, elem, ..]) if f16.contains(elem) => {
                        f16.insert(*id);
                    }
                    _ => {}
                }
            }
            let decorated: BTreeSet<u32> = insts()
                .into_iter()
                .filter(|w| w[0] & 0xFFFF == OP_DECORATE && w.get(2) == Some(&NO_CONTRACTION))
                .map(|w| w[1])
                .collect();
            // Every instruction whose result is 16-bit, as (opcode, GLSL op, result id).
            let computed: Vec<(u32, u32, u32)> = insts()
                .into_iter()
                .filter_map(|w| {
                    let code = w[0] & 0xFFFF;
                    let arith = matches!(
                        code,
                        OP_FCONVERT | OP_FADD | OP_FSUB | OP_FMUL | OP_FDIV | OP_EXT_INST
                    );
                    (arith && w.len() >= 4 && f16.contains(&w[1])).then(|| {
                        let glsl = if code == OP_EXT_INST { w[4] } else { 0 };
                        (code, glsl, w[2])
                    })
                })
                .collect();
            let want = match op {
                AirBin::FAdd => (OP_FADD, 0),
                AirBin::FSub => (OP_FSUB, 0),
                _ => (OP_EXT_INST, FMA),
            };
            let shapes: Vec<(u32, u32)> = computed.iter().map(|&(c, g, _)| (c, g)).collect();
            let (last, before) = shapes.split_last().expect("the operation");
            assert_eq!(*last, want, "{what}: the 16-bit instructions {shapes:?}");
            assert!(
                before.len() <= 2 && before.iter().all(|s| *s == (OP_FCONVERT, 0)),
                "{what}: the 16-bit instructions {shapes:?}"
            );
            let op_id = computed
                .last()
                .map(|&(_, _, id)| id)
                .expect("the operation");
            assert!(decorated.contains(&op_id), "{what}: NoContraction");
        }
    }
}

/// `from_bits` is the device's widening on every form, with no integer decode: wherever the module
/// can hold a 16-bit float (`QuantizeForm::Converted` and `Controlled`, a profile with
/// `shaderFloat16`), of the bits as 16-bit floats (one conversion of the gathered lanes); in
/// `QuantizeForm::Device` (no `shaderFloat16`), `GLSL.std.450 UnpackHalf2x16` of each lane's word
/// (one a lane, and no 16-bit type). Every half bit pattern, with junk above its 16 bits, reads as
/// its exact `f32` value but a NaN, which reads as a NaN, on one to four lanes, and
/// `narrow::widens_bits` finds the widening.
#[test]
fn a_half_decode_is_the_devices_widening_in_every_form() {
    use super::super::narrow::{f32_of_half_bits, from_bits, widens_bits};
    use crate::spirv::builder::QuantizeForm;
    let junk = |h: u32| u64::from(h | NARROW_JUNK);
    let lanes_of = |n: u32| -> Vec<Case> {
        (0..0x1_0000u32)
            .step_by(n as usize)
            .map(|h| vec![(h..h + n).map(|x| junk(x & 0xFFFF)).collect()])
            .collect()
    };
    let want = |c: &Case| {
        Some(
            c[0].iter()
                .map(|b| u64::from(f32_of_half_bits(*b as u32 & 0xFFFF)))
                .collect(),
        )
    };
    for q in [
        QuantizeForm::Device,
        QuantizeForm::Converted,
        QuantizeForm::Controlled,
    ] {
        let float16 = q != QuantizeForm::Device;
        for lanes in 1u8..=4 {
            let cases = lanes_of(u32::from(lanes));
            let cases = &cases;
            let what = format!("{lanes} lanes, {q:?}");
            let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                fb.module().set_quantize_form(q);
                from_bits(fb, vec(half(), lanes), x[0])
            };
            let (ins, out) = ([vec(int(32), lanes)], vec(float(32), lanes));
            let m = kernel_with(&ins, out, &body, true, false)
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            let budget = crate::budget::Budget::default();
            let mut meter = crate::budget::Meter::new(&budget);
            let widens = widens_bits(&m, &mut meter).expect("widens_bits");
            assert!(widens, "{what}: widens_bits");
            let insts = || m.blocks().iter().flat_map(|b| &b.insts);
            let decodes = insts().any(|n| {
                matches!(
                    n.inst,
                    Inst::Convert {
                        op: ConvOp::ConvertUToF,
                        ..
                    }
                )
            });
            let conversions = insts()
                .filter(|n| {
                    matches!(
                        n.inst,
                        Inst::Convert {
                            op: ConvOp::FConvert,
                            ..
                        }
                    )
                })
                .count();
            let unpacks = insts()
                .filter(|n| {
                    matches!(
                        n.inst,
                        Inst::Glsl {
                            op: GlslOp::UnpackHalf2x16,
                            ..
                        }
                    )
                })
                .count();
            let halves = m
                .types()
                .iter()
                .any(|(_, t)| matches!(t, Type::Float { width: 16 }));
            assert_eq!(conversions, usize::from(float16), "{what}: the conversions");
            let want_unpacks = if float16 { 0 } else { usize::from(lanes) };
            assert_eq!(unpacks, want_unpacks, "{what}: the unpacks");
            assert_eq!(halves, float16, "{what}: a 16-bit float type");
            assert!(!decodes, "{what}: an integer decode");
            check_accept(&what, &ins, out, &body, cases, &want, &|g, w| {
                g == w || (f32::from_bits(g as u32).is_nan() && f32::from_bits(w as u32).is_nan())
            });
        }
    }
}
