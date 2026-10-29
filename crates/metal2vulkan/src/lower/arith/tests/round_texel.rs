//! Interpreter differential tests of a texel's `half` (`narrow::round_texel`, through
//! `texture::held_texel`, which every texture read, sample, and gather with a `half` result takes).
//!
//! The kernel reads `f32` words, the device's decode of an image texel, converts them as a texture
//! access does, and stores the `half` bits. A half-float texel decodes to its exact `f32` value
//! (a NaN to the `f32` NaN of its sign and payload), so the conversion must give the texel's own
//! bits back: every subnormal, both zeros, both infinities and the largest half; a NaN stays a
//! NaN, its bits the store's narrowing's, the device's. A wider texel rounds to nearest even, half
//! subnormals kept. `OpQuantizeToF16` alone fails the subnormals: the interpreter gives a value
//! below the smallest normal half as the zero of its sign.

use super::texel_round::half_probe_bits;
use super::*;
use crate::lower::texture::{held_texel, TexelClass, TexelConv, TexelForm};

fn half() -> Form {
    float(16)
}

/// The `f32` bits of half `h` widened exactly: a NaN keeps its sign and its payload at the top of
/// the `f32` significand.
fn widened(h: u16) -> u64 {
    let sign = u32::from(h & 0x8000) << 16;
    let bits = if h & 0x7C00 == 0x7C00 {
        sign | 0x7F80_0000 | (u32::from(h & 0x3FF) << 13)
    } else {
        sign | ((f16_to_f64(h) as f32).to_bits() & 0x7FFF_FFFF)
    };
    u64::from(bits)
}

/// The conversion of an image texel of `lanes` components to its held `half`s.
fn texel(lanes: u8) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        let form = TexelForm {
            conv: TexelConv::F16,
            lanes,
        };
        held_texel(fb, TexelClass::Float, form, x[0])
    }
}

/// The `half` bits a texel of `f32` bits `w` reads as: a NaN keeps its sign and the top of its
/// payload; any other value rounds to nearest even.
fn want(w: u64) -> u64 {
    let w = w as u32;
    if w & 0x7FFF_FFFF > 0x7F80_0000 {
        u64::from(w >> 16 & 0x8000) | 0x7C00 | u64::from(w >> 13 & 0x3FF)
    } else {
        u64::from(f64_to_f16(f64::from(f32::from_bits(w))))
    }
}

/// Every half bit pattern, widened as the device decodes a half-float texel, reads back as itself,
/// bit for bit: subnormals, zeros, infinities and `±65504`; a NaN of every payload, signaling ones
/// included, reads back as a NaN, its bits the device's narrowing's at the store
/// (`PrecisionNote::NativeNanEncoding`); one lane at a time and four at once, as a `half4` read
/// holds them.
#[test]
fn a_half_texel_keeps_every_bit_pattern() {
    // The reference is the identity on every widened half.
    for h in 0..=0xFFFFu16 {
        assert_eq!(want(widened(h)), u64::from(h), "{h:#06x}");
    }
    let cases: Vec<Case> = (0..=0xFFFFu16).map(|h| vec![vec![widened(h)]]).collect();
    let same = |g: u64, w: u64| g == w || (is_nan(half(), g) && is_nan(half(), w));
    check_accept(
        "half texel",
        &[float(32)],
        half(),
        &texel(1),
        &cases,
        &|c| Some(vec![want(arg(c, 0))]),
        &same,
    );
    let lanes: Vec<Case> = (0..0x1_0000u32)
        .step_by(4)
        .map(|h| vec![(h..h + 4).map(|x| widened(x as u16)).collect()])
        .collect();
    check_accept(
        "half4 texel",
        &[vec(float(32), 4)],
        vec(half(), 4),
        &texel(4),
        &lanes,
        &|c| Some(c[0].iter().map(|w| want(*w)).collect()),
        &same,
    );
}

/// A wider texel (a 32-bit float format read as `half`) rounds to nearest even onto the half grid,
/// subnormals included, a zero keeping its sign, past `65504` to the infinity; a NaN stays a NaN.
#[test]
fn a_wider_texel_rounds_to_nearest_even_keeping_subnormals() {
    let cases: Vec<Case> = half_probe_bits()
        .into_iter()
        .map(|b| vec![vec![b]])
        .collect();
    check_accept(
        "f32 texel to half",
        &[float(32)],
        half(),
        &texel(1),
        &cases,
        &|c| {
            Some(vec![u64::from(f64_to_f16(f64::from(f32::from_bits(
                arg(c, 0) as u32,
            ))))])
        },
        &|g, w| g == w || (is_nan(half(), g) && is_nan(half(), w)),
    );
}

/// Under native half rounding (the device's conversion or quantize, where neither is measured to be
/// Metal's) a texel's `half` is the form's own rounding: every half-float texel and every wider
/// value but a NaN reads as before, to nearest even with half subnormals kept (the interpreter's
/// conversion rounds so; its quantize answers a subnormal as its signed zero, which SPIR-V allows),
/// and a NaN stays a NaN, its bits the form's, without the keep's unsigned compare of the texel's
/// bits.
#[test]
fn a_texel_under_native_half_rounding_takes_its_forms_nan() {
    use crate::spirv::builder::QuantizeForm;
    let mut cases: Vec<Case> = (0..=0xFFFFu16).map(|h| vec![vec![widened(h)]]).collect();
    cases.extend(half_probe_bits().into_iter().map(|b| vec![vec![b]]));
    for q in [QuantizeForm::Converted, QuantizeForm::Device] {
        let mut compares = [0usize; 2];
        for native in [false, true] {
            let what = format!("{q:?} native {native}");
            let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                fb.module().set_quantize_form(q);
                fb.module().set_native_half(native);
                texel(1)(fb, x)
            };
            let m = kernel_with(&[float(32)], half(), &body, true, false)
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            compares[usize::from(native)] = m
                .blocks()
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter(|n| {
                    matches!(
                        n.inst,
                        Inst::Binary {
                            op: BinOp::UGreaterThan,
                            ..
                        }
                    )
                })
                .count();
            check_accept(
                &what,
                &[float(32)],
                half(),
                &body,
                &cases,
                &|c| {
                    Some(vec![u64::from(f64_to_f16(f64::from(f32::from_bits(
                        arg(c, 0) as u32,
                    ))))])
                },
                &|g, w| {
                    g == w
                        || (is_nan(half(), g) && is_nan(half(), w))
                        || (q == QuantizeForm::Device && w & 0x7C00 == 0 && g == w & 0x8000)
                },
            );
        }
        // The keep's one test of the texel's bits, left out under native half rounding.
        assert_eq!(compares[0], compares[1] + 1, "{q:?}");
    }
}

/// The unsigned compares of module `m`: the keep's test of a texel's bits is one.
fn unsigned_compares(m: &crate::spirv::ir::TypedModule) -> usize {
    m.blocks()
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|n| {
            matches!(
                n.inst,
                Inst::Binary {
                    op: BinOp::UGreaterThan,
                    ..
                }
            )
        })
        .count()
}

/// Under the float controls (`QuantizeForm::Controlled`), where the module leaves a NaN's bits to
/// the device (`ModuleBuilder::native_half_nans`, every profile so rounding), a texel's `half` is
/// the conversion's own rounding: every half-float texel and every wider value but a NaN reads as
/// with the keep, to nearest even with half subnormals kept, and a NaN stays a NaN, without the
/// keep's unsigned compare of the texel's bits. Without the flag the keep stands, and every half
/// bit pattern reads back as itself, a NaN as a NaN (its bits the device's narrowing's at the
/// store).
#[test]
fn a_texel_under_the_float_controls_takes_the_conversions_nan_by_default() {
    use crate::spirv::builder::QuantizeForm;
    let halves: Vec<Case> = (0..=0xFFFFu16).map(|h| vec![vec![widened(h)]]).collect();
    let mut cases = halves.clone();
    cases.extend(half_probe_bits().into_iter().map(|b| vec![vec![b]]));
    let mut compares = [0usize; 2];
    for native in [false, true] {
        let what = format!("Controlled, NaN bits the device's {native}");
        let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
            fb.module().set_quantize_form(QuantizeForm::Controlled);
            fb.module().set_native_half_nans(native);
            texel(1)(fb, x)
        };
        let m = kernel_with(&[float(32)], half(), &body, true, false)
            .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
        compares[usize::from(native)] = unsigned_compares(&m);
        check_accept(
            &what,
            &[float(32)],
            half(),
            &body,
            &cases,
            &|c| {
                Some(vec![u64::from(f64_to_f16(f64::from(f32::from_bits(
                    arg(c, 0) as u32,
                ))))])
            },
            &|g, w| g == w || (is_nan(half(), g) && is_nan(half(), w)),
        );
        if !native {
            check_accept(
                &format!("{what}, every half"),
                &[float(32)],
                half(),
                &body,
                &halves,
                &|c| Some(vec![want(arg(c, 0))]),
                &|g, w| g == w || (is_nan(half(), g) && is_nan(half(), w)),
            );
        }
    }
    // The keep's one test of the texel's bits, left out where the NaN bits are the device's.
    assert_eq!(compares[0], compares[1] + 1, "{compares:?}");
}
