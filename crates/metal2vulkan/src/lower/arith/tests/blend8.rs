//! A fetched 8-bit unorm attachment's blend on the interpreter
//! (`extended_range::blend::blend_unorm8`), of a `half4` or `float4` output, under fast math and
//! without, against Metal's measured arithmetic. Metal clamps the output to `[0, 1]` (a NaN to 0),
//! reads the destination's code `c` as `h(c / 255)` (`f(c / 255)` for a `float4` output) and the
//! blend color clamped to `[0, 1]` (as `half` in `half` arithmetic), blends in the output's type,
//! and stores the result's code by its unorm write (a NaN 0, clamped, the exact product with 255
//! rounded to nearest). The function writes the blend's result, which the storage image's store
//! converts (the device's conversion, `PrecisionNote::DeviceColorConversion`): each lane is checked
//! through Vulkan's conversion of it ([`stored`]: a NaN 0, clamped, the product with 255 rounded to
//! nearest), which gives Metal's code.
//!
//! Premultiplied source-over (one over one minus the source alpha, added, for color and alpha):
//! fast math `h(fma(d, h(1 - a), s))`, without `h(s + h(d · h(1 - a)))`, in `float` likewise.
//! Our probe (plain Metal on the M3 Max under macOS 27.0.1, our own MSL of the guest's shape,
//! a 4096 x 961 target): every `half` alpha in `[0, 1]` against every destination code of each
//! channel, four source draws per pair (half aimed within four units in the last place of a
//! code's midpoint, 2% out of range, infinite or NaN), each of `half4` and `float4`, fast and safe:
//! this arithmetic gives Metal's code in every one of 63 million channels per variant, into
//! `BGRA8Unorm`, and `RGBA8Unorm`, `RG8Unorm` and `R8Unorm` alike. Here every alpha code's `half`
//! (or `float`) meets every destination code in some lane, beside sources aimed at the codes'
//! midpoints, out of range and NaN, and texels read a unit in the last place off `c / 255`.
//!
//! Other factors, operations, write masks and the blend color: a state of each rule
//! `Blend::unorm8` names (the color's contracted pair and alpha's lerp of source alpha over, the
//! blend color's lerps, a lone color channel's lerp, and alpha alone beside the color channels'
//! destination factor), against the arithmetic the probes measured.

use super::*;
use crate::lower::arith::extended_range::blend::{
    blend_unorm8, half_of, unorm8_tile_value, Blend, BlendArith, BlendMath,
};
use crate::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::BlendEquation;

/// `x` rounded to `half`, nearest even.
fn h(x: f64) -> f64 {
    f16_to_f64(f64_to_f16(x))
}

/// `x` clamped to `[0, 1]`, a NaN to 0.
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Metal's code of source lane `s` (alpha `a`, both as the output holds them) over destination
/// code `c`, in `arith` under `math` (the module docs).
fn metal_code(arith: BlendArith, math: BlendMath, (s, a): (f64, f64), c: u32) -> u32 {
    let (s, a) = (clamp01(s), clamp01(a));
    let v = match arith {
        BlendArith::Half => {
            let d = h(f64::from(c) / 255.0);
            let m = h(1.0 - a);
            // Products and sums of these `half`s are exact in `f64`.
            match math {
                BlendMath::Fast => h(d * m + s),
                BlendMath::Rounded => h(s + h(d * m)),
            }
        }
        BlendArith::Float => {
            let d = c as f32 / 255.0;
            let (s, a) = (s as f32, a as f32);
            let m = 1.0 - a;
            f64::from(match math {
                BlendMath::Fast => d.mul_add(m, s),
                BlendMath::Rounded => s + d * m,
            })
        }
    };
    (clamp01(v) * 255.0 + 0.5).floor() as u32
}

/// The `f32` bits of a source lane: a `half` (or a `float` where `float`) aimed within four
/// units in the last place of the midpoint between two codes of the blend of destination code
/// `c` under alpha `a`, a uniform one in `[0, 1]`, or, one time in 16, out of range, an infinity
/// or a NaN.
fn source(rng: &mut Rng, float: bool, (a, c): (f64, u32)) -> u64 {
    let w = rng.next();
    const SPECIALS: [f32; 8] = [
        -0.0,
        -1.0,
        2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.5,
        -0.25,
    ];
    if w.is_multiple_of(16) {
        return u64::from(SPECIALS[((w >> 4) % 8) as usize].to_bits());
    }
    let x = if (w >> 8) & 1 == 0 {
        ((w >> 16) & 0xFFFF) as f64 / 65535.0
    } else {
        let k = ((w >> 16) % 255) as f64;
        let ulp = if float {
            2f64.powi(-24)
        } else {
            2f64.powi(-11)
        };
        let step = ((w >> 32) % 9) as f64 - 4.0;
        (k + 0.5) / 255.0 - f64::from(c) / 255.0 * (1.0 - a) + step * ulp
    };
    let x = x.clamp(0.0, 1.0);
    match float {
        true => u64::from((x as f32).to_bits()),
        false => u64::from((h(x) as f32).to_bits()),
    }
}

/// The `f32` bits of a texel lane of code `c` as a storage image reads it: `c / 255`, a unit in
/// the last place off it one time in four (a decode the function must read as `c` all the same).
fn texel(rng: &mut Rng, c: u32) -> u64 {
    let v = c as f32 / 255.0;
    let v = match rng.next() % 8 {
        0 if c > 0 => f32::from_bits(v.to_bits() - 1),
        1 if c < 255 => f32::from_bits(v.to_bits() + 1),
        _ => v,
    };
    u64::from(v.to_bits())
}

/// A destination code's `half`, `h(c / 255)`, is the `half` of `c` times the `f32` nearest
/// `1 / 255` (`unorm8_tile_value`'s product): on the host for every code, and through the lowering
/// under each `half` rounding for every code, read as the storage image may decode it (`c / 255`,
/// and a unit in the last place off it either way).
#[test]
fn a_code_s_half_is_its_product_s() {
    for c in 0..=255u32 {
        let product = c as f32 * (1.0f32 / 255.0);
        assert_eq!(
            f64_to_f16(f64::from(product)),
            f64_to_f16(f64::from(c) / 255.0),
            "code {c}"
        );
    }
    let f4 = vec(float(32), 4);
    let mut cases: Vec<Case> = Vec::new();
    for step in [-1i64, 0, 1] {
        for k in 0..64u32 {
            let lanes = (0..4)
                .map(|l| {
                    let c = 4 * k + l;
                    let v = (c as f32 / 255.0).to_bits();
                    let v = match (step, c) {
                        (-1, 1..) => v - 1,
                        (1, ..=254) => v + 1,
                        _ => v,
                    };
                    u64::from(v)
                })
                .collect();
            cases.push(vec![lanes]);
        }
    }
    for q in HALF_FORMS {
        check(
            &format!("a destination code's half, half by {q:?}"),
            &[f4],
            f4,
            &|fb, x| {
                fb.module().set_quantize_form(q);
                unorm8_tile_value(fb, x[0], BlendArith::Half)
            },
            &cases,
            &|c| {
                let lanes = c[0].iter().map(|&b| {
                    let code = (f64::from(f32v(b)) * 255.0).round();
                    u64::from((h(code / 255.0) as f32).to_bits())
                });
                Some(lanes.collect())
            },
            false,
        );
    }
}

/// The `half` nearest `n / d` (a quotient in `[2^-14, 1]`, or zero), from the exact quotient:
/// `n` scaled to eleven significant bits over `d`, rounded to nearest even by the remainder.
fn exact_half(n: u64, d: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    // No quotient but 1 lies near a power of two, so the `f64`'s exponent is the quotient's.
    let e = (n as f64 / d as f64).log2().floor() as i32;
    assert!(e >= -14, "{n} / {d} is no normal half");
    let scaled = n << (10 - e);
    let (mut q, r) = (scaled / d, scaled % d);
    if 2 * r > d || (2 * r == d && q % 2 == 1) {
        q += 1;
    }
    q as f64 * 2f64.powi(e - 10)
}

/// An 8-bit sRGB destination read in `half` (`blend_srgb8`): the `half` of the `f32` nearest each
/// twelve-bit step over 4095 (the decode of a color byte) and each byte over 255 (alpha's), as the
/// decode gives it correctly rounded, is the `half` nearest the exact quotient: for every step and
/// byte, so the read rounds once, as Metal's.
#[test]
fn an_srgb_step_s_half_is_its_quotient_s() {
    for d in [4095u64, 255] {
        for n in 0..=d {
            let near = n as f32 / d as f32;
            assert_eq!(
                f64_to_f16(f64::from(near)),
                f64_to_f16(exact_half(n, d)),
                "{n} / {d}"
            );
        }
    }
}

/// `half_of` rounds to `half` as Metal does under each `half` rounding: to nearest even, half
/// denormals kept, an infinity from 65520 up, the sign of a zero kept. Over every finite `half` of
/// either sign: the `half` itself, the `f32`s beside it, the midpoint to the next `half` up in
/// magnitude and the `f32`s beside that midpoint; and the infinities, `f32` denormals, and the
/// edge of the infinity.
#[test]
fn half_of_rounds_as_metal_under_each_form() {
    let f4 = vec(float(32), 4);
    let mut inputs: Vec<f32> = Vec::new();
    for bits in 0..0x7C00u16 {
        for sign in [1.0f64, -1.0] {
            let v = sign * f16_to_f64(bits);
            let next = sign * f16_to_f64(bits + 1);
            let v32 = v as f32;
            let mid = ((v + next) / 2.0) as f32;
            for x in [v32, mid] {
                inputs.push(x);
                inputs.push(f32::from_bits(x.to_bits().wrapping_add(1)));
                if x.to_bits() & 0x7FFF_FFFF != 0 {
                    inputs.push(f32::from_bits(x.to_bits() - 1));
                }
            }
        }
    }
    let edges = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        f32::from_bits(0x477F_EFFF),
        65520.0,
        -65520.0,
        1.0e30,
    ];
    inputs.extend(edges);
    while !inputs.len().is_multiple_of(4) {
        inputs.push(0.0);
    }
    let cases: Vec<Case> = inputs
        .chunks(4)
        .map(|l| vec![l.iter().map(|x| u64::from(x.to_bits())).collect()])
        .collect();
    for q in HALF_FORMS {
        check(
            &format!("half_of, half by {q:?}"),
            &[f4],
            f4,
            &|fb, x| {
                fb.module().set_quantize_form(q);
                half_of(fb, f4, x[0])
            },
            &cases,
            &|c| {
                let lanes = c[0].iter().map(|&b| {
                    let x = f64::from(f32v(b));
                    u64::from((f16_to_f64(f64_to_f16(x)) as f32).to_bits())
                });
                Some(lanes.collect())
            },
            false,
        );
    }
}

/// The 8-bit unorm code Vulkan's conversion stores for the `f32` lane `bits` (Fixed-Point Data
/// Conversions: a NaN 0, the value clamped to `[0, 1]`, its product with 255 rounded to nearest).
fn stored(bits: u64) -> u64 {
    (clamp01(f64::from(f32v(bits))) * 255.0).round_ties_even() as u64
}

/// Whether lane `got`, stored by Vulkan's conversion, gives the code of lane `want`.
fn same_code(got: u64, want: u64) -> bool {
    stored(got) == stored(want)
}

#[test]
fn a_fetched_eight_bit_unorm_blend_stores_metals_code() {
    let f4 = vec(float(32), 4);
    let over = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    let one_over = f64::from(1.0f32 / 255.0);
    for arith in [BlendArith::Half, BlendArith::Float] {
        let float = arith == BlendArith::Float;
        let mut rng = Rng(0xB1E8_0C0D ^ u64::from(float));
        // Every alpha code's value against every destination code: alpha code `j`, and each
        // case's lanes the codes `4k` to `4k + 3`.
        let mut cases: Vec<Case> = Vec::new();
        for j in 0..256u32 {
            let a = match float {
                true => f64::from(j as f32 / 255.0),
                false => h(f64::from(j) / 255.0),
            };
            let a_bits = u64::from((a as f32).to_bits());
            for k in 0..64u32 {
                let codes: Vec<u32> = (0..4).map(|l| 4 * k + l).collect();
                let mut s: Vec<u64> = codes[..3]
                    .iter()
                    .map(|&c| source(&mut rng, float, (a, c)))
                    .collect();
                s.push(a_bits);
                let d = codes.iter().map(|&c| texel(&mut rng, c)).collect();
                cases.push(vec![s, d]);
            }
        }
        for math in [BlendMath::Fast, BlendMath::Rounded] {
            for pf in [PixelFormat::BGRA8Unorm, PixelFormat::R8Unorm] {
                let what = format!("{arith:?} {math:?} {pf:?}");
                let b = Blend::unorm8(&over, pf, ColorWriteMask::ALL, (math, arith))
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                for q in HALF_FORMS {
                    check_accept(
                        &format!("the fetched 8-bit unorm blend, {what}, half by {q:?}"),
                        &[f4, f4],
                        f4,
                        &|fb, x| {
                            fb.module().set_quantize_form(q);
                            blend_unorm8(fb, &b, arith, (x[0], x[1]), None)
                        },
                        &cases,
                        &|c| {
                            let a = f64::from(f32v(c[0][3]));
                            Some(
                                (0..4)
                                    .map(|l| {
                                        let code =
                                            (f64::from(f32v(c[1][l])) * 255.0).round() as u32;
                                        let s = f64::from(f32v(c[0][l]));
                                        // A format without alpha stores no alpha: the blend
                                        // leaves the lane the texel's.
                                        let got = match (pf, l) {
                                            (PixelFormat::R8Unorm, 3) => code,
                                            _ => metal_code(arith, math, (s, a), code),
                                        };
                                        u64::from(((got as f32) * one_over as f32).to_bits())
                                    })
                                    .collect(),
                            )
                        },
                        &same_code,
                    );
                }
            }
        }
    }
}

/// A factor of the second source, in any of the four factors, and another format are not built:
/// `Blend::unorm8` refuses each `NotImplemented(FramebufferFetch)`, under either math.
#[test]
fn a_second_source_or_another_format_is_not_built() {
    let over = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    let refused = |r: Result<Blend, crate::error::Error>| match r {
        Err(crate::error::Error::Unsupported(u)) => {
            u.feature == crate::error::Feature::FramebufferFetch
                && u.needs == crate::error::Needs::NotImplemented
        }
        _ => false,
    };
    let cases = [
        (
            BlendEquation {
                source_rgb: F::Source1Color,
                ..over
            },
            PixelFormat::BGRA8Unorm,
        ),
        (
            BlendEquation {
                destination_rgb: F::OneMinusSource1Color,
                ..over
            },
            PixelFormat::RGBA8Unorm,
        ),
        (
            BlendEquation {
                source_alpha: F::Source1Alpha,
                ..over
            },
            PixelFormat::RG8Unorm,
        ),
        (
            BlendEquation {
                destination_alpha: F::OneMinusSource1Alpha,
                ..over
            },
            PixelFormat::R8Unorm,
        ),
        (over, PixelFormat::RGBA8UnormSRGB),
    ];
    for (eq, pf) in cases {
        for math in [BlendMath::Fast, BlendMath::Rounded] {
            assert!(
                refused(Blend::unorm8(
                    &eq,
                    pf,
                    ColorWriteMask::ALL,
                    (math, BlendArith::Half)
                )),
                "{eq:?} {pf:?} {math:?}"
            );
        }
    }
}

/// `x` rounded in `arith`: to `half`, nearest even, or to `float`.
fn rounded(arith: BlendArith, x: f64) -> f64 {
    match arith {
        BlendArith::Half => h(x),
        BlendArith::Float => f64::from(x as f32),
    }
}

/// `a b + c` fused, rounded once in `arith` (a product of `half`s and its sum with one are exact
/// in `f64` over a blend's operands).
fn fused(arith: BlendArith, (a, b, c): (f64, f64, f64)) -> f64 {
    match arith {
        BlendArith::Half => h(a * b + c),
        BlendArith::Float => f64::from((a as f32).mul_add(b as f32, c as f32)),
    }
}

/// One lane's operands as Metal's blend reads them: the source lane `s` and alpha `sa` clamped,
/// the destination lane `d` and alpha `da` decoded, and the blend color's lane `c` and alpha `ca`
/// clamped, each rounded to the arithmetic's type.
#[derive(Clone, Copy)]
struct Lane {
    s: f64,
    sa: f64,
    d: f64,
    da: f64,
    c: f64,
    ca: f64,
}

#[test]
fn the_factor_set_masks_and_blend_color_store_metals_code() {
    let f4 = vec(float(32), 4);
    let u4 = vec(int(32), 4);
    let one_over = 1.0f32 / 255.0;
    let colors: [[f32; 4]; 2] = [[0.3, 0.6, 0.9, 0.45], [-0.25, 1.5, 0.321, 0.8]];
    let eq = |(fs, fd, op): (F, F, O), (sa, da, oa): (F, F, O)| BlendEquation {
        source_rgb: fs,
        destination_rgb: fd,
        rgb_operation: op,
        source_alpha: sa,
        destination_alpha: da,
        alpha_operation: oa,
    };
    type Form = fn(BlendArith, BlendMath, Lane) -> f64;
    // Each state, its format and mask, and the color and alpha lanes' Metal arithmetic; `None`
    // where the lane keeps its texel.
    let states: [(
        BlendEquation,
        PixelFormat,
        ColorWriteMask,
        Option<Form>,
        Option<Form>,
    ); 4] = [
        // Source alpha over: the color's source product rounded first; alpha a lerp of the
        // source alpha from the destination's.
        (
            eq(
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ),
            PixelFormat::BGRA8Unorm,
            ColorWriteMask::ALL,
            Some(|ar, math, l| {
                let m = rounded(ar, 1.0 - l.sa);
                match math {
                    BlendMath::Fast => fused(ar, (l.d, m, rounded(ar, l.s * l.sa))),
                    BlendMath::Rounded => {
                        rounded(ar, rounded(ar, l.s * l.sa) + rounded(ar, l.d * m))
                    }
                }
            }),
            Some(|ar, math, l| {
                let m = rounded(ar, 1.0 - l.sa);
                match math {
                    BlendMath::Fast => fused(ar, (l.sa, rounded(ar, l.sa - l.da), l.da)),
                    BlendMath::Rounded => {
                        rounded(ar, rounded(ar, l.sa * l.sa) + rounded(ar, l.da * m))
                    }
                }
            }),
        ),
        // The blend color and its one minus: lerps of it under fast math.
        (
            eq(
                (F::BlendColor, F::OneMinusBlendColor, O::Add),
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
            ),
            PixelFormat::RGBA8Unorm,
            ColorWriteMask::ALL,
            Some(|ar, math, l| match math {
                BlendMath::Fast => fused(ar, (l.c, rounded(ar, l.s - l.d), l.d)),
                BlendMath::Rounded => {
                    let m = rounded(ar, 1.0 - l.c);
                    rounded(ar, rounded(ar, l.s * l.c) + rounded(ar, l.d * m))
                }
            }),
            Some(|ar, math, l| match math {
                BlendMath::Fast => fused(ar, (l.ca, rounded(ar, l.sa - l.da), l.da)),
                BlendMath::Rounded => {
                    let m = rounded(ar, 1.0 - l.ca);
                    rounded(ar, rounded(ar, l.sa * l.ca) + rounded(ar, l.da * m))
                }
            }),
        ),
        // One color channel stored: a lerp of the source alpha a vector does not take.
        (
            eq(
                (F::OneMinusSourceAlpha, F::SourceAlpha, O::Add),
                (F::One, F::Zero, O::Add),
            ),
            PixelFormat::BGRA8Unorm,
            ColorWriteMask::BLUE,
            Some(|ar, math, l| match math {
                BlendMath::Fast => fused(ar, (l.sa, rounded(ar, l.d - l.s), l.s)),
                BlendMath::Rounded => {
                    let m = rounded(ar, 1.0 - l.sa);
                    rounded(ar, rounded(ar, l.s * m) + rounded(ar, l.d * l.sa))
                }
            }),
            None,
        ),
        // Alpha alone stored: the color channels' destination factor still computes one minus
        // the destination alpha, so alpha takes no lerp of it.
        (
            eq(
                (F::Zero, F::OneMinusDestinationAlpha, O::Add),
                (F::OneMinusDestinationColor, F::DestinationColor, O::Add),
            ),
            PixelFormat::BGRA8Unorm,
            ColorWriteMask::ALPHA,
            None,
            Some(|ar, math, l| {
                let m = rounded(ar, 1.0 - l.da);
                match math {
                    BlendMath::Fast => fused(ar, (l.sa, m, rounded(ar, l.da * l.da))),
                    BlendMath::Rounded => {
                        rounded(ar, rounded(ar, l.sa * m) + rounded(ar, l.da * l.da))
                    }
                }
            }),
        ),
    ];
    for arith in [BlendArith::Half, BlendArith::Float] {
        let float = arith == BlendArith::Float;
        let mut rng = Rng(0xFAC7_0B1E ^ u64::from(float));
        let mut cases: Vec<Case> = Vec::new();
        for bc in colors {
            let color: Vec<u64> = bc.iter().map(|c| u64::from(c.to_bits())).collect();
            for _ in 0..1500 {
                let codes: Vec<u32> = (0..4).map(|_| (rng.next() % 256) as u32).collect();
                let a = h(f64::from((rng.next() % 256) as u32) / 255.0);
                let s = codes
                    .iter()
                    .map(|&c| source(&mut rng, float, (a, c)))
                    .collect();
                let d = codes.iter().map(|&c| texel(&mut rng, c)).collect();
                cases.push(vec![s, d, color.clone()]);
            }
        }
        for (e, pf, mask, color_form, alpha_form) in states {
            for math in [BlendMath::Fast, BlendMath::Rounded] {
                let what = format!("{arith:?} {math:?} {pf:?} {mask:?} {e:?}");
                let b = Blend::unorm8(&e, pf, mask, (math, arith))
                    .unwrap_or_else(|err| panic!("{what}: {err:?}"));
                assert_eq!(b.reads_constant(), pf == PixelFormat::RGBA8Unorm, "{what}");
                for q in HALF_FORMS {
                    check_accept(
                        &format!("the fetched 8-bit unorm blend, {what}, half by {q:?}"),
                        &[f4, f4, u4],
                        f4,
                        &|fb, x| {
                            fb.module().set_quantize_form(q);
                            let color = b.reads_constant().then_some(x[2]);
                            blend_unorm8(fb, &b, arith, (x[0], x[1]), color)
                        },
                        &cases,
                        &|c| {
                            let code = |l: usize| (f64::from(f32v(c[1][l])) * 255.0).round() as u32;
                            let read = |x: f64| rounded(arith, clamp01(x));
                            let lane = |l: usize| Lane {
                                s: read(f64::from(f32v(c[0][l]))),
                                sa: read(f64::from(f32v(c[0][3]))),
                                d: rounded(arith, f64::from(code(l)) / 255.0),
                                da: rounded(arith, f64::from(code(3)) / 255.0),
                                c: read(f64::from(f32v(c[2][l]))),
                                ca: read(f64::from(f32v(c[2][3]))),
                            };
                            Some(
                                (0..4)
                                    .map(|l| {
                                        let form = if l < 3 { color_form } else { alpha_form };
                                        let got = match form {
                                            Some(form) => {
                                                let v = form(arith, math, lane(l));
                                                (clamp01(v) * 255.0 + 0.5).floor() as u32
                                            }
                                            None => code(l),
                                        };
                                        u64::from(((got as f32) * one_over).to_bits())
                                    })
                                    .collect(),
                            )
                        },
                        &same_code,
                    );
                }
            }
        }
    }
}

/// A format without alpha (`R8Unorm`, `RG8Unorm`) sums its products under fast math in forms of
/// its own: a lerp, common factor or contracted pair is refused `NotImplemented(FramebufferFetch)`
/// there, a fused or single product built, and without fast math every blend is built.
#[test]
fn a_format_without_alpha_builds_only_its_measured_fast_forms() {
    let eq = |fs, fd| BlendEquation {
        source_rgb: fs,
        destination_rgb: fd,
        rgb_operation: O::Add,
        source_alpha: fs,
        destination_alpha: fd,
        alpha_operation: O::Add,
    };
    // Each blend and whether fast math builds it.
    let cases = [
        (eq(F::SourceAlpha, F::OneMinusSourceAlpha), false),
        (eq(F::OneMinusSourceAlpha, F::SourceAlpha), false),
        (eq(F::DestinationColor, F::OneMinusSourceAlpha), false),
        (eq(F::One, F::OneMinusSourceAlpha), true),
        (eq(F::Zero, F::SourceColor), true),
        (eq(F::BlendColor, F::One), true),
    ];
    for pf in [PixelFormat::R8Unorm, PixelFormat::RG8Unorm] {
        for (e, fast) in cases {
            for math in [BlendMath::Fast, BlendMath::Rounded] {
                let built = Blend::unorm8(&e, pf, ColorWriteMask::ALL, (math, BlendArith::Half));
                let what = format!("{pf:?} {math:?} {e:?}");
                match built {
                    Ok(_) => assert!(fast || math == BlendMath::Rounded, "{what}"),
                    Err(crate::error::Error::Unsupported(u)) => {
                        assert!(!fast && math == BlendMath::Fast, "{what}");
                        assert_eq!(u.feature, crate::error::Feature::FramebufferFetch, "{what}");
                        assert_eq!(u.needs, crate::error::Needs::NotImplemented, "{what}");
                    }
                    Err(e) => panic!("{what}: {e:?}"),
                }
            }
        }
    }
}
