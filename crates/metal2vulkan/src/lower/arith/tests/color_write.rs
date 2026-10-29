//! A float rounded to the code Metal stores in a color channel (`color_write::round`), on the
//! interpreter, against the rules probes measured on an Apple M3 Max and an M4
//! (`runtime_lane::color_writes` holds the same rules).
//!
//! Each case writes one value to every lane of a four-lane output whose lanes take different
//! rules, so every rule runs beside the others. The reference is the value each rule writes: the
//! small float Metal stores, widened, or the value as computed. A unorm channel is the device's
//! conversion's, which `round` does not take.

use super::*;
use crate::lower::arith::color_write;
use crate::pipeline::lower_inputs::ChannelWrite;

/// Metal's code of `x` in a float channel of a five-bit exponent (bias 15) and `man` fraction
/// bits (a sign bit above where `signed`): rounded toward zero, a finite value past the largest
/// to the largest, an infinity kept, a NaN the quiet NaN, a negative value 0 where unsigned.
/// Computed in `f64` on the value's grid, apart from the emission's bit operations.
fn small_float_code(x: f32, man: u32, signed: bool) -> u32 {
    let infinity = 0x1F << man;
    if x.is_nan() {
        return infinity | (1 << (man - 1));
    }
    if x.is_sign_negative() && !signed {
        return 0;
    }
    let sign = if x.is_sign_negative() {
        1 << (man + 5)
    } else {
        0
    };
    let a = f64::from(x.abs());
    if a.is_infinite() {
        return sign | infinity;
    }
    if a >= (2.0 - 2f64.powi(-(man as i32))) * 32768.0 {
        return sign | (infinity - 1);
    }
    let man = man as i32;
    if a < 2f64.powi(-14) {
        return sign | (a / 2f64.powi(-14 - man)).floor() as u32;
    }
    let e = ((x.abs().to_bits() >> 23) & 0xFF) as i32 - 127;
    let k = (a / 2f64.powi(e - man)).floor() as u32;
    sign | (((e + 15) as u32) << man) | (k - (1 << man))
}

/// Metal's code of `x` in such a float channel rounded to nearest even (a render state's
/// half-float channel under `textureWriteRoundingMode` 2): [`small_float_code`] of `x` rounded to
/// nearest with ties to even on its grid, computed in `f64`, a finite value whose rounding reaches
/// 2^16 infinity.
fn small_float_nearest_code(x: f32, man: u32, signed: bool) -> u32 {
    if x.is_nan() || x.is_infinite() || (x.is_sign_negative() && !signed) {
        return small_float_code(x, man, signed);
    }
    let a = f64::from(x.abs());
    let e = if a < 2f64.powi(-14) {
        -14
    } else {
        ((x.abs().to_bits() >> 23) & 0xFF) as i32 - 127
    };
    let q = 2f64.powi(e - man as i32);
    let r = (a / q).round_ties_even() * q;
    let r = if r >= 65536.0 { f64::INFINITY } else { r };
    let v = (if x.is_sign_negative() { -r } else { r }) as f32;
    small_float_code(v, man, signed)
}

/// The value of code `c` of a float channel as [`small_float_code`] describes it (not a NaN).
fn small_float_value(c: u32, man: u32, signed: bool) -> f64 {
    let negative = signed && (c >> (man + 5)) & 1 == 1;
    let e = (c >> man) & 0x1F;
    let f = f64::from(c & ((1 << man) - 1));
    let man = man as i32;
    let a = match e {
        0 => f * 2f64.powi(-14 - man),
        0x1F => f64::INFINITY,
        _ => (1.0 + f * 2f64.powi(-man)) * 2f64.powi(e as i32 - 15),
    };
    if negative {
        -a
    } else {
        a
    }
}

/// The value `rule` writes for `x`, as `f32` bits.
fn written(rule: Option<ChannelWrite>, x: f32) -> u64 {
    let bits = match rule {
        None | Some(ChannelWrite::Device | ChannelWrite::BlendedNearestEven { .. }) => x.to_bits(),
        Some(
            rule @ (ChannelWrite::FloatTowardZero { mantissa, signed }
            | ChannelWrite::FloatNearestEven { mantissa, signed }),
        ) => {
            let man = u32::from(mantissa);
            let c = match rule {
                ChannelWrite::FloatNearestEven { .. } => small_float_nearest_code(x, man, signed),
                _ => small_float_code(x, man, signed),
            };
            if (c >> man) & 0x1F == 0x1F && c & ((1 << man) - 1) != 0 {
                0x7FC0_0000
            } else {
                (small_float_value(c, man, signed) as f32).to_bits()
            }
        }
        Some(rule) => unreachable!("no reference of {rule:?}: `round` takes none"),
    };
    u64::from(bits)
}

/// The values beside each code and each tie of a channel of maximum `m` (every `step`-th),
/// `n` `f32` steps either side, over `[0, 1]`.
fn around_codes(m: u32, step: usize, n: u32) -> Vec<f32> {
    let mut out = Vec::new();
    for k in (0..=m).step_by(step) {
        for c in [k as f64, k as f64 + 0.5] {
            let x = (c / f64::from(m)) as f32;
            for d in 0..=n {
                out.push(f32::from_bits(x.to_bits().wrapping_add(d)));
                out.push(f32::from_bits(x.to_bits().wrapping_sub(d)));
            }
        }
    }
    out
}

fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.0,
        -2.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-10,
        -1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xFFC0_0000),
        f32::from_bits(0x7F80_0001),
        65504.0,
        65519.0,
        65520.0,
        -65520.0,
        1e10,
        -1e10,
    ]
}

/// The half grid: every 7th finite half and the ties beside it, one `f32` step either side,
/// of either sign.
fn half_values() -> Vec<f32> {
    let mut out = Vec::new();
    for h in (0u16..0x7C00)
        .step_by(7)
        .chain([0x03FF, 0x0400, 0x7BFE, 0x7BFF])
    {
        let a = f16_to_f64(h);
        let b = if h == 0x7BFF {
            65536.0
        } else {
            f16_to_f64(h + 1)
        };
        let t = ((a + b) / 2.0) as f32;
        for v in [
            a as f32,
            t,
            f32::from_bits(t.to_bits() + 1),
            f32::from_bits(t.to_bits() - 1),
        ] {
            out.extend([v, -v]);
        }
    }
    out
}

/// An unsigned float grid of `man` fraction bits: every finite code's value and the ties beside
/// it, one `f32` step either side, and their negations.
fn small_float_values(man: u32) -> Vec<f32> {
    let mut out = Vec::new();
    let largest = (0x1F << man) - 1;
    for c in 0..=largest {
        let a = small_float_value(c, man, false);
        let b = if c == largest {
            65536.0
        } else {
            small_float_value(c + 1, man, false)
        };
        let t = ((a + b) / 2.0) as f32;
        for v in [
            a as f32,
            t,
            f32::from_bits(t.to_bits() + 1),
            f32::from_bits(t.to_bits() - 1),
        ] {
            out.extend([v, -v]);
        }
    }
    out
}

fn run_rules(what: &str, rules: [Option<ChannelWrite>; 4], values: &[f32]) {
    let f = vec(float(32), 4);
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits()); 4]])
        .collect();
    check_accept(
        what,
        &[f],
        f,
        &|fb, x| color_write::round(fb, f, x[0], &rules),
        &cases,
        &|c| {
            let x = f32::from_bits(c[0][0] as u32);
            Some(rules.iter().map(|r| written(*r, x)).collect())
        },
        &|g, w| g == w,
    );
}

/// A render state's half-float channel under `textureWriteRoundingMode` 2 rounds to nearest even
/// (`color_writes::HALF_NEAREST`): the consumer's spot values on an M3 Max and our probe's
/// on an M4 (1 + 3 * 2^-12 to `0x3C01`, the tie 1 + 3 * 2^-11 to `0x3C02`, 65519 to `0x7BFF`, from
/// 65520 infinity), and on the interpreter every 7th half and the ties beside it, the unsigned
/// 11- and 10-bit grids under the same rule, and the specials: each lane of an output beside
/// lanes of the other rules, and every lane of a vector by one rule's instructions on the whole
/// vector (`color_write::small_float`, a storage write's), both rules.
#[test]
fn nearest_even_halves_round_to_mode_twos_codes() {
    let code = |x: f32| small_float_nearest_code(x, 10, true);
    assert_eq!(code(1.0 + 3.0 * 2f32.powi(-12)), 0x3C01);
    assert_eq!(code(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
    assert_eq!(code(65519.0), 0x7BFF);
    assert_eq!(code(65520.0), 0x7C00);
    assert_eq!(code(-65520.0), 0xFC00);
    assert_eq!(code(f32::NAN), 0x7E00);
    let mut values = half_values();
    values.extend(small_float_values(6));
    values.extend(small_float_values(5));
    values.extend(specials());
    let nearest = |mantissa, signed| Some(ChannelWrite::FloatNearestEven { mantissa, signed });
    run_rules(
        "nearest-even half, half, 11-bit, 10-bit",
        [
            Some(crate::pipeline::color_writes::HALF_NEAREST),
            nearest(10, true),
            nearest(6, false),
            nearest(5, false),
        ],
        &values,
    );
    let f = vec(float(32), 4);
    for rule in [
        crate::pipeline::color_writes::HALF_NEAREST,
        crate::pipeline::color_writes::HALF,
        ChannelWrite::FloatNearestEven {
            mantissa: 6,
            signed: false,
        },
    ] {
        let cases: Vec<Case> = values
            .iter()
            .map(|x| vec![vec![u64::from(x.to_bits()); 4]])
            .collect();
        check_accept(
            &format!("{rule:?} on the whole vector"),
            &[f],
            f,
            &|fb, x| color_write::small_float(fb, f, x[0], rule),
            &cases,
            &|c| Some(vec![written(Some(rule), f32::from_bits(c[0][0] as u32)); 4]),
            &|g, w| g == w,
        );
    }
}

/// A unorm channel is the device's conversion's: `round` takes no unorm rule (it is malformed
/// there), and a lane of `ChannelWrite::Device` beside a lane of no rule is written as computed.
#[test]
fn unorm_channels_are_the_device_s_to_convert() {
    let f = vec(float(32), 4);
    for bits in [8u8, 10, 16] {
        let rules = [Some(ChannelWrite::Unorm { bits }), None, None, None];
        let built = kernel_with(
            &[f],
            f,
            &|fb, x| color_write::round(fb, f, x[0], &rules),
            true,
            false,
        );
        assert!(
            built.is_err(),
            "a {bits}-bit unorm rule is rounded in the shader"
        );
    }
    let mut values = specials();
    values.extend(half_values().into_iter().step_by(9));
    run_rules(
        "device, none, device, none",
        [
            Some(ChannelWrite::Device),
            None,
            Some(ChannelWrite::Device),
            None,
        ],
        &values,
    );
}

/// `RG11B10Float`'s unsigned 11- and 10-bit floats, beside a half.
#[test]
fn packed_float_channels_round_toward_zero() {
    let mut values = small_float_values(6);
    values.extend(small_float_values(5));
    values.extend(half_values());
    values.extend(specials());
    let f = |mantissa, signed| Some(ChannelWrite::FloatTowardZero { mantissa, signed });
    run_rules(
        "float11, float11, float10, half",
        [f(6, false), f(6, false), f(5, false), f(10, true)],
        &values,
    );
}

/// The reference rounding agrees with the half one Metal's probes gave (`0x7E00` for NaN,
/// `0x7BFF` past 65504) and with `RG11B10Float`'s spot answers.
#[test]
fn small_float_codes_give_metals_spot_answers() {
    assert_eq!(small_float_code(f32::NAN, 10, true), 0x7E00);
    assert_eq!(small_float_code(65520.0, 10, true), 0x7BFF);
    assert_eq!(small_float_code(-1.0, 10, true), 0xBC00);
    assert_eq!(small_float_code(8.940_697e-8, 10, true), 1);
    assert_eq!(small_float_code(f32::NAN, 6, false), 0x7E0);
    assert_eq!(small_float_code(f32::NAN, 5, false), 0x3F0);
    assert_eq!(small_float_code(f32::NEG_INFINITY, 6, false), 0);
    assert_eq!(small_float_code(-1.0, 5, false), 0);
    assert_eq!(small_float_code(f32::INFINITY, 6, false), 0x7C0);
    assert_eq!(small_float_code(1e10, 6, false), 0x7BF);
    assert_eq!(small_float_code(1e10, 5, false), 0x3DF);
    assert_eq!(small_float_code(1.0, 6, false), 15 << 6);
    assert_eq!(small_float_code(1.0 + 1.0 / 64.0 - 1e-7, 6, false), 15 << 6);
    for h in (0u16..0x7C00).step_by(5) {
        let x = f16_to_f64(h) as f32;
        assert_eq!(small_float_code(x, 10, true), u32::from(h), "{h:#06x}");
        assert_eq!(
            small_float_code(-x, 10, true),
            u32::from(h | 0x8000),
            "-{h:#06x}"
        );
    }
}

/// A scalar output and a half output (held widened) round as their lanes do.
#[test]
fn scalar_and_half_outputs_round() {
    let values: Vec<f32> = half_values().into_iter().chain(specials()).collect();
    let s = float(32);
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits())]])
        .collect();
    let nearest = Some(crate::pipeline::color_writes::HALF_NEAREST);
    let rules = [nearest, None, None, None];
    check_accept(
        "scalar nearest-even half",
        &[s],
        s,
        &|fb, x| color_write::round(fb, s, x[0], &rules),
        &cases,
        &|c| Some(vec![written(rules[0], f32::from_bits(c[0][0] as u32))]),
        &|g, w| g == w,
    );
    let h = vec(float(16), 2);
    let halves: Vec<u64> = (0u64..0x10000).step_by(3).collect();
    let cases: Vec<Case> = halves.iter().map(|b| vec![vec![*b, *b]]).collect();
    let toward = Some(ChannelWrite::FloatTowardZero {
        mantissa: 6,
        signed: false,
    });
    let rules = [nearest, toward, None, None];
    let out = vec(float(32), 2);
    check_accept(
        "half nearest-even half, 11-bit",
        &[h],
        out,
        &|fb, x| color_write::round(fb, h, x[0], &rules),
        &cases,
        &|c| {
            let x = f16_to_f64(c[0][0] as u16) as f32;
            Some(vec![written(rules[0], x), written(rules[1], x)])
        },
        &|g, w| (is_nan(out, g) && is_nan(out, w)) || g == w,
    );
}

/// A word attachment's encode (`stage_io::word_color::encode`) stores Metal's word for every
/// packed 16-bit layout: each lane rounded to `half`, then its code rounded to even
/// (`WordAttachment::output_word`, which the M3 Max's draws and clears matched on every texel),
/// over the values beside each code and tie of the 1-, 4-, 5- and 6-bit channels, the half grid
/// and the specials, each lane of a case a different value, and a three-lane output's alpha 1;
/// and `RGB9E5Float`'s word of a `float` output, its mantissas truncated, over those and values
/// across its range.
#[test]
fn word_attachments_store_metals_words() {
    use crate::metal::format::WordAttachment;
    use crate::metal::PixelFormat;
    let mut values = Vec::new();
    for m in [1, 15, 31, 63] {
        values.extend(around_codes(m, 1, 2));
    }
    values.extend(half_values().into_iter().step_by(5));
    values.extend(specials());
    // `RGB9E5Float`'s range: past its largest, its smallest step, and between.
    values.extend([
        70000.0,
        65408.0,
        65409.0,
        1e-7,
        2f32.powi(-24),
        3e-5,
        1000.3,
        2.5,
        0.375,
    ]);
    let n = values.len();
    let quad = |i: usize| [0, 1, 2, 3].map(|l| values[(i + l * 7919) % n]);
    for pf in [
        PixelFormat::B5G6R5Unorm,
        PixelFormat::A1BGR5Unorm,
        PixelFormat::ABGR4Unorm,
        PixelFormat::BGR5A1Unorm,
        PixelFormat::RGB9E5Float,
    ] {
        let word = WordAttachment::layout(pf).expect("a word layout");
        for lanes in [4u8, 3] {
            let f = vec(float(32), lanes);
            let cases: Vec<Case> = (0..n)
                .map(|i| {
                    let q = quad(i);
                    vec![q[..usize::from(lanes)]
                        .iter()
                        .map(|x| u64::from(x.to_bits()))
                        .collect()]
                })
                .collect();
            check_accept(
                &format!("{pf:?} word of {lanes} lanes"),
                &[f],
                int(32),
                &|fb, x| crate::lower::stage_io::word_color::encode(fb, word, (f, x[0])),
                &cases,
                &|c| {
                    let mut rgba = [0.0, 0.0, 0.0, 1.0];
                    for (out, b) in rgba.iter_mut().zip(&c[0]) {
                        *out = f32::from_bits(*b as u32);
                    }
                    Some(vec![u64::from(word.output_word(rgba))])
                },
                &|g, w| g == w,
            );
        }
    }
}
