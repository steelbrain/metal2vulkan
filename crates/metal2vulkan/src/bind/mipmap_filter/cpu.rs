//! The filter on the CPU ([`super::MipmapFilterChain::expected_levels`]) and its bound
//! ([`super::MipmapFilterChain::outside_bound`]).
//!
//! **The model** takes the kernel's (`kernel.rs`) `f32` operations in its order, each rounded once
//! as IEEE does, the transfers' `Pow` the `f64` power rounded once to `f32`, and a half's
//! narrowing to nearest even.
//!
//! **The bound.** A device computes the same operations by its own arithmetic: it may contract a
//! product and a sum into one rounding, its division and `Pow` carry Vulkan's precision, and a NaN
//! is its own. So each channel of a device's level, filtered from the level above as the device
//! wrote it, lies within [`RELATIVE`] of the sum of its terms' magnitudes (plus [`ABSOLUTE`]) of
//! the model's sum, and each field holds a code the format's encode of some value in that range
//! gives: an 8-bit or extended-range sRGB field's transfer within [`RELATIVE`] of the model's
//! (and `2^-20`) before its rounding, an `RGB9E5Float` channel within one mantissa step at the
//! device's shared exponent. A NaN channel is any NaN where the format keeps one, an infinite one
//! the model's infinity. Off a rounding boundary every field is the model's code; at one, either
//! neighbour.

use super::layout::{axis_tap, Field, Layout, Rule, Texel, EXTENDED_SCALE, SRGB_SCALE};

/// A device channel's distance from the model's, relative to the sum of the magnitudes of the
/// channel's terms: Vulkan's correctly rounded products and sums, contracted or not, err by a
/// few `2^-24` of it over at most eight taps; a decode's division and `Pow` (an sRGB field's)
/// by about `2^-19` of each tap's value. `2^-17` holds both with room.
pub(super) const RELATIVE: f64 = 1.0 / 131_072.0;

/// A device channel's further distance from the model's, `2^-140`: products of subnormal
/// magnitude that one form rounds and the other keeps in a contraction.
pub(super) const ABSOLUTE: f64 = 512.0 * f32::from_bits(1) as f64;

/// The texel's bytes as one little-endian integer.
fn word_of(texel: &[u8]) -> u128 {
    texel
        .iter()
        .rev()
        .fold(0u128, |word, &byte| (word << 8) | u128::from(byte))
}

/// Field `f`'s bits of a texel's `word`.
fn bits(word: u128, f: &Field) -> u32 {
    ((word >> f.offset) as u32) & f.mask()
}

/// `2^e` for `-126 <= e <= 127`.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The value of float bits `b` of a 5-bit exponent and `mantissa` mantissa bits (signed or not):
/// a half's or an unsigned packed float's, exactly, as the device's widening gives it.
pub(super) fn small_float_value(b: u32, mantissa: u32, signed: bool) -> f32 {
    let e = (b >> mantissa) & 31;
    let m = b & ((1 << mantissa) - 1);
    let sign = if signed { (b >> (mantissa + 5)) & 1 } else { 0 };
    let magnitude = match e {
        31 => 0x7f80_0000 | (m << (23 - mantissa)),
        0 => (m as f32 * pow2(-14 - mantissa as i32)).to_bits(),
        _ => ((e + 112) << 23) | (m << (23 - mantissa)),
    };
    f32::from_bits(magnitude | (sign << 31))
}

/// The bits of `v` as a float of a 5-bit exponent and `mantissa` mantissa bits (signed or not),
/// to nearest even: an overflow infinity, a NaN the quiet NaN, a negative value of an unsigned
/// format 0. A half's is the model of the device's narrowing; an unsigned field's is the
/// kernel's own (`kernel::small_float_bits`).
pub(super) fn small_float_bits(v: f32, mantissa: u32, signed: bool) -> u32 {
    let b = v.to_bits();
    let sign = b >> 31;
    let a = b & 0x7fff_ffff;
    let inf = 31 << mantissa;
    if a > 0x7f80_0000 {
        return inf | (1 << (mantissa - 1));
    }
    if !signed && sign == 1 {
        return 0;
    }
    let k = 23 - mantissa;
    let e = a >> 23;
    let r = if e >= 113 {
        let v = a - (112 << 23);
        ((v + (1 << (k - 1)) - 1 + ((v >> k) & 1)) >> k).min(inf)
    } else {
        let shift = k + 113 - e;
        if shift >= 32 {
            0
        } else {
            let sig = (a & 0x7f_ffff) | 0x80_0000;
            (sig + (1 << (shift - 1)) - 1 + ((sig >> shift) & 1)) >> shift
        }
    };
    if signed {
        r | (sign << (mantissa + 5))
    } else {
        r
    }
}

/// `v` with a NaN read as 0.
fn number(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// The model of the device's `Pow`: the `f64` power rounded once.
fn pow(x: f32, y: f32) -> f32 {
    f64::from(x).powf(f64::from(y)) as f32
}

/// The kernel's `f32` constant of `x`: `x` rounded once from `f64` (`Form::const_bits`).
fn k(x: f64) -> f32 {
    x as f32
}

/// The MSL sRGB-to-linear transfer of `c` in the kernel's steps (`lower::arith::pack`'s).
fn srgb_decode(c: f32) -> f32 {
    let low = c / k(12.92);
    let high = pow((c + k(0.055)) / k(1.055), k(2.4));
    if c <= k(0.040_45) {
        low
    } else {
        high
    }
}

/// The MSL linear-to-sRGB transfer of `c` in the kernel's steps (`lower::arith::pack`'s).
fn srgb_encode(c: f32) -> f32 {
    let low = c * k(12.92);
    let high = pow(c, k(1.0 / 2.4)) * k(1.055) - k(0.055);
    if c < k(0.003_130_8) {
        low
    } else {
        high
    }
}

/// `transfer(|v|)` with `v`'s sign.
fn signed_transfer(v: f32, transfer: fn(f32) -> f32) -> f32 {
    let t = transfer(v.abs());
    if v < 0.0 {
        -t
    } else {
        t
    }
}

/// An extended-range colour code of `v` clamped to `[lo, hi]` (`Rule::Extended`).
fn extended_code(v: f32, lo: f32, hi: f32) -> u32 {
    let c = (number(v).clamp(lo, hi) * 510.0).round_ties_even() as i32 + 384;
    c.clamp(0, 1023) as u32
}

/// The value of field `f`'s code `c` (`layout::Rule`).
fn value(f: &Field, c: u32) -> f32 {
    match f.rule {
        Rule::Unorm => c as f32 * f.scale(),
        Rule::Snorm => {
            let shift = 32 - f.width;
            let c = ((c << shift) as i32) >> shift;
            (c as f32 * f.scale()).max(-1.0)
        }
        Rule::Srgb8 => srgb_decode(c as f32 * SRGB_SCALE),
        Rule::Float { exponent: 8, .. } => f32::from_bits(c),
        Rule::Float {
            mantissa, signed, ..
        } => small_float_value(c, mantissa, signed),
        Rule::Extended { alpha } => {
            let c = if alpha { c.clamp(384, 894) } else { c };
            (c as i32 - 384) as f32 * EXTENDED_SCALE
        }
        Rule::ExtendedSrgb => {
            signed_transfer((c as i32 - 384) as f32 * EXTENDED_SCALE, srgb_decode)
        }
        Rule::Padding(_) => 0.0,
    }
}

/// Field `f`'s code of value `v` (`layout::Rule`), in its low `width` bits.
fn code(f: &Field, v: f32) -> u32 {
    match f.rule {
        Rule::Unorm => (number(v).clamp(0.0, 1.0) * f.den() as f32).round_ties_even() as u32,
        Rule::Snorm => {
            let c = (number(v).clamp(-1.0, 1.0) * f.den() as f32).round_ties_even() as i32;
            c as u32 & f.mask()
        }
        Rule::Srgb8 => (srgb_encode(number(v).clamp(0.0, 1.0)) * 255.0).round_ties_even() as u32,
        Rule::Float { exponent: 8, .. } => v.to_bits(),
        Rule::Float {
            mantissa, signed, ..
        } => small_float_bits(v, mantissa, signed),
        Rule::Extended { alpha } => match alpha {
            true => extended_code(v, 0.0, 1.0),
            false => extended_code(v, -1.0, 1.5),
        },
        Rule::ExtendedSrgb => extended_code(signed_transfer(number(v), srgb_encode), -1.0, 1.5),
        Rule::Padding(bits) => bits & f.mask(),
    }
}

/// The `float` channels of a texel of `layout` from its bytes.
pub(super) fn decode(layout: &Layout, texel: &[u8]) -> Vec<f32> {
    let word = word_of(texel);
    match layout.texel {
        Texel::Fields(fields) => fields
            .iter()
            .filter(|f| f.channel())
            .map(|f| value(f, bits(word, f)))
            .collect(),
        Texel::Rgb9e5 => {
            let w = word as u32;
            let scale = pow2((w >> 27) as i32 - 24);
            (0..3)
                .map(|c| ((w >> (9 * c)) & 511) as f32 * scale)
                .collect()
        }
    }
}

/// `2^(24 - e)`: a shared exponent's mantissa scale.
fn rgb9e5_scale(e: i32) -> f32 {
    pow2(24 - e)
}

/// The bytes of a texel of `layout` from its `float` channels.
pub(super) fn encode(layout: &Layout, v: &[f32]) -> Vec<u8> {
    let word: u128 = match layout.texel {
        Texel::Fields(fields) => {
            let mut values = v.iter();
            fields.iter().fold(0u128, |word, f| {
                let c = match f.rule {
                    Rule::Padding(_) => code(f, 0.0),
                    _ => code(f, *values.next().expect("a channel per field")),
                };
                word | (u128::from(c) << f.offset)
            })
        }
        Texel::Rgb9e5 => {
            let c: Vec<f32> = v.iter().map(|&x| number(x).clamp(0.0, 65408.0)).collect();
            let most = c[0].max(c[1]).max(c[2]);
            let floor = (((most.to_bits() >> 23) & 0xff) as i32 - 127).max(-16);
            let mut e = floor + 16;
            if (most * rgb9e5_scale(e) + 0.5).floor() == 512.0 {
                e += 1;
            }
            let scale = rgb9e5_scale(e);
            let mut w = (e as u32) << 27;
            for (i, x) in c.iter().enumerate() {
                let m = ((x * scale + 0.5).floor() as u32).min(511);
                w |= m << (9 * i);
            }
            u128::from(w)
        }
    };
    word.to_le_bytes()[..layout.bytes as usize].to_vec()
}

/// One destination texel's channels: each the weighted sum of its taps in tap order (the
/// model's), and the sum of the terms' magnitudes.
struct Sums {
    value: Vec<f32>,
    magnitude: Vec<f64>,
}

/// Every destination texel's [`Sums`] of one level of every layer of `layout` (`small` texels a
/// layer) from the level above (`big`): each the weighted sum of its taps' channels (`axis_tap`
/// along each axis, the depth taps where the level above is more than one texel deep), a tap of
/// weight 0 left out, in tap order (x fastest, then y, then z).
fn sums(layout: &Layout, above: &[u8], big: [u32; 3], small: [u32; 3], layers: u32) -> Vec<Sums> {
    let bytes = layout.bytes as usize;
    let depth = big[2] > 1;
    let taps = |axis: usize, i: u32| {
        let (x0, t) = axis_tap(i, big[axis], small[axis]);
        [(x0, 1.0 - t), ((x0 + 1).min(big[axis] - 1), t)]
    };
    let count = if depth { 8 } else { 4 };
    let texels = (layers * small[0] * small[1] * small[2]) as usize;
    let mut out = Vec::with_capacity(texels);
    for layer in 0..layers {
        for z in 0..small[2] {
            for y in 0..small[1] {
                for x in 0..small[0] {
                    let (ax, ay) = (taps(0, x), taps(1, y));
                    let az = taps(2, z);
                    let mut sum = Sums {
                        value: vec![0.0f32; layout.channels()],
                        magnitude: vec![0.0f64; layout.channels()],
                    };
                    for t in 0..count {
                        let (sx, wx) = ax[t & 1];
                        let (sy, wy) = ay[(t >> 1) & 1];
                        let (image, w) = if depth {
                            let (sz, wz) = az[(t >> 2) & 1];
                            (layer * big[2] + sz, wx * wy * wz)
                        } else {
                            (layer * small[2] + z, wx * wy)
                        };
                        let index = ((image * big[1] + sy) * big[0] + sx) as usize;
                        let v = decode(layout, &above[index * bytes..(index + 1) * bytes]);
                        let channels = sum.value.iter_mut().zip(&mut sum.magnitude);
                        for ((s, m), v) in channels.zip(v) {
                            let term = if w == 0.0 { 0.0 } else { w * v };
                            *s = if t == 0 { term } else { *s + term };
                            *m += f64::from(term).abs();
                        }
                    }
                    out.push(sum);
                }
            }
        }
    }
    out
}

/// One level of every layer of `layout` (`small` texels a layer) from the level above (`big`):
/// each destination texel's [`sums`] encoded.
pub(super) fn level(
    layout: &Layout,
    above: &[u8],
    big: [u32; 3],
    small: [u32; 3],
    layers: u32,
) -> Vec<u8> {
    sums(layout, above, big, small, layers)
        .iter()
        .flat_map(|s| encode(layout, &s.value))
        .collect()
}

/// The texels of `got`, one level of every layer of `layout` (`small` texels a layer) as a device
/// filtered it from `above` (`big`), outside the bound (the module docs) of the model's.
pub(super) fn outside(
    layout: &Layout,
    above: &[u8],
    got: &[u8],
    big: [u32; 3],
    small: [u32; 3],
    layers: u32,
) -> Vec<u64> {
    let bytes = layout.bytes as usize;
    sums(layout, above, big, small, layers)
        .iter()
        .zip(got.chunks_exact(bytes))
        .enumerate()
        .filter(|(_, (s, texel))| !within(layout, s, texel))
        .map(|(i, _)| i as u64)
        .collect()
}

/// Whether a device's `texel` is within the bound of the model's [`Sums`] `s`.
fn within(layout: &Layout, s: &Sums, texel: &[u8]) -> bool {
    let word = word_of(texel);
    match layout.texel {
        Texel::Fields(fields) => {
            let mut channels = s.value.iter().zip(&s.magnitude);
            fields.iter().all(|f| match f.rule {
                Rule::Padding(_) => bits(word, f) == code(f, 0.0),
                _ => channels
                    .next()
                    .is_some_and(|(&v, &m)| field_within(f, v, m, bits(word, f))),
            })
        }
        Texel::Rgb9e5 => {
            let w = word as u32;
            let step = f64::from(pow2((w >> 27) as i32 - 24));
            let got = decode(layout, texel);
            s.value
                .iter()
                .zip(&s.magnitude)
                .zip(got)
                .all(|((&v, &m), g)| {
                    let reach = RELATIVE * m + ABSOLUTE + step;
                    let clamped = |x: f64| x.clamp(0.0, 65408.0);
                    let v = f64::from(number(v));
                    (clamped(v - reach)..=clamped(v + reach)).contains(&f64::from(g))
                })
        }
    }
}

/// Whether code `got` of field `f` is within the bound of the model's channel `v` of terms of
/// magnitude `m`.
fn field_within(f: &Field, v: f32, m: f64, got: u32) -> bool {
    if !v.is_finite() {
        return match f.rule {
            Rule::Float { exponent: 8, .. } => {
                let g = f32::from_bits(got);
                if v.is_nan() {
                    g.is_nan()
                } else {
                    g == v
                }
            }
            Rule::Float {
                mantissa, signed, ..
            } => {
                let g = small_float_value(got, mantissa, signed);
                let want = small_float_value(code(f, v), mantissa, signed);
                if want.is_nan() {
                    g.is_nan()
                } else {
                    g == want
                }
            }
            _ => got == code(f, v),
        };
    }
    let reach = RELATIVE * m + ABSOLUTE;
    let (lo, hi) = (f64::from(v) - reach, f64::from(v) + reach);
    // The sRGB transfers' slack before the encode's rounding: `RELATIVE` of the value and 2^-20.
    let slack = |t: f32, toward: f64| {
        let t = f64::from(t);
        t + toward * (RELATIVE * t.abs() + 1.0 / 1_048_576.0)
    };
    match f.rule {
        Rule::Unorm | Rule::Extended { .. } => {
            (code(f, lo as f32)..=code(f, hi as f32)).contains(&got)
        }
        Rule::Snorm => {
            let signed = |c: u32| ((c << (32 - f.width)) as i32) >> (32 - f.width);
            (signed(code(f, lo as f32))..=signed(code(f, hi as f32))).contains(&signed(got))
        }
        Rule::Srgb8 => {
            let encoded = |x: f64, toward: f64| {
                let t = srgb_encode((x as f32).clamp(0.0, 1.0));
                (slack(t, toward).clamp(0.0, 1.0) * 255.0).round_ties_even() as u32
            };
            (encoded(lo, -1.0)..=encoded(hi, 1.0)).contains(&got)
        }
        Rule::ExtendedSrgb => {
            let encoded = |x: f64, toward: f64| {
                let t = signed_transfer(x as f32, srgb_encode);
                extended_code(slack(t, toward) as f32, -1.0, 1.5)
            };
            (encoded(lo, -1.0)..=encoded(hi, 1.0)).contains(&got)
        }
        Rule::Float { exponent: 8, .. } => {
            let g = f32::from_bits(got);
            if g.is_infinite() {
                let max = f64::from(f32::MAX);
                return if g > 0.0 { hi >= max } else { lo <= -max };
            }
            !g.is_nan() && (lo..=hi).contains(&f64::from(g))
        }
        Rule::Float {
            mantissa, signed, ..
        } => {
            let of = |x: f64| {
                small_float_value(
                    small_float_bits(x as f32, mantissa, signed),
                    mantissa,
                    signed,
                )
            };
            let g = small_float_value(got, mantissa, signed);
            !g.is_nan() && (of(lo)..=of(hi)).contains(&g)
        }
        Rule::Padding(_) => got == code(f, 0.0),
    }
}
