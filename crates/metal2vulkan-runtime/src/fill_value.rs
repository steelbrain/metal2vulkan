//! The texels Metal's color fills write: `fillTexture:level:slice:region:color:` and
//! `fillTexture:level:slice:region:color:pixelFormat:` (`blit_fill.rs`) pack the color by rules of
//! their own, not as a render pass's `Clear` converts it (`clear_value.rs`).
//!
//! **Metal's answer**, measured on an Apple M4 under macOS 27.0.1 with our own probe `fil.m` (no
//! guest data), every texel read back through a blit: our probe filled 4,096 colors (specials, NaNs
//! of either kind and sign, random values) a texel each into a private texture of each of 74
//! formats, and packed 256 colors of 12 pairs in another format; our probe bisected over the
//! doubles the least color storing each code of each channel (every code of a field of at most 12
//! bits, sampled pairs of the wider ones: 172,707 thresholds); our probe filled NaNs of 24 payloads
//! and signs and 26 doubles at rounding edges into each format; our probe packed 128 colors in each
//! format into textures of 18 others, which hold the packed texel repeated, as an Apple M3 Max
//! under macOS 26.7 does too, a fill at a time and 2,249 fills in one encoder alike (our probe
//! `fil5.m`). On an Apple M3 Max under macOS 27.0.1 our probe `pack.m` packed 1,144 colors (88
//! specials, 1,024 random, 32 NaNs) in each of the 88 formats Metal packs a color in, into
//! `RGBA32Uint` texels, each holding the packed texel repeated, the extended-range `_sRGB` twins'
//! too, in one encoder, an encoder a fill and a process a fill alike (`twins.m`); `classify.m`
//! found the 213 other formats pack nothing. This module reproduces every texel and every
//! threshold.
//!
//! Each channel converts its double; every rule but the integers' takes the double's `float`
//! first (rounded to nearest even, an overflow the infinity):
//! - **Unsigned normalized**: NaN is 0, the value clamped to `[0, 1]`, times the largest code `m`,
//!   rounded half up. A 4- or 8-bit channel rounds the exact product; a channel of 1, 2, 5, 6, 10
//!   or 16 bits rounds the product rounded to `float`: 0.5/65535 stores 1 in a 16-bit channel and
//!   0.5/1023 stores 1 in a 10-bit one, where a clear stores 0, and an `A1BGR5Unorm` alpha of 0.5
//!   stores 1, where a clear's `half` rounding stores 0.
//! - **Signed normalized**: NaN is 0, clamped to `[-1, 1]`, the product with `m` rounded to
//!   `float`, then halves away from zero.
//! - **sRGB** (8-bit): the code of [`SRGB_FIRST`], the least `float` storing each code, which
//!   picks the code whose decoded value is nearest the color (to within a `float` step), where a
//!   clear rounds the encoding curve; NaN is 0. Alpha is unsigned normalized.
//! - **Unsigned integer**: the double truncated toward zero, saturating; a quiet NaN (the
//!   double's top mantissa bit set) is 0 and a signalling one the largest; a 32-bit channel
//!   stores `2^31` for a double strictly between `2^31 - 1` and `2^31`.
//! - **Signed integer**: the double truncated toward zero, saturating; a quiet NaN is the
//!   smallest and a signalling one the largest.
//! - **Half**: rounded to nearest even, an overflow the infinity, subnormals kept; a NaN keeps its
//!   sign and the top 10 bits of its payload, quieted. **Float**: a NaN keeps its sign and the top
//!   22 bits of its payload, quieted. **Bfloat**: the nearest code, ties to even; a NaN keeps its
//!   sign and the top 6 bits of its payload, quieted.
//! - **`RG11B10Float`**: the nearest code, ties to even; negative values are 0, a finite value
//!   past the largest is the largest, the infinity stays, and a NaN is all ones (`0x7ff`, `0x3ff`).
//!   A value below the smallest normal is first cut toward zero to a multiple of `2^-37` (as
//!   though the `float`'s mantissa were shifted down to the result's, the bits shifted out
//!   dropped): `2^-21 (1 + 2^-17)`, past the half step to code 1, still stores 0.
//! - **`RGB9E5Float`**: NaN and negative values are 0, values past 65408 are 65408; the shared
//!   exponent is the largest channel's binade plus one (at least -15), raised by one where that
//!   channel's 9-bit mantissa rounds up to 512, and every mantissa rounds half up.
//! - **Extended range** (`BGR10_XR`, `BGRA10_XR`): the code `⌊(v + 384/510) · 510 + ½⌋`, each
//!   step a `float` operation; a color is clamped to `[-384/510, 639/510]` (NaN the low end), an
//!   alpha to `[0, 1]` (NaN 0). The `_sRGB` twins' colors take the code of [`XR_SRGB_FIRST`], NaN
//!   code 384 (0); their alpha is linear.
//! - **RGBX**: the twin's red, green and blue; the X field zeros where the twin's alpha is
//!   normalized or an integer, the alpha's conversion where it is a float.
//!
//! A texel's fields lie as Metal lays them out, little-endian from the texel's lowest bit:
//! red first, or as the format's name orders them (`BGRA8Unorm` blue first, `ARGB8Unorm` alpha
//! first, `A1BGR5Unorm` alpha in bit 0, `B5G6R5Unorm` blue in bits 0 to 4); `BGRA10_XR` is four
//! 16-bit words, blue, green, red and alpha, each code in the word's top 10 bits; `BGR10_XR` is
//! blue in bits 0 to 9, green, then red, the top two bits zero; `RGB9E5Float`'s exponent is last.
//!
//! These formats are packed by the same rules: the `RGB8`, `RGB16` and `RGB32` ones (their
//! `RGBA` twins' first three channels), `BGRA4Unorm` (blue in bits 0 to 3, then
//! green, red and alpha), `ARGB8Unorm_sRGB` (alpha first, then the sRGB red, green and blue),
//! `BGRA10Unorm` and `BGRA10Uint` (four 16-bit words, blue, green, red and alpha, each code in the
//! word's top 10 bits), and `RGBA16Float_XR` (halves of the `float` clamped to `[-384/510,
//! 639/510]`, alpha to `[0, 1]`, NaN 0, an alpha of 0 or less `+0`).
//!
//! A plain fill of a texture of a format whose packing converts a `float` ([`device_converts`])
//! and that the device renders is the device's clear of the color, not this packing
//! (`blit_fill.rs`); [`device_apart`] and [`device_nan`] say where its texel may stand apart from
//! the packed one.
//!
//! In every other format Metal packs no color (compressed, 4:2:2, depth, stencil, two-plane,
//! YCbCr, padded 10-bit, `Invalid`, the private 10- and 12-bit, PQ and HLG formats, and
//! `RGB10A2Unorm_sRGB`): its fill writes bytes it did not pack for the color (`blit_fill.rs`), and
//! there is no packing here.

use metal2vulkan::metal::PixelFormat;

/// How a color fill converts one channel (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conversion {
    /// Unsigned normalized, of this many bits.
    Unorm(u32),
    /// Signed normalized, of this many bits.
    Snorm(u32),
    /// An 8-bit sRGB-encoded color channel.
    Srgb,
    /// Unsigned integer, of this many bits.
    Uint(u32),
    /// Signed integer, of this many bits.
    Sint(u32),
    /// IEEE half.
    Half,
    /// IEEE single.
    Float,
    /// A bfloat code: the top 16 bits of a `float`.
    Bfloat,
    /// An unsigned small float with a 5-bit exponent and this many mantissa bits.
    Ufloat(u32),
    /// A field of this many bits a fill writes as zeros: an RGBX format's X field.
    Zero(u32),
    /// An extended-range color channel's 10-bit code.
    XrColor,
    /// An extended-range alpha's 10-bit code.
    XrAlpha,
    /// An extended-range `_sRGB` color channel's 10-bit code.
    XrSrgb,
    /// An extended-range half color channel: the `float` clamped to Apple's extended range.
    XrHalf,
    /// An extended-range half alpha: the `float` clamped to `[0, 1]`.
    XrHalfAlpha,
}

impl Conversion {
    /// The field's bits.
    fn width(self) -> u32 {
        match self {
            Conversion::Unorm(bits)
            | Conversion::Snorm(bits)
            | Conversion::Uint(bits)
            | Conversion::Sint(bits)
            | Conversion::Zero(bits) => bits,
            Conversion::Srgb => 8,
            Conversion::Half
            | Conversion::Bfloat
            | Conversion::XrHalf
            | Conversion::XrHalfAlpha => 16,
            Conversion::Float => 32,
            Conversion::Ufloat(mantissa) => mantissa + 5,
            Conversion::XrColor | Conversion::XrAlpha | Conversion::XrSrgb => 10,
        }
    }

    /// The field's code for the color channel `x`.
    fn code(self, x: f64) -> u64 {
        match self {
            Conversion::Unorm(bits) => unorm(x, bits),
            Conversion::Snorm(bits) => snorm(x, bits),
            Conversion::Srgb => first_at_most(&SRGB_FIRST, x, 0),
            Conversion::Uint(bits) => uint(x, bits),
            Conversion::Sint(bits) => sint(x, bits),
            Conversion::Half => half(x),
            Conversion::Float => float(x),
            Conversion::Bfloat => bfloat(x),
            Conversion::Ufloat(mantissa) => ufloat(x, mantissa),
            Conversion::Zero(_) => 0,
            Conversion::XrColor => extended(x, XR_LOW, XR_HIGH),
            Conversion::XrAlpha => extended(x, 0.0, 1.0),
            Conversion::XrSrgb => first_at_most(&XR_SRGB_FIRST, x, XR_ZERO),
            Conversion::XrHalf => {
                let v = x as f32;
                let v = if v.is_nan() {
                    0.0
                } else {
                    v.clamp(XR_LOW, XR_HIGH)
                };
                u64::from(super::clear_value::f32_to_f16(v))
            }
            // Zero or less, `-0` and NaN among them, is `+0`.
            Conversion::XrHalfAlpha => {
                let v = x as f32;
                let v = if v > 0.0 { v.min(1.0) } else { 0.0 };
                u64::from(super::clear_value::f32_to_f16(v))
            }
        }
    }
}

/// One field of a texel: the color channel it holds (0 red to 3 alpha), its lowest bit, and its
/// conversion.
#[derive(Clone, Copy, Debug)]
struct Field {
    channel: usize,
    at: u32,
    conversion: Conversion,
}

/// A texel of fields laid out one after another from bit 0, each `(channel, conversion)`.
fn packed(fields: &[(usize, Conversion)]) -> Vec<Field> {
    let mut at = 0;
    fields
        .iter()
        .map(|&(channel, conversion)| {
            let field = Field {
                channel,
                at,
                conversion,
            };
            at += conversion.width();
            field
        })
        .collect()
}

/// A texel of red, green, blue and alpha in that order, `n` of them, all converted alike, and
/// alpha (when `n` is 4) by `alpha`.
fn in_order(n: usize, conversion: Conversion, alpha: Conversion) -> Vec<Field> {
    let fields: Vec<(usize, Conversion)> = (0..n)
        .map(|c| (c, if c == 3 { alpha } else { conversion }))
        .collect();
    packed(&fields)
}

/// The fields of `pf`'s texel, `None` for a format with no packing here (the module docs) and
/// for `RGB9E5Float`, whose shared exponent [`shared_exponent`] packs.
fn fields(pf: PixelFormat) -> Option<Vec<Field>> {
    use Conversion as C;
    use PixelFormat as P;
    let same = |n, c| in_order(n, c, c);
    let xr = |srgb: bool, alpha: Option<u32>, colors: [u32; 3]| {
        let color = if srgb { C::XrSrgb } else { C::XrColor };
        // Blue, green and red from `colors`' bits, then alpha.
        let mut out: Vec<Field> = [2, 1, 0]
            .into_iter()
            .zip(colors)
            .map(|(channel, at)| Field {
                channel,
                at,
                conversion: color,
            })
            .collect();
        if let Some(at) = alpha {
            out.push(Field {
                channel: 3,
                at,
                conversion: C::XrAlpha,
            });
        }
        out
    };
    // Blue, green, red and alpha, each in the top 10 bits of a 16-bit word.
    let words = |conversion| {
        [(2, 6), (1, 22), (0, 38), (3, 54)]
            .into_iter()
            .map(|(channel, at)| Field {
                channel,
                at,
                conversion,
            })
            .collect()
    };
    Some(match pf {
        P::A8Unorm => packed(&[(3, C::Unorm(8))]),
        P::R8Unorm => same(1, C::Unorm(8)),
        P::R8UnormSRGB => same(1, C::Srgb),
        P::R8Snorm => same(1, C::Snorm(8)),
        P::R8Uint => same(1, C::Uint(8)),
        P::R8Sint => same(1, C::Sint(8)),
        P::R16Unorm => same(1, C::Unorm(16)),
        P::R16Snorm => same(1, C::Snorm(16)),
        P::R16Uint => same(1, C::Uint(16)),
        P::R16Sint => same(1, C::Sint(16)),
        P::R16Float => same(1, C::Half),
        P::R16Bfloat => same(1, C::Bfloat),
        P::RG8Unorm => same(2, C::Unorm(8)),
        P::RG8UnormSRGB => same(2, C::Srgb),
        P::RG8Snorm => same(2, C::Snorm(8)),
        P::RG8Uint => same(2, C::Uint(8)),
        P::RG8Sint => same(2, C::Sint(8)),
        P::B5G6R5Unorm => packed(&[(2, C::Unorm(5)), (1, C::Unorm(6)), (0, C::Unorm(5))]),
        P::A1BGR5Unorm => packed(&[
            (3, C::Unorm(1)),
            (2, C::Unorm(5)),
            (1, C::Unorm(5)),
            (0, C::Unorm(5)),
        ]),
        P::ABGR4Unorm => packed(&[
            (3, C::Unorm(4)),
            (2, C::Unorm(4)),
            (1, C::Unorm(4)),
            (0, C::Unorm(4)),
        ]),
        P::BGR5A1Unorm => packed(&[
            (2, C::Unorm(5)),
            (1, C::Unorm(5)),
            (0, C::Unorm(5)),
            (3, C::Unorm(1)),
        ]),
        P::R32Uint => same(1, C::Uint(32)),
        P::R32Sint => same(1, C::Sint(32)),
        P::R32Float => same(1, C::Float),
        P::RG16Unorm => same(2, C::Unorm(16)),
        P::RG16Snorm => same(2, C::Snorm(16)),
        P::RG16Uint => same(2, C::Uint(16)),
        P::RG16Sint => same(2, C::Sint(16)),
        P::RG16Float => same(2, C::Half),
        P::RG16Bfloat => same(2, C::Bfloat),
        P::RGBA8Unorm => same(4, C::Unorm(8)),
        P::RGBA8UnormSRGB => in_order(4, C::Srgb, C::Unorm(8)),
        P::RGBA8Snorm => same(4, C::Snorm(8)),
        P::RGBA8Uint => same(4, C::Uint(8)),
        P::RGBA8Sint => same(4, C::Sint(8)),
        P::BGRA8Unorm => packed(&[
            (2, C::Unorm(8)),
            (1, C::Unorm(8)),
            (0, C::Unorm(8)),
            (3, C::Unorm(8)),
        ]),
        P::BGRA8UnormSRGB => packed(&[(2, C::Srgb), (1, C::Srgb), (0, C::Srgb), (3, C::Unorm(8))]),
        P::ARGB8Unorm => packed(&[
            (3, C::Unorm(8)),
            (0, C::Unorm(8)),
            (1, C::Unorm(8)),
            (2, C::Unorm(8)),
        ]),
        P::RGB10A2Unorm => in_order(4, C::Unorm(10), C::Unorm(2)),
        P::RGB10A2Uint => in_order(4, C::Uint(10), C::Uint(2)),
        P::BGR10A2Unorm => packed(&[
            (2, C::Unorm(10)),
            (1, C::Unorm(10)),
            (0, C::Unorm(10)),
            (3, C::Unorm(2)),
        ]),
        P::RG11B10Float => packed(&[(0, C::Ufloat(6)), (1, C::Ufloat(6)), (2, C::Ufloat(5))]),
        P::RG32Uint => same(2, C::Uint(32)),
        P::RG32Sint => same(2, C::Sint(32)),
        P::RG32Float => same(2, C::Float),
        P::RGBA16Unorm => same(4, C::Unorm(16)),
        P::RGBA16Snorm => same(4, C::Snorm(16)),
        P::RGBA16Uint => same(4, C::Uint(16)),
        P::RGBA16Sint => same(4, C::Sint(16)),
        P::RGBA16Float => same(4, C::Half),
        P::RGBA16Bfloat => same(4, C::Bfloat),
        P::RGBA32Uint => same(4, C::Uint(32)),
        P::RGBA32Sint => same(4, C::Sint(32)),
        P::RGBA32Float => same(4, C::Float),
        P::RGBX8Unorm => in_order(4, C::Unorm(8), C::Zero(8)),
        P::RGBX8UnormSRGB => in_order(4, C::Srgb, C::Zero(8)),
        P::RGBX8Snorm => in_order(4, C::Snorm(8), C::Zero(8)),
        P::RGBX8Uint => in_order(4, C::Uint(8), C::Zero(8)),
        P::RGBX8Sint => in_order(4, C::Sint(8), C::Zero(8)),
        P::RGBX16Uint => in_order(4, C::Uint(16), C::Zero(16)),
        P::RGBX16Sint => in_order(4, C::Sint(16), C::Zero(16)),
        P::RGBX16Float => same(4, C::Half),
        P::RGBX32Uint => in_order(4, C::Uint(32), C::Zero(32)),
        P::RGBX32Sint => in_order(4, C::Sint(32), C::Zero(32)),
        P::RGBX32Float => same(4, C::Float),
        P::RGB8Unorm => same(3, C::Unorm(8)),
        P::RGB8Snorm => same(3, C::Snorm(8)),
        P::RGB8Uint => same(3, C::Uint(8)),
        P::RGB8Sint => same(3, C::Sint(8)),
        P::RGB16Unorm => same(3, C::Unorm(16)),
        P::RGB16Snorm => same(3, C::Snorm(16)),
        P::RGB16Uint => same(3, C::Uint(16)),
        P::RGB16Sint => same(3, C::Sint(16)),
        P::RGB16Float => same(3, C::Half),
        P::RGB16Bfloat => same(3, C::Bfloat),
        P::RGB32Uint => same(3, C::Uint(32)),
        P::RGB32Sint => same(3, C::Sint(32)),
        P::RGB32Float => same(3, C::Float),
        P::BGRA4Unorm => packed(&[
            (2, C::Unorm(4)),
            (1, C::Unorm(4)),
            (0, C::Unorm(4)),
            (3, C::Unorm(4)),
        ]),
        P::ARGB8UnormSRGB => packed(&[(3, C::Unorm(8)), (0, C::Srgb), (1, C::Srgb), (2, C::Srgb)]),
        P::RGBA16FloatXR => in_order(4, C::XrHalf, C::XrHalfAlpha),
        P::BGRA10Unorm => words(C::Unorm(10)),
        P::BGRA10Uint => words(C::Uint(10)),
        P::BGRA10XR => xr(false, Some(54), [6, 22, 38]),
        P::BGRA10XRSRGB => xr(true, Some(54), [6, 22, 38]),
        P::BGR10XR => xr(false, None, [0, 10, 20]),
        P::BGR10XRSRGB => xr(true, None, [0, 10, 20]),
        _ => return None,
    })
}

/// The bytes of the texel a color fill of `color` packed in `pf` writes (the module docs): the
/// texel's bytes, little-endian. `None` for a format with no packing here.
pub(super) fn fill_texel(pf: PixelFormat, color: [f64; 4]) -> Option<Vec<u8>> {
    let (bits, bytes) = if pf == PixelFormat::RGB9E5Float {
        (u128::from(shared_exponent(color)), 4)
    } else {
        let fields = fields(pf)?;
        let end = fields.iter().map(|f| f.at + f.conversion.width()).max()?;
        let bits = fields.iter().fold(0u128, |bits, f| {
            let mask = (1u128 << f.conversion.width()) - 1;
            bits | ((u128::from(f.conversion.code(color[f.channel])) & mask) << f.at)
        });
        (bits, end.div_ceil(8) as usize)
    };
    Some(bits.to_le_bytes()[..bytes].to_vec())
}

/// Whether a color fill of `pf` packs the color by a conversion a device's clear makes of a
/// `float` (the module docs): every field of the texel a normalized, sRGB, half, float or small
/// float one (an RGBX format's X field zeros), or the shared exponent's. A plain fill of a
/// texture of such a format that the device renders is the device's clear of the color
/// (`blit_fill.rs`). Not an integer format's, whose clear value is an integer no device converts,
/// nor a bfloat or extended-range one's, which no device clears from a `float`.
pub(super) fn device_converts(pf: PixelFormat) -> bool {
    use Conversion as C;
    pf == PixelFormat::RGB9E5Float
        || fields(pf).is_some_and(|fields| {
            fields.iter().all(|f| {
                matches!(
                    f.conversion,
                    C::Unorm(_)
                        | C::Snorm(_)
                        | C::Srgb
                        | C::Half
                        | C::Float
                        | C::Ufloat(_)
                        | C::Zero(_)
                )
            })
        })
}

/// Whether the device's clear of `color` for a texture of `pf` ([`device_converts`]) may store
/// another texel than Metal's fill packs (the module docs), announced `device-color-conversion`:
/// a normalized channel's value scaled anywhere off a whole code, where Vulkan lets the device
/// store either of the two codes closest to it (a tie among them, which the fill rounds up, half
/// away from zero signed), an sRGB channel's value inside `(0, 1)` (the fill takes the code whose
/// decoded value is nearest, the device encodes), and a half's, small float's or shared
/// exponent's value it does not hold exactly. A NaN is [`device_nan`]'s, or stores 0 on both.
pub(super) fn device_apart(pf: PixelFormat, color: [f64; 4]) -> bool {
    use Conversion as C;
    if pf == PixelFormat::RGB9E5Float {
        return !shared_exact(color);
    }
    let Some(fields) = fields(pf) else {
        return false;
    };
    fields.iter().any(|f| {
        let v = color[f.channel] as f32;
        if v.is_nan() {
            return false;
        }
        match f.conversion {
            C::Unorm(bits) | C::Snorm(bits) => {
                let (low, m) = match f.conversion {
                    C::Unorm(_) => (0.0, largest(bits) as f64),
                    _ => (-1.0, largest(bits - 1) as f64),
                };
                let r = f64::from(v.clamp(low, 1.0)) * m;
                r != r.round()
            }
            C::Srgb => v > 0.0 && v < 1.0,
            C::Half => {
                let held = super::clear_value::f16_to_f32(super::clear_value::f32_to_f16(v));
                held != v
            }
            C::Ufloat(mantissa) => {
                let largest = (2.0 - 1.0 / f64::from(1u32 << mantissa)) * 2f64.powi(15);
                let binade = super::clear_value::binade(v).max(-14);
                let spacing = 2f64.powi(binade) / f64::from(1u32 << mantissa);
                let held = v <= 0.0
                    || v == f32::INFINITY
                    || (f64::from(v) <= largest && (f64::from(v) / spacing).fract() == 0.0);
                !held
            }
            _ => false,
        }
    })
}

/// Whether `RGB9E5Float`'s texel of `color` is its red, green and blue exactly (NaN and negative
/// values 0): each a whole number of the shared exponent's spacing, the largest at most 65408.
fn shared_exact(color: [f64; 4]) -> bool {
    let values = [color[0], color[1], color[2]].map(|x| {
        let v = x as f32;
        if v.is_nan() {
            0.0
        } else {
            f64::from(v.max(0.0))
        }
    });
    let max = values.iter().copied().fold(0.0, f64::max);
    if max > 65408.0 {
        return false;
    }
    let spacing = 2f64.powi(super::clear_value::binade(max as f32).max(-16) + 1 - 9);
    values.iter().all(|v| (v / spacing).fract() == 0.0)
}

/// Whether the device's clear of `color` for a texture of `pf` ([`device_converts`]) converts a
/// NaN to a half or small float as the device likes (announced `device-nan-encoding`), where
/// Metal's fill keeps a half NaN's sign and payload and makes a small float's all ones. A
/// normalized or shared exponent's NaN stores 0 on both, and a `float`'s the bits the clear value
/// gives.
pub(super) fn device_nan(pf: PixelFormat, color: [f64; 4]) -> bool {
    fields(pf).is_some_and(|fields| {
        fields.iter().any(|f| {
            matches!(f.conversion, Conversion::Half | Conversion::Ufloat(_))
                && color[f.channel].is_nan()
        })
    })
}

/// The largest code of a field of `bits`.
fn largest(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

/// Whether the NaN `x` is quiet: its top mantissa bit set.
fn quiet(x: f64) -> bool {
    x.to_bits() & (1 << 51) != 0
}

/// An unsigned normalized code (the module docs).
fn unorm(x: f64, bits: u32) -> u64 {
    let v = x as f32;
    if v.is_nan() {
        return 0;
    }
    let v = v.clamp(0.0, 1.0);
    let m = largest(bits);
    // Exact: a `float` times a code of at most 8 bits fits a double, and so does the sum.
    let product = if matches!(bits, 4 | 8) {
        f64::from(v) * m as f64
    } else {
        f64::from(v * m as f32)
    };
    (product + 0.5).floor() as u64
}

/// A signed normalized code (the module docs), in `bits` two's complement.
fn snorm(x: f64, bits: u32) -> u64 {
    let v = x as f32;
    if v.is_nan() {
        return 0;
    }
    let m = largest(bits - 1) as f32;
    let product = f64::from(v.clamp(-1.0, 1.0) * m);
    let k = (product.abs() + 0.5).floor() as i64;
    let k = if product < 0.0 { -k } else { k };
    k as u64 & largest(bits)
}

/// An unsigned integer code (the module docs).
fn uint(x: f64, bits: u32) -> u64 {
    let max = largest(bits);
    if x.is_nan() {
        return if quiet(x) { 0 } else { max };
    }
    if x <= 0.0 {
        0
    } else if x >= max as f64 {
        max
    } else if bits == 32 && x > f64::from(i32::MAX) && x < 2147483648.0 {
        1 << 31
    } else {
        x.trunc() as u64
    }
}

/// A signed integer code (the module docs), in `bits` two's complement.
fn sint(x: f64, bits: u32) -> u64 {
    let max = largest(bits - 1) as i64;
    let min = -max - 1;
    let v = if x.is_nan() {
        if quiet(x) {
            min
        } else {
            max
        }
    } else if x >= max as f64 {
        max
    } else if x <= min as f64 {
        min
    } else {
        x.trunc() as i64
    };
    v as u64 & largest(bits)
}

/// A NaN's sign and the top `keep` bits of its payload, under the quiet NaN of a float of
/// `bits` bits whose exponent is `exponent` bits: the NaN that float stores (the module docs).
fn nan_bits(x: f64, bits: u32, exponent: u32) -> u64 {
    let d = x.to_bits();
    let mantissa = bits - 1 - exponent;
    let sign = (d >> 63) << (bits - 1);
    let ones = largest(exponent) << mantissa;
    let quiet = 1 << (mantissa - 1);
    sign | ones | quiet | ((d >> (52 - mantissa)) & largest(mantissa))
}

/// A half's code (the module docs).
fn half(x: f64) -> u64 {
    if x.is_nan() {
        return nan_bits(x, 16, 5);
    }
    u64::from(super::clear_value::f32_to_f16(x as f32))
}

/// A float's code (the module docs).
fn float(x: f64) -> u64 {
    if x.is_nan() {
        return nan_bits(x, 32, 8);
    }
    u64::from((x as f32).to_bits())
}

/// A bfloat's code (the module docs).
fn bfloat(x: f64) -> u64 {
    if x.is_nan() {
        return nan_bits(x, 16, 8);
    }
    // No sum passes `u32`: the largest non-NaN bits, `0xff80_0000`, plus `0x8000`.
    let b = (x as f32).to_bits();
    u64::from((b + 0x7fff + ((b >> 16) & 1)) >> 16)
}

/// A small unsigned float's code, of `mantissa` bits and a 5-bit exponent (the module docs).
fn ufloat(x: f64, mantissa: u32) -> u64 {
    let v = x as f32;
    let infinity = 31 << mantissa;
    if v.is_nan() {
        return infinity | largest(mantissa);
    }
    if v <= 0.0 {
        return 0;
    }
    if v.is_infinite() {
        return infinity;
    }
    let finite = infinity - 1;
    let v = f64::from(v);
    let steps = 1u64 << mantissa;
    if v < 2f64.powi(-14) {
        // Cut to a multiple of 2^-37, then to the nearest multiple of the subnormal step,
        // `2^(-14 - mantissa)`, ties to even. Exact: `v` is a `float`. A result of `steps` is
        // the smallest normal's code.
        let cut = (v * 2f64.powi(37)).floor();
        return (cut / 2f64.powi(23 - mantissa as i32)).round_ties_even() as u64;
    }
    let binade = super::clear_value::binade(v as f32);
    if binade > 15 {
        return finite;
    }
    // The nearest multiple of the binade's spacing, ties to even: `steps` to `2 steps`.
    let k = (v / 2f64.powi(binade - mantissa as i32)).round_ties_even() as u64;
    let (binade, k) = if k == 2 * steps {
        (binade + 1, steps)
    } else {
        (binade, k)
    };
    if binade > 15 {
        return finite;
    }
    (((binade + 15) as u64) << mantissa) | (k - steps)
}

/// `RGB9E5Float`'s texel (the module docs): red's, green's and blue's 9-bit mantissas from bit
/// 0, then the 5-bit exponent.
fn shared_exponent(color: [f64; 4]) -> u32 {
    const LARGEST: f32 = 65408.0;
    let values = [color[0], color[1], color[2]].map(|x| {
        let v = x as f32;
        if v.is_nan() {
            0.0
        } else {
            f64::from(v.clamp(0.0, LARGEST))
        }
    });
    let max = values.iter().copied().fold(0.0, f64::max);
    // The exponent `e` of a spacing of `2^(e - 9)`: the largest channel's binade plus one.
    let mut exponent = super::clear_value::binade(max as f32).max(-16) + 1;
    let half_up = |v: f64, exponent: i32| (v / 2f64.powi(exponent - 9) + 0.5).floor() as u32;
    if half_up(max, exponent) == 512 {
        exponent += 1;
    }
    let [r, g, b] = values.map(|v| half_up(v, exponent));
    r | (g << 9) | (b << 18) | (((exponent + 15) as u32) << 27)
}

/// The extended-range code 384 of the value 0.
const XR_ZERO: u64 = 384;
/// The least extended-range color value, `-384/510`, as a `float`.
const XR_LOW: f32 = (-384.0f64 / 510.0) as f32;
/// The largest extended-range color value, `639/510`, as a `float`.
const XR_HIGH: f32 = (639.0f64 / 510.0) as f32;

/// An extended-range code: `x`'s `float` clamped to `[low, high]` (NaN `low`), then
/// `⌊(v + 384/510) · 510 + ½⌋`, each step a `float` operation.
fn extended(x: f64, low: f32, high: f32) -> u64 {
    const OFFSET: f32 = (384.0f64 / 510.0) as f32;
    let v = x as f32;
    let v = if v.is_nan() { low } else { v.clamp(low, high) };
    ((v + OFFSET) * 510.0 + 0.5).floor() as u64
}

/// The code of `x`'s `float` under `first`, the least `float`'s bits storing each code above
/// the first (ascending in value): how many of them are at most it; `nan` for a NaN.
fn first_at_most(first: &[u32], x: f64, nan: u64) -> u64 {
    let v = x as f32;
    if v.is_nan() {
        return nan;
    }
    first.partition_point(|&t| f32::from_bits(t) <= v) as u64
}

/// The least `float` (its bits) a color fill of an 8-bit sRGB channel stores as each code from 1
/// to 255: entry `k` stores code `k + 1` (every code of `R8Unorm_sRGB`, the same
/// in each channel of `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and
/// `RGBX8Unorm_sRGB`). Each lies within a `float` step of the midpoint between the two codes'
/// decoded values: a fill stores the code whose decoded value is nearest the color.
const SRGB_FIRST: [u32; 255] = [
    0x391f22b4, 0x39eeb40e, 0x3a46eb61, 0x3a8b3e5e, 0x3ab3070b, 0x3adacfb8, 0x3b014c33, 0x3b153089,
    0x3b2914df, 0x3b3cf936, 0x3b511e78, 0x3b66218f, 0x3b7c5a8e, 0x3b89e3ee, 0x3b96370a, 0x3ba328da,
    0x3bb0bb93, 0x3bbef15d, 0x3bcdcc55, 0x3bdd4e91, 0x3bed7a19, 0x3bfe50ef, 0x3c07ea85, 0x3c11042c,
    0x3c1a7662, 0x3c244215, 0x3c2e6831, 0x3c38e9a0, 0x3c43c747, 0x3c4f0207, 0x3c5a9abf, 0x3c66924a,
    0x3c72e981, 0x3c7fa13a, 0x3c865d24, 0x3c8d1abe, 0x3c9409d1, 0x3c9b2ac4, 0x3ca27dfb, 0x3caa03d8,
    0x3cb1bcc0, 0x3cb9a912, 0x3cc1c92e, 0x3cca1d73, 0x3cd2a641, 0x3cdb63f3, 0x3ce456e5, 0x3ced7f74,
    0x3cf6ddf8, 0x3d003966, 0x3d051f24, 0x3d0a2062, 0x3d0f3d4b, 0x3d14760a, 0x3d19caca, 0x3d1f3bb6,
    0x3d24c8f7, 0x3d2a72b7, 0x3d30391e, 0x3d361c56, 0x3d3c1c88, 0x3d4239db, 0x3d487478, 0x3d4ecc85,
    0x3d55422a, 0x3d5bd58f, 0x3d6286da, 0x3d695630, 0x3d7043b9, 0x3d774f99, 0x3d7e79f7, 0x3d82e17c,
    0x3d869561, 0x3d8a58bb, 0x3d8e2b9e, 0x3d920e1c, 0x3d960047, 0x3d9a022f, 0x3d9e13e8, 0x3da23582,
    0x3da66710, 0x3daaa8a3, 0x3daefa4b, 0x3db35c1b, 0x3db7ce24, 0x3dbc5076, 0x3dc0e322, 0x3dc58638,
    0x3dca39ca, 0x3dcefde9, 0x3dd3d2a4, 0x3dd8b80d, 0x3dddae32, 0x3de2b525, 0x3de7ccf6, 0x3decf5b5,
    0x3df22f71, 0x3df77a3b, 0x3dfcd622, 0x3e01219b, 0x3e03e0c3, 0x3e06a891, 0x3e09790d, 0x3e0c523e,
    0x3e0f342c, 0x3e121edf, 0x3e15125e, 0x3e180eb0, 0x3e1b13de, 0x3e1e21ee, 0x3e2138e9, 0x3e2458d4,
    0x3e2781b9, 0x3e2ab39d, 0x3e2dee89, 0x3e313284, 0x3e347f94, 0x3e37d5c1, 0x3e3b3511, 0x3e3e9d8d,
    0x3e420f3c, 0x3e458a23, 0x3e490e4b, 0x3e4c9bb9, 0x3e503276, 0x3e53d288, 0x3e577bf6, 0x3e5b2ec7,
    0x3e5eeb02, 0x3e62b0ad, 0x3e667fcf, 0x3e6a586f, 0x3e6e3a93, 0x3e722643, 0x3e761b85, 0x3e7a1a5f,
    0x3e7e22d9, 0x3e811a7c, 0x3e832862, 0x3e853b22, 0x3e8752be, 0x3e896f3a, 0x3e8b9099, 0x3e8db6de,
    0x3e8fe20d, 0x3e921229, 0x3e944734, 0x3e968133, 0x3e98c028, 0x3e9b0416, 0x3e9d4d01, 0x3e9f9aec,
    0x3ea1edd9, 0x3ea445cc, 0x3ea6a2c8, 0x3ea904cf, 0x3eab6be6, 0x3eadd80f, 0x3eb0494d, 0x3eb2bfa4,
    0x3eb53b15, 0x3eb7bba4, 0x3eba4155, 0x3ebccc2a, 0x3ebf5c26, 0x3ec1f14c, 0x3ec48b9f, 0x3ec72b22,
    0x3ec9cfd8, 0x3ecc79c4, 0x3ecf28e9, 0x3ed1dd4a, 0x3ed496e9, 0x3ed755c9, 0x3eda19ee, 0x3edce35b,
    0x3edfb211, 0x3ee28614, 0x3ee55f67, 0x3ee83e0d, 0x3eeb2208, 0x3eee0b5c, 0x3ef0fa0a, 0x3ef3ee16,
    0x3ef6e784, 0x3ef9e655, 0x3efcea8b, 0x3efff42b, 0x3f01819c, 0x3f030bd9, 0x3f0498ce, 0x3f06287e,
    0x3f07bae9, 0x3f095011, 0x3f0ae7f7, 0x3f0c829c, 0x3f0e2002, 0x3f0fc02a, 0x3f116316, 0x3f1308c7,
    0x3f14b13d, 0x3f165c7b, 0x3f180a82, 0x3f19bb53, 0x3f1b6eef, 0x3f1d2558, 0x3f1ede8f, 0x3f209a96,
    0x3f22596d, 0x3f241b16, 0x3f25df92, 0x3f27a6e3, 0x3f29710a, 0x3f2b3e07, 0x3f2d0dde, 0x3f2ee08e,
    0x3f30b618, 0x3f328e7f, 0x3f3469c4, 0x3f3647e7, 0x3f3828ea, 0x3f3a0ccf, 0x3f3bf396, 0x3f3ddd41,
    0x3f3fc9d1, 0x3f41b947, 0x3f43aba5, 0x3f45a0ec, 0x3f47991c, 0x3f499438, 0x3f4b9240, 0x3f4d9336,
    0x3f4f971b, 0x3f519df0, 0x3f53a7b6, 0x3f55b46f, 0x3f57c41c, 0x3f59d6be, 0x3f5bec56, 0x3f5e04e5,
    0x3f60206d, 0x3f623eef, 0x3f64606c, 0x3f6684e5, 0x3f68ac5b, 0x3f6ad6d0, 0x3f6d0445, 0x3f6f34ba,
    0x3f716832, 0x3f739ead, 0x3f75d82d, 0x3f7814b2, 0x3f7a543e, 0x3f7c96d2, 0x3f7edc6f,
];

/// The least `float` (its bits) a color fill of an extended-range `_sRGB` color channel stores as
/// each code from 1 to 1023: entry `k` stores code `k + 1`, ascending from negative values (every
/// code of each color channel of `BGR10_XR_sRGB` and `BGRA10_XR_sRGB`, the
/// same in each).
const XR_SRGB_FIRST: [u32; 1023] = [
    0xbf068c95, 0xbf05c40e, 0xbf04fc37, 0xbf04350e, 0xbf036e93, 0xbf02a8c6, 0xbf01e3a7, 0xbf011f37,
    0xbf005b74, 0xbeff30be, 0xbefdabee, 0xbefc2879, 0xbefaa65e, 0xbef9259c, 0xbef7a634, 0xbef62825,
    0xbef4ab6e, 0xbef33010, 0xbef1b60a, 0xbef03d5c, 0xbeeec605, 0xbeed5005, 0xbeebdb5b, 0xbeea6808,
    0xbee8f60b, 0xbee78563, 0xbee61610, 0xbee4a812, 0xbee33b69, 0xbee1d014, 0xbee06612, 0xbedefd64,
    0xbedd9609, 0xbedc3001, 0xbedacb4b, 0xbed967e7, 0xbed805d5, 0xbed6a514, 0xbed545a3, 0xbed3e784,
    0xbed28ab5, 0xbed12f35, 0xbecfd505, 0xbece7c24, 0xbecd2492, 0xbecbce4e, 0xbeca7958, 0xbec925b0,
    0xbec7d355, 0xbec68247, 0xbec53286, 0xbec3e411, 0xbec296e7, 0xbec14b09, 0xbec00076, 0xbebeb72e,
    0xbebd6f31, 0xbebc287d, 0xbebae312, 0xbeb99ef1, 0xbeb85c19, 0xbeb71a89, 0xbeb5da42, 0xbeb49b42,
    0xbeb35d8a, 0xbeb22118, 0xbeb0e5ed, 0xbeafac08, 0xbeae7369, 0xbead3c10, 0xbeac05fc, 0xbeaad12c,
    0xbea99da1, 0xbea86b5a, 0xbea73a56, 0xbea60a96, 0xbea4dc18, 0xbea3aedc, 0xbea282e3, 0xbea1582c,
    0xbea02eb5, 0xbe9f0680, 0xbe9ddf8b, 0xbe9cb9d6, 0xbe9b9561, 0xbe9a722b, 0xbe995034, 0xbe982f7b,
    0xbe971001, 0xbe95f1c4, 0xbe94d4c5, 0xbe93b903, 0xbe929e7e, 0xbe918534, 0xbe906d26, 0xbe8f5654,
    0xbe8e40bd, 0xbe8d2c60, 0xbe8c193e, 0xbe8b0755, 0xbe89f6a6, 0xbe88e72f, 0xbe87d8f1, 0xbe86cbec,
    0xbe85c01e, 0xbe84b588, 0xbe83ac28, 0xbe82a3ff, 0xbe819d0c, 0xbe80974f, 0xbe7f2590, 0xbe7d1eeb,
    0xbe7b1aae, 0xbe7918da, 0xbe77196e, 0xbe751c67, 0xbe7321c6, 0xbe71298b, 0xbe6f33b3, 0xbe6d403f,
    0xbe6b4f2e, 0xbe69607d, 0xbe67742e, 0xbe658a3e, 0xbe63a2ad, 0xbe61bd7b, 0xbe5fdaa6, 0xbe5dfa2d,
    0xbe5c1c11, 0xbe5a404f, 0xbe5866e7, 0xbe568fd8, 0xbe54bb21, 0xbe52e8c2, 0xbe5118b9, 0xbe4f4b06,
    0xbe4d7fa8, 0xbe4bb69e, 0xbe49efe7, 0xbe482b83, 0xbe466970, 0xbe44a9ad, 0xbe42ec39, 0xbe413115,
    0xbe3f783e, 0xbe3dc1b4, 0xbe3c0d77, 0xbe3a5b85, 0xbe38abdd, 0xbe36fe7e, 0xbe355368, 0xbe33aa9a,
    0xbe320412, 0xbe305fd0, 0xbe2ebdd4, 0xbe2d1e1b, 0xbe2b80a5, 0xbe29e572, 0xbe284c81, 0xbe26b5cf,
    0xbe25215d, 0xbe238f2a, 0xbe21ff35, 0xbe20717d, 0xbe1ee600, 0xbe1d5cbe, 0xbe1bd5b6, 0xbe1a50e8,
    0xbe18ce52, 0xbe174df3, 0xbe15cfca, 0xbe1453d6, 0xbe12da17, 0xbe11628c, 0xbe0fed33, 0xbe0e7a0b,
    0xbe0d0915, 0xbe0b9a4e, 0xbe0a2db6, 0xbe08c34c, 0xbe075b0e, 0xbe05f4fd, 0xbe049116, 0xbe032f5a,
    0xbe01cfc6, 0xbe00725b, 0xbdfe2e2d, 0xbdfb7bf0, 0xbdf8cdfe, 0xbdf62453, 0xbdf37eef, 0xbdf0ddd0,
    0xbdee40f3, 0xbdeba856, 0xbde913f8, 0xbde683d6, 0xbde3f7ee, 0xbde1703f, 0xbddeecc7, 0xbddc6d83,
    0xbdd9f271, 0xbdd77b90, 0xbdd508dc, 0xbdd29a55, 0xbdd02ff9, 0xbdcdc9c4, 0xbdcb67b6, 0xbdc909cc,
    0xbdc6b004, 0xbdc45a5c, 0xbdc208d2, 0xbdbfbb63, 0xbdbd720e, 0xbdbb2cd1, 0xbdb8eba9, 0xbdb6ae95,
    0xbdb47591, 0xbdb2409d, 0xbdb00fb7, 0xbdade2db, 0xbdabba07, 0xbda9953b, 0xbda77472, 0xbda557ac,
    0xbda33ee7, 0xbda12a1f, 0xbd9f1952, 0xbd9d0c80, 0xbd9b03a5, 0xbd98febf, 0xbd96fdcb, 0xbd9500c9,
    0xbd9307b5, 0xbd91128d, 0xbd8f214f, 0xbd8d33f9, 0xbd8b4a89, 0xbd8964fc, 0xbd878350, 0xbd85a582,
    0xbd83cb91, 0xbd81f57a, 0xbd80233b, 0xbd7ca9a3, 0xbd791477, 0xbd7586ec, 0xbd7200fe, 0xbd6e82a8,
    0xbd6b0be6, 0xbd679cb3, 0xbd64350a, 0xbd60d4e7, 0xbd5d7c44, 0xbd5a2b1d, 0xbd56e16d, 0xbd539f2f,
    0xbd50645f, 0xbd4d30f7, 0xbd4a04f3, 0xbd46e04d, 0xbd43c301, 0xbd40ad0a, 0xbd3d9e63, 0xbd3a9707,
    0xbd3796f1, 0xbd349e1c, 0xbd31ac82, 0xbd2ec21f, 0xbd2bdeee, 0xbd2902e9, 0xbd262e0c, 0xbd236051,
    0xbd2099b3, 0xbd1dda2d, 0xbd1b21ba, 0xbd187054, 0xbd15c5f7, 0xbd13229c, 0xbd10863f, 0xbd0df0db,
    0xbd0b6269, 0xbd08dae4, 0xbd065a48, 0xbd03e08f, 0xbd016db2, 0xbcfe035c, 0xbcf938f8, 0xbcf47c2d,
    0xbcefccf0, 0xbceb2b37, 0xbce696f5, 0xbce21020, 0xbcdd96ad, 0xbcd92a8f, 0xbcd4cbbc, 0xbcd07a28,
    0xbccc35c7, 0xbcc7fe8e, 0xbcc3d471, 0xbcbfb764, 0xbcbba75c, 0xbcb7a44d, 0xbcb3ae2a, 0xbcafc4e7,
    0xbcabe87a, 0xbca818d5, 0xbca455ec, 0xbca09fb4, 0xbc9cf61e, 0xbc995920, 0xbc95c8ad, 0xbc9244b8,
    0xbc8ecd35, 0xbc8b6216, 0xbc880350, 0xbc84b0d5, 0xbc816a98, 0xbc7c6118, 0xbc760548, 0xbc6fc1a7,
    0xbc699619, 0xbc638283, 0xbc5d86cb, 0xbc57a2d6, 0xbc51d688, 0xbc4c21c5, 0xbc468473, 0xbc40fe76,
    0xbc3b8fb0, 0xbc363807, 0xbc30f75e, 0xbc2bcd98, 0xbc26ba98, 0xbc21be42, 0xbc1cd877, 0xbc18091a,
    0xbc13500f, 0xbc0ead36, 0xbc0a2071, 0xbc05a9a3, 0xbc0148ac, 0xbbf9fada, 0xbbf18f8f, 0xbbe94f37,
    0xbbe13993, 0xbbd94e63, 0xbbd18d66, 0xbbc9f65b, 0xbbc288ff, 0xbbbb4512, 0xbbb42729, 0xbbad348f,
    0xbba66f39, 0xbb9fce5c, 0xbb995597, 0xbb9304a4, 0xbb8cdb3a, 0xbb86d912, 0xbb80fde3, 0xbb7692c8,
    0xbb6b7694, 0xbb60a693, 0xbb56222e, 0xbb4be477, 0xbb41f257, 0xbb380028, 0xbb2e0df7, 0xbb241bc8,
    0xbb1a29a8, 0xbb103777, 0xbb064548, 0xbaf8a650, 0xbae4c1ef, 0xbad0dd90, 0xbabcf92f, 0xbaa914ef,
    0xba953090, 0xba814c2f, 0xba5acfdf, 0xba330720, 0xba0b3e5f, 0xb9c6ebbf, 0xb96eb480, 0xb89f22ff,
    0x389f2300, 0x396eb380, 0x39c6eb41, 0x3a0b3e60, 0x3a3306e0, 0x3a5acfa1, 0x3a814c30, 0x3a953070,
    0x3aa914d1, 0x3abcf930, 0x3ad0dd91, 0x3ae4c1d1, 0x3af8a630, 0x3b064549, 0x3b103769, 0x3b1a2998,
    0x3b241bc9, 0x3b2e0df8, 0x3b380018, 0x3b41f249, 0x3b4be478, 0x3b56222f, 0x3b60a694, 0x3b6b7695,
    0x3b7692c9, 0x3b80fde4, 0x3b86d913, 0x3b8cdb3b, 0x3b9304a5, 0x3b995598, 0x3b9fce5d, 0x3ba66f3a,
    0x3bad3490, 0x3bb4272a, 0x3bbb4513, 0x3bc28900, 0x3bc9f65c, 0x3bd18d67, 0x3bd94e64, 0x3be13994,
    0x3be94f38, 0x3bf18f90, 0x3bf9fadb, 0x3c0148ad, 0x3c05a9a4, 0x3c0a2072, 0x3c0ead37, 0x3c135010,
    0x3c18091b, 0x3c1cd878, 0x3c21be43, 0x3c26ba99, 0x3c2bcd99, 0x3c30f75f, 0x3c363808, 0x3c3b8fb1,
    0x3c40fe77, 0x3c468474, 0x3c4c21c6, 0x3c51d689, 0x3c57a2d7, 0x3c5d86cc, 0x3c638284, 0x3c69961a,
    0x3c6fc1a8, 0x3c760549, 0x3c7c6119, 0x3c816a99, 0x3c84b0d6, 0x3c880351, 0x3c8b6217, 0x3c8ecd36,
    0x3c9244b9, 0x3c95c8ae, 0x3c995921, 0x3c9cf61f, 0x3ca09fb5, 0x3ca455ed, 0x3ca818d6, 0x3cabe87b,
    0x3cafc4e8, 0x3cb3ae2b, 0x3cb7a44e, 0x3cbba75d, 0x3cbfb765, 0x3cc3d472, 0x3cc7fe8f, 0x3ccc35c8,
    0x3cd07a29, 0x3cd4cbbd, 0x3cd92a90, 0x3cdd96ae, 0x3ce21021, 0x3ce696f6, 0x3ceb2b38, 0x3cefccf1,
    0x3cf47c2e, 0x3cf938f9, 0x3cfe035d, 0x3d016db3, 0x3d03e090, 0x3d065a49, 0x3d08dae5, 0x3d0b626a,
    0x3d0df0dc, 0x3d108640, 0x3d13229d, 0x3d15c5f8, 0x3d187055, 0x3d1b21bb, 0x3d1dda2e, 0x3d2099b4,
    0x3d236052, 0x3d262e0d, 0x3d2902ea, 0x3d2bdeef, 0x3d2ec220, 0x3d31ac83, 0x3d349e1d, 0x3d3796f2,
    0x3d3a9708, 0x3d3d9e64, 0x3d40ad0b, 0x3d43c302, 0x3d46e04e, 0x3d4a04f4, 0x3d4d30f8, 0x3d506460,
    0x3d539f30, 0x3d56e16e, 0x3d5a2b1e, 0x3d5d7c45, 0x3d60d4e8, 0x3d64350b, 0x3d679cb4, 0x3d6b0be7,
    0x3d6e82a9, 0x3d7200ff, 0x3d7586ed, 0x3d791478, 0x3d7ca9a4, 0x3d80233c, 0x3d81f57b, 0x3d83cb92,
    0x3d85a583, 0x3d878351, 0x3d8964fd, 0x3d8b4a8a, 0x3d8d33fa, 0x3d8f2150, 0x3d91128e, 0x3d9307b6,
    0x3d9500ca, 0x3d96fdcc, 0x3d98fec0, 0x3d9b03a6, 0x3d9d0c81, 0x3d9f1953, 0x3da12a20, 0x3da33ee8,
    0x3da557ad, 0x3da77473, 0x3da9953c, 0x3dabba08, 0x3dade2dc, 0x3db00fb8, 0x3db2409e, 0x3db47592,
    0x3db6ae96, 0x3db8ebaa, 0x3dbb2cd2, 0x3dbd720f, 0x3dbfbb64, 0x3dc208d3, 0x3dc45a5d, 0x3dc6b005,
    0x3dc909cd, 0x3dcb67b7, 0x3dcdc9c5, 0x3dd02ffa, 0x3dd29a56, 0x3dd508dd, 0x3dd77b91, 0x3dd9f272,
    0x3ddc6d84, 0x3ddeecc8, 0x3de17040, 0x3de3f7ef, 0x3de683d7, 0x3de913f9, 0x3deba857, 0x3dee40f4,
    0x3df0ddd1, 0x3df37ef0, 0x3df62454, 0x3df8cdff, 0x3dfb7bf1, 0x3dfe2e2e, 0x3e00725c, 0x3e01cfc7,
    0x3e032f5b, 0x3e049117, 0x3e05f4fe, 0x3e075b0f, 0x3e08c34d, 0x3e0a2db7, 0x3e0b9a4f, 0x3e0d0916,
    0x3e0e7a0c, 0x3e0fed34, 0x3e11628d, 0x3e12da18, 0x3e1453d7, 0x3e15cfcb, 0x3e174df4, 0x3e18ce53,
    0x3e1a50e9, 0x3e1bd5b7, 0x3e1d5cbf, 0x3e1ee601, 0x3e20717e, 0x3e21ff36, 0x3e238f2b, 0x3e25215e,
    0x3e26b5d0, 0x3e284c82, 0x3e29e573, 0x3e2b80a6, 0x3e2d1e1c, 0x3e2ebdd5, 0x3e305fd1, 0x3e320413,
    0x3e33aa9b, 0x3e355369, 0x3e36fe7f, 0x3e38abde, 0x3e3a5b86, 0x3e3c0d78, 0x3e3dc1b5, 0x3e3f783f,
    0x3e413116, 0x3e42ec3a, 0x3e44a9ae, 0x3e466971, 0x3e482b84, 0x3e49efe8, 0x3e4bb69f, 0x3e4d7fa9,
    0x3e4f4b07, 0x3e5118ba, 0x3e52e8c3, 0x3e54bb22, 0x3e568fd9, 0x3e5866e8, 0x3e5a4050, 0x3e5c1c12,
    0x3e5dfa2e, 0x3e5fdaa7, 0x3e61bd7c, 0x3e63a2ae, 0x3e658a3f, 0x3e67742f, 0x3e69607e, 0x3e6b4f2f,
    0x3e6d4040, 0x3e6f33b4, 0x3e71298c, 0x3e7321c7, 0x3e751c68, 0x3e77196f, 0x3e7918db, 0x3e7b1aaf,
    0x3e7d1eec, 0x3e7f2591, 0x3e809750, 0x3e819d0d, 0x3e82a400, 0x3e83ac29, 0x3e84b589, 0x3e85c01f,
    0x3e86cbed, 0x3e87d8f2, 0x3e88e730, 0x3e89f6a7, 0x3e8b0756, 0x3e8c193f, 0x3e8d2c61, 0x3e8e40be,
    0x3e8f5655, 0x3e906d27, 0x3e918535, 0x3e929e7f, 0x3e93b904, 0x3e94d4c6, 0x3e95f1c5, 0x3e971002,
    0x3e982f7c, 0x3e995035, 0x3e9a722c, 0x3e9b9562, 0x3e9cb9d7, 0x3e9ddf8c, 0x3e9f0681, 0x3ea02eb6,
    0x3ea1582d, 0x3ea282e4, 0x3ea3aedd, 0x3ea4dc19, 0x3ea60a97, 0x3ea73a57, 0x3ea86b5b, 0x3ea99da2,
    0x3eaad12d, 0x3eac05fd, 0x3ead3c11, 0x3eae736a, 0x3eafac09, 0x3eb0e5ee, 0x3eb22119, 0x3eb35d8b,
    0x3eb49b43, 0x3eb5da43, 0x3eb71a8a, 0x3eb85c1a, 0x3eb99ef2, 0x3ebae313, 0x3ebc287e, 0x3ebd6f32,
    0x3ebeb72f, 0x3ec00077, 0x3ec14b0a, 0x3ec296e8, 0x3ec3e412, 0x3ec53287, 0x3ec68248, 0x3ec7d356,
    0x3ec925b1, 0x3eca7959, 0x3ecbce4f, 0x3ecd2493, 0x3ece7c25, 0x3ecfd506, 0x3ed12f36, 0x3ed28ab6,
    0x3ed3e785, 0x3ed545a4, 0x3ed6a515, 0x3ed805d6, 0x3ed967e8, 0x3edacb4c, 0x3edc3002, 0x3edd960a,
    0x3edefd65, 0x3ee06613, 0x3ee1d015, 0x3ee33b6a, 0x3ee4a813, 0x3ee61611, 0x3ee78564, 0x3ee8f60c,
    0x3eea6809, 0x3eebdb5c, 0x3eed5006, 0x3eeec606, 0x3ef03d5d, 0x3ef1b60b, 0x3ef33011, 0x3ef4ab6f,
    0x3ef62826, 0x3ef7a635, 0x3ef9259d, 0x3efaa65f, 0x3efc287a, 0x3efdabef, 0x3eff30bf, 0x3f005b75,
    0x3f011f38, 0x3f01e3a8, 0x3f02a8c7, 0x3f036e94, 0x3f04350f, 0x3f04fc38, 0x3f05c40f, 0x3f068c96,
    0x3f0755cb, 0x3f081fb0, 0x3f08ea44, 0x3f09b587, 0x3f0a817a, 0x3f0b4e1d, 0x3f0c1b6f, 0x3f0ce972,
    0x3f0db825, 0x3f0e8788, 0x3f0f579c, 0x3f102861, 0x3f10f9d7, 0x3f11cbfe, 0x3f129ed6, 0x3f13725f,
    0x3f14469b, 0x3f151b88, 0x3f15f127, 0x3f16c778, 0x3f179e7b, 0x3f187630, 0x3f194e99, 0x3f1a27b4,
    0x3f1b0182, 0x3f1bdc03, 0x3f1cb738, 0x3f1d9320, 0x3f1e6fbb, 0x3f1f4d0a, 0x3f202b0d, 0x3f2109c5,
    0x3f21e930, 0x3f22c950, 0x3f23aa24, 0x3f248bae, 0x3f256dec, 0x3f2650df, 0x3f273487, 0x3f2818e5,
    0x3f28fdf9, 0x3f29e3c1, 0x3f2aca40, 0x3f2bb175, 0x3f2c9960, 0x3f2d8201, 0x3f2e6b59, 0x3f2f5568,
    0x3f30402d, 0x3f312ba9, 0x3f3217dc, 0x3f3304c7, 0x3f33f26a, 0x3f34e0c3, 0x3f35cfd5, 0x3f36bf9e,
    0x3f37b020, 0x3f38a15a, 0x3f39934c, 0x3f3a85f7, 0x3f3b795a, 0x3f3c6d76, 0x3f3d624c, 0x3f3e57da,
    0x3f3f4e22, 0x3f404524, 0x3f413cdf, 0x3f423554, 0x3f432e83, 0x3f44286c, 0x3f45230f, 0x3f461e6c,
    0x3f471a84, 0x3f481757, 0x3f4914e5, 0x3f4a132e, 0x3f4b1232, 0x3f4c11f1, 0x3f4d126c, 0x3f4e13a3,
    0x3f4f1595, 0x3f501844, 0x3f511bae, 0x3f521fd4, 0x3f5324b8, 0x3f542a58, 0x3f5530b4, 0x3f5637cd,
    0x3f573fa3, 0x3f584837, 0x3f595188, 0x3f5a5b96, 0x3f5b6662, 0x3f5c71eb, 0x3f5d7e33, 0x3f5e8b39,
    0x3f5f98fc, 0x3f60a77f, 0x3f61b6c0, 0x3f62c6bf, 0x3f63d77e, 0x3f64e8fb, 0x3f65fb38, 0x3f670e33,
    0x3f6821ee, 0x3f693669, 0x3f6a4ba3, 0x3f6b619e, 0x3f6c7858, 0x3f6d8fd2, 0x3f6ea80d, 0x3f6fc108,
    0x3f70dac4, 0x3f71f540, 0x3f73107e, 0x3f742c7c, 0x3f75493c, 0x3f7666be, 0x3f778500, 0x3f78a404,
    0x3f79c3ca, 0x3f7ae452, 0x3f7c059c, 0x3f7d27a8, 0x3f7e4a77, 0x3f7f6e08, 0x3f80492e, 0x3f80dbb9,
    0x3f816ea6, 0x3f8201f4, 0x3f8295a4, 0x3f8329b6, 0x3f83be29, 0x3f8452fe, 0x3f84e835, 0x3f857dce,
    0x3f8613c9, 0x3f86aa26, 0x3f8740e6, 0x3f87d807, 0x3f886f8b, 0x3f890772, 0x3f899fbb, 0x3f8a3866,
    0x3f8ad174, 0x3f8b6ae5, 0x3f8c04b8, 0x3f8c9eee, 0x3f8d3987, 0x3f8dd483, 0x3f8e6fe2, 0x3f8f0ba5,
    0x3f8fa7ca, 0x3f904453, 0x3f90e13f, 0x3f917e8e, 0x3f921c41, 0x3f92ba58, 0x3f9358d2, 0x3f93f7af,
    0x3f9496f1, 0x3f953696, 0x3f95d69f, 0x3f96770c, 0x3f9717dd, 0x3f97b912, 0x3f985aab, 0x3f98fca9,
    0x3f999f0a, 0x3f9a41d1, 0x3f9ae4fc, 0x3f9b888b, 0x3f9c2c7e, 0x3f9cd0d7, 0x3f9d7594, 0x3f9e1ab6,
    0x3f9ec03d, 0x3f9f6628, 0x3fa00c79, 0x3fa0b32e, 0x3fa15a49, 0x3fa201c9, 0x3fa2a9ae, 0x3fa351f9,
    0x3fa3faa9, 0x3fa4a3be, 0x3fa54d39, 0x3fa5f71a, 0x3fa6a160, 0x3fa74c0c, 0x3fa7f71e, 0x3fa8a296,
    0x3fa94e73, 0x3fa9fab7, 0x3faaa761, 0x3fab5471, 0x3fac01e7, 0x3facafc3, 0x3fad5e05, 0x3fae0cae,
    0x3faebbbe, 0x3faf6b34, 0x3fb01b10, 0x3fb0cb54, 0x3fb17bfe, 0x3fb22d0f, 0x3fb2de87, 0x3fb39066,
    0x3fb442ac, 0x3fb4f558, 0x3fb5a86c, 0x3fb65be8, 0x3fb70fca, 0x3fb7c414, 0x3fb878c5, 0x3fb92dde,
    0x3fb9e35e, 0x3fba9946, 0x3fbb4f95, 0x3fbc064d, 0x3fbcbd6c, 0x3fbd74f3, 0x3fbe2ce2, 0x3fbee538,
    0x3fbf9df7, 0x3fc0571e, 0x3fc110ae, 0x3fc1caa5, 0x3fc28505, 0x3fc33fcd, 0x3fc3fafe, 0x3fc4b697,
    0x3fc57299, 0x3fc62f04, 0x3fc6ebd7, 0x3fc7a913, 0x3fc866b8, 0x3fc924c5, 0x3fc9e33c, 0x3fcaa21c,
    0x3fcb6165, 0x3fcc2117, 0x3fcce133, 0x3fcda1b7, 0x3fce62a5, 0x3fcf23fd, 0x3fcfe5bd, 0x3fd0a7e8,
    0x3fd16a7c, 0x3fd22d79, 0x3fd2f0e1, 0x3fd3b4b2, 0x3fd478ee, 0x3fd53d93, 0x3fd602a2,
];
