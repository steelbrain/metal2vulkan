//! Metal's color resolve arithmetic over every input it measured, on the device the cfg selects,
//! under the validation layer.
//!
//! Metal probes (an Apple M3 Max and an M4, macOS 26.7, byte-identical) resolved every pair of
//! 8-bit samples and hundreds of thousands of 2- and 4-sample texels per format, and closed each
//! format's average as one rule (every case measured):
//!
//! - a unorm channel of `b` bits: `(sum + n / 2) / n` over the integer codes (8, 10, 16 and the
//!   2-bit alpha);
//! - a snorm channel: `floor(sum / n + 1 / 2)` over the codes, `-2^(b-1)` read as
//!   `-(2^(b-1) - 1)` (both are -1.0);
//! - an sRGB channel: each code's linear value as Metal's 12-bit table holds it, their integer
//!   average `(sum + n / 2) / n`, and Metal's encode of that 12-bit value
//!   (`bind::SRGB_RESOLVE_LINEAR`, `bind::SRGB_RESOLVE_THRESHOLDS`);
//! - a 32-bit float channel: the samples summed in order in `f32`, each input, partial sum and the
//!   result times `1 / n` a denormal flushed to the zero of its sign, and a NaN anywhere the
//!   canonical `0x7fc00000`.
//!
//! A probe of an M3 Max under macOS 27.0.1 (our own `resolve.swift`) confirmed those and closed the
//! rest at 2 and 4 samples (`bind::ColorResolveFormat`'s module docs state each rule):
//!
//! - a packed 16-bit unorm channel: each code read through the `half` nearest its value, times the
//!   code's scale rounded toward zero to a `half`, the average of those rounded half up;
//! - a `half` channel: the samples floored to a grid 14 bits below their largest exponent (at 4,
//!   samples 0 and 1 first, as a two's complement adder normalizes their sum) and summed, the
//!   average rounded to a `half` with a tie toward +∞, a NaN counting 0 and an infinity 65536;
//! - an `RG11B10Float` channel: the `half` rule, then that `half` to nearest even on the channel's
//!   grid, held to its largest finite value unless infinite;
//! - an `RGB9E5Float` texel: the `half` rule per channel, then `EXT_texture_shared_exponent`'s
//!   encode.
//!
//! The rules built on the `half` rule are not applied past 4 samples (the runtime resolves those by
//! the device's average), so their formats are compared at 2 and 4 only.
//!
//! Each case draws a multisampled texture sample by sample (the owned fixture `rt6_samples`'s
//! fill, each sample's code as the float that encodes it exactly), reads every sample back
//! through an unsigned view (`candidate::runtime_samples`, the oracle's readback), resolves it in
//! a pass of no draws that loads it, and compares every channel of the resolve texture with the
//! rule applied to the samples read back: every pair of codes of an 8-bit channel at 2 samples,
//! every pair as `(a, a, b, b)` and random texels at 4, and random texels with the edge values of
//! a wider channel. The device's average resolves every one of these formats in the pass,
//! announced where it is not measured to be Metal's, and [`the_device_resolve_rounds_as_announced`]
//! measures it against the rules. Metal has no texture of more than 4 samples on Apple silicon, so
//! a resolve of 8 is compared with the same rules and printed as unmeasured, never as Metal's. A
//! device without `imageViewFormatReinterpretation` (MoltenVK) views no texel as an unsigned format
//! of other components, so the formats only such a view reads back are printed as unread there. The
//! runs before the runtime resolved so (at 2b41d71d) measured each driver's own average, which
//! `DriverFact::ColorResolveAverageIsMetals` records.

use super::resolves::{lane, quiet, Lane};
use crate::candidate::runtime_samples::{encode_through, library, shape};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::runtime_lane::buffer_with;
use metal2vulkan::bind::{SRGB_RESOLVE_LINEAR, SRGB_RESOLVE_THRESHOLDS};
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::format::WordAttachment;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// How a channel's bits encode its value.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Channel {
    Unorm(u32),
    /// A packed 16-bit format's unorm channel, which Metal reads through a `half`.
    HalfUnorm(u32),
    Snorm(u32),
    Srgb,
    Float,
    Half,
    /// An unsigned float of 5 exponent bits and this many mantissa bits.
    SmallFloat(u32),
    /// A 9-bit mantissa of the texel's shared exponent at bits 27 to 31.
    Shared,
}

impl Channel {
    fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::HalfUnorm(bits) | Channel::Snorm(bits) => bits,
            Channel::Srgb => 8,
            Channel::Float => 32,
            Channel::Half => 16,
            Channel::SmallFloat(mantissa) => mantissa + 5,
            Channel::Shared => 9,
        }
    }

    /// Whether Metal's rule of the channel holds at `n` samples: one built on the `half` rule at
    /// 2 and 4 only.
    fn resolves(self, n: u32) -> bool {
        match self {
            Channel::Half | Channel::SmallFloat(_) | Channel::Shared => n == 2 || n == 4,
            _ => true,
        }
    }
}

/// A format the probes closed: its texel bytes and each channel's bit offset in the texel, its
/// component (0 to 3 for R, G, B, A) and its encoding.
struct Case {
    pf: PixelFormat,
    bytes: usize,
    channels: &'static [(u32, usize, Channel)],
}

const U8: Channel = Channel::Unorm(8);
const S8: Channel = Channel::Snorm(8);
const U16: Channel = Channel::Unorm(16);
const S16: Channel = Channel::Snorm(16);
const U10: Channel = Channel::Unorm(10);
const U2: Channel = Channel::Unorm(2);
const SRGB: Channel = Channel::Srgb;
const F32: Channel = Channel::Float;
const F16: Channel = Channel::Half;
const H1: Channel = Channel::HalfUnorm(1);
const H4: Channel = Channel::HalfUnorm(4);
const H5: Channel = Channel::HalfUnorm(5);
const H6: Channel = Channel::HalfUnorm(6);
const F11: Channel = Channel::SmallFloat(6);
const F10: Channel = Channel::SmallFloat(5);
const E9: Channel = Channel::Shared;

const fn case(pf: PixelFormat, bytes: usize, channels: &'static [(u32, usize, Channel)]) -> Case {
    Case {
        pf,
        bytes,
        channels,
    }
}

const CASES: &[Case] = &[
    case(PixelFormat::R8Unorm, 1, &[(0, 0, U8)]),
    case(PixelFormat::RG8Unorm, 2, &[(0, 0, U8), (8, 1, U8)]),
    case(
        PixelFormat::RGBA8Unorm,
        4,
        &[(0, 0, U8), (8, 1, U8), (16, 2, U8), (24, 3, U8)],
    ),
    case(
        PixelFormat::BGRA8Unorm,
        4,
        &[(0, 2, U8), (8, 1, U8), (16, 0, U8), (24, 3, U8)],
    ),
    case(PixelFormat::R8UnormSRGB, 1, &[(0, 0, SRGB)]),
    case(PixelFormat::RG8UnormSRGB, 2, &[(0, 0, SRGB), (8, 1, SRGB)]),
    case(
        PixelFormat::RGBA8UnormSRGB,
        4,
        &[(0, 0, SRGB), (8, 1, SRGB), (16, 2, SRGB), (24, 3, U8)],
    ),
    case(
        PixelFormat::BGRA8UnormSRGB,
        4,
        &[(0, 2, SRGB), (8, 1, SRGB), (16, 0, SRGB), (24, 3, U8)],
    ),
    case(PixelFormat::R8Snorm, 1, &[(0, 0, S8)]),
    case(PixelFormat::RG8Snorm, 2, &[(0, 0, S8), (8, 1, S8)]),
    case(
        PixelFormat::RGBA8Snorm,
        4,
        &[(0, 0, S8), (8, 1, S8), (16, 2, S8), (24, 3, S8)],
    ),
    case(PixelFormat::R16Unorm, 2, &[(0, 0, U16)]),
    case(PixelFormat::RG16Unorm, 4, &[(0, 0, U16), (16, 1, U16)]),
    case(
        PixelFormat::RGBA16Unorm,
        8,
        &[(0, 0, U16), (16, 1, U16), (32, 2, U16), (48, 3, U16)],
    ),
    case(PixelFormat::R16Snorm, 2, &[(0, 0, S16)]),
    case(PixelFormat::RG16Snorm, 4, &[(0, 0, S16), (16, 1, S16)]),
    case(
        PixelFormat::RGBA16Snorm,
        8,
        &[(0, 0, S16), (16, 1, S16), (32, 2, S16), (48, 3, S16)],
    ),
    case(
        PixelFormat::RGB10A2Unorm,
        4,
        &[(0, 0, U10), (10, 1, U10), (20, 2, U10), (30, 3, U2)],
    ),
    case(
        PixelFormat::BGR10A2Unorm,
        4,
        &[(0, 2, U10), (10, 1, U10), (20, 0, U10), (30, 3, U2)],
    ),
    case(PixelFormat::R32Float, 4, &[(0, 0, F32)]),
    case(PixelFormat::RG32Float, 8, &[(0, 0, F32), (32, 1, F32)]),
    case(
        PixelFormat::RGBA32Float,
        16,
        &[(0, 0, F32), (32, 1, F32), (64, 2, F32), (96, 3, F32)],
    ),
    case(PixelFormat::R16Float, 2, &[(0, 0, F16)]),
    case(PixelFormat::RG16Float, 4, &[(0, 0, F16), (16, 1, F16)]),
    case(
        PixelFormat::RGBA16Float,
        8,
        &[(0, 0, F16), (16, 1, F16), (32, 2, F16), (48, 3, F16)],
    ),
    // Metal's packed formats are named from the least significant bit.
    case(
        PixelFormat::B5G6R5Unorm,
        2,
        &[(11, 0, H5), (5, 1, H6), (0, 2, H5)],
    ),
    case(
        PixelFormat::A1BGR5Unorm,
        2,
        &[(11, 0, H5), (6, 1, H5), (1, 2, H5), (0, 3, H1)],
    ),
    case(
        PixelFormat::ABGR4Unorm,
        2,
        &[(12, 0, H4), (8, 1, H4), (4, 2, H4), (0, 3, H4)],
    ),
    case(
        PixelFormat::BGR5A1Unorm,
        2,
        &[(10, 0, H5), (5, 1, H5), (0, 2, H5), (15, 3, H1)],
    ),
    case(
        PixelFormat::RG11B10Float,
        4,
        &[(0, 0, F11), (11, 1, F11), (22, 2, F10)],
    ),
    case(
        PixelFormat::RGB9E5Float,
        4,
        &[(0, 0, E9), (9, 1, E9), (18, 2, E9)],
    ),
];

impl Case {
    /// Whether the case's texels share an exponent (`RGB9E5Float`'s, at bits 27 to 31).
    fn shared(&self) -> bool {
        self.channels.iter().any(|&(_, _, c)| c == Channel::Shared)
    }

    /// The most codes the device's average of a channel of a code may stand from Metal's rule
    /// (`device-rounded-resolve`'s bound): two in a texel of sRGB channels (NVIDIA's at 4
    /// samples), one otherwise.
    fn device_codes(&self) -> u64 {
        match self.channels.iter().any(|&(_, _, c)| c == Channel::Srgb) {
            true => 2,
            false => 1,
        }
    }

    /// Whether Metal's rules of every channel hold at `n` samples.
    fn resolves(&self, n: u32) -> bool {
        self.channels.iter().all(|&(_, _, c)| c.resolves(n))
    }
}

/// How a case's samples are read back: the unsigned view, the fixture's kernel that packs a texel
/// of it into words, and the readback's texel size. A view of the texel's own components where the
/// fixture packs one, else the unsigned format of the texel's size, which needs
/// `imageViewFormatReinterpretation`.
fn readback_view(case: &Case) -> (PixelFormat, &'static str, ReadbackView) {
    use PixelFormat as P;
    match case.pf {
        P::R8Unorm | P::R8UnormSRGB | P::R8Snorm => {
            (P::R8Uint, "rt6_read_words1", ReadbackView::R8Uint)
        }
        P::RGBA8Unorm | P::RGBA8UnormSRGB | P::RGBA8Snorm | P::BGRA8Unorm | P::BGRA8UnormSRGB => {
            (P::RGBA8Uint, "rt6_read_bytes4", ReadbackView::R32Uint)
        }
        P::R16Unorm | P::R16Snorm => (P::R16Uint, "rt6_read_words1", ReadbackView::R16Uint),
        P::RG16Unorm | P::RG16Snorm => (P::RG16Uint, "rt6_read_halves2", ReadbackView::R32Uint),
        P::RGBA16Unorm | P::RGBA16Snorm | P::RGBA16Float => {
            (P::RGBA16Uint, "rt6_read_halves4", ReadbackView::Rg32Uint)
        }
        P::R16Float => (P::R16Uint, "rt6_read_words1", ReadbackView::R16Uint),
        P::RG16Float => (P::RG16Uint, "rt6_read_halves2", ReadbackView::R32Uint),
        P::R32Float => (P::R32Uint, "rt6_read_words1", ReadbackView::R32Uint),
        P::RG32Float => (P::RG32Uint, "rt6_read_words2", ReadbackView::Rg32Uint),
        P::RGBA32Float => (P::RGBA32Uint, "rt6_read_words4", ReadbackView::Rgba32Uint),
        _ => match case.bytes {
            2 => (P::R16Uint, "rt6_read_words1", ReadbackView::R16Uint),
            _ => (P::R32Uint, "rt6_read_words1", ReadbackView::R32Uint),
        },
    }
}

/// The float a fragment writes for `code` of `channel`: the value that encodes it exactly (an
/// sRGB code's linear value, a snorm code below -1.0 as -1.0, a float's own bits, a small float's
/// value). A shared exponent's mantissa needs its texel's exponent ([`shared_value`]).
fn encoding(channel: Channel, code: u64) -> f32 {
    match channel {
        Channel::Unorm(bits) | Channel::HalfUnorm(bits) => {
            (code as f64 / ((1u64 << bits) - 1) as f64) as f32
        }
        Channel::Half => half_value(code as u32) as f32,
        Channel::SmallFloat(mantissa) => half_value((code as u32) << (10 - mantissa)) as f32,
        Channel::Shared => panic!("a shared exponent's mantissa is no value alone"),
        Channel::Snorm(bits) => {
            let half = 1i64 << (bits - 1);
            let v = code as i64;
            let v = if v >= half { v - 2 * half } else { v };
            (v.max(-(half - 1)) as f64 / (half - 1) as f64) as f32
        }
        Channel::Srgb => {
            let c = code as f64 / 255.0;
            let linear = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
            linear as f32
        }
        Channel::Float => f32::from_bits(code as u32),
    }
}

/// The value of a shared-exponent texel's mantissa `code` under its exponent field `exponent`.
fn shared_value(code: u64, exponent: u64) -> f64 {
    code as f64 * pow2(exponent as i32 - 24)
}

/// 2^k, for a `k` of a normal `f64`.
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The exponent of a nonzero finite `f64` (its leading bit's), from its bits.
fn exponent(x: f64) -> i32 {
    ((x.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// The value of a half's bits: its NaNs and infinities too.
fn half_value(h: u32) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = f64::from(h & 0x3ff);
    sign * match e {
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        0 => m * pow2(-24),
        _ => (1024.0 + m) * pow2(e - 25),
    }
}

/// SplitMix64, the probes' generator.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Float bits that reach every rule: denormals, zeros, infinities, NaNs, the largest finite, and
/// pairs of close values whose sum is a denormal.
fn float_bits(r: &mut SplitMix) -> u64 {
    let pick = r.next();
    let sign = (pick >> 8) & 1;
    let mantissa = r.next() & 0x7f_ffff;
    let bits = match pick % 16 {
        0 => mantissa.max(1),               // a denormal
        1 => 0,                             // zero
        2 => 0x7f80_0000,                   // infinity
        3 => 0x7f80_0000 | mantissa.max(1), // a NaN
        4 => 0x7f7f_ffff - (r.next() % 4),  // near the largest finite
        5 => 0x0080_0000 + (r.next() % 64), // near the smallest normal
        6 => (1 << 23) | mantissa,          // the smallest normal binade
        _ => r.next() % 0x7f80_0000,        // finite
    };
    (sign << 31) | bits
}

/// Half bits that reach every case of the half rule: any, the infinities, NaNs, the largest,
/// subnormals and zeros.
fn half_bits(r: &mut SplitMix) -> u64 {
    let pick = r.next();
    let sign = (pick >> 8) & 1;
    let bits = match pick % 12 {
        0 => [0x7c00, 0x7e00, 0x7c01, 0x7fff][(r.next() % 4) as usize],
        1 => 0x7bff - r.next() % 4,
        2 => r.next() % 0x800,
        3 => 0,
        _ => r.next() % 0x7c00,
    };
    (sign << 15) | bits
}

/// Small-float bits of `mantissa` bits that reach every case of the small-float rule: any, the
/// infinity, NaNs, the largest finite values, subnormals and zero.
fn small_float_bits(r: &mut SplitMix, mantissa: u32) -> u64 {
    let inf = 31u64 << mantissa;
    match r.next() % 10 {
        0 => inf,
        1 => inf | (r.next() & ((1 << mantissa) - 1)).max(1),
        2 => inf - 1 - r.next() % 4,
        3 => r.next() & ((1 << mantissa) - 1),
        4 => 0,
        _ => r.next() % inf,
    }
}

/// The value of channel `channel` of sample `s` of pixel `(x, y)` among `n`, on an image `width`
/// wide: an 8-bit or narrower channel takes every pair `(x, y)` (2 samples), and `(a, a, b, b)`
/// over every pair in the image's first 256 rows at more; the rest is random, wider integer
/// channels drawing their edge values a quarter of the time, floats every class.
fn value(
    channel: Channel,
    index: usize,
    (x, y): (u32, u32),
    (s, n): (u32, u32),
    r: &mut SplitMix,
) -> u64 {
    let bits = channel.bits();
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let spread = |v: u32| -> u64 {
        // A bijection of the codes per channel, so channels differ.
        (u64::from(v).wrapping_mul(2 * index as u64 + 1) + 53 * index as u64) & mask
    };
    if bits <= 8 && y < 256 {
        let pick = if n == 2 {
            [x, y][s as usize]
        } else if s < n / 2 {
            x
        } else {
            y
        };
        return spread(pick) & mask;
    }
    match channel {
        Channel::Float => float_bits(r),
        Channel::Half => half_bits(r),
        Channel::SmallFloat(mantissa) => small_float_bits(r, mantissa),
        _ if bits > 8 && r.next().is_multiple_of(4) => {
            let half = 1u64 << (bits - 1);
            let edges = [0, 1, half - 1, half, half + 1, mask - 1, mask];
            edges[(r.next() % edges.len() as u64) as usize]
        }
        _ => r.next() & mask,
    }
}

/// Metal's resolve of one channel's `samples` (the module docs).
fn metal(channel: Channel, samples: &[u64]) -> u64 {
    let n = samples.len() as i64;
    match channel {
        Channel::Unorm(_) => {
            let sum: i64 = samples.iter().map(|&s| s as i64).sum();
            ((sum + n / 2) / n) as u64
        }
        Channel::Snorm(bits) => {
            let half = 1i64 << (bits - 1);
            let sum: i64 = samples
                .iter()
                .map(|&s| {
                    let v = s as i64;
                    let v = if v >= half { v - 2 * half } else { v };
                    v.max(-(half - 1))
                })
                .sum();
            let resolved = (2 * sum + n).div_euclid(2 * n);
            (resolved as u64) & ((1u64 << bits) - 1)
        }
        Channel::Srgb => {
            let sum: i64 = samples
                .iter()
                .map(|&s| i64::from(SRGB_RESOLVE_LINEAR[s as usize]))
                .sum();
            let linear = ((sum + n / 2) / n) as u16;
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear) as u64
        }
        Channel::Float => {
            let flush = |v: f32| {
                if v.is_subnormal() {
                    0.0f32.copysign(v)
                } else {
                    v
                }
            };
            let mut sum = flush(f32::from_bits(samples[0] as u32));
            for &s in &samples[1..] {
                sum = flush(sum + flush(f32::from_bits(s as u32)));
            }
            let resolved = flush(sum * (1.0 / n as f32));
            if resolved.is_nan() {
                0x7fc0_0000
            } else {
                u64::from(resolved.to_bits())
            }
        }
        Channel::HalfUnorm(bits) => half_unorm_rule(bits, samples),
        Channel::Half => {
            let halves: Vec<u32> = samples.iter().map(|&s| s as u32).collect();
            u64::from(half_rule(&halves))
        }
        Channel::SmallFloat(mantissa) => small_rule(mantissa, samples),
        Channel::Shared => panic!("a shared exponent resolves by texel ([`shared_rule`])"),
    }
}

/// Metal's resolve of a unorm channel of `bits` bits read through a `half` (the module docs): each
/// code `c` of `m` becomes the `half` nearest `c / m` (ties to even), times `m` rounded toward zero
/// to a `half`; the average of those rounds half up. In `f64`, where `c / m` lies far from every
/// tie of the `half` grid (`m` is odd) and each other step is exact.
fn half_unorm_rule(bits: u32, samples: &[u64]) -> u64 {
    let m = ((1u64 << bits) - 1) as f64;
    let sum: f64 = samples
        .iter()
        .map(|&c| {
            let v = c as f64 / m;
            if v == 0.0 {
                return 0.0;
            }
            let step = pow2(exponent(v) - 10);
            let product = (v / step).round_ties_even() * step * m;
            let step = pow2(exponent(product) - 10);
            (product / step).floor() * step
        })
        .sum();
    (sum / samples.len() as f64 + 0.5).floor() as u64
}

/// A half's value in units of 2^-24, exactly: a NaN 0, an infinity 2^16 of its sign.
fn half_units(h: u32) -> i64 {
    let sign = if h & 0x8000 != 0 { -1 } else { 1 };
    let e = (h >> 10) & 0x1f;
    let m = i64::from(h & 0x3ff);
    sign * match (e, m) {
        (31, 0) => 1 << 40,
        (31, _) => 0,
        (0, _) => m,
        _ => (1024 + m) << (e - 1),
    }
}

/// The half's bits of `v` units of 2^-24 on the `half` grid: 2^16 and more an infinity, a zero +0.
fn half_of_units(v: i64) -> u32 {
    let sign = if v < 0 { 0x8000 } else { 0 };
    let a = v.unsigned_abs();
    if a >= 1 << 40 {
        return sign | 0x7c00;
    }
    if a == 0 {
        return 0;
    }
    if a < 1024 {
        return sign | a as u32;
    }
    let e = 64 - a.leading_zeros() - 11;
    sign | ((e + 1) << 10) | ((a >> e) as u32 - 1024)
}

/// The exponent of `v` units of 2^-28 (its leading bit's), -100 for zero.
fn fine_exponent(v: i64) -> i32 {
    if v == 0 {
        return -100;
    }
    63 - v.unsigned_abs().leading_zeros() as i32 - 28
}

/// `v` units of 2^-28 floored to the multiple of 2^(e - 14).
fn floored(v: i64, e: i32) -> i64 {
    let step = 1i64 << (e + 14);
    v.div_euclid(step) * step
}

/// Metal's 2- and 4-sample half resolve (the module docs) in exact integers, units of 2^-28: each
/// half's value (a NaN 0, an infinity 2^16 of its sign) and exponent (at least -14); at 2 the two
/// floored to the larger exponent's grid and summed; at 4 samples 0 and 1 so summed into `A`,
/// whose exponent is its leading bit's (one lower for a negative power of two) but at most the
/// larger of theirs, and `A` and samples 2 and 3 floored to the largest exponent's grid and
/// summed; the average rounded to the `half` grid with a tie toward +∞. The half's bits.
fn half_rule(halves: &[u32]) -> u32 {
    let fine: Vec<(i64, i32)> = halves
        .iter()
        .map(|&h| {
            let v = half_units(h) * 16;
            (v, fine_exponent(v).max(-14))
        })
        .collect();
    let total = match *fine.as_slice() {
        [(a, ea), (b, eb)] => {
            let e = ea.max(eb);
            floored(a, e) + floored(b, e)
        }
        [(a, ea), (b, eb), (c, ec), (d, ed)] => {
            let pair = ea.max(eb);
            let first = floored(a, pair) + floored(b, pair);
            let mut e_first = fine_exponent(first);
            if first < 0 && first.unsigned_abs().is_power_of_two() {
                e_first -= 1;
            }
            let e = e_first.min(pair).max(ec).max(ed);
            floored(first, e) + floored(c, e) + floored(d, e)
        }
        _ => panic!(
            "a half's rule holds at 2 and 4 samples, not {}",
            halves.len()
        ),
    };
    let n = halves.len() as i64;
    // The grid's step in units of 2^-28: 2^-24 below 2^-13, else the binade's.
    let mut step = 16i64;
    while total.abs() >= n * (step << 11) {
        step <<= 1;
    }
    let q = (2 * total + n * step).div_euclid(2 * n * step);
    half_of_units(q * step / 16)
}

/// Metal's resolve of a small float of `mantissa` bits (the module docs): the samples' half
/// resolve (a small float's bits are a half's leading bits), an infinity infinite, else that half
/// rounded to nearest even onto the small float's grid and held to its largest finite value. The
/// small float's bits.
fn small_rule(mantissa: u32, samples: &[u64]) -> u64 {
    let halves: Vec<u32> = samples
        .iter()
        .map(|&c| (c as u32) << (10 - mantissa))
        .collect();
    let h = half_rule(&halves);
    if h == 0x7c00 {
        return 31 << mantissa;
    }
    let v = half_value(h);
    if v == 0.0 {
        return 0;
    }
    let m = mantissa as i32;
    let step = pow2(exponent(v).max(-14) - m);
    let v = ((v / step).round_ties_even() * step).min((2.0 - pow2(-m)) * 32768.0);
    let e = exponent(v);
    if e < -14 {
        return (v / pow2(-14 - m)) as u64;
    }
    (((e + 15) as u64) << mantissa) | ((v / pow2(e - m)) as u64 - (1 << mantissa))
}

/// The three channels (R, G, B) of a shared-exponent texel `word`.
fn shared_values(word: u64) -> [f64; 3] {
    [0, 9, 18].map(|at| shared_value((word >> at) & 0x1ff, (word >> 27) & 0x1f))
}

/// Metal's resolve of shared-exponent texels `words` (the module docs): each channel's half
/// resolve, then `EXT_texture_shared_exponent`'s encode. The values the resolved texel holds.
fn shared_rule(words: &[u64]) -> [f64; 3] {
    let resolved = [0, 1, 2].map(|c| {
        // A channel's value in units of 2^-24 is its mantissa shifted by its exponent field.
        let halves: Vec<u32> = words
            .iter()
            .map(|&w| half_of_units((((w >> (9 * c)) & 0x1ff) << ((w >> 27) & 0x1f)) as i64))
            .collect();
        half_value(half_rule(&halves)).clamp(0.0, 65408.0)
    });
    let most = resolved[0].max(resolved[1]).max(resolved[2]);
    let mut e = if most > 0.0 {
        exponent(most).max(-16)
    } else {
        -16
    } + 16;
    if (most / pow2(e - 24) + 0.5).floor() == 512.0 {
        e += 1;
    }
    let step = pow2(e - 24);
    resolved.map(|v| (v / step + 0.5).floor() * step)
}

/// The bits of `texel` (little-endian) at `offset`, `bits` wide.
fn field(texel: &[u8], offset: u32, bits: u32) -> u64 {
    let mut word = 0u128;
    for (i, &b) in texel.iter().enumerate() {
        word |= u128::from(b) << (8 * i);
    }
    ((word >> offset) as u64)
        & if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
}

/// `texel`'s bytes with `value` at `offset`, `bits` wide.
fn set_field(texel: &mut [u8], offset: u32, bits: u32, value: u64) {
    for bit in 0..bits {
        let at = offset + bit;
        let byte = &mut texel[(at / 8) as usize];
        let mask = 1u8 << (at % 8);
        if (value >> bit) & 1 == 1 {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }
}

/// What one case's resolve gave against Metal's rule.
struct Outcome {
    channels: usize,
    differ: usize,
    /// Samples the fill did not store as drawn (a float the device changed on the way).
    changed: usize,
    /// Of the channels of a code (unorm, snorm, sRGB, packed unorm) that differ, those above
    /// Metal's code and those below it, and the most codes between the two.
    above: usize,
    below: usize,
    most_codes: u64,
    examples: Vec<String>,
}

/// The width and height of a case's texture at `n` samples.
fn case_extent(n: u32) -> [u32; 2] {
    [256, if n == 2 { 256 } else { 512 }]
}

/// Draw a texture of `case` at `n` samples ([`case_extent`]), read its samples back, resolve it,
/// and compare.
fn resolve_case(lane: &Lane, case: &Case, n: u32) -> Result<Outcome, String> {
    resolve_case_into(lane, case, n, case)
}

/// [`resolve_case`] resolving into a texture of `into`, a format of `case`'s channel layout (its
/// sRGB twin), whose channels' rules the resolve is compared with.
fn resolve_case_into(lane: &Lane, case: &Case, n: u32, into: &Case) -> Result<Outcome, String> {
    let [width, height] = case_extent(n);
    let mut r = SplitMix(0x5250_4c00 ^ (u64::from(n) << 32) ^ case.pf as u64);
    let pixels = (width * height) as usize;
    let texels = pixels * n as usize;
    // Each sample's codes, as drawn, and the float4 that draws them.
    let mut drawn = vec![0u8; texels * case.bytes];
    let mut floats = Vec::with_capacity(texels * 16);
    for y in 0..height {
        for x in 0..width {
            let mut sample_floats = vec![[0f32; 4]; n as usize];
            // A shared-exponent texel's exponent field, each sample's.
            let exponents: Vec<u64> = (0..n)
                .map(|_| if case.shared() { r.next() % 32 } else { 0 })
                .collect();
            for (index, &(offset, component, channel)) in case.channels.iter().enumerate() {
                for (s, lanes) in sample_floats.iter_mut().enumerate() {
                    let code = value(channel, index, (x, y), (s as u32, n), &mut r);
                    let at = (((y * width + x) * n) as usize + s) * case.bytes;
                    let texel = &mut drawn[at..at + case.bytes];
                    set_field(texel, offset, channel.bits(), code);
                    lanes[component] = match channel {
                        Channel::Shared => {
                            set_field(texel, 27, 5, exponents[s]);
                            shared_value(code, exponents[s]) as f32
                        }
                        _ => encoding(channel, code),
                    };
                }
            }
            for lanes in sample_floats {
                for lane in lanes {
                    floats.extend_from_slice(&lane.to_le_bytes());
                }
            }
        }
    }
    let device = &lane.device;
    let mut descriptor = TextureDescriptor::texture_2d(case.pf, width.into(), height.into(), false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = n.into();
    descriptor.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::private();
    let msaa = device
        .new_texture(&descriptor)
        .map_err(|e| format!("new_texture: {e}"))?;
    let (view_pf, kernel, readback) = readback_view(case);
    let view = msaa
        .new_texture_view(view_pf)
        .map_err(|e| format!("unread: no {view_pf:?} view ({e})"))?;
    let library = library(device).map_err(|e| e.to_string())?;
    let function = |name: &str| {
        library
            .new_function_with_name(name)
            .map(|f| f.function_descriptor())
            .map_err(|e| format!("{name}: {e}"))
    };
    let mut pipeline =
        RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")?));
    pipeline.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")?));
    pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(case.pf));
    pipeline.raster_sample_count = n;
    let pipeline = device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .map_err(|e| format!("the fill's pipeline: {e}"))?;
    let values = buffer_with(device, &floats).map_err(|e| e.to_string())?;
    let shape = shape(device, width, n).map_err(|e| e.to_string())?;
    let mut single = TextureDescriptor::texture_2d(into.pf, width.into(), height.into(), false);
    single.usage = TextureUsage::RENDER_TARGET;
    single.resource_options = ResourceOptions::private();
    let resolved: Texture = device
        .new_texture(&single)
        .map_err(|e| format!("new_texture: {e}"))?;
    let command_buffer = lane.queue.command_buffer().map_err(|e| e.to_string())?;
    let mut fill = RenderPassDescriptor::new();
    let target = &mut fill.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = LoadAction::DontCare;
    target.store_action = StoreAction::Store;
    let mut encoder = command_buffer
        .render_command_encoder(&fill)
        .map_err(|e| format!("the fill pass: {e}"))?;
    let draw = |e: metal2vulkan_runtime::Error| format!("the fill: {e}");
    encoder.set_render_pipeline_state(&pipeline).map_err(draw)?;
    encoder
        .set_fragment_buffer(Some(&values), 0, 0)
        .map_err(draw)?;
    encoder
        .set_fragment_buffer(Some(&shape), 0, 1)
        .map_err(draw)?;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .map_err(draw)?;
    encoder.end_encoding().map_err(draw)?;
    let samples = encode_through(
        device,
        &command_buffer,
        &view,
        kernel,
        SampleRead::Color(readback),
        [width, height],
        n,
    )
    .map_err(|e| format!("the readback: {e}"))?;
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = LoadAction::Load;
    target.store_action = StoreAction::MultisampleResolve;
    target.resolve_texture = Some(resolved.clone());
    let encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(|e| format!("the resolve pass: {e}"))?;
    encoder.end_encoding().map_err(|e| e.to_string())?;
    command_buffer.commit().map_err(|e| e.to_string())?;
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .map_err(|e| format!("{e:?}"))?;
    if let Some(error) = command_buffer.error() {
        return Err(format!("the command buffer failed: {error:?}"));
    }
    let stored = samples.bytes().map_err(|e| e.to_string())?;
    let got = lane.read(&resolved, (0, 0), case.bytes as u64, BlitOption::None);
    let mut outcome = Outcome {
        channels: 0,
        differ: 0,
        changed: 0,
        above: 0,
        below: 0,
        most_codes: 0,
        examples: Vec::new(),
    };
    for p in 0..pixels {
        let at = p * case.bytes;
        if case.shared() {
            compare_shared(&mut outcome, (&drawn, &stored, &got), p, n as usize);
            continue;
        }
        for (&(offset, _, drawn_as), &(_, _, channel)) in case.channels.iter().zip(into.channels) {
            outcome.channels += 1;
            let bits = drawn_as.bits();
            let field_of = |texels: &[u8], s: usize| {
                let at = (p * n as usize + s) * case.bytes;
                field(&texels[at..at + case.bytes], offset, bits)
            };
            let samples: Vec<u64> = (0..n as usize).map(|s| field_of(&stored, s)).collect();
            outcome.changed += (0..n as usize)
                .filter(|&s| field_of(&drawn, s) != samples[s])
                .count();
            let device_value = field(&got[at..at + case.bytes], offset, bits);
            let metal_value = metal(channel, &samples);
            if device_value != metal_value {
                outcome.differ += 1;
                if let Some((device_code, metal_code)) = codes(channel, device_value, metal_value) {
                    match device_code > metal_code {
                        true => outcome.above += 1,
                        false => outcome.below += 1,
                    }
                    outcome.most_codes = outcome.most_codes.max(device_code.abs_diff(metal_code));
                }
                if outcome.examples.len() < 4 {
                    let shown: Vec<String> = samples.iter().map(|v| format!("{v:#x}")).collect();
                    outcome.examples.push(format!(
                        "[{}] at bit {offset}: {device_value:#x}, Metal's rule {metal_value:#x}",
                        shown.join(", ")
                    ));
                }
            }
        }
    }
    Ok(outcome)
}

/// A channel's device and Metal values as the codes they order by, where `channel` is one of a
/// code (a snorm channel's two's complement read signed); `None` for a float channel.
fn codes(channel: Channel, device: u64, metal: u64) -> Option<(i64, i64)> {
    match channel {
        Channel::Unorm(_) | Channel::HalfUnorm(_) | Channel::Srgb => {
            Some((device as i64, metal as i64))
        }
        Channel::Snorm(bits) => {
            let signed = |v: u64| ((v << (64 - bits)) as i64) >> (64 - bits);
            Some((signed(device), signed(metal)))
        }
        Channel::Float | Channel::Half | Channel::SmallFloat(_) | Channel::Shared => None,
    }
}

/// Compare pixel `p`'s shared-exponent texel of 4 bytes, `n` samples each in `drawn` and
/// `stored`, resolved in `got`, with Metal's rule by value: the encode may write one value in
/// more than one way (a zero under any exponent).
fn compare_shared(
    outcome: &mut Outcome,
    (drawn, stored, got): (&[u8], &[u8], &[u8]),
    p: usize,
    n: usize,
) {
    let word = |texels: &[u8], i: usize| field(&texels[4 * i..4 * i + 4], 0, 32);
    let words: Vec<u64> = (0..n).map(|s| word(stored, p * n + s)).collect();
    outcome.changed += (0..n)
        .filter(|&s| shared_values(word(drawn, p * n + s)) != shared_values(words[s]))
        .count();
    let device_values = shared_values(word(got, p));
    let metal_values = shared_rule(&words);
    outcome.channels += 3;
    let differ = (0..3)
        .filter(|&c| device_values[c] != metal_values[c])
        .count();
    outcome.differ += differ;
    if differ != 0 && outcome.examples.len() < 4 {
        let shown: Vec<String> = words.iter().map(|w| format!("{w:#x}")).collect();
        outcome.examples.push(format!(
            "[{}]: {device_values:?}, Metal's rule {metal_values:?}",
            shown.join(", ")
        ));
    }
}

/// Every closed format at every sample count the device takes resolves by the device's
/// `AVERAGE` in the pass, but one the device renders through a word view
/// (`metal::format::WordAttachment`, as the NVIDIA RTX 5080 Laptop renders `R8Unorm_sRGB`),
/// whose codes the crate-owned resolve averages: `shader-resolve` is announced, once, exactly
/// where such a format resolved, and `device-rounded-resolve` exactly where the device's average
/// is not measured to be Metal's (`DriverFact::ColorResolveAverageIsMetals`, at 2 and 4 samples),
/// where every channel is Metal's. Elsewhere what the device's average gives against Metal's rule
/// is printed, a measurement: the channels that differ, those of a code above and below Metal's,
/// and the most codes between the two, held at 2 and 4 samples to the bound
/// `device-rounded-resolve` states ([`Case::device_codes`]; a float channel's differences are
/// counted, not bounded).
#[test]
fn the_device_resolve_rounds_as_announced() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let description = &lane.device.properties().description;
    eprintln!(
        "the device's resolve on {name}: driver {:?}, driverVersion {:#x}, vendor {:#x}",
        description.driver_id, description.driver_version, description.vendor_id
    );
    let metals = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::ColorResolveAverageIsMetals);
    messages.take_notices();
    let mut differ = Vec::new();
    let mut past_bound = Vec::new();
    let mut resolved_past_metal = false;
    // A format the device renders through a word view (`WordAttachment::of`), whose codes no
    // device's average reads, resolved: the crate-owned resolve (`color_resolve::shader_rules`).
    let profile = lane.device.compile_device().profile();
    let mut worded = Vec::new();
    for n in [2u32, 4, 8] {
        let supported = lane
            .device
            .properties()
            .supports_texture_sample_count(n.into())
            .unwrap();
        if !supported {
            eprintln!("the device's resolve on {name}: no texture of {n} samples");
            continue;
        }
        for case in CASES {
            let pf = case.pf;
            match resolve_case(&lane, case, n) {
                Err(why) => eprintln!("the device's resolve {pf:?} x{n} on {name}: {why}"),
                Ok(outcome) => {
                    resolved_past_metal |= n > 4;
                    if WordAttachment::of(pf, profile).is_some() {
                        worded.push(format!("{pf:?} x{n}"));
                    }
                    if outcome.differ != 0 && n <= 4 {
                        differ.push(format!("{pf:?} x{n}: {} channels", outcome.differ));
                    }
                    if outcome.most_codes > case.device_codes() && n <= 4 {
                        past_bound.push(format!("{pf:?} x{n}: {} codes", outcome.most_codes));
                    }
                    eprintln!(
                        "the device's resolve {pf:?} x{n} on {name}: {} of {} channels differ \
                         from Metal's rule{} ({} above Metal's code, {} below, by {} codes at \
                         most; {} samples stored otherwise than drawn){}",
                        outcome.differ,
                        outcome.channels,
                        if case.resolves(n) {
                            ""
                        } else {
                            " (applied unmeasured)"
                        },
                        outcome.above,
                        outcome.below,
                        outcome.most_codes,
                        outcome.changed,
                        if outcome.examples.is_empty() {
                            String::new()
                        } else {
                            format!(": {}", outcome.examples.join("; "))
                        }
                    );
                }
            }
        }
    }
    let counts = super::degrade_notices::counts(&messages.take_notices());
    eprintln!("the device's resolve on {name}: announced {counts:?}");
    eprintln!("the device's resolve on {name}: through a word view {worded:?}");
    assert_eq!(
        counts.get(&Degradation::ShaderResolve),
        (!worded.is_empty()).then_some(&1),
        "shader-resolve on {name}"
    );
    // At more than 4 samples no device's average is measured to be Metal's.
    let rounded = !metals || resolved_past_metal;
    assert_eq!(
        counts.contains_key(&Degradation::DeviceRoundedResolve),
        rounded,
        "device-rounded-resolve on {name}: {counts:?}"
    );
    if metals {
        assert!(
            differ.is_empty(),
            "the resolve of a driver measured to average as Metal does, not by Metal's rules on \
             {name}: {differ:#?}"
        );
    }
    assert!(
        past_bound.is_empty(),
        "the device's resolve past device-rounded-resolve's codes on {name}: {past_bound:#?}"
    );
    quiet(lane, &messages);
}

/// A resolve of a 4x or 2x `RGBA8Unorm` or `BGRA8Unorm` attachment into its sRGB twin, the
/// crate-owned resolve on every device (no subpass resolve writes the twin's encoding), takes the
/// device's own sRGB decode and encode, every device having those formats
/// (`bind::ColorResolveFormat::device_srgb`): it announces `device-rounded-resolve`, and every
/// channel stands within the codes the note states of Metal's rule ([`metal`]: its 12-bit
/// tables), over every pair of codes at 2 samples and the lane's pattern at 4 ([`value`]). What
/// the device gives against Metal's rule is printed, a measurement.
#[test]
fn a_twin_resolve_into_srgb_rounds_as_announced() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let case = |pf: PixelFormat| CASES.iter().find(|c| c.pf == pf).unwrap();
    messages.take_notices();
    let mut past_bound = Vec::new();
    for n in [2u32, 4] {
        let supported = lane
            .device
            .properties()
            .supports_texture_sample_count(n.into())
            .unwrap();
        if !supported {
            eprintln!("the twin resolve on {name}: no texture of {n} samples");
            continue;
        }
        for (from, into) in [
            (PixelFormat::RGBA8Unorm, PixelFormat::RGBA8UnormSRGB),
            (PixelFormat::BGRA8Unorm, PixelFormat::BGRA8UnormSRGB),
        ] {
            let what = format!("{from:?} x{n} into {into:?} on {name}");
            let outcome = resolve_case_into(&lane, case(from), n, case(into))
                .unwrap_or_else(|why| panic!("{what}: {why}"));
            if outcome.most_codes > case(into).device_codes() {
                past_bound.push(format!("{what}: {} codes", outcome.most_codes));
            }
            eprintln!(
                "the twin resolve {what}: {} of {} channels differ from Metal's rule ({} above \
                 Metal's code, {} below, by {} codes at most; {} samples stored otherwise than \
                 drawn){}",
                outcome.differ,
                outcome.channels,
                outcome.above,
                outcome.below,
                outcome.most_codes,
                outcome.changed,
                if outcome.examples.is_empty() {
                    String::new()
                } else {
                    format!(": {}", outcome.examples.join("; "))
                }
            );
        }
    }
    let counts = super::degrade_notices::counts(&messages.take_notices());
    eprintln!("the twin resolve on {name}: announced {counts:?}");
    assert!(
        counts.contains_key(&Degradation::DeviceRoundedResolve),
        "device-rounded-resolve on {name}: {counts:?}"
    );
    assert!(
        past_bound.is_empty(),
        "the twin resolve past device-rounded-resolve's codes on {name}: {past_bound:#?}"
    );
    quiet(lane, &messages);
}

/// The side of [`a_maps_form_pass_resolves_bgra8_as_announced`]'s targets.
const MAPS_SIDE: u32 = 64;

/// One pass of `pf` at 4 samples beside Maps' depth form (a 4x `Depth32Float_Stencil8` of
/// `RenderTarget` usage alone, cleared to 0, its test `GreaterEqual` with writes, which the fill at
/// depth 0 and the half at 0.5 pass): `random`, a
/// fullscreen fill whose every sample takes its own random codes ([`value`]), else a clear to
/// `(0.2, 0.4, 0.6, 1)`; then half the target (`rt6_resolve_half`, its diagonal's pixels half
/// covered) drawn per sample by `rt6_resolve_samples` (red `s / 3`, green 1 on sample 3, blue
/// 0.5). The color is stored and resolved in the pass (`StoreAndMultisampleResolve`); its samples
/// are read back after it. Each channel of the resolve is compared with Metal's rule ([`metal`])
/// over the samples read back; the count of differing channels, and of channels on half-covered
/// pixels whose samples differ, are returned.
fn maps_form_pass(
    lane: &Lane,
    case: &Case,
    random: bool,
    run: u64,
) -> Result<(usize, usize), String> {
    let (n, side) = (4u32, MAPS_SIDE);
    let device = &lane.device;
    let pixels = (side * side) as usize;
    let mut r = SplitMix(0x4d41_5053 ^ (run << 40) ^ case.pf as u64);
    let mut floats = Vec::with_capacity(pixels * n as usize * 16);
    for y in 0..side {
        for x in 0..side {
            for s in 0..n {
                let mut lanes = [0f32; 4];
                for (index, &(_, component, channel)) in case.channels.iter().enumerate() {
                    let code = value(channel, index, (x, y + 256), (s, n), &mut r);
                    lanes[component] = encoding(channel, code);
                }
                for lane in lanes {
                    floats.extend_from_slice(&lane.to_le_bytes());
                }
            }
        }
    }
    let texture = |pf, samples: u32, usage| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, side.into(), side.into(), false);
        if samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = samples.into();
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        device
            .new_texture(&descriptor)
            .map_err(|e| format!("new_texture {pf:?} x{samples}: {e}"))
    };
    let msaa = texture(
        case.pf,
        n,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
    )?;
    let depth = texture(
        PixelFormat::Depth32FloatStencil8,
        n,
        TextureUsage::RENDER_TARGET,
    )?;
    let resolved = texture(case.pf, 1, TextureUsage::RENDER_TARGET)?;
    let (view_pf, kernel, readback) = readback_view(case);
    let view = msaa
        .new_texture_view(view_pf)
        .map_err(|e| format!("unread: no {view_pf:?} view ({e})"))?;
    let samples_library = library(device).map_err(|e| e.to_string())?;
    let ds = Some(PixelFormat::Depth32FloatStencil8);
    let fill = {
        let function = |name: &str| {
            samples_library
                .new_function_with_name(name)
                .map(|f| f.function_descriptor())
                .map_err(|e| format!("{name}: {e}"))
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")?));
        descriptor.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")?));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(case.pf));
        descriptor.depth_format = ds;
        descriptor.stencil_format = ds;
        descriptor.raster_sample_count = n;
        device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|e| format!("the fill's pipeline: {e}"))?
    };
    let half = lane.pipeline(
        ("rt6_resolve_half", "rt6_resolve_samples"),
        &[case.pf],
        (ds, ds),
        4,
    );
    let mut tested = DepthStencilDescriptor::default();
    tested.depth_compare_function = CompareFunction::GreaterEqual;
    tested.depth_write_enabled = true;
    let tested = device
        .new_depth_stencil_state(&tested)
        .map_err(|e| e.to_string())?;
    let values = buffer_with(device, &floats).map_err(|e| e.to_string())?;
    let shape = shape(device, side, n).map_err(|e| e.to_string())?;
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = if random {
        LoadAction::DontCare
    } else {
        LoadAction::Clear
    };
    target.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
    target.store_action = StoreAction::StoreAndMultisampleResolve;
    target.resolve_texture = Some(resolved.clone());
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 0.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::DontCare;
    let command_buffer = lane.queue.command_buffer().map_err(|e| e.to_string())?;
    let mut encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(|e| format!("the pass: {e}"))?;
    let draw = |e: metal2vulkan_runtime::Error| format!("the draws: {e}");
    encoder.set_depth_stencil_state(&tested).map_err(draw)?;
    if random {
        encoder.set_render_pipeline_state(&fill).map_err(draw)?;
        encoder
            .set_fragment_buffer(Some(&values), 0, 0)
            .map_err(draw)?;
        encoder
            .set_fragment_buffer(Some(&shape), 0, 1)
            .map_err(draw)?;
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .map_err(draw)?;
    }
    encoder.set_render_pipeline_state(&half).map_err(draw)?;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .map_err(draw)?;
    encoder.end_encoding().map_err(draw)?;
    let samples = encode_through(
        device,
        &command_buffer,
        &view,
        kernel,
        SampleRead::Color(readback),
        [side, side],
        n,
    )
    .map_err(|e| format!("the readback: {e}"))?;
    command_buffer.commit().map_err(|e| e.to_string())?;
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .map_err(|e| format!("{e:?}"))?;
    if let Some(error) = command_buffer.error() {
        return Err(format!("the command buffer failed: {error:?}"));
    }
    let stored = samples.bytes().map_err(|e| e.to_string())?;
    let got = lane.read(&resolved, (0, 0), case.bytes as u64, BlitOption::None);
    let (mut differ, mut mixed) = (0, 0);
    for p in 0..pixels {
        let at = p * case.bytes;
        for &(offset, _, channel) in case.channels {
            let bits = channel.bits();
            let samples: Vec<u64> = (0..n as usize)
                .map(|s| {
                    let at = (p * n as usize + s) * case.bytes;
                    field(&stored[at..at + case.bytes], offset, bits)
                })
                .collect();
            mixed += usize::from(samples.iter().any(|&v| v != samples[0]));
            if field(&got[at..at + case.bytes], offset, bits) != metal(channel, &samples) {
                differ += 1;
            }
        }
    }
    Ok((differ, mixed))
}

/// Maps' color formats at 4 samples (`BGRA8Unorm_sRGB`, `BGRA8Unorm`), each in a pass beside its
/// depth form, resolved in the pass by the device's average: on a cleared and on a randomly filled
/// target, three runs each, the channels of the resolve that differ from the rule over the samples
/// read back (half-covered diagonal pixels and per-sample colors among them) are printed. Where
/// the device's average is measured to be Metal's (`DriverFact::ColorResolveAverageIsMetals`)
/// every channel is the rule and no pass announces `device-rounded-resolve`; elsewhere a pass
/// announces it.
#[test]
fn a_maps_form_pass_resolves_bgra8_as_announced() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let metals = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::ColorResolveAverageIsMetals);
    messages.take_notices();
    for pf in [PixelFormat::BGRA8UnormSRGB, PixelFormat::BGRA8Unorm] {
        let case = CASES.iter().find(|c| c.pf == pf).expect("a closed format");
        for random in [false, true] {
            for run in 0..3 {
                let what = format!(
                    "{pf:?} x4 beside Maps' depth form, {} target, run {run}, on {name}",
                    if random { "a random" } else { "a cleared" }
                );
                match maps_form_pass(&lane, case, random, run) {
                    Err(why) if why.starts_with("unread") => {
                        eprintln!("Maps' form resolve {what}: {why}");
                    }
                    Err(why) => panic!("{what}: {why}"),
                    Ok((differ, mixed)) => {
                        eprintln!(
                            "Maps' form resolve {what}: {differ} channels differ from Metal's \
                             rule ({mixed} channels over samples that differ)"
                        );
                        if metals {
                            assert_eq!(differ, 0, "{what}");
                        }
                        assert!(mixed > 0, "{what}: no pixel's samples differ");
                    }
                }
            }
        }
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    eprintln!("Maps' form resolve on {name}: announced {notices:?}");
    assert_eq!(
        notices.contains_key(&Degradation::DeviceRoundedResolve),
        !metals,
        "device-rounded-resolve on {name}"
    );
    quiet(lane, &messages);
}

/// A 4x sRGB texture of `RenderTarget` usage alone, cleared in a pass of no draws that resolves
/// it, resolves to the clear's codes: the bytes the device's clear of a 1x texture of the format
/// stores (Metal's bytes, `renders.rs`' `COLOR_CLEARS`, where its conversion is Metal's, printed),
/// as every sample holds the clear, which Metal's rule and the device's average give back,
/// whatever form the device keeps the clear in. Where the device announced
/// `device-color-conversion` (once a device: these passes' or the earlier format's), the device
/// converts the clear its own way, which may differ between
/// its 1x and its 4x images (the Radeon 780M under RADV: a 1x clear of 0.214 stores `0x80`, the 4x
/// one resolves to `0x7f`): there each channel of the resolve is one of the two codes nearest the
/// clear's value, the sRGB curve's for a colour channel and the value itself for alpha.
#[test]
fn cleared_srgb_attachments_resolve_to_the_clear_s_codes() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let side = 64u32;
    for (pf, [r, g, b, a], metal) in [
        (
            PixelFormat::RGBA8UnormSRGB,
            [0.5, 0.214, 0.0031308, 1.0],
            [0xbc, 0x7f, 0x0a, 0xff],
        ),
        (
            PixelFormat::BGRA8UnormSRGB,
            [0.5, 0.214, 0.0031308, 0.5],
            [0x0a, 0x7f, 0xbc, 0x80],
        ),
    ] {
        let texture = |samples: u64| {
            let mut d = TextureDescriptor::texture_2d(pf, side.into(), side.into(), false);
            if samples > 1 {
                d.texture_type = TextureType::Type2DMultisample;
                d.sample_count = samples;
            }
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            lane.device.new_texture(&d).unwrap()
        };
        let (msaa, resolved, single) = (texture(4), texture(1), texture(1));
        let clear = |texture: &Texture, resolve: Option<&Texture>| {
            let mut pass = RenderPassDescriptor::new();
            let target = &mut pass.color_attachments[0];
            target.texture = Some(texture.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = ClearColor::new(r, g, b, a);
            target.store_action = match resolve {
                Some(_) => StoreAction::MultisampleResolve,
                None => StoreAction::Store,
            };
            target.resolve_texture = resolve.cloned();
            let command_buffer = lane.queue.command_buffer().unwrap();
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
                .wait_until_completed_for(super::WAIT)
                .unwrap();
            assert_eq!(command_buffer.error(), None);
        };
        messages.take_notices();
        clear(&single, None);
        clear(&msaa, Some(&resolved));
        // A device announces each kind once: an earlier format's clear may have announced it.
        let notices = super::degrade_notices::counts(&messages.take_notices());
        let converted = (messages.taken_kinds()).contains(&Degradation::DeviceColorConversion);
        // The clear's value in codes, in the texel's byte order: the sRGB curve's for a colour
        // channel, alpha as it is.
        let encode = |v: f64| match v <= 0.0031308 {
            true => 12.92 * v,
            false => 1.055 * v.powf(1.0 / 2.4) - 0.055,
        };
        let [r, g, b] = [r, g, b].map(|v| 255.0 * encode(v));
        let alpha = 255.0 * a;
        let exact = match pf {
            PixelFormat::BGRA8UnormSRGB => [b, g, r, alpha],
            _ => [r, g, b, alpha],
        };
        let nearest = |texel: &[u8]| {
            (texel.iter().zip(exact))
                .all(|(&code, v)| [v.floor(), v.ceil()].contains(&f64::from(code)))
        };
        let cleared = super::renders::blit_read(&lane.device, &single, 4);
        let bytes: [u8; 4] = cleared[..4].try_into().unwrap();
        eprintln!(
            "cleared {pf:?} on {name}: {bytes:02x?}, Metal's {metal:02x?}{}",
            if bytes == metal {
                ""
            } else {
                " (device-color-conversion)"
            }
        );
        let read = super::renders::blit_read(&lane.device, &resolved, 4);
        let other = read.chunks(4).filter(|texel| *texel != bytes).count();
        let wrong = (read.chunks(4))
            .filter(|texel| *texel != bytes && !(converted && nearest(texel)))
            .count();
        eprintln!(
            "cleared resolve {pf:?} x4 on {name}: {other} texels other than the clear's \
             ({:02x?} first), {wrong} off the codes nearest it; announced {notices:?}",
            &read[..4]
        );
        assert_eq!(wrong, 0, "{pf:?} on {name}: {:02x?}", &read[..4]);
    }
    quiet(lane, &messages);
}
