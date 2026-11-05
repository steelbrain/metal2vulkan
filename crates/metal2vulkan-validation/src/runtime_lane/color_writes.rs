//! What a draw stores when its fragment function writes a `float` to a normalized, sRGB or
//! half-float color attachment, against Metal's measured conversion, on the device the cfg
//! selects, under the validation layer.
//!
//! Metal probes (an Apple M3 Max, macOS 26.7; and an M4, 27.0.1) drew every code's exact value,
//! every tie between neighbouring codes with two `f32` steps either side, and specials into each
//! format below, plain and through blending with factors one and zero, and closed each conversion
//! as one rule (every input measured):
//!
//! - a unorm channel of `b` bits: `x` clamped to `[0, 1]` (a NaN 0), times `2^b - 1` exactly,
//!   rounded to nearest (its only tie, `x = 0.5`, up);
//! - a snorm channel: `x` clamped to `[-1, 1]` (a NaN 0), times `2^(b-1) - 1` exactly, rounded to
//!   nearest (its only ties, `x = ±0.5`, away from zero), so -1.0 stores `-(2^(b-1) - 1)`;
//!   through blending a NaN stores -1.0's code;
//! - an sRGB channel: `x` clamped to `[0, 1]` (a NaN 0), times 4095 exactly, rounded to nearest
//!   (half up), encoded by Metal's thresholds (`bind::SRGB_RESOLVE_THRESHOLDS`), the same
//!   12-bit encode as its `pack_float_to_srgb_unorm4x8`; the alpha of an sRGB format is unorm;
//! - a half-float channel: `x` rounded toward zero to `half`, a finite value past 65504 stored as
//!   65504, an infinity as itself, and a NaN as `0x7e00`;
//! - `RG11B10Float`'s unsigned 11- and 10-bit floats (probe 6): `x` rounded toward zero, a
//!   negative value (`-inf` included) stored as 0, a finite value past the largest as the
//!   largest, an infinity as itself, and a NaN as the quiet NaN (`0x7e0`, `0x3f0`);
//! - `A8Unorm`'s alpha (probe 6): as an 8-bit unorm channel;
//! - a packed 16-bit format's channel (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`,
//!   `BGR5A1Unorm`; an M3 Max): `x` rounded to `half` to nearest even first, then
//!   as a unorm channel of its bits, a tie (a 1-bit channel's 0.5) to even.
//!
//! A `half` output (probe 5, an M4) converts as its value does, and a half-float channel stores it
//! as it is, a NaN's payload included; the half-output cases draw every `half`.
//!
//! A packed format the device samples but does not render (the RTX 5080's `A1BGR5Unorm` and
//! `ABGR4Unorm`) draws through the view of its word, the word the fragment function encodes
//! (`metal::format::WordAttachment`): its cases must draw there, plain and blended alike, and say
//! so.
//!
//! Each case draws the same inputs into the format, every channel of a texel the same input,
//! plain and through blending with factors one and zero, reads the attachment back, and compares
//! every channel with the rule. One difference is the recorded near-equivalent
//! `DeviceNanEncoding`, not a failure, where the fragment stage records it: a NaN a half-float or
//! unsigned small float channel stores as another NaN (ANV and RADV store `0xfe00`; lavapipe
//! quiets or canonicalizes a `half` NaN; RADV and NVIDIA keep a small float NaN's payload). Such
//! channels are counted and printed apart, with the codes stored; where the fragment stage does
//! not record the note (the driver is measured to store Metal's NaN,
//! `DriverDeviation::MetalFloatColorNans` and `MetalHalfColorNans`), and for every other channel, a
//! NaN's class included, the lane is exact. The fragment function writes the output as computed and the device converts it: where the
//! device is measured to convert the kind of channel as Metal does (MoltenVK on Apple's GPUs,
//! `DriverDeviation::MetalColorWrites`; `MetalSmallFloatColorWrites`, a `float` output to a
//! half-float or `RG11B10Float` channel; `MetalNorm16ColorWrites`, an output to a 16-bit unorm or
//! snorm channel) the lane holds the device's conversion to Metal's, the measurement of the
//! deviation on that device; elsewhere the fragment stage records the device's conversion
//! (`PrecisionNote::DeviceColorConversion`), and the lane prints the channels that differ from
//! Metal's without holding them. A packed 16-bit word the fragment function encodes is held to
//! Metal's rule; an 8-bit sRGB word's codes are the device's encode (the MSL transfer through its
//! `Pow`, `PrecisionNote::DevicePackConversion`: `metal::format::WordAttachment`), where Metal
//! encodes through its twelve-bit table, so they are printed against Metal's and not held (the
//! RTX 5080, which renders neither `R8_SRGB` nor `R8G8_SRGB`, stored 3,170 of the lane's 53,907
//! `R8Unorm_sRGB` channels otherwise than Metal, each printed example a code apart).

use super::degrade_notices::counts;
use super::resolves::{lane, quiet, Lane};
use crate::candidate::runtime_samples::{library, shape};
use crate::runtime_lane::buffer_with;
use metal2vulkan::bind::SRGB_RESOLVE_THRESHOLDS;
use metal2vulkan::metal::format::WordAttachment;
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage, TextureWriteRoundingMode,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, Reconvergence, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, StoreAction, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The kinds [`pipeline_states_announce_the_shader_conversion_once`] takes.
pub(super) const COLOR_WRITES: [Degradation; 1] = [Degradation::ShaderColorConversion];

/// The texels of a row.
const WIDTH: usize = 1024;

/// How a channel stores a value.
#[derive(Clone, Copy, Debug)]
pub(super) enum Channel {
    Unorm(u32),
    Snorm(u32),
    Srgb,
    Half,
    /// An unsigned float of a five-bit exponent and this many fraction bits.
    UnsignedFloat(u32),
    /// A packed 16-bit format's unorm channel of this many bits, through `half`.
    HalfUnorm(u32),
}

impl Channel {
    pub(super) fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::Snorm(bits) | Channel::HalfUnorm(bits) => bits,
            Channel::Srgb => 8,
            Channel::Half => 16,
            Channel::UnsignedFloat(mantissa) => mantissa + 5,
        }
    }
}

/// The fragment function's output: `float4` or `half4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Output {
    Float,
    Half,
}

/// A format of the probe: its texel bytes, each channel's bit offset and encoding, its inputs
/// (for a `half` output, each a `half` widened exactly, a NaN's payload included), and the
/// output that writes them.
pub(super) struct Case {
    pub(super) pf: PixelFormat,
    pub(super) bytes: usize,
    pub(super) channels: &'static [(u32, Channel)],
    pub(super) values: fn() -> Vec<f32>,
    pub(super) output: Output,
}

const U8: Channel = Channel::Unorm(8);
const U16: Channel = Channel::Unorm(16);
const S8: Channel = Channel::Snorm(8);
const S16: Channel = Channel::Snorm(16);
const SRGB: Channel = Channel::Srgb;
const HALF: Channel = Channel::Half;

pub(super) const CASES: &[Case] = &[
    Case {
        pf: PixelFormat::RGBA8Unorm,
        bytes: 4,
        channels: &[(0, U8), (8, U8), (16, U8), (24, U8)],
        values: || [unorm(8), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGBA8UnormSRGB,
        bytes: 4,
        channels: &[(0, SRGB), (8, SRGB), (16, SRGB), (24, U8)],
        values: || [srgb(), unorm(8), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::BGRA8UnormSRGB,
        bytes: 4,
        channels: &[(0, SRGB), (8, SRGB), (16, SRGB), (24, U8)],
        values: || [srgb(), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGBA16Unorm,
        bytes: 8,
        channels: &[(0, U16), (16, U16), (32, U16), (48, U16)],
        values: || [unorm(16), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGB10A2Unorm,
        bytes: 4,
        channels: &[
            (0, Channel::Unorm(10)),
            (10, Channel::Unorm(10)),
            (20, Channel::Unorm(10)),
            (30, Channel::Unorm(2)),
        ],
        values: || [unorm(10), unorm(2), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGBA8Snorm,
        bytes: 4,
        channels: &[(0, S8), (8, S8), (16, S8), (24, S8)],
        values: || [snorm(8), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGBA16Snorm,
        bytes: 8,
        channels: &[(0, S16), (16, S16), (32, S16), (48, S16)],
        values: || [snorm(16), specials()].concat(),
        output: Output::Float,
    },
    Case {
        pf: PixelFormat::RGBA16Float,
        bytes: 8,
        channels: &[(0, HALF), (16, HALF), (32, HALF), (48, HALF)],
        values: || [half(), specials()].concat(),
        output: Output::Float,
    },
    // Probe 6's formats.
    float_case(PixelFormat::R8Unorm, 1, &[(0, U8)], || {
        [unorm(8), specials()].concat()
    }),
    float_case(PixelFormat::RG8Unorm, 2, &[(0, U8), (8, U8)], || {
        [unorm(8), specials()].concat()
    }),
    float_case(
        PixelFormat::BGRA8Unorm,
        4,
        &[(0, U8), (8, U8), (16, U8), (24, U8)],
        || [unorm(8), specials()].concat(),
    ),
    float_case(PixelFormat::R8UnormSRGB, 1, &[(0, SRGB)], || {
        [srgb(), specials()].concat()
    }),
    float_case(
        PixelFormat::RG8UnormSRGB,
        2,
        &[(0, SRGB), (8, SRGB)],
        || [srgb(), specials()].concat(),
    ),
    float_case(PixelFormat::R16Unorm, 2, &[(0, U16)], || {
        [unorm(16), specials()].concat()
    }),
    float_case(PixelFormat::RG16Unorm, 4, &[(0, U16), (16, U16)], || {
        [unorm(16), specials()].concat()
    }),
    float_case(PixelFormat::BGR10A2Unorm, 4, TEN_TEN_TEN_TWO, || {
        [unorm(10), unorm(2), specials()].concat()
    }),
    float_case(PixelFormat::R8Snorm, 1, &[(0, S8)], || {
        [snorm(8), specials()].concat()
    }),
    float_case(PixelFormat::RG8Snorm, 2, &[(0, S8), (8, S8)], || {
        [snorm(8), specials()].concat()
    }),
    float_case(PixelFormat::R16Snorm, 2, &[(0, S16)], || {
        [snorm(16), specials()].concat()
    }),
    float_case(PixelFormat::RG16Snorm, 4, &[(0, S16), (16, S16)], || {
        [snorm(16), specials()].concat()
    }),
    float_case(PixelFormat::R16Float, 2, &[(0, HALF)], || {
        [half(), specials()].concat()
    }),
    float_case(PixelFormat::RG16Float, 4, &[(0, HALF), (16, HALF)], || {
        [half(), specials()].concat()
    }),
    float_case(PixelFormat::A8Unorm, 1, &[(0, U8)], || {
        [unorm(8), specials()].concat()
    }),
    float_case(
        PixelFormat::RG11B10Float,
        4,
        &[
            (0, Channel::UnsignedFloat(6)),
            (11, Channel::UnsignedFloat(6)),
            (22, Channel::UnsignedFloat(5)),
        ],
        || [unsigned_float(6), unsigned_float(5), specials()].concat(),
    ),
    float_case(
        PixelFormat::B5G6R5Unorm,
        2,
        &[
            (0, Channel::HalfUnorm(5)),
            (5, Channel::HalfUnorm(6)),
            (11, Channel::HalfUnorm(5)),
        ],
        || [packed(), specials()].concat(),
    ),
    float_case(
        PixelFormat::A1BGR5Unorm,
        2,
        &[
            (0, Channel::HalfUnorm(1)),
            (1, Channel::HalfUnorm(5)),
            (6, Channel::HalfUnorm(5)),
            (11, Channel::HalfUnorm(5)),
        ],
        || [packed(), specials()].concat(),
    ),
    float_case(
        PixelFormat::ABGR4Unorm,
        2,
        &[
            (0, Channel::HalfUnorm(4)),
            (4, Channel::HalfUnorm(4)),
            (8, Channel::HalfUnorm(4)),
            (12, Channel::HalfUnorm(4)),
        ],
        || [packed(), specials()].concat(),
    ),
    float_case(
        PixelFormat::BGR5A1Unorm,
        2,
        &[
            (0, Channel::HalfUnorm(5)),
            (5, Channel::HalfUnorm(5)),
            (10, Channel::HalfUnorm(5)),
            (15, Channel::HalfUnorm(1)),
        ],
        || [packed(), specials()].concat(),
    ),
    // A `half` output, every `half`.
    half_case(
        PixelFormat::RGBA8Unorm,
        4,
        &[(0, U8), (8, U8), (16, U8), (24, U8)],
    ),
    half_case(
        PixelFormat::RGBA8UnormSRGB,
        4,
        &[(0, SRGB), (8, SRGB), (16, SRGB), (24, U8)],
    ),
    half_case(
        PixelFormat::RGBA16Unorm,
        8,
        &[(0, U16), (16, U16), (32, U16), (48, U16)],
    ),
    half_case(PixelFormat::RGB10A2Unorm, 4, TEN_TEN_TEN_TWO),
    half_case(
        PixelFormat::RGBA8Snorm,
        4,
        &[(0, S8), (8, S8), (16, S8), (24, S8)],
    ),
    half_case(
        PixelFormat::RGBA16Snorm,
        8,
        &[(0, S16), (16, S16), (32, S16), (48, S16)],
    ),
    half_case(
        PixelFormat::RGBA16Float,
        8,
        &[(0, HALF), (16, HALF), (32, HALF), (48, HALF)],
    ),
];

const TEN_TEN_TEN_TWO: &[(u32, Channel)] = &[
    (0, Channel::Unorm(10)),
    (10, Channel::Unorm(10)),
    (20, Channel::Unorm(10)),
    (30, Channel::Unorm(2)),
];

const fn float_case(
    pf: PixelFormat,
    bytes: usize,
    channels: &'static [(u32, Channel)],
    values: fn() -> Vec<f32>,
) -> Case {
    Case {
        pf,
        bytes,
        channels,
        values,
        output: Output::Float,
    }
}

const fn half_case(pf: PixelFormat, bytes: usize, channels: &'static [(u32, Channel)]) -> Case {
    Case {
        pf,
        bytes,
        channels,
        values: every_half,
        output: Output::Half,
    }
}

/// Every `half`, widened exactly (a NaN's payload in the `f32` significand's top bits).
fn every_half() -> Vec<f32> {
    (0..=u16::MAX).map(widen).collect()
}

/// `half` bits `h` widened exactly, a NaN's payload included.
fn widen(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let fraction = u32::from(h & 0x3ff);
    if h & 0x7c00 == 0x7c00 {
        return f32::from_bits(sign | 0x7f80_0000 | (fraction << 13));
    }
    let magnitude = half_value(h & 0x7fff);
    f32::from_bits(sign | magnitude.to_bits())
}

/// The `half` an exactly widened value `x` holds ([`widen`]'s inverse).
fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if x.is_nan() || x.is_infinite() {
        return sign | 0x7c00 | ((bits >> 13) & 0x3ff) as u16;
    }
    sign | half_toward_zero(x.abs())
}

/// `x` and its two neighbours on each side, in `f32` steps.
fn around(x: f32) -> [f32; 5] {
    [
        x.next_down().next_down(),
        x.next_down(),
        x,
        x.next_up(),
        x.next_up().next_up(),
    ]
}

pub(super) fn specials() -> Vec<f32> {
    let mut out = vec![
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
        1e-45,
        -1e-45,
        1.175_494_2e-38,
        -1.175_494_2e-38,
        1e-10,
        -1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        65519.0,
        65520.0,
        1e10,
        -1e10,
    ];
    out.extend(
        [0x7fc0_0000u32, 0xffc0_0000, 0x7f80_0001, 0x7fff_ffff]
            .into_iter()
            .map(f32::from_bits),
    );
    out
}

/// Unorm of `bits`: every code's exact value, and every tie between neighbouring codes with two
/// `f32` steps either side.
fn unorm(bits: u32) -> Vec<f32> {
    let max = f64::from((1u32 << bits) - 1);
    let mut out: Vec<f32> = (0..1u32 << bits)
        .map(|k| (f64::from(k) / max) as f32)
        .collect();
    for k in 0..(1u32 << bits) - 1 {
        out.extend(around(((f64::from(k) + 0.5) / max) as f32));
    }
    out
}

/// Snorm of `bits`: every code's exact value, and every tie, with two `f32` steps either side.
fn snorm(bits: u32) -> Vec<f32> {
    let max = (1i32 << (bits - 1)) - 1;
    let mut out: Vec<f32> = (-max..=max)
        .map(|k| (f64::from(k) / f64::from(max)) as f32)
        .collect();
    for k in -max..max {
        out.extend(around(((f64::from(k) + 0.5) / f64::from(max)) as f32));
    }
    out
}

fn srgb_decode(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB: every multiple of 2⁻¹⁴ in `[0, 1.0078]`, every half in `[0, 1]`, each code's exact
/// linear value, the exact encode's boundaries between codes and the 12-bit grid's ties, each
/// boundary and tie with two `f32` steps either side.
fn srgb() -> Vec<f32> {
    let mut out: Vec<f32> = (0..=16512).map(|j| j as f32 / 16384.0).collect();
    out.extend((0..=0x3c00u16).map(half_value));
    out.extend((0..=255).map(|k| srgb_decode(f64::from(k) / 255.0) as f32));
    for k in 0..255 {
        out.extend(around(srgb_decode((f64::from(k) + 0.5) / 255.0) as f32));
    }
    for i in 0..4095 {
        out.extend(around(((f64::from(i) + 0.5) / 4095.0) as f32));
    }
    out
}

/// The value of a `half`'s bits.
fn half_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => fraction * 2f64.powi(-24),
        0x1f if fraction == 0.0 => f64::INFINITY,
        0x1f => f64::NAN,
        _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    };
    (sign * magnitude) as f32
}

/// Half: every tie between neighbouring finite halves of either sign, with one `f32` step either
/// side, and every finite half.
pub(super) fn half() -> Vec<f32> {
    let mut out = Vec::new();
    for bits in 0..0x7c00u16 {
        let a = f64::from(half_value(bits));
        let b = if bits == 0x7bff {
            65536.0
        } else {
            f64::from(half_value(bits + 1))
        };
        let tie = ((a + b) / 2.0) as f32;
        for v in [tie.next_down(), tie, tie.next_up(), a as f32] {
            out.push(v);
            out.push(-v);
        }
    }
    out
}

/// The packed formats' inputs (our probe's): every half in `[0, 1]` and every midpoint
/// between neighbouring halves there with one `f32` step either side, and every code's exact
/// value and tie of 1, 4, 5 and 6 bits with two `f32` steps either side.
fn packed() -> Vec<f32> {
    let mut out: Vec<f32> = (0..=0x3c00u16).map(half_value).collect();
    for bits in 0..0x3c00u16 {
        let tie = ((f64::from(half_value(bits)) + f64::from(half_value(bits + 1))) / 2.0) as f32;
        out.extend([tie.next_down(), tie, tie.next_up()]);
    }
    for bits in [1, 4, 5, 6] {
        out.extend(unorm(bits));
    }
    out
}

/// `x` rounded to `half` to nearest even, as an `f32`: a finite value from 65520 up infinite, a
/// NaN a NaN.
fn half_nearest(x: f32) -> f32 {
    let a = f64::from(x.abs());
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    let exponent = if a < 2f64.powi(-14) {
        -14
    } else {
        ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127
    };
    let step = 2f64.powi(exponent - 10);
    let r = (a / step).round_ties_even() * step;
    let r = if r >= 65536.0 { f64::INFINITY } else { r };
    (if x.is_sign_negative() { -r } else { r }) as f32
}

/// The value of code `c` of an unsigned float of `mantissa` fraction bits (not a NaN).
fn unsigned_float_value(c: u32, mantissa: u32) -> f64 {
    let exponent = (c >> mantissa) & 0x1f;
    let fraction = f64::from(c & ((1 << mantissa) - 1));
    let m = mantissa as i32;
    match exponent {
        0 => fraction * 2f64.powi(-14 - m),
        0x1f => f64::INFINITY,
        _ => (1.0 + fraction * 2f64.powi(-m)) * 2f64.powi(exponent as i32 - 15),
    }
}

/// An unsigned float of `mantissa` fraction bits: every finite code's value and every tie between
/// neighbouring codes with one `f32` step either side, and their negations.
pub(super) fn unsigned_float(mantissa: u32) -> Vec<f32> {
    let largest = (0x1f << mantissa) - 1;
    let mut out = Vec::new();
    for c in 0..=largest {
        let a = unsigned_float_value(c, mantissa);
        let b = if c == largest {
            65536.0
        } else {
            unsigned_float_value(c + 1, mantissa)
        };
        let tie = ((a + b) / 2.0) as f32;
        for v in [tie.next_down(), tie, tie.next_up(), a as f32] {
            out.push(v);
            out.push(-v);
        }
    }
    out
}

/// `x` rounded toward zero to an unsigned float of `mantissa` fraction bits (the module docs).
pub(super) fn unsigned_float_toward_zero(x: f32, mantissa: u32) -> u64 {
    let infinity = 0x1fu64 << mantissa;
    if x.is_nan() {
        return infinity | (1 << (mantissa - 1));
    }
    if x.is_sign_negative() {
        return 0;
    }
    if x.is_infinite() {
        return infinity;
    }
    let a = f64::from(x);
    let m = mantissa as i32;
    if a >= (2.0 - 2f64.powi(-m)) * 32768.0 {
        return infinity - 1;
    }
    if a < 2f64.powi(-14) {
        return (a / 2f64.powi(-14 - m)).floor() as u64;
    }
    let exponent = ((x.to_bits() >> 23) & 0xff) as i32 - 127;
    let k = (a / 2f64.powi(exponent - m)).floor() as u64;
    (((exponent + 15) as u64) << mantissa) | (k - (1 << mantissa))
}

/// `x` (finite, at least 0) times `max` exactly, rounded to nearest, a tie up.
pub(super) fn round_product(x: f32, max: u32) -> u32 {
    let bits = x.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let (mantissa, scale) = if exponent == 0 {
        (u128::from(bits & 0x7f_ffff), 149u32)
    } else {
        (u128::from((bits & 0x7f_ffff) | 0x80_0000), 150 - exponent)
    };
    // x * max + 1/2 = (2 * mantissa * max + 2^scale) / 2^(scale + 1), scale at most 150.
    if scale >= 100 {
        return 0;
    }
    let twice = 2 * mantissa * u128::from(max) + (1u128 << scale);
    (twice >> (scale + 1)) as u32
}

/// Metal's code of `x`, written as `output`, in a channel (the module docs).
fn metal_of(channel: Channel, x: f32, blended: bool, output: Output) -> u64 {
    match (channel, output) {
        (Channel::Half, Output::Half) => u64::from(narrow(x)),
        _ => metal(channel, x, blended),
    }
}

/// Metal's code of `x` in a channel, as the module docs give it; `blended` for a write through
/// blending with factors one and zero.
pub(super) fn metal(channel: Channel, x: f32, blended: bool) -> u64 {
    let unit = |x: f32| if x >= 0.0 { x.min(1.0) } else { 0.0 };
    match channel {
        Channel::Unorm(bits) => u64::from(round_product(unit(x), (1 << bits) - 1)),
        Channel::Snorm(bits) => {
            let max = (1u32 << (bits - 1)) - 1;
            let code = if x.is_nan() {
                if blended {
                    -(max as i64)
                } else {
                    0
                }
            } else {
                let magnitude = i64::from(round_product(x.abs().min(1.0), max));
                if x < 0.0 {
                    -magnitude
                } else {
                    magnitude
                }
            };
            (code as u64) & ((1 << bits) - 1)
        }
        Channel::Srgb => {
            let index = round_product(unit(x), 4095);
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| u32::from(t) <= index) as u64
        }
        Channel::Half => u64::from(half_toward_zero(x)),
        Channel::UnsignedFloat(mantissa) => unsigned_float_toward_zero(x, mantissa),
        // The half's exact product with a maximum below 2^6, exact in `f64`.
        Channel::HalfUnorm(bits) => {
            let h = f64::from(unit(half_nearest(x)));
            (h * f64::from((1u32 << bits) - 1)).round_ties_even() as u64
        }
    }
}

/// `x` rounded toward zero to `half`: a finite value past 65504 is 65504, a NaN `0x7e00`.
pub(super) fn half_toward_zero(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a.is_infinite() {
        return sign | 0x7c00;
    }
    if a >= 65504.0 {
        return sign | 0x7bff;
    }
    if a < 2f32.powi(-14) {
        // Exact: a power-of-two scaling of a value below 2^-14.
        return sign | (a * 2f32.powi(24)).floor() as u16;
    }
    let bits = a.to_bits();
    let exponent = ((bits >> 23) & 0xff) as u16 - (127 - 15);
    sign | (exponent << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// Whether `code`, of `channel`, is a NaN: a half, or an unsigned small float, with every
/// exponent bit and a fraction.
fn float_nan(channel: Channel, code: u64) -> bool {
    let fraction = match channel {
        Channel::Half => 10,
        Channel::UnsignedFloat(bits) => bits,
        _ => return false,
    };
    let exponent = 0x1f << fraction;
    code & exponent == exponent && code & ((1 << fraction) - 1) != 0
}

/// The `bits` bits at `offset` of a texel.
pub(super) fn field(texel: &[u8], offset: u32, bits: u32) -> u64 {
    let mut word = 0u64;
    for (i, &b) in texel.iter().enumerate() {
        word |= u64::from(b) << (8 * i);
    }
    (word >> offset) & ((1u64 << bits) - 1)
}

/// What a draw of a case stored, and how its fragment stage was compiled.
struct Drawn {
    /// The texels.
    stored: Vec<u8>,
    /// Whether the fragment stage records `PrecisionNote::DeviceNanEncoding`.
    noted: bool,
    /// Whether the fragment stage records `PrecisionNote::DeviceColorConversion`, or
    /// `PrecisionNote::DevicePackConversion` for an 8-bit sRGB word it encodes: the device's own
    /// conversion of the attachment's channels, printed against Metal's and not held to it.
    converted: bool,
}

/// How a draw blends: not at all, with factors one and zero, or by premultiplied source-over (one
/// and one minus the source alpha, added, for color and alpha).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawBlend {
    Off,
    Replace,
    Over,
}

/// The texels a draw of `values` into `case`'s format stores, blended by `blend`, and how the
/// fragment stage was compiled; or why it could not draw. Where `masked`, under that write mask
/// over a pass that clears to that color.
fn drawn(
    lane: &Lane,
    case: &Case,
    values: &[f32],
    blend: DrawBlend,
    masked: Option<(ColorWriteMask, [f64; 4])>,
) -> Result<Drawn, String> {
    let device = &lane.device;
    let height = values.len().div_ceil(WIDTH);
    let library = library(device).map_err(|e| e.to_string())?;
    let function = |name: &str| {
        library
            .new_function_with_name(name)
            .map(|f| f.function_descriptor())
            .map_err(|e| format!("{name}: {e}"))
    };
    let mut pipeline =
        RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")?));
    let fragment = match case.output {
        Output::Float => "rt6_fill_rgba32float",
        Output::Half => "rt6_fill_rgba16float",
    };
    pipeline.fragment = Some(StageSource::Function(function(fragment)?));
    let mut attachment = ColorAttachmentDescriptor::new(case.pf);
    attachment.blending_enabled = blend != DrawBlend::Off;
    if blend == DrawBlend::Over {
        attachment.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        attachment.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    }
    if let Some((mask, _)) = masked {
        attachment.write_mask = mask;
    }
    pipeline.color_attachments[0] = Some(attachment);
    let compiled = device
        .compile_device()
        .compile_render_pipeline(&pipeline, &Budget::standard())
        .map_err(|e| format!("unsupported: the pipeline ({e})"))?;
    let fragment = compiled
        .stages
        .iter()
        .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
        .ok_or("the pipeline has no fragment stage")?;
    let noted = fragment
        .precision_notes
        .contains(&PrecisionNote::DeviceNanEncoding);
    let srgb_word = matches!(
        WordAttachment::of(case.pf, device.compile_device().profile()),
        Some(WordAttachment::Srgb8 { .. })
    );
    let notes = &fragment.precision_notes;
    let converted = notes.contains(&PrecisionNote::DeviceColorConversion)
        || (srgb_word && notes.contains(&PrecisionNote::DevicePackConversion));
    let pipeline = device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .map_err(|e| format!("unsupported: the pipeline ({e})"))?;
    let mut floats = Vec::with_capacity(WIDTH * height * 16);
    for i in 0..WIDTH * height {
        let v = values.get(i).copied().unwrap_or(0.0);
        for _ in 0..4 {
            match case.output {
                Output::Float => floats.extend_from_slice(&v.to_le_bytes()),
                Output::Half => floats.extend_from_slice(&narrow(v).to_le_bytes()),
            }
        }
    }
    let values_buffer = buffer_with(device, &floats).map_err(|e| e.to_string())?;
    let shape = shape(device, WIDTH as u32, 1).map_err(|e| e.to_string())?;
    let mut descriptor = TextureDescriptor::texture_2d(case.pf, WIDTH as u64, height as u64, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device
        .new_texture(&descriptor)
        .map_err(|e| format!("unsupported: the texture ({e})"))?;
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    if let Some((_, [r, g, b, a])) = masked {
        color.clear_color = ClearColor::new(r, g, b, a);
    }
    color.store_action = StoreAction::Store;
    lane.run(&pass, |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&values_buffer), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .map_err(|e| format!("the pass: {e}"))?;
    let stored = lane.read(&target, (0, 0), case.bytes as u64, BlitOption::None);
    Ok(Drawn {
        stored,
        noted,
        converted,
    })
}

/// Every format, plain and blended, printed against Metal's conversion, and held to it where the
/// fragment stage does not record the device's own conversion (`DeviceColorConversion`, or
/// `DevicePackConversion` for an 8-bit sRGB word: the module docs).
#[test]
fn draws_store_metals_codes() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let mut differ = Vec::new();
    for case in CASES {
        let values = (case.values)();
        for blended in [false, true] {
            let label = format!(
                "{:?}{}{}",
                case.pf,
                if case.output == Output::Half {
                    " from half"
                } else {
                    ""
                },
                if blended { " blended (one, zero)" } else { "" }
            );
            // A packed format the device samples but does not render draws through the view of
            // its word (`metal::format::WordAttachment`), plain and blended alike.
            let word = WordAttachment::of(case.pf, lane.device.compile_device().profile());
            let blend = match blended {
                true => DrawBlend::Replace,
                false => DrawBlend::Off,
            };
            let Drawn {
                stored,
                noted,
                converted,
            } = match drawn(&lane, case, &values, blend, None) {
                Ok(drawn) => drawn,
                Err(why) => {
                    eprintln!("color write {label} on {name}: {why}");
                    if word.is_some() {
                        differ.push(format!("{label}: not drawn through its word view: {why}"));
                    }
                    continue;
                }
            };
            if word.is_some() {
                eprintln!("color write {label} on {name}: through the view of its word");
            }
            let (mut channels, mut wrong, mut examples) = (0usize, 0usize, Vec::new());
            let (mut nans, mut stores) = (0usize, BTreeSet::new());
            for (i, &x) in values.iter().enumerate() {
                let texel = &stored[i * case.bytes..(i + 1) * case.bytes];
                for &(offset, channel) in case.channels {
                    channels += 1;
                    let got = field(texel, offset, channel.bits());
                    let want = metal_of(channel, x, blended, case.output);
                    if got != want && noted && float_nan(channel, got) && float_nan(channel, want) {
                        nans += 1;
                        if stores.len() < 4 {
                            stores.insert(format!("{want:#06x} as {got:#06x}"));
                        }
                    } else if got != want {
                        wrong += 1;
                        if examples.len() < 6 {
                            examples.push(format!(
                                "{x:e} ({:#010x}) at bit {offset}: {got:#x}, Metal's {want:#x}",
                                x.to_bits()
                            ));
                        }
                    }
                }
            }
            if nans > 0 {
                let stores = stores.into_iter().collect::<Vec<_>>().join(", ");
                eprintln!(
                    "color write {label} on {name}: {nans} NaN channels store another NaN \
                     (DeviceNanEncoding, recorded: {stores})"
                );
            } else if noted {
                eprintln!(
                    "color write {label} on {name}: DeviceNanEncoding recorded, no NaN channel \
                     differs"
                );
            }
            let recorded = match converted {
                true => " (DeviceColorConversion, recorded)",
                false => "",
            };
            if wrong == 0 {
                eprintln!(
                    "color write {label} on {name}: Metal's, 0 of {channels} channels \
                     differ{recorded}"
                );
            } else {
                eprintln!(
                    "color write {label} on {name}: {wrong} of {channels} channels \
                     differ{recorded}: {}",
                    examples.join("; ")
                );
                if !converted {
                    differ.push(format!("{label}: {wrong} of {channels}"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "color writes not Metal's: {differ:#?}");
    quiet(lane, &messages);
}

/// A packed 16-bit or two-channel 8-bit sRGB format's draw under a write mask of some of its
/// channels, over a pass that clears it: each written channel stores Metal's code of the output
/// and each kept one the code of the clear (Metal's rule of both), held so where the fragment
/// stage does not record the device's own conversion (`DeviceColorConversion`, or
/// `DevicePackConversion` for an 8-bit sRGB word's encode). A format the device holds but does
/// not render must draw so through the view of its word (`metal::format::WordAttachment`),
/// which the fragment function merges into the texel's, in the pixel interlock or over the
/// attachment's input attachment on a device that orders fetch through them (lavapipe; an 8-bit
/// sRGB word), and says so.
#[test]
fn partly_masked_packed_formats_keep_their_fields() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let profile = lane.device.compile_device().profile().clone();
    // Each clear channel's code is clear of a tie at 4, 5 and 1 bits, so a device that clears a
    // format it renders by its own conversion stores Metal's code too.
    let clear = [0.2, 0.6, 0.8, 0.4];
    let masks = [
        ColorWriteMask::RED,
        ColorWriteMask::RED | ColorWriteMask::GREEN,
        ColorWriteMask::ALPHA,
        ColorWriteMask::RED | ColorWriteMask::GREEN | ColorWriteMask::BLUE,
    ];
    let mut differ = Vec::new();
    let cases = CASES.iter().filter(|c| {
        matches!(
            c.pf,
            PixelFormat::A1BGR5Unorm | PixelFormat::ABGR4Unorm | PixelFormat::RG8UnormSRGB
        ) && c.output == Output::Float
    });
    for case in cases {
        let values = (case.values)();
        let word = WordAttachment::of(case.pf, &profile);
        for mask in masks {
            let label = format!("{:?} under {mask:?}", case.pf);
            let (stored, converted) =
                match drawn(&lane, case, &values, DrawBlend::Off, Some((mask, clear))) {
                    Ok(drawn) => (drawn.stored, drawn.converted),
                    Err(why) => {
                        eprintln!("masked color write {label} on {name}: {why}");
                        if word.is_some() {
                            differ.push(format!("{label}: not drawn through its word: {why}"));
                        }
                        continue;
                    }
                };
            if word.is_some() {
                eprintln!("masked color write {label} on {name}: its word merged into the texel's");
            }
            let (mut channels, mut wrong, mut examples) = (0usize, 0usize, Vec::new());
            for (i, &x) in values.iter().enumerate() {
                let texel = &stored[i * case.bytes..(i + 1) * case.bytes];
                // The case lists red, green, blue and alpha by their fields, least significant
                // first; each channel's mask bit names it.
                for &(offset, channel) in case.channels {
                    let lane_of = channel_lane(case.pf, offset);
                    let bit = [
                        ColorWriteMask::RED,
                        ColorWriteMask::GREEN,
                        ColorWriteMask::BLUE,
                        ColorWriteMask::ALPHA,
                    ][lane_of];
                    channels += 1;
                    let got = field(texel, offset, channel.bits());
                    let want = match mask.contains(bit) {
                        true => metal(channel, x, false),
                        false => metal(channel, clear[lane_of] as f32, false),
                    };
                    if got != want {
                        wrong += 1;
                        if examples.len() < 6 {
                            examples.push(format!(
                                "{x:e} at bit {offset}: {got:#x}, Metal's {want:#x}"
                            ));
                        }
                    }
                }
            }
            if wrong == 0 {
                eprintln!(
                    "masked color write {label} on {name}: Metal's, 0 of {channels} channels differ"
                );
            } else {
                eprintln!(
                    "masked color write {label} on {name}: {wrong} of {channels} channels differ: {}",
                    examples.join("; ")
                );
                if !converted {
                    differ.push(format!("{label}: {wrong} of {channels}"));
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "masked color writes not Metal's: {differ:#?}"
    );
    quiet(lane, &messages);
}

/// `x` rounded to `half`, nearest even, for `x` in `[0, 2]`.
fn half_even(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let step = 2f64.powi((x.log2().floor() as i32).max(-14) - 10);
    (x / step).round_ties_even() * step
}

/// A packed 16-bit format's draw blended by premultiplied source-over (one and one minus the
/// source alpha, added, for color and alpha) over a pass that clears it to codes it holds exactly,
/// so Metal's tile holds the codes' values: each channel stores Metal's code of the blend, under
/// the fixture's fast math `h(fma(d, h(1 - a), s))`, the output clamped to `[0, 1]` (a NaN 0) and
/// rounded to `half` to nearest even, `a` its alpha, each code `c` of the clear read as
/// `h(c / m)`, the result's code to nearest even (the packed 16-bit blend sweep of the M3 Max
/// under macOS 27.0.1: every channel of this state, into each packed format, from `half4` and
/// `float4` outputs). A format the device samples but does not render must draw so through the
/// pixel interlock, which computes the blend over its word, and says so; a format the device
/// renders takes the device's blend, printed against Metal's and not held to it
/// (`DeviceColorConversion`).
#[test]
fn blended_packed_formats_store_metals_blend() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let profile = lane.device.compile_device().profile().clone();
    let mut differ = Vec::new();
    let cases = CASES
        .iter()
        .filter(|c| matches!(c.pf, PixelFormat::A1BGR5Unorm | PixelFormat::ABGR4Unorm));
    for case in cases {
        let values = (case.values)();
        let word = WordAttachment::of(case.pf, &profile);
        let layout = WordAttachment::layout(case.pf).expect("a packed format");
        // Red, green, blue and alpha codes, each the clear's exact value.
        let codes = match case.pf {
            PixelFormat::A1BGR5Unorm => [6u32, 18, 25, 1],
            _ => [3, 9, 12, 6],
        };
        let maxima = layout
            .fields()
            .map(|f| f.map_or(1, |f| (1u32 << f.bits) - 1));
        let clear: [f64; 4] = std::array::from_fn(|l| f64::from(codes[l]) / f64::from(maxima[l]));
        let label = format!(
            "{:?}{}",
            case.pf,
            if case.output == Output::Half {
                " from half"
            } else {
                ""
            }
        );
        let stored = match drawn(
            &lane,
            case,
            &values,
            DrawBlend::Over,
            Some((ColorWriteMask::ALL, clear)),
        ) {
            Ok(drawn) => drawn.stored,
            Err(why) => {
                eprintln!("blended color write {label} on {name}: {why}");
                if word.is_some() {
                    differ.push(format!("{label}: not drawn through the interlock: {why}"));
                }
                continue;
            }
        };
        if word.is_some() {
            eprintln!(
                "blended color write {label} on {name}: blended over its word in the interlock"
            );
        }
        let (mut channels, mut wrong, mut examples) = (0usize, 0usize, Vec::new());
        for (i, &x) in values.iter().enumerate() {
            let texel = &stored[i * case.bytes..(i + 1) * case.bytes];
            // The output as the function returns it, clamped and rounded to `half`.
            let out = match case.output {
                Output::Float => f64::from(x),
                Output::Half => f64::from(widen(narrow(x))),
            };
            let s = match out.is_nan() {
                true => 0.0,
                false => half_even(out.clamp(0.0, 1.0)),
            };
            let minus = half_even(1.0 - s);
            for &(offset, channel) in case.channels {
                let l = channel_lane(case.pf, offset);
                channels += 1;
                let m = f64::from(maxima[l]);
                let d = half_even(f64::from(codes[l]) / m);
                let v = half_even(d * minus + s);
                let want = (v.clamp(0.0, 1.0) * m).round_ties_even() as u64;
                let got = field(texel, offset, channel.bits());
                if got != want {
                    wrong += 1;
                    if examples.len() < 6 {
                        examples.push(format!(
                            "{x:e} at bit {offset}: {got:#x}, Metal's {want:#x}"
                        ));
                    }
                }
            }
        }
        if wrong == 0 {
            eprintln!(
                "blended color write {label} on {name}: Metal's, 0 of {channels} channels differ"
            );
        } else {
            eprintln!(
                "blended color write {label} on {name}: {wrong} of {channels} channels differ: {}",
                examples.join("; ")
            );
            if word.is_some() {
                differ.push(format!("{label}: {wrong} of {channels}"));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "blended color writes not Metal's: {differ:#?}"
    );
    quiet(lane, &messages);
}

/// The channel (0 red to 3 alpha) whose field starts at bit `offset` of a texel of packed 16-bit
/// format `pf` (`metal::format::WordAttachment::layout`).
fn channel_lane(pf: PixelFormat, offset: u32) -> usize {
    WordAttachment::layout(pf)
        .and_then(|w| {
            w.fields()
                .iter()
                .position(|f| f.is_some_and(|f| u32::from(f.shift) == offset))
        })
        .expect("a field of a packed format")
}

/// Render pipeline states, each made twice, announce their fragment stage's color writes once,
/// with their stages' other kinds not announced before, and nothing else. A fragment function
/// writing `float4` to an `RGBA16Float` attachment under `TextureWriteRoundingMode::NearestEven`
/// announces `shader-color-conversion` on every device, its fragment stage listing the attachment
/// as rounded (the one rounding the function makes: no device's conversion rounds a half to
/// nearest even). One writing `half4` to an `RGBA16Float` attachment at Metal's default mode
/// announces no shader conversion, its stage never listing the attachment as rounded: the half is
/// stored as the device converts it.
#[test]
fn pipeline_states_announce_the_shader_conversion_once() {
    let (messages, _cache, lane) = lane();
    let device = &lane.device;
    let library = library(device).unwrap();
    let function = |name: &str| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let compile = device.compile_device();
    // Each kind is announced once per device: the second state's are those the first did not.
    let mut seen = BTreeSet::new();
    for (fragment, rounding, kind, listed) in [
        (
            "rt6_fill_rgba32float",
            TextureWriteRoundingMode::NearestEven,
            Degradation::ShaderColorConversion,
            true,
        ),
        (
            "rt6_fill_rgba16float",
            TextureWriteRoundingMode::TowardZero,
            Degradation::ShaderColorConversion,
            false,
        ),
    ] {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")));
        descriptor.fragment = Some(StageSource::Function(function(fragment)));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        descriptor.texture_write_rounding_mode = rounding;
        let compiled = compile
            .compile_render_pipeline(&descriptor, &Budget::standard())
            .unwrap();
        let lists = |list: &dyn Fn(&metal2vulkan::pipeline::StageModule) -> &[u32]| {
            compiled.stages.iter().map(list).collect::<Vec<&[u32]>>()
        };
        let rounded = lists(&|s| s.rounded_colors.as_slice());
        let want_rounded: &[&[u32]] = match listed {
            true => &[&[], &[0]],
            false => &[&[], &[]],
        };
        assert_eq!(rounded, want_rounded, "{fragment}");
        let mut expected: BTreeSet<Degradation> = compiled
            .stages
            .iter()
            .flat_map(|s| s.precision_notes.iter())
            .map(|&note| Degradation::from_precision_note(note))
            .collect();
        if compiled
            .stages
            .iter()
            .any(|s| s.reconvergence == Reconvergence::ImplementationDefined)
        {
            expected.insert(Degradation::ImplementationDefinedReconvergence);
        }
        if listed {
            expected.insert(kind);
        }
        expected.retain(|k| !seen.contains(k));
        for _ in 0..2 {
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap();
        }
        let counts = counts(&messages.take_notices());
        let announced: BTreeSet<Degradation> = counts.keys().copied().collect();
        assert_eq!(announced, expected, "{fragment}: {counts:?}");
        assert!(counts.values().all(|&n| n == 1), "{fragment}: {counts:?}");
        seen.extend(announced);
    }
    quiet(lane, &messages);
}

/// The rules reproduce the probe's spot answers (module docs).
#[test]
fn the_rules_give_metals_spot_answers() {
    assert_eq!(metal(U8, 0.5, false), 0x80);
    assert_eq!(metal(U16, 0.5, false), 0x8000);
    assert_eq!(metal(SRGB, 0.5, false), 0xbc);
    assert_eq!(metal(S8, -0.5, false), 0xc0);
    assert_eq!(metal(S8, -1.5, false), 0x81);
    assert_eq!(metal(S8, f32::NAN, false), 0);
    assert_eq!(metal(S8, f32::NAN, true), 0x81);
    assert_eq!(metal(U8, f32::NAN, false), 0);
    assert_eq!(metal(HALF, 8.940_697e-8, false), 1);
    assert_eq!(metal(HALF, 65520.0, false), 0x7bff);
    assert_eq!(metal(HALF, f32::NAN, false), 0x7e00);
    assert_eq!(metal(HALF, -1.0, false), 0xbc00);
    assert_eq!(metal(Channel::Unorm(2), 1.0 / 6.0 + 1e-7, false), 1);
    let (f11, f10) = (Channel::UnsignedFloat(6), Channel::UnsignedFloat(5));
    assert_eq!(metal(f11, f32::NAN, false), 0x7e0);
    assert_eq!(metal(f10, f32::NAN, true), 0x3f0);
    assert_eq!(metal(f11, f32::NEG_INFINITY, false), 0);
    assert_eq!(metal(f11, -1e-10, false), 0);
    assert_eq!(metal(f11, f32::INFINITY, false), 0x7c0);
    assert_eq!(metal(f11, 1e10, false), 0x7bf);
    assert_eq!(metal(f10, 65520.0, false), 0x3df);
    assert_eq!(metal(f11, 1.0, false), 15 << 6);
    assert_eq!(metal(f10, 1.0 + 1.0 / 32.0 - 1e-7, false), 15 << 5);
    assert_eq!(metal(f11, 2f32.powi(-20), false), 1);
    assert_eq!(metal(f11, 2f32.powi(-20) * 0.99, false), 0);
    assert!(float_nan(HALF, 0x7e00) && float_nan(HALF, 0xfe00) && float_nan(HALF, 0x7c01));
    assert!(!float_nan(HALF, 0x7c00) && !float_nan(HALF, 0xfc00) && !float_nan(U16, 0x7e00));
    assert!(float_nan(f11, 0x7e0) && float_nan(f11, 0x7c1) && float_nan(f10, 0x3f0));
    assert!(!float_nan(f11, 0x7c0) && !float_nan(f10, 0x3e0));
    for c in 0..0x7c0 {
        let x = unsigned_float_value(c, 6) as f32;
        assert_eq!(metal(f11, x, false), u64::from(c), "{c:#x}");
    }
    for h in 0..=u16::MAX {
        assert_eq!(narrow(widen(h)), h, "{h:#06x}");
    }
    assert_eq!(metal_of(HALF, widen(0x7e01), false, Output::Half), 0x7e01);
    assert_eq!(metal_of(U8, widen(0x3800), false, Output::Half), 0x80);
}
