//! The formats the filter computes, texel by texel: how each texel's bytes, in the Vulkan format
//! the runtime holds its pixel format in, decode to the `float` channels the filter sums and how
//! the sums encode back ([`layout`]); and the taps of each destination texel ([`axis_tap`]).

use crate::metal::PixelFormat;

/// How one field of a texel converts to and from its `float` value (a padding field has none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// A normalized unsigned code `c` of the field's width `b`: `c` times `1 / (2^b - 1)`
    /// ([`Field::scale`]); a value encodes
    /// as `RoundEven(clamp(v, 0, 1) (2^b - 1))`, a NaN 0.
    Unorm,
    /// A normalized signed code `c` (two's complement) of the field's width `b`: `max(c` times
    /// `1 / (2^(b-1) - 1), -1)`; a value encodes as `RoundEven(clamp(v, -1, 1) (2^(b-1) - 1))`, a NaN
    /// 0.
    Snorm,
    /// An 8-bit sRGB colour code `c`: `c` times `1 / 255`, as a read of its UNORM twin gives it,
    /// through the MSL sRGB-to-linear transfer by the device's division and `Pow`
    /// (`lower::arith::pack`'s, as a shader's read of the twin decodes it); a value encodes by
    /// the MSL linear-to-sRGB transfer by the device's `Pow` of `clamp(v, 0, 1)` (a NaN 0), then
    /// as an 8-bit unorm: `RoundEven(t 255)`, as `unorm4x8_srgb` does.
    Srgb8,
    /// A float of `exponent` exponent bits (5 or 8) and `mantissa` mantissa bits, with a sign bit
    /// where `signed`. A 32-bit float is its bits both ways (a NaN as the device computes it). A
    /// half decodes by the device's widening (`UnpackHalf2x16`) and encodes by its narrowing
    /// (`PackHalf2x16`). `RG11B10Float`'s unsigned fields decode as the half of the same exponent
    /// and mantissa bits, by the same widening; no device instruction narrows to them, so a value
    /// encodes to nearest even by integer steps, an overflow to infinity, a NaN as the field's
    /// quiet NaN and a negative value as 0.
    Float {
        exponent: u32,
        mantissa: u32,
        signed: bool,
    },
    /// An extended-range 10-bit code `c`: `c - 384` times `1 / 510` ([`EXTENDED_SCALE`]), an
    /// alpha code clamped to `[384,
    /// 894]` first (`[0, 1]`); a value encodes as `RoundEven(v 510) + 384`, `v` clamped to `[-1,
    /// 1.5]` (alpha to `[0, 1]`) and the code to `[0, 1023]`, a NaN 0.
    Extended { alpha: bool },
    /// An extended-range sRGB colour code `c`: its encoded value `s`, `c - 384` times `1 / 510`,
    /// decodes by the MSL sRGB-to-linear transfer of `|s|` by the device's division and `Pow`,
    /// the sign kept; a value `v` (a NaN 0) encodes by the MSL linear-to-sRGB transfer of `|v|`
    /// by the device's `Pow`, the sign kept, then as [`Rule::Extended`]'s colour code.
    ExtendedSrgb,
    /// Bits written as this constant, which no channel reads (`BGR10_XR`'s top two bits: 3).
    Padding(u32),
}

/// The `f32` reciprocal of 255, rounded once: an 8-bit sRGB code's encoded value is the code
/// times it (`Rule::Srgb8`).
pub(super) const SRGB_SCALE: f32 = 1.0 / 255.0;

/// The `f32` reciprocal of 510, rounded once: an extended-range code's value is its offset from
/// 384 times it (`Rule::Extended`, and `Rule::ExtendedSrgb`'s encoded value).
pub(super) const EXTENDED_SCALE: f32 = 1.0 / 510.0;

/// One field of a texel: `width` bits at bit `offset` of the texel's little-endian bytes, never
/// across a 32-bit word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Field {
    pub(super) offset: u32,
    pub(super) width: u32,
    pub(super) rule: Rule,
}

impl Field {
    /// The field's mask, `2^width - 1`.
    pub(super) fn mask(&self) -> u32 {
        if self.width == 32 {
            u32::MAX
        } else {
            (1 << self.width) - 1
        }
    }

    /// The largest code of a normalized field: `2^width - 1` (unorm) or `2^(width-1) - 1`
    /// (snorm).
    pub(super) fn den(&self) -> u32 {
        match self.rule {
            Rule::Snorm => (1 << (self.width - 1)) - 1,
            _ => self.mask(),
        }
    }

    /// The `f32` reciprocal of [`den`](Self::den), rounded once: a normalized code's value is the
    /// code times it, a product every device rounds alike, where a quotient's precision is the
    /// device's.
    pub(super) fn scale(&self) -> f32 {
        1.0 / self.den() as f32
    }

    /// Whether a channel reads the field (every field but padding).
    pub(super) fn channel(&self) -> bool {
        !matches!(self.rule, Rule::Padding(_))
    }
}

/// What a texel holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Texel {
    /// Fields that each decode and encode on their own, a channel each but padding.
    Fields(&'static [Field]),
    /// `RGB9E5Float`'s word: three 9-bit mantissas from bit 0 and a 5-bit exponent they share in
    /// bits 27 to 31, three channels. A channel reads `m 2^(e - 24)`; the three encode by
    /// Vulkan's shared-exponent conversion, each clamped to `[0, 65408]` first (a NaN 0).
    Rgb9e5,
}

/// A format the filter computes: its texel's bytes and what they hold, in the bytes of the Vulkan
/// format the runtime holds it in (a red/blue twin swaps only fields of one rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) bytes: u32,
    pub(super) texel: Texel,
}

impl Layout {
    /// The `float` channels a texel decodes to.
    pub(super) fn channels(&self) -> usize {
        match self.texel {
            Texel::Fields(fields) => fields.iter().filter(|f| f.channel()).count(),
            Texel::Rgb9e5 => 3,
        }
    }
}

const fn at(offset: u32, width: u32, rule: Rule) -> Field {
    Field {
        offset,
        width,
        rule,
    }
}

const U: Rule = Rule::Unorm;
const S: Rule = Rule::Snorm;
const SRGB: Rule = Rule::Srgb8;
const F32: Rule = Rule::Float {
    exponent: 8,
    mantissa: 23,
    signed: true,
};
const F16: Rule = Rule::Float {
    exponent: 5,
    mantissa: 10,
    signed: true,
};
const UF6: Rule = Rule::Float {
    exponent: 5,
    mantissa: 6,
    signed: false,
};
const UF5: Rule = Rule::Float {
    exponent: 5,
    mantissa: 5,
    signed: false,
};
const XR: Rule = Rule::Extended { alpha: false };
const XR_ALPHA: Rule = Rule::Extended { alpha: true };
const XR_SRGB: Rule = Rule::ExtendedSrgb;

static UNORM8: [Field; 4] = [at(0, 8, U), at(8, 8, U), at(16, 8, U), at(24, 8, U)];
static SNORM8: [Field; 4] = [at(0, 8, S), at(8, 8, S), at(16, 8, S), at(24, 8, S)];
/// Three sRGB colour bytes from bit 0 and a linear alpha byte.
static SRGB8: [Field; 4] = [
    at(0, 8, SRGB),
    at(8, 8, SRGB),
    at(16, 8, SRGB),
    at(24, 8, U),
];
static UNORM16: [Field; 4] = [at(0, 16, U), at(16, 16, U), at(32, 16, U), at(48, 16, U)];
static SNORM16: [Field; 4] = [at(0, 16, S), at(16, 16, S), at(32, 16, S), at(48, 16, S)];
static FLOAT16: [Field; 4] = [
    at(0, 16, F16),
    at(16, 16, F16),
    at(32, 16, F16),
    at(48, 16, F16),
];
static FLOAT32: [Field; 4] = [
    at(0, 32, F32),
    at(32, 32, F32),
    at(64, 32, F32),
    at(96, 32, F32),
];
/// Ten-bit colour codes from bit 0 and a 2-bit alpha.
static UNORM10_2: [Field; 4] = [at(0, 10, U), at(10, 10, U), at(20, 10, U), at(30, 2, U)];
/// Five, six and five bits from bit 0.
static UNORM565: [Field; 3] = [at(0, 5, U), at(5, 6, U), at(11, 5, U)];
/// A 1-bit alpha in bit 0, then three 5-bit fields.
static UNORM1555: [Field; 4] = [at(0, 1, U), at(1, 5, U), at(6, 5, U), at(11, 5, U)];
/// Three 5-bit fields from bit 0, then a 1-bit alpha in bit 15.
static UNORM5551: [Field; 4] = [at(0, 5, U), at(5, 5, U), at(10, 5, U), at(15, 1, U)];
/// Four 4-bit fields from bit 0.
static UNORM4: [Field; 4] = [at(0, 4, U), at(4, 4, U), at(8, 4, U), at(12, 4, U)];
/// `RG11B10Float`'s two 11-bit and one 10-bit unsigned floats from bit 0.
static UFLOAT11_11_10: [Field; 3] = [at(0, 11, UF6), at(11, 11, UF6), at(22, 10, UF5)];
/// `BGR10_XR`'s blue, green and red codes from bit 0 (`R32_UINT`), and the two top bits Metal
/// sets.
static BGR10_XR: [Field; 4] = [
    at(0, 10, XR),
    at(10, 10, XR),
    at(20, 10, XR),
    at(30, 2, Rule::Padding(3)),
];
static BGR10_XR_SRGB: [Field; 4] = [
    at(0, 10, XR_SRGB),
    at(10, 10, XR_SRGB),
    at(20, 10, XR_SRGB),
    at(30, 2, Rule::Padding(3)),
];
/// `BGRA10_XR`'s four 16-bit lanes (`R16G16B16A16_UINT`), blue first, each code in its lane's top
/// ten bits (the low six 0).
static BGRA10_XR: [Field; 4] = [
    at(6, 10, XR),
    at(22, 10, XR),
    at(38, 10, XR),
    at(54, 10, XR_ALPHA),
];
static BGRA10_XR_SRGB: [Field; 4] = [
    at(6, 10, XR_SRGB),
    at(22, 10, XR_SRGB),
    at(38, 10, XR_SRGB),
    at(54, 10, XR_ALPHA),
];

const fn fields(bytes: u32, f: &'static [Field]) -> Layout {
    Layout {
        bytes,
        texel: Texel::Fields(f),
    }
}

/// The layout of `pf` where the filter computes its mipmaps: every colour format Metal filters
/// (normalized, sRGB, float, packed and shared-exponent ones) and the extended-range `BGR10_XR`
/// and `BGRA10_XR` formats and their sRGB twins, held raw in their integer formats. `None` for
/// any other format: an integer format (whose mipmaps are picked texels), a compressed, depth,
/// stencil or YCbCr one, and the other extended-range forms, Metal's mipmaps of which are not
/// measured.
pub(super) fn layout(pf: PixelFormat) -> Option<Layout> {
    use PixelFormat as P;
    Some(match pf {
        P::R8Unorm | P::A8Unorm => fields(1, &UNORM8[..1]),
        P::RG8Unorm => fields(2, &UNORM8[..2]),
        P::RGBA8Unorm | P::BGRA8Unorm | P::ARGB8Unorm => fields(4, &UNORM8),
        P::R8Snorm => fields(1, &SNORM8[..1]),
        P::RG8Snorm => fields(2, &SNORM8[..2]),
        P::RGBA8Snorm => fields(4, &SNORM8),
        P::R8UnormSRGB => fields(1, &SRGB8[..1]),
        P::RG8UnormSRGB => fields(2, &SRGB8[..2]),
        P::RGBA8UnormSRGB | P::BGRA8UnormSRGB => fields(4, &SRGB8),
        P::RGB10A2Unorm | P::BGR10A2Unorm => fields(4, &UNORM10_2),
        P::B5G6R5Unorm => fields(2, &UNORM565),
        P::A1BGR5Unorm => fields(2, &UNORM1555),
        P::BGR5A1Unorm => fields(2, &UNORM5551),
        P::ABGR4Unorm | P::BGRA4Unorm => fields(2, &UNORM4),
        P::R16Unorm => fields(2, &UNORM16[..1]),
        P::RG16Unorm => fields(4, &UNORM16[..2]),
        P::RGBA16Unorm => fields(8, &UNORM16),
        P::R16Snorm => fields(2, &SNORM16[..1]),
        P::RG16Snorm => fields(4, &SNORM16[..2]),
        P::RGBA16Snorm => fields(8, &SNORM16),
        P::R16Float => fields(2, &FLOAT16[..1]),
        P::RG16Float => fields(4, &FLOAT16[..2]),
        P::RGBA16Float => fields(8, &FLOAT16),
        P::R32Float => fields(4, &FLOAT32[..1]),
        P::RG32Float => fields(8, &FLOAT32[..2]),
        P::RGBA32Float => fields(16, &FLOAT32),
        P::RG11B10Float => fields(4, &UFLOAT11_11_10),
        P::RGB9E5Float => Layout {
            bytes: 4,
            texel: Texel::Rgb9e5,
        },
        P::BGR10XR => fields(4, &BGR10_XR),
        P::BGR10XRSRGB => fields(4, &BGR10_XR_SRGB),
        P::BGRA10XR => fields(8, &BGRA10_XR),
        P::BGRA10XRSRGB => fields(8, &BGRA10_XR_SRGB),
        _ => return None,
    })
}

/// The two taps along one axis of destination texel `i` of `small` texels from `big`: the first
/// source texel `x0` and the weight `t` of the next, `min(x0 + 1, big - 1)`, the first weighing
/// `1 - t`. They are a sampler's linear taps at the texel's centre: the centre `(i + 1/2) big /
/// small` less a half is `x0 + t`, exactly `((2i + 1) big - small) / (2 small)`, its fraction
/// rounded once to `f32`.
pub(super) fn axis_tap(i: u32, big: u32, small: u32) -> (u32, f32) {
    let num = (2 * u64::from(i) + 1) * u64::from(big) - u64::from(small);
    let den = 2 * u64::from(small);
    let x0 = (num / den) as u32;
    let t = ((num % den) as f64 / den as f64) as f32;
    (x0, t)
}

// ---- The words a dispatch reads ([`super::MipmapFilterModule::PARAMS_BINDING`]) ----

/// Words of the level's header.
pub(super) const HEADER_WORDS: u32 = 11;

/// Words of one entry of a tap table: `x0`, then the `f32` bits of `t` ([`axis_tap`]).
pub(super) const TAP_WORDS: u32 = 2;

/// Header words, in order: the destination texels of the level (every layer), the invocations
/// of a dispatch row, the level's width, height and depth, the level above's, and where the x,
/// y and z tap tables start (words). The tap tables follow the header, one entry
/// ([`TAP_WORDS`]) per destination coordinate along their axis.
pub(super) mod header {
    pub const TEXELS: u32 = 0;
    pub const ROW: u32 = 1;
    pub const EXTENT: u32 = 2;
    pub const ABOVE: u32 = 5;
    pub const TAPS: u32 = 8;
}
