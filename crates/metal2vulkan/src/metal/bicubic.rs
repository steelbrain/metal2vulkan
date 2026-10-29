//! Metal's bicubic filter (`filter::bicubic`) per bound pixel format.
//!
//! Metal documents no bicubic filter; its standard library declares `filter::bicubic` under
//! `__HAVE_BICUBIC_FILTERING__`. Read on an Apple M3 Max (macOS 26.7), a 2D texture sampled
//! through a bicubic sampler is filtered by the Catmull-Rom kernel over its 4x4 footprint, or
//! bilinearly over the inner 2x2, and what is done with the sum depends on the bound pixel format
//! rather than on the shader: so the translation emulates the filter by fetches
//! (`lower::sampler`, `SampleOp::Bicubic`) and reads the format's treatment from the
//! `SpecKey::BicubicTexels` specialization of the texture slot, which [`texels`] computes from the
//! bound format.
//!
//! Measured per format (authored cases `bicubic-impulse-weights-per-bound-format` and
//! `bicubic-random-texels-per-bound-format`, and 32 formats probed through the implicit sample;
//! the other sample forms were read through `rgba16Float`, `rgba32Float`, and `rgba8Unorm`; the
//! weights and each format's rounding were read by the cases). [`texels`] maps `RG8UnormSRGB`
//! (normalized unsigned) by family alone, never probed; `RG16Snorm`, mapped so at first, is read by
//! the cases. Those cases read the [`EXACT`] rule on `R8Unorm`, `RGBA8Unorm`, the 8-bit snorm R and
//! RGBA formats, and every 16-bit unorm and snorm format but `RG16Unorm`; `RG8Unorm`, `RG8Snorm`,
//! `RG16Unorm`, `BGRA8Unorm`, and `A8Unorm` take it by family.
//!
//! - 32-bit float channels (`R32Float`, `RG32Float`, `RGBA32Float`, `Depth32Float`): bilinear
//!   over the inner taps with the same 1/256 fraction ([`LINEAR`]).
//! - Normalized unsigned channels (8- and 16-bit, sRGB, the packed 16-bit and 10:10:10:2 formats,
//!   `A8Unorm`, `Depth16Unorm`): Catmull-Rom, the sum clamped into `[0, 1]` ([`UNORM`]).
//! - Normalized signed channels: Catmull-Rom, the sum clamped into `[-1, 1]` ([`SNORM`]). The
//!   upper bound is measured; the lower is inferred from the format's range, since no sum read
//!   went below `-1`.
//! - Half-float channels (`R16Float`, `RG16Float`, `RGBA16Float`) and the small floats
//!   (`RG11B10Float`, `RGB9E5Float`): Catmull-Rom, unclamped ([`HALF`]).
//!
//! Metal's arithmetic of the sum (its fixed-point weight table, an exact sum of an 8- or 16-bit
//! format's codes rounded half up onto its grid, its linear and two-pass accumulators of the float
//! formats, its rounding of a half format's sum to `half`) is not reproduced: the filter is the
//! Catmull-Rom polynomial's weights computed from the fraction and summed in the device's
//! arithmetic (`lower::sampler::bicubic::routines`, `PrecisionNote::BicubicFilterArithmetic`). The
//! bits that named that arithmetic stay the classes the binder states: [`EXACT`] and [`WIDE16`]
//! the 8- and 16-bit normalized formats of one channel width (`R8Unorm`, `RG8Unorm`,
//! `RGBA8Unorm`, `BGRA8Unorm`, `A8Unorm`, the 8-bit signed formats, and the 16-bit unsigned and
//! signed ones but `Depth16Unorm`), and [`SRGB`] and [`EXTENDED`] the classes of the in-place
//! linear filter's exact sum (`lower::sampler::bicubic::routines::Quad`).
//!
//! A tap outside the texture under `clamp_to_zero` or `clamp_to_border` reads Metal's border texel
//! (`metal::sampler::border_texel`; the sampler's border color under either mode): the border color
//! in the format's channels, a missing red, green, or blue 0 and a missing alpha 1 (`r8Unorm` reads
//! `(0, 0, 0, 1)` for a transparent-black border, `(1, 0, 0, 1)` for an opaque white one),
//! `a8Unorm`'s alpha the color's red (white `(0, 0, 0, 1)`, opaque black `(0, 0, 0, 0)`), then the
//! texture's swizzle. The channels are the bits from [`PRESENT`]; the texel for each border color
//! is the bits from [`BORDER`], which [`texels`] sets for a texture without a swizzle and the
//! binder for the bound texture's (`BicubicTexels::with_border`). A format of no measured family
//! (compressed, YUV, stencil, integer) has no treatment: [`texels`] is `None` and the binder
//! refuses it. An extended-range format has none here either: its slot's variant fetches the raw
//! texels and filters their decoded values under the format's bits, known as it is emitted
//! (`lower::sampler::extended`, measured).
//!
//! [`LINEAR`]: BicubicTexels::LINEAR
//! [`UNORM`]: BicubicTexels::UNORM
//! [`SNORM`]: BicubicTexels::SNORM
//! [`HALF`]: BicubicTexels::HALF
//! [`EXACT`]: BicubicTexels::EXACT
//! [`WIDE16`]: BicubicTexels::WIDE16
//! [`SRGB`]: BicubicTexels::SRGB
//! [`EXTENDED`]: BicubicTexels::EXTENDED
//! [`PRESENT`]: BicubicTexels::PRESENT
//! [`BORDER`]: BicubicTexels::BORDER

use crate::metal::{PixelFormat, TextureSwizzleChannels};

/// How Metal's bicubic filter treats the texels of a bound pixel format: the bits of the
/// `SpecKey::BicubicTexels` specialization constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BicubicTexels(pub u32);

impl BicubicTexels {
    /// The texels are filtered bilinearly over the footprint's inner 2x2 taps (32-bit float
    /// channels), not by the Catmull-Rom kernel.
    pub const LINEAR: u32 = 1 << 0;
    /// The filtered value is clamped into `[0, 1]` (normalized unsigned channels).
    pub const UNORM: u32 = 1 << 1;
    /// The filtered value is clamped into `[-1, 1]` (normalized signed channels).
    pub const SNORM: u32 = 1 << 2;
    /// Half-float and small-float channels: the filtered value is unclamped.
    pub const HALF: u32 = 1 << 3;
    /// Bit `PRESENT + c` is set when the format has channel `c` (0 red, 1 green, 2 blue, 3
    /// alpha): a tap outside the texture under a border address reads the border's channel `c`
    /// there, and otherwise 0 for red, green, and blue and 1 for alpha.
    pub const PRESENT: u32 = 4;
    /// The channels are 8- or 16-bit normalized codes of one width, whose sum Metal computes
    /// exactly on the format's grid (module docs): `1/(16 · 255)` or `1/(16 · 127)`, or with
    /// [`WIDE16`](Self::WIDE16) `1/65535` or `1/32767`. Set with [`UNORM`](Self::UNORM) or
    /// [`SNORM`](Self::SNORM). The bicubic filter sums them in the device's arithmetic.
    pub const EXACT: u32 = 1 << 8;
    /// With [`EXACT`](Self::EXACT): the codes are 16-bit.
    pub const WIDE16: u32 = 1 << 9;
    /// The channels are 8-bit sRGB codes (`metal::texel_format::srgb_coded`), set by
    /// `metal::texel_format::texel_format`, never by [`texels`]: each tap is the device's decode of
    /// its code, and the bicubic sum the device's arithmetic of the decoded taps.
    pub const SRGB: u32 = 1 << 10;
    /// The channels are Apple's extended-range codes (`metal::format::ExtendedRange`), set with
    /// [`EXACT`](Self::EXACT) where a function samples such a texture, and with
    /// [`SRGB`](Self::SRGB) for an `_sRGB` twin. Each tap arrives as its decoded value, and the
    /// linear, 3D, level and bicubic filters sum the decoded taps in the device's arithmetic
    /// (`lower::sampler::extended`). [`texels`] never sets it: an extended-range slot's samples
    /// read the format's bits as constants under its variant.
    pub const EXTENDED: u32 = 1 << 11;
    /// Bit `BORDER + 3c + b` is set when channel `c` (0 red to 3 alpha) of a tap outside the
    /// texture under a border address reads 1, not 0, for border color `b`, `MTLSamplerBorderColor`'s
    /// value: 0 transparent black, 1 opaque black, 2 opaque white. That is Metal's border texel
    /// (`metal::sampler::border_texel`) for the format's channels and the texture's swizzle.
    pub const BORDER: u32 = 12;

    /// The treatment of a slot bound to nothing the binding can read, whose neutral texture
    /// reads zero: every channel present, the sum unchanged, the border texels of a four-channel
    /// format without a swizzle (alpha 1 under opaque black, every channel 1 under white).
    pub const NEUTRAL: BicubicTexels =
        BicubicTexels(0xf << BicubicTexels::PRESENT | 0b110_100_100_100 << BicubicTexels::BORDER);

    /// The channels mask (red, green, blue, alpha in bits 0 to 3) and the flags.
    const fn new(flags: u32, channels: u32) -> BicubicTexels {
        BicubicTexels(flags | channels << BicubicTexels::PRESENT)
    }

    /// Whether the format has channel `c` (0 red to 3 alpha).
    pub fn has_channel(self, c: u32) -> bool {
        self.0 >> (BicubicTexels::PRESENT + c) & 1 != 0
    }

    /// These bits with the [`BORDER`](Self::BORDER) texels of a texture of `pf`, whose channels
    /// are these bits' [`PRESENT`](Self::PRESENT), made with `swizzle`.
    pub fn with_border(self, pf: PixelFormat, swizzle: TextureSwizzleChannels) -> BicubicTexels {
        let present = std::array::from_fn(|c| self.has_channel(c as u32));
        let mut bits = self.0 & !(0xfff << BicubicTexels::BORDER);
        for (b, color) in (0u32..).zip(crate::metal::sampler::BORDER_COLORS) {
            let texel = crate::metal::sampler::border_texel(pf, present, swizzle, color);
            for (c, one) in (0u32..).zip(texel) {
                if one {
                    bits |= 1 << (BicubicTexels::BORDER + 3 * c + b);
                }
            }
        }
        BicubicTexels(bits)
    }
}

const R: u32 = 0b0001;
const RG: u32 = 0b0011;
const RGB: u32 = 0b0111;
const RGBA: u32 = 0b1111;
const A: u32 = 0b1000;

/// Whether Metal's bicubic filter is measured through a runtime sampler whose min or mag filter
/// is `Cubic` and whose word (`metal::sampler::SamplerWord`, its `CUBIC_STATE` bits or all of
/// them) is `word`, whatever a function samples through it: no mip filter, isotropic, the
/// weighted-average reduction, a border color Metal names, and on every axis an address mode the
/// bicubic readings cover (not `MirrorClampToEdge`). A function that samples through such a
/// sampler refuses any other word (`lower::sampler::bicubic`), so the binder refuses it before it
/// names the cubic-sampler variant (`VariantKey::cubic_samplers`).
pub(crate) fn runtime_cubic_measured(word: u32) -> bool {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::{SamplerAddressMode, SamplerBorderColor};
    let field = |shift: u32, mask: u32| (word & mask) >> shift;
    let addressed = SamplerWord::ADDRESS_SHIFT
        .into_iter()
        .zip(SamplerWord::ADDRESS)
        .all(|(shift, mask)| {
            matches!(
                SamplerAddressMode::try_from(field(shift, mask)),
                Ok(SamplerAddressMode::ClampToEdge
                    | SamplerAddressMode::Repeat
                    | SamplerAddressMode::MirrorRepeat
                    | SamplerAddressMode::ClampToZero
                    | SamplerAddressMode::ClampToBorderColor)
            )
        });
    let bordered = matches!(
        SamplerBorderColor::try_from(field(SamplerWord::BORDER_SHIFT, SamplerWord::BORDER)),
        Ok(SamplerBorderColor::TransparentBlack
            | SamplerBorderColor::OpaqueBlack
            | SamplerBorderColor::OpaqueWhite)
    );
    let unfiltered = SamplerWord::MIP_FILTER | SamplerWord::ANISOTROPIC | SamplerWord::REDUCTION;
    word & unfiltered == 0 && addressed && bordered
}

/// How Metal's bicubic filter treats the texels of `pf` (module docs), or `None` for a format of
/// no measured family.
pub fn texels(pf: PixelFormat) -> Option<BicubicTexels> {
    use BicubicTexels as T;
    use PixelFormat as P;
    let exact8u = T::UNORM | T::EXACT;
    let exact16u = exact8u | T::WIDE16;
    let exact8s = T::SNORM | T::EXACT;
    let exact16s = exact8s | T::WIDE16;
    let (flags, channels) = match pf {
        P::R32Float | P::Depth32Float => (T::LINEAR, R),
        P::RG32Float => (T::LINEAR, RG),
        P::RGBA32Float => (T::LINEAR, RGBA),
        P::R8Unorm => (exact8u, R),
        P::RG8Unorm => (exact8u, RG),
        P::A8Unorm => (exact8u, A),
        P::RGBA8Unorm | P::BGRA8Unorm => (exact8u, RGBA),
        P::R16Unorm => (exact16u, R),
        P::RG16Unorm => (exact16u, RG),
        P::RGBA16Unorm => (exact16u, RGBA),
        P::R8UnormSRGB | P::Depth16Unorm => (T::UNORM, R),
        P::RG8UnormSRGB => (T::UNORM, RG),
        P::B5G6R5Unorm => (T::UNORM, RGB),
        P::RGBA8UnormSRGB
        | P::BGRA8UnormSRGB
        | P::RGB10A2Unorm
        | P::BGR10A2Unorm
        | P::A1BGR5Unorm
        | P::ABGR4Unorm
        | P::BGR5A1Unorm => (T::UNORM, RGBA),
        P::R8Snorm => (exact8s, R),
        P::RG8Snorm => (exact8s, RG),
        P::RGBA8Snorm => (exact8s, RGBA),
        P::R16Snorm => (exact16s, R),
        P::RG16Snorm => (exact16s, RG),
        P::RGBA16Snorm => (exact16s, RGBA),
        P::R16Float => (T::HALF, R),
        P::RG16Float => (T::HALF, RG),
        P::RG11B10Float | P::RGB9E5Float => (T::HALF, RGB),
        P::RGBA16Float => (T::HALF, RGBA),
        // An RGBX format reads as its RGBA twin with alpha 1: the twin's family,
        // red, green and blue.
        P::RGBX32Float => (T::LINEAR, RGB),
        P::RGBX8Unorm => (exact8u, RGB),
        P::RGBX8UnormSRGB => (T::UNORM, RGB),
        P::RGBX8Snorm => (exact8s, RGB),
        P::RGBX16Float => (T::HALF, RGB),
        // A 4:2:2 format reads as an `RGBA8Unorm` texture of its decoded codes, alpha 1 (our probe
        // on an Apple M3 Max under macOS 27.0.1: reads, nearest, linear and bicubic samples and
        // gathers bit for bit, `bind::ChromaDecode`): `RGBX8Unorm`'s treatment.
        P::GBGR422 | P::BGRG422 => (exact8u, RGB),
        // A BC, PVRTC, ETC2, EAC or ASTC sRGB form's texels are 8-bit sRGB codes once decoded,
        // which Metal filters as an 8-bit sRGB format's; an RGB one's alpha 1.
        P::BC1RGBASRGB
        | P::BC2RGBASRGB
        | P::BC3RGBASRGB
        | P::BC7RGBAUnormSRGB
        | P::PVRTCRGBA2BPPSRGB
        | P::PVRTCRGBA4BPPSRGB
        | P::ETC2RGB8A1SRGB
        | P::EACRGBA8SRGB
        | P::ASTC4x4SRGB
        | P::ASTC5x4SRGB
        | P::ASTC5x5SRGB
        | P::ASTC6x5SRGB
        | P::ASTC6x6SRGB
        | P::ASTC8x5SRGB
        | P::ASTC8x6SRGB
        | P::ASTC8x8SRGB
        | P::ASTC10x5SRGB
        | P::ASTC10x6SRGB
        | P::ASTC10x8SRGB
        | P::ASTC10x10SRGB
        | P::ASTC12x10SRGB
        | P::ASTC12x12SRGB => (T::UNORM, RGBA),
        P::PVRTCRGB2BPPSRGB | P::PVRTCRGB4BPPSRGB | P::ETC2RGB8SRGB => (T::UNORM, RGB),
        _ => return None,
    };
    Some(T::new(flags, channels).with_border(pf, TextureSwizzleChannels::DEFAULT))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each family's flags, and the channels the border substitution reads (the border texels
    /// apart: `border_texels_are_metals`).
    #[test]
    fn each_measured_family_has_its_treatment() {
        let t =
            |pf| BicubicTexels(texels(pf).expect("measured").0 & !(0xfff << BicubicTexels::BORDER));
        assert_eq!(
            t(PixelFormat::RGBA32Float).0,
            BicubicTexels::LINEAR | 0xf << 4
        );
        let (exact, wide) = (BicubicTexels::EXACT, BicubicTexels::WIDE16);
        assert_eq!(
            t(PixelFormat::R8Unorm).0,
            BicubicTexels::UNORM | exact | 1 << 4
        );
        assert_eq!(
            t(PixelFormat::RGBA8Snorm).0,
            BicubicTexels::SNORM | exact | 0xf << 4
        );
        assert_eq!(
            t(PixelFormat::RG16Unorm).0,
            BicubicTexels::UNORM | exact | wide | 0x3 << 4
        );
        assert_eq!(
            t(PixelFormat::RGBA16Snorm).0,
            BicubicTexels::SNORM | exact | wide | 0xf << 4
        );
        // sRGB, packed, and depth formats take the emulated sum here (an sRGB slot's raw codes
        // take the exact one by the emission's `SRGB | EXACT`).
        for pf in [
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::B5G6R5Unorm,
            PixelFormat::RGB10A2Unorm,
            PixelFormat::Depth16Unorm,
        ] {
            assert_eq!(t(pf).0 & (exact | wide), 0, "{pf:?}");
        }
        assert_eq!(
            t(PixelFormat::RG11B10Float).0,
            BicubicTexels::HALF | 0x7 << 4
        );
        let a8 = t(PixelFormat::A8Unorm);
        assert!(!a8.has_channel(0) && a8.has_channel(3));
        assert!(BicubicTexels::NEUTRAL.has_channel(0) && BicubicTexels::NEUTRAL.has_channel(3));
        // Unmeasured families have none.
        for pf in [
            PixelFormat::BGR10XR,
            PixelFormat::BC1RGBA,
            PixelFormat::ETC2RGB8,
            PixelFormat::RGBA8Uint,
            PixelFormat::Stencil8,
        ] {
            assert_eq!(texels(pf), None, "{pf:?}");
        }
        // A 4:2:2 format takes `RGBX8Unorm`'s treatment.
        assert_eq!(t(PixelFormat::GBGR422).0, t(PixelFormat::RGBX8Unorm).0);
    }
    /// The border texels: per border color, the channels reading 1 after the
    /// format's channels and the swizzle. `R8Unorm` reads `(0, 0, 0, 1)` under transparent black
    /// and `(1, 0, 0, 1)` under white; `A8Unorm`'s alpha is the color's red, so its opaque black
    /// reads `(0, 0, 0, 0)`; a swizzle moves the channels, `ga0r` on `RGBA16Float` under opaque
    /// black reading `(0, 1, 0, 0)`. The neutral is a four-channel format's without a swizzle.
    #[test]
    fn border_texels_are_metals() {
        use crate::metal::TextureSwizzle as S;
        let ones = |t: BicubicTexels, b: u32| -> [bool; 4] {
            std::array::from_fn(|c| t.0 >> (BicubicTexels::BORDER + 3 * c as u32 + b) & 1 != 0)
        };
        let (zero, black, white) = (0, 1, 2);
        let r8 = texels(PixelFormat::R8Unorm).expect("measured");
        assert_eq!(ones(r8, zero), [false, false, false, true]);
        assert_eq!(ones(r8, black), [false, false, false, true]);
        assert_eq!(ones(r8, white), [true, false, false, true]);
        let a8 = texels(PixelFormat::A8Unorm).expect("measured");
        assert_eq!(ones(a8, zero), [false; 4]);
        assert_eq!(ones(a8, black), [false; 4]);
        assert_eq!(ones(a8, white), [false, false, false, true]);
        let ga0r = TextureSwizzleChannels::new(S::Green, S::Alpha, S::Zero, S::Red);
        let rgba16 = texels(PixelFormat::RGBA16Float)
            .expect("measured")
            .with_border(PixelFormat::RGBA16Float, ga0r);
        assert_eq!(ones(rgba16, black), [false, true, false, false]);
        assert_eq!(ones(rgba16, white), [true, true, false, true]);
        let rrr1 = TextureSwizzleChannels::new(S::Red, S::Red, S::Red, S::One);
        assert_eq!(
            ones(a8.with_border(PixelFormat::A8Unorm, rrr1), zero),
            [false, false, false, true]
        );
        assert_eq!(
            BicubicTexels(0xf << BicubicTexels::PRESENT)
                .with_border(PixelFormat::RGBA8Unorm, TextureSwizzleChannels::DEFAULT),
            BicubicTexels::NEUTRAL
        );
    }
}
