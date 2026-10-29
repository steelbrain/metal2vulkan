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
//! weights and each format's rounding were pinned by the cases). [`texels`] maps `RG8UnormSRGB`
//! (normalized unsigned) by family alone, never probed; `RG16Snorm`, mapped so at first, is read by
//! the cases. Those cases read the [`EXACT`] rule on `R8Unorm`, `RGBA8Unorm`, the 8-bit snorm R and
//! RGBA formats, and every 16-bit unorm and snorm format but `RG16Unorm`; `RG8Unorm`, `RG8Snorm`,
//! `RG16Unorm`, `BGRA8Unorm`, and `A8Unorm` take it by family.
//!
//! - 32-bit float channels (`R32Float`, `RG32Float`, `RGBA32Float`, `Depth32Float`): bilinear
//!   over the inner taps with the same 1/256 fraction, the sum rounded to `f32` with ties away
//!   from zero, a tap of zero weight skipped ([`LINEAR`]). The sum is Metal's linear accumulator
//!   (each term truncated toward zero onto a grid 38 bits below the largest term's tap and weight
//!   exponents, the terms summed exactly; `Depth32Float`'s by family), and a denormal result is
//!   `+0`.
//! - Normalized unsigned channels (8- and 16-bit, sRGB, the packed 16-bit and 10:10:10:2 formats,
//!   `A8Unorm`, `Depth16Unorm`): Catmull-Rom, the sum clamped into `[0, 1]` ([`UNORM`]).
//! - Normalized signed channels: Catmull-Rom, the sum clamped into `[-1, 1]` ([`SNORM`]). The
//!   upper bound is measured; the lower is inferred from the format's range, since no sum read
//!   went below `-1`.
//! - Half-float channels (`R16Float`, `RG16Float`, `RGBA16Float`) and the small floats
//!   (`RG11B10Float`, `RGB9E5Float`): Catmull-Rom, the sum rounded to `half` with ties away from
//!   zero, unclamped ([`HALF`]). A 1D sum is Metal's linear accumulator with a grid 17 bits below
//!   the reference, a 2D one its two-pass accumulator (each row onto a grid 17 bits below the
//!   row's reference, then the rows onto one 20 bits below theirs), an axis whose four taps clamp
//!   onto one texel weighed as that texel alone at the unit weight, and a NaN or an infinity
//!   anywhere in a 2D footprint decides the result whatever its weight.
//!
//! The weights are Metal's fixed-point table (`weights`), the same on both axes and for every
//! format measured. The 8- and 16-bit normalized formats with channels of one width
//! (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`, `A8Unorm`, the 8-bit signed formats, and
//! the 16-bit unsigned and signed ones but `Depth16Unorm`) sum their codes exactly and round the
//! sum half up onto `1/(16 · 255)`, `1/(16 · 127)`, `1/65535`, or `1/32767`: [`EXACT`], with
//! [`WIDE16`] for the 16-bit grids. That is all of Metal's arithmetic for a 1D texture and for a
//! 2D one but next to a rounding tie. An 8-bit sRGB format sums its decoded taps exactly too, onto
//! `1/4095` (color) and `1/(16 · 255)` (alpha), where a function reads the raw codes ([`SRGB`] with
//! [`EXACT`], set by the emission, not here; each tap the device's decode of its code). The 32-bit float and half-float formats' sums are
//! Metal's accumulators. The other formats' sums are an emulation in `f32` operations:
//! `PrecisionNote::BicubicFilterArithmetic`.
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
    /// The filtered value is rounded to `half` (half-float and small-float channels).
    pub const HALF: u32 = 1 << 3;
    /// Bit `PRESENT + c` is set when the format has channel `c` (0 red, 1 green, 2 blue, 3
    /// alpha): a tap outside the texture under a border address reads the border's channel `c`
    /// there, and otherwise 0 for red, green, and blue and 1 for alpha.
    pub const PRESENT: u32 = 4;
    /// The channels are 8- or 16-bit normalized codes of one width whose sum Metal computes
    /// exactly and rounds half up onto the format's grid (module docs): `1/(16 · 255)` or
    /// `1/(16 · 127)`, or with [`WIDE16`](Self::WIDE16) `1/65535` or `1/32767`. Set with
    /// [`UNORM`](Self::UNORM) or [`SNORM`](Self::SNORM).
    pub const EXACT: u32 = 1 << 8;
    /// With [`EXACT`](Self::EXACT): the codes are 16-bit.
    pub const WIDE16: u32 = 1 << 9;
    /// The channels are 8-bit sRGB codes (`metal::texel_format::srgb_coded`). With
    /// [`EXACT`](Self::EXACT), set only where a function decodes the codes itself (the slot's
    /// `SrgbTexture` variant: a format the device holds as its UNORM twin), each tap's color value
    /// is the device's decode of its code (the MSL transfer through its `Pow`), read on the grid
    /// `1/4095`, and its alpha a code on `1/(16 · 255)`, so the exact sum rounds half up onto those
    /// grids. [`texels`] never sets it;
    /// `metal::texel_format::texel_format` does, without `EXACT`, where the device decodes the
    /// taps and the sum is the emulated one.
    pub const SRGB: u32 = 1 << 10;
    /// The channels are Apple's extended-range codes (`metal::format::ExtendedRange`), set with
    /// [`EXACT`](Self::EXACT) where a function samples such a texture, and with
    /// [`SRGB`](Self::SRGB) for an `_sRGB` twin. Each tap arrives as its value (the shader's decode
    /// of the raw code): a color lane's `(c - 384) / 510` on the grid `1/(16 · 510)` (an `_sRGB`
    /// twin's `±s / 8192` on `1/8192`), and alpha's `(c - 384) / 510`, its code clamped to `[384,
    /// 894]` first, on `1/(8 · 510)`. The sums are signed and reach past `±1` (up to twice the
    /// grid); a 2D or 3D linear sum truncates a color lane's terms onto `2^-7` of its grid steps
    /// and alpha's onto `2^-4`, where a bicubic sum (the 2D exact routine's) truncates none; and a
    /// `half` result rounds an `_sRGB` color lane's value with ties away from zero (an Apple M3 Max
    /// and an Apple M4; the bicubic filter's, `lower::sampler::extended`). [`texels`] never sets
    /// it: an extended-range slot's samples read the format's bits as constants under its variant.
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

/// Metal's Catmull-Rom weights of taps `i - 1 .. i + 2` at fraction `k / 256` (`k < 256`), in
/// units of `2^-16`; they sum to `2^16`.
///
/// A compressed encoding of the table Metal answers (authored case
/// `bicubic-weight-table-at-every-fraction-on-rgba16-unorm`, `628c2f53`, both axes alike; checked against the
/// table by `metal::bicubic::tests::the_weights_are_the_measured_table`), not a derivation: the
/// cube `C = k³ / 256` is rounded to the nearest unit for `k` in `76 ..= 95`, `161 ..= 176`, and
/// `255`, and to the nearest 16 units elsewhere, and each weight is half of `-C + 2k² - 256k`,
/// `3C - 5k² + 2^17`, `-3C + 4k² + 256k`, or `C - k²` (the Catmull-Rom polynomials scaled by
/// `2^17`), rounded to nearest with ties away from zero. The ranges are fitted to the readings.
#[cfg(test)]
pub fn weights(k: u32) -> [i32; 4] {
    let k = (k & 255) as i32;
    let cube = k * k * k;
    let fine = (76..=95).contains(&k) || (161..=176).contains(&k) || k == 255;
    let c = if fine {
        (cube + 128) >> 8
    } else {
        ((cube + 2048) >> 12) << 4
    };
    let (k2, k256) = (k * k, 256 * k);
    let half_away = |a: i32| a.signum() * ((a.abs() + 1) >> 1);
    [
        half_away(-c + 2 * k2 - k256),
        half_away(3 * c - 5 * k2 + (1 << 17)),
        half_away(-3 * c + 4 * k2 + k256),
        half_away(c - k2),
    ]
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
    /// Metal's weights at every fraction `k`, four per `k` in units of `2^-16`, read from the
    /// impulse responses of authored case `628c2f53` (both axes answered alike). `k = 75` and
    /// `181` were not solved uniquely there: their rows are the rule's, whose sums reproduce that
    /// case's readings at both fractions.
    const MEASURED: [i32; 1024] = [
        0, 65536, 0, 0, -127, 65534, 130, -1, -252, 65526, 264, -2, -375, 65514, 402, -5, -496,
        65496, 544, -8, -615, 65474, 690, -13, -732, 65446, 840, -18, -847, 65414, 994, -25, -960,
        65376, 1152, -32, -1071, 65334, 1314, -41, -1180, 65286, 1480, -50, -1287, 65234, 1650,
        -61, -1392, 65176, 1824, -72, -1503, 65138, 1978, -77, -1604, 65070, 2160, -90, -1703,
        64998, 2346, -105, -1800, 64920, 2536, -120, -1895, 64838, 2730, -137, -1988, 64750, 2928,
        -154, -2087, 64682, 3106, -165, -2176, 64584, 3312, -184, -2263, 64482, 3522, -205, -2356,
        64398, 3712, -218, -2439, 64286, 3930, -241, -2520, 64168, 4152, -264, -2607, 64070, 4354,
        -281, -2684, 63942, 4584, -306, -2767, 63834, 4794, -325, -2840, 63696, 5032, -352, -2919,
        63578, 5250, -373, -2996, 63454, 5472, -394, -3063, 63302, 5722, -425, -3136, 63168, 5952,
        -448, -3207, 63030, 6186, -473, -3276, 62886, 6424, -498, -3335, 62714, 6690, -533, -3400,
        62560, 6936, -560, -3463, 62402, 7186, -589, -3524, 62238, 7440, -618, -3583, 62070, 7698,
        -649, -3648, 61920, 7936, -672, -3703, 61742, 8202, -705, -3756, 61558, 8472, -738, -3807,
        61370, 8746, -773, -3864, 61200, 9000, -800, -3911, 61002, 9282, -837, -3964, 60822, 9544,
        -866, -4007, 60614, 9834, -905, -4056, 60424, 10104, -936, -4103, 60230, 10378, -969,
        -4148, 60030, 10656, -1002, -4183, 59802, 10962, -1045, -4224, 59592, 11248, -1080, -4263,
        59378, 11538, -1117, -4300, 59158, 11832, -1154, -4343, 58958, 12106, -1185, -4376, 58728,
        12408, -1224, -4407, 58494, 12714, -1265, -4444, 58278, 13000, -1298, -4471, 58034, 13314,
        -1341, -4504, 57808, 13608, -1376, -4527, 57554, 13930, -1421, -4556, 57318, 14232, -1458,
        -4583, 57078, 14538, -1497, -4608, 56832, 14848, -1536, -4631, 56582, 15162, -1577, -4652,
        56326, 15480, -1618, -4671, 56066, 15802, -1661, -4696, 55824, 16104, -1696, -4711, 55554,
        16434, -1741, -4732, 55302, 16744, -1778, -4743, 55022, 17082, -1825, -4760, 54760, 17400,
        -1864, -4775, 54494, 17722, -1905, -4788, 54222, 18048, -1946, -4799, 53946, 18378, -1989,
        -4810, 53669, 18708, -2031, -4819, 53388, 19040, -2073, -4827, 53107, 19371, -2115, -4834,
        52823, 19705, -2158, -4840, 52536, 20040, -2200, -4845, 52248, 20376, -2243, -4849, 51957,
        20713, -2285, -4852, 51665, 21051, -2328, -4854, 51369, 21392, -2371, -4855, 51072, 21732,
        -2413, -4855, 50774, 22073, -2456, -4853, 50472, 22416, -2499, -4851, 50169, 22759, -2541,
        -4848, 49865, 23103, -2584, -4844, 49558, 23448, -2626, -4839, 49250, 23794, -2669, -4833,
        48939, 24141, -2711, -4826, 48627, 24489, -2754, -4818, 48312, 24838, -2796, -4810, 47997,
        25187, -2838, -4800, 47680, 25536, -2880, -4791, 47366, 25882, -2921, -4780, 47046, 26232,
        -2962, -4767, 46722, 26586, -3005, -4752, 46392, 26944, -3048, -4743, 46082, 27282, -3085,
        -4724, 45742, 27648, -3130, -4711, 45422, 27994, -3169, -4696, 45096, 28344, -3208, -4679,
        44766, 28698, -3249, -4660, 44430, 29056, -3290, -4639, 44090, 29418, -3333, -4624, 43768,
        29760, -3368, -4599, 43418, 30130, -3413, -4580, 43086, 30480, -3450, -4559, 42750, 30834,
        -3489, -4536, 42408, 31192, -3528, -4511, 42062, 31554, -3569, -4492, 41734, 31896, -3602,
        -4463, 41378, 32266, -3645, -4440, 41040, 32616, -3680, -4415, 40698, 32970, -3717, -4388,
        40350, 33328, -3754, -4359, 39998, 33690, -3793, -4336, 39664, 34032, -3824, -4311, 39326,
        34378, -3857, -4276, 38958, 34752, -3898, -4247, 38610, 35106, -3933, -4216, 38256, 35464,
        -3968, -4191, 37922, 35802, -3997, -4156, 37558, 36168, -4034, -4127, 37214, 36514, -4065,
        -4096, 36864, 36864, -4096, -4063, 36510, 37218, -4129, -4028, 36150, 37576, -4162, -3999,
        35810, 37914, -4189, -3968, 35464, 38256, -4216, -3927, 35090, 38626, -4253, -3892, 34734,
        38976, -4282, -3863, 34398, 39306, -4305, -3824, 34032, 39664, -4336, -3791, 33686, 40002,
        -4361, -3756, 33334, 40344, -4386, -3719, 32978, 40690, -4413, -3680, 32616, 41040, -4440,
        -3639, 32250, 41394, -4469, -3604, 31902, 41728, -4490, -3567, 31550, 42066, -4513, -3528,
        31192, 42408, -4536, -3487, 30830, 42754, -4561, -3452, 30486, 43080, -4578, -3415, 30138,
        43410, -4597, -3368, 29760, 43768, -4624, -3335, 29426, 44082, -4637, -3292, 29062, 44424,
        -4658, -3255, 28718, 44746, -4673, -3208, 28344, 45096, -4696, -3167, 27990, 45426, -4713,
        -3132, 27654, 45736, -4722, -3087, 27290, 46074, -4741, -3048, 26944, 46392, -4752, -3007,
        26594, 46714, -4765, -2964, 26238, 47040, -4778, -2919, 25878, 47370, -4793, -2880, 25536,
        47680, -4800, -2838, 25187, 47997, -4810, -2796, 24838, 48312, -4818, -2754, 24489, 48627,
        -4826, -2711, 24141, 48939, -4833, -2669, 23794, 49250, -4839, -2626, 23448, 49558, -4844,
        -2584, 23103, 49865, -4848, -2541, 22759, 50169, -4851, -2499, 22416, 50472, -4853, -2456,
        22073, 50774, -4855, -2413, 21732, 51072, -4855, -2371, 21392, 51369, -4854, -2328, 21051,
        51665, -4852, -2285, 20713, 51957, -4849, -2243, 20376, 52248, -4845, -2200, 20040, 52536,
        -4840, -2159, 19710, 52818, -4833, -2116, 19374, 53104, -4826, -2071, 19034, 53394, -4821,
        -2032, 18712, 53664, -4808, -1991, 18386, 53938, -4797, -1948, 18054, 54216, -4786, -1903,
        17718, 54498, -4777, -1864, 17400, 54760, -4760, -1823, 17078, 55026, -4745, -1780, 16750,
        55296, -4730, -1735, 16418, 55570, -4717, -1696, 16104, 55824, -4696, -1655, 15786, 56082,
        -4677, -1620, 15486, 56320, -4650, -1575, 15158, 56586, -4633, -1536, 14848, 56832, -4608,
        -1495, 14534, 57082, -4585, -1460, 14238, 57312, -4554, -1415, 13914, 57570, -4533, -1376,
        13608, 57808, -4504, -1343, 13322, 58026, -4469, -1300, 13006, 58272, -4442, -1263, 12710,
        58498, -4409, -1224, 12408, 58728, -4376, -1191, 12126, 58938, -4337, -1148, 11814, 59176,
        -4306, -1111, 11522, 59394, -4269, -1080, 11248, 59592, -4224, -1039, 10946, 59818, -4189,
        -1004, 10662, 60024, -4146, -967, 10374, 60234, -4105, -936, 10104, 60424, -4056, -903,
        9830, 60618, -4009, -868, 9550, 60816, -3962, -831, 9266, 61018, -3917, -800, 9000, 61200,
        -3864, -767, 8730, 61386, -3813, -740, 8478, 61552, -3754, -703, 8198, 61746, -3705, -672,
        7936, 61920, -3648, -647, 7694, 62074, -3585, -612, 7422, 62256, -3530, -583, 7170, 62418,
        -3469, -560, 6936, 62560, -3400, -527, 6674, 62730, -3341, -500, 6430, 62880, -3274, -471,
        6182, 63034, -3209, -448, 5952, 63168, -3136, -423, 5718, 63306, -3065, -396, 5478, 63448,
        -2994, -375, 5258, 63570, -2917, -352, 5032, 63696, -2840, -327, 4802, 63826, -2765, -300,
        4566, 63960, -2690, -279, 4350, 64074, -2609, -264, 4152, 64168, -2520, -239, 3926, 64290,
        -2441, -220, 3718, 64392, -2354, -199, 3506, 64498, -2269, -184, 3312, 64584, -2176, -167,
        3114, 64674, -2085, -148, 2910, 64768, -1994, -135, 2726, 64842, -1897, -120, 2536, 64920,
        -1800, -103, 2342, 65002, -1705, -92, 2166, 65064, -1602, -79, 1986, 65130, -1501, -72,
        1824, 65176, -1392, -55, 1634, 65250, -1293, -52, 1486, 65280, -1178, -39, 1310, 65338,
        -1073, -32, 1152, 65376, -960, -23, 990, 65418, -849, -20, 846, 65440, -730, -15, 698,
        65466, -613, -8, 544, 65496, -496, -7, 410, 65506, -373, -4, 270, 65520, -250, -1, 130,
        65534, -127,
    ];

    /// [`weights`] is the measured table at every fraction, and each row sums to `2^16`.
    #[test]
    fn the_weights_are_the_measured_table() {
        for k in 0..256u32 {
            let w = weights(k);
            let at = 4 * k as usize;
            assert_eq!(w[..], MEASURED[at..at + 4], "k = {k}");
            assert_eq!(w.iter().sum::<i32>(), 1 << 16, "k = {k}");
            // The outer taps never weigh positively, the inner ones never negatively.
            assert!(w[0] <= 0 && w[3] <= 0 && w[1] >= 0 && w[2] >= 0, "k = {k}");
        }
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
