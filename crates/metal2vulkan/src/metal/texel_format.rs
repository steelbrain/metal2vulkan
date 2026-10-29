//! How Metal converts the texels of a bound pixel format to a `half` result, and filters them
//! linearly.
//!
//! A `half` texture (`texture2d<half>` and the like) reads or samples whatever format is bound
//! to it, and Metal's conversion of the texel to `half` depends on the format, which only the
//! binder knows. The translation reads the bound format's treatment from the texture slot's
//! `SpecKey::TexelFormat` specialization, which [`texel_format`] computes: the bits of
//! [`BicubicTexels`] (its family flags and channels) for a format whose conversion is measured,
//! an 8-bit sRGB format's own bits (below), a cube gather's corner alone for some other formats
//! (below), else [`TexelFormat::OTHER`].
//!
//! Measured on an Apple M3 Max (macOS 26.7; authored cases
//! `half-conversion-of-wider-texels-by-read-and-sample`, `f8f81b81`, and the probe cases), a read
//! or gather converts the texel's value to nearest even, and a sample (nearest or linear) converts
//! its filtered value by the family:
//!
//! - 32-bit float channels ([`BicubicTexels::LINEAR`]): a read or gather to nearest even, past
//!   `65504` to the infinity, half denormals kept; a sample toward zero, saturating at `±65504`.
//! - Half-float channels ([`BicubicTexels::HALF`]): the texel is a half already; a linear
//!   sample's sum to nearest with ties away from zero.
//! - 8- and 16-bit normalized channels ([`BicubicTexels::EXACT`], with
//!   [`BicubicTexels::UNORM`] or [`BicubicTexels::SNORM`], and [`BicubicTexels::WIDE16`] for 16
//!   bits): the value rounded half up onto the grid `1/4080` (`1/(16 · 255)`), `1/2032`
//!   (`1/(16 · 127)`), `1/65535`, or `1/32767`, then to nearest even. On a single texel's value,
//!   a read's or a nearest sample's, the grid is the identity, so that is the code's value
//!   `c / (2^b - 1)` correctly rounded (a signed code `-2^(b-1)` reads `-1`). A packed
//!   normalized format's class is this family too, each channel on the grid its
//!   [`TexelFormat::PACKED`] field names (below).
//!
//! A linear sample's footprint and weights are the bicubic filter's bilinear ones
//! (`metal::bicubic`): per plane axis the base tap `floor(256 (x - 1/2) + 1/2) >> 8` and the
//! fraction's 1/256 steps, the taps weighed `(256 - k) / 256` and `k / 256` and summed as the
//! family's rule above states (a 32-bit float format's sum rounded to `f32` with ties away from
//! zero first, a denormal result `+0`), but for a normalized format's 2D sum, which truncates
//! each term first, onto `2^-8` of a 16-bit code and `2^-4` of an sRGB step (Apple M4 and Apple M3
//! Max), and a 3D sample's weights and terms.
//!
//! `RGBA32Float`, `RGBA16Float`, `RGBA16Unorm`, `RGBA8Unorm`, `RGBA8Snorm`, and `RGBA16Snorm`
//! were read, and so were `RG16Unorm`, `RG8Unorm`, `BGRA8Unorm`, `A8Unorm`, and `RG8Snorm`
//! (reads, nearest and linear samples, `half` and `float` results, on an Apple M3 Max and an
//! Apple M4: the probe). The other formats of each family (`R32Float`, `RG32Float`, `R16Float`,
//! `RG16Float`, `R16Unorm`, `R8Unorm`, `R8Snorm`, `R16Snorm`, `RG16Snorm`) take its rule by family,
//! an inference: the conversion is per channel, and their channels are the read formats' channels.
//! Every other format is [`TexelFormat::OTHER`], so the translation keeps rounding the device's
//! `f32` texel exactly to nearest even (`narrow::round_texel`) and filtering it on the device, and
//! records `PrecisionNote::HalfTexelConversion`.
//!
//! **Packed formats**. Metal reads, gathers, and samples one texel of the packed formats the probe
//! read (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`, `RGB10A2Unorm`, `BGR10A2Unorm`,
//! `RG11B10Float`, `RGB9E5Float`) as each channel's exact value, `c / (2^b - 1)` correctly rounded
//! to `half` or `f32` (every read and nearest word of the committed rows), and reads
//! `RG11B10Float`'s NaN codes as `0x7fc00000`. Some devices convert their channels otherwise: ANV
//! and NVIDIA replicate a 5- or 6-bit channel's bits to 8, lavapipe multiplies by the reciprocal.
//! Their class holds their cube gather's corner (below) and their [`PackedTexels`] field
//! ([`TexelFormat::PACKED`]), whose code scales recover the code from the device's value, beside
//! their family:
//!
//! - **Metal's linear filter of them** (an Apple M3 Max on macOS 26.7
//!   and an Apple M4 on macOS 27.0.1, byte-identical: random texels of the eight formats, 2,048
//!   1D, 4,096 2D, and 4,096 3D samples each, `float` and `half` results, every channel). A packed
//!   normalized format sums as an 8-bit one does, each channel on its own grid
//!   ([`PackedTexels::filter_grids`]: the code widened by replicating its bits to 8, 10, or 12
//!   bits, on `1/(16 · 255)`, `1/(4 · 1023)`, or `1/4095`), the 2D and 3D terms truncated onto
//!   `2^-4` of a grid step: every sample. Its class is the normalized family with
//!   [`BicubicTexels::EXACT`], the field naming the grids. A small float sums as a half-float
//!   format does (its value is a `half`), its `float` result the `half` sum widened: every 1D and
//!   2D sample, and in 3D the half-float residue (about 0.3% of channels half a `half` ulp off).
//!   Its class is the half-float family.
//! - **A cube's linear sample** of a packed format is not measured. A packed normalized format's
//!   ([`TexelFormat::packed_exact`]) keeps the device's filter: its made-up corner texel lies off
//!   the code scale its sum recovers the codes on. A small float's takes the half-float family's
//!   emulation, an inference from its parts, each measured on the small floats: the face
//!   position (format-independent), the 2D sum above, and the corner tap, the `x` face's texel
//!   (its gathers', below).
//! - **A swizzled texture's class** drops the field, and a packed normalized format's family with
//!   it ([`TexelFormat::swizzled`]).
//!
//! `ARGB8Unorm` takes a field too ([`PackedTexels::Argb8`]): Metal reads its one texel as `c / 255`
//! correctly rounded, its bytes rotated in the view; its cube gathers and its linear filter are not
//! measured, so it has no corner and no family. The unorm forms of BC1 to BC3 take the same field
//! beside their cube corner: Metal reads their decoded 8-bit codes so, and a
//! device's view of a decoded twin's codes may not (lavapipe's is an ulp high). So do the 8-bit
//! linear ETC2 and EAC formats (`RGB8`, `RGB8A1`, `RGBA8`), beside their corner, and the linear
//! PVRTC formats, alone: Metal reads their decoded 8-bit codes as `c / 255` correctly rounded too
//! (an Apple M3 Max, `bind::EtcDecode`, `bind::PvrtcDecode`), their filters unmeasured. A texture buffer's
//! texels convert as a texture's of the same format (`RGBA32Float`, `RGBA16Float`, `RGBA8Unorm`,
//! and `RGBA8Snorm` read: the P2a probe).
//!
//! The two-plane YCbCr formats' views take a field too, one per grid of Metal's reads:
//! [`PackedTexels::YCbCr2048`], [`PackedTexels::YCbCr8192`] and
//! [`PackedTexels::YCbCrExtended8192`]. Their channels are Metal's reads exactly, steps `k / D`,
//! which Metal's linear filter sums on that grid with each 2D term truncated by `2^7`; a `half`
//! sample rounds the step's value with ties away from zero. The formats whose reads are the
//! codes take the families of their codes: the 8-bit ones `RGBA8Unorm`'s, the 10-bit ones
//! [`PackedTexels::Rgb10A2`]'s.
//!
//! **A `float` result.** One texel's value (a read, a 2D gather, or a sample that reads one texel
//! of one level) is its code's `c / (2^b - 1)` correctly rounded (lavapipe's reciprocal answers
//! some codes one ulp off). A linear sample's value is the same sum: a normalized format's step `n
//! / D` rounded to `f32` to nearest even, a 32-bit float format's sum rounded to `f32` with ties
//! away from zero, and a half-float format's `half` sum widened (the formats read above and
//! `R8Unorm`, `R16Float`, and `R32Float`, in 1D and 2D: the P2a probe).
//!
//! **sRGB.** Metal decodes an 8-bit sRGB color channel through its twelve-bit table (the code `c`
//! to the step `k`, the value `k / 4095` correctly rounded to the result, `lower::arith::srgb`)
//! and reads alpha as `c / 255`; its linear and bicubic filters sum the decoded steps
//! (`RGBA8Unorm_sRGB`, every code: `f8f81b81`, the probe cases). The device decodes such a texel
//! itself, through a view of the image in the sRGB format (Vulkan's texel decode applies the sRGB
//! EOTF to the codes before filtering, within the device's precision), and that is the only form
//! wherever the device samples the format: a texture slot's reads, samples, gathers and bicubic
//! taps take it, recording `PrecisionNote::SrgbTexelDecode` where the bound format's class has
//! [`SRGB`](BicubicTexels::SRGB) on a device that does not decode through Metal's own texture
//! unit (`VkProperty::MetalSampler`, whose decode is Metal's). The class is
//! [`UNORM`](BicubicTexels::UNORM) and [`SRGB`](BicubicTexels::SRGB) with its channels, never
//! [`EXACT`](BicubicTexels::EXACT): a `half` result rounds the device's `float` value, and a
//! bicubic sample sums the device's decoded taps. The formats ([`srgb_coded`]) are
//! `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`, `R8Unorm_sRGB` and `RG8Unorm_sRGB` (the probe), their
//! `RGBX8Unorm_sRGB` kin, and the sRGB forms of BC1 to BC3, BC7, PVRTC, ETC2, EAC and ASTC, whose
//! decoded 8-bit codes Metal decodes alike (an Apple M4, an Apple M3 Max): a device without such
//! a compressed format samples its decoded twin through an `R8G8B8A8_SRGB` view of the twin's
//! codes (`bind::AstcDecode`, `bind::PvrtcDecode`, `bind::EtcDecode`), its own decode again.
//!
//! A device that cannot sample an 8-bit sRGB format holds a sampled texture of it as the format's
//! UNORM twin (`metal::format::srgb_raw_twin`, [`srgb_twin_on`]: `R8Unorm_sRGB` and
//! `RG8Unorm_sRGB` on lavapipe, `RG8Unorm_sRGB` on ANV and NVIDIA among the committed profiles),
//! whose views read the codes as `c / 255`. Having no sRGB decode there, the slot's
//! `RuntimeFact::SrgbTexture` variant decodes what the view reads (one texel's codes, or the
//! device's filter of them) by the MSL transfer through the device's `Pow`
//! (`PrecisionNote::SrgbTexelDecode`).
//!
//! **Depth formats**. A depth texture has no `half` result, and no linear filter is emulated on
//! one, so its class is read by a read, plain gather or sample of one texel with a `float` result
//! (`Depth16Unorm`'s every code `c / 65535` correctly rounded, read and nearest-sampled), and by a
//! cube's gather and comparison gather, whose tap past both edges of a face
//! is the format's (Apple M4, the probes). `Depth32Float` takes `R32Float`'s bits (the 32-bit float
//! family, [`BicubicTexels::LINEAR`]): that tap is the `x` face's texel. `Depth16Unorm` takes
//! `R16Unorm`'s ([`BicubicTexels::UNORM`], [`BicubicTexels::EXACT`], [`BicubicTexels::WIDE16`]):
//! that tap is the made-up texel of the three faces' codes. `Depth32Float_Stencil8` reads its `x`
//! face's texel there, compared as given (below); every other depth format is
//! [`TexelFormat::OTHER`].
//!
//! **Cube gather corners** (the probes, Apple M4 on macOS 27.0.1, with and
//! without Metal's validation layer; the Apple M3 Max's run of the second on macOS 26.7 equals the
//! M4's byte for byte). Outside the measured families, a cube gather's tap past both edges of a
//! face is measured for these formats, on random texels and blocks of faces of 3 to 8 texels at 600
//! directions near each cube's corners, every lane and channel bit for bit: the `x` face's texel
//! for `Depth32Float_Stencil8`'s depth (its comparison gather compares that texel as
//! `Depth32Float`'s), `RG11B10Float`, `RGB9E5Float`, BC6H, and ASTC HDR (read through void-extent
//! blocks, the other block sizes by family); and for the packed normalized and compressed formats a
//! texel made up as a normalized family's is, on each channel's decoded grid ([`CubeCorner`]). The
//! class of such a format is its [`CubeCorner`] field ([`TexelFormat::CORNER`]) alone: its
//! conversions and filters are not measured, so every other access takes it as
//! [`TexelFormat::OTHER`].

use crate::metal::bicubic::{texels, BicubicTexels};
use crate::metal::{PixelFormat, TextureSwizzleChannels};

/// The treatment of a bound format's texels with a `half` result: the value of the
/// `SpecKey::TexelFormat` specialization constant, the bits of [`BicubicTexels`] for a measured
/// format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexelFormat(pub u32);

impl TexelFormat {
    /// A format whose conversion to `half` is not measured, and the slot's default: the device's
    /// decode rounded to `half` (`narrow::round_texel`) and filtered by the device. A slot bound to nothing reads zero, which
    /// every treatment converts alike.
    pub const OTHER: TexelFormat = TexelFormat(0);

    /// Bits `CORNER` to `CORNER + 3`: the [`CubeCorner`] of a format outside the measured
    /// families whose cube gather's tap past both edges of a face is measured (module docs), else
    /// 0. No other access reads them: such a format's class is otherwise [`OTHER`](Self::OTHER)'s.
    pub const CORNER: u32 = 24;

    /// The [`CORNER`](Self::CORNER) field's mask, in place.
    pub const CORNER_MASK: u32 = 0xF << TexelFormat::CORNER;

    /// Bits `PACKED` to `PACKED + 3`: the [`PackedTexels`] of a packed format, whose one texel a
    /// read, a 2D gather, or a sample reading one texel converts from its codes, else 0. The binder
    /// clears it for a swizzled texture, but at a slot whose function applies the swizzle after
    /// converting.
    pub const PACKED: u32 = 28;

    /// The [`PACKED`](Self::PACKED) field's mask, in place.
    pub const PACKED_MASK: u32 = 0xF << TexelFormat::PACKED;

    /// The [`PACKED`](Self::PACKED) field's top bit, set in exactly the fields of the grid `8192`
    /// ([`PackedTexels::YCbCr8192`], [`PackedTexels::YCbCrExtended8192`] and
    /// [`PackedTexels::Rgb10A2Srgb`]), whose one texel's `half` rounds with ties away from zero: what the `half` routine of a device that
    /// samples through Metal's own texture unit reads of the field (`lower::texture::half`), so
    /// that every other field's class specializes as a slot bound to nothing does.
    pub const YCBCR_8192: u32 = 8 << TexelFormat::PACKED;

    /// The bits of a class whose `half` conversion is measured on a device that samples through
    /// Metal's own texture unit: a measured family ([`BicubicTexels::LINEAR`],
    /// [`BicubicTexels::HALF`], [`BicubicTexels::EXACT`]), an 8-bit sRGB format's
    /// ([`BicubicTexels::SRGB`]) or a packed field. A class with none of them
    /// ([`unmeasured_on_metal_sampler`](Self::unmeasured_on_metal_sampler)) converts by the
    /// `OTHER` arm, whose rounding of Metal's `float` texel Metal's `half` may not be.
    pub const MEASURED_ON_METAL_SAMPLER: u32 = BicubicTexels::LINEAR
        | BicubicTexels::HALF
        | BicubicTexels::EXACT
        | BicubicTexels::SRGB
        | TexelFormat::PACKED_MASK;

    /// Whether the class has none of [`MEASURED_ON_METAL_SAMPLER`](Self::MEASURED_ON_METAL_SAMPLER)'s
    /// bits: a format outside the measured families, or a swizzled packed normalized format
    /// ([`swizzled`](Self::swizzled)), whose `half` conversion the module notes on such a device
    /// (`SpecKey::UnmeasuredTexels`).
    pub fn unmeasured_on_metal_sampler(self) -> bool {
        self.0 & TexelFormat::MEASURED_ON_METAL_SAMPLER == 0
    }

    /// Whether, on a device that samples through Metal's own texture unit, every access's `half`
    /// result of the class is the device's `float` texel rounded to nearest even
    /// (`SpecKey::PlainHalfTexels`): the class has no YCbCr field on the grid `8192`
    /// ([`TexelFormat::YCBCR_8192`], ties away), the routine's one arm other than its rounding to
    /// nearest even (`lower::texture::half`).
    pub fn plain_on_metal_sampler(self) -> bool {
        self.0 & TexelFormat::YCBCR_8192 == 0
    }

    /// The families whose texels decode to no NaN: normalized channels, signed or not
    /// ([`BicubicTexels::UNORM`], [`BicubicTexels::SNORM`]), 8-bit sRGB codes
    /// ([`BicubicTexels::SRGB`]) and Apple's extended-range codes ([`BicubicTexels::EXTENDED`]).
    /// A `half` read, sample or gather of a texture of a class with any of them hands back no
    /// NaN, on Metal or a device, so the device's encoding of a NaN texel cannot show there.
    /// Every other class may hold one: the half-float, small-float and 32-bit float formats, and
    /// those outside the families. Plain Metal on an Apple M4 (macOS 27.0.1)
    /// hands back a 16-bit float texel's own NaN bits, and `0x7e00` for every NaN of a 32-bit
    /// float or `RG11B10Float` texel; `RGB9E5Float` decodes its every code to a finite value.
    pub const NAN_FREE: u32 =
        BicubicTexels::UNORM | BicubicTexels::SNORM | BicubicTexels::SRGB | BicubicTexels::EXTENDED;

    /// Whether the class states channels ([`BicubicTexels::PRESENT`]): a measured family's does,
    /// a [`CubeCorner`] alone does not, so the binder states no border texel for it.
    pub fn has_channels(self) -> bool {
        self.0 >> BicubicTexels::PRESENT & 0xF != 0
    }

    /// Whether the class sums its channels on its [`PACKED`](Self::PACKED) field's grids
    /// ([`PackedTexels::filter_grids`]): a packed normalized format's [`BicubicTexels::EXACT`]
    /// class (module docs).
    pub fn packed_exact(self) -> bool {
        self.0 & BicubicTexels::EXACT != 0 && self.0 & TexelFormat::PACKED_MASK != 0
    }

    /// The class of a texture of this class's format made with a swizzle, which permutes the
    /// channels before the function converts or filters them: without the
    /// [`PACKED`](Self::PACKED) field, whose code scales are per channel, and for a
    /// [`packed_exact`](Self::packed_exact) class without the family, whose grids are the
    /// field's, its channels, and its border texels too, leaving the [`CORNER`](Self::CORNER)
    /// field alone (its class before Metal's filter of it was measured).
    pub fn swizzled(self) -> TexelFormat {
        match self.packed_exact() {
            true => TexelFormat(self.0 & TexelFormat::CORNER_MASK),
            false => TexelFormat(self.0 & !TexelFormat::PACKED_MASK),
        }
    }
}

/// How Metal's cube gather reads the tap past both edges of a face of a format outside the
/// measured families (module docs): the `x` face's texel, or a texel made up of the three faces'
/// corner texels on each channel's grid `D`, `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋` of their codes `X`,
/// `Y`, `Z` on it (a signed grid's codes read from `-D` up), as the value `n / D` correctly
/// rounded. The grid is the channel's decoded precision, not its stored width: a channel of 1, 2,
/// or 4 bits is gathered on `1/255`, of 5 bits on `1/1023`, of 6 on `1/4095`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CubeCorner {
    /// The `x` face's texel: `Depth32Float_Stencil8`'s depth, the small floats, BC6H, ASTC HDR.
    XFace = 1,
    /// Every channel on `1/255`: `ABGR4Unorm` and the 8-bit compressed formats (BC1, BC2, BC3,
    /// BC7, ETC2 RGB8 and RGB8A1, EAC RGBA8).
    Grid255 = 2,
    /// Red, green, and blue on `1/1023`, alpha on `1/255`: the 10:10:10:2 formats and the 5-bit
    /// formats with a 1-bit alpha.
    Grid1023 = 3,
    /// Red and blue on `1/1023`, green on `1/4095`: `B5G6R5Unorm` (alpha 1).
    Grid565 = 4,
    /// Every channel on `1/2047`: the unsigned EAC R11 and RG11 formats.
    Grid2047 = 5,
    /// Every channel on the signed grid `1/1023`: the signed EAC R11 and RG11 formats.
    Signed1023 = 6,
    /// Every channel on `1/16320` (`1/(64 · 255)`): the unsigned BC4 and BC5 formats.
    Grid16320 = 7,
    /// Every channel on the signed grid `1/8128` (`1/(64 · 127)`): the signed BC4 and BC5 formats.
    Signed8128 = 8,
}

impl CubeCorner {
    /// The corner of `pf`'s cube gather, where it is measured outside the families (the
    /// probes, Apple M4 on macOS 27.0.1). ASTC's HDR formats were read
    /// through `ASTC_4x4_HDR`'s void-extent blocks; the other block sizes and blocks take its
    /// corner by family, as every float format's is its `x` face's texel.
    pub fn of(pf: PixelFormat) -> Option<CubeCorner> {
        use PixelFormat as P;
        Some(match pf {
            P::Depth32FloatStencil8
            | P::RG11B10Float
            | P::RGB9E5Float
            | P::BC6HRGBFloat
            | P::BC6HRGBUfloat
            | P::ASTC4x4HDR
            | P::ASTC5x4HDR
            | P::ASTC5x5HDR
            | P::ASTC6x5HDR
            | P::ASTC6x6HDR
            | P::ASTC8x5HDR
            | P::ASTC8x6HDR
            | P::ASTC8x8HDR
            | P::ASTC10x5HDR
            | P::ASTC10x6HDR
            | P::ASTC10x8HDR
            | P::ASTC10x10HDR
            | P::ASTC12x10HDR
            | P::ASTC12x12HDR => CubeCorner::XFace,
            P::ABGR4Unorm
            | P::BC1RGBA
            | P::BC2RGBA
            | P::BC3RGBA
            | P::BC7RGBAUnorm
            | P::ETC2RGB8
            | P::ETC2RGB8A1
            | P::EACRGBA8 => CubeCorner::Grid255,
            P::RGB10A2Unorm | P::BGR10A2Unorm | P::A1BGR5Unorm | P::BGR5A1Unorm => {
                CubeCorner::Grid1023
            }
            P::B5G6R5Unorm => CubeCorner::Grid565,
            P::EACR11Unorm | P::EACRG11Unorm => CubeCorner::Grid2047,
            P::EACR11Snorm | P::EACRG11Snorm => CubeCorner::Signed1023,
            P::BC4RUnorm | P::BC5RGUnorm => CubeCorner::Grid16320,
            P::BC4RSnorm | P::BC5RGSnorm => CubeCorner::Signed8128,
            _ => return None,
        })
    }

    /// The class of a format of this corner: the field alone.
    pub fn class(self) -> TexelFormat {
        TexelFormat((self as u32) << TexelFormat::CORNER)
    }
}

/// The channels of a packed format whose one texel Metal reads as its codes' exact values (module
/// docs): measured on an Apple M3 Max by the committed rows of a probe cases
/// (`float-results-of-packed-16-bit-formats-by-read-and-sample`, its 32-bit twin, and their `half`
/// twins), every read and nearest word `c / (2^b - 1)` correctly rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PackedTexels {
    /// Red and blue of 5 bits, green of 6, no alpha (it reads 1): `B5G6R5Unorm`.
    Rgb565 = 1,
    /// Red, green, and blue of 5 bits, alpha of 1: `A1BGR5Unorm` and `BGR5A1Unorm`.
    Rgb5A1 = 2,
    /// Every channel of 4 bits: `ABGR4Unorm`.
    Rgba4 = 3,
    /// Red, green, and blue of 10 bits, alpha of 2: `RGB10A2Unorm` and `BGR10A2Unorm`.
    Rgb10A2 = 4,
    /// The small floats, `RG11B10Float` and `RGB9E5Float`: the device's value is exact, and a
    /// NaN reads `0x7fc00000` (Metal's, every NaN code of `RG11B10Float`'s rows).
    SmallFloat = 5,
    /// Every channel of 8 bits, held in its bytes' rotation: `ARGB8Unorm`, whose reads Metal
    /// answers `c / 255` correctly rounded (and its plane views over a buffer), where lavapipe's
    /// are an ulp high. Its class has no family, as its border passes through the rotation. Also
    /// the unorm forms of BC1 to BC3, whose blocks Metal decodes to 8-bit codes it reads as
    /// `c / 255` correctly rounded, where lavapipe's view of a decoded twin's codes
    /// (`R8G8B8A8_UNORM`) is an ulp high: beside their cube corner, with no family, Metal's linear
    /// filter of them not being measured. And the 8-bit linear ETC2 and EAC formats (beside their
    /// corner) and the linear PVRTC formats (alone), whose decoded 8-bit codes Metal reads so too
    /// (an Apple M3 Max, `bind::EtcDecode`, `bind::PvrtcDecode`), and which a twin holds as
    /// `R8G8B8A8_UNORM` codes on a device without them.
    Argb8 = 6,
    /// The 8- and 10-bit two-plane YCbCr formats' channels as their view holds them: Metal reads a
    /// texel as its converted color, each channel a step `k / 2048` of `[0, 1]`, and alpha as 1,
    /// which the view's half-float channels hold exactly: a read's value is the device's, as a
    /// small float's is, so the field has no code scales. Metal's linear filter sums the steps on
    /// that grid, each 2D term `k · w` truncated by `2^7` (an Apple M3 Max on macOS 26.7 and an
    /// Apple M4 on macOS 27.0.1, byte-identical, every channel of `YCBCR8_420_2P`; every 8- and
    /// 10-bit format, packed or not, at each subsampling and range;
    /// [`filter_truncation`](Self::filter_truncation)).
    YCbCr2048 = 7,
    /// The `_sRGB` twins' and the unpacked 12-bit YCbCr formats' channels as their view holds them:
    /// steps `k / 8192` of `[0, 1]`, alpha 1, which Metal's linear filter sums as
    /// [`YCbCr2048`](Self::YCbCr2048)'s on the grid `8192`, a `half` result the step's value
    /// rounded with ties away from zero (every channel of every format).
    YCbCr8192 = 8,
    /// The extended-range (`_XR`) YCbCr formats' channels, but the packed 12-bit ones': signed
    /// steps `k / 8192` within `[-2, 2]`, alpha 1, which Metal's linear filter sums as
    /// [`YCbCr8192`](Self::YCbCr8192)'s, the steps and the sums signed
    /// ([`filter_extended`](Self::filter_extended)).
    YCbCrExtended8192 = 9,
    /// `RGB10A2Unorm_sRGB`'s channels as the function decodes its held word
    /// (`ExtendedRange::Rgb10A2Srgb`): red, green and blue steps `k / 8192` of `[0, 1]`, which
    /// Metal's linear filter sums as [`YCbCr8192`](Self::YCbCr8192)'s, and alpha's 2-bit code
    /// `a` read `a / 3`, which it sums on the grid `16 · 255` (the code `1360 a`), each 2D term
    /// truncated by `2^4`; a `half` result's color steps rounded with ties away from zero (an
    /// Apple M3 Max under macOS 27.0.1: every channel of 4,096 2D samples of a 16 x 16 texture,
    /// `float` and `half` results). Its filter of more than one level is not measured.
    Rgb10A2Srgb = 10,
}

impl PackedTexels {
    /// Every value, in field order.
    pub const ALL: [PackedTexels; 10] = [
        PackedTexels::Rgb565,
        PackedTexels::Rgb5A1,
        PackedTexels::Rgba4,
        PackedTexels::Rgb10A2,
        PackedTexels::SmallFloat,
        PackedTexels::Argb8,
        PackedTexels::YCbCr2048,
        PackedTexels::YCbCr8192,
        PackedTexels::YCbCrExtended8192,
        PackedTexels::Rgb10A2Srgb,
    ];

    /// The packed channels of `pf`, where Metal's reads of them are measured, a two-plane YCbCr
    /// format's by its layout ([`of_ycbcr`](Self::of_ycbcr)).
    pub fn of(pf: PixelFormat) -> Option<PackedTexels> {
        use PixelFormat as P;
        if let Some(y) = crate::metal::format::ycbcr_format(pf) {
            return PackedTexels::of_ycbcr(y);
        }
        Some(match pf {
            P::B5G6R5Unorm => PackedTexels::Rgb565,
            P::A1BGR5Unorm | P::BGR5A1Unorm => PackedTexels::Rgb5A1,
            P::ABGR4Unorm => PackedTexels::Rgba4,
            P::RGB10A2Unorm | P::BGR10A2Unorm => PackedTexels::Rgb10A2,
            P::RG11B10Float | P::RGB9E5Float => PackedTexels::SmallFloat,
            P::ARGB8Unorm => PackedTexels::Argb8,
            P::BC1RGBA | P::BC2RGBA | P::BC3RGBA => PackedTexels::Argb8,
            P::PVRTCRGB2BPP
            | P::PVRTCRGB4BPP
            | P::PVRTCRGBA2BPP
            | P::PVRTCRGBA4BPP
            | P::ETC2RGB8
            | P::ETC2RGB8A1
            | P::EACRGBA8 => PackedTexels::Argb8,
            _ => return None,
        })
    }

    /// The field of a two-plane YCbCr format of layout `y`, by the grid of Metal's reads: the 8-
    /// and 10-bit converted formats' `1/2048`, the `_sRGB` twins' and the
    /// 12-bit formats' `1/8192`, the extended-range ones' signed `1/8192`, and the 10-bit codes'
    /// `RGB10A2Unorm`'s. None where the filter is not measured exactly: the packed 12-bit formats
    /// (Metal's sample of them is no bilinear sum of their reads), the PQ and HLG twins (whose
    /// reads lie on no grid), and the 8-bit codes, whose family is `RGBA8Unorm`'s with no field
    /// (`texel_format`).
    pub fn of_ycbcr(y: crate::metal::format::YCbCrFormat) -> Option<PackedTexels> {
        use crate::metal::format::YCbCrReads as R;
        let packed12 = y.bits == 12 && y.packed;
        Some(match y.reads {
            R::Converted if y.bits <= 10 => PackedTexels::YCbCr2048,
            R::Converted | R::Srgb if !packed12 => PackedTexels::YCbCr8192,
            R::ExtendedRange if !packed12 => PackedTexels::YCbCrExtended8192,
            R::Codes if y.bits == 10 => PackedTexels::Rgb10A2,
            _ => return None,
        })
    }

    /// Each channel's code scale `2^b - 1`, red to alpha (an absent alpha's 1, which its value
    /// 1 recovers), or `None` where the device's value is Metal's exactly: the small floats, the
    /// YCbCr fields and `RGB10A2Unorm_sRGB`'s, whose function decodes Metal's values itself.
    pub fn scales(self) -> Option<[u32; 4]> {
        Some(match self {
            PackedTexels::Rgb565 => [31, 63, 31, 1],
            PackedTexels::Rgb5A1 => [31, 31, 31, 1],
            PackedTexels::Rgba4 => [15; 4],
            PackedTexels::Rgb10A2 => [1023, 1023, 1023, 3],
            PackedTexels::Argb8 => [255; 4],
            PackedTexels::SmallFloat
            | PackedTexels::YCbCr2048
            | PackedTexels::YCbCr8192
            | PackedTexels::YCbCrExtended8192
            | PackedTexels::Rgb10A2Srgb => return None,
        })
    }

    /// This field in place.
    pub fn class(self) -> TexelFormat {
        TexelFormat((self as u32) << TexelFormat::PACKED)
    }

    /// Each channel's grid `D` of Metal's linear filter, red to alpha, where it is measured
    /// (module docs): the 12-bit grid of the channel's code widened by replicating its bits to 8,
    /// 10, or 12 bits, `16 · 255` for a channel of 1, 2, or 4 bits, `4 · 1023` for one of 5 or 10,
    /// `4095` for one of 6; the YCbCr fields' `2048` or `8192`, on which their steps are their
    /// codes (`S = D` below). Each
    /// is a multiple of the channel's code scale ([`scales`](Self::scales)), so a code `c` lies on
    /// it at `c · D / S`. `None` for the small floats, whose sum is a half-float format's, and for
    /// `ARGB8Unorm`, whose filter is not measured.
    pub fn filter_grids(self) -> Option<[u32; 4]> {
        const G8: u32 = 16 * 255;
        const G10: u32 = 4 * 1023;
        Some(match self {
            PackedTexels::Rgb565 => [G10, 4095, G10, G8],
            PackedTexels::Rgb5A1 | PackedTexels::Rgb10A2 => [G10, G10, G10, G8],
            PackedTexels::Rgba4 => [G8; 4],
            PackedTexels::YCbCr2048 => [2048; 4],
            PackedTexels::YCbCr8192 | PackedTexels::YCbCrExtended8192 => [8192; 4],
            PackedTexels::Rgb10A2Srgb => [8192, 8192, 8192, G8],
            PackedTexels::SmallFloat | PackedTexels::Argb8 => return None,
        })
    }

    /// The bits `s` Metal's 2D and 3D linear sums drop from each term where
    /// [`filter_grids`](Self::filter_grids) names the grids: the term of a code `c` on its grid
    /// under the weight `w` (in units of `2^-16`) is `⌊c · w / 2^s⌋`, and the sum's grid step
    /// `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋` (module docs). 4 for a packed normalized format's 12-bit
    /// grids, as for an 8-bit format's, and 7 for the YCbCr fields', measured on their 2D samples;
    /// `RGB10A2Unorm_sRGB`'s 7 for its color channels and 4 for alpha. Red to alpha.
    pub fn filter_truncation(self) -> Option<[u32; 4]> {
        self.filter_grids().map(|_| match self {
            PackedTexels::Rgb10A2Srgb => [7, 7, 7, 4],
            _ if self.ycbcr() => [7; 4],
            _ => [4; 4],
        })
    }

    /// Whether a `half` result of the field's one texel or linear filter rounds with ties away
    /// from zero: the two-plane YCbCr fields' ([`ycbcr`](Self::ycbcr)) and
    /// `RGB10A2Unorm_sRGB`'s, whose color steps lie on the grid `8192` (its alpha's values are
    /// never a tie).
    pub fn ties_away(self) -> bool {
        self.ycbcr() || self == PackedTexels::Rgb10A2Srgb
    }

    /// Whether the field is a two-plane YCbCr format's: its view holds Metal's reads exactly, and a
    /// `half` result of its linear filter is the step's value rounded with ties away from zero (on
    /// the grid `8192`, where a step's value may lie between two halves; the grid `2048`'s are
    /// halves), as is a `half` read or nearest sample of one texel (every channel of
    /// `YCBCR8_420_2P_sRGB`, `YCBCR10_420_2P_sRGB` and three 12-bit and
    /// extended-range formats on the grid `8192`).
    pub fn ycbcr(self) -> bool {
        matches!(
            self,
            PackedTexels::YCbCr2048 | PackedTexels::YCbCr8192 | PackedTexels::YCbCrExtended8192
        )
    }

    /// Whether the filter's codes and steps are signed and reach `±2D`
    /// ([`YCbCrExtended8192`](Self::YCbCrExtended8192)): each step divided by `2D` and doubled,
    /// exactly.
    pub fn filter_extended(self) -> bool {
        self == PackedTexels::YCbCrExtended8192
    }
}

/// The treatment of `pf`'s texels with a `half` result, or a depth format's cube gather (module
/// docs).
pub fn texel_format(pf: PixelFormat) -> TexelFormat {
    use PixelFormat as P;
    let measured = matches!(
        pf,
        P::R32Float
            | P::RG32Float
            | P::RGBA32Float
            | P::R16Float
            | P::RG16Float
            | P::RGBA16Float
            | P::R8Unorm
            | P::RG8Unorm
            | P::RGBA8Unorm
            | P::BGRA8Unorm
            | P::A8Unorm
            | P::R16Unorm
            | P::RG16Unorm
            | P::RGBA16Unorm
            | P::R8Snorm
            | P::RG8Snorm
            | P::RGBA8Snorm
            | P::R16Snorm
            | P::RG16Snorm
            | P::RGBA16Snorm
            | P::RGBX32Float
            | P::RGBX16Float
            | P::RGBX8Unorm
            | P::RGBX8Snorm
            // Read, sampled (nearest, linear and bicubic, `float` and `half`) and gathered as an
            // `RGBX8Unorm` texture of their decoded codes (an Apple M3 Max, `bind::ChromaDecode`).
            | P::GBGR422
            | P::BGRG422
    );
    // A depth format's cube gather: its one channel's family (module docs).
    match pf {
        P::Depth32Float => return texel_format(P::R32Float),
        P::Depth16Unorm => return texel_format(P::R16Unorm),
        _ => {}
    }
    match texels(pf) {
        Some(t) if measured => TexelFormat(t.0),
        // An sRGB format's family, its channels sRGB codes (module docs).
        Some(t) if srgb_coded(pf) => TexelFormat(t.0 | BicubicTexels::SRGB),
        // A cube gather's corner, measured alone, and a packed format's codes, with its family
        // where Metal's linear filter of it is measured: a packed normalized format's `EXACT`
        // sum on its field's grids, a small float's half-float one (module docs).
        t => {
            let corner = CubeCorner::of(pf).map_or(TexelFormat::OTHER, CubeCorner::class);
            let packed = PackedTexels::of(pf);
            let family = match (t, packed) {
                (Some(t), Some(p)) if p.filter_grids().is_some() => t.0 | BicubicTexels::EXACT,
                (Some(t), Some(PackedTexels::SmallFloat)) => t.0,
                (None, field) => two_plane_family(pf, field),
                _ => 0,
            };
            let packed = packed.map_or(TexelFormat::OTHER, PackedTexels::class);
            TexelFormat(family | corner.0 | packed.0)
        }
    }
}

/// A two-plane YCbCr format's family, where its filter is measured exactly: with a field of grids
/// ([`PackedTexels::of_ycbcr`]), the normalized family with `EXACT`, signed where the field's steps
/// are ([`PackedTexels::filter_extended`]); the 8-bit codes' (no field), the 8-bit unsigned
/// normalized family's, as `RGBA8Unorm`'s; each with its channels, red, green and blue (its alpha
/// reads 1, at the border too), and their border texels. 0 for any other format. The bicubic
/// filter's treatment ([`texels`]) has none: Metal's bicubic filter of them is not measured.
fn two_plane_family(pf: PixelFormat, field: Option<PackedTexels>) -> u32 {
    use crate::metal::format::YCbCrReads;
    let Some(y) = crate::metal::format::ycbcr_format(pf) else {
        return 0;
    };
    let sign = match field {
        Some(p) if p.filter_extended() => BicubicTexels::SNORM,
        Some(p) if p.filter_grids().is_some() => BicubicTexels::UNORM,
        None if y.reads == YCbCrReads::Codes && y.bits == 8 => BicubicTexels::UNORM,
        _ => return 0,
    };
    // Red, green and blue: every two-plane format reads alpha 1.
    let flags = sign | BicubicTexels::EXACT | 0b0111 << BicubicTexels::PRESENT;
    BicubicTexels(flags)
        .with_border(pf, TextureSwizzleChannels::DEFAULT)
        .0
}

/// The class of a texture of pixel format `pf` made with `swizzle`, with the border texels of
/// that swizzle where the class states its channels (a cube gather's corner alone states none):
/// [`texel_format`]'s bits, which a function whose slot does not apply the swizzle itself reads as
/// [`TexelFormat::swizzled`] where the swizzle is not the identity (the binder's `TexelFormat`
/// specialization).
pub fn with_border(pf: PixelFormat, swizzle: TextureSwizzleChannels) -> TexelFormat {
    match texel_format(pf) {
        class if !class.has_channels() => class,
        class => TexelFormat(BicubicTexels(class.0).with_border(pf, swizzle).0),
    }
}

/// The class a heap texture of pixel format `pf` made with `swizzle` carries in its handle-table
/// entry: a texture without a swizzle takes its slot's class ([`with_border`]); a swizzled one
/// takes [`TexelFormat::OTHER`], whose texels the function leaves to the device, so the classes a
/// heap names are [`heap_classes`]' few.
pub fn heap_class(pf: PixelFormat, swizzle: TextureSwizzleChannels) -> TexelFormat {
    match reads_unswizzled(pf, swizzle) {
        true => with_border(pf, swizzle),
        false => TexelFormat::OTHER,
    }
}

/// Whether a texture of pixel format `pf` made with `swizzle` reads every texel, border texels
/// included, as one without a swizzle does: each channel from itself, or, for a channel `pf` lacks
/// (its class's channels, else `metal::format::channels`), as the value it reads anyway (0 for
/// red, green and blue, 1 for alpha). The runtime reads such a swizzle where a texture's alpha reads 1 (a two-plane YCbCr
/// format's), which keeps the format's class: its field and family.
pub fn reads_unswizzled(pf: PixelFormat, swizzle: TextureSwizzleChannels) -> bool {
    use crate::metal::TextureSwizzle as S;
    // The class's channels where it states them (a two-plane format's, whatever this crate
    // holds of it), else the format's.
    let class = BicubicTexels(texel_format(pf).0);
    let present = match TexelFormat(class.0).has_channels() {
        true => std::array::from_fn(|c| class.has_channel(c as u32)),
        false => crate::metal::format::channels(pf).unwrap_or([true; 4]),
    };
    let own = [S::Red, S::Green, S::Blue, S::Alpha];
    let lacking = [S::Zero, S::Zero, S::Zero, S::One];
    let sources = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
    (0..4).all(|c| sources[c] == own[c] || (!present[c] && sources[c] == lacking[c]))
}

/// Every class a heap texture can carry ([`heap_class`] of each pixel format without a swizzle),
/// sorted and without repeats, so [`TexelFormat::OTHER`] (0) is first: a texture's handle-table
/// entry names its class by its index here ([`heap_class_index`]), which a function reading the
/// class looks up in its own copy of this table. The binder and the function come from one build,
/// whose `SEMANTICS_VERSION` changes with any class.
///
/// After them, from [`heap_extended_base`], one [`TexelFormat::OTHER`] per form of
/// `ExtendedRange::HEAP_READ` in its order: the index of a heap texture of that form, whose texels
/// a function decodes from its raw codes where the index names it, and which every routine that
/// reads the class takes as `OTHER`.
///
/// After those, from [`heap_numeric_base`], two more `OTHER` entries: the index of a heap texture
/// of a signed and of an unsigned integer color format ([`heap_numeric`]), which a read of another
/// numeric class tests to read the texture in its own class; then, from the own-only base, the
/// classes of the formats such a read does not take ([`crosses_classes`]: depth, stencil,
/// extended-range and two-plane formats), sorted, `OTHER` first, each the class of its own: a float
/// read's routine takes it as the class of the same value among the own ones, and only a crossing
/// float format's texture names an own class.
pub fn heap_classes() -> &'static [u32] {
    &heap_layout().classes
}

/// [`heap_classes`] and the indices where its parts start: the forms a heap reads, the numeric
/// markers, and the classes of the formats no read of another class takes.
struct HeapLayout {
    classes: Vec<u32>,
    extended: u32,
    numeric: u32,
    own_only: u32,
}

/// The heap's classes ([`heap_classes`]), built once.
fn heap_layout() -> &'static HeapLayout {
    static LAYOUT: std::sync::OnceLock<HeapLayout> = std::sync::OnceLock::new();
    LAYOUT.get_or_init(|| {
        let sorted = |classes: Vec<u32>| {
            let mut classes: Vec<u32> = std::iter::once(TexelFormat::OTHER.0)
                .chain(classes)
                .collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        let of = |pf: PixelFormat| heap_class(pf, TextureSwizzleChannels::DEFAULT).0;
        let mut classes = sorted(PixelFormat::ALL.iter().map(|&pf| of(pf)).collect());
        let index = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let extended = index(classes.len());
        classes.extend(
            crate::metal::format::ExtendedRange::HEAP_READ
                .iter()
                .map(|_| TexelFormat::OTHER.0),
        );
        let numeric = index(classes.len());
        classes.extend([TexelFormat::OTHER.0; 2]);
        let own_only = index(classes.len());
        classes.extend(sorted(
            PixelFormat::ALL
                .iter()
                .filter(|&&pf| !crosses_classes(pf))
                .map(|&pf| of(pf))
                .collect(),
        ));
        HeapLayout {
            classes,
            extended,
            numeric,
            own_only,
        }
    })
}

/// The index in [`heap_classes`] of the first form of `ExtendedRange::HEAP_READ`: a heap texture's
/// class index at or past it names the form `index - base` of that list.
pub fn heap_extended_base() -> u32 {
    heap_layout().extended
}

/// The index in [`heap_classes`] of a heap texture of a signed integer color format; the next one
/// is an unsigned integer color format's ([`heap_numeric`]).
pub fn heap_numeric_base() -> u32 {
    heap_layout().numeric
}

/// Whether Metal's read of a texture of `pf` in another numeric class than its format's is
/// measured: a color format of the float, signed or unsigned integer class, not a depth, stencil,
/// extended-range (bfloat included) or two-plane YCbCr one. A slot of another class takes such a
/// texture under its cross-class variant (`bind::plan::resources::cross_class`), and a heap group's
/// read of another class reads it in its own class ([`heap_numeric`]).
pub fn crosses_classes(pf: PixelFormat) -> bool {
    use crate::metal::format::{is_depth_stencil, sampled_type, ycbcr_format, ExtendedRange};
    use crate::pipeline::sym::SampledType;
    matches!(
        sampled_type(pf),
        Some(SampledType::Float | SampledType::Sint | SampledType::Uint)
    ) && !is_depth_stencil(pf)
        && ExtendedRange::of(pf).is_none()
        && ycbcr_format(pf).is_none()
}

/// The numeric class of `pf` where a heap texture of it carries a numeric marker
/// ([`heap_numeric_base`]): `Sint` or `Uint` for an integer format a read of another class takes
/// ([`crosses_classes`]), whatever its swizzle; `None` for every other format.
pub fn heap_numeric(pf: PixelFormat) -> Option<crate::pipeline::sym::SampledType> {
    use crate::pipeline::sym::SampledType;
    match crate::metal::format::sampled_type(pf) {
        Some(t @ (SampledType::Sint | SampledType::Uint)) if crosses_classes(pf) => Some(t),
        _ => None,
    }
}

/// The index in [`heap_classes`] of `A8Unorm`'s class, which no other format's texture carries:
/// a storage read or write through a heap handle carrying it converts the member's texels where
/// the device holds such a texture's alpha in red (`metal::format::alpha_in_red`), which a heap's
/// storage array views without a component mapping. A swizzled `A8Unorm` texture that reads as
/// none carries it too; another swizzle's class is `OTHER`, which a heap's storage array does not
/// hold there (`bind::heap`'s storage elements).
pub fn heap_alpha_in_red_index() -> u32 {
    heap_class_index(PixelFormat::A8Unorm, TextureSwizzleChannels::DEFAULT)
}

/// Whether a texture of pixel format `pf` holds 16-bit float channels of its own: its class is
/// the half-float family's without a packed field.
pub fn half_channels(pf: PixelFormat) -> bool {
    use crate::metal::bicubic::BicubicTexels;
    texel_format(pf).0 & (BicubicTexels::HALF | TexelFormat::PACKED_MASK) == BicubicTexels::HALF
}

/// Whether Metal's `textureWriteRoundingMode` 2 stores a fragment function's write to a texture of
/// `pf` otherwise than its default, where the translation builds only the default's rule
/// (`RuntimeFact::StorageTextureWriteRounding`, `RuntimeFact::HeapWriteRounding`): an
/// `RG11B10Float` or `RGB9E5Float` texture's, and where `halves` (a `float` texel) a half-float
/// one's ([`half_channels`]).
pub fn mode_two_rounds(pf: PixelFormat, halves: bool) -> bool {
    matches!(pf, PixelFormat::RG11B10Float | PixelFormat::RGB9E5Float)
        || (halves && half_channels(pf))
}

/// The index in [`heap_classes`] of the class a heap texture of pixel format `pf` made with
/// `swizzle` carries ([`heap_class`]); for an unswizzled texture of a form of
/// `ExtendedRange::HEAP_READ`, that form's index past [`heap_extended_base`]; for an integer color
/// format, its numeric marker ([`heap_numeric_base`]), and for a format a read of another class
/// does not take, its class among those past the own-only base ([`heap_classes`]).
pub fn heap_class_index(pf: PixelFormat, swizzle: TextureSwizzleChannels) -> u32 {
    let layout = heap_layout();
    match heap_numeric(pf) {
        Some(crate::pipeline::sym::SampledType::Sint) => return layout.numeric,
        Some(_) => return layout.numeric + 1,
        None => {}
    }
    let base = layout.extended;
    let read = crate::metal::format::ExtendedRange::of(pf).and_then(|f| f.heap_read());
    if let (Some(k), true) = (read, swizzle.is_default()) {
        return base + k;
    }
    let (from, own) = match crosses_classes(pf) {
        true => (0, &layout.classes[..layout.extended as usize]),
        false => (layout.own_only, &layout.classes[layout.own_only as usize..]),
    };
    let class = heap_class(pf, swizzle).0;
    own.binary_search(&class)
        .map_or(from, |i| from + u32::try_from(i).unwrap_or(0))
}

/// Whether `pf`'s color channels are 8-bit sRGB codes, which Metal decodes through its twelve-bit
/// table (module docs): the 8-bit sRGB formats, and the sRGB forms of BC1 to BC3, BC7, PVRTC, ETC2,
/// EAC and ASTC, whose blocks decode to such codes.
pub fn srgb_coded(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        pf,
        P::RGBA8UnormSRGB
            | P::BGRA8UnormSRGB
            | P::R8UnormSRGB
            | P::RG8UnormSRGB
            | P::RGBX8UnormSRGB
            | P::BC1RGBASRGB
            | P::BC2RGBASRGB
            | P::BC3RGBASRGB
            | P::BC7RGBAUnormSRGB
            | P::PVRTCRGB2BPPSRGB
            | P::PVRTCRGB4BPPSRGB
            | P::PVRTCRGBA2BPPSRGB
            | P::PVRTCRGBA4BPPSRGB
            | P::ETC2RGB8SRGB
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
            | P::ASTC12x12SRGB
    )
}

/// Whether `profile`'s device holds a sampled texture of `pf` as its 8-bit sRGB format's UNORM twin
/// (`metal::format::srgb_raw_twin`): it samples no format of `pf`'s and samples the twin's, whose
/// views read the codes, so a function decodes them itself (module docs). The runtime's image plan
/// holds the twin by the same format features.
pub fn srgb_twin_on(pf: PixelFormat, profile: &crate::device::profile::DeviceProfile) -> bool {
    use crate::device::requirements::FormatUsage;
    let usages = [FormatUsage::SampledImage].into();
    let samples = |pf| crate::metal::format::vk_format(pf, &usages, profile).is_ok();
    !samples(pf) && crate::metal::format::srgb_raw_twin(pf).is_some_and(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use BicubicTexels as T;

    /// An sRGB-coded format's twin (`metal::format::srgb_raw_twin`) is a format whose direct image
    /// format is the UNORM format of the sRGB format's own class: the same channels and bits, read
    /// as their codes; nothing else has one, nor an ASTC sRGB format, whose unorm form decodes the
    /// blocks otherwise.
    #[test]
    fn an_srgb_format_s_twin_holds_its_codes() {
        for &pf in PixelFormat::ALL {
            match crate::metal::format::srgb_raw_twin(pf) {
                Some(twin) => {
                    assert!(srgb_coded(pf), "{pf:?}");
                    let own = crate::metal::format::direct_format(pf).and_then(|f| f.block());
                    let codes = crate::metal::format::direct_format(twin).and_then(|f| f.block());
                    // A PVRTC RGB format and its twin are held only through decoded twins.
                    let (own, codes) = match (own, codes) {
                        (Some(own), Some(codes)) => (own, codes),
                        (None, None) => continue,
                        _ => panic!("{pf:?}: one of the pair without a direct format"),
                    };
                    assert_eq!(own.color, Some(crate::vk::NumericFormat::Srgb), "{pf:?}");
                    assert_eq!(codes.color, Some(crate::vk::NumericFormat::Unorm), "{pf:?}");
                    assert_eq!(
                        (own.class, own.size, own.component_bits),
                        (codes.class, codes.size, codes.component_bits),
                        "{pf:?}"
                    );
                }
                None => assert!(
                    !srgb_coded(pf) || crate::bind::AstcDecode::of(pf).is_some(),
                    "{pf:?}"
                ),
            }
        }
    }

    /// A 4:2:2 format reads, samples and gathers as an `RGBA8Unorm` texture of its decoded codes
    /// with alpha 1 (our probe on an Apple M3 Max under macOS 27.0.1, `bind::ChromaDecode`):
    /// `RGBX8Unorm`'s class, its border texels too.
    #[test]
    fn four_two_two_formats_convert_as_rgbx8_unorm() {
        for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
            assert_eq!(
                texel_format(pf),
                texel_format(PixelFormat::RGBX8Unorm),
                "{pf:?}"
            );
        }
    }

    /// An RGBX format converts as its RGBA twin with red, green and blue (every
    /// read, sample, gather and bicubic sample equals the twin's but for alpha 1): its family,
    /// an `_sRGB` one's sRGB codes.
    #[test]
    fn rgbx_formats_convert_as_their_twins_with_three_channels() {
        use PixelFormat as P;
        let family = |pf| texel_format(pf).0 & !(0xF << T::PRESENT | 0xfff << T::BORDER);
        for (pf, twin) in [
            (P::RGBX8Unorm, P::RGBA8Unorm),
            (P::RGBX8UnormSRGB, P::RGBA8UnormSRGB),
            (P::RGBX8Snorm, P::RGBA8Snorm),
            (P::RGBX16Float, P::RGBA16Float),
            (P::RGBX32Float, P::RGBA32Float),
        ] {
            assert_eq!(family(pf), family(twin), "{pf:?}");
            let class = BicubicTexels(texel_format(pf).0);
            assert_eq!(
                (0..4).map(|c| class.has_channel(c)).collect::<Vec<_>>(),
                [true, true, true, false],
                "{pf:?}"
            );
            assert_eq!(srgb_coded(pf), srgb_coded(twin), "{pf:?}");
        }
        // The integer formats, as their twins, convert by no family.
        for pf in [P::RGBX8Uint, P::RGBX16Sint, P::RGBX32Uint] {
            assert_eq!(texels(pf), None, "{pf:?}");
        }
    }

    /// Each measured family keeps its bicubic bits, an 8-bit or BC sRGB format its own; every
    /// other format is `OTHER`.
    #[test]
    fn the_measured_families_take_their_bicubic_bits() {
        let family = |pf| texel_format(pf).0 & !(0xF << T::PRESENT | 0xfff << T::BORDER);
        assert_eq!(family(PixelFormat::RGBA32Float), T::LINEAR);
        assert_eq!(family(PixelFormat::RG16Float), T::HALF);
        assert_eq!(family(PixelFormat::BGRA8Unorm), T::UNORM | T::EXACT);
        assert_eq!(
            family(PixelFormat::R16Snorm),
            T::SNORM | T::EXACT | T::WIDE16
        );
        // The depth formats whose cube gather is measured (module docs).
        assert_eq!(
            texel_format(PixelFormat::Depth32Float),
            texel_format(PixelFormat::R32Float)
        );
        assert_eq!(
            texel_format(PixelFormat::Depth16Unorm),
            texel_format(PixelFormat::R16Unorm)
        );
        let a8 = BicubicTexels(texel_format(PixelFormat::A8Unorm).0);
        assert!(a8.has_channel(3) && !a8.has_channel(0));
        // An 8-bit sRGB format: normalized, sRGB, its channels, never `EXACT` (module docs).
        for (pf, channels) in [
            (PixelFormat::RGBA8UnormSRGB, 0xF),
            (PixelFormat::BGRA8UnormSRGB, 0xF),
            (PixelFormat::RG8UnormSRGB, 0x3),
            (PixelFormat::R8UnormSRGB, 0x1),
            (PixelFormat::BC1RGBASRGB, 0xF),
            (PixelFormat::BC7RGBAUnormSRGB, 0xF),
            (PixelFormat::PVRTCRGBA2BPPSRGB, 0xF),
            (PixelFormat::PVRTCRGB4BPPSRGB, 0x7),
            (PixelFormat::ETC2RGB8SRGB, 0x7),
            (PixelFormat::ETC2RGB8A1SRGB, 0xF),
            (PixelFormat::EACRGBA8SRGB, 0xF),
            (PixelFormat::ASTC4x4SRGB, 0xF),
            (PixelFormat::ASTC12x12SRGB, 0xF),
        ] {
            assert_eq!(
                texel_format(pf).0 & !(0xfff << T::BORDER),
                T::UNORM | T::SRGB | channels << T::PRESENT,
                "{pf:?}"
            );
        }
        for pf in [
            PixelFormat::BGRA10XRSRGB,
            PixelFormat::RGBA8Uint,
            PixelFormat::ASTC4x4LDR,
        ] {
            assert_eq!(texel_format(pf), TexelFormat::OTHER, "{pf:?}");
        }
    }

    /// A format of a measured cube corner outside the families is its corner's field alone (no
    /// family, channel, or border bit, so every other access reads it as `OTHER`), but for a
    /// packed format's family and field beside it; each field fits its four bits, and each grid
    /// suits `texel_round::grid_value` (`1 <= D < 2^16`).
    #[test]
    fn a_cube_corner_alone_is_its_field() {
        let mut seen = std::collections::BTreeSet::new();
        for &pf in PixelFormat::ALL {
            let Some(corner) = CubeCorner::of(pf) else {
                assert_eq!(texel_format(pf).0 & TexelFormat::CORNER_MASK, 0, "{pf:?}");
                continue;
            };
            // A packed format's class holds its family beside its corner; a swizzled texture's
            // of a packed normalized one keeps the corner alone.
            let class = texel_format(pf);
            assert_eq!(
                class.0 & TexelFormat::CORNER_MASK,
                corner.class().0,
                "{pf:?}"
            );
            let alone = match PackedTexels::of(pf) {
                Some(_) => TexelFormat(class.0 & TexelFormat::CORNER_MASK),
                None => class,
            };
            if class.packed_exact() {
                assert_eq!(class.swizzled(), alone, "{pf:?}");
            }
            assert_eq!(alone, corner.class(), "{pf:?}");
            assert!(!alone.has_channels(), "{pf:?}");
            seen.insert(corner as u32);
        }
        let every = [
            CubeCorner::XFace,
            CubeCorner::Grid255,
            CubeCorner::Grid1023,
            CubeCorner::Grid565,
            CubeCorner::Grid2047,
            CubeCorner::Signed1023,
            CubeCorner::Grid16320,
            CubeCorner::Signed8128,
        ];
        for (k, corner) in every.into_iter().enumerate() {
            assert_eq!(corner as u32, k as u32 + 1);
            assert_eq!(corner.class().0 >> TexelFormat::CORNER, corner as u32);
            assert!(corner.class().0 & TexelFormat::CORNER_MASK == corner.class().0);
            assert!(seen.contains(&(corner as u32)), "{corner:?} has no format");
        }
        // The measured families keep no corner field.
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::Depth32Float,
            PixelFormat::R16Float,
        ] {
            assert_eq!(texel_format(pf).0 & TexelFormat::CORNER_MASK, 0, "{pf:?}");
            assert!(texel_format(pf).has_channels(), "{pf:?}");
        }
    }

    /// Each packed format takes its field beside its corner, and nothing else does: the field fits
    /// its four bits, each value has a format, each code scale suits `texel_round::grid_value` (`1
    /// <= S < 2^16`), and the field touches no other field's bit. `ARGB8Unorm`'s cube gathers and
    /// linear filter are not measured: its field stands alone, as the linear PVRTC formats' does;
    /// BC1 to BC3's unorm forms and the 8-bit linear ETC2 and EAC formats take it beside their
    /// corner. No cube gather of a two-plane YCbCr format is measured: its field has
    /// no corner. A packed normalized format's class is its normalized family's with `EXACT`, a
    /// small float's the half-float family's, each with its channels and border texels, and a
    /// two-plane format's the normalized family's with `EXACT` (signed where its field's steps are)
    /// and its format's channels. A field with grids states its sums' truncation, and only it.
    #[test]
    fn a_packed_format_takes_its_field_beside_its_corner() {
        let mut seen = std::collections::BTreeSet::new();
        for &pf in PixelFormat::ALL {
            let class = texel_format(pf);
            let field = class.0 & TexelFormat::PACKED_MASK;
            match PackedTexels::of(pf) {
                Some(p) => {
                    assert_eq!(field, p.class().0, "{pf:?}");
                    let two_plane = crate::metal::format::ycbcr_format(pf).is_some();
                    let blocks = matches!(
                        pf,
                        PixelFormat::BC1RGBA
                            | PixelFormat::BC2RGBA
                            | PixelFormat::BC3RGBA
                            | PixelFormat::ETC2RGB8
                            | PixelFormat::ETC2RGB8A1
                            | PixelFormat::EACRGBA8
                    );
                    assert_eq!(
                        CubeCorner::of(pf).is_some(),
                        (p != PackedTexels::Argb8 || blocks) && !two_plane,
                        "{pf:?}"
                    );
                    assert_eq!(p.ycbcr(), two_plane && p != PackedTexels::Rgb10A2, "{pf:?}");
                    let family = class.0 & !(TexelFormat::PACKED_MASK | TexelFormat::CORNER_MASK);
                    let want = match p {
                        PackedTexels::SmallFloat => T::HALF,
                        PackedTexels::Argb8 => 0,
                        _ if p.filter_extended() => T::SNORM | T::EXACT,
                        _ => T::UNORM | T::EXACT,
                    };
                    assert_eq!(family & !(0xF << T::PRESENT | 0xfff << T::BORDER), want);
                    // A two-plane format's channels are its format's, which the bicubic filter's
                    // treatment does not state: red, green and blue, alpha reading 1 at the border.
                    let t = match two_plane {
                        true => {
                            assert!(texels(pf).is_none(), "{pf:?}");
                            T(0b0111 << T::PRESENT)
                                .with_border(pf, TextureSwizzleChannels::DEFAULT)
                                .0
                        }
                        false => texels(pf).map_or(0, |t| t.0),
                    };
                    assert_eq!(family, t | want, "{pf:?}");
                    assert_eq!(class.packed_exact(), p.filter_grids().is_some(), "{pf:?}");
                    seen.insert(p as u32);
                }
                None => {
                    assert_eq!(field, 0, "{pf:?}");
                    assert!(!class.packed_exact(), "{pf:?}");
                }
            }
        }
        // The two-plane YCbCr formats by the grid of Metal's reads: 9 on
        // `1/2048`, 12 on `1/8192`, 9 on the signed `1/8192`, 6 of 10-bit codes; the 3 of 8-bit
        // codes `RGBA8Unorm`'s family without a field; the 30 others (PQ, HLG, packed 12 bits) no
        // class.
        use crate::metal::format::YCbCrReads;
        let mut by = std::collections::BTreeMap::new();
        for &pf in PixelFormat::ALL {
            let Some(y) = crate::metal::format::ycbcr_format(pf) else {
                continue;
            };
            let class = texel_format(pf);
            let key = match PackedTexels::of(pf) {
                Some(p) => format!("{p:?}"),
                None if class.0 == 0 => "other".to_owned(),
                None => {
                    let rgb = T(T::UNORM | T::EXACT | 0b0111 << T::PRESENT)
                        .with_border(pf, TextureSwizzleChannels::DEFAULT);
                    assert_eq!(class.0, rgb.0, "{pf:?}");
                    assert_eq!((y.bits, y.reads), (8, YCbCrReads::Codes), "{pf:?}");
                    "codes8".to_owned()
                }
            };
            *by.entry(key).or_insert(0) += 1;
        }
        let want = [
            ("Rgb10A2", 6),
            ("YCbCr2048", 9),
            ("YCbCr8192", 12),
            ("YCbCrExtended8192", 9),
            ("codes8", 3),
            ("other", 30),
        ];
        assert_eq!(
            by,
            want.map(|(k, n)| (k.to_owned(), n)).into_iter().collect()
        );
        for (pf, p) in [
            (
                PixelFormat::YCBCR104442PPACKED,
                Some(PackedTexels::YCbCr2048),
            ),
            (
                PixelFormat::YCBCR104202PPACKEDSRGB,
                Some(PackedTexels::YCbCr8192),
            ),
            (PixelFormat::YCBCR124222P, Some(PackedTexels::YCbCr8192)),
            (
                PixelFormat::YCBCR104202PPACKEDXR,
                Some(PackedTexels::YCbCrExtended8192),
            ),
            (
                PixelFormat::YCBCR124442PXR,
                Some(PackedTexels::YCbCrExtended8192),
            ),
            (PixelFormat::YCBCR124202PPACKED, None),
            (PixelFormat::YCBCR124202PPACKEDXR, None),
            (PixelFormat::YCBCR104202PPQ, None),
            (PixelFormat::YCBCR124442PHLG, None),
            (PixelFormat::RGB104202PPACKED, Some(PackedTexels::Rgb10A2)),
        ] {
            assert_eq!(PackedTexels::of(pf), p, "{pf:?}");
        }
        for (k, p) in PackedTexels::ALL.into_iter().enumerate() {
            assert_eq!(p as u32, k as u32 + 1);
            assert_eq!(p.class().0 & !TexelFormat::PACKED_MASK, 0);
            for s in p.scales().into_iter().flatten() {
                assert!((1..1 << 16).contains(&s), "{p:?}");
            }
            // Each filter grid lies on its channel's code scale, below `2^12`.
            if let (Some(s), Some(g)) = (p.scales(), p.filter_grids()) {
                for (s, g) in s.into_iter().zip(g) {
                    assert_eq!(g % s, 0, "{p:?}");
                    assert!(g < 1 << 12, "{p:?}");
                }
            }
            assert_eq!(
                p.filter_truncation().is_some(),
                p.filter_grids().is_some(),
                "{p:?}"
            );
            // `RGB10A2Unorm_sRGB`'s field is its extended-range form's class's
            // (`lower::sampler::extended`), which no format's own class takes.
            let extended = p == PackedTexels::Rgb10A2Srgb;
            assert!(
                seen.contains(&(p as u32)) || extended,
                "{p:?} has no format"
            );
        }
        // A swizzle drops the field, and a packed normalized format's family with it.
        for (pf, alone) in [
            (PixelFormat::B5G6R5Unorm, true),
            (PixelFormat::RGB10A2Unorm, true),
            (PixelFormat::RG11B10Float, false),
            (PixelFormat::ARGB8Unorm, false),
            (PixelFormat::RGBA8Unorm, false),
        ] {
            let class = texel_format(pf);
            let swizzled = class.swizzled();
            assert_eq!(swizzled.0 & TexelFormat::PACKED_MASK, 0, "{pf:?}");
            let want = match alone {
                true => class.0 & TexelFormat::CORNER_MASK,
                false => class.0 & !TexelFormat::PACKED_MASK,
            };
            assert_eq!(swizzled.0, want, "{pf:?}");
        }
        assert_eq!(TexelFormat::PACKED_MASK & TexelFormat::CORNER_MASK, 0);
        assert_eq!(TexelFormat::PACKED_MASK & 0xff_ffff, 0);
    }

    /// The heap classes are few enough for a handle-table entry's class field, `OTHER` first; each
    /// unswizzled format's index names its slot class with its border texels, and a swizzled
    /// texture's names `OTHER`. An unswizzled texture of a form a heap reads names that form past
    /// the base, an `OTHER` entry; a swizzled one `OTHER`'s. An integer color format names its
    /// numeric marker; a format a read of another class does not take, its class past the own-only
    /// base, so only a crossing float format's texture names an own class (below the extended
    /// base).
    #[test]
    fn a_heap_texture_names_its_class_by_index() {
        use crate::memory::handles::FLAG_TEXEL_CLASS;
        use crate::metal::format::ExtendedRange;
        let classes = heap_classes();
        let base = heap_extended_base() as usize;
        assert_eq!(classes.first(), Some(&TexelFormat::OTHER.0));
        assert!(classes[..base].windows(2).all(|w| w[0] < w[1]));
        let numeric = heap_numeric_base() as usize;
        assert_eq!(numeric, base + ExtendedRange::HEAP_READ.len());
        let own_only = heap_layout().own_only as usize;
        assert_eq!(own_only, numeric + 2);
        assert!(classes[base..own_only]
            .iter()
            .all(|&c| c == TexelFormat::OTHER.0));
        assert_eq!(classes.get(own_only), Some(&TexelFormat::OTHER.0));
        assert!(classes[own_only..].windows(2).all(|w| w[0] < w[1]));
        let field = FLAG_TEXEL_CLASS >> FLAG_TEXEL_CLASS.trailing_zeros();
        assert!(
            classes.len() <= field as usize + 1,
            "{} classes",
            classes.len()
        );
        let swizzle = TextureSwizzleChannels::new(
            crate::metal::TextureSwizzle::Blue,
            crate::metal::TextureSwizzle::Green,
            crate::metal::TextureSwizzle::Red,
            crate::metal::TextureSwizzle::Alpha,
        );
        for &pf in PixelFormat::ALL {
            let i = heap_class_index(pf, TextureSwizzleChannels::DEFAULT);
            // An integer color format names its numeric marker, swizzled or not, and loses no
            // class: its own is `OTHER`.
            let marker = match heap_numeric(pf) {
                Some(crate::pipeline::sym::SampledType::Sint) => Some(numeric),
                Some(_) => Some(numeric + 1),
                None => None,
            };
            if let Some(m) = marker {
                assert_eq!(i as usize, m, "{pf:?}");
                assert_eq!(heap_class_index(pf, swizzle) as usize, m, "{pf:?}");
                assert_eq!(
                    with_border(pf, TextureSwizzleChannels::DEFAULT),
                    TexelFormat::OTHER,
                    "{pf:?}"
                );
                continue;
            }
            match ExtendedRange::of(pf).and_then(ExtendedRange::heap_read) {
                Some(k) => assert_eq!(i as usize, base + k as usize, "{pf:?}"),
                None => {
                    assert_eq!(
                        classes[i as usize],
                        with_border(pf, TextureSwizzleChannels::DEFAULT).0,
                        "{pf:?}"
                    );
                    assert_eq!((i as usize) < base, crosses_classes(pf), "{pf:?}");
                }
            }
            let swizzled = match crosses_classes(pf) {
                true => 0,
                false => own_only,
            };
            assert_eq!(heap_class_index(pf, swizzle) as usize, swizzled, "{pf:?}");
        }
        assert!(heap_numeric(PixelFormat::RGBA8Uint).is_some());
        assert!(heap_numeric(PixelFormat::RGBA32Sint).is_some());
        assert!(heap_numeric(PixelFormat::RGBX8Uint).is_some());
        assert!(heap_numeric(PixelFormat::Stencil8).is_none());
        for pf in [
            PixelFormat::Stencil8,
            PixelFormat::Depth32Float,
            PixelFormat::Depth16Unorm,
        ] {
            let i = heap_class_index(pf, TextureSwizzleChannels::DEFAULT) as usize;
            assert!(i >= own_only, "{pf:?}");
        }
        // The measured families each carry a class of their own.
        let rgba8 = heap_class_index(PixelFormat::RGBA8Unorm, TextureSwizzleChannels::DEFAULT);
        let r32 = heap_class_index(PixelFormat::R32Float, TextureSwizzleChannels::DEFAULT);
        assert!(rgba8 != 0 && r32 != 0 && rgba8 != r32);
    }

    /// `A8Unorm`'s heap class index, which a heap storage access tests to convert a member held in
    /// red, names no other format's texture, swizzled or not, nor `OTHER`.
    #[test]
    fn the_alpha_in_red_heap_class_is_a8_unorms_alone() {
        use crate::metal::TextureSwizzle as S;
        let a8 = heap_alpha_in_red_index();
        assert_ne!(heap_classes()[a8 as usize], TexelFormat::OTHER.0);
        let swizzles = [
            TextureSwizzleChannels::DEFAULT,
            TextureSwizzleChannels::new(S::Blue, S::Green, S::Red, S::Alpha),
            TextureSwizzleChannels::new(S::Alpha, S::Alpha, S::Alpha, S::Alpha),
            TextureSwizzleChannels::new(S::Red, S::Red, S::Red, S::One),
        ];
        for &pf in PixelFormat::ALL {
            if pf == PixelFormat::A8Unorm {
                continue;
            }
            for swizzle in swizzles {
                assert_ne!(heap_class_index(pf, swizzle), a8, "{pf:?} {swizzle:?}");
            }
        }
        let alpha_only = TextureSwizzleChannels::new(S::Zero, S::Zero, S::Zero, S::Alpha);
        assert_eq!(heap_class_index(PixelFormat::A8Unorm, alpha_only), a8);
        let moved = TextureSwizzleChannels::new(S::Alpha, S::Zero, S::Zero, S::One);
        assert_ne!(heap_class_index(PixelFormat::A8Unorm, moved), a8);
    }

    /// The heap classes of the half-float family without a packed field are the unswizzled
    /// half-float formats' and no other format's: `RGBX16Float`'s too, `RGBA16Float`'s texels with
    /// three channels.
    #[test]
    fn a_heap_write_finds_the_half_formats_by_index() {
        use crate::metal::bicubic::BicubicTexels;
        let half: Vec<u32> = heap_classes()
            .iter()
            .enumerate()
            .filter(|(_, &c)| {
                c & (BicubicTexels::HALF | TexelFormat::PACKED_MASK) == BicubicTexels::HALF
            })
            .filter_map(|(i, _)| u32::try_from(i).ok())
            .collect();
        let wanted = [
            PixelFormat::R16Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::RGBX16Float,
        ];
        assert_eq!(half.len(), wanted.len(), "{half:?}");
        for &pf in PixelFormat::ALL {
            let i = heap_class_index(pf, TextureSwizzleChannels::DEFAULT);
            assert_eq!(half.contains(&i), wanted.contains(&pf), "{pf:?}");
        }
    }

    /// The bits of a class the `half` routine of a device that samples through Metal's own
    /// texture unit reads (`lower::texture::half`).
    const READ_ON_METAL_SAMPLER: u32 = T::LINEAR | T::SNORM | T::WIDE16 | TexelFormat::YCBCR_8192;

    /// [`TexelFormat::YCBCR_8192`] is set in exactly the fields of the grid `8192`, the packed
    /// fields whose one texel's `half` rounds with ties away from zero.
    #[test]
    fn the_ycbcr_8192_bit_names_exactly_the_grid_8192_fields() {
        for p in PackedTexels::ALL {
            let field = (p as u32) << TexelFormat::PACKED;
            assert_eq!(field & TexelFormat::PACKED_MASK, field, "{p:?}");
            assert_eq!(
                field & TexelFormat::YCBCR_8192 != 0,
                matches!(
                    p,
                    PackedTexels::YCbCr8192
                        | PackedTexels::YCbCrExtended8192
                        | PackedTexels::Rgb10A2Srgb
                ),
                "{p:?}"
            );
            assert_eq!(
                field & TexelFormat::YCBCR_8192 != 0,
                p.ties_away() && p != PackedTexels::YCbCr2048,
                "{p:?}"
            );
        }
    }

    /// A 16-bit grid's class (`WIDE16`) is an exact family's with no packed field, so the Metal
    /// sampler's routine, which reads the grid's bit and not the packed field, selects its grid
    /// for nothing else.
    #[test]
    fn a_wide16_class_is_exact_and_unpacked() {
        for &pf in PixelFormat::ALL {
            let c = texel_format(pf).0;
            if c & T::WIDE16 != 0 {
                assert_ne!(c & T::EXACT, 0, "{pf:?}");
                assert_eq!(c & TexelFormat::PACKED_MASK, 0, "{pf:?}");
                assert_eq!(c & T::LINEAR, 0, "{pf:?}");
            }
        }
    }

    /// On a device that samples through Metal's own texture unit the 8-bit unsigned normalized,
    /// half-float, sRGB, packed, small float and YCbCr-2048 formats' classes, swizzled or not,
    /// read as a slot bound to nothing does, and are measured; the 32-bit float and 16-bit grids',
    /// an 8-bit signed format's, and a YCbCr-8192 field's do not; a compressed format's class is
    /// unmeasured.
    #[test]
    fn the_metal_sampler_s_read_bits_fold_the_measured_families() {
        let folded = [
            PixelFormat::RGBA8Unorm,
            PixelFormat::BGRA8Unorm,
            PixelFormat::R8Unorm,
            PixelFormat::RG8Unorm,
            PixelFormat::A8Unorm,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::R16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::B5G6R5Unorm,
            PixelFormat::RGB10A2Unorm,
            PixelFormat::RG11B10Float,
            PixelFormat::RGB9E5Float,
        ];
        for pf in folded {
            let c = texel_format(pf);
            assert_eq!(c.0 & READ_ON_METAL_SAMPLER, 0, "{pf:?}");
            assert!(!c.unmeasured_on_metal_sampler(), "{pf:?}");
            assert_eq!(c.swizzled().0 & READ_ON_METAL_SAMPLER, 0, "{pf:?}");
        }
        for pf in [
            PixelFormat::RGBA8Snorm,
            PixelFormat::R16Unorm,
            PixelFormat::RGBA16Snorm,
            PixelFormat::RGBA32Float,
        ] {
            let c = texel_format(pf);
            assert_ne!(c.0 & READ_ON_METAL_SAMPLER, 0, "{pf:?}");
            assert!(!c.unmeasured_on_metal_sampler(), "{pf:?}");
        }
        for p in [PackedTexels::YCbCr8192, PackedTexels::YCbCrExtended8192] {
            let c = TexelFormat((p as u32) << TexelFormat::PACKED);
            assert_ne!(c.0 & READ_ON_METAL_SAMPLER, 0, "{p:?}");
            assert!(!c.unmeasured_on_metal_sampler(), "{p:?}");
        }
        let ycbcr = TexelFormat((PackedTexels::YCbCr2048 as u32) << TexelFormat::PACKED);
        assert_eq!(ycbcr.0 & READ_ON_METAL_SAMPLER, 0);
        assert!(!ycbcr.unmeasured_on_metal_sampler());
        assert!(texel_format(PixelFormat::BC7RGBAUnorm).unmeasured_on_metal_sampler());
        assert!(!texel_format(PixelFormat::BC1RGBA).unmeasured_on_metal_sampler());
        assert!(TexelFormat(0).unmeasured_on_metal_sampler());
    }
}
