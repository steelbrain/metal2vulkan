//! Pixel-format mapping: `MTLPixelFormat` → Vulkan image formats, storage-image choices, and
//! texture view types.
//!
//! Each Metal pixel format has a fixed, ordered list of **candidate** representations, chosen from
//! its bit layout alone (never from a device name). [`vk_format`] returns the first candidate that
//! is valid for every requested [`FormatUsage`] and that the profile's format tables support, and
//! otherwise a typed refusal naming what each candidate lacks. Every candidate is exact for the
//! usages it admits, except [`FormatEmulation::DepthWidened`], which the mapping reports so the
//! caller can decide.
//!
//! The Metal packed formats are named from the least significant bit (`B5G6R5Unorm` holds blue in
//! bits 0–4); Vulkan packed formats are named from the most significant bit. The table below pairs
//! them by layout (`B5G6R5Unorm` ↔ `R5G6B5_UNORM_PACK16`).
//!
//! A candidate whose Vulkan format enum value a core version or an extension defines
//! (`A8_UNORM`, the ASTC HDR and PVRTC blocks) is usable only where the profile exposes the value
//! ([`vk::Format::exposure`]): `Device::new` keeps only exposed formats in the profile's tables,
//! and a refusal names the version or extension that would expose it.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::bind::neutral::numeric_type;
use crate::device::enables::format_exposure_gap;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{Capability, DescriptorError, DriverDeviation, Error, Feature, Unsupported};
use crate::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use crate::pipeline::lower_inputs::{
    AliasView, ChannelWrite, StorageImageChoice, StorageImageClass,
};
use crate::pipeline::sym::SampledType;
use crate::vk;

pub(crate) mod extended_srgb;
pub(crate) mod padded_srgb;
mod word;
pub use word::{WordAttachment, WordField};

/// How a Metal pixel format is represented on a device, for a set of usages.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatMapping {
    /// The format of the Vulkan image (and of its views).
    pub format: vk::Format,
    /// The view's component mapping that presents Metal's channels (identity unless
    /// [`FormatEmulation::SwizzledView`] or [`FormatEmulation::AlphaInRed`]). Texture swizzles
    /// the guest requests compose on top.
    pub components: vk::ComponentMapping,
    /// The aspects the Metal format has (a sampled view selects one of them).
    pub aspects: vk::ImageAspectFlags,
    pub emulation: FormatEmulation,
}

impl FormatMapping {
    /// A direct mapping: the image is `format` itself, viewed with the identity component
    /// mapping, with the aspects `aspects` ([`FormatEmulation::Direct`]). For an image the
    /// consumer makes in a format it chose rather than a Metal pixel format's, such as a stand-in
    /// attachment or a scratch image.
    pub fn direct(format: vk::Format, aspects: vk::ImageAspectFlags) -> FormatMapping {
        FormatMapping {
            format,
            components: vk::ComponentMapping::IDENTITY,
            aspects,
            emulation: FormatEmulation::Direct,
        }
    }
}

/// How a candidate differs from a direct one-to-one Vulkan format.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FormatEmulation {
    /// The Vulkan format has exactly the Metal format's layout and numeric format.
    Direct,
    /// The bytes are read through a Vulkan format with the same bit layout but a different channel
    /// order, and [`FormatMapping::components`] moves the channels back. Exact for sampling, except that
    /// Vulkan leaves an opaque-black border read through a non-identity mapping undefined unless
    /// the device swizzles borders
    /// ([`check_border_swizzle`](crate::metal::sampler::check_border_swizzle)); admitted only for
    /// sampled-image usages, because texel buffers, storage images, and attachments ignore the
    /// component mapping.
    SwizzledView,
    /// `A8Unorm`'s byte held in `R8_UNORM`, on a device without `A8_UNORM` for the usages (it is
    /// core only from Vulkan 1.4, else `VK_KHR_maintenance5`): sampled through
    /// [`FormatMapping::components`] `(Zero, Zero, Zero, R)` as [`FormatEmulation::SwizzledView`]
    /// is (exact, and at the border as that is); a storage image whose shader reads the byte as
    /// alpha and writes a texel's alpha into it (`StorageImageClass::AlphaInRed`); and a color
    /// attachment the fragment function writes its alpha to (`LoweringInputs::rotated_colors`,
    /// whose rotation puts Metal's alpha in the image's red), whose write mask's alpha bit, clear
    /// color's alpha and blend (Metal's alpha factors and operation, each factor that reads an
    /// alpha reading the red it lands in, `pipeline::rotated_blend::alpha_in_red_blend`) the
    /// consumer and link move to red. Metal stores, reads, masks, clears, blends and resolves
    /// `A8Unorm` exactly so (probe `a8`, an M3 Max under macOS 27.0.1: kernel and render writes
    /// store the alpha, every read is `(0, 0, 0, a)`, only the alpha bit of a write mask writes,
    /// each alpha factor reads alphas, and a 4-sample resolve is `R8Unorm`'s). An output to such
    /// an attachment holds Metal's alpha in the image's alpha lane too, which alpha to coverage
    /// reads, and a fetch of one reads its red as Metal's alpha, `(0, 0, 0, r)`
    /// (`LoweringInputs::alpha_in_red_colors`; probe `a8`'s uses: Metal's coverage through an
    /// `A8Unorm` attachment 0 is `RGBA8Unorm`'s of the same alpha, and its fetch `(0, 0, 0, a)`).
    /// A texture buffer's bytes take an `R8_UNORM` view wherever `A8_UNORM` does not serve its use
    /// (`texel_buffer_alpha_in_red`), and an argument buffer holds a storage texture of it as its
    /// `R8_UNORM` image, whose texels a heap read or write converts where the handle carries
    /// `A8Unorm`'s class (`bind::heap`; probe `a8`'s argument buffers: a member reads `(0, 0, 0, a)`
    /// and a write stores its alpha). Admitted for the image usages.
    AlphaInRed,
    /// A stencil-only format held in a combined depth/stencil format; only the stencil aspect is
    /// used. Exact.
    StencilInCombined,
    /// A depth-only format held in a combined depth/stencil format of the same depth format; only
    /// the depth aspect is used. Exact.
    DepthInCombined,
    /// 24-bit normalized depth held as 32-bit float depth. Every 24-bit value is representable,
    /// but the stored values and depth-test results near equality can differ from Metal: this is
    /// **not exact**, and the caller decides whether to accept it.
    DepthWidened,
    /// One of Apple's extended-range formats ([`ExtendedRange`]), which Vulkan has no format for:
    /// the texels' bytes held in an unsigned integer format of the same layout, which a function
    /// reads and writes through its slot's [`ExtendedRange`] variant
    /// (`LoweringInputs::extended_range`), decoding and encoding Metal's values in the shader.
    /// Exact for a sampled image a function fetches from, for a storage image, and for a color
    /// attachment a fragment function writes encoded texels to
    /// (`LoweringInputs::extended_range_colors`); admitted for those usages only (no filter, blend,
    /// or texel buffer).
    ExtendedRange,
    /// Apple's two-plane YCbCr formats this crate holds ([`ycbcr`]), which Metal makes only over
    /// an IOSurface's two planes and reads as RGB: held as texels of exactly Metal's values,
    /// which the consumer computes from the planes' bytes (`bind::ycbcr_convert_module`):
    /// `R16G16B16A16_SFLOAT` of a converted format's (a grid of `1/2048`),
    /// `R32G32B32A32_SFLOAT` of an `_sRGB` twin's (a grid of `1/8192`), and
    /// `A2B10G10R10_UNORM_PACK32` of the codes. Exact for a sampled image a function reads or
    /// fetches from, and for a linear sample the function emulates (Metal's filter of the texels on
    /// their grid, `metal::texel_format::PackedTexels::YCbCr2048` and `YCbCr8192`); a linear sample
    /// the emulation does not reach is the device's filter, noted. Admitted for sampling only:
    /// Metal drops a function's writes to a 4:2:0 or 4:2:2 format ([`writes_dropped`]), stores a
    /// 4:4:4 one's into both planes (which a slot that only writes one of [`ycbcr_writes_stored`]
    /// does through the texture's write shadow; the others are refused), and renders into such a
    /// texture by an inverse conversion, which a pass into one of [`ycbcr_renders`] does through
    /// [`FormatEmulation::YCbCrRendered`] (a packed one's pass is refused, its pipeline made,
    /// [`ycbcr_pass_refused`]; the others are not modelled).
    YCbCr,
    /// `ARGB8Unorm`'s bytes (alpha, then red, green and blue) held in `R8G8B8A8_UNORM`, whose red,
    /// green, blue and alpha are Metal's alpha, red, green and blue ([`to_rotated_image`]);
    /// [`FormatMapping::components`] rotates them back for sampling. Exact for sampling as
    /// [`FormatEmulation::SwizzledView`] is (Metal's border texel is the border color in the
    /// image's channels, read through the rotation), for a storage image the shader rotates
    /// (`StorageImageClass::RotateArgb`), and for a color attachment a fragment function writes
    /// rotated (`LoweringInputs::rotated_colors`), whose clear color and write mask the consumer
    /// rotates too; Metal stores, clears, masks, resolves and blends the format as `RGBA8Unorm`
    /// with its channels rotated. Admitted for those usages, and for blending
    /// through the factors `pipeline::rotated_blend` finds Metal's blend in, which refuses the
    /// rest.
    RotatedView,
    /// A render pass's color attachment of a two-plane YCbCr format whose store Metal's planes
    /// are matched for ([`ycbcr_renders`]): `R32G32B32A32_SFLOAT` of the values the pass leaves
    /// in each texel (the fragment function's output, the clear color through `float`, or the
    /// texel the pass loads), which the consumer encodes into the surface's planes as the pass
    /// stores (`bind::ycbcr_store`, the encode kernel's store of every texel). Exact for an
    /// attachment a fragment function writes unblended, under any write mask (a masked channel
    /// keeps the texel's value, as Metal's tile keeps it). Admitted for a color attachment and
    /// its blend (where the device blends `R32G32B32A32_SFLOAT`), which the device computes in
    /// floats where Metal's tile blends halves: within a code of Metal's, which the consumer
    /// announces.
    YCbCrRendered,
    /// A pipeline's color attachment of a packed 10-bit two-plane YCbCr format whose pass is
    /// refused ([`ycbcr_pass_refused`]): `R32G32B32A32_SFLOAT`, which no pass renders into.
    /// Metal makes such a pipeline, and its pass into such a texture fails (measured as
    /// [`ycbcr_pass_refused`] states), so the pipeline is made and the runtime refuses the pass
    /// at its start. Admitted for a color attachment only (no blend: the pipeline refuses one).
    YCbCrPassRefused,
    /// One of Apple's RGBX formats ([`x_channel`]): the bytes of its RGBA twin, whose fourth
    /// field (Metal's X) Metal reads as alpha 1 and its GPU writes as the green field's bits,
    /// held in the twin's Vulkan format. [`FormatMapping::components`] reads alpha as one for
    /// sampling (exact at the border too: Metal's border texels read alpha 1 there), and a storage
    /// image's slot converts its texels in the shader (`StorageImageChoice::x_channel`). Admitted
    /// for sampling and for storage images.
    XChannel,
    /// One of Apple's bfloat formats ([`ExtendedRange::R16Bfloat`], `RG16Bfloat`, `RGBA16Bfloat`):
    /// each channel's 16-bit code held raw in the unsigned integer format of its words, which a
    /// function reads and writes through its slot's [`ExtendedRange`] variant
    /// (`LoweringInputs::extended_range`), converting Metal's values in the shader. Exact for a
    /// sampled image a function fetches from and for a storage image; admitted for those usages
    /// and for a color attachment, whose render pass clears and loads the raw codes as Metal's does
    /// (`clear_texel`; the runtime's load pass). Metal filters no bfloat texel (a linear sample
    /// reads the nearest texel) and blends none (its validation asserts), so a filter and a blend
    /// are not admitted.
    Bfloat,
    /// A compressed format the device has no format for, today an ASTC LDR, a PVRTC, an ETC2 or
    /// an EAC one (or its `_sRGB` form), or a 4:2:2 one (`GBGR422`, `BGRG422`, whose Vulkan
    /// formats reconstruct chroma by the implementation's rules): the texture's blocks are held
    /// for copies alone, and its sampled views present a decoded twin, an image of Metal's decode
    /// of them ([`AstcDecode`](crate::bind::AstcDecode), [`PvrtcDecode`](crate::bind::PvrtcDecode),
    /// [`EtcDecode`](crate::bind::EtcDecode), [`ChromaDecode`](crate::bind::ChromaDecode)), which
    /// the consumer decodes after each write. The candidate's format is the twin's view of compact
    /// texels, `R8G8B8A8_UNORM` (an `_sRGB` form's `R8G8B8A8_SRGB`), whose texels are 8-bit codes
    /// of Metal's values (a PVRTC, ETC2, EAC or 4:2:2 format's own codes, an sRGB one's read raw,
    /// `texel_format::srgb_raw`), or an EAC 11-bit
    /// format's 16-bit normalized codes nearest Metal's values (`R16_UNORM`, `R16_SNORM` and their
    /// two-channel forms); for an ASTC or an EAC 11-bit format the consumer may hold Metal's values
    /// exactly as `float`s instead ([`AstcTexels`](crate::bind::AstcTexels),
    /// [`EtcTexels`](crate::bind::EtcTexels)), whose views it presents for the candidate's.
    /// Admitted for sampling only: Metal's functions do not write a compressed texture, and no
    /// pass renders into one.
    Decoded,
}

impl FormatEmulation {
    /// Whether a candidate with this emulation is valid for `usage` (independent of the device).
    fn admits(self, usage: FormatUsage) -> bool {
        use FormatUsage as U;
        match self {
            FormatEmulation::Direct => true,
            FormatEmulation::SwizzledView => matches!(
                usage,
                U::SampledImage | U::SampledImageFilterLinear | U::SampledImageFilterMinmax
            ),
            FormatEmulation::AlphaInRed => matches!(
                usage,
                U::SampledImage
                    | U::SampledImageFilterLinear
                    | U::SampledImageFilterMinmax
                    | U::StorageImage
                    | U::ColorAttachment
                    | U::ColorAttachmentBlend
            ),
            FormatEmulation::RotatedView => matches!(
                usage,
                U::SampledImage
                    | U::SampledImageFilterLinear
                    | U::SampledImageFilterMinmax
                    | U::ColorAttachment
                    | U::ColorAttachmentBlend
            ),
            FormatEmulation::ExtendedRange => {
                matches!(
                    usage,
                    U::SampledImage | U::StorageImage | U::ColorAttachment
                )
            }
            FormatEmulation::YCbCr => {
                matches!(usage, U::SampledImage | U::SampledImageFilterLinear)
            }
            FormatEmulation::YCbCrRendered => {
                matches!(usage, U::ColorAttachment | U::ColorAttachmentBlend)
            }
            FormatEmulation::YCbCrPassRefused => usage == U::ColorAttachment,
            FormatEmulation::XChannel => matches!(
                usage,
                U::SampledImage
                    | U::SampledImageFilterLinear
                    | U::SampledImageFilterMinmax
                    | U::StorageImage
            ),
            FormatEmulation::Bfloat => {
                matches!(
                    usage,
                    U::SampledImage | U::StorageImage | U::ColorAttachment
                )
            }
            FormatEmulation::Decoded => matches!(
                usage,
                U::SampledImage | U::SampledImageFilterLinear | U::SampledImageFilterMinmax
            ),
            FormatEmulation::StencilInCombined
            | FormatEmulation::DepthInCombined
            | FormatEmulation::DepthWidened => matches!(
                usage,
                U::SampledImage
                    | U::SampledImageFilterLinear
                    | U::SampledImageFilterMinmax
                    | U::DepthStencilAttachment
            ),
        }
    }
}

/// One of Apple's extended-range pixel formats, as a texture slot bound in it reads and writes
/// its raw texels. Metal's answers, measured on an Apple M3 Max (macOS 26.7) and an Apple M4 (macOS
/// 27.0.1) alike, every code and channel:
///
/// - **Codes.** A channel's 10-bit code `c` is the value `(c - 384) / 510` correctly rounded to
///   `f32`, `[-0.752941, 1.252941]`; an `_sRGB` format's color channel is that value through the
///   sRGB transfer, on a grid of `2^-13` (`lower::arith::extended_range`'s table), and its alpha
///   is linear. Alpha reads clamped to `[0, 1]`.
/// - **Writes.** A value is the code of its exact product with 510 rounded to nearest even, plus
///   384, clamped to the codes (alpha to `[384, 894]`); a NaN is 384. An `_sRGB` color channel's
///   magnitude is rounded to `2^-13` (nearest even) and encoded by Metal's own thresholds
///   (`lower::arith::extended_range`'s table), symmetrically about 384.
/// - **Bytes.** `BGR10_XR`: one little-endian word, blue in bits 0-9, green 10-19, red 20-29;
///   bits 30-31 are not read and are written as ones; alpha reads 1. `BGRA10_XR`: four 16-bit
///   fields, blue, green, red, alpha from the lowest, each code in a field's top 10 bits; the low
///   6 bits are not read and are written as zeros.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtendedRange {
    /// `BGR10_XR`, or `BGR10_XR_sRGB` where `srgb`: held as `R32_UINT`.
    Bgr10 {
        /// The `_sRGB` twin.
        srgb: bool,
    },
    /// `BGRA10_XR`, or `BGRA10_XR_sRGB` where `srgb`: held as `R16G16B16A16_UINT`.
    Bgra10 {
        /// The `_sRGB` twin.
        srgb: bool,
    },
    /// `RGB10A8_2P_XR10`, or `RGB10A8_2P_XR10_sRGB` where `srgb`: two planes on Metal, the
    /// color word as `BGR10_XR`'s and an alpha byte, held side by side in one `R32G32_UINT`
    /// image (the word in the first component, the byte in the second). The image's texels are
    /// not Metal's bytes, so CPU copies, blits and views of such a texture are refused.
    Planes {
        /// The `_sRGB` twin.
        srgb: bool,
    },
    /// `R10Unorm_X6` (576), a 10-bit unsigned normalized red code in the top 10 bits of a
    /// little-endian 16-bit word, held as `R16_UINT`. Metal makes a texture of it in memory of
    /// its own as readily as over an IOSurface's plane or a buffer ([`padded_words`], which has
    /// the creators measured). Measured on an Apple M4 under macOS 27.0.1 and an Apple M3
    /// Max under macOS 26.7: a read is `c / 1023` correctly rounded (its `half` to nearest even),
    /// the low 6 bits unread; a kernel write stores the code of the exact product of the value,
    /// clamped to `[0, 1]` (a NaN 0), with 1023, rounded to nearest, the low 6 bits zero; a render
    /// pass's store rounds the output to `half` toward zero first; a clear goes through `float` and
    /// then `half`, each to nearest even.
    R10X6,
    /// `RG10Unorm_X12` (578): two such words, red then green, held as `R16G16_UINT`.
    Rg10X12,
    /// `R10Unorm_X6_sRGB` (577): `R10Unorm_X6`'s word, held as `R16_UINT`, its code sRGB-encoded
    /// through Metal's own tables (an Apple M4 and an Apple M3 Max, byte-identical): a read is the
    /// decode step `s / 8192` of the code (its `half` rounded with ties away from zero); a write
    /// rounds the value, clamped to `[0, 1]` (a NaN 0), to `2^-13` (nearest even) and stores the
    /// largest code whose first encode index is at most that.
    R10X6Srgb,
    /// `RG10Unorm_X12_sRGB` (579): `RG10Unorm_X12`'s two words, held as `R16G16_UINT`. Its reads
    /// are `RG10Unorm_X12`'s, `c / 1023`; a write stores, for the code `m` `RG10Unorm_X12` writes,
    /// Metal's code of `m` (a measured table of 1024 codes up to 137).
    Rg10X12Srgb,
    /// `RGB10A2Unorm_sRGB` (586): `RGB10A2Unorm`'s little-endian word, red in bits 0-9, green
    /// 10-19, blue 20-29 and alpha 30-31, held as `R32_UINT`. Measured on an Apple M3 Max under
    /// macOS 27.0.1, which makes it in every storage mode and usage: each color channel's code
    /// converts as `R10Unorm_X6_sRGB`'s (its tables, every code read and 16,385 values written;
    /// a `half` read with ties away from zero), and alpha's 2-bit code `a` reads `a / 3`
    /// correctly rounded and a write stores the exact product of the value, clamped to `[0, 1]`
    /// (a NaN 0), with 3, rounded to nearest. A render pass's store rounds a `float` output to
    /// `half` toward zero first, a clear goes through `half` (nearest even), and a pass that loads
    /// it and draws elsewhere writes each color code back as `R10Unorm_X6_sRGB`'s
    /// ([`load_code`](Self::load_code)), alpha kept.
    Rgb10A2Srgb,
    /// `R16Bfloat` (26): a bfloat code (the top 16 bits of an IEEE `float`) in a little-endian
    /// 16-bit word, held as `R16_UINT` ([`FormatEmulation::Bfloat`]). Measured on an Apple M4
    /// under macOS 27.0.1, every code of each bfloat format: a read (`read` and `read_write`, a
    /// nearest or linear sample, a gather) is the `float` of the code's bits shifted up 16, any NaN
    /// code the quiet NaN `0x7fc00000`, a channel the format lacks 0 and alpha 1; a `half` read or
    /// gather rounds that `float` to nearest even, a `half` sample toward zero (saturating at
    /// `65504`). A sample under either filter reads the texel at `floor` of the unsnapped
    /// coordinate; a gather reads the linear footprint's taps. A texture swizzle applies to the
    /// decoded texel, border texels included. A kernel write stores the value's top 16 bits (toward
    /// zero), a NaN `0x7fc0`, a `half` value widened first. A render pass's clear stores the color
    /// through `float` and then `half`, each rounded to nearest even, then the bfloat code nearest
    /// that, ties to even, a NaN `0x7fc0` ([`clear_texel`](Self::clear_texel)). Copies and the
    /// CPU's accesses keep the bytes.
    R16Bfloat,
    /// `RG16Bfloat` (66): two such words, red then green, held as `R16G16_UINT`.
    Rg16Bfloat,
    /// `RGBA16Bfloat` (116): four such words, red, green, blue and alpha, held as
    /// `R16G16B16A16_UINT`.
    Rgba16Bfloat,
}

impl ExtendedRange {
    /// The extended-range format `pf` is, if it is one held in one image.
    pub fn of(pf: PixelFormat) -> Option<ExtendedRange> {
        match pf {
            PixelFormat::BGR10XR => Some(ExtendedRange::Bgr10 { srgb: false }),
            PixelFormat::BGR10XRSRGB => Some(ExtendedRange::Bgr10 { srgb: true }),
            PixelFormat::BGRA10XR => Some(ExtendedRange::Bgra10 { srgb: false }),
            PixelFormat::BGRA10XRSRGB => Some(ExtendedRange::Bgra10 { srgb: true }),
            PixelFormat::RGB10A82PXR10 => Some(ExtendedRange::Planes { srgb: false }),
            PixelFormat::RGB10A82PXR10SRGB => Some(ExtendedRange::Planes { srgb: true }),
            PixelFormat::R10UnormX6 => Some(ExtendedRange::R10X6),
            PixelFormat::RG10UnormX12 => Some(ExtendedRange::Rg10X12),
            PixelFormat::R10UnormX6SRGB => Some(ExtendedRange::R10X6Srgb),
            PixelFormat::RG10UnormX12SRGB => Some(ExtendedRange::Rg10X12Srgb),
            PixelFormat::RGB10A2UnormSRGB => Some(ExtendedRange::Rgb10A2Srgb),
            PixelFormat::R16Bfloat => Some(ExtendedRange::R16Bfloat),
            PixelFormat::RG16Bfloat => Some(ExtendedRange::Rg16Bfloat),
            PixelFormat::RGBA16Bfloat => Some(ExtendedRange::Rgba16Bfloat),
            _ => None,
        }
    }

    /// The unsigned integer format of the texels' bytes, which holds them.
    pub fn raw_format(self) -> vk::Format {
        match self {
            ExtendedRange::Bgr10 { .. } | ExtendedRange::Rgb10A2Srgb => vk::Format::R32_UINT,
            ExtendedRange::Bgra10 { .. } => vk::Format::R16G16B16A16_UINT,
            ExtendedRange::Planes { .. } => vk::Format::R32G32_UINT,
            ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => vk::Format::R16_UINT,
            ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => vk::Format::R16G16_UINT,
            ExtendedRange::R16Bfloat => vk::Format::R16_UINT,
            ExtendedRange::Rg16Bfloat => vk::Format::R16G16_UINT,
            ExtendedRange::Rgba16Bfloat => vk::Format::R16G16B16A16_UINT,
        }
    }

    /// Whether the color channels are sRGB-encoded through the extended-range formats' tables
    /// (`lower::arith::extended_range::declare`). The padded `_sRGB` twins have tables of their
    /// own ([`padded_srgb`](Self::padded_srgb)).
    pub fn srgb(self) -> bool {
        match self {
            ExtendedRange::Bgr10 { srgb }
            | ExtendedRange::Bgra10 { srgb }
            | ExtendedRange::Planes { srgb } => srgb,
            ExtendedRange::R10X6
            | ExtendedRange::Rg10X12
            | ExtendedRange::R10X6Srgb
            | ExtendedRange::Rg10X12Srgb
            | ExtendedRange::Rgb10A2Srgb
            | ExtendedRange::R16Bfloat
            | ExtendedRange::Rg16Bfloat
            | ExtendedRange::Rgba16Bfloat => false,
        }
    }

    /// Whether the codes are bfloat ones ([`R16Bfloat`](Self::R16Bfloat), `RG16Bfloat`,
    /// `RGBA16Bfloat`), each the top 16 bits of an IEEE `float`.
    pub fn bfloat(self) -> bool {
        matches!(
            self,
            ExtendedRange::R16Bfloat | ExtendedRange::Rg16Bfloat | ExtendedRange::Rgba16Bfloat
        )
    }

    /// Whether the codes convert through the padded `_sRGB` formats' tables
    /// (`R10Unorm_X6_sRGB`, `RG10Unorm_X12_sRGB`, and `RGB10A2Unorm_sRGB`'s color channels
    /// through `R10Unorm_X6_sRGB`'s; `lower::arith::extended_range::declare_padded`).
    pub fn padded_srgb(self) -> bool {
        matches!(
            self,
            ExtendedRange::R10X6Srgb | ExtendedRange::Rg10X12Srgb | ExtendedRange::Rgb10A2Srgb
        )
    }

    /// Whether the codes are 10-bit unsigned normalized ones rather than Apple's extended-range
    /// `(c - 384) / 510`: the padded formats' (`R10Unorm_X6`, `RG10Unorm_X12` and their `_sRGB`
    /// twins), each in a 16-bit word's top 10 bits, and `RGB10A2Unorm_sRGB`'s packed fields
    /// ([`Rgb10A2Srgb`](Self::Rgb10A2Srgb), its alpha a 2-bit code).
    pub fn padded(self) -> bool {
        matches!(
            self,
            ExtendedRange::R10X6
                | ExtendedRange::Rg10X12
                | ExtendedRange::R10X6Srgb
                | ExtendedRange::Rg10X12Srgb
                | ExtendedRange::Rgb10A2Srgb
        )
    }

    /// Whether Metal's linear mip filter of the form's samples is built, a blend of two levels'
    /// samples (`lower::sampler::extended`): `BGR10_XR`'s and `BGRA10_XR`'s and their `_sRGB`
    /// twins', measured on an Apple M3 Max. Another form's sample through a sampler with a linear
    /// mip filter is refused.
    pub(crate) fn blends_levels(self) -> bool {
        matches!(
            self,
            ExtendedRange::Bgr10 { .. } | ExtendedRange::Bgra10 { .. }
        )
    }

    /// The forms an argument buffer's texture is read in (a heap element), in the order their
    /// classes follow `metal::texel_format::heap_classes`' own: a function decodes such a texture's
    /// raw texels, read through the heap's unsigned integer array, without a table. The `_sRGB`
    /// forms' decodes take the module's tables and the two-plane forms are held in two planes, so a
    /// heap does not hold them.
    pub const HEAP_READ: [ExtendedRange; 7] = [
        ExtendedRange::Bgr10 { srgb: false },
        ExtendedRange::Bgra10 { srgb: false },
        ExtendedRange::R10X6,
        ExtendedRange::Rg10X12,
        ExtendedRange::R16Bfloat,
        ExtendedRange::Rg16Bfloat,
        ExtendedRange::Rgba16Bfloat,
    ];

    /// This form's index in [`HEAP_READ`](Self::HEAP_READ), if a heap reads it.
    pub fn heap_read(self) -> Option<u32> {
        Self::HEAP_READ
            .iter()
            .position(|&f| f == self)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Whether the image holds Metal's bytes of each texel, which CPU copies, blits and views
    /// then move unchanged: every form but [`Planes`](Self::Planes), whose two planes share one
    /// image's texels.
    pub fn holds_metal_bytes(self) -> bool {
        !matches!(self, ExtendedRange::Planes { .. })
    }

    /// The channels a color write mask selects among: red, green and blue, and alpha but for
    /// `BGR10_XR` and its twin, which have none.
    pub fn channels(self) -> crate::metal::ColorWriteMask {
        use crate::metal::ColorWriteMask as M;
        let rgb = M::RED.union(M::GREEN).union(M::BLUE);
        match self {
            ExtendedRange::Bgr10 { .. } => rgb,
            ExtendedRange::Bgra10 { .. }
            | ExtendedRange::Planes { .. }
            | ExtendedRange::Rgb10A2Srgb => rgb.union(M::ALPHA),
            ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => M::RED,
            ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => M::RED.union(M::GREEN),
            ExtendedRange::R16Bfloat => M::RED,
            ExtendedRange::Rg16Bfloat => M::RED.union(M::GREEN),
            ExtendedRange::Rgba16Bfloat => rgb.union(M::ALPHA),
        }
    }

    /// The components of [`raw_format`](Self::raw_format) a texel occupies: `BGR10_XR`'s one
    /// word, the two-plane formats' color word and alpha byte, `BGRA10_XR`'s four fields.
    pub fn raw_components(self) -> u32 {
        match self {
            ExtendedRange::Bgr10 { .. }
            | ExtendedRange::R10X6
            | ExtendedRange::R10X6Srgb
            | ExtendedRange::Rgb10A2Srgb => 1,
            ExtendedRange::Planes { .. } | ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => 2,
            ExtendedRange::Bgra10 { .. } => 4,
            ExtendedRange::R16Bfloat => 1,
            ExtendedRange::Rg16Bfloat => 2,
            ExtendedRange::Rgba16Bfloat => 4,
        }
    }

    /// The raw texel (the components of [`raw_format`](Self::raw_format)) Metal's `Clear` load
    /// action stores for the clear color `color` (red, green, blue, alpha; measured on an Apple M3
    /// Max and an M4): each channel goes through `float` and then `half`,
    /// each rounded to nearest even, and is stored as a kernel write stores that value; the bits
    /// Metal does not read are written as a write writes them. That is Metal's route `route` where
    /// it is [`ClearRoute::ThroughHalf`]; a texel by [`ClearRoute::Straight`] is that route's. A
    /// bfloat form's code is the one `route` gives. (A color fill, Apple SPI, goes through neither:
    /// the runtime's `blit_fill`.)
    pub fn clear_texel(self, color: [f64; 4], route: ClearRoute) -> [u32; 4] {
        let half =
            |v: f64| crate::float16::f16_bits_to_f32(crate::float16::f32_to_f16_bits(v as f32));
        let straight = route == ClearRoute::Straight
            && matches!(
                self,
                ExtendedRange::Bgr10 { .. } | ExtendedRange::Bgra10 { .. }
            );
        let codes = std::array::from_fn(|c| match (self.bfloat(), route) {
            (true, ClearRoute::ThroughHalf) => bfloat_nearest_even(half(color[c])),
            (true, ClearRoute::Straight) => bfloat_of_double(color[c]),
            (false, _) if straight => self.straight_code(color[c] as f32, c == 3),
            (false, _) => self.code(half(color[c]), c == 3),
        });
        let texel = self.texel(codes);
        match (self, straight) {
            // Bits 30 and 31 zero.
            (ExtendedRange::Bgr10 { .. }, true) => [texel[0] & 0x3fff_ffff, 0, 0, 0],
            _ => texel,
        }
    }

    /// The code Metal's straight clear route ([`ClearRoute::Straight`]) stores in a `BGR10_XR`
    /// or `BGRA10_XR` channel, or its alpha where `alpha`, for the `float` `v` (an Apple M4 under
    /// macOS 27.0.1: every channel of 2,574 colors of each form; the consumer's M3 Max under 26.7
    /// alike). A linear channel's: `(v + 384/510) · 510`, each step a `float` rounded to nearest
    /// even, rounded to the nearest code with ties up, clamped to `[0, 1023]`, a NaN 0 (`0.25` is
    /// 511 and `-0.25` is 257, where a write stores 512 and 256); alpha's that of `v` clamped to
    /// `[0, 1]` first, a NaN 384. An `_sRGB` color channel's: the offset from code 384, toward the
    /// sign of `v`, of the thresholds of Metal's own encode of that sign at most `|v|`
    /// (`extended_srgb::STRAIGHT_SRGB_FIRST`), a NaN 384.
    fn straight_code(self, v: f32, alpha: bool) -> u32 {
        if self.srgb() && !alpha {
            let bits = v.abs().to_bits();
            let at = |thresholds: &[u32]| thresholds.partition_point(|&t| t <= bits) as u32;
            return match (v.is_nan(), v < 0.0) {
                (true, _) => 384,
                (false, true) => 384 - at(&extended_srgb::STRAIGHT_SRGB_FIRST_NEGATIVE),
                (false, false) => 384 + at(&extended_srgb::STRAIGHT_SRGB_FIRST),
            };
        }
        let v = match (v.is_nan(), alpha) {
            (true, true) => 0.0,
            (true, false) => return 0,
            (false, true) => v.clamp(0.0, 1.0),
            (false, false) => v,
        };
        let x = (v + 384.0 / 510.0) * 510.0;
        // Ties up, of the `float`: its sum with a half is exact in `f64`.
        (f64::from(x) + 0.5).floor().clamp(0.0, 1023.0) as u32
    }

    /// Whether a render pass's `Clear` load of the clear color `color` leaves Metal's tile memory
    /// holding the values the clear's stored codes ([`clear_texel`](Self::clear_texel)) read
    /// back, rounded to `half`. Metal's tile memory holds each channel of a pass's
    /// `BGR10_XR`, `BGRA10_XR` or two-plane color, and their `_sRGB` twins, as `half` until the
    /// pass stores it, and a `Clear` load as the clear color through `float` and then `half`
    /// (on an Apple M4 and the same probe on an Apple M3 Max): a draw that reads
    /// or blends into the color then sees those halves, where an image of the codes holds the
    /// codes nearest them. `None` for the padded and bfloat forms, whose tile memory is
    /// unmeasured.
    pub fn clear_read_back(self, color: [f64; 4]) -> Option<bool> {
        if self.padded() || self.bfloat() {
            return None;
        }
        let half = |v: f32| crate::float16::f16_bits_to_f32(crate::float16::f32_to_f16_bits(v));
        let channels = match self {
            ExtendedRange::Bgr10 { .. } => 3,
            _ => 4,
        };
        Some((0..channels).all(|c| {
            let held = half(color[c] as f32);
            held == half(self.code_value(self.code(held, c == 3), c == 3))
        }))
    }

    /// The `half` values Metal's tile memory holds for a color of this form after a render
    /// pass's `Clear` load of the clear color `color` (red, green, blue, alpha), as `f32`: each
    /// channel through `float` and then `half`, each rounded to nearest even, and `BGR10_XR`'s
    /// alpha 1, as a read of the tile reads it. The runtime clears the attachment's `half` shadow
    /// to them. `None` for the padded and bfloat forms, whose tile memory is unmeasured.
    pub fn clear_tile_value(self, color: [f64; 4]) -> Option<[f32; 4]> {
        if self.padded() || self.bfloat() {
            return None;
        }
        let half = |v: f32| crate::float16::f16_bits_to_f32(crate::float16::f32_to_f16_bits(v));
        Some(std::array::from_fn(|c| match (self, c) {
            (ExtendedRange::Bgr10 { .. }, 3) => 1.0,
            _ => half(color[c] as f32),
        }))
    }

    /// The value Metal reads from the code `code` of a color channel, or of alpha where `alpha`,
    /// of a `BGR10_XR`, `BGRA10_XR` or two-plane format or an `_sRGB` twin: `(code - 384) / 510`
    /// correctly rounded, an `_sRGB` color channel's table step, the two-plane formats' alpha byte
    /// over 255.
    fn code_value(self, code: u32, alpha: bool) -> f32 {
        let n = code as i32 - 384;
        match (self, alpha) {
            (ExtendedRange::Planes { .. }, true) => code as f32 / 255.0,
            (_, false) if self.srgb() => {
                let step = extended_srgb::SRGB_DECODE_STEPS[n.unsigned_abs() as usize];
                (f32::from(step) / 8192.0).copysign(n as f32)
            }
            _ => n as f32 / 510.0,
        }
    }

    /// The code Metal's write stores for the value `v` of a color channel, or of alpha where
    /// `alpha` (the kernel write rule): a NaN is 0; a color channel's code is `384 ±
    /// RoundHalfEven(min(|v|, 1.5) · 510)` of the exact product, clamped to `[0, 1023]`, or for an
    /// `_sRGB` form the offset of the largest first index at most `RoundEven(min(|v|, 2) · 8192)`
    /// (at most 384 below code 384); alpha is clamped to `[0, 1]` and stored as `384 +
    /// RoundHalfEven(a · 510)`, or as the two-plane formats' byte `RoundHalfEven(a · 255)`.
    fn code(self, v: f32, alpha: bool) -> u32 {
        let v = if v.is_nan() { 0.0 } else { v };
        // A padded format's code: the exact product of the value clamped to [0, 1] with 1023,
        // rounded to nearest even (its one tie, 0.5, stores 512); an `_sRGB`
        // twin's through Metal's tables. `RGB10A2Unorm_sRGB`'s alpha: the product with 3, never a
        // tie.
        if self.padded() {
            let v = f64::from(v.clamp(0.0, 1.0));
            let linear = (v * 1023.0).round_ties_even() as usize;
            return match self {
                ExtendedRange::Rgb10A2Srgb if alpha => (v * 3.0).round_ties_even() as u32,
                ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb => {
                    // `v` times 2^13 is exact.
                    let i = (v * 8192.0).round_ties_even() as u16;
                    let first = &padded_srgb::R10X6_SRGB_ENCODE_FIRST;
                    first.partition_point(|&f| f <= i).saturating_sub(1) as u32
                }
                ExtendedRange::Rg10X12Srgb => {
                    u32::from(padded_srgb::RG10X12_SRGB_WRITE_CODES[linear])
                }
                _ => linear as u32,
            };
        }
        if alpha {
            let a = f64::from(v.clamp(0.0, 1.0));
            return match self {
                ExtendedRange::Planes { .. } => (a * 255.0).round_ties_even() as u32,
                _ => 384 + (a * 510.0).round_ties_even() as u32,
            };
        }
        let neg = v < 0.0;
        let m = f64::from(v.abs());
        if self.srgb() {
            // `|v|` below 2 times 2^13 is exact.
            let i = (m.min(2.0) * 8192.0).round_ties_even() as u32;
            let first = &extended_srgb::SRGB_ENCODE_FIRST;
            let n = first
                .partition_point(|&f| u32::from(f) <= i)
                .saturating_sub(1) as u32;
            return match neg {
                true => 384 - n.min(384),
                false => 384 + n,
            };
        }
        let k = (m.min(1.5) * 510.0).round_ties_even() as i64;
        let code = if neg { 384 - k } else { 384 + k };
        code.clamp(0, 1023) as u32
    }

    /// The raw texel of the red, green, blue and alpha codes `rgba` (alpha the two-plane
    /// formats' byte there), with the bits Metal does not read as its writes write them:
    /// `BGR10_XR`'s bits 30 and 31 ones, plane 0's zero, `BGRA10_XR`'s low 6 bits of each field
    /// zero.
    fn texel(self, rgba: [u32; 4]) -> [u32; 4] {
        let word = rgba[0] << 20 | rgba[1] << 10 | rgba[2];
        match self {
            ExtendedRange::Bgr10 { .. } => [0xC000_0000 | word, 0, 0, 0],
            ExtendedRange::Planes { .. } => [word, rgba[3], 0, 0],
            ExtendedRange::Bgra10 { .. } => {
                [rgba[2] << 6, rgba[1] << 6, rgba[0] << 6, rgba[3] << 6]
            }
            // Each code in its word's top 10 bits, the low 6 bits zero.
            ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => [rgba[0] << 6, 0, 0, 0],
            ExtendedRange::Rgb10A2Srgb => [
                rgba[3] << 30 | rgba[2] << 20 | rgba[1] << 10 | rgba[0],
                0,
                0,
                0,
            ],
            ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => {
                [rgba[0] << 6, rgba[1] << 6, 0, 0]
            }
            // A bfloat code a word.
            ExtendedRange::R16Bfloat => [rgba[0], 0, 0, 0],
            ExtendedRange::Rg16Bfloat => [rgba[0], rgba[1], 0, 0],
            ExtendedRange::Rgba16Bfloat => rgba,
        }
    }

    /// The code a padded form's word of code `c` holds after a render pass that loads it and
    /// draws, where no draw writes it: Metal's pass reads the word and writes it back, its
    /// decoded value encoded again (an Apple M4 and an Apple M3 Max, byte-identical). `c` itself
    /// for `R10Unorm_X6` and `RG10Unorm_X12`; `R10Unorm_X6_sRGB`'s write of its read, which moves
    /// the 23 codes no write stores, and so does each color channel's code of `RGB10A2Unorm_sRGB`
    /// (an Apple M3 Max under macOS 27.0.1, every code; its alpha's code stays);
    /// `RG10Unorm_X12_sRGB`'s write code of `c`. A bfloat form's code
    /// `c` (each channel's 16-bit word) holds the code a `Clear` of its value stores under
    /// [`ClearRoute::ThroughHalf`] ([`clear_texel`](Self::clear_texel)): the `float` of its bits
    /// rounded to `half` to nearest even (past the largest `half`, the infinity), then the bfloat
    /// code nearest that, ties to even, a NaN `0x7fc0` (an Apple M4 under macOS 27.0.1: every code
    /// of each bfloat format through a pass that loads it and draws into another attachment; 56,829
    /// codes move). The M4's pass alone is measured. `None` for another form, or a code past the
    /// form's (1023, or a bfloat form's `0xffff`).
    pub fn load_code(self, c: u32) -> Option<u32> {
        if self.bfloat() {
            let c = u16::try_from(c).ok()?;
            let value = f64::from(f32::from_bits(u32::from(c) << 16));
            return Some(self.clear_texel([value; 4], ClearRoute::ThroughHalf)[0]);
        }
        let at = usize::try_from(c).ok().filter(|&c| c < 1024)?;
        match self {
            ExtendedRange::R10X6 | ExtendedRange::Rg10X12 => Some(c),
            ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb => {
                let step = padded_srgb::R10X6_SRGB_DECODE_STEPS[at];
                let first = &padded_srgb::R10X6_SRGB_ENCODE_FIRST;
                Some(first.partition_point(|&f| f <= step).saturating_sub(1) as u32)
            }
            ExtendedRange::Rg10X12Srgb => {
                Some(u32::from(padded_srgb::RG10X12_SRGB_WRITE_CODES[at]))
            }
            _ => None,
        }
    }
}

/// How Metal converts a render pass's clear color to a bfloat attachment's codes
/// ([`ExtendedRange::clear_texel`]). Its two routes depend on the pass and the attachment, not on
/// the release: an Apple M4 under macOS 27.0.1 (16 colors and 39 NaN colors, 34 shapes, each bfloat
/// format) and the consumer's Apple M3 Max under macOS 26.7 take each as the shape says. The
/// runtime chooses by its clear's pass and attachment. A `BGR10_XR` or `BGRA10_XR` attachment and
/// their `_sRGB` twins take the same two routes by the same shapes (the M4 under 27.0.1; the
/// consumer's M3 Max under 26.7 byte for byte alike); a two-plane form's are one (each of its 2,574
/// colors' planes alike).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ClearRoute {
    /// Each channel through `float` and then `half`, each rounded to nearest even (past the
    /// largest `half`, the infinity), then the bfloat code nearest that `half`, ties to even; a
    /// NaN `0x7fc0`, its sign and payload dropped. Metal's clear of a texture it does not keep
    /// losslessly compressed (a level under 16 texels a side, `ShaderWrite` or `PixelFormatView`
    /// usage, `allowGPUOptimizedContents` off), and of any texture in a pass that records a draw
    /// call, even one that draws nothing.
    ThroughHalf,
    /// Each channel through `float`, rounded to nearest even, then the bfloat code nearest that,
    /// ties to even, `float` subnormals kept: `-3e38` is `0xff62`, `1e-40` is `0x0001`, `70000` is
    /// `0x4789`, `1 + 2^-8 + 2^-23` is `0x3f81`. A NaN is the `float` NaN of its sign and its top
    /// 23 payload bits, quieted, cut to its top 16 bits (`0xfff0_0400_0000_0000` is `0xffc0`,
    /// `0x7ff8_4000_0000_0000` is `0x7fc2`, `0x7fff_ffff_ffff_ffff` is `0x7fff`). Metal's clear of
    /// a texture it keeps losslessly compressed, the attachment's level 16 texels or more a side,
    /// by a pass that records no draw call (the consumer's M3 Max rows of a 16x16 `Private`
    /// texture). A `BGR10_XR` or `BGRA10_XR` channel's code there is straight from the `float` too:
    /// a linear channel's `(v + 384/510) · 510` in `float` steps, ties up, a color NaN 0 and an
    /// alpha NaN 384, an `_sRGB` color channel's by Metal's own encode, and `BGR10_XR`'s bits 30
    /// and 31 zero.
    Straight,
}

/// The bfloat code of `v` by [`ClearRoute::Straight`]: its `float` rounded to nearest even
/// (Rust's conversion), then [`bfloat_nearest_even`]; a NaN the `float` NaN of its sign and top
/// 23 payload bits, quieted, cut to its top 16 bits.
fn bfloat_of_double(v: f64) -> u32 {
    if v.is_nan() {
        let bits = v.to_bits();
        let sign = ((bits >> 63) as u32) << 31;
        let payload = ((bits >> 29) & 0x7f_ffff) as u32;
        return (sign | 0x7fc0_0000 | payload) >> 16;
    }
    bfloat_nearest_even(v as f32)
}

/// The bfloat code nearest `v`, ties to even (an overflow past the largest finite code the
/// infinity), a NaN the quiet NaN `0x7fc0`.
pub(crate) fn bfloat_nearest_even(v: f32) -> u32 {
    if v.is_nan() {
        return 0x7fc0;
    }
    // No sum passes `u32`: the largest non-NaN bits, `0xff80_0000`, plus `0x8000`.
    let b = v.to_bits();
    (b + 0x7fff + ((b >> 16) & 1)) >> 16
}

/// The layout of a two-plane YCbCr format ([`ycbcr_format`]): Metal's texture over an IOSurface's
/// luma plane (plane 0, a code a texel) and its chroma plane (plane 1, a Cb and a Cr code for
/// each texel, each 2 x 1 texels or each 2 x 2 texels), which a function reads as RGB. The
/// facts are the ones the format's name states, which the names Metal prints for the raw values
/// carry (an Apple M4 under macOS 27.0.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct YCbCrFormat {
    /// The bits of each luma and chroma code: 8, 10 or 12.
    pub bits: u32,
    /// How many luma texels share a chroma pair.
    pub subsampling: ChromaSubsampling,
    /// Whether the codes are packed (`_PACKED`), rather than one to a byte (8 bits) or to a
    /// 16-bit word, in its high bits (10 and 12 bits).
    pub packed: bool,
    /// What a texel reads.
    pub reads: YCbCrReads,
}

/// How many luma texels of a two-plane YCbCr format share a chroma pair ([`YCbCrFormat`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChromaSubsampling {
    /// 4:2:0: each 2 x 2 texels.
    Chroma420,
    /// 4:2:2: each 2 x 1 texels.
    Chroma422,
    /// 4:4:4: each texel.
    Chroma444,
}

impl ChromaSubsampling {
    /// Whether the chroma plane holds a pair for each two luma columns, and for each two rows.
    pub fn halved(self) -> [bool; 2] {
        match self {
            ChromaSubsampling::Chroma420 => [true, true],
            ChromaSubsampling::Chroma422 => [true, false],
            ChromaSubsampling::Chroma444 => [false, false],
        }
    }
}

/// What a texel of a two-plane YCbCr format reads ([`YCbCrFormat`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum YCbCrReads {
    /// The codes, unconverted: red Cr, green Y, blue Cb (`RGB*_2P`).
    Codes,
    /// RGB converted from the codes by the surface's range and matrix (`YCBCR*_2P`).
    Converted,
    /// The converted RGB through the sRGB curve (`_sRGB`).
    Srgb,
    /// The converted RGB through the PQ curve (`_PQ`).
    Pq,
    /// The converted RGB through the HLG curve (`_HLG`).
    Hlg,
    /// The converted RGB in the extended range (`_XR`).
    ExtendedRange,
}

/// The two-plane YCbCr formats: their table, [`ycbcr_table`], and the pattern
/// `two_plane_ycbcr!()` matching each, for this module's matches that name every format.
macro_rules! ycbcr_formats {
    ($($pf:ident: $bits:literal, $subsampling:ident, $packed:literal, $reads:ident;)*) => {
        macro_rules! two_plane_ycbcr {
            () => {
                $(PixelFormat::$pf)|*
            };
        }

        /// [`ycbcr_format`]'s table.
        fn ycbcr_table(pf: PixelFormat) -> Option<YCbCrFormat> {
            Some(match pf {
                $(PixelFormat::$pf => YCbCrFormat {
                    bits: $bits,
                    subsampling: ChromaSubsampling::$subsampling,
                    packed: $packed,
                    reads: YCbCrReads::$reads,
                },)*
                _ => return None,
            })
        }
    };
}

/// The layout of `pf` if it is one of the 69 two-plane YCbCr formats Metal makes a texture of
/// over an IOSurface's planes (every one over a surface of a matching
/// OSType, at plane 0 and plane 1; none from a descriptor alone), `None` for every other format.
/// Which ones this crate holds a texture of is [`ycbcr`]'s answer.
pub fn ycbcr_format(pf: PixelFormat) -> Option<YCbCrFormat> {
    ycbcr_table(pf)
}

ycbcr_formats! {
    YCBCR84202P: 8, Chroma420, false, Converted;
    YCBCR84222P: 8, Chroma422, false, Converted;
    YCBCR84442P: 8, Chroma444, false, Converted;
    YCBCR104202P: 10, Chroma420, false, Converted;
    YCBCR104222P: 10, Chroma422, false, Converted;
    YCBCR104442P: 10, Chroma444, false, Converted;
    YCBCR104202PPACKED: 10, Chroma420, true, Converted;
    YCBCR104222PPACKED: 10, Chroma422, true, Converted;
    YCBCR104442PPACKED: 10, Chroma444, true, Converted;
    YCBCR84202PSRGB: 8, Chroma420, false, Srgb;
    YCBCR84222PSRGB: 8, Chroma422, false, Srgb;
    YCBCR84442PSRGB: 8, Chroma444, false, Srgb;
    YCBCR104202PSRGB: 10, Chroma420, false, Srgb;
    YCBCR104222PSRGB: 10, Chroma422, false, Srgb;
    YCBCR104442PSRGB: 10, Chroma444, false, Srgb;
    YCBCR104202PPACKEDSRGB: 10, Chroma420, true, Srgb;
    YCBCR104222PPACKEDSRGB: 10, Chroma422, true, Srgb;
    YCBCR104442PPACKEDSRGB: 10, Chroma444, true, Srgb;
    RGB84202P: 8, Chroma420, false, Codes;
    RGB84222P: 8, Chroma422, false, Codes;
    RGB84442P: 8, Chroma444, false, Codes;
    RGB104202P: 10, Chroma420, false, Codes;
    RGB104222P: 10, Chroma422, false, Codes;
    RGB104442P: 10, Chroma444, false, Codes;
    RGB104202PPACKED: 10, Chroma420, true, Codes;
    RGB104222PPACKED: 10, Chroma422, true, Codes;
    RGB104442PPACKED: 10, Chroma444, true, Codes;
    YCBCR104202PPQ: 10, Chroma420, false, Pq;
    YCBCR104222PPQ: 10, Chroma422, false, Pq;
    YCBCR104442PPQ: 10, Chroma444, false, Pq;
    YCBCR104202PPACKEDPQ: 10, Chroma420, true, Pq;
    YCBCR104222PPACKEDPQ: 10, Chroma422, true, Pq;
    YCBCR104442PPACKEDPQ: 10, Chroma444, true, Pq;
    YCBCR124202P: 12, Chroma420, false, Converted;
    YCBCR124222P: 12, Chroma422, false, Converted;
    YCBCR124442P: 12, Chroma444, false, Converted;
    YCBCR124202PPQ: 12, Chroma420, false, Pq;
    YCBCR124222PPQ: 12, Chroma422, false, Pq;
    YCBCR124442PPQ: 12, Chroma444, false, Pq;
    YCBCR124202PPACKED: 12, Chroma420, true, Converted;
    YCBCR124222PPACKED: 12, Chroma422, true, Converted;
    YCBCR124442PPACKED: 12, Chroma444, true, Converted;
    YCBCR124202PPACKEDPQ: 12, Chroma420, true, Pq;
    YCBCR124222PPACKEDPQ: 12, Chroma422, true, Pq;
    YCBCR124442PPACKEDPQ: 12, Chroma444, true, Pq;
    YCBCR104202PXR: 10, Chroma420, false, ExtendedRange;
    YCBCR104222PXR: 10, Chroma422, false, ExtendedRange;
    YCBCR104442PXR: 10, Chroma444, false, ExtendedRange;
    YCBCR104202PPACKEDXR: 10, Chroma420, true, ExtendedRange;
    YCBCR104222PPACKEDXR: 10, Chroma422, true, ExtendedRange;
    YCBCR104442PPACKEDXR: 10, Chroma444, true, ExtendedRange;
    YCBCR124202PXR: 12, Chroma420, false, ExtendedRange;
    YCBCR124222PXR: 12, Chroma422, false, ExtendedRange;
    YCBCR124442PXR: 12, Chroma444, false, ExtendedRange;
    YCBCR124202PPACKEDXR: 12, Chroma420, true, ExtendedRange;
    YCBCR124222PPACKEDXR: 12, Chroma422, true, ExtendedRange;
    YCBCR124442PPACKEDXR: 12, Chroma444, true, ExtendedRange;
    YCBCR104202PHLG: 10, Chroma420, false, Hlg;
    YCBCR104222PHLG: 10, Chroma422, false, Hlg;
    YCBCR104442PHLG: 10, Chroma444, false, Hlg;
    YCBCR104202PPACKEDHLG: 10, Chroma420, true, Hlg;
    YCBCR104222PPACKEDHLG: 10, Chroma422, true, Hlg;
    YCBCR104442PPACKEDHLG: 10, Chroma444, true, Hlg;
    YCBCR124202PHLG: 12, Chroma420, false, Hlg;
    YCBCR124222PHLG: 12, Chroma422, false, Hlg;
    YCBCR124442PHLG: 12, Chroma444, false, Hlg;
    YCBCR124202PPACKEDHLG: 12, Chroma420, true, Hlg;
    YCBCR124222PPACKEDHLG: 12, Chroma422, true, Hlg;
    YCBCR124442PPACKEDHLG: 12, Chroma444, true, Hlg;
}

/// One candidate representation.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    format: vk::Format,
    components: vk::ComponentMapping,
    emulation: FormatEmulation,
}

const fn direct(format: vk::Format) -> Candidate {
    Candidate {
        format,
        components: vk::ComponentMapping::IDENTITY,
        emulation: FormatEmulation::Direct,
    }
}

/// The same bit layout with red and blue exchanged, read back through the view's mapping.
const fn red_blue_twin(format: vk::Format) -> Candidate {
    use vk::ComponentSwizzle as S;
    Candidate {
        format,
        components: vk::ComponentMapping {
            r: S::B,
            g: S::IDENTITY,
            b: S::R,
            a: S::IDENTITY,
        },
        emulation: FormatEmulation::SwizzledView,
    }
}

const fn combined(format: vk::Format, emulation: FormatEmulation) -> Candidate {
    Candidate {
        format,
        components: vk::ComponentMapping::IDENTITY,
        emulation,
    }
}

/// What the translator knows about a Metal pixel format.
enum Repr {
    /// Ordered candidates with the format's aspects.
    Candidates(vk::ImageAspectFlags, Vec<Candidate>),
    /// Representable only through a mechanism this crate does not model.
    NotImplemented,
    /// Not a concrete format (`Invalid`, `Unspecialized`).
    NotAFormat,
}

fn color(candidates: Vec<Candidate>) -> Repr {
    Repr::Candidates(vk::ImageAspectFlags::COLOR, candidates)
}

fn one(format: vk::Format) -> Repr {
    color(vec![direct(format)])
}

/// The candidate table's entry for a two-plane YCbCr format ([`ycbcr_format`]) whose layout the
/// conversion reads (`bind::YCbCrCodes::of`): one read converted held as Metal's converted values,
/// all on the grid of 1/2048 in [0, 1], which half floats hold exactly; one read as 10-bit codes
/// held as the `RGB10A2Unorm` texels of its codes, one read as 8-bit codes as the `RGBA8Unorm` ones
/// ([`ycbcr`]). The others, whose reads are not yet modelled exactly, `NotImplemented`.
fn ycbcr_repr(pf: PixelFormat) -> Repr {
    use crate::bind::{YCbCrCodes, YCbCrOutput};
    let format = match ycbcr_format(pf)
        .and_then(YCbCrCodes::of)
        .map(|c| (c.output, c.bits))
    {
        Some((YCbCrOutput::Converted, _)) => vk::Format::R16G16B16A16_SFLOAT,
        Some((YCbCrOutput::Srgb, _)) => vk::Format::R32G32B32A32_SFLOAT,
        Some((YCbCrOutput::Codes, 8)) => vk::Format::R8G8B8A8_UNORM,
        Some((YCbCrOutput::Codes, _)) => vk::Format::A2B10G10R10_UNORM_PACK32,
        None => return Repr::NotImplemented,
    };
    let sampled = combined(format, FormatEmulation::YCbCr);
    // A pass renders a format whose store is matched in floats, which the consumer encodes; a
    // packed format's pipeline is made in the same floats, and its pass refused.
    let attachment = match (ycbcr_renders(pf), ycbcr_pass_refused(pf)) {
        (true, _) => Some(FormatEmulation::YCbCrRendered),
        (false, true) => Some(FormatEmulation::YCbCrPassRefused),
        (false, false) => None,
    };
    match attachment {
        Some(emulation) => color(vec![
            sampled,
            combined(vk::Format::R32G32B32A32_SFLOAT, emulation),
        ]),
        None => color(vec![sampled]),
    }
}

/// The candidates of a compressed format whose block is `block`: the block, then the decoded twin
/// whose view is `twin` ([`FormatEmulation::Decoded`]).
fn decoded(block: vk::Format, twin: vk::Format) -> Repr {
    color(vec![
        direct(block),
        combined(twin, FormatEmulation::Decoded),
    ])
}

/// The pattern of ASTC's 4x2 and 8x4 formats, LDR, sRGB and HDR: Metal makes a 2D texture of
/// each from a descriptor alone (an Apple M4 under macOS 27.0.1), and no Vulkan
/// ASTC block has either footprint, so this crate holds none of them (`repr`'s `NotImplemented`).
macro_rules! astc_without_a_vulkan_block {
    () => {
        PixelFormat::ASTC4x2LDR
            | PixelFormat::ASTC4x2SRGB
            | PixelFormat::ASTC4x2HDR
            | PixelFormat::ASTC8x4LDR
            | PixelFormat::ASTC8x4SRGB
            | PixelFormat::ASTC8x4HDR
    };
}

/// The pattern of the formats Metal names that this crate holds no texture of yet (`repr`'s
/// `NotImplemented`): every raw value from 0 to 1100 a texture descriptor's description names on
/// an Apple M4 under macOS 27.0.1 that no format above takes. Every Vulkan format of each is
/// refused `NotImplemented(Format)`, so the runtime refuses a texture of one, a view in one and a
/// heap's size query for one where it plans the image, after the descriptor rules Metal stops for
/// every format. Their reads, writes and samples are measured one family at a time before any is
/// held.
macro_rules! formats_named_not_held {
    () => {
        PixelFormat::BGRA4Unorm
            | PixelFormat::RGB8Unorm
            | PixelFormat::RGB8Snorm
            | PixelFormat::RGB8Uint
            | PixelFormat::RGB8Sint
            | PixelFormat::ARGB8UnormSRGB
            | PixelFormat::RGB16Unorm
            | PixelFormat::RGB16Snorm
            | PixelFormat::RGB16Uint
            | PixelFormat::RGB16Sint
            | PixelFormat::RGB16Float
            | PixelFormat::RGB16Bfloat
            | PixelFormat::RGB32Uint
            | PixelFormat::RGB32Sint
            | PixelFormat::RGB32Float
            | PixelFormat::YCBCR84221P
            | PixelFormat::YCBCR104441P
            | PixelFormat::YCBCR84221PSRGB
            | PixelFormat::YCBCR104441PSRGB
            | PixelFormat::YCBCRA84441P
            | PixelFormat::YCBCR84221PREV
            | PixelFormat::YCBCR104441PPQ
            | PixelFormat::RGB10A2UnormPQ
            | PixelFormat::R10UnormPACKED
            | PixelFormat::RG10UnormPACKED
            | PixelFormat::YCBCR104441PXR
            | PixelFormat::R12UnormX4
            | PixelFormat::R12UnormX4PQ
            | PixelFormat::RG12UnormX8
            | PixelFormat::R10UnormX6PQ
            | PixelFormat::YCBCR104441PHLG
            | PixelFormat::R10UnormX6HLG
            | PixelFormat::R12UnormX4HLG
            | PixelFormat::RGB10A2UnormHLG
            | PixelFormat::RGB10A82P
            | PixelFormat::RGB10A82PSRGB
            | PixelFormat::RGB10A82PPQ
            | PixelFormat::RGB10A82PHLG
            | PixelFormat::BGR10A2UnormSRGB
            | PixelFormat::BGR10A2UnormPQ
            | PixelFormat::BGRA10UintPACKED
            | PixelFormat::BGRA10XRPACKED
            | PixelFormat::BGRA10XRPACKEDSRGB
            | PixelFormat::BGRA10UnormPACKED
            | PixelFormat::BGRA10UnormPACKEDSRGB
            | PixelFormat::BGRA10UnormPACKEDHLG
            | PixelFormat::BGRA10UnormPACKEDPQ
            | PixelFormat::BGRA10Uint
            | PixelFormat::BGRA10Unorm
            | PixelFormat::BGRA10UnormSRGB
            | PixelFormat::BGRA10UnormHLG
            | PixelFormat::BGRA10UnormPQ
            | PixelFormat::R10UintX6
            | PixelFormat::R10UintPACKED
            | PixelFormat::RG10UintX12
            | PixelFormat::RG10UintPACKED
            | PixelFormat::R12UintX4
            | PixelFormat::R12UintPACKED
            | PixelFormat::RG12UintX8
            | PixelFormat::RG12UintPACKED
            | PixelFormat::RGB10A8Uint2P
    };
}

/// Whether a render pipeline on `profile` blends into a color attachment of `pf`, one of Apple's
/// extended-range formats ([`ExtendedRange::of`]). No device blends such an attachment's raw
/// texels: the fragment function computes Metal's blend in the shader, inside the pixel
/// interlock, from the texel it reads. `Ok` where the device orders fetch through the pixel
/// interlock alone and that form is exact for an attachment of `pf` written whole under a blend
/// Metal's model holds; else what blocks it. A pipeline's own blend, write mask or output may still
/// be refused by name (`Feature::ExtendedRangeBlend`), as [`rotated_channels`]' blends are per
/// pipeline. Another format answers `NotImplemented(Format)`.
pub fn extended_range_blend(pf: PixelFormat, profile: &DeviceProfile) -> Result<(), Unsupported> {
    crate::lower::fragment::extended_range_blend(pf, profile)
}

/// Whether `pf` holds each of its 10-bit codes in the top 10 bits of its own little-endian
/// 16-bit word, the low 6 unread: the padded formats `R10Unorm_X6` and `RG10Unorm_X12` and their
/// `_sRGB` twins. `RGB10A2Unorm_sRGB` packs its codes in one 32-bit word instead, so it is not
/// one of these ([`ExtendedRange::padded`] names both sets).
///
/// Metal makes a texture of such a format wherever it makes one of any format, in memory of its
/// own as much as over an IOSurface's plane or a buffer. Measured on an Apple M3 Max under macOS
/// 27.0.1 (26A434) for all four, 2D 8x8 unless said: `newTextureWithDescriptor:` answers a
/// texture in `Shared`, `Private` and `Memoryless` storage, under `ShaderRead`,
/// `ShaderRead | ShaderWrite` and `ShaderRead | RenderTarget` usage, with four levels, as an
/// array of three slices, and as a texture buffer of 64 texels; an automatic, a placement and a
/// sparse heap each place one, and `heapTextureSizeAndAlignWithDescriptor:` answers the tight
/// linear size at an alignment of 128. A shader read answers the code's value and a shader write
/// stores the code, a blit from a buffer lands exactly, and a render pass clears and draws into
/// one. Only `newSharedTextureWithDescriptor:` answers nil, and only a multisample descriptor is
/// stopped by Metal's validation.
///
/// Metal's own host copies of such a texture are exact only while it keeps the texture linear —
/// `allocatedSize` equal to the tight `width * bytes * height`, which held for every height up
/// to 8 at widths to 1024, for heights to 12 at a width of 16, and at any extent whose usage
/// includes `ShaderWrite`. Where Metal picks its compressed layout instead (an extra 128 bytes
/// or more in `allocatedSize`: 16x16 and 64x64 under `ShaderRead` alone) both host copies are
/// silent no-ops: `getBytes:` leaves its destination untouched and `replaceRegion:` leaves the
/// texture zero as a shader reads it, while a blit either way still moves the texels. The
/// runtime's copies are exact at every extent (docs/API.md, accepted differences).
pub fn padded_words(pf: PixelFormat) -> bool {
    matches!(
        pf,
        PixelFormat::R10UnormX6
            | PixelFormat::RG10UnormX12
            | PixelFormat::R10UnormX6SRGB
            | PixelFormat::RG10UnormX12SRGB
    )
}

/// The texel extent of one block of the compressed format `pf`, `None` for an uncompressed one.
///
/// Metal states this nowhere: `MTLPixelFormat` has no block-extent getter, and a program that
/// needs the number reads it off Apple's format tables. It is still a fact of the format and not
/// of any one use of it, so it lives here beside the rest of what a format is, and every caller
/// that has to round an extent up to whole blocks — a copy's region, a level's row pitch, the
/// blocks a `replaceRegion:` hands over — reads this rather than keeping a table of its own.
///
/// Every compressed format Metal names answers, the ones this crate holds no texture of
/// included: ASTC's 4x2 and 8x4 footprints, which no Vulkan ASTC block has, answer `[4, 2]` and
/// `[8, 4]`, because the extent is what decides the type rules Metal applies to a texture of the
/// format and those rules hold whether or not a device can sample it. An ASTC format's three
/// forms — `_LDR`, `_sRGB` and `_HDR` — share one footprint, since the profile changes how a
/// block decodes and not how many texels it covers.
///
/// A block's **bytes** are not here: they are a property of the Vulkan format this crate maps
/// `pf` to, which [`vk_format`] answers, and the two numbers are wanted in different places.
pub fn block_extent(pf: PixelFormat) -> Option<[u32; 2]> {
    use PixelFormat as P;
    Some(match pf {
        P::BC1RGBA
        | P::BC1RGBASRGB
        | P::BC2RGBA
        | P::BC2RGBASRGB
        | P::BC3RGBA
        | P::BC3RGBASRGB
        | P::BC4RUnorm
        | P::BC4RSnorm
        | P::BC5RGUnorm
        | P::BC5RGSnorm
        | P::BC6HRGBFloat
        | P::BC6HRGBUfloat
        | P::BC7RGBAUnorm
        | P::BC7RGBAUnormSRGB
        | P::EACR11Unorm
        | P::EACR11Snorm
        | P::EACRG11Unorm
        | P::EACRG11Snorm
        | P::EACRGBA8
        | P::EACRGBA8SRGB
        | P::ETC2RGB8
        | P::ETC2RGB8SRGB
        | P::ETC2RGB8A1
        | P::ETC2RGB8A1SRGB
        | P::PVRTCRGB4BPP
        | P::PVRTCRGB4BPPSRGB
        | P::PVRTCRGBA4BPP
        | P::PVRTCRGBA4BPPSRGB
        | P::ASTC4x4LDR
        | P::ASTC4x4SRGB
        | P::ASTC4x4HDR => [4, 4],
        P::PVRTCRGB2BPP | P::PVRTCRGB2BPPSRGB | P::PVRTCRGBA2BPP | P::PVRTCRGBA2BPPSRGB => [8, 4],
        P::ASTC5x4LDR | P::ASTC5x4SRGB | P::ASTC5x4HDR => [5, 4],
        P::ASTC5x5LDR | P::ASTC5x5SRGB | P::ASTC5x5HDR => [5, 5],
        P::ASTC6x5LDR | P::ASTC6x5SRGB | P::ASTC6x5HDR => [6, 5],
        P::ASTC6x6LDR | P::ASTC6x6SRGB | P::ASTC6x6HDR => [6, 6],
        P::ASTC8x5LDR | P::ASTC8x5SRGB | P::ASTC8x5HDR => [8, 5],
        P::ASTC8x6LDR | P::ASTC8x6SRGB | P::ASTC8x6HDR => [8, 6],
        P::ASTC8x8LDR | P::ASTC8x8SRGB | P::ASTC8x8HDR => [8, 8],
        P::ASTC10x5LDR | P::ASTC10x5SRGB | P::ASTC10x5HDR => [10, 5],
        P::ASTC10x6LDR | P::ASTC10x6SRGB | P::ASTC10x6HDR => [10, 6],
        P::ASTC10x8LDR | P::ASTC10x8SRGB | P::ASTC10x8HDR => [10, 8],
        P::ASTC10x10LDR | P::ASTC10x10SRGB | P::ASTC10x10HDR => [10, 10],
        P::ASTC12x10LDR | P::ASTC12x10SRGB | P::ASTC12x10HDR => [12, 10],
        P::ASTC12x12LDR | P::ASTC12x12SRGB | P::ASTC12x12HDR => [12, 12],
        P::ASTC4x2LDR | P::ASTC4x2SRGB | P::ASTC4x2HDR => [4, 2],
        P::ASTC8x4LDR | P::ASTC8x4SRGB | P::ASTC8x4HDR => [8, 4],
        _ => return None,
    })
}

/// Whether `pf`'s image holds Metal's channels rotated ([`FormatEmulation::RotatedView`],
/// `ARGB8Unorm`): a color attachment of it takes the function's output, its clear color and its
/// write mask through [`to_rotated_image`], and its border texel is read through the rotation.
pub fn rotated_channels(pf: PixelFormat) -> bool {
    match repr(pf) {
        Repr::Candidates(_, candidates) => candidates
            .first()
            .is_some_and(|c| c.emulation == FormatEmulation::RotatedView),
        Repr::NotImplemented | Repr::NotAFormat => false,
    }
}

/// Whether a texture of `pf` holds Metal's alpha in its image's red on `profile`
/// ([`FormatEmulation::AlphaInRed`]: `A8Unorm` on a device that holds no `A8_UNORM` image, for
/// sampling, storage or attachments): its image is [`alpha_in_red_format`]'s for every use, its
/// storage choice `StorageImageClass::AlphaInRed`, and a color attachment of it takes the
/// function's output, its clear color and its write mask through [`to_rotated_image`] as
/// [`rotated_channels`]' formats do (the rotation's red is Metal's alpha), and its blend in red
/// (`pipeline::rotated_blend::alpha_in_red_blend`).
pub fn alpha_in_red(pf: PixelFormat, profile: &DeviceProfile) -> bool {
    let held = |own: vk::Format| {
        [
            FormatUsage::SampledImage,
            FormatUsage::StorageImage,
            FormatUsage::ColorAttachment,
        ]
        .into_iter()
        .any(|usage| profile.supports_format(own, usage))
    };
    alpha_in_red_format(pf).is_some() && direct_format(pf).is_some_and(|own| !held(own))
}

/// The format a texel buffer view of a texture buffer of `pf` takes for `usage` (a texel buffer
/// usage) on `profile` where the format's own does not serve it and the bytes hold Metal's alpha in
/// red ([`FormatEmulation::AlphaInRed`]): `R8_UNORM` for `A8Unorm` on a device that views no
/// `A8_UNORM` texels for the usage (a Vulkan 1.2 device without `VK_KHR_maintenance5`), where it
/// views `R8_UNORM` ones. A view reads `(r, 0, 0, 1)` where Metal reads `(0, 0, 0, r)`: a storage
/// view's reads and writes convert by its choice's class (`StorageImageClass::AlphaInRed`,
/// [`storage_choice`]), and a uniform view's reads take the alpha from red through the slot's
/// swizzle (`TextureSwizzleChannels::alpha_in_red`). `None` elsewhere.
pub(crate) fn texel_buffer_alpha_in_red(
    pf: PixelFormat,
    usage: FormatUsage,
    profile: &DeviceProfile,
) -> Option<vk::Format> {
    let own = direct_format(pf)?;
    let held = alpha_in_red_format(pf)?;
    (!profile.supports_format(own, usage) && profile.supports_format(held, usage)).then_some(held)
}

/// The format of `pf`'s image where the device holds no image of its direct format and `pf`
/// holds Metal's alpha in red ([`FormatEmulation::AlphaInRed`]): `R8_UNORM` for `A8Unorm`,
/// `None` for every other format.
pub fn alpha_in_red_format(pf: PixelFormat) -> Option<vk::Format> {
    match repr(pf) {
        Repr::Candidates(_, candidates) => candidates
            .iter()
            .find(|c| c.emulation == FormatEmulation::AlphaInRed)
            .map(|c| c.format),
        Repr::NotImplemented | Repr::NotAFormat => None,
    }
}

/// The candidate of an RGBX format held in its RGBA twin `format`
/// ([`FormatEmulation::XChannel`]): sampled through a view that reads alpha as one.
fn x_channel_twin(format: vk::Format) -> Repr {
    use vk::ComponentSwizzle as S;
    color(vec![Candidate {
        format,
        components: vk::ComponentMapping {
            r: S::IDENTITY,
            g: S::IDENTITY,
            b: S::IDENTITY,
            a: S::ONE,
        },
        emulation: FormatEmulation::XChannel,
    }])
}

/// Whether `pf` is one of Apple's RGBX formats, held in its RGBA twin
/// ([`FormatEmulation::XChannel`]): `RGBX8Unorm`, its `_sRGB` twin, `RGBX8Snorm`, `RGBX8Uint`,
/// `RGBX8Sint`, `RGBX16Uint`, `RGBX16Sint`, `RGBX16Float`, `RGBX32Uint`, `RGBX32Sint` and
/// `RGBX32Float`. Measured on an Apple M4 under macOS 27.0.1 (our own probes, each format beside
/// its RGBA twin on the same bytes and values): a texel is the twin's bytes, and its red, green and
/// blue read, sample, gather and convert as the twin's; every read of the fourth field (Metal's X)
/// gives alpha 1 (`1.0`, or the integer 1): a function's read and sample, a gather of alpha, a read
/// out of bounds, a border texel of any border color, a texture buffer's read, a framebuffer fetch
/// and a blend's destination alpha. The GPU's stores write X as the bits of the green field they
/// store: a function's writes (the texels written alone; a texture buffer's too), and, through
/// `RGBX8Unorm`, `RGBX16Float` and `RGBX32Float`, the formats Metal renders into, a render pass's
/// tiles, a resolve, the samples of a multisample texture and the levels `generateMipmaps`
/// computes. A CPU copy and a blit move the bytes as they are, and a view of another format reads
/// and writes them as its own.
pub fn x_channel(pf: PixelFormat) -> bool {
    match repr(pf) {
        Repr::Candidates(_, candidates) => candidates
            .first()
            .is_some_and(|c| c.emulation == FormatEmulation::XChannel),
        Repr::NotImplemented | Repr::NotAFormat => false,
    }
}

/// Metal's channels `[red, green, blue, alpha]` as the image of a
/// [`FormatEmulation::RotatedView`] format holds them, in its red, green, blue and alpha: alpha,
/// red, green, blue.
pub fn to_rotated_image<T: Copy>(metal: [T; 4]) -> [T; 4] {
    [metal[3], metal[0], metal[1], metal[2]]
}

/// The candidate table's entry for `pf` ([`make_repr`]): the table is made once, for every format,
/// at the first ask. A draw asks it of its pass's and its textures' formats, a dozen times each,
/// and each entry made anew allocated its candidates.
fn repr(pf: PixelFormat) -> &'static Repr {
    static TABLE: OnceLock<Vec<Repr>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let len = PixelFormat::ALL
            .iter()
            .map(|pf| pf.as_raw() as usize + 1)
            .max()
            .unwrap_or(0);
        let mut table: Vec<Repr> = (0..len).map(|_| Repr::NotAFormat).collect();
        for &pf in PixelFormat::ALL {
            table[pf.as_raw() as usize] = make_repr(pf);
        }
        table
    });
    &table[pf.as_raw() as usize]
}

/// The candidate table. Order is preference: the direct format first.
fn make_repr(pf: PixelFormat) -> Repr {
    use vk::ComponentSwizzle as S;
    use vk::Format as F;
    use FormatEmulation as E;
    use PixelFormat as P;
    let depth = vk::ImageAspectFlags::DEPTH;
    let stencil = vk::ImageAspectFlags::STENCIL;
    let depth_stencil = depth | stencil;
    match pf {
        P::Invalid | P::Unspecialized => Repr::NotAFormat,
        // Metal presents (0, 0, 0, a), as `VK_FORMAT_A8_UNORM` does (core 1.4 or
        // `VK_KHR_maintenance5`). Without it, the byte is `R8_UNORM`'s, its alpha in red.
        P::A8Unorm => color(vec![
            direct(F::A8_UNORM),
            Candidate {
                format: F::R8_UNORM,
                components: vk::ComponentMapping {
                    r: S::ZERO,
                    g: S::ZERO,
                    b: S::ZERO,
                    a: S::R,
                },
                emulation: E::AlphaInRed,
            },
        ]),
        P::R8Unorm => one(F::R8_UNORM),
        P::R8UnormSRGB => one(F::R8_SRGB),
        P::R8Snorm => one(F::R8_SNORM),
        P::R8Uint => one(F::R8_UINT),
        P::R8Sint => one(F::R8_SINT),
        P::R16Unorm => one(F::R16_UNORM),
        P::R16Snorm => one(F::R16_SNORM),
        P::R16Uint => one(F::R16_UINT),
        P::R16Sint => one(F::R16_SINT),
        P::R16Float => one(F::R16_SFLOAT),
        P::RG8Unorm => one(F::R8G8_UNORM),
        P::RG8UnormSRGB => one(F::R8G8_SRGB),
        P::RG8Snorm => one(F::R8G8_SNORM),
        P::RG8Uint => one(F::R8G8_UINT),
        P::RG8Sint => one(F::R8G8_SINT),
        P::B5G6R5Unorm => color(vec![
            direct(F::R5G6B5_UNORM_PACK16),
            red_blue_twin(F::B5G6R5_UNORM_PACK16),
        ]),
        P::A1BGR5Unorm => color(vec![
            direct(F::R5G5B5A1_UNORM_PACK16),
            red_blue_twin(F::B5G5R5A1_UNORM_PACK16),
        ]),
        P::ABGR4Unorm => color(vec![
            direct(F::R4G4B4A4_UNORM_PACK16),
            red_blue_twin(F::B4G4R4A4_UNORM_PACK16),
        ]),
        P::BGR5A1Unorm => one(F::A1R5G5B5_UNORM_PACK16),
        P::R32Uint => one(F::R32_UINT),
        P::R32Sint => one(F::R32_SINT),
        P::R32Float => one(F::R32_SFLOAT),
        P::RG16Unorm => one(F::R16G16_UNORM),
        P::RG16Snorm => one(F::R16G16_SNORM),
        P::RG16Uint => one(F::R16G16_UINT),
        P::RG16Sint => one(F::R16G16_SINT),
        P::RG16Float => one(F::R16G16_SFLOAT),
        P::RGBA8Unorm => color(vec![
            direct(F::R8G8B8A8_UNORM),
            red_blue_twin(F::B8G8R8A8_UNORM),
        ]),
        P::RGBA8UnormSRGB => color(vec![
            direct(F::R8G8B8A8_SRGB),
            red_blue_twin(F::B8G8R8A8_SRGB),
        ]),
        P::RGBA8Snorm => one(F::R8G8B8A8_SNORM),
        P::RGBA8Uint => one(F::R8G8B8A8_UINT),
        P::RGBA8Sint => one(F::R8G8B8A8_SINT),
        P::BGRA8Unorm => color(vec![
            direct(F::B8G8R8A8_UNORM),
            red_blue_twin(F::R8G8B8A8_UNORM),
        ]),
        P::BGRA8UnormSRGB => color(vec![
            direct(F::B8G8R8A8_SRGB),
            red_blue_twin(F::R8G8B8A8_SRGB),
        ]),
        // Alpha in byte 0, then red, green, and blue (measured): no Vulkan format has that order,
        // so its bytes are `R8G8B8A8_UNORM` read through a rotating view.
        P::ARGB8Unorm => color(vec![Candidate {
            format: F::R8G8B8A8_UNORM,
            components: vk::ComponentMapping {
                r: S::G,
                g: S::B,
                b: S::A,
                a: S::R,
            },
            emulation: FormatEmulation::RotatedView,
        }]),
        P::RGB10A2Unorm => color(vec![
            direct(F::A2B10G10R10_UNORM_PACK32),
            red_blue_twin(F::A2R10G10B10_UNORM_PACK32),
        ]),
        P::RGB10A2Uint => color(vec![
            direct(F::A2B10G10R10_UINT_PACK32),
            red_blue_twin(F::A2R10G10B10_UINT_PACK32),
        ]),
        P::RG11B10Float => one(F::B10G11R11_UFLOAT_PACK32),
        P::RGB9E5Float => one(F::E5B9G9R9_UFLOAT_PACK32),
        P::BGR10A2Unorm => color(vec![
            direct(F::A2R10G10B10_UNORM_PACK32),
            red_blue_twin(F::A2B10G10R10_UNORM_PACK32),
        ]),
        // Apple's extended-range formats: 10-bit codes `c` of the value `(c - 384) / 510`
        // (sRGB-encoded in the `_sRGB` twins), which no Vulkan format has, held raw in the
        // integer format of their bytes.
        P::BGR10XR | P::BGR10XRSRGB => color(vec![combined(F::R32_UINT, E::ExtendedRange)]),
        P::BGRA10XR | P::BGRA10XRSRGB => {
            color(vec![combined(F::R16G16B16A16_UINT, E::ExtendedRange)])
        }
        // Two planes, the extended-range color word and an alpha byte, held side by side in one
        // image.
        P::RGB10A82PXR10 | P::RGB10A82PXR10SRGB => {
            color(vec![combined(F::R32G32_UINT, E::ExtendedRange)])
        }
        // The padded 10-bit formats: each code in a 16-bit word's top bits, held raw in the
        // integer format of the words ([`ExtendedRange::R10X6`]).
        P::R10UnormX6 | P::R10UnormX6SRGB => color(vec![combined(F::R16_UINT, E::ExtendedRange)]),
        P::RG10UnormX12 | P::RG10UnormX12SRGB => {
            color(vec![combined(F::R16G16_UINT, E::ExtendedRange)])
        }
        // `RGB10A2Unorm`'s word, its color codes sRGB-encoded by Metal's own tables, which no
        // Vulkan format has: held raw ([`ExtendedRange::Rgb10A2Srgb`]).
        P::RGB10A2UnormSRGB => color(vec![combined(F::R32_UINT, E::ExtendedRange)]),
        // The bfloat formats: each code held raw in the integer format of its words.
        P::R16Bfloat => color(vec![combined(F::R16_UINT, E::Bfloat)]),
        P::RG16Bfloat => color(vec![combined(F::R16G16_UINT, E::Bfloat)]),
        P::RGBA16Bfloat => color(vec![combined(F::R16G16B16A16_UINT, E::Bfloat)]),
        P::RG32Uint => one(F::R32G32_UINT),
        P::RG32Sint => one(F::R32G32_SINT),
        P::RG32Float => one(F::R32G32_SFLOAT),
        P::RGBA16Unorm => one(F::R16G16B16A16_UNORM),
        P::RGBA16Snorm => one(F::R16G16B16A16_SNORM),
        P::RGBA16Uint => one(F::R16G16B16A16_UINT),
        P::RGBA16Sint => one(F::R16G16B16A16_SINT),
        P::RGBA16Float => one(F::R16G16B16A16_SFLOAT),
        P::RGBA32Uint => one(F::R32G32B32A32_UINT),
        P::RGBA32Sint => one(F::R32G32B32A32_SINT),
        P::RGBA32Float => one(F::R32G32B32A32_SFLOAT),
        // Apple's RGBX formats: the RGBA twin's bytes, the fourth field read as alpha 1.
        P::RGBX8Unorm => x_channel_twin(F::R8G8B8A8_UNORM),
        P::RGBX8UnormSRGB => x_channel_twin(F::R8G8B8A8_SRGB),
        P::RGBX8Snorm => x_channel_twin(F::R8G8B8A8_SNORM),
        P::RGBX8Uint => x_channel_twin(F::R8G8B8A8_UINT),
        P::RGBX8Sint => x_channel_twin(F::R8G8B8A8_SINT),
        P::RGBX16Uint => x_channel_twin(F::R16G16B16A16_UINT),
        P::RGBX16Sint => x_channel_twin(F::R16G16B16A16_SINT),
        P::RGBX16Float => x_channel_twin(F::R16G16B16A16_SFLOAT),
        P::RGBX32Uint => x_channel_twin(F::R32G32B32A32_UINT),
        P::RGBX32Sint => x_channel_twin(F::R32G32B32A32_SINT),
        P::RGBX32Float => x_channel_twin(F::R32G32B32A32_SFLOAT),
        P::BC1RGBA => one(F::BC1_RGBA_UNORM_BLOCK),
        P::BC1RGBASRGB => one(F::BC1_RGBA_SRGB_BLOCK),
        P::BC2RGBA => one(F::BC2_UNORM_BLOCK),
        P::BC2RGBASRGB => one(F::BC2_SRGB_BLOCK),
        P::BC3RGBA => one(F::BC3_UNORM_BLOCK),
        P::BC3RGBASRGB => one(F::BC3_SRGB_BLOCK),
        P::BC4RUnorm => one(F::BC4_UNORM_BLOCK),
        P::BC4RSnorm => one(F::BC4_SNORM_BLOCK),
        P::BC5RGUnorm => one(F::BC5_UNORM_BLOCK),
        P::BC5RGSnorm => one(F::BC5_SNORM_BLOCK),
        P::BC6HRGBFloat => one(F::BC6H_SFLOAT_BLOCK),
        P::BC6HRGBUfloat => one(F::BC6H_UFLOAT_BLOCK),
        P::BC7RGBAUnorm => one(F::BC7_UNORM_BLOCK),
        P::BC7RGBAUnormSRGB => one(F::BC7_SRGB_BLOCK),
        // PVRTC1 blocks behind `VK_IMG_format_pvrtc` (never promoted), else the decoded twin's
        // view of their codes (`FormatEmulation::Decoded`). Vulkan has only the RGBA formats;
        // reading Metal's RGB formats through them needs the alpha a decoder returns for RGB
        // data, which is not established (a swizzle to one would assume it), so those take the
        // twin alone.
        P::PVRTCRGBA2BPP => decoded(F::PVRTC1_2BPP_UNORM_BLOCK_IMG, F::R8G8B8A8_UNORM),
        P::PVRTCRGBA2BPPSRGB => decoded(F::PVRTC1_2BPP_SRGB_BLOCK_IMG, F::R8G8B8A8_SRGB),
        P::PVRTCRGBA4BPP => decoded(F::PVRTC1_4BPP_UNORM_BLOCK_IMG, F::R8G8B8A8_UNORM),
        P::PVRTCRGBA4BPPSRGB => decoded(F::PVRTC1_4BPP_SRGB_BLOCK_IMG, F::R8G8B8A8_SRGB),
        P::PVRTCRGB2BPP | P::PVRTCRGB4BPP => {
            color(vec![combined(F::R8G8B8A8_UNORM, FormatEmulation::Decoded)])
        }
        P::PVRTCRGB2BPPSRGB | P::PVRTCRGB4BPPSRGB => {
            color(vec![combined(F::R8G8B8A8_SRGB, FormatEmulation::Decoded)])
        }
        // ETC2 and EAC: the block, else the decoded twin's view (`FormatEmulation::Decoded`) of
        // its codes, or an 11-bit format's compact texels (`bind::EtcTexels`).
        P::EACR11Unorm => decoded(F::EAC_R11_UNORM_BLOCK, F::R16_UNORM),
        P::EACR11Snorm => decoded(F::EAC_R11_SNORM_BLOCK, F::R16_SNORM),
        P::EACRG11Unorm => decoded(F::EAC_R11G11_UNORM_BLOCK, F::R16G16_UNORM),
        P::EACRG11Snorm => decoded(F::EAC_R11G11_SNORM_BLOCK, F::R16G16_SNORM),
        P::EACRGBA8 => decoded(F::ETC2_R8G8B8A8_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::EACRGBA8SRGB => decoded(F::ETC2_R8G8B8A8_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ETC2RGB8 => decoded(F::ETC2_R8G8B8_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ETC2RGB8SRGB => decoded(F::ETC2_R8G8B8_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ETC2RGB8A1 => decoded(F::ETC2_R8G8B8A1_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ETC2RGB8A1SRGB => decoded(F::ETC2_R8G8B8A1_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        // ASTC LDR: the block, else the decoded twin's view (`FormatEmulation::Decoded`).
        P::ASTC4x4SRGB => decoded(F::ASTC_4X4_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC5x4SRGB => decoded(F::ASTC_5X4_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC5x5SRGB => decoded(F::ASTC_5X5_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC6x5SRGB => decoded(F::ASTC_6X5_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC6x6SRGB => decoded(F::ASTC_6X6_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC8x5SRGB => decoded(F::ASTC_8X5_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC8x6SRGB => decoded(F::ASTC_8X6_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC8x8SRGB => decoded(F::ASTC_8X8_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC10x5SRGB => decoded(F::ASTC_10X5_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC10x6SRGB => decoded(F::ASTC_10X6_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC10x8SRGB => decoded(F::ASTC_10X8_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC10x10SRGB => decoded(F::ASTC_10X10_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC12x10SRGB => decoded(F::ASTC_12X10_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC12x12SRGB => decoded(F::ASTC_12X12_SRGB_BLOCK, F::R8G8B8A8_SRGB),
        P::ASTC4x4LDR => decoded(F::ASTC_4X4_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC5x4LDR => decoded(F::ASTC_5X4_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC5x5LDR => decoded(F::ASTC_5X5_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC6x5LDR => decoded(F::ASTC_6X5_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC6x6LDR => decoded(F::ASTC_6X6_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC8x5LDR => decoded(F::ASTC_8X5_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC8x6LDR => decoded(F::ASTC_8X6_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC8x8LDR => decoded(F::ASTC_8X8_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC10x5LDR => decoded(F::ASTC_10X5_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC10x6LDR => decoded(F::ASTC_10X6_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC10x8LDR => decoded(F::ASTC_10X8_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC10x10LDR => decoded(F::ASTC_10X10_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC12x10LDR => decoded(F::ASTC_12X10_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        P::ASTC12x12LDR => decoded(F::ASTC_12X12_UNORM_BLOCK, F::R8G8B8A8_UNORM),
        // The ASTC HDR profile: the `SFLOAT` blocks (core 1.3 or
        // `VK_EXT_texture_compression_astc_hdr`). Like every compressed format they are usable
        // where the format properties say so; `textureCompressionASTC_HDR` only guarantees that.
        P::ASTC4x4HDR => one(F::ASTC_4X4_SFLOAT_BLOCK),
        P::ASTC5x4HDR => one(F::ASTC_5X4_SFLOAT_BLOCK),
        P::ASTC5x5HDR => one(F::ASTC_5X5_SFLOAT_BLOCK),
        P::ASTC6x5HDR => one(F::ASTC_6X5_SFLOAT_BLOCK),
        P::ASTC6x6HDR => one(F::ASTC_6X6_SFLOAT_BLOCK),
        P::ASTC8x5HDR => one(F::ASTC_8X5_SFLOAT_BLOCK),
        P::ASTC8x6HDR => one(F::ASTC_8X6_SFLOAT_BLOCK),
        P::ASTC8x8HDR => one(F::ASTC_8X8_SFLOAT_BLOCK),
        P::ASTC10x5HDR => one(F::ASTC_10X5_SFLOAT_BLOCK),
        P::ASTC10x6HDR => one(F::ASTC_10X6_SFLOAT_BLOCK),
        P::ASTC10x8HDR => one(F::ASTC_10X8_SFLOAT_BLOCK),
        P::ASTC10x10HDR => one(F::ASTC_10X10_SFLOAT_BLOCK),
        P::ASTC12x10HDR => one(F::ASTC_12X10_SFLOAT_BLOCK),
        P::ASTC12x12HDR => one(F::ASTC_12X12_SFLOAT_BLOCK),
        // 4:2:2: `G8B8G8R8_422_UNORM`/`B8G8R8G8_422_UNORM` exist (core 1.1), but a view of them
        // needs a sampler Y′CbCr conversion (VUID-VkImageViewCreateInfo-format-06415), whose
        // chroma reconstruction is the implementation's. The decoded twin's view
        // (`FormatEmulation::Decoded`) holds Metal's texels instead (`bind::ChromaDecode`).
        P::GBGR422 | P::BGRG422 => {
            color(vec![combined(F::R8G8B8A8_UNORM, FormatEmulation::Decoded)])
        }
        two_plane_ycbcr!() => ycbcr_repr(pf),
        // Formats Metal makes no texture of from a descriptor alone; one over
        // an IOSurface plane is not modelled.
        P::RGBA16FloatXR | P::AnyDepth | P::AnyStencil => Repr::NotImplemented,
        // ASTC's 4x2 and 8x4 footprints, which Metal makes textures of: no
        // Vulkan ASTC block has them, and decoding their blocks is not built.
        astc_without_a_vulkan_block!() => Repr::NotImplemented,
        formats_named_not_held!() => Repr::NotImplemented,
        P::Depth16Unorm => Repr::Candidates(depth, vec![direct(F::D16_UNORM)]),
        P::Depth32Float => Repr::Candidates(
            depth,
            vec![
                direct(F::D32_SFLOAT),
                combined(F::D32_SFLOAT_S8_UINT, E::DepthInCombined),
            ],
        ),
        P::Stencil8 => Repr::Candidates(
            stencil,
            vec![
                direct(F::S8_UINT),
                combined(F::D32_SFLOAT_S8_UINT, E::StencilInCombined),
                combined(F::D24_UNORM_S8_UINT, E::StencilInCombined),
            ],
        ),
        P::Depth24UnormStencil8 => Repr::Candidates(
            depth_stencil,
            vec![
                direct(F::D24_UNORM_S8_UINT),
                combined(F::D32_SFLOAT_S8_UINT, E::DepthWidened),
            ],
        ),
        P::Depth32FloatStencil8 => {
            Repr::Candidates(depth_stencil, vec![direct(F::D32_SFLOAT_S8_UINT)])
        }
        // Stencil views of the combined formats: the same candidates as the texture they view,
        // so a view maps to its texture's format for the same usages.
        P::X32Stencil8 => Repr::Candidates(stencil, vec![direct(F::D32_SFLOAT_S8_UINT)]),
        P::X24Stencil8 => Repr::Candidates(
            stencil,
            vec![
                direct(F::D24_UNORM_S8_UINT),
                combined(F::D32_SFLOAT_S8_UINT, E::StencilInCombined),
            ],
        ),
    }
}

/// The Vulkan representation of `pf` valid for every usage in `usages` on `profile`.
///
/// Storage usages ask whether the image itself holds a storage-capable format; the shader-side
/// declaration and any alias view come from [`storage_image_choice`].
///
/// Errors: `InvalidDescriptor` for `Invalid`/`Unspecialized`, and
/// `InvalidDescriptor(PixelFormatUsage)` for an attachment usage the format's aspects do not
/// have (a color attachment of a depth or stencil format, a depth/stencil attachment of a color
/// format: Metal rejects both descriptors); `Unsupported{Format}` with `NotRepresentable` or
/// `NotImplemented` for formats without a candidate or when no candidate admits the usages;
/// otherwise `Unsupported{Format}` needing, per candidate, what the profile lacks: the version or
/// extension that exposes its format value where the profile does not, then the `(format, usage)`
/// pairs it does not support.
pub fn vk_format(
    pf: PixelFormat,
    usages: &BTreeSet<FormatUsage>,
    profile: &DeviceProfile,
) -> Result<FormatMapping, Error> {
    let (aspects, candidates) = candidates(pf)?;
    if let Some(&usage) = usages.iter().find(|&&u| !aspects_admit(aspects, u)) {
        return Err(Error::InvalidDescriptor(
            DescriptorError::PixelFormatUsage {
                format: pf.as_raw(),
                usage,
            },
        ));
    }
    let mut alternatives = Vec::new();
    for c in candidates
        .iter()
        .filter(|c| usages.iter().all(|&u| c.emulation.admits(u)))
    {
        let missing: Vec<Capability> = exposure_gap(c.format, profile)
            .into_iter()
            .chain(
                usages
                    .iter()
                    .filter(|&&usage| !profile.supports_format(c.format, usage))
                    .map(|&usage| Capability::Format {
                        format: c.format,
                        usage,
                    }),
            )
            .chain(view_mapping_gap(c.components, profile))
            .collect();
        if missing.is_empty() {
            return Ok(FormatMapping {
                format: c.format,
                components: c.components,
                aspects,
                emulation: c.emulation,
            });
        }
        alternatives.push(missing);
    }
    if alternatives.is_empty() {
        // No candidate is valid for these usages on any device (e.g. an `A8Unorm` attachment).
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    Err(Unsupported::needs_any_of(Feature::Format, alternatives).into())
}

/// The swizzle, over the channels of `own`, through which a texture of `own` reads what a view
/// of `pf` made with `swizzle` reads, where `pf` holds its texels in the very bits of `own`'s and
/// presents them in other channels: `own` samples as its direct Vulkan format on `profile`, and
/// one of `pf`'s candidates is [`FormatEmulation::SwizzledView`] or
/// [`FormatEmulation::AlphaInRed`] of that format (`A8Unorm` over `R8Unorm`, read through
/// `(Zero, Zero, Zero, Red)`). Exact for sampling and reads as that
/// emulation is: the same bits, the same filter, the channels moved by the view's mapping, which
/// Vulkan applies to gathered texels and border texels as to any other ("Texel Input Operations":
/// replacement, then the component swizzle). `None` elsewhere, for `pf` equal to `own`, where
/// `profile` takes no view of the composed mapping
/// (`VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`), and where an opaque-black border
/// would read otherwise: Vulkan leaves it undefined through a mapping other than the identity
/// unless the device swizzles borders (`VK_EXT_border_color_swizzle`), so the swizzle is given only
/// where Metal reads the transparent-black texel for opaque black through it
/// ([`opaque_black_reads_transparent`](crate::metal::sampler::opaque_black_reads_transparent)),
/// whose sampler every device reads exactly; transparent black and opaque white are defined through
/// any mapping. A write through it stores as a swizzled texture's storage binding does
/// (`LoweringInputs::swizzled_storage`): each of the image's channels from the first lane whose
/// source it is, else from the channel's own lane, as Metal stores each of the view's (probe
/// `a8write`, an M3 Max under macOS 27.0.1: an `A8Unorm` view of an `R8Unorm` texture stores the
/// first lane whose source is alpha, else the alpha lane). So `None` too where a channel of the
/// view that holds one of the image's is no lane's source (an `A8Unorm` view of swizzle
/// `(Red, Green, Blue, One)`), where the two fallbacks name different lanes.
pub fn swizzle_over(
    pf: PixelFormat,
    swizzle: TextureSwizzleChannels,
    own: PixelFormat,
    profile: &DeviceProfile,
) -> Option<TextureSwizzleChannels> {
    if pf == own {
        return None;
    }
    let sampled = [FormatUsage::SampledImage].into();
    let image = vk_format(own, &sampled, profile).ok()?;
    if image.emulation != FormatEmulation::Direct {
        return None;
    }
    let (_, candidates) = candidates(pf).ok()?;
    let candidate = candidates.iter().find(|c| {
        matches!(
            c.emulation,
            FormatEmulation::SwizzledView | FormatEmulation::AlphaInRed
        ) && c.format == image.format
    })?;
    let mapping = swizzle.compose(candidate.components);
    if view_mapping_gap(mapping, profile).is_some() {
        return None;
    }
    let over = swizzle_of(mapping);
    // A write stores each of the image's channels from the first lane whose source it is, else
    // from the channel's own lane: the view's channel holding it must be some lane's source, so
    // that the image's channel is too and both take the same lane.
    let held = swizzle_of(candidate.components);
    let lanes = [over.red, over.green, over.blue, over.alpha];
    let writes = [held.red, held.green, held.blue, held.alpha]
        .into_iter()
        .filter(|c| !matches!(c, TextureSwizzle::Zero | TextureSwizzle::One))
        .all(|c| lanes.contains(&c));
    let reads = crate::metal::sampler::opaque_black_reads_transparent(own, over);
    (reads && writes).then_some(over)
}

/// The swizzle that reads as component mapping `m` (in `TextureSwizzleChannels::compose`'s
/// canonical form over a format's own channels): each channel from the one `m` names, from itself
/// where `m` names none, or a constant.
fn swizzle_of(m: vk::ComponentMapping) -> TextureSwizzleChannels {
    use crate::metal::TextureSwizzle as T;
    use vk::ComponentSwizzle as S;
    let named = [
        (S::ZERO, T::Zero),
        (S::ONE, T::One),
        (S::R, T::Red),
        (S::G, T::Green),
        (S::B, T::Blue),
        (S::A, T::Alpha),
    ];
    let channel = |c: vk::ComponentSwizzle, own: T| {
        named.iter().find(|(s, _)| *s == c).map_or(own, |&(_, t)| t)
    };
    TextureSwizzleChannels::new(
        channel(m.r, T::Red),
        channel(m.g, T::Green),
        channel(m.b, T::Blue),
        channel(m.a, T::Alpha),
    )
}

/// What `profile` lacks to make an image view whose component mapping is `components`:
/// `imageViewFormatSwizzle`, where its portability subset withholds it and the mapping is not the
/// identity (`VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`). A swizzled candidate's
/// view and a Metal texture's swizzle composed onto it are such mappings.
pub(crate) fn view_mapping_gap(
    components: vk::ComponentMapping,
    profile: &DeviceProfile,
) -> Option<Capability> {
    let feature = VkFeature::ImageViewFormatSwizzle;
    (!crate::metal::sampler::is_identity(components)
        && profile.fields().enabled.portability_withholds(feature))
    .then_some(Capability::Feature(feature))
}

/// Whether a format with `aspects` can serve `usage`: color attachments need the color aspect,
/// depth/stencil attachments a depth or stencil aspect; every other usage is decided by the
/// candidates and the profile.
fn aspects_admit(aspects: vk::ImageAspectFlags, usage: FormatUsage) -> bool {
    let depth_stencil = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    match usage {
        FormatUsage::ColorAttachment | FormatUsage::ColorAttachmentBlend => {
            aspects == vk::ImageAspectFlags::COLOR
        }
        FormatUsage::DepthStencilAttachment => {
            !aspects.is_empty() && depth_stencil.contains(aspects)
        }
        _ => true,
    }
}

/// The version or extension `profile` lacks to use `format`'s enum value.
fn exposure_gap(format: vk::Format, profile: &DeviceProfile) -> Option<Capability> {
    let f = profile.fields();
    format_exposure_gap(format, f.api_version, &f.enabled.extensions)
}

/// The candidates of `pf`, or the refusal for a format without one.
fn candidates(pf: PixelFormat) -> Result<(vk::ImageAspectFlags, &'static [Candidate]), Error> {
    match repr(pf) {
        Repr::Candidates(aspects, candidates) => Ok((*aspects, candidates)),
        Repr::NotImplemented => Err(Unsupported::not_implemented(Feature::Format).into()),
        Repr::NotAFormat => Err(Error::InvalidDescriptor(
            DescriptorError::PlaceholderEnumValue {
                ty: PixelFormat::METAL_NAME,
                value: pf.as_raw(),
            },
        )),
    }
}

/// The UNORM format of an 8-bit sRGB format whose texels Metal decodes from their raw codes
/// (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`; a BC, PVRTC, ETC2 or
/// EAC sRGB format's unorm form, the same blocks): the same channels and bits, its texels the sRGB format's codes
/// read as `c / 255`. A device that cannot sample the sRGB format holds a sampled texture of it in
/// this format, and a pipeline reads the raw codes through its view and decodes them as Metal does.
/// `None` for any other format.
pub fn srgb_raw_twin(pf: PixelFormat) -> Option<PixelFormat> {
    use PixelFormat as P;
    Some(match pf {
        P::R8UnormSRGB => P::R8Unorm,
        P::RG8UnormSRGB => P::RG8Unorm,
        P::RGBA8UnormSRGB => P::RGBA8Unorm,
        P::BGRA8UnormSRGB => P::BGRA8Unorm,
        P::RGBX8UnormSRGB => P::RGBX8Unorm,
        P::BC1RGBASRGB => P::BC1RGBA,
        P::BC2RGBASRGB => P::BC2RGBA,
        P::BC3RGBASRGB => P::BC3RGBA,
        P::BC7RGBAUnormSRGB => P::BC7RGBAUnorm,
        P::PVRTCRGB2BPPSRGB => P::PVRTCRGB2BPP,
        P::PVRTCRGB4BPPSRGB => P::PVRTCRGB4BPP,
        P::PVRTCRGBA2BPPSRGB => P::PVRTCRGBA2BPP,
        P::PVRTCRGBA4BPPSRGB => P::PVRTCRGBA4BPP,
        P::ETC2RGB8SRGB => P::ETC2RGB8,
        P::ETC2RGB8A1SRGB => P::ETC2RGB8A1,
        P::EACRGBA8SRGB => P::EACRGBA8,
        _ => return None,
    })
}

/// The bit width of each channel of integer format `f`, red to alpha (a channel `f` lacks takes
/// the first one's width; nothing is stored there): the ranges into which Metal saturates each
/// channel of an integer texel's write, as the format's signedness reads the word. `None` for a
/// format whose channels are not integers.
pub(crate) fn integer_channel_bits(f: vk::Format) -> Option<[u32; 4]> {
    use crate::vk::NumericFormat as N;
    let block = f.block()?;
    if !matches!(block.color, Some(N::Sint | N::Uint)) {
        return None;
    }
    match (f, block.component_bits) {
        (vk::Format::A2B10G10R10_UINT_PACK32 | vk::Format::A2B10G10R10_SINT_PACK32, _) => {
            Some([10, 10, 10, 2])
        }
        (_, Some(bits)) => Some([bits; 4]),
        _ => None,
    }
}

/// The word `PushField::TextureIntegerChannels` carries for a texture of pixel format `pf`: byte
/// `c` holds `32 - bits` for channel `c` (red to alpha) of an integer format whose channel is
/// narrower than 32 bits ([`integer_channel_bits`]), else 0. A write saturates each lane to the
/// range `bits` spans in the image's signedness, so a word of 0 (a float, 32-bit integer or
/// unbound texture's) leaves every lane as it is.
pub(crate) fn integer_channels_word(pf: PixelFormat) -> u32 {
    direct_format(pf)
        .and_then(integer_channel_bits)
        .map_or(0, |bits| {
            bits.iter()
                .zip([0u32, 8, 16, 24])
                .fold(0, |word, (b, at)| word | (32 - (*b).min(32)) << at)
        })
}

/// Whether the device's conversion decides the NaN a `half` texel written to a texture of pixel
/// format `pf` stores, once the write hands it the half widened (16-bit float channels,
/// `metal::texel_format::half_channels`) or `0x7fc00000` (a packed small float's, `RG11B10Float`, `RGB9E5Float`):
/// the stores `DriverDeviation::MetalHalfTexelWrites` states Metal's. A 32-bit float channel stores
/// its bits, and a normalized or integer one no NaN.
pub fn half_write_nans_converted(pf: PixelFormat) -> bool {
    use crate::vk::NumericFormat as N;
    crate::metal::texel_format::half_channels(pf)
        || direct_format(pf).and_then(|f| f.block()).is_some_and(|b| {
            b.packed_bits.is_some() && matches!(b.color, Some(N::Ufloat | N::Sfloat))
        })
}

/// The class of pixel format `pf` (`metal::texel_format::TexelFormat`'s word, as its `half`
/// conversion takes it): the value a specialized note's test of
/// `SpecKey::BoundTexelFormat` reads for a texture of `pf` bound at its slot
/// (`StageModule::bound_format_clauses`), which a runtime takes from each draw's or dispatch's
/// bound textures.
pub fn bound_texel_format(pf: PixelFormat) -> u32 {
    crate::metal::texel_format::texel_format(pf).0
}

/// The direct Vulkan format of `pf` (its first candidate), independent of any device: `None` for
/// a format held only through its decoded twin's view ([`FormatEmulation::Decoded`], the PVRTC
/// RGB formats), whose blocks no Vulkan format holds.
pub fn direct_format(pf: PixelFormat) -> Option<vk::Format> {
    match repr(pf) {
        Repr::Candidates(_, c) => c
            .first()
            .filter(|c| c.emulation != FormatEmulation::Decoded)
            .map(|c| c.format),
        _ => None,
    }
}

/// The direct Vulkan format of `pf` where its image holds Metal's bytes of each texel, which CPU
/// copies, blits, views and textures over a buffer then move unchanged: [`direct_format`], but
/// `None` for the two-plane extended-range formats (`ExtendedRange::holds_metal_bytes`), whose
/// image holds both planes side by side.
pub fn byte_format(pf: PixelFormat) -> Option<vk::Format> {
    if ExtendedRange::of(pf).is_some_and(|f| !f.holds_metal_bytes()) || ycbcr(pf) {
        return None;
    }
    direct_format(pf)
}

/// Whether `pf` is a two-plane YCbCr format ([`ycbcr_format`]) this crate holds as the texels
/// Metal reads of its planes ([`FormatEmulation::YCbCr`]): `YCBCR8_420_2P`, `YCBCR10_420_2P`,
/// their `_sRGB` twins, `RGB8_420_2P` and `RGB10_420_2P`, and the 4:2:2 and 4:4:4 formats of
/// each (`bind::YCbCrCodes::of`), Metal's texture over an IOSurface's luma plane (plane 0, one code
/// a texel) and its chroma plane (plane 1, a Cb and a Cr code for each 2 x 2, 2 x 1 or single
/// texel), each code a byte (8 bits) or a 16-bit word's high bits (10 bits), which a function reads
/// as RGB. Measured on an Apple M4 under macOS 27.0.1 (our own probes over CoreVideo surfaces):
/// each texel reads `v / 2048` per channel for a format read converted, `v / 8192` for an `_sRGB`
/// twin, its codes `code / (2^b - 1)` (red Cr, green Y, blue Cb) for one read as codes, `v` the
/// conversion `bind::ycbcr_convert` states, alpha 1; a nearest sample and a gather read those
/// texels; a CPU copy or a blit between the texture and a buffer moves the luma plane's bytes (a
/// code's bytes a texel), and a blit between two such textures both planes'; a function's writes
/// store nothing. `byte_format` is `None`: the image holds the texels read, not the planes' bytes.
pub fn ycbcr(pf: PixelFormat) -> bool {
    matches!(
        repr(pf),
        Repr::Candidates(_, c) if c.first().is_some_and(|c| c.emulation == FormatEmulation::YCbCr)
    )
}

/// Whether Metal completes a function's writes to a texture of `pf` and stores nothing: the
/// 4:2:2 formats, the YCbCr formats ([`ycbcr`]) whose chroma is subsampled, 4:2:0 and 4:2:2 (a
/// kernel's write leaves both planes' bytes as they were, and Metal's validation stops the
/// binding), and the packed ones, `YCBCR10_444_2P_PACKED` and its `_sRGB` twin among them (the
/// M3 Max under macOS 27.0.1, our probe `ycw` over `p444` and `pf44` surfaces: every byte of both
/// planes kept, the command buffer completed; an Apple M4 stores them). A storage slot a function
/// only writes takes its neutral there (`bind::plan`), which the writes leave as Metal leaves the
/// texture.
///
/// An unpacked 4:4:4 YCbCr format's writes are not dropped: Metal stores each texel's codes into
/// both planes, which a slot that only writes one of [`ycbcr_writes_stored`]
/// does as Metal does; a slot that writes another is refused (`storage_choice`).
pub fn writes_dropped(pf: PixelFormat) -> bool {
    matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422)
        || (ycbcr(pf)
            && ycbcr_format(pf)
                .is_some_and(|f| f.subsampling != ChromaSubsampling::Chroma444 || f.packed))
}

/// Whether a function's writes to a two-plane YCbCr texture of `pf` ([`ycbcr`]) store into its
/// planes as Metal stores them, at a slot that only writes it
/// (`pipeline::lower_inputs::StorageImageClass::YCbCrWrite`): the 4:4:4 formats of unpacked 8-
/// and 10-bit codes read converted or as codes, `YCBCR8_444_2P`, `YCBCR10_444_2P`,
/// `RGB8_444_2P` and `RGB10_444_2P`. Metal stores each written texel's luma code into the luma
/// plane and its own chroma pair into the chroma plane, the codes `bind::ycbcr_write` states
/// (an Apple M4 under macOS 27.0.1, and an Apple M3 Max under macOS 26.7, byte for byte the same).
/// Not the `_sRGB` twins, whose writes no arithmetic found matches (a slot that writes one is
/// refused, `storage_choice`), nor the packed formats, whose writes Metal drops
/// ([`writes_dropped`]).
pub fn ycbcr_writes_stored(pf: PixelFormat) -> bool {
    ycbcr(pf)
        && ycbcr_format(pf).is_some_and(|f| {
            f.subsampling == ChromaSubsampling::Chroma444
                && !f.packed
                && matches!(f.bits, 8 | 10)
                && matches!(f.reads, YCbCrReads::Converted | YCbCrReads::Codes)
        })
}

/// Whether a render pass stores its color attachment of a two-plane YCbCr texture of `pf`
/// ([`ycbcr`]) into the texture's planes as Metal stores it: the formats of unpacked 8- and
/// 10-bit codes read converted, `YCBCR8_420_2P`, `YCBCR8_422_2P`, `YCBCR8_444_2P`,
/// `YCBCR10_420_2P`, `YCBCR10_422_2P` and `YCBCR10_444_2P`, or read as codes, `RGB8_420_2P` to
/// `RGB10_444_2P`. Metal re-encodes every texel of the attachment, drawn or not, into both planes
/// at the pass's store, by the codes `bind::ycbcr_store` states (an Apple M4 under macOS 27.0.1:
/// every byte of the 8-bit formats' clears and full-screen draws over `420v`, `420f`, `422v` and
/// `444v` surfaces, no matrix and BT.709; the consumer's run of the same probe on an Apple M3 Max
/// under macOS 26.7: the M4's bytes in every case; the same M4: every byte of the 10-bit formats'
/// draws and clears over `x420`, `xf20`, `x422`, `xf22`, `x444` and `xf44`; an Apple M3 Max under
/// macOS 27.0.1: every byte of the code formats' draws).
/// Such an attachment renders into `R32G32B32A32_SFLOAT` ([`FormatEmulation::YCbCrRendered`]),
/// through any write mask and, where the device blends that format, any blend (within a code of
/// Metal's, the runtime announces it; refused elsewhere, `Unsupported(Format)` naming the need).
/// Not the `_sRGB` or packed formats, whose stores no rule matches yet (a packed one's pass is
/// refused, [`ycbcr_pass_refused`]).
pub fn ycbcr_renders(pf: PixelFormat) -> bool {
    // Every such layout is held ([`ycbcr`]), whose representation asks this.
    ycbcr_format(pf).is_some_and(|f| {
        !f.packed
            && matches!(f.bits, 8 | 10)
            && matches!(f.reads, YCbCrReads::Converted | YCbCrReads::Codes)
    })
}

/// Whether `pf` is a packed 10-bit two-plane YCbCr format this crate holds ([`ycbcr`]), whose
/// pipeline a color attachment of is made ([`FormatEmulation::YCbCrPassRefused`]) and whose
/// render pass the runtime refuses at its start, `NotImplemented(Format)`: `YCBCR10_420_2P_PACKED`,
/// `YCBCR10_422_2P_PACKED`, `YCBCR10_444_2P_PACKED` and their `_sRGB` twins (the code twins
/// `RGB10_*_2P_PACKED` join once a texture of them is held). Metal makes the pipeline and fails
/// the pass (an Apple M4 under macOS 27.0.1, a clear and a draw of each over its packed surface, 16
/// x 8, plain; the consumer's runs of the same probe on an Apple M3 Max under macOS 26.7): the
/// command buffer ends in an error, `Hang` in most runs and `InnocentVictim` in the rest, which
/// code varying from run to run on an idle GPU; the pass changes 39 or 40 bytes of two rows of the
/// luma plane and none of the chroma plane, their values varying between runs of a case; a second
/// such failure in a process has Metal ignore the queue's later command buffers. No rule gives
/// those bytes or that code, so no exact form exists: the runtime's refusal is its gap.
pub fn ycbcr_pass_refused(pf: PixelFormat) -> bool {
    // Held: read converted or through the sRGB curve (`bind::YCbCrCodes::of`).
    ycbcr_format(pf)
        .is_some_and(|f| f.packed && f.bits == 10 && crate::bind::YCbCrCodes::of(f).is_some())
}

/// Metal's `imageblockSampleLength` of the color attachments of `formats`, by color index: each
/// set format's texel in index order, at its component's alignment (a packed format's at its
/// packed word), the total not rounded. An unset or `Invalid` format takes no room, and every set
/// one counts whether a function reaches it or not. Measured on an M3 Max and an M4:
/// `[R8Unorm, R32Float]` is 8, `[R32Float, R8Unorm]` 5, and eight `RGBA32Float` 128, for render and
/// tile pipeline states alike. `None` when a format has no direct Vulkan format or no whole-byte
/// component or packed word.
pub fn imageblock_sample_length(formats: &[Option<PixelFormat>]) -> Option<u32> {
    let mut end = 0u32;
    for (offset, size) in color_record(formats)?.into_iter().flatten() {
        end = end.max(offset.checked_add(size)?);
    }
    Some(end)
}

/// Where each color attachment of `formats` lies in Metal's per-sample record of a pixel's colors,
/// by color index: its texel's byte offset and size, packed as [`imageblock_sample_length`]
/// counts them (each set format's texel in index order at its component's alignment, a packed
/// format's at its packed word); `None` at an unset or `Invalid` format, which takes no room. A
/// render pipeline state writes its fragment function's colors into this record by its own
/// formats, and a render pass loads and stores each attachment from it by the pass's: measured
/// on an M4 (a state of `[R8Unorm, RGBA8Unorm]` in a pass of `[R8Unorm, R32Float]` stores the
/// state's color 1 at byte 1, which the pass's color 1 reads from byte 4; a state with color 1
/// alone stores it at byte 0, the pass's color 0). `None` when a format has no direct Vulkan format
/// or no whole-byte component or packed word.
pub fn color_record(formats: &[Option<PixelFormat>]) -> Option<Vec<Option<(u32, u32)>>> {
    let mut end = 0u32;
    let mut record = Vec::with_capacity(formats.len());
    for &format in formats {
        let Some(format) = format.filter(|&f| f != PixelFormat::Invalid) else {
            record.push(None);
            continue;
        };
        let block = direct_format(format)?.block()?;
        let align = block.vertex_alignment()?;
        let offset = end.checked_next_multiple_of(align)?;
        end = offset.checked_add(block.size)?;
        record.push(Some((offset, block.size)));
    }
    Some(record)
}

/// Metal's `imageblockSampleLength` of a render pipeline whose fragment function declares an
/// explicit imageblock of `data_size` bytes (`air.imageblock_data_size`) whose layouts align to
/// `alignment`, beside color attachments that pack to `colors` bytes
/// ([`imageblock_sample_length`]): the data, then the colors rounded up to 8 bytes and then to
/// `alignment`; no colors add nothing. Measured on an M3 Max and an M4 over nine layouts and 23
/// sets of color formats: a `char` with `R8Unorm` is 9, a `float4` with `R8Unorm` 32, a
/// `packed_float4` with it 24, and a `float4` and a `float` (32 bytes) with
/// `R8Unorm` and `RGBA32Float` 64. `None` past `u32`.
pub fn explicit_imageblock_sample_length(
    data_size: u32,
    alignment: u32,
    colors: u32,
) -> Option<u32> {
    let colors = colors
        .checked_next_multiple_of(8)?
        .checked_next_multiple_of(alignment.max(1))?;
    data_size.checked_add(colors)
}

/// The aspects of `pf` (`COLOR`, `DEPTH`, `STENCIL`, or `DEPTH | STENCIL`), for formats with a
/// Vulkan representation.
pub fn aspects(pf: PixelFormat) -> Option<vk::ImageAspectFlags> {
    match repr(pf) {
        Repr::Candidates(aspects, _) => Some(*aspects),
        _ => None,
    }
}

/// Whether `pf` has a depth or stencil aspect.
pub fn is_depth_stencil(pf: PixelFormat) -> bool {
    aspects(pf)
        .is_some_and(|a| a.intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL))
}

/// The numeric type a shader reads `pf` as: `Uint`/`Sint` for integer formats and `Uint` for
/// stencil-only views, `Float` otherwise (normalized, float, sRGB, compressed, depth).
///
/// Decided by the registry numeric format of the first (direct) candidate's Vulkan format.
pub fn sampled_type(pf: PixelFormat) -> Option<SampledType> {
    let (aspects, candidates) = candidates(pf).ok()?;
    if aspects == vk::ImageAspectFlags::STENCIL {
        return Some(SampledType::Uint);
    }
    if aspects != vk::ImageAspectFlags::COLOR {
        // Depth (or combined depth/stencil read through its depth aspect).
        return Some(SampledType::Float);
    }
    // An extended-range format's values are floats, held raw in an integer format, and a bfloat
    // format's too.
    if matches!(
        candidates[0].emulation,
        FormatEmulation::ExtendedRange | FormatEmulation::Bfloat
    ) {
        return Some(SampledType::Float);
    }
    numeric_type(candidates[0].format)
}

/// The fewest components a fragment color output to a color attachment of `pf` must have: the
/// position of the format's last channel in `(r, g, b, a)` order. Metal refuses a render pipeline
/// whose output at an attachment has fewer ("fragment shader color output does not have enough
/// components for the pixel format", whatever the write mask or blending), and `A8Unorm`, whose one
/// channel is alpha, needs all four ("requires 4 outputs"). Measured on an Apple M3 Max, macOS 26:
/// `R32Float` 1; `RG8Unorm` and `RG16Float` 2; `RG11B10Float` and `RGB9E5Float` 3; `RGB10A2Unorm`,
/// `RGBA8Unorm`, `BGRA8Unorm`, `RGBA16Float`, the `RGBA32` formats, and `A8Unorm` 4. The other
/// formats follow their channels, the two-plane YCbCr formats a pass renders ([`ycbcr_renders`])
/// too: 3, red, green and blue (not measured). `None` for a format that is no color attachment
/// (depth, stencil, compressed, and the placeholders).
pub(crate) fn color_output_components(pf: PixelFormat) -> Option<u8> {
    use PixelFormat as P;
    Some(match pf {
        P::R8Unorm
        | P::R8UnormSRGB
        | P::R8Snorm
        | P::R8Uint
        | P::R8Sint
        | P::R16Unorm
        | P::R16Snorm
        | P::R16Uint
        | P::R16Sint
        | P::R16Float
        | P::R32Uint
        | P::R32Sint
        | P::R32Float
        | P::R10UnormX6
        | P::R10UnormX6SRGB
        | P::R16Bfloat => 1,
        P::RG10UnormX12 | P::RG10UnormX12SRGB | P::RG16Bfloat => 2,
        P::RG8Unorm
        | P::RG8UnormSRGB
        | P::RG8Snorm
        | P::RG8Uint
        | P::RG8Sint
        | P::RG16Unorm
        | P::RG16Snorm
        | P::RG16Uint
        | P::RG16Sint
        | P::RG16Float
        | P::RG32Uint
        | P::RG32Sint
        | P::RG32Float => 2,
        P::B5G6R5Unorm | P::RG11B10Float | P::RGB9E5Float | P::BGR10XR | P::BGR10XRSRGB => 3,
        P::A8Unorm
        | P::A1BGR5Unorm
        | P::ABGR4Unorm
        | P::BGR5A1Unorm
        | P::RGBA8Unorm
        | P::RGBA8UnormSRGB
        | P::RGBA8Snorm
        | P::RGBA8Uint
        | P::RGBA8Sint
        | P::ARGB8Unorm
        | P::BGRA8Unorm
        | P::BGRA8UnormSRGB
        | P::RGB10A2Unorm
        | P::RGB10A2Uint
        | P::BGR10A2Unorm
        | P::RGBA16Unorm
        | P::RGBA16Snorm
        | P::RGBA16Uint
        | P::RGBA16Sint
        | P::RGBA16Float
        | P::BGRA10XR
        | P::BGRA10XRSRGB
        | P::RGB10A82PXR10
        | P::RGB10A82PXR10SRGB
        | P::RGB10A2UnormSRGB
        | P::RGBA16Bfloat
        | P::RGBA32Uint
        | P::RGBA32Sint
        | P::RGBA32Float => 4,
        // A two-plane YCbCr format a pipeline renders follows its channels, red, green and
        // blue (alpha reads 1; `ycbcr_renders`, `ycbcr_pass_refused`).
        _ if ycbcr_renders(pf) || ycbcr_pass_refused(pf) => 3,
        _ => return None,
    })
}

/// How a color attachment of a pixel format takes a fragment function's float output
/// ([`write_conversion`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteConversion {
    /// The format takes no float output (integer formats), or holds the function's floats for
    /// the consumer's store to convert (a two-plane YCbCr format a pass renders, which no
    /// pipeline blends).
    Exact,
    /// A 32-bit float format: it stores the value as written, but a blend other than factors one
    /// and zero is the device's arithmetic, whose rounding and NaN need not be Metal's (NVIDIA's
    /// 615.71.09 on the RTX 3090 stores some source-over sums one `f32` ulp from Metal's, and a
    /// blended NaN as `0x7fffffff` where Metal stores `0x7fc00000`).
    Float,
    /// Metal converts each channel by its rule (red, green, blue, alpha; `None` for a channel
    /// the format lacks), as probes measured.
    Rules([Option<ChannelWrite>; 4]),
    /// The format converts the value, by a rule not measured on Metal.
    Unmeasured,
    /// One of Apple's extended-range formats: the attachment holds raw texels, which the
    /// fragment function encodes itself (`LoweringInputs::extended_range_colors`).
    Encoded,
}

/// How a color attachment of `pf` takes a fragment function's float output: Metal's measured
/// conversion of each channel (probes 4 to 6 on an Apple M3 Max and an M4 drew every code's exact
/// value and every tie with two `f32` steps either side, plain and blended with factors one and
/// zero): the 8-, 10-, 16- and 2-bit unorm and the 8- and 16-bit snorm channels round the exact
/// product to nearest, the 8-bit sRGB channels encode through Metal's twelve-bit table, and the
/// half-float channels and `RG11B10Float`'s unsigned ones round toward zero. The packed 16-bit
/// unorm formats' channels round the value to a half first and then as a unorm channel
/// (`ChannelWrite::HalfUnorm`; every half in `[0, 1]`, the midpoints between them and every code's
/// value and tie of 1, 4, 5 and 6 bits with `f32` steps either side, plain and blended with factors
/// one and zero, every channel of the four formats). `RGB9E5Float` was measured only with equal
/// channels: its conversion stays unmeasured. `ARGB8Unorm` converts as `RGBA8Unorm` and stores the
/// codes rotated. `None` for a format that is no color attachment.
pub(crate) fn write_conversion(pf: PixelFormat) -> Option<WriteConversion> {
    use ChannelWrite as C;
    use PixelFormat as P;
    let u = |bits| Some(C::Unorm { bits });
    let s = |bits| {
        Some(C::Snorm {
            bits,
            blended: false,
        })
    };
    let f = |mantissa, signed| Some(C::FloatTowardZero { mantissa, signed });
    let h = f(10, true);
    let hu = |bits| Some(C::HalfUnorm { bits });
    let srgb = Some(C::Srgb);
    if ExtendedRange::of(pf).is_some() {
        return Some(WriteConversion::Encoded);
    }
    // A two-plane YCbCr attachment a pass renders holds the function's floats, which the
    // consumer's store converts as Metal's does (`FormatEmulation::YCbCrRendered`); a packed
    // one's pass is refused, so nothing converts its writes (`ycbcr_pass_refused`).
    if ycbcr_renders(pf) || ycbcr_pass_refused(pf) {
        return Some(WriteConversion::Exact);
    }
    let rules = match pf {
        P::R8Unorm => [u(8), None, None, None],
        P::RG8Unorm => [u(8), u(8), None, None],
        P::RGBA8Unorm | P::BGRA8Unorm => [u(8); 4],
        // Stored as `RGBA8Unorm`, its channels rotated after the conversion (16
        // colors, ties at half a code, NaN and infinities among them, byte for byte).
        P::ARGB8Unorm => [u(8); 4],
        P::R8UnormSRGB => [srgb, None, None, None],
        P::RG8UnormSRGB => [srgb, srgb, None, None],
        P::RGBA8UnormSRGB | P::BGRA8UnormSRGB => [srgb, srgb, srgb, u(8)],
        P::R16Unorm => [u(16), None, None, None],
        P::RG16Unorm => [u(16), u(16), None, None],
        P::RGBA16Unorm => [u(16); 4],
        P::RGB10A2Unorm | P::BGR10A2Unorm => [u(10), u(10), u(10), u(2)],
        P::R8Snorm => [s(8), None, None, None],
        P::RG8Snorm => [s(8), s(8), None, None],
        P::RGBA8Snorm => [s(8); 4],
        P::R16Snorm => [s(16), None, None, None],
        P::RG16Snorm => [s(16), s(16), None, None],
        P::RGBA16Snorm => [s(16); 4],
        P::R16Float => [h, None, None, None],
        P::RG16Float => [h, h, None, None],
        P::RGBA16Float => [h; 4],
        P::RG11B10Float => [f(6, false), f(6, false), f(5, false), None],
        P::A8Unorm => [None, None, None, u(8)],
        P::B5G6R5Unorm => [hu(5), hu(6), hu(5), None],
        P::A1BGR5Unorm | P::BGR5A1Unorm => [hu(5), hu(5), hu(5), hu(1)],
        P::ABGR4Unorm => [hu(4); 4],
        _ => {
            return color_output_components(pf).map(|_| match sampled_type(pf) {
                Some(SampledType::Float)
                    if matches!(pf, P::R32Float | P::RG32Float | P::RGBA32Float) =>
                {
                    WriteConversion::Float
                }
                Some(SampledType::Float) => WriteConversion::Unmeasured,
                _ => WriteConversion::Exact,
            });
        }
    };
    Some(WriteConversion::Rules(rules))
}

/// Whether an out-of-bounds `read` of a texture (or texture buffer) of `pf` reads alpha 1 rather
/// than 0: whether `pf` has no alpha component. Metal reads a zero texel out of bounds, each
/// component the format has 0 and each it lacks at its default, 0 for red, green, and blue and 1
/// for alpha (MSL 4.1 sections 2.9.1 and 6.13; measured on an Apple M3 Max, macOS 26.7, for 2D
/// textures of 30 color formats including `A8Unorm`, `B5G6R5Unorm`, `RG11B10Float`, `RGB9E5Float`,
/// `RGB10A2Uint`, `BC1_RGBA`, and `BC4_RUnorm`, every texture kind, and texture buffers). A depth
/// or stencil format's missing components are undefined in Metal; Vulkan's component substitution
/// reads their alpha as 1, which this keeps. `None` for the placeholders `Invalid` and
/// `Unspecialized`, which are no format. The match names every format, so a new one must state its
/// answer.
pub(crate) fn out_of_bounds_alpha_one(pf: PixelFormat) -> Option<bool> {
    use PixelFormat as P;
    Some(match pf {
        P::Invalid | P::Unspecialized => return None,
        // Formats of which this crate holds no texture (`repr`'s `NotImplemented`).
        P::RGBA16FloatXR | P::AnyDepth | P::AnyStencil => return None,
        astc_without_a_vulkan_block!() => return None,
        formats_named_not_held!() => return None,
        // No alpha: a missing alpha reads 1.
        P::R10UnormX6 | P::RG10UnormX12 | P::R10UnormX6SRGB | P::RG10UnormX12SRGB => true,
        // The bfloat formats, by their channels.
        P::R16Bfloat | P::RG16Bfloat => true,
        P::RGBA16Bfloat => false,
        // A two-plane YCbCr texture's texels read alpha 1; outside them
        // unmeasured. `None` for the ones of which this crate holds no texture.
        two_plane_ycbcr!() => return ycbcr(pf).then_some(true),
        // No alpha component.
        P::R8Unorm
        | P::R8UnormSRGB
        | P::R8Snorm
        | P::R8Uint
        | P::R8Sint
        | P::R16Unorm
        | P::R16Snorm
        | P::R16Uint
        | P::R16Sint
        | P::R16Float
        | P::RG8Unorm
        | P::RG8UnormSRGB
        | P::RG8Snorm
        | P::RG8Uint
        | P::RG8Sint
        | P::B5G6R5Unorm
        | P::R32Uint
        | P::R32Sint
        | P::R32Float
        | P::RG16Unorm
        | P::RG16Snorm
        | P::RG16Uint
        | P::RG16Sint
        | P::RG16Float
        | P::RG11B10Float
        | P::RGB9E5Float
        | P::BGR10XR
        | P::BGR10XRSRGB
        | P::RG32Uint
        | P::RG32Sint
        | P::RG32Float
        | P::BC4RUnorm
        | P::BC4RSnorm
        | P::BC5RGUnorm
        | P::BC5RGSnorm
        | P::BC6HRGBFloat
        | P::BC6HRGBUfloat
        | P::PVRTCRGB2BPP
        | P::PVRTCRGB2BPPSRGB
        | P::PVRTCRGB4BPP
        | P::PVRTCRGB4BPPSRGB
        | P::EACR11Unorm
        | P::EACR11Snorm
        | P::EACRG11Unorm
        | P::EACRG11Snorm
        | P::ETC2RGB8
        | P::ETC2RGB8SRGB
        | P::GBGR422
        | P::BGRG422
        | P::Depth16Unorm
        | P::Depth32Float
        | P::Stencil8
        | P::Depth24UnormStencil8
        | P::Depth32FloatStencil8
        | P::X32Stencil8
        | P::X24Stencil8 => true,
        // The RGBX formats read alpha 1 everywhere, out of bounds too.
        P::RGBX8Unorm
        | P::RGBX8UnormSRGB
        | P::RGBX8Snorm
        | P::RGBX8Uint
        | P::RGBX8Sint
        | P::RGBX16Uint
        | P::RGBX16Sint
        | P::RGBX16Float
        | P::RGBX32Uint
        | P::RGBX32Sint
        | P::RGBX32Float => true,
        // An alpha component.
        P::A8Unorm
        | P::A1BGR5Unorm
        | P::ABGR4Unorm
        | P::BGR5A1Unorm
        | P::RGBA8Unorm
        | P::RGBA8UnormSRGB
        | P::RGBA8Snorm
        | P::RGBA8Uint
        | P::RGBA8Sint
        | P::BGRA8Unorm
        | P::BGRA8UnormSRGB
        | P::RGB10A2Unorm
        | P::RGB10A2Uint
        | P::BGR10A2Unorm
        | P::RGBA16Unorm
        | P::RGBA16Snorm
        | P::RGBA16Uint
        | P::RGBA16Sint
        | P::RGBA16Float
        | P::BGRA10XR
        | P::BGRA10XRSRGB
        | P::ARGB8Unorm
        | P::RGB10A82PXR10
        | P::RGB10A82PXR10SRGB
        | P::RGB10A2UnormSRGB
        | P::RGBA32Uint
        | P::RGBA32Sint
        | P::RGBA32Float
        | P::BC1RGBA
        | P::BC1RGBASRGB
        | P::BC2RGBA
        | P::BC2RGBASRGB
        | P::BC3RGBA
        | P::BC3RGBASRGB
        | P::BC7RGBAUnorm
        | P::BC7RGBAUnormSRGB
        | P::PVRTCRGBA2BPP
        | P::PVRTCRGBA2BPPSRGB
        | P::PVRTCRGBA4BPP
        | P::PVRTCRGBA4BPPSRGB
        | P::EACRGBA8
        | P::EACRGBA8SRGB
        | P::ETC2RGB8A1
        | P::ETC2RGB8A1SRGB
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
        | P::ASTC4x4LDR
        | P::ASTC5x4LDR
        | P::ASTC5x5LDR
        | P::ASTC6x5LDR
        | P::ASTC6x6LDR
        | P::ASTC8x5LDR
        | P::ASTC8x6LDR
        | P::ASTC8x8LDR
        | P::ASTC10x5LDR
        | P::ASTC10x6LDR
        | P::ASTC10x8LDR
        | P::ASTC10x10LDR
        | P::ASTC12x10LDR
        | P::ASTC12x12LDR
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
        | P::ASTC12x12HDR => false,
    })
}

/// The channels a color texture of `pf` holds, `[red, green, blue, alpha]`: those its texels
/// carry, which Metal reads from them, where it reads each channel the format lacks as the
/// constant 0 (red, green, blue) or 1 (alpha) at every texel, mapped, unmapped or the border
/// (MSL 4.1 sections 2.9.1 and 6.13). `A8Unorm` holds alpha alone; a format without alpha holds
/// red, then green, then blue, as many as it has ([`color_output_components`], and the compressed
/// formats by their names). `None` for a depth or stencil format, whose missing components Metal
/// leaves undefined, for a format of no stated channels, and for the placeholders.
pub(crate) fn channels(pf: PixelFormat) -> Option<[bool; 4]> {
    use PixelFormat as P;
    if is_depth_stencil(pf) {
        return None;
    }
    let alpha = !out_of_bounds_alpha_one(pf)?;
    let color = match pf {
        P::A8Unorm => 0,
        _ if alpha => 3,
        P::BC4RUnorm | P::BC4RSnorm | P::EACR11Unorm | P::EACR11Snorm => 1,
        P::BC5RGUnorm | P::BC5RGSnorm | P::EACRG11Unorm | P::EACRG11Snorm => 2,
        P::BC6HRGBFloat
        | P::BC6HRGBUfloat
        | P::PVRTCRGB2BPP
        | P::PVRTCRGB2BPPSRGB
        | P::PVRTCRGB4BPP
        | P::PVRTCRGB4BPPSRGB
        | P::ETC2RGB8
        | P::ETC2RGB8SRGB
        | P::GBGR422
        | P::BGRG422 => 3,
        two_plane_ycbcr!() => 3,
        _ if x_channel(pf) => 3,
        _ => color_output_components(pf)?,
    };
    Some([color >= 1, color >= 2, color >= 3, alpha])
}

/// The non-storage format usages a texture created with Metal `usage` needs: `ShaderRead` →
/// sampled image; `RenderTarget` → color or depth/stencil attachment. Shader writes and atomics go
/// through [`storage_image_choice`], and filtering is checked against the sampler at plan time.
pub fn texture_usages(pf: PixelFormat, usage: TextureUsage) -> BTreeSet<FormatUsage> {
    let mut out = BTreeSet::new();
    if usage.contains(TextureUsage::SHADER_READ) {
        out.insert(FormatUsage::SampledImage);
    }
    if usage.contains(TextureUsage::RENDER_TARGET) {
        out.insert(if is_depth_stencil(pf) {
            FormatUsage::DepthStencilAttachment
        } else {
            FormatUsage::ColorAttachment
        });
    }
    out
}

/// The view type of a Metal texture type (`None` for texture buffers, which are texel buffers).
pub fn image_view_type(ty: TextureType) -> Option<vk::ImageViewType> {
    use vk::ImageViewType as V;
    use TextureType as T;
    Some(match ty {
        T::Type1D => V::TYPE_1D,
        T::Type1DArray => V::TYPE_1D_ARRAY,
        T::Type2D | T::Type2DMultisample => V::TYPE_2D,
        T::Type2DArray | T::Type2DMultisampleArray => V::TYPE_2D_ARRAY,
        T::Cube => V::CUBE,
        T::CubeArray => V::CUBE_ARRAY,
        T::Type3D => V::TYPE_3D,
        T::TextureBuffer => return None,
    })
}

/// The image type and create flags of a Metal texture type (`None` for texture buffers).
pub fn image_type(ty: TextureType) -> Option<(vk::ImageType, vk::ImageCreateFlags)> {
    use vk::ImageType as I;
    use TextureType as T;
    let none = vk::ImageCreateFlags::empty();
    Some(match ty {
        T::Type1D | T::Type1DArray => (I::TYPE_1D, none),
        T::Type2D | T::Type2DArray | T::Type2DMultisample | T::Type2DMultisampleArray => {
            (I::TYPE_2D, none)
        }
        T::Cube | T::CubeArray => (I::TYPE_2D, vk::ImageCreateFlags::CUBE_COMPATIBLE),
        T::Type3D => (I::TYPE_3D, none),
        T::TextureBuffer => return None,
    })
}

/// What a storage choice serves: a texture (a storage image, whose format's optimal-tiling features
/// apply) or a texture buffer (a storage texel buffer, whose buffer features apply), and whether
/// the function reads and writes texels through it. Vulkan grants access without a format per
/// format, per usage, and per access (`VkFormatFeatureFlags2`'s `STORAGE_READ_WITHOUT_FORMAT` and
/// `STORAGE_WRITE_WITHOUT_FORMAT`), not by the device's features alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StorageUse {
    /// A texture buffer; else a texture.
    pub texel_buffer: bool,
    pub read: bool,
    pub write: bool,
}

impl StorageUse {
    /// A texture read and written: the use a choice made without knowing the function's accesses
    /// serves ([`storage_image_choice`]).
    pub const READ_WRITE_TEXTURE: StorageUse = StorageUse {
        texel_buffer: false,
        read: true,
        write: true,
    };

    /// A texture the function reads (`read`) and writes (`write`).
    pub fn texture(read: bool, write: bool) -> StorageUse {
        StorageUse {
            texel_buffer: false,
            read,
            write,
        }
    }

    /// A texture buffer the function reads (`read`) and writes (`write`).
    pub fn texel_buffer(read: bool, write: bool) -> StorageUse {
        StorageUse {
            texel_buffer: true,
            read,
            write,
        }
    }

    /// The usage the bound format must support: `StorageImage` or `StorageTexelBuffer`.
    pub fn usage(self) -> FormatUsage {
        if self.texel_buffer {
            FormatUsage::StorageTexelBuffer
        } else {
            FormatUsage::StorageImage
        }
    }

    /// The without-format usages the use's accesses need, with the device feature of each: read,
    /// then write.
    fn without_format(self) -> impl Iterator<Item = (FormatUsage, VkFeature)> {
        use FormatUsage as U;
        let (read, write) = if self.texel_buffer {
            (
                U::StorageTexelBufferReadWithoutFormat,
                U::StorageTexelBufferWriteWithoutFormat,
            )
        } else {
            (
                U::StorageImageReadWithoutFormat,
                U::StorageImageWriteWithoutFormat,
            )
        };
        [
            (
                self.read,
                read,
                VkFeature::ShaderStorageImageReadWithoutFormat,
            ),
            (
                self.write,
                write,
                VkFeature::ShaderStorageImageWriteWithoutFormat,
            ),
        ]
        .into_iter()
        .filter_map(|(used, usage, feature)| used.then_some((usage, feature)))
    }
}

/// Whether a storage image or texel buffer viewed as `format` is declared without a format for
/// `use_` by preference on `profile`, so that one declaration, and with it one module, serves every
/// format a function accesses alike: for each access, the device's
/// `shaderStorageImage{Read,Write}WithoutFormat` and the format's without-format usage (the
/// profile's, which carries the per-format bits and what the specification implies). A device
/// without the feature that accesses formats by their own bits alone
/// ([`DeviceProfile::without_format_by_format_bits`]) keeps a typed declaration wherever SPIR-V
/// names the format, and takes one without a format only where nothing else declares the view
/// ([`without_format_gaps`]). A use that neither reads nor writes needs nothing.
pub(crate) fn accessed_without_format(
    format: vk::Format,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> bool {
    let images = &profile.fields().images;
    use_.without_format().all(|(usage, feature)| {
        let enabled = match feature {
            VkFeature::ShaderStorageImageReadWithoutFormat => images.storage_read_without_format,
            _ => images.storage_write_without_format,
        };
        enabled && profile.supports_format(format, usage)
    })
}

/// What `format` lacks for `use_` through a declaration without a format on `profile`, empty where
/// it can be accessed so: for each access, the device's `shaderStorageImage{Read,Write}WithoutFormat`
/// or the formats' own bits ([`DeviceProfile::without_format_by_format_bits`]: Vulkan 1.3 or
/// `VK_KHR_format_feature_flags2`), else the feature; then the format's without-format usage.
pub(crate) fn without_format_gaps(
    format: vk::Format,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> Vec<Capability> {
    let images = &profile.fields().images;
    let bits = profile.without_format_by_format_bits();
    let mut gaps = Vec::new();
    for (usage, feature) in use_.without_format() {
        let enabled = match feature {
            VkFeature::ShaderStorageImageReadWithoutFormat => images.storage_read_without_format,
            _ => images.storage_write_without_format,
        };
        if !(enabled || bits) {
            gaps.push(Capability::Feature(feature));
        } else if !profile.supports_format(format, usage) {
            gaps.push(Capability::Format { format, usage });
        }
    }
    gaps
}

/// How a storage texture of format `pf` is declared in the shader and viewed for a function that
/// reads and writes it.
///
/// `Identity` when the direct format supports `StorageImage` and the shader can declare it: with
/// the format `Unknown` where the device reads and writes the format without one (the device's
/// `shaderStorageImage{Read,Write}WithoutFormat` and the format's own
/// `STORAGE_{READ,WRITE}_WITHOUT_FORMAT` optimal-tiling bits, or what the specification implies
/// for them), else with its SPIR-V image format. Otherwise an alias view (bgra8 through rgba8 and
/// bgr10a2 through rgb10a2 with `SwizzleBgra`, the 8-bit sRGB formats through their UNORM twin with
/// `SrgbEncode` or bgra8 sRGB through rgba8 with `SrgbEncodeBgra`, `RGB9E5Float` through `R32_UINT`
/// with `SharedExponentBits`, `A8Unorm` through `R8_UNORM` with `AlphaInRed`), else
/// `Unsupported{Format}`. A read-and-write choice serves any access to a storage image, so a
/// caller that does not know the function's accesses uses it. It does not serve a texture
/// buffer: Vulkan implies no buffer bit, so a format formatless as an image can need a declared
/// format as a texel buffer. The binder's plan chooses for the slot's own uses, image or texel
/// buffer, and names the variant it needs wherever that choice differs.
pub fn storage_image_choice(
    pf: PixelFormat,
    profile: &DeviceProfile,
) -> Result<StorageImageChoice, Error> {
    storage_choice(pf, StorageUse::READ_WRITE_TEXTURE, profile)
}

/// How a storage texture or texture buffer of format `pf` is declared in the shader and viewed
/// for `use_`.
///
/// `Identity` when the direct format supports the use's storage usage and the shader can declare
/// it: with the format `Unknown` where the format serves the use without one
/// ([`accessed_without_format`]; every such format of one numeric type shares one key, and
/// `StorageImageChoice::numeric` keeps the type), else with its SPIR-V image format. A half-float
/// format the use writes takes the class its writes need there instead (`half_storage_class`).
/// Otherwise an alias view of a `MUTABLE_FORMAT` image (or the same bytes viewed as the alias
/// format, for a texture buffer): bgra8 through an rgba8 view and bgr10a2 through an rgb10a2 view
/// (`A2R10G10B10` through `A2B10G10R10`, which NVIDIA and lavapipe store where they do not store
/// the former) with `SwizzleBgra`, the 8-bit sRGB formats through their UNORM twin with
/// `SrgbEncode` (`BGRA8Unorm_sRGB`, where its twin is not accessed without a format, through
/// `R8G8B8A8_UNORM` with `SrgbEncodeBgra`), `RGB9E5Float` through `R32_UINT` with
/// `SharedExponentBits`, and `A8Unorm` through `R8_UNORM` with `AlphaInRed` (no alias: such a
/// texture's image is `R8_UNORM` itself, [`FormatEmulation::AlphaInRed`]; a texture buffer's bytes
/// take that view wherever `A8_UNORM` does not serve the use, [`texel_buffer_alpha_in_red`]).
/// Anything else is `Unsupported{Format}` naming what each route lacks, a format's without-format
/// usage included.
///
/// An RGBX format's choice ([`x_channel`]) is its RGBA twin's, which converts its texels in the
/// shader (`StorageImageChoice::x_channel`); its texture buffer, whose reads a texel buffer's view
/// cannot give alpha 1, is `NotImplemented(Format)`.
pub(crate) fn storage_choice(
    pf: PixelFormat,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> Result<StorageImageChoice, Error> {
    storage_choice_viewed(pf, use_, profile, true)
}

/// [`storage_choice`] of a texture whose image takes views in other formats only where
/// `aliased`: an image viewed in its own format alone (a wrapped one made without
/// `MUTABLE_FORMAT`) takes no half-float alias (`StorageImageClass::HalfBits`) and keeps its
/// format's own choice, as before.
pub(crate) fn storage_choice_viewed(
    pf: PixelFormat,
    use_: StorageUse,
    profile: &DeviceProfile,
    aliased: bool,
) -> Result<StorageImageChoice, Error> {
    let x = x_channel(pf);
    if x && use_.texel_buffer {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    let mut choice = held_storage_choice(pf, use_, profile, aliased)?;
    choice.x_channel = x;
    Ok(choice)
}

/// [`storage_choice`] of the format's held texels, without its RGBX conversion.
fn held_storage_choice(
    pf: PixelFormat,
    use_: StorageUse,
    profile: &DeviceProfile,
    aliased: bool,
) -> Result<StorageImageChoice, Error> {
    use vk::Format as F;
    let (aspects, candidates) = candidates(pf)?;
    if aspects != vk::ImageAspectFlags::COLOR {
        // Metal has no shader writes to depth or stencil textures.
        return Err(Unsupported::not_representable(Feature::Format).into());
    }
    // A YCbCr texture's image holds converted values, not the planes' bytes a write would store
    // into: Metal drops a write-only slot's writes to a 4:2:0 or 4:2:2 format
    // ([`writes_dropped`]), and a write-only slot of a format whose writes Metal stores into both
    // planes ([`ycbcr_writes_stored`]) writes the texture's write shadow (`bind::plan`); the other
    // 4:4:4 formats' writes, and a slot that reads and writes any of them, are not modelled.
    if candidates[0].emulation == FormatEmulation::YCbCr {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    // The first candidate is direct for every format but `ARGB8Unorm`'s, whose bytes the shader
    // rotates (a storage image ignores the view's component mapping), and an extended-range
    // format's, whose raw bytes its variant reads and writes.
    let image = candidates[0].format;
    let usage = use_.usage();
    let storage = |f: vk::Format| profile.supports_format(f, usage);
    // Declared without a format by preference where the device's features grant it, by its SPIR-V
    // image format where it has one, else without a format where the format's own bits allow it.
    let typed = |f: vk::Format| spirv_image_format(f).is_some();
    let formatless = |f: vk::Format| {
        accessed_without_format(f, use_, profile)
            || (!typed(f) && without_format_gaps(f, use_, profile).is_empty())
    };
    let declarable = |f: vk::Format| formatless(f) || typed(f);
    let rotated = candidates[0].emulation == FormatEmulation::RotatedView;
    let alias = match pf {
        PixelFormat::BGRA8Unorm => Some((F::R8G8B8A8_UNORM, StorageImageClass::SwizzleBgra)),
        // `A2R10G10B10`'s fields are `A2B10G10R10`'s with red and blue exchanged.
        PixelFormat::BGR10A2Unorm => {
            Some((F::A2B10G10R10_UNORM_PACK32, StorageImageClass::SwizzleBgra))
        }
        PixelFormat::ARGB8Unorm => Some((F::R8G8B8A8_UNORM, StorageImageClass::RotateArgb)),
        // `A8Unorm`'s byte in red, where the device holds no `A8_UNORM`: an image of its own. A
        // texture buffer's bytes take the view wherever `A8_UNORM` does not serve the use, whatever
        // the device's images hold (a texel buffer's view names its format alone).
        PixelFormat::A8Unorm if use_.texel_buffer || alpha_in_red(pf, profile) => {
            Some((F::R8_UNORM, StorageImageClass::AlphaInRed))
        }
        PixelFormat::RGB9E5Float => Some((F::R32_UINT, StorageImageClass::SharedExponentBits)),
        PixelFormat::R8UnormSRGB => Some((F::R8_UNORM, StorageImageClass::SrgbEncode)),
        PixelFormat::RG8UnormSRGB => Some((F::R8G8_UNORM, StorageImageClass::SrgbEncode)),
        PixelFormat::RGBA8UnormSRGB => Some((F::R8G8B8A8_UNORM, StorageImageClass::SrgbEncode)),
        PixelFormat::RGBX8UnormSRGB => Some((F::R8G8B8A8_UNORM, StorageImageClass::SrgbEncode)),
        PixelFormat::BGRA8UnormSRGB => Some((F::B8G8R8A8_UNORM, StorageImageClass::SrgbEncode)),
        _ => None,
    };
    // bgra8 sRGB takes both conversions through an rgba8 view where its own UNORM twin, which
    // has no SPIR-V image format, is not accessed without one.
    let second = match pf {
        PixelFormat::BGRA8UnormSRGB => Some((F::R8G8B8A8_UNORM, StorageImageClass::SrgbEncodeBgra)),
        _ => None,
    };
    // An RGBX format's write stores the green field's bits as X: green's sRGB code
    // for `RGBX8Unorm_sRGB`. A device storing into the sRGB image itself encodes red, green and
    // blue and stores the fourth lane linearly, as alpha, so that format's writes take the UNORM
    // alias, whose encode the shader applies before it moves green into the fourth lane.
    let srgb_x =
        x_channel(pf) && use_.write && matches!(alias, Some((_, StorageImageClass::SrgbEncode)));
    // A half-float texture the function only writes, on a device not measured to store a written
    // half's NaN as its bits: viewed through its unsigned integer alias, through which the shader
    // writes the bits Metal stores (`StorageImageClass::HalfBits`).
    if let Some(view) = half_bits_view(pf, use_, profile).filter(|_| aliased) {
        let mut choice = StorageImageChoice::new(view, StorageImageClass::HalfBits);
        choice.alias = Some(AliasView::new(image));
        return Ok(choice);
    }
    // Vulkan's conversion to a shared-exponent texel rounds each mantissa half up, where Metal
    // truncates it: an `RGB9E5Float` write takes its word alias, whose encode is Metal's, even on a
    // device that stores the format.
    let word = matches!(alias, Some((_, StorageImageClass::SharedExponentBits)));
    if storage(image) && declarable(image) && !rotated && !srgb_x && !word {
        let class = half_storage_class(pf, use_, profile);
        if !formatless(image) {
            return Ok(StorageImageChoice::new(image, class));
        }
        // The declaration names no format; the choice keeps the format's numeric type for a
        // function that depends on it.
        let mut choice = StorageImageChoice::new(F::UNDEFINED, class);
        choice.numeric = crate::bind::neutral::numeric_type(image);
        return Ok(choice);
    }
    for (view, class) in alias.into_iter().chain(second) {
        if storage(view) && declarable(view) {
            let mut choice = StorageImageChoice::new(view, class);
            // `ARGB8Unorm`'s image is the view's format: its choice rotates without an alias. So is
            // the image of an `A8Unorm` texture held in red, whose `R8_UNORM` views no `A8_UNORM`
            // image (a class of its own).
            let own = view == image || class == StorageImageClass::AlphaInRed;
            choice.alias = (!own).then(|| AliasView::new(image));
            return Ok(choice);
        }
    }
    // What each route lacks: the direct format, then the alias view.
    let lacks = |f: vk::Format| {
        let mut missing: Vec<Capability> = exposure_gap(f, profile).into_iter().collect();
        if !storage(f) {
            missing.push(Capability::Format { format: f, usage });
        }
        if !declarable(f) {
            missing.extend(without_format_gaps(f, use_, profile));
        }
        missing
    };
    let mut alternatives = vec![lacks(image)];
    alternatives.extend(alias.into_iter().chain(second).map(|(view, _)| lacks(view)));
    Err(Unsupported::needs_any_of(Feature::Format, alternatives).into())
}

/// The unsigned integer alias through which a function that writes a half-float texture of `pf`
/// (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`: every format of 16-bit float
/// channels, `metal::texel_format::half_channels`), and may read it, stores the bits Metal stores
/// (`StorageImageClass::HalfBits`), on a device not measured to store a written `half`'s NaN as
/// its bits (`DriverDeviation::MetalHalfTexelWrites`; there the format's own view serves, as
/// before): `R16_UINT`, `R16G16_UINT` or `R16G16B16A16_UINT`, where the device stores it for
/// `use_` and the shader can declare it by its SPIR-V image format. `None` elsewhere, where the
/// format's own choice stands. `RGBX16Float`'s image is `RGBA16Float`'s: its RGBX conversion moves
/// green's bits into the fourth lane after the class's (`StorageImageChoice:: x_channel`), and its
/// read takes alpha 1.
fn half_bits_view(
    pf: PixelFormat,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> Option<vk::Format> {
    use vk::Format as F;
    let view = match pf {
        PixelFormat::R16Float => F::R16_UINT,
        PixelFormat::RG16Float => F::R16G16_UINT,
        PixelFormat::RGBA16Float | PixelFormat::RGBX16Float => F::R16G16B16A16_UINT,
        _ => return None,
    };
    if !use_.write || profile.takes_driver_deviation(DriverDeviation::MetalHalfTexelWrites) {
        return None;
    }
    let declared = spirv_image_format(view)?;
    let declarable = !crate::spirv::requirements::extended_format(declared)
        || profile.fields().images.storage_extended_formats;
    (declarable && profile.supports_format(view, use_.usage())).then_some(view)
}

/// The unsigned integer alias through which a function reads a 16-bit float texture's texels
/// without a device decode seeing them (`DevicePolicy::half_texel_bits`,
/// [`HalfTexelBitsPolicy::Exact`](crate::device::HalfTexelBitsPolicy::Exact)): `R16_UINT`,
/// `R16G16_UINT` or `R16G16B16A16_UINT` for `R16Float`, `RG16Float`, and `RGBA16Float` or
/// `RGBX16Float`, where `profile`'s device samples the alias, so the slot's second binding can
/// view the texture in it. `None` for every other format, and where the device does not sample the
/// alias. `RGBX16Float`'s alias is `RGBA16Float`'s: its read takes alpha 1 after the widening, as
/// its own view's does.
///
/// Metal's `texture2d<half>` never widens a 16-bit float texel, so the texel's NaN comes back as
/// its own code; a device whose decode into `f32` canonicalizes a NaN has destroyed the code
/// before the shader sees it (MoltenVK on an Apple GPU, measured; ANV on Intel's GPUs, from the
/// conformance suite's replay). Reading the alias hands the shader the 16 bits themselves, which
/// it widens exactly ([`lower::arith::narrow::from_bits`](crate::lower) is the translation's).
pub fn half_bits_read_view(pf: PixelFormat, profile: &DeviceProfile) -> Option<vk::Format> {
    use vk::Format as F;
    let view = match pf {
        PixelFormat::R16Float => F::R16_UINT,
        PixelFormat::RG16Float => F::R16G16_UINT,
        PixelFormat::RGBA16Float | PixelFormat::RGBX16Float => F::R16G16B16A16_UINT,
        _ => return None,
    };
    profile
        .supports_format(view, FormatUsage::SampledImage)
        .then_some(view)
}

/// The unsigned integer alias through which a heap's twin (`StorageFormatClass::HalfBits`, a
/// formatless declaration) holds a written half-float texture of `pf` on `profile`:
/// [`half_bits_view`]'s for a write, where the device writes it without a format. `None` for
/// every other format, and on a device of `DriverDeviation::MetalHalfTexelWrites`.
pub(crate) fn heap_half_bits_view(pf: PixelFormat, profile: &DeviceProfile) -> Option<vk::Format> {
    let use_ = StorageUse::texture(false, true);
    let view = half_bits_view(pf, use_, profile)?;
    without_format_gaps(view, use_, profile)
        .is_empty()
        .then_some(view)
}

/// Whether a function's `half` and `float` writes through a heap handle store a half-float
/// member's bits through the heap's unsigned integer twin on `profile`
/// (`StorageFormatClass::HalfBits`): every format of
/// `metal::texel_format::HALF_BITS_FORMATS` has its alias there ([`heap_half_bits_view`]), so
/// the twin's indices (`heap_half_bits_indices`) name only members it holds.
pub(crate) fn heap_half_bits(profile: &DeviceProfile) -> bool {
    crate::metal::texel_format::HALF_BITS_FORMATS
        .iter()
        .all(|&pf| heap_half_bits_view(pf, profile).is_some())
}

/// The formats a storage binding views a texture of `pf` as on `profile`, for any use: the
/// read-and-write choice's ([`storage_image_choice`]: its alias view's format, else the direct
/// format) and a written-only one's (a half-float texture's unsigned integer alias on a device
/// not measured to store a written `half`'s NaN as its bits). A texture's image lists each among
/// its view formats. Empty where the device stores no texture of `pf`.
pub fn storage_view_formats(pf: PixelFormat, profile: &DeviceProfile) -> Vec<vk::Format> {
    let mut formats = Vec::new();
    for use_ in [
        StorageUse::READ_WRITE_TEXTURE,
        StorageUse::texture(false, true),
    ] {
        let Ok(choice) = storage_choice(pf, use_, profile) else {
            continue;
        };
        let format = match (choice.alias, choice.class) {
            (Some(_), _) | (None, StorageImageClass::AlphaInRed) => Some(choice.vk_format),
            (None, _) => direct_format(pf),
        };
        if let Some(format) = format {
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
    }
    formats
}

/// The RGBA-order twin of a BGRA-order direct format, whose fields are the format's with red and
/// blue exchanged: `R8G8B8A8_UNORM` for `B8G8R8A8_UNORM` and `A2B10G10R10_UNORM_PACK32` for
/// `A2R10G10B10_UNORM_PACK32`, each of the format's texel size and compatibility class (the
/// kernels' `SwizzleBgra` alias views, [`storage_choice`]). SPIR-V names no image format for
/// either BGRA-order format. `None` for every other format.
pub(crate) fn rgba_twin(format: vk::Format) -> Option<vk::Format> {
    match format {
        vk::Format::B8G8R8A8_UNORM => Some(vk::Format::R8G8B8A8_UNORM),
        vk::Format::A2R10G10B10_UNORM_PACK32 => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        _ => None,
    }
}

/// What a storage view in `twin`, an RGBA-order twin ([`rgba_twin`]), lacks on `profile` for a
/// function that reads and writes it: the format's storage usage, and a declaration, without a
/// format where the device grants the format that access without one, else by its SPIR-V image
/// format (`shaderStorageImageExtendedFormats` for an extended one). Empty where it lacks
/// nothing.
pub(crate) fn rgba_twin_gaps(twin: vk::Format, profile: &DeviceProfile) -> Vec<Capability> {
    let use_ = StorageUse::READ_WRITE_TEXTURE;
    let mut gaps = Vec::new();
    if !profile.supports_format(twin, FormatUsage::StorageImage) {
        gaps.push(Capability::Format {
            format: twin,
            usage: FormatUsage::StorageImage,
        });
    }
    if !accessed_without_format(twin, use_, profile) {
        match spirv_image_format(twin) {
            Some(declared)
                if crate::spirv::requirements::extended_format(declared)
                    && !profile.fields().images.storage_extended_formats =>
            {
                gaps.push(Capability::Feature(
                    VkFeature::ShaderStorageImageExtendedFormats,
                ));
            }
            Some(_) => {}
            None => gaps.extend(without_format_gaps(twin, use_, profile)),
        }
    }
    gaps
}

/// The format other than `format` a fetching draw's storage view of a color attachment of
/// `format` takes in the storage form of framebuffer fetch on `profile`
/// (`StorageAttachmentUse::view_alias_of`, its public door): an 8-bit sRGB format's
/// unsigned-integer format of its texel size on every device, and a BGRA-order format's
/// RGBA-order twin ([`rgba_twin`]) where the device neither reads nor writes the format without a
/// format ([`accessed_without_format`]; SPIR-V names no image format for it, so no declaration
/// reaches it otherwise) and the twin lacks nothing ([`rgba_twin_gaps`]). `None` elsewhere: the
/// view is in `format`, and a device that accesses a BGRA-order format without one keeps that.
pub(crate) fn fetch_view_alias(format: vk::Format, profile: &DeviceProfile) -> Option<vk::Format> {
    use vk::Format as F;
    match format {
        F::B8G8R8A8_SRGB | F::R8G8B8A8_SRGB => Some(F::R32_UINT),
        F::R8G8_SRGB => Some(F::R16_UINT),
        F::R8_SRGB => Some(F::R8_UINT),
        _ => rgba_twin(format).filter(|&twin| {
            !accessed_without_format(format, StorageUse::READ_WRITE_TEXTURE, profile)
                && rgba_twin_gaps(twin, profile).is_empty()
        }),
    }
}

/// The class of a storage choice that views `pf`'s own format for `use_`: for a half-float
/// format (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`) the function writes, by what the device is
/// measured to store for a written `float` (Metal rounds each channel toward zero and stores
/// every NaN as `0x7e00`): `Identity` where it rounds and stores NaNs as Metal
/// does (`DriverDeviation::MetalHalfStorageWrites` and `MetalHalfStorageNans`), `HalfQuietNan`
/// where it rounds alone, else `HalfTowardZero`. An `RG11B10Float` texture the function writes
/// takes `SmallFloatQuietNan` on every device, and one of a 1-bit alpha ([`one_bit_alpha`])
/// `OneBitAlpha` on a device of `DriverDeviation::MetalNormStorageWrites` (elsewhere the shader
/// rounds it by its class). Otherwise `Identity`.
pub(crate) fn half_storage_class(
    pf: PixelFormat,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> StorageImageClass {
    let half = matches!(
        pf,
        PixelFormat::R16Float
            | PixelFormat::RG16Float
            | PixelFormat::RGBA16Float
            | PixelFormat::RGBX16Float
    );
    match (half, pf == PixelFormat::RG11B10Float, use_.write) {
        (true, _, true) => half_write_class(profile),
        (_, true, true) => StorageImageClass::SmallFloatQuietNan,
        (_, _, true)
            if one_bit_alpha(pf)
                && profile.takes_driver_deviation(DriverDeviation::MetalNormStorageWrites) =>
        {
            StorageImageClass::OneBitAlpha
        }
        _ => StorageImageClass::Identity,
    }
}

/// Whether `pf`'s alpha is a 1-bit unorm channel (`crate::metal::texel_format::PackedTexels::Rgb5A1`:
/// `A1BGR5Unorm`, `BGR5A1Unorm`), whose `float` write Metal stores as 1 only above one half, where
/// a device measured to store every other normalized channel as Metal does rounds the half up
/// (RADV).
pub(crate) fn one_bit_alpha(pf: PixelFormat) -> bool {
    crate::metal::texel_format::PackedTexels::of(pf)
        == Some(crate::metal::texel_format::PackedTexels::Rgb5A1)
}

/// Whether `profile` stores a texture (or texture buffer) of a format with a 1-bit alpha
/// ([`one_bit_alpha`]) for `use_`: a heap write that may reach one tests its handle's class.
pub(crate) fn one_bit_storage(use_: StorageUse, profile: &DeviceProfile) -> bool {
    PixelFormat::ALL
        .iter()
        .filter(|&&pf| one_bit_alpha(pf))
        .any(|&pf| storage_choice(pf, use_, profile).is_ok())
}

/// The class a `float` write to a half-float storage image takes on `profile`
/// ([`half_storage_class`]).
pub(crate) fn half_write_class(profile: &DeviceProfile) -> StorageImageClass {
    let rounds = profile.takes_driver_deviation(DriverDeviation::MetalHalfStorageWrites);
    let nans = profile.takes_driver_deviation(DriverDeviation::MetalHalfStorageNans);
    match (rounds, nans) {
        (true, true) => StorageImageClass::Identity,
        (true, false) => StorageImageClass::HalfQuietNan,
        (false, _) => StorageImageClass::HalfTowardZero,
    }
}

/// The SPIR-V `Image Format` a storage image of Vulkan format `format` is declared with (Vulkan
/// spec "Compatibility Between SPIR-V Image Formats and Vulkan Formats"); `None` for formats only
/// a formatless (`Unknown`) declaration can access. Crate-private: only the lowering and the
/// planner read it, and its result is the grammar module's (`spv`) type.
pub(crate) fn spirv_image_format(format: vk::Format) -> Option<crate::spv::ImageFormat> {
    use crate::spv::ImageFormat as I;
    use vk::Format as F;
    Some(match format {
        F::R32G32B32A32_SFLOAT => I::Rgba32f,
        F::R16G16B16A16_SFLOAT => I::Rgba16f,
        F::R32_SFLOAT => I::R32f,
        F::R8G8B8A8_UNORM => I::Rgba8,
        F::R8G8B8A8_SNORM => I::Rgba8Snorm,
        F::R32G32_SFLOAT => I::Rg32f,
        F::R16G16_SFLOAT => I::Rg16f,
        F::B10G11R11_UFLOAT_PACK32 => I::R11fG11fB10f,
        F::R16_SFLOAT => I::R16f,
        F::R16G16B16A16_UNORM => I::Rgba16,
        F::A2B10G10R10_UNORM_PACK32 => I::Rgb10A2,
        F::R16G16_UNORM => I::Rg16,
        F::R8G8_UNORM => I::Rg8,
        F::R16_UNORM => I::R16,
        F::R8_UNORM => I::R8,
        F::R16G16B16A16_SNORM => I::Rgba16Snorm,
        F::R16G16_SNORM => I::Rg16Snorm,
        F::R8G8_SNORM => I::Rg8Snorm,
        F::R16_SNORM => I::R16Snorm,
        F::R8_SNORM => I::R8Snorm,
        F::R32G32B32A32_SINT => I::Rgba32i,
        F::R16G16B16A16_SINT => I::Rgba16i,
        F::R8G8B8A8_SINT => I::Rgba8i,
        F::R32_SINT => I::R32i,
        F::R32G32_SINT => I::Rg32i,
        F::R16G16_SINT => I::Rg16i,
        F::R8G8_SINT => I::Rg8i,
        F::R16_SINT => I::R16i,
        F::R8_SINT => I::R8i,
        F::R32G32B32A32_UINT => I::Rgba32ui,
        F::R16G16B16A16_UINT => I::Rgba16ui,
        F::R8G8B8A8_UINT => I::Rgba8ui,
        F::R32_UINT => I::R32ui,
        F::A2B10G10R10_UINT_PACK32 => I::Rgb10a2ui,
        F::R32G32_UINT => I::Rg32ui,
        F::R16G16_UINT => I::Rg16ui,
        F::R8G8_UINT => I::Rg8ui,
        F::R16_UINT => I::R16ui,
        F::R8_UINT => I::R8ui,
        F::R64_UINT => I::R64ui,
        F::R64_SINT => I::R64i,
        _ => return None,
    })
}

/// Every Vulkan format a Metal pixel format (every candidate of every format) or vertex format
/// can map to, and `R64_UINT`, the view of a `ulong` texture that a `ulong` atomic updates, sorted
/// and without duplicates: `FormatSupport::queries()` (some are defined by an extension or a later
/// core version; `FormatSupport::queries_for` filters them). Consumers call
/// `FormatSupport::queries`; crate-private.
pub(crate) fn queried_formats() -> Vec<vk::Format> {
    let set: BTreeSet<vk::Format> = PixelFormat::ALL
        .iter()
        .filter_map(|&pf| match repr(pf) {
            Repr::Candidates(_, c) => Some(c),
            _ => None,
        })
        .flatten()
        .map(|c| c.format)
        .chain(crate::metal::vertex::vertex_formats_queried())
        .chain([vk::Format::R64_UINT])
        .collect();
    set.into_iter().collect()
}

/// The Vulkan formats of `pf`'s candidates, none for a format without one.
#[cfg(test)]
pub(crate) fn candidate_formats(pf: PixelFormat) -> Vec<vk::Format> {
    match repr(pf) {
        Repr::Candidates(_, c) => c.iter().map(|c| c.format).collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
pub(crate) mod tests;
