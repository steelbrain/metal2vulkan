//! Color attachments a device renders through an unsigned integer view of the texture's words.
//!
//! A device may hold a format's image, and sample it, but not render into it: the RTX 5080 has
//! `R5G5B5A1_UNORM_PACK16` and `R4G4B4A4_UNORM_PACK16` images without `COLOR_ATTACHMENT`, and
//! `R8_SRGB` and `E5B9G9R9_UFLOAT_PACK32` without it (`R8G8_SRGB` it holds no image of: the runtime
//! holds such a texture in its UNORM twin). Such a texture's image is made mutable, with extended
//! usage, and its attachment is a view of the same bits as the unsigned integer format of its word
//! ([`WordAttachment::word_format`]), which every Vulkan device renders. The fragment function
//! writes its output's word (`LoweringInputs::word_colors`): the word Metal stores, but an 8-bit
//! sRGB format's codes, which are the device's encode (below); and a clear writes the word of
//! Metal's clear (the runtime's clear values). The texture's other views read the same bits.
//!
//! Metal's store into a packed 16-bit format (measured on an Apple M3 Max under macOS 27.0.1:
//! every texel of 65,536 `float` and `half` outputs and 4,096 clear colors into each of
//! `B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm` and `BGR5A1Unorm`): each channel's value rounded to
//! `half` to nearest even (a NaN 0), clamped to `[0, 1]`, times the channel's maximum code, rounded
//! to nearest even; the only ties are at one half, which store the even code (an `A1BGR5Unorm`
//! alpha of 0.5 is 0). That is `ChannelWrite::HalfUnorm`'s rule. `BGRA4Unorm`'s four fields take
//! it too, measured the same way on the same device, its red at bit 8, green 4, blue 0 and alpha
//! 12, read back off Metal's own words.
//!
//! Metal's store into an 8-bit sRGB format (`R8Unorm_sRGB`, `RG8Unorm_sRGB`; the lane's rule,
//! `runtime_lane::color_writes`, and its in-guest probe's clears and draws on the M3 Max): each
//! channel's value clamped to `[0, 1]` (a NaN 0), times 4095 exactly, rounded to nearest, encoded
//! by Metal's twelve-bit table, as `pack_float_to_srgb_unorm4x8` encodes a color channel. The
//! fragment function encodes the channel by the MSL transfer through the device's `Pow`, as its
//! `pack_float_to_srgb_unorm4x8` does (`lower::arith::pack`, `PrecisionNote::DevicePackConversion`). A write
//! of some of its channels merges the word into the texel's: in the pixel interlock
//! (`color_writes::stored_word`), or over the attachment's input attachment on a device that orders
//! fetch through rasterization-order input attachments (`color_writes::merged_word`). A blend of
//! such an attachment, other than one and zero, is not built.
//!
//! Metal's store of a `float` output into `RGB9E5Float` (measured on the M3 Max: 65,536 draws and
//! clears, and the format scout's draws): each of red, green and blue clamped to `[0, 65408]` (a
//! NaN 0), the shared exponent from the largest, and every mantissa truncated (the rule of its
//! kernel writes). Its store of a `half` output rounds the mantissas half up, raising the exponent
//! where the largest reaches 512, from the `float` the `half` was converted from (Metal's compiler
//! folds the conversion into the store): that store is not built.
//!
//! A blend of factors one and zero, added, stores the same (Metal's draws through it matched too).
//! A write mask of some of the format's channels keeps the other fields: where the device orders
//! fetch through the pixel interlock, the fragment function stores the word in the interlocked
//! storage form, merging the kept fields of the word it reads first (`lower::fragment::fetch`),
//! which hold their codes as Metal's tile holds them. The device blends no such view otherwise,
//! resolves none, and reads none as the format's input attachment: a pipeline that blends a word
//! attachment, covers by its alpha, or writes some of its channels on a device without that
//! interlock is refused as the device lacking the format's attachment, as without the view.
//!
//! A fragment function's color input of a packed 16-bit word attachment reads the word and
//! decodes it, in the interlocked storage form (`lower::fragment::fetch`): each channel's code
//! over its maximum code rounded to `half`, nearest even, which is the value Metal's programmable
//! blending hands the function (measured on an Apple M3 Max under macOS 27.0.1: a `float4` and a
//! `half4` color input of `BGRA4Unorm`, `ABGR4Unorm` and `B5G6R5Unorm`, every code of each 4-bit
//! channel and a sweep of the 5- and 6-bit ones, stored raw into an `RGBA32Float` attachment; an
//! `RGBA8Unorm` control reads the `f32` quotient through a `float4` input, so the two rules are
//! told apart). Metal's tile holds a pass's clear and an earlier draw's write finer than the
//! codes, as it holds them for a blend (`PrecisionNote::DeviceColorConversion`), so a read after
//! one in the same pass reads the code the attachment holds. A color input of an 8-bit sRGB word
//! and of an `RGB9E5Float` one, and a pass's resolve of any word attachment, are
//! `NotImplemented(Format)`.

use crate::device::requirements::FormatUsage;
use crate::device::DeviceProfile;
use crate::metal::PixelFormat;
use crate::vk;

/// A color format whose attachment a device renders through an unsigned integer view of its words
/// (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WordAttachment {
    /// A packed 16-bit unsigned normalized format (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`,
    /// `BGR5A1Unorm`, `BGRA4Unorm`), held as `R16_UINT`: each channel's field in the
    /// little-endian word, red, green, blue and alpha, as its lowest bit and its width; `None`
    /// for a channel the format lacks.
    Packed16 {
        /// The fields of red, green, blue and alpha.
        fields: [Option<WordField>; 4],
    },
    /// An 8-bit sRGB format of one or two channels (`R8Unorm_sRGB`, `RG8Unorm_sRGB`), held as
    /// `R8_UINT` or `R16_UINT`: each channel's byte in the little-endian word, red then green,
    /// Metal's twelve-bit encode of the channel's value.
    Srgb8 {
        /// The fields of red, green, blue and alpha.
        fields: [Option<WordField>; 4],
    },
    /// `RGB9E5Float`, held as `R32_UINT`: red's, green's and blue's nine-bit mantissas from bit
    /// 0, under the exponent in bits 27 to 31. Its fields are the mantissas, which no write mask
    /// keeps apart: a write of some of its channels is not built.
    Rgb9e5,
}

/// One channel's bits in a [`WordAttachment`]'s word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WordField {
    /// The field's lowest bit.
    pub shift: u8,
    /// The field's width in bits.
    pub bits: u8,
}

impl WordAttachment {
    /// The word layout of `pf`, whatever the device: `Some` for the formats a word view can
    /// render.
    pub fn layout(pf: PixelFormat) -> Option<WordAttachment> {
        let f = |shift: u8, bits: u8| Some(WordField { shift, bits });
        // Metal names a packed format from its least significant bit.
        let fields = match pf {
            PixelFormat::B5G6R5Unorm => [f(11, 5), f(5, 6), f(0, 5), None],
            PixelFormat::A1BGR5Unorm => [f(11, 5), f(6, 5), f(1, 5), f(0, 1)],
            PixelFormat::ABGR4Unorm => [f(12, 4), f(8, 4), f(4, 4), f(0, 4)],
            PixelFormat::BGRA4Unorm => [f(8, 4), f(4, 4), f(0, 4), f(12, 4)],
            PixelFormat::BGR5A1Unorm => [f(10, 5), f(5, 5), f(0, 5), f(15, 1)],
            PixelFormat::R8UnormSRGB => {
                let fields = [f(0, 8), None, None, None];
                return Some(WordAttachment::Srgb8 { fields });
            }
            PixelFormat::RG8UnormSRGB => {
                let fields = [f(0, 8), f(8, 8), None, None];
                return Some(WordAttachment::Srgb8 { fields });
            }
            PixelFormat::RGB9E5Float => return Some(WordAttachment::Rgb9e5),
            _ => return None,
        };
        Some(WordAttachment::Packed16 { fields })
    }

    /// How profile `p`'s device renders an attachment of `pf` through a word view: `Some` where
    /// `pf` has a word layout ([`layout`](Self::layout)), the device holds the texture (it samples
    /// `pf`'s own Vulkan format, or an 8-bit sRGB format's UNORM twin, in which the runtime holds
    /// it, `srgb_raw_twin`) but does not render `pf`'s format, and renders the word's format.
    /// `None` where it renders `pf` itself, or does not hold it.
    pub fn of(pf: PixelFormat, p: &DeviceProfile) -> Option<WordAttachment> {
        let word = WordAttachment::layout(pf)?;
        let direct = super::direct_format(pf)?;
        let twin = super::srgb_raw_twin(pf).and_then(super::direct_format);
        let held = [Some(direct), twin]
            .into_iter()
            .flatten()
            .any(|f| p.supports_format(f, FormatUsage::SampledImage));
        (held
            && !p.supports_format(direct, FormatUsage::ColorAttachment)
            && p.supports_format(word.word_format(), FormatUsage::ColorAttachment))
        .then_some(word)
    }

    /// The unsigned integer format of the word, the attachment view's.
    pub fn word_format(self) -> vk::Format {
        match self {
            WordAttachment::Packed16 { .. } => vk::Format::R16_UINT,
            WordAttachment::Srgb8 { fields } if fields[1].is_none() => vk::Format::R8_UINT,
            WordAttachment::Srgb8 { .. } => vk::Format::R16_UINT,
            WordAttachment::Rgb9e5 => vk::Format::R32_UINT,
        }
    }

    /// The channels' fields: red, green, blue and alpha (`RGB9E5Float`'s mantissas).
    pub fn fields(self) -> [Option<WordField>; 4] {
        let f = |shift: u8| Some(WordField { shift, bits: 9 });
        match self {
            WordAttachment::Packed16 { fields } | WordAttachment::Srgb8 { fields } => fields,
            WordAttachment::Rgb9e5 => [f(0), f(9), f(18), None],
        }
    }

    /// The word of the channels' codes `codes` (red, green, blue, alpha; a channel the format
    /// lacks ignored, each code masked to its field).
    pub fn pack(self, codes: [u32; 4]) -> u32 {
        self.fields()
            .iter()
            .zip(codes)
            .filter_map(|(f, c)| f.map(|f| (c & ((1u32 << f.bits) - 1)) << f.shift))
            .fold(0, |w, c| w | c)
    }

    /// The word Metal stores for a `float` (or `half`) output `rgba`, which the fragment
    /// function's encode stores too: a packed 16-bit format's channel by
    /// `ChannelWrite::HalfUnorm`'s rule, a channel the format lacks ignored; `RGB9E5Float`'s word
    /// of a `float` output (the module docs), which a clear stores as well. `None` for an 8-bit
    /// sRGB format, whose codes the fragment function takes from the device's encode (the module
    /// docs).
    pub fn output_word(self, rgba: [f32; 4]) -> Option<u32> {
        match self {
            WordAttachment::Rgb9e5 => Some(rgb9e5_word([rgba[0], rgba[1], rgba[2]])),
            WordAttachment::Srgb8 { .. } => None,
            WordAttachment::Packed16 { fields } => {
                let mut codes = [0u32; 4];
                for (code, (f, x)) in codes.iter_mut().zip(fields.iter().zip(rgba)) {
                    if let Some(f) = f {
                        *code = half_unorm_code(x, f.bits);
                    }
                }
                Some(self.pack(codes))
            }
        }
    }
}

/// The `RGB9E5Float` word Metal stores for `float` channels `rgb` (a draw's output, a clear's
/// color through `float`, a kernel's write): each clamped to `[0, 65408]` (a NaN, a negative value
/// and `-0` 0), the exponent `e = max(floor(log2 m), -16) + 16` of the largest clamped channel
/// `m`, and each mantissa the clamped channel times `2^(24 - e)` (exact) truncated, at most 511.
pub(crate) fn rgb9e5_word(rgb: [f32; 3]) -> u32 {
    let clamped = rgb.map(|x| match x.is_nan() || x.is_sign_negative() {
        true => 0.0,
        false => x.min(65408.0),
    });
    let largest = clamped.iter().copied().fold(0.0, f32::max);
    // The exponent field of the largest: `floor(log2 m)` + 127 for a normal `m`.
    let field = (largest.to_bits() >> 23).max(111);
    let e = field - 111;
    let scale = f64::from(2.0f32).powi(24 - e as i32);
    clamped
        .iter()
        .enumerate()
        .map(|(c, &x)| (((f64::from(x) * scale) as u32).min(511)) << (9 * c))
        .fold(e << 27, |w, m| w | m)
}

/// The code of `x` in a unorm channel of `bits` bits that Metal reads through a `half`: `x`
/// rounded to `half`, nearest even (a NaN 0), clamped to `[0, 1]`, times `2^bits - 1`, rounded to
/// nearest even (the product of a `half` and a code of at most 16 bits is exact in `f64`).
pub(crate) fn half_unorm_code(x: f32, bits: u8) -> u32 {
    let h = f64::from(crate::float16::f16_bits_to_f32(
        crate::float16::f32_to_f16_bits(x),
    ));
    let h = if h.is_nan() { 0.0 } else { h.clamp(0.0, 1.0) };
    let m = f64::from((1u32 << bits) - 1);
    (h * m).round_ties_even() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layouts_are_metals_least_significant_first() {
        let w = WordAttachment::layout(PixelFormat::A1BGR5Unorm).unwrap();
        // Alpha in bit 0, blue in 1..=5, green in 6..=10, red in 11..=15.
        assert_eq!(w.pack([31, 0, 0, 0]), 0xf800);
        assert_eq!(w.pack([0, 31, 0, 0]), 0x07c0);
        assert_eq!(w.pack([0, 0, 31, 0]), 0x003e);
        assert_eq!(w.pack([0, 0, 0, 1]), 0x0001);
        let w = WordAttachment::layout(PixelFormat::ABGR4Unorm).unwrap();
        assert_eq!(w.pack([1, 2, 3, 4]), 0x1234);
        let w = WordAttachment::layout(PixelFormat::B5G6R5Unorm).unwrap();
        assert_eq!(w.pack([31, 63, 31, 1]), 0xffff);
        let w = WordAttachment::layout(PixelFormat::BGR5A1Unorm).unwrap();
        assert_eq!(w.pack([0, 0, 0, 1]), 0x8000);
        assert_eq!(WordAttachment::layout(PixelFormat::RGBA8Unorm), None);
    }

    /// The probe's answers: just below a half the `half` rounds to the half, which the code's
    /// tie takes to even; a half itself is even; NaN and negatives are 0, past one the maximum.
    #[test]
    fn output_words_round_through_half_then_to_even() {
        let below_half = f32::from_bits(0.5f32.to_bits() - 1);
        assert_eq!(half_unorm_code(below_half, 5), 16);
        assert_eq!(half_unorm_code(below_half, 4), 8);
        assert_eq!(half_unorm_code(0.5, 6), 32);
        assert_eq!(half_unorm_code(0.5, 1), 0);
        assert_eq!(half_unorm_code(f32::NAN, 5), 0);
        assert_eq!(half_unorm_code(-1.0, 5), 0);
        assert_eq!(half_unorm_code(7.0, 4), 15);
        // `half` 0.09998 times 15 is 1.4997: an ABGR4 alpha of 0.1 is 1.
        assert_eq!(half_unorm_code(0.1, 4), 1);
        let w = WordAttachment::layout(PixelFormat::A1BGR5Unorm).unwrap();
        assert_eq!(w.output_word([1.0, 0.0, 1.0, 0.5]), Some(0xf83e));
    }

    /// An 8-bit sRGB word's codes are the device's encode: no reference word.
    #[test]
    fn srgb_words_have_no_reference_word() {
        let r8 = WordAttachment::layout(PixelFormat::R8UnormSRGB).unwrap();
        assert_eq!(r8.word_format(), vk::Format::R8_UINT);
        assert_eq!(r8.output_word([0.2, 0.4, 0.6, 0.8]), None);
        let rg8 = WordAttachment::layout(PixelFormat::RG8UnormSRGB).unwrap();
        assert_eq!(rg8.word_format(), vk::Format::R16_UINT);
        assert_eq!(rg8.output_word([0.2, 0.4, 0.6, 0.8]), None);
    }

    /// The format scout's draw on the M3 Max: (2.5, 0.375, -0.75) stores `0x88006140`; the
    /// probe's clear of (0.25, 0, 1.125) `0x84800040`; past 65408 is 65408, NaN and negatives 0,
    /// and a channel below the exponent's step truncates to 0.
    #[test]
    fn rgb9e5_words_truncate_under_the_largest_exponent() {
        let w = WordAttachment::layout(PixelFormat::RGB9E5Float).unwrap();
        assert_eq!(w.word_format(), vk::Format::R32_UINT);
        assert_eq!(w.output_word([2.5, 0.375, -0.75, 0.5]), Some(0x8800_6140));
        assert_eq!(rgb9e5_word([0.25, 0.0, 1.125]), 0x8480_0040);
        assert_eq!(rgb9e5_word([70000.0, f32::NAN, -0.0]), (31 << 27) | 511);
        assert_eq!(rgb9e5_word([0.0; 3]), 0);
        // 1 + 2^-9 under the exponent of 1 (a step of 2^-8) truncates to 1; 2^-10 is 0 there.
        assert_eq!(
            rgb9e5_word([1.0 + 2f32.powi(-9), 2f32.powi(-10), 0.0]),
            (16 << 27) | 256
        );
        // The smallest exponent: 2^-24 is mantissa 1 under exponent 0.
        assert_eq!(rgb9e5_word([2f32.powi(-24), 0.0, 0.0]), 1);
    }

    /// `BGRA4Unorm`'s fields, Metal's own words (the probe `packed16_word_fetch`'s controls read
    /// each format's fields out of single-channel texels): blue in bits 0 to 3, then green, red
    /// and alpha. Its word of a codes vector is `ABGR4Unorm`'s rotated, and never its own.
    #[test]
    fn bgra4_is_blue_then_green_then_red_then_alpha() {
        let w = WordAttachment::layout(PixelFormat::BGRA4Unorm).unwrap();
        assert_eq!(w.word_format(), vk::Format::R16_UINT);
        assert_eq!(w.pack([15, 0, 0, 0]), 0x0f00);
        assert_eq!(w.pack([0, 15, 0, 0]), 0x00f0);
        assert_eq!(w.pack([0, 0, 15, 0]), 0x000f);
        assert_eq!(w.pack([0, 0, 0, 15]), 0xf000);
        assert_eq!(w.pack([1, 2, 3, 4]), 0x4123);
        // The same codes under `ABGR4Unorm`'s order: a different word, the same channels.
        let abgr4 = WordAttachment::layout(PixelFormat::ABGR4Unorm).unwrap();
        assert_eq!(abgr4.pack([1, 2, 3, 4]), 0x1234);
        for codes in [[1, 2, 3, 4], [15, 0, 8, 1], [0, 9, 9, 14]] {
            assert_ne!(w.pack(codes), abgr4.pack(codes), "{codes:?}");
            for (lane, (f, g)) in w.fields().iter().zip(abgr4.fields()).enumerate() {
                let (f, g) = (f.unwrap(), g.unwrap());
                let code = |w: u32, f: WordField| (w >> f.shift) & ((1 << f.bits) - 1);
                assert_eq!(code(w.pack(codes), f), code(abgr4.pack(codes), g), "{lane}");
            }
        }
    }

    /// The value Metal's programmable blending hands a fragment function for each code of a
    /// 4-bit channel of `BGRA4Unorm`: the code over 15 rounded to `half`, nearest even, which is
    /// what the fetch's decode computes (`stage_io::word_color::decode`). These sixteen `f32`
    /// bit patterns are Metal's own, from the probe `packed16_word_fetch` on an Apple M3 Max
    /// under macOS 27.0.1: every code of every channel of `BGRA4Unorm` and `ABGR4Unorm`, through
    /// a `float4` and a `half4` color input, gave the same pattern per code, and twelve fresh
    /// processes agreed. The `f32` quotient, which an `RGBA8Unorm` control did read through a
    /// `float4` input, differs at fourteen of the sixteen.
    #[test]
    fn a_four_bit_channels_fetched_value_is_the_half_of_its_quotient() {
        const METAL: [u32; 16] = [
            0x0000_0000,
            0x3d88_8000,
            0x3e08_8000,
            0x3e4c_c000,
            0x3e88_8000,
            0x3eaa_a000,
            0x3ecc_c000,
            0x3eee_e000,
            0x3f08_8000,
            0x3f19_a000,
            0x3f2a_a000,
            0x3f3b_c000,
            0x3f4c_c000,
            0x3f5d_e000,
            0x3f6e_e000,
            0x3f80_0000,
        ];
        let mut unlike_the_quotient = 0;
        for (code, metal) in METAL.into_iter().enumerate() {
            let code = code as u32;
            let quotient = code as f32 / 15.0;
            let decoded =
                crate::float16::f16_bits_to_f32(crate::float16::f32_to_f16_bits(quotient));
            assert_eq!(decoded.to_bits(), metal, "code {code}");
            if quotient.to_bits() != metal {
                unlike_the_quotient += 1;
            }
            // And the value a function writes back stores the code it read.
            assert_eq!(half_unorm_code(decoded, 4), code, "code {code}");
        }
        // The two rules are told apart: the `f32` quotient is Metal's value at 0 and 1 alone.
        assert_eq!(unlike_the_quotient, 14);
    }

    #[test]
    fn a_word_view_serves_where_the_device_holds_but_cannot_render_the_format() {
        use crate::pipeline::testing::profile;
        let nvidia = profile(
            "nvidia-rtx5080-laptop",
            crate::device::DevicePolicy::default(),
        )
        .unwrap();
        let lavapipe = profile("lavapipe", crate::device::DevicePolicy::default()).unwrap();
        for pf in [
            PixelFormat::A1BGR5Unorm,
            PixelFormat::ABGR4Unorm,
            PixelFormat::BGRA4Unorm,
        ] {
            assert_eq!(
                WordAttachment::of(pf, &nvidia),
                WordAttachment::layout(pf),
                "{pf:?}"
            );
            assert_eq!(WordAttachment::of(pf, &lavapipe), None, "{pf:?}");
        }
        // The 5080 renders neither `R8_SRGB` nor `R8G8_SRGB` (it holds the latter in its twin).
        for pf in [PixelFormat::R8UnormSRGB, PixelFormat::RG8UnormSRGB] {
            assert_eq!(
                WordAttachment::of(pf, &nvidia),
                WordAttachment::layout(pf),
                "{pf:?}"
            );
        }
        // The 5080 samples `E5B9G9R9_UFLOAT_PACK32` and does not render it; RADV renders it.
        assert_eq!(
            WordAttachment::of(PixelFormat::RGB9E5Float, &nvidia),
            Some(WordAttachment::Rgb9e5)
        );
        let radv = profile("amd-rdna3-radv", crate::device::DevicePolicy::default()).unwrap();
        assert_eq!(WordAttachment::of(PixelFormat::RGB9E5Float, &radv), None);
        // Vulkan makes `R5G6B5` and `A1R5G5B5` attachments mandatory.
        for pf in [PixelFormat::B5G6R5Unorm, PixelFormat::BGR5A1Unorm] {
            assert_eq!(WordAttachment::of(pf, &nvidia), None, "{pf:?}");
        }
        assert_eq!(WordAttachment::of(PixelFormat::RGBA8Unorm, &nvidia), None);
    }
}
