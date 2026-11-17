//! The high word of a texture's `gpuResourceID`: what a shader reads of a texture it learns only
//! at run time, as a pure function of the Metal texture descriptor, so a guest computes it with
//! no round trip. `docs/RESOURCE-ID.md` is the specification; a sampler's word is
//! [`SamplerWord`](crate::metal::sampler::SamplerWord).

use crate::metal::{PixelFormat, TextureSwizzleChannels};

/// The high word of a texture's `gpuResourceID`, from its pixel format and swizzle alone. The
/// bits:
///
/// | bits | field |
/// |---|---|
/// | 0-11 | [`BORDER_TEXELS`](Self::BORDER_TEXELS): Metal's border texel for each border color |
/// | 12 | [`OUT_OF_BOUNDS_ALPHA_ONE`](Self::OUT_OF_BOUNDS_ALPHA_ONE): the format has no alpha |
/// | 13-15 | 0 |
/// | 16-31 | [`PIXEL_FORMAT`](Self::PIXEL_FORMAT): the `MTLPixelFormat` raw value |
///
/// Bit `3c + b` of the border texels is set where channel `c` (0 red to 3 alpha) of the texel a
/// Metal 4.1 read through a sampler hands back outside the level on a border axis, for border
/// color `b` (`MTLSamplerBorderColor`'s raw value: transparent black, opaque black, opaque
/// white), reads 1 rather than 0, after the swizzle. A color format's channels are its own; a
/// depth or stencil format reads as red, green and blue, alpha 1. An out-of-bounds `read` of a
/// format without alpha reads alpha 1 (MSL 4.1 sections 2.9.1 and 6.13). Every pixel format's
/// raw value is above 0, so a texture's word is never 0, and the nil `gpuResourceID`, 0, has the
/// word 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureWord(pub u32);

impl TextureWord {
    /// The border texels' bits.
    pub const BORDER_TEXELS: u32 = 0xfff;
    /// The format has no alpha: an out-of-bounds read reads alpha 1.
    pub const OUT_OF_BOUNDS_ALPHA_ONE: u32 = 1 << 12;
    /// The lowest bit of the pixel format.
    pub const PIXEL_FORMAT_SHIFT: u32 = 16;
    /// The pixel format's bits.
    pub const PIXEL_FORMAT: u32 = 0xffff << Self::PIXEL_FORMAT_SHIFT;

    /// The word of a texture of `pixel_format` made with `swizzle`. `None` for the placeholders
    /// (`Invalid`, `Unspecialized`) and the formats of which the translator holds no texture or
    /// which it only copies, which no shader reads.
    pub fn of(pixel_format: PixelFormat, swizzle: TextureSwizzleChannels) -> Option<TextureWord> {
        let border = crate::metal::sampler::border_texels(pixel_format, swizzle)?;
        let alpha_one = crate::metal::format::out_of_bounds_alpha_one(pixel_format)?;
        let mut w = border | (pixel_format.as_raw() << Self::PIXEL_FORMAT_SHIFT);
        if alpha_one {
            w |= Self::OUT_OF_BOUNDS_ALPHA_ONE;
        }
        Some(TextureWord(w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::TextureSwizzle;

    /// Each field lands in its bits: `RGBA8Unorm` (raw 70) has every channel, so white reads 1 in
    /// each and opaque black in alpha; `R8Unorm` (raw 10) has red alone, so its green and blue
    /// read 0 under every color, its alpha 1 under every color, and its out-of-bounds alpha is 1;
    /// a swizzle of red from one and green from zero sets and clears those channels' bits;
    /// `Invalid` has no word.
    #[test]
    fn each_field_lands_in_its_bits() {
        let rgba = TextureWord::of(PixelFormat::RGBA8Unorm, TextureSwizzleChannels::DEFAULT);
        assert_eq!(rgba, Some(TextureWord(70 << 16 | 0b110_100_100_100)));
        let r = TextureWord::of(PixelFormat::R8Unorm, TextureSwizzleChannels::DEFAULT);
        assert_eq!(r, Some(TextureWord(10 << 16 | 1 << 12 | 0b111_000_000_100)));
        let ones = TextureSwizzleChannels::new(
            TextureSwizzle::One,
            TextureSwizzle::Zero,
            TextureSwizzle::Blue,
            TextureSwizzle::Alpha,
        );
        let swizzled = TextureWord::of(PixelFormat::RGBA8Unorm, ones);
        assert_eq!(swizzled, Some(TextureWord(70 << 16 | 0b110_100_000_111)));
        assert_eq!(
            TextureWord::of(PixelFormat::Invalid, TextureSwizzleChannels::DEFAULT),
            None
        );
    }
}
