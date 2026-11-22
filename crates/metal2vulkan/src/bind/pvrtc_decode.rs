//! A PVRTC texture's texels as Metal decodes them, from its blocks: the texels a decoded twin of
//! the texture holds on a device without the format (every device but Apple's and PowerVR's).
//!
//! **The decode** (PVRTC1, fitted on an Apple M3 Max under macOS 27.0.1, our own probe: every
//! texel of a 256 x 256 texture of seeded random blocks of each of the eight formats read as
//! `float`, equal to this decode's in every channel). A block is a 64-bit little-endian word of
//! 4 x 4 texels (4 bits a pixel) or 8 x 4 (2 bits); the blocks lie row-major and wrap at the
//! level's edges, a level of fewer texels than a block holding one.
//!
//! - **Colors.** Color B is bits 48 to 63: RGB555 where its bit 15 is set, else A3R4G4B4; color A
//!   is bits 32 to 47, its bit 0 the block's mode bit: RGB554 (blue's 4 bits at bits 1 to 4) where
//!   its bit 15 is set, else A3R4G4B3. Each widens to 5-bit channels (a 4-bit one `c << 1 | c >> 3`,
//!   a 3-bit one `c << 2 | c >> 1`) and a 4-bit alpha (`a << 1`, an opaque color's 15).
//! - **Upscale.** Each texel's colors A and B are the bilinear upscale of the four blocks whose
//!   centres (`(w/2, h/2)` in a block of `w` x `h`) surround it: the integer sum `V` of each
//!   channel times the weights, at a scale of `w h`, widened to 8 bits: 4 bits a pixel, RGB
//!   `(V >> 1) + (V >> 6)` and alpha `V + (V >> 4)`; 2 bits, RGB `(V >> 2) + (V >> 7)` and alpha
//!   `(V >> 1) + (V >> 5)`.
//! - **4 bits a pixel.** Each texel's 2-bit index (bits 0 to 31, 2 a texel, row-major) weighs B:
//!   0, 3, 5 or 8 eighths; in mode 1 (punch-through) 0, 4, 4 or 8, index 2 also making alpha 0.
//! - **2 bits a pixel.** In mode 0 each texel has one bit (bits 0 to 31): 0 or 8 eighths. In mode
//!   1 the texels of even `x + y` hold 2-bit values (0, 3, 5 or 8 eighths) on the checkerboard,
//!   row-major; the others take the four neighbours' average `(s + 2) / 4`, or where bit 0 is set
//!   the two horizontal neighbours' `(a + b + 1) / 2` (the vertical ones' where bit 20 is set).
//!   Before the values are read, bit 21 is copied into bit 20 where bit 0 is set, then bit 1 into
//!   bit 0. Neighbours come from the adjacent blocks, wrapping.
//! - **The texel.** Each channel is `(A (8 - w) + B w) / 8`, an 8-bit code; an RGB format's alpha
//!   is 255. Metal reads a code `c / 255`, an sRGB format's color channel through its twelve-bit
//!   table (`k / 4095` for `lower::arith::srgb`'s step `k`).
//!
//! **The twin's texels** are the codes, in `R8G8B8A8_UNORM`,
//! exact: a linear format's read is the code's `c / 255`, and an sRGB format's slot reads them
//! through the twin's `R8G8B8A8_SRGB` view, the device's decode (`PrecisionNote::SrgbTexelDecode`),
//! as for `RGBA8Unorm_sRGB`. A format's linear and sRGB forms decode to the same codes, so one twin
//! serves the views of both.

use crate::lower::arith::srgb::SRGB_DECODE_STEPS;
use crate::metal::PixelFormat;

/// How a PVRTC format's blocks decode (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PvrtcDecode {
    /// 2 bits a pixel (8 x 4 blocks), else 4 (4 x 4 blocks).
    pub two_bits: bool,
    /// An RGBA format, whose alpha is decoded; an RGB format's is 255.
    pub alpha: bool,
    /// An sRGB format, whose color channels Metal reads through its twelve-bit table.
    pub srgb: bool,
}

impl PvrtcDecode {
    /// The decode of `pf`'s blocks: the eight PVRTC formats; `None` for every other format.
    pub fn of(pf: PixelFormat) -> Option<PvrtcDecode> {
        use PixelFormat as P;
        let (two_bits, alpha, srgb) = match pf {
            P::PVRTCRGB2BPP => (true, false, false),
            P::PVRTCRGB2BPPSRGB => (true, false, true),
            P::PVRTCRGB4BPP => (false, false, false),
            P::PVRTCRGB4BPPSRGB => (false, false, true),
            P::PVRTCRGBA2BPP => (true, true, false),
            P::PVRTCRGBA2BPPSRGB => (true, true, true),
            P::PVRTCRGBA4BPP => (false, true, false),
            P::PVRTCRGBA4BPPSRGB => (false, true, true),
            _ => return None,
        };
        Some(PvrtcDecode {
            two_bits,
            alpha,
            srgb,
        })
    }

    /// The `float` Metal reads of channel `ch` (red, green, blue, alpha) of code `c`.
    pub fn metal_value(self, ch: usize, c: u8) -> f32 {
        match self.srgb && ch < 3 {
            true => (f64::from(SRGB_DECODE_STEPS[usize::from(c)]) / 4095.0) as f32,
            false => (f64::from(c) / 255.0) as f32,
        }
    }
}

#[cfg(test)]
mod tests;
