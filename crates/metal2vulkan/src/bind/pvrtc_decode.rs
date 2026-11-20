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
//! **The twin's texels** are the codes, in `R8G8B8A8_UNORM` ([`PvrtcDecode::texel_format`]),
//! exact: a linear format's read is the code's `c / 255`, and an sRGB format's slot reads them
//! through the twin's `R8G8B8A8_SRGB` view, the device's decode (`PrecisionNote::SrgbTexelDecode`),
//! as for `RGBA8Unorm_sRGB`. A format's linear and sRGB forms decode to the same codes, so one twin
//! serves the views of both.

use crate::lower::arith::srgb::SRGB_DECODE_STEPS;
use crate::metal::PixelFormat;
use crate::vk::Format;

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

/// The weights of B, in eighths, of a 2-bit index.
const WEIGHTS: [u32; 4] = [0, 3, 5, 8];

/// The weights of B of a 4-bit-a-pixel block's punch-through mode.
const PUNCH_WEIGHTS: [u32; 4] = [0, 4, 4, 8];

impl PvrtcDecode {
    /// The bytes of every block.
    pub const BLOCK_BYTES: u32 = 8;

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

    /// The texels across and down of a block: 8 x 4 or 4 x 4.
    pub fn block_extent(self) -> [u32; 2] {
        match self.two_bits {
            true => [8, 4],
            false => [4, 4],
        }
    }

    /// The bytes of a twin's texel: its four codes.
    pub const TEXEL_BYTES: u32 = 4;

    /// The decode of the format's linear form, whose blocks decode to the same codes.
    pub fn linear(self) -> PvrtcDecode {
        PvrtcDecode {
            srgb: false,
            ..self
        }
    }

    /// The Vulkan format of a twin's texels: `R8G8B8A8_UNORM`, its codes.
    pub fn texel_format(self) -> Format {
        Format::R8G8B8A8_UNORM
    }

    /// The `float` Metal reads of channel `ch` (red, green, blue, alpha) of code `c`.
    pub fn metal_value(self, ch: usize, c: u8) -> f32 {
        match self.srgb && ch < 3 {
            true => (f64::from(SRGB_DECODE_STEPS[usize::from(c)]) / 4095.0) as f32,
            false => (f64::from(c) / 255.0) as f32,
        }
    }

    /// The twin's texels (each its four codes, red in the low byte) of the `depth` images of a
    /// whole level `width` x `height` whose `blocks` lie row-major (each image's after the last, a
    /// level of fewer texels than a block holding one), laid out as a copy into the twin reads
    /// them: the module docs' decode, wrapping at each image's blocks.
    ///
    /// # Panics
    ///
    /// Where `blocks` holds fewer blocks than the level.
    pub fn decode(self, blocks: &[u8], [width, height, depth]: [u32; 3]) -> Vec<u8> {
        let image = self.level_bytes([width, height]);
        let texels = (width * height) as usize * Self::TEXEL_BYTES as usize;
        let mut out = vec![0; texels * depth as usize];
        for (i, out) in out.chunks_exact_mut(texels.max(1)).enumerate() {
            let blocks = &blocks[i * image..(i + 1) * image];
            self.decode_rows(blocks, [width, height], [0, height], out);
        }
        out
    }

    /// The bytes of the blocks of one image of a level `width` x `height`.
    pub fn level_bytes(self, [width, height]: [u32; 2]) -> usize {
        let [bw, bh] = self.block_extent();
        (width.div_ceil(bw) * height.div_ceil(bh) * Self::BLOCK_BYTES) as usize
    }

    /// The twin's texels of rows `first` to `first + rows` of one image of a level `width` x
    /// `height` whose `blocks` (all of the image's, row-major) wrap, into `out` (exactly their
    /// bytes), laid out as a copy into the twin reads them.
    ///
    /// # Panics
    ///
    /// Where `blocks` holds fewer blocks than the image, or `out` another size than the rows'.
    pub fn decode_rows(
        self,
        blocks: &[u8],
        [width, height]: [u32; 2],
        [first, rows]: [u32; 2],
        out: &mut [u8],
    ) {
        let [bw, bh] = self.block_extent();
        let (across, down) = (width.div_ceil(bw), height.div_ceil(bh));
        let words: Vec<u64> = blocks[..self.level_bytes([width, height])]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| u64::from_le_bytes(*bytes))
            .collect();
        let level = Level {
            decode: self,
            words: &words,
            across,
            down,
        };
        assert_eq!(out.len(), (width * rows * Self::TEXEL_BYTES) as usize);
        let mut texels = out.as_chunks_mut::<4>().0.iter_mut();
        for y in first..first + rows {
            for x in 0..width {
                if let Some(texel) = texels.next() {
                    *texel = level.texel(x, y);
                }
            }
        }
    }
}

/// Colors A and B of the block `word`, each red, green and blue of 5 bits and alpha of 4.
fn colors(word: u64) -> [[u32; 4]; 2] {
    let high = (word >> 32) as u32;
    let (a, b) = (high & 0xffff, high >> 16);
    let four = |c: u32| c << 1 | c >> 3;
    let b = match b & 0x8000 != 0 {
        true => [(b >> 10) & 31, (b >> 5) & 31, b & 31, 15],
        false => [
            four((b >> 8) & 15),
            four((b >> 4) & 15),
            four(b & 15),
            ((b >> 12) & 7) << 1,
        ],
    };
    let a = match a & 0x8000 != 0 {
        true => [(a >> 10) & 31, (a >> 5) & 31, four((a >> 1) & 15), 15],
        false => {
            let blue = (a >> 1) & 7;
            [
                four((a >> 8) & 15),
                four((a >> 4) & 15),
                blue << 2 | blue >> 1,
                ((a >> 12) & 7) << 1,
            ]
        }
    };
    [a, b]
}

/// One image of a level's blocks.
struct Level<'a> {
    decode: PvrtcDecode,
    words: &'a [u64],
    across: u32,
    down: u32,
}

impl Level<'_> {
    /// The block `(bx, by)`, wrapping.
    fn word(&self, bx: i64, by: i64) -> u64 {
        let bx = bx.rem_euclid(i64::from(self.across)) as usize;
        let by = by.rem_euclid(i64::from(self.down)) as usize;
        self.words[by * self.across as usize + bx]
    }

    /// The weight of B, in eighths, a 2-bit-a-pixel block stores at texel `(x, y)` of the level
    /// (wrapping), of even `x + y` in mode 1 or any in mode 0.
    fn stored(&self, x: i64, y: i64) -> u32 {
        let [bw, bh] = self.decode.block_extent().map(i64::from);
        let word = self.word(x.div_euclid(bw), y.div_euclid(bh));
        let (x, y) = (x.rem_euclid(bw) as u32, y.rem_euclid(bh) as u32);
        let mut m = word as u32;
        if (word >> 32) & 1 == 0 {
            return WEIGHTS[3 * ((m >> (8 * y + x)) & 1) as usize];
        }
        // Bit 21 into bit 20 where bit 0 is set, then bit 1 into bit 0.
        if m & 1 != 0 {
            m = (m & !(1 << 20)) | ((m >> 21) & 1) << 20;
        }
        m = (m & !1) | ((m >> 1) & 1);
        // The checkerboard's texels, row-major, hold 2-bit values in turn.
        let k = (8 * y + x) / 2;
        WEIGHTS[((m >> (2 * k)) & 3) as usize]
    }

    /// The weight of B, in eighths, of texel `(x, y)` of a 2-bit-a-pixel level.
    fn weight_two_bits(&self, x: i64, y: i64) -> u32 {
        let [bw, bh] = self.decode.block_extent().map(i64::from);
        let word = self.word(x.div_euclid(bw), y.div_euclid(bh));
        let (bx, by) = (x.rem_euclid(bw), y.rem_euclid(bh));
        if (word >> 32) & 1 == 0 || (bx ^ by) & 1 == 0 {
            return self.stored(x, y);
        }
        let m = word as u32;
        if m & 1 == 0 {
            let sum = self.stored(x, y - 1)
                + self.stored(x, y + 1)
                + self.stored(x - 1, y)
                + self.stored(x + 1, y);
            return (sum + 2) / 4;
        }
        // Bit 20 as stored chooses the vertical neighbours.
        match (m >> 20) & 1 != 0 {
            true => (self.stored(x, y - 1) + self.stored(x, y + 1)).div_ceil(2),
            false => (self.stored(x - 1, y) + self.stored(x + 1, y)).div_ceil(2),
        }
    }

    /// The codes of texel `(x, y)`.
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let decode = self.decode;
        let [bw, bh] = decode.block_extent().map(i64::from);
        let (x, y) = (i64::from(x), i64::from(y));
        let (px, py) = (x - bw / 2, y - bh / 2);
        let (bx, by) = (px.div_euclid(bw), py.div_euclid(bh));
        let (u, v) = (px - bx * bw, py - by * bh);
        let corners = [
            (colors(self.word(bx, by)), (bw - u) * (bh - v)),
            (colors(self.word(bx + 1, by)), u * (bh - v)),
            (colors(self.word(bx, by + 1)), (bw - u) * v),
            (colors(self.word(bx + 1, by + 1)), u * v),
        ];
        let up = |which: usize, ch: usize| -> u32 {
            let sum: i64 = corners
                .iter()
                .map(|(c, weight)| i64::from(c[which][ch]) * weight)
                .sum();
            let v = sum as u32;
            match (decode.two_bits, ch < 3) {
                (false, true) => (v >> 1) + (v >> 6),
                (false, false) => v + (v >> 4),
                (true, true) => (v >> 2) + (v >> 7),
                (true, false) => (v >> 1) + (v >> 5),
            }
        };
        let word = self.word(x.div_euclid(bw), y.div_euclid(bh));
        let (weight, punch) = match decode.two_bits {
            true => (self.weight_two_bits(x, y), false),
            false => {
                let index = (word >> (2 * (4 * y.rem_euclid(4) + x.rem_euclid(4)))) & 3;
                match (word >> 32) & 1 != 0 {
                    true => (PUNCH_WEIGHTS[index as usize], index == 2),
                    false => (WEIGHTS[index as usize], false),
                }
            }
        };
        let mut texel = [0u8; 4];
        for (ch, code) in texel.iter_mut().enumerate() {
            *code = ((up(0, ch) * (8 - weight) + up(1, ch) * weight) / 8) as u8;
        }
        if punch {
            texel[3] = 0;
        }
        if !decode.alpha {
            texel[3] = 255;
        }
        texel
    }
}

#[cfg(test)]
mod tests;
