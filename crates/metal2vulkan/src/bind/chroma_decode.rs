//! A 4:2:2 texture's texels as Metal reads them, from its pairs: the texels the decoded twin of a
//! `GBGR422` or `BGRG422` texture holds.
//!
//! **The decode.** Two texels share 4 bytes: their own green codes and one blue and one red code
//! (`G0 B G1 R` for `GBGR422`, `B G0 R G1` for `BGRG422`), sited between the two. Metal reads
//! texel `x` as its own green code and, for red and blue, the codes interpolated at its centre:
//! `(3 c + n + 2) >> 2` of its pair's code `c` and the code `n` of the pair beside it on its side
//! (the pair before an even texel, after an odd one; its own pair at the row's ends), each
//! `code / 255`, alpha 1. Our probe on an Apple M3 Max under macOS 27.0.1 (both formats, 16 x 4
//! texels of random bytes): every `float` and `half` read, nearest, linear and bicubic sample
//! (`float` and `half`) and gather is bit for bit the one of an `RGBA8Unorm` texture holding those
//! codes, whose treatment their class takes (`metal::texel_format`, `RGBX8Unorm`'s). The
//! twin holds them (`R8G8B8A8_UNORM`, alpha 255), exact, and the device samples and filters it as
//! it does any 8-bit format's texture, which is how Metal samples them.
//!
//! A texel's codes read its row's pairs alone, so a decode is of whole rows.

use crate::metal::PixelFormat;
use crate::vk::Format;

/// How a 4:2:2 format's pairs lie (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChromaDecode {
    /// `GBGR422`: `G0 B G1 R`.
    Gbgr,
    /// `BGRG422`: `B G0 R G1`.
    Bgrg,
}

impl ChromaDecode {
    /// The decode of a texture of `pf`; `None` for a format of no pairs.
    pub fn of(pf: PixelFormat) -> Option<ChromaDecode> {
        match pf {
            PixelFormat::GBGR422 => Some(ChromaDecode::Gbgr),
            PixelFormat::BGRG422 => Some(ChromaDecode::Bgrg),
            _ => None,
        }
    }

    /// Both decodes.
    pub const ALL: [ChromaDecode; 2] = [ChromaDecode::Gbgr, ChromaDecode::Bgrg];

    /// The texels across and down of a pair.
    pub fn block_extent(self) -> [u32; 2] {
        [2, 1]
    }

    /// The bytes of a pair.
    pub const BLOCK_BYTES: u32 = 4;

    /// The bytes of a decoded texel.
    pub const TEXEL_BYTES: u32 = 4;

    /// The twin's format: `R8G8B8A8_UNORM` codes.
    pub fn texel_format(self) -> Format {
        Format::R8G8B8A8_UNORM
    }

    /// The byte of a pair (its 4 bytes as a little-endian word, byte `k` at bits `8k`) holding
    /// red, blue, and the green of the pair's texel `t` (0 or 1).
    fn bytes(self) -> (u32, u32, [u32; 2]) {
        match self {
            ChromaDecode::Gbgr => (3, 1, [0, 2]),
            ChromaDecode::Bgrg => (2, 0, [1, 3]),
        }
    }

    /// Texel `t` (0 or 1) of `pair` beside `next` (the pair before an even texel, after an odd
    /// one), each its 4 bytes as a little-endian word: the twin's codes as a little-endian word
    /// (red lowest, alpha 255).
    fn texel(self, pair: u32, next: u32, t: usize) -> u32 {
        let (r, b, g) = self.bytes();
        let byte = |word: u32, k: u32| (word >> (8 * k)) & 0xff;
        let chroma = |k: u32| (3 * byte(pair, k) + byte(next, k) + 2) >> 2;
        chroma(r) | byte(pair, g[t]) << 8 | chroma(b) << 16 | 0xff << 24
    }

    /// The twin's texels of whole rows `width` x `height` x `depth` (images one after another)
    /// whose `pairs` are laid out as a copy of the rows out of the texture's carrier lays them
    /// (each texel's 2 bytes, tight rows), laid out as a copy into the twin reads them.
    pub fn decode(self, pairs: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let mut out = vec![0; width * height * depth * Self::TEXEL_BYTES as usize];
        self.decode_into(pairs, size, &mut out);
        out
    }

    /// [`decode`](Self::decode) into `out`, which holds exactly the rows' texels. A row of an odd
    /// width ends inside a pair, whose bytes it holds whole.
    ///
    /// # Panics
    ///
    /// Where `pairs` holds fewer pairs than the rows, or `out` another length.
    pub fn decode_into(self, pairs: &[u8], [width, height, depth]: [u32; 3], out: &mut [u8]) {
        let (width, rows) = (width as usize, height as usize * depth as usize);
        let across = width.div_ceil(2);
        assert_eq!(out.len(), width * rows * 4, "the rows' texels");
        let word = |at: usize| {
            u32::from_le_bytes([pairs[at], pairs[at + 1], pairs[at + 2], pairs[at + 3]])
        };
        for (y, row) in out.chunks_exact_mut(width * 4).enumerate() {
            for (x, texel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let p = x / 2;
                let next = match x % 2 {
                    0 => p.saturating_sub(1),
                    _ => (p + 1).min(across - 1),
                };
                let (pair, next) = (word((y * across + p) * 4), word((y * across + next) * 4));
                *texel = self.texel(pair, next, x % 2).to_le_bytes();
            }
        }
    }
}

#[cfg(test)]
mod tests;
