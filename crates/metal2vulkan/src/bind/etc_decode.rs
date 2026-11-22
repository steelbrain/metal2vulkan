//! An ETC2 or EAC texture's texels as Metal decodes them, from its blocks: the texels the decoded
//! twin of the texture holds on a device without the format, for a consumer that keeps one.
//!
//! **The decode** is the Khronos Data Format Specification's (its ETC2 and EAC chapters), written
//! here from the specification. A block is 64 bits read most significant byte first; its texels
//! are numbered down each column (texel `(x, y)` is index `4x + y`).
//!
//! - **ETC2 RGB** ([`EtcDecode::Rgb8`]): bit 33 chooses the individual mode (two 4-bit base
//!   colors) or the differential one (a 5-bit base and a 3-bit signed delta per channel), where a
//!   red, green or blue sum outside `[0, 31]` selects the T, H or planar mode instead. A texel is
//!   its subblock's base plus the modifier its table and 2-bit index pick, a paint color of the T
//!   and H modes, or the planar mode's gradient `(x (H - O) + y (V - O) + 4 O + 2) >> 2`, each
//!   channel clamped to `[0, 255]`; alpha is 255.
//! - **ETC2 RGB8A1** ([`EtcDecode::Rgb8A1`]): bit 33 is the opaque flag and the individual mode
//!   does not exist. With the flag clear, index 2 is transparent black (all four channels 0) in the
//!   differential, T and H modes, and the differential mode's modifiers are `0, b, 0, -b`; a planar
//!   block is opaque.
//! - **EAC** (the alpha of [`EtcDecode::Rgba8`], [`EtcDecode::R11Unorm`] and the others): an
//!   8-bit base, a 4-bit multiplier `m`, one of sixteen modifier tables and a 3-bit index per
//!   texel. Alpha is `clamp(base + modifier m, 0, 255)`; an unsigned 11-bit channel is
//!   `clamp(8 base + 4 + 8 modifier m, 0, 2047)` and a signed one `clamp(8 base + 8 modifier m,
//!   -1023, 1023)`, where `m = 0` takes the modifier itself instead of `8 modifier m` and a signed
//!   base of -128 is -127. `EAC_RGBA8`'s block is the alpha block, then the RGB block; `RG11`'s is
//!   red's block, then green's.
//!
//! **Metal's answers** (an Apple M3 Max under macOS 27.0.1, our own probe: 4096 random blocks of
//! each format, every texel read as `float`, equal to this decode's in every channel): an 8-bit
//! channel reads `c / 255`, an sRGB format's color channel the step of Metal's twelve-bit table
//! (`lower::arith::srgb`, as BC1 to BC3's sRGB forms read, `metal::texel_format::srgb_coded`),
//! alpha `c / 255`; an unsigned 11-bit channel reads `c / 2047` and a signed one `c / 1023`,
//! correctly rounded to `f32` (not the 16-bit code the specification widens them to).
//!
//! **The twin's texels** ([`EtcTexels`]): `R8G8B8A8_UNORM` codes of the ETC2 formats and
//! `EAC_RGBA8` either way (an sRGB form's codes too, which the device decodes through the twin's
//! `R8G8B8A8_SRGB` view). An 11-bit format's: [`EtcTexels::Compact`] holds the 16-bit
//! normalized code nearest each value (`R16_UNORM`, `R16_SNORM` and their two-channel forms), at
//! most half a 16-bit step off it; [`EtcTexels::Float`] holds the values exactly (`R32_SFLOAT`,
//! `R32G32_SFLOAT`).

use crate::metal::PixelFormat;
#[cfg(test)]
use crate::vk::Format;

/// How an ETC2 or EAC format's blocks decode (the module docs). The sRGB forms decode to their
/// unorm forms' codes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EtcDecode {
    /// `ETC2_RGB8` and its sRGB form: one RGB block.
    Rgb8,
    /// `ETC2_RGB8A1` and its sRGB form: one RGB block with punch-through alpha.
    Rgb8A1,
    /// `EAC_RGBA8` and its sRGB form: an alpha block, then an RGB block.
    Rgba8,
    /// `EAC_R11Unorm`: one unsigned 11-bit block.
    R11Unorm,
    /// `EAC_R11Snorm`: one signed 11-bit block.
    R11Snorm,
    /// `EAC_RG11Unorm`: two unsigned 11-bit blocks, red then green.
    Rg11Unorm,
    /// `EAC_RG11Snorm`: two signed 11-bit blocks, red then green.
    Rg11Snorm,
}

/// The texels a decoded twin of an ETC2 or EAC texture holds (the module docs): they differ for
/// the 11-bit formats alone.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EtcTexels {
    /// 8-bit codes, and an 11-bit format's 16-bit normalized codes nearest Metal's values.
    Compact,
    /// 8-bit codes, and an 11-bit format's values exactly, as `float`s.
    Float,
}

/// The individual and differential modes' modifier pairs `(a, b)` of each table: index 0 is `a`,
/// 1 is `b`, 2 is `-a` and 3 is `-b`.
const MODIFIERS: [(i32, i32); 8] = [
    (2, 8),
    (5, 17),
    (9, 29),
    (13, 42),
    (18, 60),
    (24, 80),
    (33, 106),
    (47, 183),
];

/// The T and H modes' distances.
const DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

/// The EAC modifier tables.
const EAC_MODIFIERS: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

impl EtcDecode {
    /// The decode of `pf`'s blocks: the ETC2 and EAC formats, the sRGB forms decoding to the unorm
    /// forms' codes; `None` for every other format.
    pub fn of(pf: PixelFormat) -> Option<EtcDecode> {
        use PixelFormat as P;
        Some(match pf {
            P::ETC2RGB8 | P::ETC2RGB8SRGB => EtcDecode::Rgb8,
            P::ETC2RGB8A1 | P::ETC2RGB8A1SRGB => EtcDecode::Rgb8A1,
            P::EACRGBA8 | P::EACRGBA8SRGB => EtcDecode::Rgba8,
            P::EACR11Unorm => EtcDecode::R11Unorm,
            P::EACR11Snorm => EtcDecode::R11Snorm,
            P::EACRG11Unorm => EtcDecode::Rg11Unorm,
            P::EACRG11Snorm => EtcDecode::Rg11Snorm,
            _ => return None,
        })
    }

    /// The bytes of a block: 8, or 16 for `EAC_RGBA8` and the `RG11` formats.
    pub fn block_bytes(self) -> u32 {
        match self {
            EtcDecode::Rgba8 | EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm => 16,
            _ => 8,
        }
    }

    /// Whether the decode's values are 11-bit ones (the `R11` and `RG11` formats'), which no 8-bit
    /// code holds.
    pub fn eleven_bit(self) -> bool {
        !matches!(self, EtcDecode::Rgb8 | EtcDecode::Rgb8A1 | EtcDecode::Rgba8)
    }

    /// The bytes of a twin's texel of `texels`: 4 for the 8-bit codes; an 11-bit format's 2 or 4
    /// a channel.
    pub fn texel_bytes(self, texels: EtcTexels) -> u32 {
        let channel = match texels {
            EtcTexels::Compact => 2,
            EtcTexels::Float => 4,
        };
        match self {
            EtcDecode::Rgb8 | EtcDecode::Rgb8A1 | EtcDecode::Rgba8 => 4,
            EtcDecode::R11Unorm | EtcDecode::R11Snorm => channel,
            EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm => 2 * channel,
        }
    }

    /// The values of the 16 texels of `block`, row-major (texel `(x, y)` at `4y + x`): the packed
    /// 8-bit codes, red in the low byte, or an 11-bit format's values (red, then an `RG11`
    /// format's green; 0 elsewhere).
    pub fn block_values(self, block: &[u8]) -> [[i32; 2]; 16] {
        let codes = |[r, g, b, a]: [u8; 4]| i32::from_le_bytes([r, g, b, a]);
        let word = |block: &[u8]| u64::from_be_bytes(block[..8].try_into().expect("8 bytes"));
        match self {
            EtcDecode::Rgb8 | EtcDecode::Rgb8A1 => {
                let texels = rgb_block(word(block), self == EtcDecode::Rgb8A1);
                std::array::from_fn(|j| [codes(texels[j]), 0])
            }
            EtcDecode::Rgba8 => {
                let alpha = eac_block(word(block), Channel::Alpha);
                let texels = rgb_block(word(&block[8..]), false);
                std::array::from_fn(|j| {
                    let [r, g, b, _] = texels[j];
                    [codes([r, g, b, alpha[j] as u8]), 0]
                })
            }
            EtcDecode::R11Unorm | EtcDecode::R11Snorm => {
                let red = eac_block(word(block), self.channel());
                std::array::from_fn(|j| [red[j], 0])
            }
            EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm => {
                let red = eac_block(word(block), self.channel());
                let green = eac_block(word(&block[8..]), self.channel());
                std::array::from_fn(|j| [red[j], green[j]])
            }
        }
    }

    /// The `float` Metal reads of an 11-bit format's value `c`: `c / 2047` (unsigned) or
    /// `c / 1023` (signed), correctly rounded. (`c` over the divisor lies on no `f32` tie, so
    /// rounding the `f64` quotient again is the correctly rounded `f32`.)
    pub fn metal_value(self, c: i32) -> f32 {
        (f64::from(c) / f64::from(self.channel().divisor())) as f32
    }

    /// The channel kind of an 11-bit format's blocks.
    fn channel(self) -> Channel {
        match self {
            EtcDecode::R11Snorm | EtcDecode::Rg11Snorm => Channel::Signed,
            _ => Channel::Unsigned,
        }
    }

    /// The twin's texel of `texels` of a texel whose values are `v` ([`block_values`]), into
    /// `out` ([`texel_bytes`](Self::texel_bytes) of them).
    ///
    /// [`block_values`]: Self::block_values
    fn twin_texel(self, texels: EtcTexels, v: [i32; 2], out: &mut [u8]) {
        if !self.eleven_bit() {
            out.copy_from_slice(&v[0].to_le_bytes());
            return;
        }
        let divisor = self.channel().divisor();
        for (k, out) in out
            .chunks_exact_mut(out.len() / self.channels())
            .enumerate()
        {
            match texels {
                // The 16-bit normalized code nearest `c / divisor`: `c · (2^b - 1) / divisor`
                // rounded, which is never a tie (the divisor is odd).
                EtcTexels::Compact => {
                    let top = match self.channel() {
                        Channel::Signed => 32767,
                        _ => 65535,
                    };
                    let c = v[k];
                    let code = c.signum() * ((2 * c.abs() * top + divisor) / (2 * divisor));
                    out.copy_from_slice(&(code as i16 as u16).to_le_bytes());
                }
                EtcTexels::Float => out.copy_from_slice(&self.metal_value(v[k]).to_le_bytes()),
            }
        }
    }

    /// The channels of a decoded texel: 4 codes count as one word, an `RG11` format's 2.
    fn channels(self) -> usize {
        match self {
            EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm => 2,
            _ => 1,
        }
    }

    /// The twin's texels of `texels` of a region `width` x `height` x `depth` whose `blocks` are
    /// laid out as a copy of the region out of the texture lays them (tight rows of blocks, each
    /// image's after the last), laid out as a copy into the twin reads them.
    pub fn decode(self, texels: EtcTexels, blocks: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let bytes = self.texel_bytes(texels) as usize;
        let mut out = vec![0; width * height * depth * bytes];
        self.decode_into(texels, blocks, size, &mut out);
        out
    }

    /// [`decode`](Self::decode) into `out`, which holds exactly the region's texels: so bands of
    /// a region's block rows can be decoded on several threads into one buffer.
    ///
    /// # Panics
    ///
    /// Where `blocks` holds fewer blocks than the region, or `out` another length.
    pub fn decode_into(
        self,
        texels: EtcTexels,
        blocks: &[u8],
        [width, height, depth]: [u32; 3],
        out: &mut [u8],
    ) {
        let size = self.block_bytes() as usize;
        let (width, height) = (width as usize, height as usize);
        let (across, down) = (width.div_ceil(4), height.div_ceil(4));
        let bytes = self.texel_bytes(texels) as usize;
        let row = width * bytes;
        assert_eq!(
            out.len(),
            row * height * depth as usize,
            "the region's texels"
        );
        for b in 0..across * down * depth as usize {
            let values = self.block_values(&blocks[b * size..(b + 1) * size]);
            let (z, by, bx) = (b / (across * down), b / across % down, b % across);
            for (ty, y) in (by * 4..(by * 4 + 4).min(height)).enumerate() {
                for (tx, x) in (bx * 4..(bx * 4 + 4).min(width)).enumerate() {
                    let at = (z * height + y) * row + x * bytes;
                    self.twin_texel(texels, values[ty * 4 + tx], &mut out[at..at + bytes]);
                }
            }
        }
    }

    /// Every decode.
    #[cfg(test)]
    pub(crate) const ALL: [EtcDecode; 7] = [
        EtcDecode::Rgb8,
        EtcDecode::Rgb8A1,
        EtcDecode::Rgba8,
        EtcDecode::R11Unorm,
        EtcDecode::R11Snorm,
        EtcDecode::Rg11Unorm,
        EtcDecode::Rg11Snorm,
    ];

    /// The Vulkan format of a twin's texels of `texels`: `R8G8B8A8_UNORM` codes; an 11-bit
    /// format's 16-bit normalized codes or `float` values.
    #[cfg(test)]
    pub(crate) fn texel_format(self, texels: EtcTexels) -> Format {
        use EtcTexels as T;
        match (self, texels) {
            (EtcDecode::Rgb8 | EtcDecode::Rgb8A1 | EtcDecode::Rgba8, _) => Format::R8G8B8A8_UNORM,
            (EtcDecode::R11Unorm, T::Compact) => Format::R16_UNORM,
            (EtcDecode::R11Snorm, T::Compact) => Format::R16_SNORM,
            (EtcDecode::Rg11Unorm, T::Compact) => Format::R16G16_UNORM,
            (EtcDecode::Rg11Snorm, T::Compact) => Format::R16G16_SNORM,
            (EtcDecode::R11Unorm | EtcDecode::R11Snorm, T::Float) => Format::R32_SFLOAT,
            (EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm, T::Float) => Format::R32G32_SFLOAT,
        }
    }
}

/// Bits `lo..=hi` of `v`.
fn bits(v: u64, hi: u32, lo: u32) -> i32 {
    ((v >> lo) & ((1 << (hi - lo + 1)) - 1)) as i32
}

/// `v`, of `n` bits, widened to 8 by replicating its high bits.
fn widen(v: i32, n: u32) -> i32 {
    (v << (8 - n)) | (v >> (2 * n - 8))
}

/// A 3-bit two's complement delta.
fn delta(v: i32) -> i32 {
    if v & 4 != 0 {
        v - 8
    } else {
        v
    }
}

/// `v` clamped to a byte.
fn byte(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Each of a color's channels plus `d`, clamped.
fn shift([r, g, b]: [i32; 3], d: i32) -> [u8; 3] {
    [byte(r + d), byte(g + d), byte(b + d)]
}

/// The texels of an ETC2 RGB block `v` (most significant byte first), row-major, red to alpha;
/// `punch`: the RGB8A1 form, whose bit 33 is the opaque flag.
fn rgb_block(v: u64, punch: bool) -> [[u8; 4]; 16] {
    let flag = bits(v, 33, 33) == 1;
    let opaque = !punch || flag;
    let differential = punch || flag;
    // Texel `p`'s 2-bit index (texels numbered down each column).
    let index = |p: u32| (bits(v, 16 + p, 16 + p) << 1 | bits(v, p, p)) as usize;
    let mut out = [[0u8; 4]; 16];
    let mut put = |p: u32, rgb: Option<[u8; 3]>| {
        let (x, y) = (p / 4, p % 4);
        out[(4 * y + x) as usize] = match rgb {
            Some([r, g, b]) => [r, g, b, 255],
            None => [0; 4],
        };
    };
    let (base1, base2) = if differential {
        let (r, g, b) = (bits(v, 63, 59), bits(v, 55, 51), bits(v, 47, 43));
        let (dr, dg, db) = (
            delta(bits(v, 58, 56)),
            delta(bits(v, 50, 48)),
            delta(bits(v, 42, 40)),
        );
        let outside = |c: i32| !(0..=31).contains(&c);
        if outside(r + dr) || outside(g + dg) {
            // The T mode (red overflows) or the H mode (green does): four paint colors.
            let paint = if outside(r + dr) {
                let c1 = [
                    bits(v, 60, 59) << 2 | bits(v, 57, 56),
                    bits(v, 55, 52),
                    bits(v, 51, 48),
                ]
                .map(|c| widen(c, 4));
                let c2 = [bits(v, 47, 44), bits(v, 43, 40), bits(v, 39, 36)].map(|c| widen(c, 4));
                let d = DISTANCES[(bits(v, 35, 34) << 1 | bits(v, 32, 32)) as usize];
                [shift(c1, 0), shift(c2, d), shift(c2, 0), shift(c2, -d)]
            } else {
                let c1 = [
                    bits(v, 62, 59),
                    bits(v, 58, 56) << 1 | bits(v, 52, 52),
                    bits(v, 51, 51) << 3 | bits(v, 49, 47),
                ]
                .map(|c| widen(c, 4));
                let c2 = [bits(v, 46, 43), bits(v, 42, 39), bits(v, 38, 35)].map(|c| widen(c, 4));
                let key = |[r, g, b]: [i32; 3]| r << 16 | g << 8 | b;
                let low = i32::from(key(c1) >= key(c2));
                let d = DISTANCES[(bits(v, 34, 34) << 2 | bits(v, 32, 32) << 1 | low) as usize];
                [shift(c1, d), shift(c1, -d), shift(c2, d), shift(c2, -d)]
            };
            for p in 0..16 {
                let i = index(p);
                put(p, (opaque || i != 2).then_some(paint[i]));
            }
            return out;
        }
        if outside(b + db) {
            // The planar mode: always opaque.
            let o = [
                widen(bits(v, 62, 57), 6),
                widen(bits(v, 56, 56) << 6 | bits(v, 54, 49), 7),
                widen(
                    bits(v, 48, 48) << 5 | bits(v, 44, 43) << 3 | bits(v, 41, 39),
                    6,
                ),
            ];
            let h = [
                widen(bits(v, 38, 34) << 1 | bits(v, 32, 32), 6),
                widen(bits(v, 31, 25), 7),
                widen(bits(v, 24, 19), 6),
            ];
            let vert = [
                widen(bits(v, 18, 13), 6),
                widen(bits(v, 12, 6), 7),
                widen(bits(v, 5, 0), 6),
            ];
            for p in 0..16 {
                let (x, y) = ((p / 4) as i32, (p % 4) as i32);
                let channel =
                    |k: usize| byte((x * (h[k] - o[k]) + y * (vert[k] - o[k]) + 4 * o[k] + 2) >> 2);
                put(p, Some([channel(0), channel(1), channel(2)]));
            }
            return out;
        }
        (
            [r, g, b].map(|c| widen(c, 5)),
            [r + dr, g + dg, b + db].map(|c| widen(c, 5)),
        )
    } else {
        (
            [bits(v, 63, 60), bits(v, 55, 52), bits(v, 47, 44)].map(|c| widen(c, 4)),
            [bits(v, 59, 56), bits(v, 51, 48), bits(v, 43, 40)].map(|c| widen(c, 4)),
        )
    };
    let (table1, table2) = (bits(v, 39, 37) as usize, bits(v, 36, 34) as usize);
    let flip = bits(v, 32, 32) == 1;
    for p in 0..16 {
        let (x, y) = (p / 4, p % 4);
        let second = if flip { y >= 2 } else { x >= 2 };
        let (base, (a, b)) = match second {
            true => (base2, MODIFIERS[table2]),
            false => (base1, MODIFIERS[table1]),
        };
        let i = index(p);
        let modifier = match opaque {
            true => [a, b, -a, -b][i],
            false => [0, b, 0, -b][i],
        };
        put(p, (opaque || i != 2).then(|| shift(base, modifier)));
    }
    out
}

/// What an EAC block's values are.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// `EAC_RGBA8`'s alpha: 8-bit codes.
    Alpha,
    /// An unsigned 11-bit channel, `[0, 2047]`.
    Unsigned,
    /// A signed 11-bit channel, `[-1023, 1023]`.
    Signed,
}

impl Channel {
    /// The divisor of an 11-bit channel's value: 2047, or 1023 for a signed one.
    fn divisor(self) -> i32 {
        match self {
            Channel::Signed => 1023,
            _ => 2047,
        }
    }
}

/// The values of an EAC block `v` (most significant byte first), row-major.
fn eac_block(v: u64, channel: Channel) -> [i32; 16] {
    let base = bits(v, 63, 56);
    let multiplier = bits(v, 55, 52);
    let table = EAC_MODIFIERS[bits(v, 51, 48) as usize];
    let mut out = [0; 16];
    for p in 0..16u32 {
        let modifier = table[bits(v, 47 - 3 * p, 45 - 3 * p) as usize];
        // An 11-bit block with a multiplier of 0 adds the modifier itself, as an eighth of a step.
        let step = match multiplier {
            0 => modifier,
            m => 8 * modifier * m,
        };
        let value = match channel {
            Channel::Alpha => (base + modifier * multiplier).clamp(0, 255),
            Channel::Unsigned => (8 * base + 4 + step).clamp(0, 2047),
            Channel::Signed => {
                let base = (base as u8 as i8).max(-127);
                (8 * i32::from(base) + step).clamp(-1023, 1023)
            }
        };
        let (x, y) = (p / 4, p % 4);
        out[(4 * y + x) as usize] = value;
    }
    out
}

#[cfg(test)]
mod tests;
