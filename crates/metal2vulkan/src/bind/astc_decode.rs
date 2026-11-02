//! An ASTC texture's texels as Metal decodes them, from its blocks, for a device without its ASTC
//! block: the texels a decoded twin of the texture holds (the runtime's `texture/decoded_twin.rs`).
//!
//! **The decode** is the Khronos Data Format Specification's (its ASTC chapter): the block mode,
//! the integer sequence encoding of the colour endpoints and the weights, the colour endpoint
//! modes, the partition function and the weight grid's infill, written here from the
//! specification. Each channel of a texel comes out as its 16-bit interpolation result `C` of the
//! two endpoints widened to 16 bits: an LDR mode's 8-bit endpoints `e · 257`, or `e · 256 + 128`
//! for an `_sRGB` format; an HDR mode's 12-bit ones (an `_HDR` format's alone, `hdr.rs`) `e · 16`.
//!
//! **Metal's answers** (an Apple M3 Max under macOS 27.0.1, our own probe: 4096 blocks of each of
//! the fourteen footprints Vulkan's ASTC formats have, about 70% valid blocks of random modes,
//! void-extent blocks and error blocks, every texel read as `float` and `half` and sampled at its
//! centre, linear and sRGB, every channel equal to this decode's; and Apple's 4x2 and 8x4
//! footprints, which no Vulkan format has, in every profile, the probe
//! `astc_passes_writes_footprints`'s seeded blocks read alike):
//!
//! - **Linear:** each channel reads `(C >> 4) / 4095`, a 12-bit value, correctly rounded to `f32`
//!   (a `half` read rounds that to nearest even).
//! - **sRGB:** each colour channel reads Metal's twelve-bit table of the code `C >> 8`
//!   (`k / 4095` for `lower::arith::srgb`'s step `k`), alpha `(C >> 8) / 255`.
//! - **Error blocks** read magenta, `(1, 0, 1, 1)`: a reserved block mode, a weight grid larger
//!   than the block, more than 64 weights, fewer than 24 or more than 96 weight bits, two planes
//!   of four partitions, more than 18 colour values, too few bits for a colour range of 6 levels,
//!   an HDR void-extent block, a void-extent block whose reserved bits (10 and 11) are not both
//!   set or whose extent is empty. A partition of an HDR colour endpoint mode reads magenta at its
//!   own texels, the block's other partitions decoding as they are.
//! - A void-extent block's texels are its four 16-bit values, as `C`.
//!
//! **The HDR profile** (an `_HDR` format; Metal's answers on the same M3 Max, our probe
//! `astc_hdr_textures` and the conformance suite's recordings `astc-*-hdr`): every colour
//! endpoint mode decodes, each texel's channels half-precision values, which Metal reads exactly
//! (as `float` and as `half`):
//!
//! - an HDR mode's channel (each of modes 2, 3, 7, 11 and 15, and mode 14's colour) is the
//!   logarithmic `C`'s half: its top five bits the exponent, its low eleven a mantissa mapped
//!   piecewise linearly onto the half's ten, the largest finite half (65504) past it;
//! - an LDR mode's channel, and mode 14's alpha, is `C / 65536` truncated to a half, 1 for
//!   `C = 65535`;
//! - a void-extent block's values are halves where its HDR bit (9) is set (infinities kept, every
//!   NaN read as the quiet NaN `0x7FC00000`), else 16-bit values `C` read as an LDR channel is;
//! - an error block reads the quiet NaN in every channel.
//!
//! **The twin's texels** ([`AstcTexels`]): [`AstcTexels::Float`] holds each texel's value exactly
//! (`R32G32B32A32_SFLOAT`); [`AstcTexels::Half`] an HDR format's exactly as halves
//! (`R16G16B16A16_SFLOAT`); [`AstcTexels::Unorm8`] an LDR format's 8-bit codes
//! (`R8G8B8A8_UNORM`): a linear channel the code nearest its value, at most `1/510` off it, and an
//! sRGB format's channels the code `C >> 8` itself, which Metal decodes through its table
//! (`metal::texel_format::srgb_coded`) and the device decodes through the image's `R8G8B8A8_SRGB`
//! view (`PrecisionNote::SrgbTexelDecode`). Whatever its policy, the runtime holds an LDR or sRGB
//! format's twin in 8-bit codes and an HDR format's in halves.
//!
//! **The GPU's decode** ([`astc_decode_module`]) is this decode in integer operations, each
//! texel's from its block alone, writing the texels [`decode`](AstcDecode::decode) writes; its
//! bindings, shape and dispatch are the block decode's ([`BlockDecodeModule`]), its blocks and
//! texels laid out as [`decode`](AstcDecode::decode)'s are.

use std::sync::OnceLock;

use super::block_decode::{decode_module, BlockDecodeModule};
use crate::device::Device;
use crate::error::Error;
use crate::lower::arith::srgb::SRGB_DECODE_STEPS;
use crate::metal::PixelFormat;
use crate::vk::Format;

mod hdr;
mod kernel;

/// The module of `device` that decodes `decode`'s blocks into the texels of `texels` on the GPU
/// (the module docs): the bindings, shape and dispatch of [`BlockDecodeModule`], of which it
/// reads the shape's first three words (the region's width, height and depth), its blocks
/// `decode`'s, `⌈width / w⌉` across and `⌈height / h⌉` down of each layer for the footprint
/// `w` x `h`.
pub fn astc_decode_module(
    device: &Device,
    decode: AstcDecode,
    texels: AstcTexels,
) -> Result<BlockDecodeModule, Error> {
    decode_module(device, kernel::typed_module(decode, texels)?, "ASTC decode")
}

/// How an ASTC format's blocks decode: its footprint, and its profile (LDR, sRGB-encoded LDR, or
/// HDR).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AstcDecode {
    /// The block's texels across.
    pub width: u32,
    /// The block's texels down.
    pub height: u32,
    /// Whether the format is an `_sRGB` one.
    pub srgb: bool,
    /// Whether the format is an `_HDR` one: every colour endpoint mode decodes, into half-precision
    /// values (the module docs).
    pub hdr: bool,
}

/// The texels a decoded twin of an ASTC texture holds (the module docs). An LDR or sRGB format
/// has no half form and an HDR format no 8-bit one: each holds its compact form for the other.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AstcTexels {
    /// `R8G8B8A8_UNORM` codes: a linear channel the code nearest Metal's 12-bit value (at most
    /// `1/510` off it), an sRGB format's channels Metal's own codes, which a function reads raw
    /// and decodes through Metal's table. An HDR format's are [`Half`](Self::Half)'s.
    Unorm8,
    /// `R32G32B32A32_SFLOAT`: each channel Metal's value exactly.
    Float,
    /// `R16G16B16A16_SFLOAT`: an HDR format's channels, Metal's half-precision values exactly.
    /// An LDR or sRGB format's are [`Unorm8`](Self::Unorm8)'s.
    Half,
}

/// The most texels a block holds (12 x 12).
const MOST_TEXELS: usize = 144;

impl AstcDecode {
    /// The bytes of every block.
    pub const BLOCK_BYTES: u32 = 16;

    /// The decode of `pf`'s blocks: every ASTC format, LDR, `_sRGB` and `_HDR`, of the fourteen 2D
    /// footprints Vulkan's ASTC formats have and of Apple's 4x2 and 8x4, which no Vulkan format
    /// has; `None` for every other format.
    pub fn of(pf: PixelFormat) -> Option<AstcDecode> {
        let (srgb, hdr) = Self::profile(pf)?;
        let [width, height] = crate::metal::format::block_extent(pf)?;
        Some(AstcDecode {
            width,
            height,
            srgb,
            hdr,
        })
    }

    /// Whether the ASTC format `pf` is an `_sRGB` one and whether an `_HDR` one; `None` for any
    /// other format.
    fn profile(pf: PixelFormat) -> Option<(bool, bool)> {
        use PixelFormat as P;
        Some(match pf {
            P::ASTC4x2SRGB
            | P::ASTC4x4SRGB
            | P::ASTC5x4SRGB
            | P::ASTC5x5SRGB
            | P::ASTC6x5SRGB
            | P::ASTC6x6SRGB
            | P::ASTC8x4SRGB
            | P::ASTC8x5SRGB
            | P::ASTC8x6SRGB
            | P::ASTC8x8SRGB
            | P::ASTC10x5SRGB
            | P::ASTC10x6SRGB
            | P::ASTC10x8SRGB
            | P::ASTC10x10SRGB
            | P::ASTC12x10SRGB
            | P::ASTC12x12SRGB => (true, false),
            P::ASTC4x2LDR
            | P::ASTC4x4LDR
            | P::ASTC5x4LDR
            | P::ASTC5x5LDR
            | P::ASTC6x5LDR
            | P::ASTC6x6LDR
            | P::ASTC8x4LDR
            | P::ASTC8x5LDR
            | P::ASTC8x6LDR
            | P::ASTC8x8LDR
            | P::ASTC10x5LDR
            | P::ASTC10x6LDR
            | P::ASTC10x8LDR
            | P::ASTC10x10LDR
            | P::ASTC12x10LDR
            | P::ASTC12x12LDR => (false, false),
            P::ASTC4x2HDR
            | P::ASTC4x4HDR
            | P::ASTC5x4HDR
            | P::ASTC5x5HDR
            | P::ASTC6x5HDR
            | P::ASTC6x6HDR
            | P::ASTC8x4HDR
            | P::ASTC8x5HDR
            | P::ASTC8x6HDR
            | P::ASTC8x8HDR
            | P::ASTC10x5HDR
            | P::ASTC10x6HDR
            | P::ASTC10x8HDR
            | P::ASTC10x10HDR
            | P::ASTC12x10HDR
            | P::ASTC12x12HDR => (false, true),
            _ => return None,
        })
    }

    /// The form `texels` holds for this decode: [`AstcTexels::Float`], or the profile's compact
    /// form, [`AstcTexels::Half`] for an HDR format and [`AstcTexels::Unorm8`] for the others.
    pub fn form(self, texels: AstcTexels) -> AstcTexels {
        match (texels, self.hdr) {
            (AstcTexels::Float, _) => AstcTexels::Float,
            (_, true) => AstcTexels::Half,
            (_, false) => AstcTexels::Unorm8,
        }
    }

    /// The block's footprint, texels across and down.
    pub fn block_extent(self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// The Vulkan format of the twin's texels of `texels` ([`form`](Self::form)'s):
    /// `R8G8B8A8_UNORM` (an sRGB format's viewed as `R8G8B8A8_SRGB` too, where its linear samples
    /// are the device's filter), `R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT`.
    pub fn texel_format(self, texels: AstcTexels) -> Format {
        match self.form(texels) {
            AstcTexels::Unorm8 => Format::R8G8B8A8_UNORM,
            AstcTexels::Half => Format::R16G16B16A16_SFLOAT,
            AstcTexels::Float => Format::R32G32B32A32_SFLOAT,
        }
    }

    /// The Metal pixel format whose texels the twin's of `texels` are: `RGBA8Unorm`,
    /// `RGBA16Float` or `RGBA32Float`.
    pub fn texel_pixel_format(self, texels: AstcTexels) -> PixelFormat {
        match self.form(texels) {
            AstcTexels::Unorm8 => PixelFormat::RGBA8Unorm,
            AstcTexels::Half => PixelFormat::RGBA16Float,
            AstcTexels::Float => PixelFormat::RGBA32Float,
        }
    }

    /// The bytes of a twin's texel of `texels` ([`form`](Self::form)'s): 4, 8 or 16.
    pub fn texel_bytes(self, texels: AstcTexels) -> u32 {
        match self.form(texels) {
            AstcTexels::Unorm8 => 4,
            AstcTexels::Half => 8,
            AstcTexels::Float => 16,
        }
    }

    /// The block's texels, each channel's 16-bit interpolation result `C` (red, green, blue,
    /// alpha), or for an HDR format its half-precision value's bits, row-major; `None` for an
    /// error block (whose texels read magenta, or NaN for an HDR format).
    pub fn block_values(self, block: &[u8; 16]) -> Option<Vec<[u16; 4]>> {
        self.block_values_why(block).ok()
    }

    /// [`block_values`](Self::block_values), with why a block is an error block.
    pub(crate) fn block_values_why(self, block: &[u8; 16]) -> Result<Vec<[u16; 4]>, Illegal> {
        let mut out = [[0; 4]; MOST_TEXELS];
        decode_block(self, u128::from_le_bytes(*block), &mut out)?;
        Ok(out[..(self.width * self.height) as usize].to_vec())
    }

    /// The `float` channels Metal reads of a texel whose 16-bit interpolation results (an HDR
    /// format's half-precision bits) are `c` (the module docs).
    pub fn metal_value(self, c: [u16; 4]) -> [f32; 4] {
        if self.hdr {
            return c.map(hdr::half_value);
        }
        let table = twelve_bit_values();
        std::array::from_fn(|ch| match (self.srgb, ch) {
            (true, 3) => (f64::from(c[ch] >> 8) / 255.0) as f32,
            (true, _) => table[usize::from(SRGB_DECODE_STEPS[usize::from(c[ch] >> 8)])],
            (false, _) => table[usize::from(c[ch] >> 4)],
        })
    }

    /// The twin's texel of `texels` of a texel whose 16-bit interpolation results (an HDR
    /// format's half-precision bits) are `c`, its bytes in `out`
    /// ([`texel_bytes`](Self::texel_bytes) of them).
    fn twin_texel(self, texels: AstcTexels, c: [u16; 4], out: &mut [u8]) {
        match self.form(texels) {
            AstcTexels::Half => {
                for (half, c) in out.as_chunks_mut::<2>().0.iter_mut().zip(c) {
                    *half = c.to_le_bytes();
                }
            }
            AstcTexels::Unorm8 => {
                for (byte, c) in out.iter_mut().zip(c) {
                    *byte = match self.srgb {
                        true => (c >> 8) as u8,
                        // The code nearest `k / 4095`: no `k` lies on a half code.
                        false => ((u32::from(c >> 4) * 255 + 2047) / 4095) as u8,
                    };
                }
            }
            AstcTexels::Float => {
                for (word, value) in out
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(self.metal_value(c))
                {
                    *word = value.to_le_bytes();
                }
            }
        }
    }

    /// The twin's texels of `texels` of a region `width` x `height` x `depth` whose `blocks` are
    /// laid out as a copy of the region out of the texture lays them (tight rows of blocks, each
    /// image's after the last), laid out as a copy into the twin reads them.
    pub fn decode(self, texels: AstcTexels, blocks: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let bytes = self.texel_bytes(texels) as usize;
        let mut out = vec![0; width * height * depth * bytes];
        self.decode_into(texels, blocks, size, &mut out);
        out
    }

    /// [`decode`](Self::decode) into `out`, which holds exactly the region's texels: so bands of a
    /// region's block rows can be decoded on several threads into one buffer.
    ///
    /// # Panics
    ///
    /// Where `blocks` holds fewer blocks than the region, or `out` another length.
    pub fn decode_into(
        self,
        texels: AstcTexels,
        blocks: &[u8],
        [width, height, depth]: [u32; 3],
        out: &mut [u8],
    ) {
        let (bw, bh) = (self.width as usize, self.height as usize);
        let (width, height) = (width as usize, height as usize);
        let (across, down) = (width.div_ceil(bw), height.div_ceil(bh));
        let bytes = self.texel_bytes(texels) as usize;
        let row = width * bytes;
        assert_eq!(
            out.len(),
            row * height * depth as usize,
            "the region's texels"
        );
        let mut values = [[0u16; 4]; MOST_TEXELS];
        for b in 0..across * down * depth as usize {
            let block: [u8; 16] = blocks[b * 16..b * 16 + 16]
                .try_into()
                .expect("a block's 16 bytes");
            if decode_block(self, u128::from_le_bytes(block), &mut values).is_err() {
                values.fill(self.error_texel());
            }
            let (z, by, bx) = (b / (across * down), b / across % down, b % across);
            for (ty, y) in (by * bh..(by * bh + bh).min(height)).enumerate() {
                for (tx, x) in (bx * bw..(bx * bw + bw).min(width)).enumerate() {
                    let at = (z * height + y) * row + x * bytes;
                    self.twin_texel(texels, values[ty * bw + tx], &mut out[at..at + bytes]);
                }
            }
        }
    }
}

/// `k / 4095` correctly rounded to `f32`, for every 12-bit `k`. (`k / 4095` lies on no `f32`
/// tie, so rounding its `f64` quotient once more is the correctly rounded `f32`.)
fn twelve_bit_values() -> &'static [f32; 4096] {
    static TABLE: OnceLock<[f32; 4096]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|k| (k as f64 / 4095.0) as f32))
}

impl AstcDecode {
    /// An error texel's values: magenta, or NaN for an HDR format.
    fn error_texel(self) -> [u16; 4] {
        match self.hdr {
            true => [HDR_ERROR; 4],
            false => ERROR_TEXEL,
        }
    }
}

/// Why a block is an error block (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Illegal {
    ReservedMode,
    GridPastBlock,
    TooManyWeights,
    WeightBits,
    DualPlaneFourPartitions,
    ConfigPastWeights,
    ColorPastWeights,
    TooManyValues,
    NoColorRange,
    HdrVoidExtent,
    VoidExtentReserved,
    VoidExtentEmpty,
}

/// The error colour's 16-bit values: magenta.
const ERROR_TEXEL: [u16; 4] = [0xFFFF, 0, 0xFFFF, 0xFFFF];

/// An HDR format's error channel: a quiet NaN half.
const HDR_ERROR: u16 = 0x7E00;

// ---------------------------------------------------------------------------------------------
// The integer sequence encoding.

/// A range of values the integer sequence encoding holds: `levels` values, `bits` low bits each
/// and a trit (3), a quint (5) or neither (1) on top.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    bits: u32,
    /// 1, 3 or 5.
    base: u32,
}

impl Range {
    const fn levels(self) -> u32 {
        self.base << self.bits
    }

    /// The bits `count` values of the range take.
    const fn ise_bits(self, count: u32) -> u32 {
        count * self.bits
            + match self.base {
                3 => (8 * count).div_ceil(5),
                5 => (7 * count).div_ceil(3),
                _ => 0,
            }
    }
}

const fn r(base: u32, bits: u32) -> Range {
    Range { bits, base }
}

/// Every range, by their levels: 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96,
/// 128, 160, 192 and 256.
const RANGES: [Range; 21] = [
    r(1, 1),
    r(3, 0),
    r(1, 2),
    r(5, 0),
    r(3, 1),
    r(1, 3),
    r(5, 1),
    r(3, 2),
    r(1, 4),
    r(5, 2),
    r(3, 3),
    r(1, 5),
    r(5, 3),
    r(3, 4),
    r(1, 6),
    r(5, 4),
    r(3, 5),
    r(1, 7),
    r(5, 5),
    r(3, 6),
    r(1, 8),
];

/// The five trits of a trit block's 8 bits `t`.
fn trits(t: u32) -> [u32; 5] {
    let bit = |v: u32, i: u32| (v >> i) & 1;
    let (c, t4, t3);
    if (t >> 2) & 7 == 7 {
        c = ((t >> 5) & 7) << 2 | (t & 3);
        t4 = 2;
        t3 = 2;
    } else {
        c = t & 0x1F;
        if (t >> 5) & 3 == 3 {
            t4 = 2;
            t3 = bit(t, 7);
        } else {
            t4 = bit(t, 7);
            t3 = (t >> 5) & 3;
        }
    }
    let (t2, t1, t0);
    if c & 3 == 3 {
        t2 = 2;
        t1 = bit(c, 4);
        t0 = bit(c, 3) << 1 | (bit(c, 2) & !bit(c, 3) & 1);
    } else if (c >> 2) & 3 == 3 {
        t2 = 2;
        t1 = 2;
        t0 = c & 3;
    } else {
        t2 = bit(c, 4);
        t1 = (c >> 2) & 3;
        t0 = bit(c, 1) << 1 | (bit(c, 0) & !bit(c, 1) & 1);
    }
    [t0, t1, t2, t3, t4]
}

/// The three quints of a quint block's 7 bits `q`.
fn quints(q: u32) -> [u32; 3] {
    let bit = |v: u32, i: u32| (v >> i) & 1;
    if (q >> 1) & 3 == 3 && (q >> 5) & 3 == 0 {
        let q2 = bit(q, 0) << 2 | (bit(q, 4) & !bit(q, 0) & 1) << 1 | (bit(q, 3) & !bit(q, 0) & 1);
        return [4, 4, q2];
    }
    let (q2, c);
    if (q >> 1) & 3 == 3 {
        q2 = 4;
        c = ((q >> 3) & 3) << 3 | (!(q >> 5) & 3) << 1 | bit(q, 0);
    } else {
        q2 = (q >> 5) & 3;
        c = q & 0x1F;
    }
    let (q1, q0) = match c & 7 == 5 {
        true => (4, (c >> 3) & 3),
        false => ((c >> 3) & 3, c & 7),
    };
    [q0, q1, q2]
}

/// `count` values of `range` from the stream `stream` (its bits past the encoding's are zero),
/// each the trit or quint times `2^bits` plus its low bits.
fn ise_decode(stream: u128, range: Range, count: usize, out: &mut [u32]) {
    let n = range.bits;
    let mut at = 0u32;
    let mut take = |width: u32| -> u32 {
        let v = ((stream >> at) as u32) & ((1u32 << width) - 1);
        at += width;
        v
    };
    match range.base {
        3 => {
            for group in 0..count.div_ceil(5) {
                let mut m = [0u32; 5];
                let mut t = 0;
                m[0] = take(n);
                t |= take(2);
                m[1] = take(n);
                t |= take(2) << 2;
                m[2] = take(n);
                t |= take(1) << 4;
                m[3] = take(n);
                t |= take(2) << 5;
                m[4] = take(n);
                t |= take(1) << 7;
                let tr = trits(t);
                for k in 0..5 {
                    if let Some(slot) = out.get_mut(group * 5 + k).filter(|_| group * 5 + k < count)
                    {
                        *slot = tr[k] << n | m[k];
                    }
                }
            }
        }
        5 => {
            for group in 0..count.div_ceil(3) {
                let mut m = [0u32; 3];
                let mut q = 0;
                m[0] = take(n);
                q |= take(3);
                m[1] = take(n);
                q |= take(2) << 3;
                m[2] = take(n);
                q |= take(2) << 5;
                let qu = quints(q);
                for k in 0..3 {
                    if let Some(slot) = out.get_mut(group * 3 + k).filter(|_| group * 3 + k < count)
                    {
                        *slot = qu[k] << n | m[k];
                    }
                }
            }
        }
        _ => {
            for slot in out.iter_mut().take(count) {
                *slot = take(n);
            }
        }
    }
}

/// The `bits` bits of `block` from bit `start`.
fn field(block: u128, start: u32, bits: u32) -> u128 {
    if bits == 0 {
        return 0;
    }
    (block >> start) & (u128::MAX >> (128 - bits))
}

// ---------------------------------------------------------------------------------------------
// Unquantization.

/// A colour endpoint value `v` of `range` unquantized to 0 to 255.
fn unquantize_color(range: Range, v: u32) -> u32 {
    if range.base == 1 {
        // Replicate the bits up to 8.
        let n = range.bits;
        let mut out = 0;
        let mut have = 0;
        while have < 8 {
            let shift = 8 - have;
            out |= if shift >= n {
                v << (shift - n)
            } else {
                v >> (n - shift)
            };
            have += n;
        }
        return out & 0xFF;
    }
    let n = range.bits;
    let d = v >> n;
    let m = v & ((1 << n) - 1);
    let bit = |i: u32| (m >> i) & 1;
    let a = if m & 1 == 1 { 0x1FF } else { 0 };
    let (b, c) = match (range.base, n) {
        (3, 1) => (0, 204),
        (3, 2) => {
            let b = bit(1);
            // b000b0bb0
            (b << 8 | b << 4 | b << 2 | b << 1, 93)
        }
        (3, 3) => {
            let (b, c) = (bit(1), bit(2));
            // cb000cbcb
            (c << 8 | b << 7 | c << 3 | b << 2 | c << 1 | b, 44)
        }
        (3, 4) => {
            let (b, c, d) = (bit(1), bit(2), bit(3));
            // dcb000dcb
            (d << 8 | c << 7 | b << 6 | d << 2 | c << 1 | b, 22)
        }
        (3, 5) => {
            let (b, c, d, e) = (bit(1), bit(2), bit(3), bit(4));
            // edcb000ed
            (e << 8 | d << 7 | c << 6 | b << 5 | e << 1 | d, 11)
        }
        (3, 6) => {
            let (b, c, d, e, f) = (bit(1), bit(2), bit(3), bit(4), bit(5));
            // fedcb000f
            (f << 8 | e << 7 | d << 6 | c << 5 | b << 4 | f, 5)
        }
        (5, 1) => (0, 113),
        (5, 2) => {
            let b = bit(1);
            // b0000bb00
            (b << 8 | b << 3 | b << 2, 54)
        }
        (5, 3) => {
            let (b, c) = (bit(1), bit(2));
            // cb0000cbc
            (c << 8 | b << 7 | c << 2 | b << 1 | c, 26)
        }
        (5, 4) => {
            let (b, c, d) = (bit(1), bit(2), bit(3));
            // dcb0000dc
            (d << 8 | c << 7 | b << 6 | d << 1 | c, 13)
        }
        (5, 5) => {
            let (b, c, d, e) = (bit(1), bit(2), bit(3), bit(4));
            // edcb0000e
            (e << 8 | d << 7 | c << 6 | b << 5 | e, 6)
        }
        _ => unreachable!("colour endpoints take 6 levels or more"),
    };
    let t = (d * c + b) ^ a;
    (a & 0x80) | (t >> 2)
}

/// A weight `v` of `range` unquantized to 0 to 64.
fn unquantize_weight(range: Range, v: u32) -> u32 {
    let t = if range.base == 1 {
        let n = range.bits;
        let mut out = 0;
        let mut have = 0;
        while have < 6 {
            let shift = 6 - have;
            out |= if shift >= n {
                v << (shift - n)
            } else {
                v >> (n - shift)
            };
            have += n;
        }
        out & 0x3F
    } else {
        let n = range.bits;
        let d = v >> n;
        let m = v & ((1 << n) - 1);
        match (range.base, n) {
            (3, 0) => [0, 32, 63][d as usize],
            (5, 0) => [0, 16, 32, 47, 63][d as usize],
            _ => {
                let bit = |i: u32| (m >> i) & 1;
                let a = if m & 1 == 1 { 0x7F } else { 0 };
                let (b, c) = match (range.base, n) {
                    (3, 1) => (0, 50),
                    (3, 2) => {
                        let b = bit(1);
                        // b000b0b
                        (b << 6 | b << 2 | b, 23)
                    }
                    (3, 3) => {
                        let (b, c) = (bit(1), bit(2));
                        // cb000cb
                        (c << 6 | b << 5 | c << 1 | b, 11)
                    }
                    (5, 1) => (0, 28),
                    (5, 2) => {
                        let b = bit(1);
                        // b0000b0
                        (b << 6 | b << 1, 13)
                    }
                    _ => unreachable!("weights take 32 levels or fewer"),
                };
                let t = (d * c + b) ^ a;
                (a & 0x20) | (t >> 2)
            }
        }
    };
    if t > 32 {
        t + 1
    } else {
        t
    }
}

// ---------------------------------------------------------------------------------------------
// Colour endpoints.

/// Moves a bit from `a` (the offset) to `b` (the base): `a` becomes a signed 6-bit offset.
fn bit_transfer_signed(a: i32, b: i32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let mut a = (a >> 1) & 0x3F;
    if a & 0x20 != 0 {
        a -= 0x40;
    }
    (a, b)
}

fn blue_contract(r: i32, g: i32, b: i32, a: i32) -> [i32; 4] {
    [(r + b) >> 1, (g + b) >> 1, b, a]
}

fn clamp8(e: [i32; 4]) -> [u32; 4] {
    e.map(|c| c.clamp(0, 255) as u32)
}

/// A partition's two endpoints, each channel widened to 16 bits, and which channels are
/// logarithmic (an HDR mode's).
#[derive(Clone, Copy)]
struct Ends {
    e: [[u32; 4]; 2],
    lns: [bool; 4],
}

/// The endpoints of colour endpoint mode `cem` of `fmt` from its unquantized values `v`, widened
/// (the module docs); `None` for an HDR mode of an LDR or sRGB format.
fn endpoints(fmt: AstcDecode, cem: u32, v: &[i32; 8]) -> Option<Ends> {
    if let Some([e0, e1]) = ldr_endpoints(cem, v) {
        let widen = |e: [u32; 4]| {
            e.map(|c| match fmt.srgb {
                true => c << 8 | 0x80,
                false => c * 257,
            })
        };
        return Some(Ends {
            e: [widen(e0), widen(e1)],
            lns: [false; 4],
        });
    }
    if !fmt.hdr {
        return None;
    }
    let ([e0, e1], alpha_lns) = hdr::endpoints(cem, v);
    let widen = |e: [i32; 4]| {
        std::array::from_fn(|ch| match ch < 3 || alpha_lns {
            true => (e[ch] as u32) << 4,
            false => e[ch] as u32 * 257,
        })
    };
    Some(Ends {
        e: [widen(e0), widen(e1)],
        lns: [true, true, true, alpha_lns],
    })
}

/// The two 8-bit endpoints of an LDR colour endpoint mode `cem` from its unquantized values `v`;
/// `None` for an HDR mode.
fn ldr_endpoints(cem: u32, v: &[i32; 8]) -> Option<[[u32; 4]; 2]> {
    let u = |i: usize| v[i] as u32;
    Some(match cem {
        0 => [[u(0), u(0), u(0), 255], [u(1), u(1), u(1), 255]],
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xC0);
            let l1 = (l0 + (v[1] & 0x3F)).min(255);
            let (l0, l1) = (l0 as u32, l1 as u32);
            [[l0, l0, l0, 255], [l1, l1, l1, 255]]
        }
        4 => [[u(0), u(0), u(0), u(2)], [u(1), u(1), u(1), u(3)]],
        5 => {
            let (o1, b0) = bit_transfer_signed(v[1], v[0]);
            let (o3, b2) = bit_transfer_signed(v[3], v[2]);
            [
                clamp8([b0, b0, b0, b2]),
                clamp8([b0 + o1, b0 + o1, b0 + o1, b2 + o3]),
            ]
        }
        6 | 10 => {
            let (a0, a1) = match cem {
                10 => (u(4), u(5)),
                _ => (255, 255),
            };
            let scaled = |i: usize| ((v[i] * v[3]) >> 8) as u32;
            [
                [scaled(0), scaled(1), scaled(2), a0],
                [u(0), u(1), u(2), a1],
            ]
        }
        8 | 12 => {
            let (a0, a1) = match cem {
                12 => (v[6], v[7]),
                _ => (255, 255),
            };
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                [
                    clamp8([v[0], v[2], v[4], a0]),
                    clamp8([v[1], v[3], v[5], a1]),
                ]
            } else {
                [
                    clamp8(blue_contract(v[1], v[3], v[5], a1)),
                    clamp8(blue_contract(v[0], v[2], v[4], a0)),
                ]
            }
        }
        9 | 13 => {
            let (o1, b0) = bit_transfer_signed(v[1], v[0]);
            let (o3, b2) = bit_transfer_signed(v[3], v[2]);
            let (o5, b4) = bit_transfer_signed(v[5], v[4]);
            let (o7, b6) = match cem {
                13 => bit_transfer_signed(v[7], v[6]),
                _ => (0, 255),
            };
            if o1 + o3 + o5 >= 0 {
                [
                    clamp8([b0, b2, b4, b6]),
                    clamp8([b0 + o1, b2 + o3, b4 + o5, b6 + o7]),
                ]
            } else {
                [
                    clamp8(blue_contract(b0 + o1, b2 + o3, b4 + o5, b6 + o7)),
                    clamp8(blue_contract(b0, b2, b4, b6)),
                ]
            }
        }
        // 2, 3, 7, 11, 14 and 15: HDR modes.
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------
// Partitions.

fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// The partition function of a block: its partition index `seed`, its partition count and
/// whether the block is small (fewer than 31 texels), worked out once for its texels.
struct Partitions {
    /// The twelve seeds, squared and shifted.
    s: [u32; 12],
    /// The hash.
    rnum: u32,
    count: u32,
    small: bool,
}

impl Partitions {
    fn new(seed: u32, count: u32, texels: u32) -> Partitions {
        let seed = seed + (count - 1) * 1024;
        let rnum = hash52(seed);
        let mut s = [
            rnum & 0xF,
            (rnum >> 4) & 0xF,
            (rnum >> 8) & 0xF,
            (rnum >> 12) & 0xF,
            (rnum >> 16) & 0xF,
            (rnum >> 20) & 0xF,
            (rnum >> 24) & 0xF,
            (rnum >> 28) & 0xF,
            (rnum >> 18) & 0xF,
            (rnum >> 22) & 0xF,
            (rnum >> 26) & 0xF,
            rnum.rotate_left(2) & 0xF,
        ];
        let (sh1, sh2) = match seed & 1 {
            1 => (
                if seed & 2 != 0 { 4 } else { 5 },
                if count == 3 { 6 } else { 5 },
            ),
            _ => (
                if count == 3 { 6 } else { 5 },
                if seed & 2 != 0 { 4 } else { 5 },
            ),
        };
        let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
        for (i, v) in s.iter_mut().enumerate() {
            *v *= *v;
            *v >>= match i {
                0 | 2 | 4 | 6 => sh1,
                1 | 3 | 5 | 7 => sh2,
                _ => sh3,
            };
        }
        Partitions {
            s,
            rnum,
            count,
            small: texels < 31,
        }
    }

    /// The partition of texel `(x, y)` (a 2D block's `z` is 0).
    fn of(&self, x: u32, y: u32) -> usize {
        let (x, y) = match self.small {
            true => (x << 1, y << 1),
            false => (x, y),
        };
        let s = &self.s;
        let rnum = self.rnum;
        let a = (s[0] * x + s[1] * y + (rnum >> 14)) & 0x3F;
        let b = (s[2] * x + s[3] * y + (rnum >> 10)) & 0x3F;
        let c = match self.count < 3 {
            true => 0,
            false => (s[4] * x + s[5] * y + (rnum >> 6)) & 0x3F,
        };
        let d = match self.count < 4 {
            true => 0,
            false => (s[6] * x + s[7] * y + (rnum >> 2)) & 0x3F,
        };
        if a >= b && a >= c && a >= d {
            0
        } else if b >= c && b >= d {
            1
        } else if c >= d {
            2
        } else {
            3
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Blocks.

/// The weights' ranges by a block mode's precision bit and range bits: levels 2, 3, 4, 5, 6 and 8
/// (low precision), then 10, 12, 16, 20, 24 and 32 (high).
const WEIGHT_RANGES: [Range; 12] = [
    r(1, 1),
    r(3, 0),
    r(1, 2),
    r(5, 0),
    r(3, 1),
    r(1, 3),
    r(5, 1),
    r(3, 2),
    r(1, 4),
    r(5, 2),
    r(3, 3),
    r(1, 5),
];

/// A block mode's weight grid and weights: grid across and down, two planes, the weights'
/// range.
struct BlockMode {
    grid: [u32; 2],
    dual: bool,
    range: Range,
}

/// The block mode `mode` (11 bits) describes; `None` for a reserved one.
fn block_mode(mode: u32) -> Option<BlockMode> {
    let bit = |i: u32| (mode >> i) & 1;
    let bits = |i: u32, n: u32| (mode >> i) & ((1 << n) - 1);
    let (range_bits, grid, high, dual);
    if mode & 3 != 0 {
        range_bits = bit(4) | (mode & 3) << 1;
        high = bit(9);
        dual = bit(10);
        let a = bits(5, 2);
        let b = bits(7, 2);
        grid = match bits(2, 2) {
            0 => [b + 4, a + 2],
            1 => [b + 8, a + 2],
            2 => [a + 2, b + 8],
            _ => match bit(8) {
                0 => [a + 2, (b & 1) + 6],
                _ => [(b & 1) + 2, a + 2],
            },
        };
    } else {
        if mode & 0xF == 0 {
            return None;
        }
        range_bits = bit(4) | bits(2, 2) << 1;
        let a = bits(5, 2);
        (grid, high, dual) = match bits(7, 2) {
            0 => ([12, a + 2], bit(9), bit(10)),
            1 => ([a + 2, 12], bit(9), bit(10)),
            2 => ([a + 6, bits(9, 2) + 6], 0, 0),
            _ => match a {
                0 => ([6, 10], bit(9), bit(10)),
                1 => ([10, 6], bit(9), bit(10)),
                _ => return None,
            },
        };
    }
    if range_bits < 2 {
        return None;
    }
    Some(BlockMode {
        grid,
        dual: dual == 1,
        range: WEIGHT_RANGES[(range_bits - 2) as usize + if high == 1 { 6 } else { 0 }],
    })
}

/// Decodes `block` of `fmt` into `out` (its texels' 16-bit interpolation results, row-major), or
/// says why it is an error block.
fn decode_block(fmt: AstcDecode, block: u128, out: &mut [[u16; 4]]) -> Result<(), Illegal> {
    let (bw, bh) = (fmt.width, fmt.height);
    let mode = field(block, 0, 11) as u32;
    if mode & 0x1FF == 0x1FC {
        return void_extent(fmt, block, &mut out[..(bw * bh) as usize]);
    }
    let bm = block_mode(mode).ok_or(Illegal::ReservedMode)?;
    let [gw, gh] = bm.grid;
    if gw > bw || gh > bh {
        return Err(Illegal::GridPastBlock);
    }
    let planes = if bm.dual { 2 } else { 1 };
    let weight_count = gw * gh * planes;
    if weight_count > 64 {
        return Err(Illegal::TooManyWeights);
    }
    let weight_bits = bm.range.ise_bits(weight_count);
    if !(24..=96).contains(&weight_bits) {
        return Err(Illegal::WeightBits);
    }
    let partitions = field(block, 11, 2) as u32 + 1;
    if bm.dual && partitions == 4 {
        return Err(Illegal::DualPlaneFourPartitions);
    }
    let below_weights = 128 - weight_bits;
    let mut cems = [0u32; 4];
    let (color_start, extra, seed);
    if partitions == 1 {
        cems[0] = field(block, 13, 4) as u32;
        (color_start, extra, seed) = (17, 0, 0);
    } else {
        seed = field(block, 13, 10) as u32;
        color_start = 29;
        let low = field(block, 23, 6) as u32;
        if low & 3 == 0 {
            extra = 0;
            cems[..partitions as usize].fill(low >> 2);
        } else {
            // The selector's class, then each partition's class bit and its 2-bit mode, the
            // fields past the first 6 bits just below the weights.
            extra = 3 * partitions - 4;
            let at = below_weights
                .checked_sub(extra)
                .ok_or(Illegal::ConfigPastWeights)?;
            let config = low | (field(block, at, extra) as u32) << 6;
            let base = (config & 3) - 1;
            for (i, cem) in (0..).zip(cems.iter_mut().take(partitions as usize)) {
                let class = base + ((config >> (2 + i)) & 1);
                let m = (config >> (2 + partitions + 2 * i)) & 3;
                *cem = class << 2 | m;
            }
        }
    }
    let color_end = below_weights
        .checked_sub(extra + 2 * (planes - 1))
        .ok_or(Illegal::ColorPastWeights)?;
    // The plane-2 channel, just below the weights and the configuration's high bits.
    let ccs = match bm.dual {
        true => field(block, color_end, 2) as usize,
        false => usize::MAX,
    };
    let value_count: u32 = cems[..partitions as usize]
        .iter()
        .map(|c| 2 * ((c >> 2) + 1))
        .sum();
    if value_count > 18 {
        return Err(Illegal::TooManyValues);
    }
    let color_bits = color_end
        .checked_sub(color_start)
        .ok_or(Illegal::ColorPastWeights)?;
    // The largest range whose encoding of the values fits, of 6 levels or more.
    let color_range = *RANGES
        .iter()
        .rev()
        .find(|range| range.ise_bits(value_count) <= color_bits)
        .filter(|range| range.levels() >= 6)
        .ok_or(Illegal::NoColorRange)?;
    let mut values = [0u32; 18];
    ise_decode(
        field(block, color_start, color_range.ise_bits(value_count)),
        color_range,
        value_count as usize,
        &mut values,
    );
    let mut ends: [Option<Ends>; 4] = [None; 4];
    let mut at = 0;
    for (p, end) in ends.iter_mut().enumerate().take(partitions as usize) {
        let n = (2 * ((cems[p] >> 2) + 1)) as usize;
        let mut v = [0i32; 8];
        for (slot, &raw) in v.iter_mut().zip(&values[at..at + n]) {
            *slot = unquantize_color(color_range, raw) as i32;
        }
        at += n;
        *end = endpoints(fmt, cems[p], &v);
    }
    // The weights, read from bit 127 down.
    let mut weights = [0u32; 64];
    ise_decode(
        field(block.reverse_bits(), 0, weight_bits),
        bm.range,
        weight_count as usize,
        &mut weights,
    );
    for w in weights.iter_mut().take(weight_count as usize) {
        *w = unquantize_weight(bm.range, *w);
    }
    let partition = (partitions > 1).then(|| Partitions::new(seed, partitions, bw * bh));
    // The infill's steps of a texel across and down, in 1/1024ths of the block.
    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    for t in 0..bh {
        let gt = (dt * t * (gh - 1) + 32) >> 6;
        let (jt, ft) = (gt >> 4, gt & 0xF);
        for s in 0..bw {
            let gs = (ds * s * (gw - 1) + 32) >> 6;
            let (js, fs) = (gs >> 4, gs & 0xF);
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            let w00 = 16 + w11 - fs - ft;
            let v0 = js + jt * gw;
            // A tap of weight 0 may lie past the grid, read as 0.
            let tap = |i: u32, plane: u32| -> u32 {
                weights
                    .get((i * planes + plane) as usize)
                    .copied()
                    .unwrap_or(0)
            };
            let infill = |plane: u32| -> u32 {
                let p00 = tap(v0, plane);
                let p01 = if w01 != 0 { tap(v0 + 1, plane) } else { 0 };
                let p10 = if w10 != 0 { tap(v0 + gw, plane) } else { 0 };
                let p11 = if w11 != 0 { tap(v0 + gw + 1, plane) } else { 0 };
                (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4
            };
            let w0 = infill(0);
            let w1 = if bm.dual { infill(1) } else { w0 };
            let p = partition.as_ref().map_or(0, |parts| parts.of(s, t));
            let texel = &mut out[(t * bw + s) as usize];
            let Some(Ends { e: [e0, e1], lns }) = ends[p] else {
                *texel = fmt.error_texel();
                continue;
            };
            for ch in 0..4 {
                let w = if ch == ccs { w1 } else { w0 };
                let c = (e0[ch] * (64 - w) + e1[ch] * w + 32) >> 6;
                texel[ch] = match (fmt.hdr, lns[ch]) {
                    (false, _) => c as u16,
                    (true, true) => hdr::lns_to_half(c),
                    (true, false) => hdr::unorm16_to_half(c),
                };
            }
        }
    }
    Ok(())
}

/// A void-extent block's texels: its constant colour, or why it is an error block. An HDR
/// format's are halves: an HDR block's values themselves, an LDR one's as an LDR channel reads.
fn void_extent(fmt: AstcDecode, block: u128, out: &mut [[u16; 4]]) -> Result<(), Illegal> {
    let hdr_block = (block >> 9) & 1 == 1;
    if hdr_block && !fmt.hdr {
        return Err(Illegal::HdrVoidExtent);
    }
    if (block >> 10) & 3 != 3 {
        return Err(Illegal::VoidExtentReserved);
    }
    let coord = |i: u32| field(block, 12 + 13 * i, 13) as u32;
    let (s0, s1, t0, t1) = (coord(0), coord(1), coord(2), coord(3));
    let all_ones = [s0, s1, t0, t1].iter().all(|&c| c == 0x1FFF);
    if !all_ones && (s0 >= s1 || t0 >= t1) {
        return Err(Illegal::VoidExtentEmpty);
    }
    let color: [u16; 4] = std::array::from_fn(|i| field(block, 64 + 16 * i as u32, 16) as u16);
    let color = match (fmt.hdr, hdr_block) {
        (true, false) => color.map(|c| hdr::unorm16_to_half(u32::from(c))),
        // Metal reads every NaN as the one quiet NaN.
        (true, true) => color.map(|c| match c & 0x7FFF > 0x7C00 {
            true => HDR_ERROR,
            false => c,
        }),
        _ => color,
    };
    out.fill(color);
    Ok(())
}

#[cfg(test)]
mod tests;
