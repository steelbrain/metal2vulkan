//! A YCbCr texture's texels as Metal reads them, converted from its IOSurface's two planes.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1; our own probes over CoreVideo surfaces). A
//! texture of a 4:2:0 two-plane format of `w x h` texels made over an IOSurface's plane 0 reads the
//! surface's luma plane (plane 0, a code `Y` a texel) and its chroma plane (plane 1, a Cb and then
//! a Cr code for each 2 x 2 texels, `⌈w / 2⌉ x ⌈h / 2⌉` of them at the plane's rows, whatever the
//! surface's own chroma layout). A code of `b` bits ([`YCbCrCodes`]) is a byte (8 bits) or the high
//! `b` bits of a little-endian 16-bit word (10 bits), or, packed, one of three 10-bit codes of a
//! little-endian 32-bit word (below). Texel `(x, y)` reads, for `YCBCR8_420_2P`
//! over `420v` and `420f` surfaces (every 8-bit code compared), and for `YCBCR10_420_2P` and
//! `RGB10_420_2P` over `x420`, `xf20`, `x422`, `xf22`, `x444` and `xf44` surfaces (random
//! codes), with no YCbCr matrix, BT.601, BT.709 and BT.2020 (500, 505, 520, 525 and 543, every
//! texel of 28 surfaces):
//!
//! - **Chroma**: the four nearest chroma texels, the nearer column and row weighted 3 and the
//!   farther 1 (`9 / 3 / 3 / 1`): column `⌊x / 2⌋` and the one after it for an odd `x`, the one
//!   before it for an even `x`, each clamped to the plane, and rows the same way. Each sum `S`
//!   rounds to a code of `b` bits, `C = (S + 8) >> 4`.
//! - **4:2:2 and 4:4:4 chroma** ([`YCbCrCodes::subsampling`]): a 4:2:2 format's
//!   chroma plane holds a pair for each 2 x 1 texels, `⌈w / 2⌉ x h` of them, a 4:4:4 one's a pair
//!   for each texel, `w x h`. Along an axis the plane does not halve, texel `(x, y)` reads the pair
//!   at its own column (or row) alone, weighted 4: a 4:2:2 texel the nearer of its row's two pairs
//!   weighted 3 and the farther 1, `C = (S + 2) >> 2`, a 4:4:4 texel its own pair's codes. Measured
//!   for `YCBCR8_422_2P`, `YCBCR8_444_2P`, `YCBCR10_422_2P`, `YCBCR10_444_2P` and their `_sRGB`
//!   twins over their own surfaces (`422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444`,
//!   `xf44`), every texel of random planes under no matrix, BT.601, BT.709 and BT.2020.
//! - **Packed codes** ([`YCbCrCodes::packed`]): the `_PACKED` formats' rows hold three 10-bit codes
//!   to a little-endian 32-bit word, at its bits 0, 10 and 20, in order: luma code `x` of a row in
//!   word `⌊x / 3⌋`, the chroma plane's pair `x` as codes `2 x` (Cb) and `2 x + 1` (Cr). Their
//!   texels read as the unpacked 10-bit formats' of the same codes, filter, range and matrix alike.
//!   Measured for `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED`, `YCBCR10_444_2P_PACKED` and
//!   their `_sRGB` twins over `p420`, `pf20`, `p422`, `pf22`, `p444` and `pf44`, every texel of
//!   random planes under no matrix, BT.601, BT.709 and BT.2020.
//! - **Codes** ([`YCbCrOutput::Codes`], `RGB10_420_2P`, `RGB8_420_2P` and their 4:2:2 and 4:4:4
//!   formats): red `Cr`, green `Y` and blue `Cb`, each `code / (2^b - 1)`, alpha 1: an
//!   `RGB10A2Unorm` or an `RGBA8Unorm` texel of the codes. `RGB8_420_2P` over `420v` and `420f`
//!   reads them so (under BT.709 and BT.2020 alike), and `RGB8_422_2P`, `RGB8_444_2P`,
//!   `RGB10_422_2P` and `RGB10_444_2P` over their own surfaces, after their chroma filter (every
//!   texel of 24 surfaces; and our probes under every matrix).
//! - **Converted** ([`YCbCrOutput::Converted`]): the codes counted as 10-bit codes
//!   (`u = 10 - b`), `Cb' = (Cb << u) - 512`, `Cr' = (Cr << u) - 512`; luma video range (`420v`,
//!   `x4..`) `L = 9576 ((Y << u) - 64)`, full range (`420f`, `xf..`)
//!   `L = 256 ⌊(32768 Y + ⌊m / 2⌋) / m⌋` with `m = 2^b - 1`. Red, green and blue are each
//!   `v / 2048`, where `v = clamp((L + kb Cb' + kr Cr' + 2048) >> 12, 0, 2048)` (an arithmetic
//!   shift) with the channel's `(kb, kr)` of the range and matrix; alpha is 1.
//!   Every value lies on the grid of `1 / 2048`, which a half float holds exactly.
//! - **sRGB** ([`YCbCrOutput::Srgb`], the `_sRGB` twins `YCBCR8_420_2P_sRGB` over `420v` and `420f`
//!   and `YCBCR10_420_2P_sRGB` over `x420`, `xf20`, `x422` and `xf44`, as textures and as views of
//!   `YCBCR8_420_2P` and `YCBCR10_420_2P`; every channel, and every one of the 8193 steps below):
//!   the converted sum through the sRGB curve on the grid of `1 / 8192`. With `T = L + kb Cb' + kr
//!   Cr'` as above, the step `q = clamp((T + 512) >> 10, 0, 8192)` is decoded by the curve's knots
//!   at every 16th step, `K[j]` the step nearest `8192 · srgb(16 j / 8192)` (`srgb(c)` is `c /
//!   12.92` up to `0.04045`, `((c + 0.055) / 1.055)^2.4` above), joined by straight lines: with `i
//!   = min(q >> 4, 511)` and `f = q - 16 i`, the step `(16 K[i] + (K[i + 1] - K[i]) f) / 16`
//!   rounded to nearest even, each channel that step `/ 8192`; alpha is 1. A half float does not
//!   hold every step (above `1/4`).
//!
//! A consumer that holds such a texture as its texels computes them from the planes' bytes by
//! these rules.

use crate::metal::format::ChromaSubsampling;

/// The range of a YCbCr surface's codes, which its pixel format names.
#[cfg(test)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum YCbCrRange {
    /// Video range: luma codes 16 to 235 span black to white (CoreVideo's `420v`).
    Video,
    /// Full range: luma codes 0 to 255 (CoreVideo's `420f`).
    Full,
}

/// The YCbCr matrix of a conversion: the surface's `IOSurfaceYCbCrMatrix` as Metal reads it when
/// it makes the texture (none reads as BT.601).
#[cfg(test)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum YCbCrMatrix {
    /// ITU-R BT.601 (`ITU_R_601_4`, or no matrix).
    Bt601,
    /// ITU-R BT.709 (`ITU_R_709_2`).
    Bt709,
    /// ITU-R BT.2020 (`ITU_R_2020`).
    Bt2020,
}

/// What a YCbCr texture's texels read of its codes (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YCbCrOutput {
    /// The color converted by the surface's range and matrix, each channel a multiple of
    /// `1 / 2048`.
    Converted,
    /// The codes themselves, red Cr, green Y and blue Cb, each `code / (2^b - 1)`.
    Codes,
    /// The converted color through the sRGB curve, each channel a multiple of `1 / 8192`.
    Srgb,
}

/// How a two-plane format's planes hold its codes, and what its texels read of them: the layouts
/// Metal's conversion is modelled for (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct YCbCrCodes {
    /// The bits of each code: 8 or 10.
    pub bits: u32,
    /// The bytes of the unit a plane's row holds its codes in, one after the other: 1, a byte of
    /// one code; 2, a little-endian 16-bit word of one code in its high
    /// [`bits`](Self::bits) bits; or 4, a little-endian 32-bit word of three 10-bit codes at its
    /// bits 0, 10 and 20 ([`packed`](Self::packed)).
    pub bytes: u32,
    /// What a texel reads.
    pub output: YCbCrOutput,
    /// How many luma texels share a chroma pair, which picks the chroma filter (the module
    /// docs).
    pub subsampling: ChromaSubsampling,
    /// Whether the codes are packed (`_PACKED`), three to a 4-byte unit.
    pub packed: bool,
}

impl YCbCrCodes {
    /// The codes of a two-plane format of `format`'s layout, `None` for a layout the conversion
    /// does not model: the 4:2:0, 4:2:2 and 4:4:4 formats of unpacked codes, 8- and 10-bit ones read
    /// converted (`YCBCR8_420_2P`, `YCBCR10_444_2P`, ...), through the sRGB curve (their
    /// `_sRGB` twins) or as codes (`RGB8_420_2P`, `RGB10_422_2P`, ...), and the packed 10-bit ones
    /// read converted or through the sRGB curve (`YCBCR10_420_2P_PACKED`,
    /// `YCBCR10_444_2P_PACKED_sRGB`, ...), as Metal reads them (the module docs).
    pub fn of(format: crate::metal::format::YCbCrFormat) -> Option<YCbCrCodes> {
        use crate::metal::format::YCbCrReads;
        let output = match (format.reads, format.bits, format.packed) {
            (YCbCrReads::Converted, 8 | 10, false) | (YCbCrReads::Converted, 10, true) => {
                YCbCrOutput::Converted
            }
            (YCbCrReads::Srgb, 8 | 10, false) | (YCbCrReads::Srgb, 10, true) => YCbCrOutput::Srgb,
            (YCbCrReads::Codes, 8 | 10, false) => YCbCrOutput::Codes,
            _ => return None,
        };
        Some(YCbCrCodes {
            bits: format.bits,
            bytes: match format.packed {
                true => 4,
                false => format.bits.div_ceil(8),
            },
            output,
            subsampling: format.subsampling,
            packed: format.packed,
        })
    }

    /// The codes a unit holds: 3 packed, 1 otherwise.
    #[cfg(test)]
    fn per_unit(&self) -> u32 {
        match self.packed {
            true => 3,
            false => 1,
        }
    }

    /// The units that hold `n` codes from a row's start: `⌈n / 3⌉` packed, `n` otherwise. A
    /// plane's row of `n` codes is that many units of [`bytes`](Self::bytes) bytes.
    #[cfg(test)]
    pub(crate) fn units(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.per_unit()))
    }
}

/// The `(kb, kr)` of red, green and blue for `range` and `matrix`, in the units of the module
/// docs' conversion (measured, the module docs).
#[cfg(test)]
pub(crate) fn coefficients(range: YCbCrRange, matrix: YCbCrMatrix) -> [(i32, i32); 3] {
    match (range, matrix) {
        (YCbCrRange::Video, YCbCrMatrix::Bt601) => [(-9, 13123), (-3218, -6686), (16591, 9)],
        (YCbCrRange::Video, YCbCrMatrix::Bt709) => [(-1, 14743), (-1753, -4383), (17373, 1)],
        (YCbCrRange::Full, YCbCrMatrix::Bt601) => [(-8, 11483), (-2816, -5850), (14518, 8)],
        (YCbCrRange::Full, YCbCrMatrix::Bt709) => [(-1, 12900), (-1534, -3835), (15201, 1)],
        (YCbCrRange::Video, YCbCrMatrix::Bt2020) => [(0, 13806), (-1540, -5349), (17614, -1)],
        (YCbCrRange::Full, YCbCrMatrix::Bt2020) => [(0, 12080), (-1348, -4680), (15412, -1)],
    }
}

/// Where a texture's two planes lie in its surface's bytes, and how they convert.
#[cfg(test)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct YCbCrPlanes {
    /// The texture's width and height, in luma texels.
    pub size: [u32; 2],
    /// The luma plane's first byte, and the bytes from one of its rows to the next.
    pub luma: [u32; 2],
    /// The chroma plane's first byte, and the bytes from one of its rows to the next.
    pub chroma: [u32; 2],
    pub range: YCbCrRange,
    pub matrix: YCbCrMatrix,
    /// How the planes hold the codes, and what a texel reads.
    pub codes: YCbCrCodes,
}

#[cfg(test)]
impl YCbCrPlanes {
    /// The planes of a texture of `size` texels whose luma plane starts at byte `luma[0]` with
    /// rows `luma[1]` bytes apart, and whose chroma plane at `chroma[0]` with rows `chroma[1]`
    /// apart, holding `codes`.
    pub(crate) fn new(
        size: [u32; 2],
        luma: [u32; 2],
        chroma: [u32; 2],
        (range, matrix): (YCbCrRange, YCbCrMatrix),
        codes: YCbCrCodes,
    ) -> YCbCrPlanes {
        YCbCrPlanes {
            size,
            luma,
            chroma,
            range,
            matrix,
            codes,
        }
    }

    /// The chroma plane's texels across and down: `⌈w / 2⌉ x ⌈h / 2⌉` for 4:2:0 codes,
    /// `⌈w / 2⌉ x h` for 4:2:2 ones and `w x h` for 4:4:4 ones.
    pub(crate) fn chroma_size(&self) -> [u32; 2] {
        let halved = self.codes.subsampling.halved();
        [0, 1].map(|axis| match halved[axis] {
            true => self.size[axis].div_ceil(2),
            false => self.size[axis],
        })
    }
}

/// The luma term of the conversion for the code `y` of `bits` bits (the module docs).
#[cfg(test)]
fn luma_term(range: YCbCrRange, bits: u32, y: u32) -> i32 {
    let max = (1u32 << bits) - 1;
    match range {
        YCbCrRange::Video => 9576 * (((y << (10 - bits)) as i32) - 64),
        YCbCrRange::Full => 256 * ((y * 32768 + max / 2) / max) as i32,
    }
}

/// The sRGB curve's knots on the grid of `1 / 8192` (the module docs): `K[j]`, the step nearest
/// `8192 · srgb(16 j / 8192)`, for `j` from 0 to 512 (`the_srgb_knots_are_the_curves`).
#[cfg(test)]
const SRGB_KNOTS: [u16; 513] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31,
    33, 34, 36, 37, 39, 41, 42, 44, 46, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 68, 70, 72, 74, 77,
    79, 82, 84, 87, 89, 92, 95, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133,
    137, 140, 144, 147, 151, 154, 158, 161, 165, 169, 173, 176, 180, 184, 188, 192, 197, 201, 205,
    209, 213, 218, 222, 227, 231, 236, 240, 245, 250, 255, 259, 264, 269, 274, 279, 284, 289, 295,
    300, 305, 311, 316, 322, 327, 333, 338, 344, 350, 356, 361, 367, 373, 379, 385, 392, 398, 404,
    410, 417, 423, 430, 436, 443, 450, 456, 463, 470, 477, 484, 491, 498, 505, 512, 519, 527, 534,
    542, 549, 557, 564, 572, 580, 587, 595, 603, 611, 619, 627, 635, 644, 652, 660, 669, 677, 686,
    694, 703, 712, 720, 729, 738, 747, 756, 765, 774, 784, 793, 802, 812, 821, 831, 840, 850, 860,
    870, 879, 889, 899, 909, 920, 930, 940, 950, 961, 971, 982, 992, 1003, 1014, 1025, 1035, 1046,
    1057, 1068, 1080, 1091, 1102, 1113, 1125, 1136, 1148, 1159, 1171, 1183, 1194, 1206, 1218, 1230,
    1242, 1255, 1267, 1279, 1291, 1304, 1316, 1329, 1342, 1354, 1367, 1380, 1393, 1406, 1419, 1432,
    1445, 1458, 1472, 1485, 1499, 1512, 1526, 1539, 1553, 1567, 1581, 1595, 1609, 1623, 1637, 1652,
    1666, 1680, 1695, 1709, 1724, 1739, 1753, 1768, 1783, 1798, 1813, 1828, 1844, 1859, 1874, 1890,
    1905, 1921, 1936, 1952, 1968, 1984, 2000, 2016, 2032, 2048, 2064, 2081, 2097, 2114, 2130, 2147,
    2163, 2180, 2197, 2214, 2231, 2248, 2265, 2282, 2300, 2317, 2335, 2352, 2370, 2387, 2405, 2423,
    2441, 2459, 2477, 2495, 2513, 2532, 2550, 2569, 2587, 2606, 2625, 2643, 2662, 2681, 2700, 2719,
    2738, 2758, 2777, 2796, 2816, 2835, 2855, 2875, 2895, 2914, 2934, 2954, 2975, 2995, 3015, 3035,
    3056, 3076, 3097, 3118, 3138, 3159, 3180, 3201, 3222, 3243, 3265, 3286, 3307, 3329, 3350, 3372,
    3394, 3416, 3437, 3459, 3481, 3504, 3526, 3548, 3570, 3593, 3615, 3638, 3661, 3684, 3706, 3729,
    3752, 3776, 3799, 3822, 3845, 3869, 3892, 3916, 3940, 3964, 3987, 4011, 4035, 4060, 4084, 4108,
    4132, 4157, 4181, 4206, 4231, 4256, 4280, 4305, 4331, 4356, 4381, 4406, 4432, 4457, 4483, 4508,
    4534, 4560, 4586, 4612, 4638, 4664, 4690, 4717, 4743, 4769, 4796, 4823, 4849, 4876, 4903, 4930,
    4957, 4985, 5012, 5039, 5067, 5094, 5122, 5150, 5177, 5205, 5233, 5261, 5290, 5318, 5346, 5375,
    5403, 5432, 5460, 5489, 5518, 5547, 5576, 5605, 5634, 5664, 5693, 5722, 5752, 5782, 5811, 5841,
    5871, 5901, 5931, 5961, 5992, 6022, 6053, 6083, 6114, 6145, 6175, 6206, 6237, 6268, 6300, 6331,
    6362, 6394, 6425, 6457, 6489, 6520, 6552, 6584, 6616, 6649, 6681, 6713, 6746, 6778, 6811, 6844,
    6876, 6909, 6942, 6975, 7009, 7042, 7075, 7109, 7142, 7176, 7210, 7243, 7277, 7311, 7345, 7380,
    7414, 7448, 7483, 7517, 7552, 7587, 7622, 7657, 7692, 7727, 7762, 7797, 7833, 7868, 7904, 7940,
    7975, 8011, 8047, 8083, 8119, 8156, 8192,
];

/// The step of `1 / 8192` Metal reads for the converted sum's step `q` (at most 8192) of an sRGB
/// format (the module docs): the knots joined by straight lines, rounded to nearest even.
#[cfg(test)]
fn srgb_step(q: u32) -> u16 {
    let i = (q >> 4).min(511) as usize;
    let f = q - 16 * i as u32;
    let (a, b) = (u32::from(SRGB_KNOTS[i]), u32::from(SRGB_KNOTS[i + 1]));
    let n = 16 * a + (b - a) * f;
    ((n + 7 + ((n >> 4) & 1)) >> 4) as u16
}

/// Metal's red, green and blue of a texel of luma code `y` whose chroma rounded to the 8-bit
/// codes `cb` and `cr` (the module docs), each `v` of the value `v / 2048`: [`ycbcr_read`] of
/// 8-bit codes read converted.
#[cfg(test)]
pub(crate) fn ycbcr_texel(
    range: YCbCrRange,
    matrix: YCbCrMatrix,
    y: u8,
    cb: u8,
    cr: u8,
) -> [u16; 3] {
    let codes = YCbCrCodes {
        bits: 8,
        bytes: 1,
        output: YCbCrOutput::Converted,
        subsampling: ChromaSubsampling::Chroma420,
        packed: false,
    };
    ycbcr_read(codes, (range, matrix), [y, cb, cr].map(u16::from))
}

/// Metal's red, green and blue of a texel of `codes` whose luma code is `y` and whose chroma
/// rounded to the codes `cb` and `cr` (the module docs): each `v` of the value `v / 2048` for
/// codes read converted, by `range` and `matrix`, and of `v / 8192` for codes read through the
/// sRGB curve; the codes `cr`, `y` and `cb` of `code / (2^bits - 1)` for codes read as codes.
#[cfg(test)]
pub(crate) fn ycbcr_read(
    codes: YCbCrCodes,
    (range, matrix): (YCbCrRange, YCbCrMatrix),
    [y, cb, cr]: [u16; 3],
) -> [u16; 3] {
    // The converted sums `T = L + kb Cb' + kr Cr'`.
    let sums = || {
        let l = luma_term(range, codes.bits, u32::from(y));
        let up = 10 - codes.bits;
        let (cb, cr) = (i32::from(cb) << up, i32::from(cr) << up);
        let (cb, cr) = (cb - 512, cr - 512);
        coefficients(range, matrix).map(|(kb, kr)| l + kb * cb + kr * cr)
    };
    match codes.output {
        YCbCrOutput::Codes => [cr, y, cb],
        YCbCrOutput::Converted => sums().map(|t| ((t + 2048) >> 12).clamp(0, 2048) as u16),
        YCbCrOutput::Srgb => sums().map(|t| srgb_step(((t + 512) >> 10).clamp(0, 8192) as u32)),
    }
}

#[cfg(test)]
mod tests;
