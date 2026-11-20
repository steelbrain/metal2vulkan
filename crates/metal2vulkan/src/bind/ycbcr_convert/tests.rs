//! The YCbCr conversion's CPU model, against the conversion Metal's reads were measured to follow
//! and against Metal's own reads of our probes' surfaces.

use super::{
    srgb_step, ycbcr_read, ycbcr_texel, YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrPlanes,
    YCbCrRange, SRGB_KNOTS,
};
use crate::metal::format::{ycbcr_format, ChromaSubsampling};
use crate::metal::PixelFormat;

/// The codes of `pf`'s layout.
fn codes_of(pf: PixelFormat) -> YCbCrCodes {
    YCbCrCodes::of(ycbcr_format(pf).expect("a two-plane format")).expect("converted")
}

/// The 4:2:0 formats whose layouts the model converts.
const HELD: [PixelFormat; 6] = [
    PixelFormat::YCBCR84202P,
    PixelFormat::YCBCR104202P,
    PixelFormat::RGB104202P,
    PixelFormat::YCBCR84202PSRGB,
    PixelFormat::YCBCR104202PSRGB,
    PixelFormat::RGB84202P,
];

/// The 4:2:2 and 4:4:4 formats whose layouts the model converts, each [`HELD`]'s layout at its
/// place with that chroma (the codes).
const HELD_SUBSAMPLED: [[PixelFormat; 6]; 2] = [
    [
        PixelFormat::YCBCR84222P,
        PixelFormat::YCBCR104222P,
        PixelFormat::RGB104222P,
        PixelFormat::YCBCR84222PSRGB,
        PixelFormat::YCBCR104222PSRGB,
        PixelFormat::RGB84222P,
    ],
    [
        PixelFormat::YCBCR84442P,
        PixelFormat::YCBCR104442P,
        PixelFormat::RGB104442P,
        PixelFormat::YCBCR84442PSRGB,
        PixelFormat::YCBCR104442PSRGB,
        PixelFormat::RGB84442P,
    ],
];

/// The packed formats whose layouts the model converts: 4:2:0, 4:2:2 and 4:4:4, converted and
/// their `_sRGB` twins.
const HELD_PACKED: [PixelFormat; 6] = [
    PixelFormat::YCBCR104202PPACKED,
    PixelFormat::YCBCR104222PPACKED,
    PixelFormat::YCBCR104442PPACKED,
    PixelFormat::YCBCR104202PPACKEDSRGB,
    PixelFormat::YCBCR104222PPACKEDSRGB,
    PixelFormat::YCBCR104442PPACKEDSRGB,
];

/// The packed layouts the model converts.
fn packed_layouts() -> [YCbCrCodes; 6] {
    HELD_PACKED.map(codes_of)
}

/// The 4:2:0 layouts the model converts: `YCBCR8_420_2P`'s, `YCBCR10_420_2P`'s,
/// `RGB10_420_2P`'s, the two `_sRGB` twins', and `RGB8_420_2P`'s.
fn layouts() -> [YCbCrCodes; 6] {
    HELD.map(codes_of)
}

/// The 4:2:2 and 4:4:4 layouts the model converts.
fn subsampled_layouts() -> impl Iterator<Item = YCbCrCodes> {
    HELD_SUBSAMPLED.into_iter().flatten().map(codes_of)
}

/// The conversion models the 4:2:0, 4:2:2 and 4:4:4 layouts of unpacked 8- and 10-bit codes
/// read converted, through the sRGB curve or as codes, and of packed 10-bit codes read converted
/// or through the sRGB curve, and no other two-plane format's ([`YCbCrCodes::of`]).
#[test]
fn the_model_converts_the_measured_layouts_alone() {
    let held: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ycbcr_format(pf).and_then(YCbCrCodes::of).is_some())
        .collect();
    let listed: Vec<PixelFormat> = HELD
        .into_iter()
        .chain(HELD_SUBSAMPLED.into_iter().flatten())
        .chain(HELD_PACKED)
        .collect();
    assert_eq!(held.len(), listed.len());
    assert!(listed.iter().all(|pf| held.contains(pf)));
    let [eight, ten, codes, eight_srgb, ten_srgb, eight_codes] = layouts();
    for (formats, subsampling) in HELD_SUBSAMPLED
        .iter()
        .zip([ChromaSubsampling::Chroma422, ChromaSubsampling::Chroma444])
    {
        for (&pf, own) in formats.iter().zip(layouts()) {
            let want = YCbCrCodes { subsampling, ..own };
            assert_eq!(codes_of(pf), want, "{pf:?}");
        }
    }
    assert!(layouts()
        .iter()
        .all(|c| c.subsampling == ChromaSubsampling::Chroma420));
    assert_eq!((eight.bits, eight.bytes), (8, 1));
    assert_eq!((ten.bits, ten.bytes), (10, 2));
    assert_eq!(
        (codes.bits, codes.bytes, codes.output),
        (10, 2, YCbCrOutput::Codes)
    );
    assert_eq!(
        (eight_srgb.bits, eight_srgb.bytes, eight_srgb.output),
        (8, 1, YCbCrOutput::Srgb)
    );
    assert_eq!(
        (ten_srgb.bits, ten_srgb.bytes, ten_srgb.output),
        (10, 2, YCbCrOutput::Srgb)
    );
    assert_eq!(
        (eight_codes.bits, eight_codes.bytes, eight_codes.output),
        (8, 1, YCbCrOutput::Codes)
    );
    assert!(layouts()
        .into_iter()
        .chain(subsampled_layouts())
        .all(|c| !c.packed && c.units(7) == 7));
    // The packed layouts: three 10-bit codes to a 4-byte unit, `ten`'s and `ten_srgb`'s at
    // each subsampling.
    let subsamplings = [
        ChromaSubsampling::Chroma420,
        ChromaSubsampling::Chroma422,
        ChromaSubsampling::Chroma444,
    ];
    for (i, (pf, codes)) in HELD_PACKED.into_iter().zip(packed_layouts()).enumerate() {
        let own = if i < 3 { ten } else { ten_srgb };
        let want = YCbCrCodes {
            bytes: 4,
            packed: true,
            subsampling: subsamplings[i % 3],
            ..own
        };
        assert_eq!(codes, want, "{pf:?}");
        assert_eq!([7, 9, 1, 0].map(|n| codes.units(n)), [3, 3, 1, 0]);
    }
}

/// One surface: its planes' bytes and where they lie.
struct Surface {
    bytes: Vec<u8>,
    planes: YCbCrPlanes,
}

impl Surface {
    /// A surface of `size` texels whose luma plane starts at byte `at` with rows `rows[0]` apart,
    /// and whose chroma plane follows it at the next multiple of 16 with rows `rows[1]` apart,
    /// holding `codes`; `luma` and `chroma` give each plane's codes by texel (a 16-bit word's low
    /// bits `0b10_1101` past its code), every other byte `0xA5` (a packed word's bits past its
    /// codes, and its codes past a row's, among them).
    fn new(
        size: [u32; 2],
        at: u32,
        rows: [u32; 2],
        (conversion, codes): ((YCbCrRange, YCbCrMatrix), YCbCrCodes),
        luma: &dyn Fn(u32, u32) -> u16,
        chroma: &dyn Fn(u32, u32) -> [u16; 2],
    ) -> Surface {
        let [w, h] = size;
        let chroma_at = (at + rows[0] * h).next_multiple_of(16);
        let planes = YCbCrPlanes::new(size, [at, rows[0]], [chroma_at, rows[1]], conversion, codes);
        let [cw, ch] = planes.chroma_size();
        let end = chroma_at + rows[1] * ch;
        let mut bytes = vec![0xA5; end.next_multiple_of(4) as usize];
        let mut put = |row: u32, index: u32, code: u16| {
            let (at, shift) = place(codes, row, index);
            let at = at as usize;
            match (codes.packed, codes.bytes) {
                (true, _) => {
                    let mut word = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                    word &= !(0x3FF << shift);
                    word |= u32::from(code) << shift;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
                (false, 1) => bytes[at] = code as u8,
                (false, _) => {
                    let word = (code << shift) | (0b10_1101 & ((1 << shift) - 1));
                    bytes[at..at + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
        };
        for y in 0..h {
            for x in 0..w {
                put(at + y * rows[0], x, luma(x, y));
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                let [cb, cr] = chroma(x, y);
                let row = chroma_at + y * rows[1];
                put(row, 2 * x, cb);
                put(row, 2 * x + 1, cr);
            }
        }
        Surface { bytes, planes }
    }

    /// A surface of `size` texels over `planes`, the luma plane's bytes with rows `rows[0]` apart
    /// and then the chroma plane's with rows `rows[1]` apart, as a probe fills a surface's planes.
    fn of_planes(
        size: [u32; 2],
        (luma, chroma): (&[u8], &[u8]),
        rows: [u32; 2],
        conversion: (YCbCrRange, YCbCrMatrix),
        codes: YCbCrCodes,
    ) -> Surface {
        let mut bytes = [luma, chroma].concat();
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        let chroma_at = luma.len() as u32;
        assert_eq!(chroma_at % 16, 0);
        let planes = YCbCrPlanes::new(size, [0, rows[0]], [chroma_at, rows[1]], conversion, codes);
        Surface { bytes, planes }
    }

    /// Code `index` of the row whose first byte is at `row`.
    fn code(&self, row: u32, index: u32) -> u32 {
        let codes = self.planes.codes;
        let (at, shift) = place(codes, row, index);
        let at = at as usize;
        let unit = &self.bytes[at..at + codes.bytes as usize];
        let word = unit
            .iter()
            .rev()
            .fold(0u32, |word, &b| (word << 8) | u32::from(b));
        (word >> shift) & ((1 << codes.bits) - 1)
    }

    /// The texels every texel reads, row by row, by the module docs' rule.
    fn modelled(&self) -> Vec<[u16; 3]> {
        let p = &self.planes;
        let [w, h] = p.size;
        let [cw, ch] = p.chroma_size();
        let [across, down] = p.codes.subsampling.halved();
        // A halved axis's two nearest chroma texels, weighted 3 and 1; a direct one's own.
        let taps = |q: u32, n: u32, halved: bool| -> [(u32, u32); 2] {
            let c = q / 2;
            let clamp = |i: i64| i.clamp(0, i64::from(n) - 1) as u32;
            if !halved {
                [(q, 4), (q, 0)]
            } else if q % 2 == 1 {
                [(clamp(i64::from(c)), 3), (clamp(i64::from(c) + 1), 1)]
            } else {
                [(clamp(i64::from(c) - 1), 1), (clamp(i64::from(c)), 3)]
            }
        };
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let mut sums = [0u32; 2];
                for (cy, wy) in taps(y, ch, down) {
                    for (cx, wx) in taps(x, cw, across) {
                        let row = p.chroma[0] + cy * p.chroma[1];
                        for (k, sum) in sums.iter_mut().enumerate() {
                            *sum += wx * wy * self.code(row, 2 * cx + k as u32);
                        }
                    }
                }
                let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
                let luma = self.code(p.luma[0] + y * p.luma[1], x) as u16;
                out.push(ycbcr_read(p.codes, (p.range, p.matrix), [luma, cb, cr]));
            }
        }
        out
    }
}

/// Where code `index` of the row whose first byte is at `row` lies: its unit's first byte, and
/// the shift that takes the unit's little-endian bytes to it (the module docs).
fn place(codes: YCbCrCodes, row: u32, index: u32) -> (u32, u32) {
    match codes.packed {
        true => (row + 4 * (index / 3), 10 * (index % 3)),
        false => (row + codes.bytes * index, 8 * codes.bytes - codes.bits),
    }
}

/// The ranges and matrices the model converts.
const CONVERSIONS: [(YCbCrRange, YCbCrMatrix); 6] = [
    (YCbCrRange::Video, YCbCrMatrix::Bt601),
    (YCbCrRange::Video, YCbCrMatrix::Bt709),
    (YCbCrRange::Full, YCbCrMatrix::Bt601),
    (YCbCrRange::Full, YCbCrMatrix::Bt709),
    (YCbCrRange::Video, YCbCrMatrix::Bt2020),
    (YCbCrRange::Full, YCbCrMatrix::Bt2020),
];

/// Our probes' generator of random bytes (`ycc1.swift`, `ycc2.swift`; `ycf3.swift` starts it
/// at `seed · 0x9E3779B97F4A7C15 | 1`).
struct Random(u64);

impl Random {
    /// `ycf3.swift`'s generator for `seed`.
    fn ycf3(seed: u64) -> Random {
        Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }

    /// `n` bytes.
    fn bytes(&mut self, n: u32) -> Vec<u8> {
        (0..n).map(|_| self.next()).collect()
    }

    /// The bytes of a `w x h` surface's luma plane and then its chroma plane, packed, as the
    /// probes fill a surface.
    fn planes(&mut self, [w, h]: [u32; 2]) -> (Vec<u8>, Vec<u8>) {
        let luma = self.bytes(w * h);
        let chroma = self.bytes(2 * w.div_ceil(2) * h.div_ceil(2));
        (luma, chroma)
    }
}

/// A surface of `size` holding `planes` (packed 8-bit codes), at `at` with `rows`.
fn surface_of(
    size: [u32; 2],
    at: u32,
    rows: [u32; 2],
    conversion: (YCbCrRange, YCbCrMatrix),
    (luma, chroma): &(Vec<u8>, Vec<u8>),
) -> Surface {
    let [w, _] = size;
    let cw = w.div_ceil(2);
    Surface::new(
        size,
        at,
        rows,
        (conversion, codes_of(PixelFormat::YCBCR84202P)),
        &|x, y| u16::from(luma[(y * w + x) as usize]),
        &|x, y| {
            let i = (2 * (y * cw + x)) as usize;
            [chroma[i], chroma[i + 1]].map(u16::from)
        },
    )
}

/// Metal's `read()` of a 40 x 24 `YCBCR8_420_2P` texture over a CoreVideo `420v` surface with no
/// YCbCr matrix, its planes filled from our probes' generator at seed 5 (an
/// Apple M4 under macOS 27.0.1): the FNV-1a digest of each texel's red, green and blue `v` as
/// little-endian 16-bit words, and the first texels.
const METAL_READ_DIGEST: u32 = 0x26e2_da15;
const METAL_FIRST_TEXELS: [[u16; 3]; 4] = [
    [0, 869, 493],
    [114, 1948, 1237],
    [373, 2048, 957],
    [239, 2048, 333],
];

fn digest(texels: &[[u16; 3]]) -> u32 {
    texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|v| v.to_le_bytes()))
        .fold(2_166_136_261u32, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(16_777_619)
        })
}

/// The model reads one surface exactly as Metal read it.
#[test]
fn the_model_reads_a_surface_as_metal_does() {
    let size = [40, 24];
    let planes = Random(5).planes(size);
    let surface = surface_of(size, 0, [64, 64], CONVERSIONS[0], &planes);
    let got = surface.modelled();
    assert_eq!(got[..4], METAL_FIRST_TEXELS);
    assert_eq!(digest(&got), METAL_READ_DIGEST);
}

/// Metal's `read()` of a two-plane texture over a CoreVideo surface whose planes `ycf3.swift`
/// filled at a seed.
struct MetalRead {
    pf: PixelFormat,
    size: [u32; 2],
    /// The luma plane's row bytes and rows, then the chroma plane's.
    rows: [u32; 4],
    /// The range and the matrix (no matrix reads as BT.601).
    conversion: (YCbCrRange, YCbCrMatrix),
    seed: u64,
    /// The digest of each texel's `v` (or code), and the first texels.
    digest: u32,
    first: [[u16; 3]; 4],
}

/// Forty of the probes' reads (the BT.2020 ones, `RGB8_420_2P`'s, the 4:2:2 and 4:4:4 ones, their
/// code twins' and the packed ones, each an Apple M4 under macOS 27.0.1). A 4:2:0 format over a
/// 4:2:2 or 4:4:4 surface reads its chroma plane as a 4:2:0 one at its rows, and a 4:2:2 one over a
/// 4:4:4 surface its pairs at their places in the rows.
const METAL_PROBE_READS: [MetalRead; 40] = [
    // `YCBCR10_420_2P_PACKED` over `p420`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104202PPACKED,
        size: [33, 17],
        rows: [64, 17, 64, 9],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 903,
        digest: 0x05da_ecf2,
        first: [
            [878, 1687, 2048],
            [932, 1769, 1715],
            [309, 1202, 138],
            [0, 935, 0],
        ],
    },
    // `YCBCR10_420_2P_PACKED` over `pf20`, no matrix.
    MetalRead {
        pf: PixelFormat::YCBCR104202PPACKED,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 902,
        digest: 0xb93e_7a6f,
        first: [
            [1878, 66, 1967],
            [1472, 0, 1467],
            [2048, 420, 1959],
            [1766, 250, 1263],
        ],
    },
    // `YCBCR10_420_2P_PACKED` over `p444`, no matrix: a 4:2:0 read of a 4:4:4 plane, at its rows.
    MetalRead {
        pf: PixelFormat::YCBCR104202PPACKED,
        size: [64, 64],
        rows: [128, 64, 256, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 908,
        digest: 0x476b_106f,
        first: [
            [0, 401, 0],
            [916, 1875, 1089],
            [616, 2048, 630],
            [346, 1158, 0],
        ],
    },
    // `YCBCR10_422_2P_PACKED` over `pf22`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104222PPACKED,
        size: [33, 17],
        rows: [64, 17, 64, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 946,
        digest: 0xa0d3_3ea6,
        first: [
            [807, 179, 95],
            [1355, 989, 1189],
            [49, 206, 974],
            [958, 1783, 2034],
        ],
    },
    // `YCBCR10_422_2P_PACKED` over `p422`, BT.601, 7 x 5.
    MetalRead {
        pf: PixelFormat::YCBCR104222PPACKED,
        size: [7, 5],
        rows: [64, 5, 64, 5],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 947,
        digest: 0x4e0a_7b31,
        first: [
            [0, 1374, 1608],
            [0, 502, 838],
            [0, 260, 806],
            [1137, 2048, 2048],
        ],
    },
    // `YCBCR10_444_2P_PACKED` over `p444`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104442PPACKED,
        size: [33, 17],
        rows: [64, 17, 128, 17],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 987,
        digest: 0x50b6_4b53,
        first: [
            [0, 912, 1978],
            [2048, 301, 0],
            [1723, 1748, 2048],
            [1917, 333, 152],
        ],
    },
    // `YCBCR10_420_2P_PACKED_sRGB` over `pf20`, BT.601, 16 x 8.
    MetalRead {
        pf: PixelFormat::YCBCR104202PPACKEDSRGB,
        size: [16, 8],
        rows: [64, 8, 64, 4],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 1032,
        digest: 0xc75e_e3f3,
        first: [
            [124, 8192, 3375],
            [633, 5578, 2375],
            [8192, 5460, 4825],
            [815, 0, 0],
        ],
    },
    // `YCBCR10_422_2P_PACKED_sRGB` over `p422`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104222PPACKEDSRGB,
        size: [33, 17],
        rows: [64, 17, 64, 17],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 1071,
        digest: 0xbf96_7d37,
        first: [
            [1491, 8192, 3726],
            [0, 12, 0],
            [1019, 159, 0],
            [8192, 2110, 2363],
        ],
    },
    // `YCBCR10_444_2P_PACKED_sRGB` over `pf44`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104442PPACKEDSRGB,
        size: [33, 17],
        rows: [64, 17, 128, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 1114,
        digest: 0xdee8_b202,
        first: [
            [1221, 430, 0],
            [6417, 4688, 8192],
            [8192, 4011, 8192],
            [1055, 5, 3138],
        ],
    },
    // `YCBCR10_444_2P_PACKED_sRGB` over `p444`, BT.601, 7 x 5.
    MetalRead {
        pf: PixelFormat::YCBCR104442PPACKEDSRGB,
        size: [7, 5],
        rows: [64, 5, 64, 5],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 1115,
        digest: 0x4810_e154,
        first: [
            [658, 5933, 0],
            [0, 3471, 1608],
            [8192, 3124, 4468],
            [413, 0, 0],
        ],
    },
    // 4:2:2 `YCBCR8_422_2P` over `422f`, BT.2020, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR84222P,
        size: [33, 17],
        rows: [64, 17, 64, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt2020),
        seed: 416,
        digest: 0x51c5_c9fc,
        first: [
            [0, 447, 419],
            [1118, 1695, 2048],
            [1079, 904, 2048],
            [649, 158, 1687],
        ],
    },
    // 4:4:4 `YCBCR8_444_2P` over `444v`, BT.709.
    MetalRead {
        pf: PixelFormat::YCBCR84442P,
        size: [64, 64],
        rows: [64, 64, 128, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 421,
        digest: 0x2ef7_98c9,
        first: [
            [941, 1430, 678],
            [2048, 579, 2048],
            [1650, 0, 1904],
            [604, 1659, 0],
        ],
    },
    // 4:2:2 `YCBCR10_422_2P` over `xf22`, BT.601.
    MetalRead {
        pf: PixelFormat::YCBCR104222P,
        size: [64, 64],
        rows: [128, 64, 128, 64],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 443,
        digest: 0x3bf8_2c8f,
        first: [
            [0, 646, 350],
            [318, 1102, 1273],
            [362, 0, 932],
            [1075, 593, 1348],
        ],
    },
    // 4:4:4 `YCBCR10_444_2P` over `x444`, no matrix, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104442P,
        size: [33, 17],
        rows: [128, 17, 192, 17],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 450,
        digest: 0x001c_2648,
        first: [
            [2001, 1425, 1258],
            [959, 0, 256],
            [2048, 1054, 664],
            [1065, 933, 0],
        ],
    },
    // `YCBCR8_422_2P_sRGB` over `422v`, BT.709.
    MetalRead {
        pf: PixelFormat::YCBCR84222PSRGB,
        size: [64, 64],
        rows: [64, 64, 64, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 517,
        digest: 0x6df2_0fb4,
        first: [
            [0, 47, 4849],
            [2446, 4353, 8192],
            [4633, 4648, 956],
            [572, 586, 0],
        ],
    },
    // `YCBCR8_444_2P_sRGB` over `444f`, no matrix, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR84442PSRGB,
        size: [33, 17],
        rows: [64, 17, 128, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 538,
        digest: 0xc01b_f61f,
        first: [
            [8192, 3596, 592],
            [6738, 8192, 95],
            [2943, 0, 1331],
            [1873, 450, 3464],
        ],
    },
    // `YCBCR10_422_2P_sRGB` over `x422`, BT.2020.
    MetalRead {
        pf: PixelFormat::YCBCR104222PSRGB,
        size: [64, 64],
        rows: [128, 64, 128, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        seed: 551,
        digest: 0xe9d1_8877,
        first: [
            [3198, 4104, 1286],
            [462, 120, 18],
            [6903, 871, 3378],
            [4457, 37, 4271],
        ],
    },
    // `YCBCR10_444_2P_sRGB` over `xf44`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104442PSRGB,
        size: [33, 17],
        rows: [128, 17, 192, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 574,
        digest: 0x170f_fcec,
        first: [
            [5752, 0, 0],
            [2524, 2429, 8192],
            [0, 2723, 3455],
            [0, 145, 0],
        ],
    },
    // `YCBCR8_420_2P` over `420v`.
    MetalRead {
        pf: PixelFormat::YCBCR84202P,
        size: [64, 64],
        rows: [64, 64, 64, 32],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        seed: 301,
        digest: 0xb78b_8d92,
        first: [
            [0, 1172, 2048],
            [604, 1364, 1919],
            [1688, 1549, 1460],
            [2048, 1464, 1269],
        ],
    },
    // `YCBCR8_420_2P_sRGB` over `420f`, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR84202PSRGB,
        size: [33, 17],
        rows: [64, 17, 64, 9],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt2020),
        seed: 308,
        digest: 0xbb35_1938,
        first: [[262, 7677, 5655], [0, 1272, 78], [0, 604, 0], [0, 3055, 0]],
    },
    // `YCBCR10_420_2P` over `x420`, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104202P,
        size: [33, 17],
        rows: [128, 17, 128, 9],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        seed: 310,
        digest: 0xf001_95e4,
        first: [
            [1303, 551, 2048],
            [2048, 1362, 2048],
            [2048, 1280, 763],
            [459, 0, 0],
        ],
    },
    // `YCBCR10_420_2P` over `xf20`.
    MetalRead {
        pf: PixelFormat::YCBCR104202P,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt2020),
        seed: 311,
        digest: 0x079f_57b3,
        first: [
            [940, 1759, 778],
            [526, 1239, 720],
            [1076, 1579, 1987],
            [771, 784, 1561],
        ],
    },
    // `YCBCR10_420_2P_sRGB` over `x420`.
    MetalRead {
        pf: PixelFormat::YCBCR104202PSRGB,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        seed: 317,
        digest: 0xd3d3_414e,
        first: [
            [753, 1542, 424],
            [3433, 1795, 2471],
            [8192, 1613, 8192],
            [8192, 1908, 7036],
        ],
    },
    // `YCBCR8_420_2P_sRGB` over `420f`, BT.709.
    MetalRead {
        pf: PixelFormat::YCBCR84202PSRGB,
        size: [64, 64],
        rows: [64, 64, 64, 32],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 109,
        digest: 0x5078_acf7,
        first: [
            [142, 1245, 7502],
            [0, 453, 5351],
            [137, 3816, 8192],
            [91, 1176, 8192],
        ],
    },
    // `YCBCR10_420_2P_sRGB` over `xf20`, BT.601.
    MetalRead {
        pf: PixelFormat::YCBCR104202PSRGB,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 120,
        digest: 0xc474_89d6,
        first: [
            [124, 381, 8192],
            [2673, 3343, 8192],
            [556, 461, 7115],
            [2282, 499, 4342],
        ],
    },
    // `YCBCR10_420_2P_sRGB` over `x420`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::YCBCR104202PSRGB,
        size: [33, 17],
        rows: [128, 17, 128, 9],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 130,
        digest: 0x416a_aa41,
        first: [
            [8192, 696, 0],
            [8167, 903, 719],
            [3650, 2548, 8192],
            [3904, 7362, 8192],
        ],
    },
    // `YCBCR8_420_2P_sRGB` over `420v`, no matrix: steps of `1 / 8192`.
    MetalRead {
        pf: PixelFormat::YCBCR84202PSRGB,
        size: [64, 64],
        rows: [64, 64, 64, 32],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 74,
        digest: 0xbf2e_3b63,
        first: [
            [0, 2274, 2082],
            [3382, 8192, 8192],
            [1659, 3748, 4038],
            [2856, 7235, 2754],
        ],
    },
    // `YCBCR10_420_2P_sRGB` over `x420`, no matrix.
    MetalRead {
        pf: PixelFormat::YCBCR104202PSRGB,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 75,
        digest: 0x12e2_9f12,
        first: [
            [0, 985, 2219],
            [1006, 1526, 6975],
            [3469, 0, 5809],
            [8192, 672, 8192],
        ],
    },
    // `x420`, no matrix.
    MetalRead {
        pf: PixelFormat::YCBCR104202P,
        size: [33, 17],
        rows: [128, 17, 128, 9],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 3,
        digest: 0x96d5_da95,
        first: [
            [1834, 2048, 1523],
            [2048, 2048, 2048],
            [1932, 748, 2048],
            [1868, 137, 1819],
        ],
    },
    // `xf20`, BT.709.
    MetalRead {
        pf: PixelFormat::YCBCR104202P,
        size: [64, 64],
        rows: [128, 64, 128, 32],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 12,
        digest: 0x87fc_51cb,
        first: [
            [69, 2048, 275],
            [835, 2048, 1484],
            [1196, 1922, 2048],
            [747, 990, 2048],
        ],
    },
    // `x444`, BT.709.
    MetalRead {
        pf: PixelFormat::YCBCR104202P,
        size: [33, 17],
        rows: [128, 17, 192, 17],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt709),
        seed: 31,
        digest: 0xc179_b394,
        first: [
            [793, 102, 0],
            [2006, 1866, 1446],
            [948, 1904, 2048],
            [717, 1653, 2048],
        ],
    },
    // `RGB8_420_2P` over `420v`, no matrix: the 8-bit codes.
    MetalRead {
        pf: PixelFormat::RGB84202P,
        size: [64, 64],
        rows: [64, 64, 64, 32],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 541,
        digest: 0x9a1c_4dae,
        first: [
            [152, 191, 246],
            [115, 126, 214],
            [40, 151, 150],
            [64, 118, 105],
        ],
    },
    // `RGB8_420_2P` over `420f`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::RGB84202P,
        size: [33, 17],
        rows: [64, 17, 64, 9],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 544,
        digest: 0xd068_c106,
        first: [[113, 202, 72], [96, 168, 69], [61, 143, 62], [57, 109, 44]],
    },
    // `RGB8_420_2P` over `420v`, BT.601, 7 x 5.
    MetalRead {
        pf: PixelFormat::RGB84202P,
        size: [7, 5],
        rows: [64, 5, 64, 3],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 545,
        digest: 0x348c_b55c,
        first: [
            [101, 5, 31],
            [136, 245, 86],
            [205, 231, 195],
            [191, 40, 245],
        ],
    },
    // `x422`, BT.601: the codes, which no matrix changes.
    MetalRead {
        pf: PixelFormat::RGB104202P,
        size: [33, 17],
        rows: [128, 17, 128, 17],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 53,
        digest: 0xb608_4c0f,
        first: [
            [618, 718, 588],
            [711, 680, 495],
            [896, 9, 308],
            [849, 700, 205],
        ],
    },
    // `RGB8_422_2P` over `422v`, no matrix.
    MetalRead {
        pf: PixelFormat::RGB84222P,
        size: [64, 64],
        rows: [64, 64, 64, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 601,
        digest: 0x5c2b_90de,
        first: [
            [47, 211, 195],
            [74, 121, 161],
            [127, 78, 94],
            [149, 227, 106],
        ],
    },
    // `RGB8_422_2P` over `444v`: a 4:2:2 read of a 4:4:4 plane, at its pairs and rows.
    MetalRead {
        pf: PixelFormat::RGB84222P,
        size: [64, 64],
        rows: [64, 64, 128, 64],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 609,
        digest: 0x1b74_f9b2,
        first: [[68, 94, 35], [99, 103, 69], [160, 239, 137], [189, 71, 135]],
    },
    // `RGB8_444_2P` over `444f`, BT.709, odd sizes.
    MetalRead {
        pf: PixelFormat::RGB84442P,
        size: [33, 17],
        rows: [64, 17, 128, 17],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt709),
        seed: 626,
        digest: 0xc587_ff68,
        first: [[13, 220, 122], [11, 51, 8], [202, 1, 133], [4, 173, 113]],
    },
    // `RGB10_422_2P` over `x422`, BT.601, 7 x 5.
    MetalRead {
        pf: PixelFormat::RGB104222P,
        size: [7, 5],
        rows: [64, 5, 64, 5],
        conversion: (YCbCrRange::Video, YCbCrMatrix::Bt601),
        seed: 649,
        digest: 0xd9ff_89e2,
        first: [
            [628, 60, 683],
            [581, 232, 709],
            [486, 706, 760],
            [512, 772, 744],
        ],
    },
    // `RGB10_444_2P` over `xf44`, BT.601, 16 x 8.
    MetalRead {
        pf: PixelFormat::RGB104442P,
        size: [16, 8],
        rows: [64, 8, 64, 8],
        conversion: (YCbCrRange::Full, YCbCrMatrix::Bt601),
        seed: 672,
        digest: 0xac01_ef97,
        first: [
            [469, 565, 143],
            [78, 686, 553],
            [914, 599, 1016],
            [269, 496, 84],
        ],
    },
];

/// The model reads 10-bit surfaces, converted and as codes, the sRGB twins' 8- and 10-bit ones,
/// 8-bit ones as codes, and packed ones, exactly as Metal read them.
#[test]
fn the_model_reads_the_probes_surfaces_as_metal_does() {
    for r in METAL_PROBE_READS {
        let [luma_row, luma_rows, chroma_row, chroma_rows] = r.rows;
        let mut random = Random::ycf3(r.seed);
        let luma = random.bytes(luma_row * luma_rows);
        let chroma = random.bytes(chroma_row * chroma_rows);
        let surface = Surface::of_planes(
            r.size,
            (&luma, &chroma),
            [luma_row, chroma_row],
            r.conversion,
            codes_of(r.pf),
        );
        let got = surface.modelled();
        assert_eq!(got[..4], r.first, "{:?} seed {}", r.pf, r.seed);
        assert_eq!(digest(&got), r.digest, "{:?} seed {}", r.pf, r.seed);
    }
}

/// Metal's measured texels of single codes (the probe's model, which every 8-bit luma and
/// chroma code was compared against): black of zero chroma reads green, a video-range one more
/// than a full-range one.
#[test]
fn codes_of_zero_read_metals_green() {
    let video = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, 0, 0, 0);
    let full = ycbcr_texel(YCbCrRange::Full, YCbCrMatrix::Bt601, 0, 0, 0);
    assert_eq!(video, [0, 1088, 0]);
    assert_eq!(full, [0, 1083, 0]);
    let white = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt709, 235, 128, 128);
    assert_eq!(white, [2048, 2048, 2048]);
    // A 10-bit code four times an 8-bit one reads as it in video range.
    let ten = codes_of(PixelFormat::YCBCR104202P);
    for (y, cb, cr) in [(0u16, 0u16, 0u16), (235, 128, 128), (90, 200, 17)] {
        for conversion in [CONVERSIONS[0], CONVERSIONS[1]] {
            let eight = ycbcr_texel(conversion.0, conversion.1, y as u8, cb as u8, cr as u8);
            assert_eq!(ycbcr_read(ten, conversion, [4 * y, 4 * cb, 4 * cr]), eight);
        }
    }
}

/// The sRGB knots are the curve's (the module docs' sRGB rule): each the step of `1 / 8192` nearest
/// `8192 · srgb(16 j / 8192)`, which no knot lies within `10^-4` of a tie of, so `f64`'s curve
/// decides each; and the line between two knots rounds a remainder of 8 to the even step.
#[test]
fn the_srgb_knots_are_the_curves() {
    let srgb = |c: f64| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    for (j, &k) in SRGB_KNOTS.iter().enumerate() {
        let v = 8192.0 * srgb(16.0 * j as f64 / 8192.0);
        assert!((v - v.floor() - 0.5).abs() > 1e-4, "knot {j} near a tie");
        assert_eq!(u32::from(k), (v + 0.5).floor() as u32, "knot {j}");
        assert_eq!(srgb_step(16 * j as u32), k, "the line meets knot {j}");
    }
    // Knots 0 and 1 are 0 and 1: step 8 lies halfway, at 0 (even); knots 1 and 2 are 1 and 2:
    // step 24 lies halfway, at 2 (even). Measured: 8 reads 0, 24 reads 2; and every step from 0 to
    // 8192 reads the line's (the probe's 512 x 512 surfaces).
    assert_eq!(SRGB_KNOTS[..3], [0, 1, 2]);
    assert_eq!(srgb_step(8), 0);
    assert_eq!(srgb_step(24), 2);
    assert_eq!(srgb_step(8192), 8192);
}
