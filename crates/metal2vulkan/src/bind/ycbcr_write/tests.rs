//! The YCbCr write kernel on the CPU interpreter, against the CPU references [`ycbcr_write`]
//! and [`ycbcr_store`], and the references against Metal's own writes of our probes' surfaces
//! and its stores of our probe's passes.

use super::{typed_module, ycbcr_store, ycbcr_write, YCbCrWriteModule, BYTES, SHAPE, WRITTEN};
use crate::bind::{YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrRange};
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::metal::format::{ycbcr_format, ycbcr_writes_stored};
use crate::metal::PixelFormat;
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

#[test]
fn the_ycbcr_write_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

/// The formats whose writes Metal stores into both planes: `YCBCR8_444_2P`, `YCBCR10_444_2P`,
/// their `_sRGB` twins, `RGB8_444_2P` and `RGB10_444_2P` (503, 507, 523, 527, 542, 545).
const STORED: [PixelFormat; 6] = [
    PixelFormat::YCBCR84442P,
    PixelFormat::YCBCR104442P,
    PixelFormat::YCBCR84442PSRGB,
    PixelFormat::YCBCR104442PSRGB,
    PixelFormat::RGB84442P,
    PixelFormat::RGB104442P,
];

/// The codes of `pf`'s layout.
fn codes_of(pf: PixelFormat) -> YCbCrCodes {
    YCbCrCodes::of(ycbcr_format(pf).expect("a two-plane format")).expect("a layout")
}

/// The formats whose writes are stored are the six above, the reference encodes each of their
/// layouts, and it encodes no other two-plane layout: not the packed formats' or the subsampled
/// ones'.
#[test]
fn the_stored_formats_are_the_unpacked_four_four_four_ones() {
    let stored: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ycbcr_writes_stored(pf))
        .collect();
    assert_eq!(stored, STORED.to_vec());
    let texel = [0.25, 0.5, 0.75];
    let video = (YCbCrRange::Video, YCbCrMatrix::Bt709);
    for pf in PixelFormat::ALL.iter().copied() {
        let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) else {
            continue;
        };
        let encoded = ycbcr_write(codes, video, texel);
        assert_eq!(encoded.is_some(), STORED.contains(&pf), "{pf:?}");
    }
    assert_eq!(
        [PixelFormat::RGB84442P, PixelFormat::RGB104442P].map(|pf| codes_of(pf).output),
        [YCbCrOutput::Codes; 2]
    );
}

/// The pseudo-random generator of our probes (the `ycw3` and `ycw4`), which made each case's
/// written values.
struct Gen(u64);

impl Gen {
    fn new(seed: u64) -> Gen {
        Gen(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() % (1 << 24)) as f32 / (1u32 << 24) as f32
    }
}

/// The `f32` after `x` toward positive infinity, `-0` read as `+0`.
fn next_up(x: f32) -> f32 {
    let b = match x.to_bits() {
        0x8000_0000 => 0,
        b => b,
    };
    f32::from_bits(if b < 0x8000_0000 { b + 1 } else { b - 1 })
}

/// The `f32` before `x` toward negative infinity.
fn next_down(x: f32) -> f32 {
    let b = x.to_bits();
    if b == 0 {
        return f32::from_bits(0x8000_0001);
    }
    f32::from_bits(if b < 0x8000_0000 { b - 1 } else { b + 1 })
}

/// The probes' special values: NaNs, infinities, zeros, the edges of `[0, 1]`, a subnormal,
/// out-of-range magnitudes, and values on and beside the grid of `1 / 2048`.
fn specials() -> [f32; 25] {
    let f = f32::from_bits;
    let tie = 2047.5f32 / 2048.0;
    let half_step = 0.5f32 / 2048.0;
    [
        f(0x7fc0_0000),
        f(0xffc0_0000),
        f(0x7fa0_0001),
        f(0xffc0_1234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        f(1),
        f(0x0040_0000),
        f(0x8000_0001),
        next_up(1.0),
        next_down(1.0),
        next_up(0.5),
        next_down(0.5),
        1e30,
        -1e30,
        0.25 / 2048.0,
        0.75 / 2048.0,
        tie,
        next_up(tie),
        next_down(tie),
        half_step,
        next_down(half_step),
    ]
}

/// How a probe case made its values.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// A mix of kinds per texel: on the grid, finer, ties and an ulp either side, any value of
    /// `[0, 1]`, out of range, and the specials.
    Mix,
    /// Every texel gray, `i / (n - 1)` of texel `i`.
    Gray,
    /// Ties of the code grid of `scale` (255 or 1023) and up to three ulps either side.
    Ties(f64),
}

/// The `n` texels `[r, g, b, a]` a probe case of `seed` wrote, its generator first advanced
/// `skip` times (the plane fill before them).
fn values(seed: u64, n: usize, skip: usize, mode: Mode) -> Vec<[f32; 4]> {
    let mut g = Gen::new(seed);
    for _ in 0..skip {
        g.next();
    }
    let specials = specials();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = [0.0f32, 0.0, 0.0, 1.0];
        let s = i as f32 / (n - 1) as f32;
        match mode {
            Mode::Mix => {
                let kind = g.next() % 16;
                for c in &mut v {
                    *c = match kind {
                        0..=5 => (g.next() % 2049) as f32 / 2048.0,
                        6 => (g.next() % 65537) as f32 / 65536.0,
                        7 => ((g.next() % 2048) as f32 + 0.5) / 2048.0,
                        8 => {
                            let t = ((g.next() % 2048) as f32 + 0.5) / 2048.0;
                            if g.next().is_multiple_of(2) {
                                next_up(t)
                            } else {
                                next_down(t)
                            }
                        }
                        9 | 10 => g.unit(),
                        11 => g.unit() * 4.0 - 1.5,
                        12 => {
                            if g.next().is_multiple_of(2) {
                                specials[(g.next() % specials.len() as u64) as usize]
                            } else {
                                g.unit()
                            }
                        }
                        13 => (g.next() % 4097) as f32 / 4096.0,
                        14 => (g.next() % 16385) as f32 / 16384.0,
                        _ => (g.next() % 8193) as f32 / 8192.0,
                    };
                }
            }
            Mode::Ties(scale) => {
                for c in &mut v[..3] {
                    let k = (g.next() % scale as u64) as f64;
                    let mut t = ((k + 0.5) / scale) as f32;
                    let steps = (g.next() % 7) as i64 - 3;
                    for _ in 0..steps.max(0) {
                        t = next_up(t);
                    }
                    for _ in 0..(-steps).max(0) {
                        t = next_down(t);
                    }
                    *c = t;
                }
            }
            Mode::Gray => v = [s, s, s, 1.0],
        }
        out.push(v);
    }
    out
}

/// FNV-1a over `codes`, each a little-endian `u16`.
fn digest(codes: impl IntoIterator<Item = u16>) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for byte in codes.into_iter().flat_map(|c| c.to_le_bytes()) {
        h = (h ^ u32::from(byte)).wrapping_mul(16_777_619);
    }
    h
}

/// One of our probes' cases (an Apple M4 under macOS 27.0.1; the `ycw3` cases alike on an Apple M3
/// Max under macOS 26.7): a 256 x 256 texture of `format` over a surface of `range` and `matrix`,
/// every texel written `float` values, or `half` ones (`half`), of `mode` from `seed`; and the
/// digest of the luma, Cb and Cr codes Metal stored, texel by texel, row by row.
struct ProbeCase {
    format: PixelFormat,
    surface: (YCbCrRange, YCbCrMatrix),
    half: bool,
    mode: Mode,
    seed: u64,
    digest: u32,
}

/// The cases (the handoff): the `444v`, `444f`, `x444` and `xf44` surfaces, every matrix, `float`
/// and `half` writes.
fn probe_cases() -> [ProbeCase; 11] {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    let case = |format, surface, half, mode, seed, digest| ProbeCase {
        format,
        surface,
        half,
        mode,
        seed,
        digest,
    };
    [
        case(
            P::YCBCR84442P,
            (Video, Bt601),
            false,
            Mode::Mix,
            7301,
            0x8744_59b6,
        ),
        case(
            P::YCBCR84442P,
            (Full, Bt2020),
            false,
            Mode::Mix,
            7308,
            0xd6da_89cd,
        ),
        case(
            P::YCBCR104442P,
            (Video, Bt709),
            false,
            Mode::Mix,
            7311,
            0xf1df_ca8c,
        ),
        case(
            P::YCBCR104442P,
            (Full, Bt601),
            false,
            Mode::Mix,
            7314,
            0xfbdd_80a3,
        ),
        case(
            P::YCBCR84442P,
            (Full, Bt709),
            true,
            Mode::Mix,
            7333,
            0x9a9e_c4ef,
        ),
        case(
            P::YCBCR104442P,
            (Video, Bt2020),
            true,
            Mode::Mix,
            7336,
            0x46c3_a3a8,
        ),
        case(
            P::RGB84442P,
            (Video, Bt601),
            false,
            Mode::Mix,
            7339,
            0x2530_18f6,
        ),
        case(
            P::RGB104442P,
            (Full, Bt601),
            true,
            Mode::Mix,
            7346,
            0xfaae_867c,
        ),
        case(
            P::YCBCR104442P,
            (Full, Bt709),
            false,
            Mode::Gray,
            7370,
            0xeee6_ae65,
        ),
        case(
            P::RGB84442P,
            (Full, Bt601),
            false,
            Mode::Ties(255.0),
            8103,
            0x9917_9744,
        ),
        case(
            P::RGB104442P,
            (Video, Bt601),
            false,
            Mode::Ties(1023.0),
            8105,
            0xd5d0_17d7,
        ),
    ]
}

impl ProbeCase {
    /// The texels the case wrote, `[r, g, b]`: a `half` write's widened from its `half`.
    fn written(&self) -> Vec<[f32; 3]> {
        let codes = codes_of(self.format);
        let (w, h) = (256usize, 256usize);
        let unit = codes.bytes as usize;
        // The generator filled both planes' codes first: a luma code and a chroma pair a texel.
        let skip = (w * unit + 2 * w * unit) * h;
        values(self.seed, w * h, skip, self.mode)
            .into_iter()
            .map(|[r, g, b, _]| {
                let v = [r, g, b];
                match self.half {
                    true => v.map(|c| f16_bits_to_f32(f32_to_f16_bits(c))),
                    false => v,
                }
            })
            .collect()
    }

    /// The reference's codes of every texel.
    fn encoded(&self) -> Vec<[u16; 3]> {
        let codes = codes_of(self.format);
        self.written()
            .into_iter()
            .map(|t| ycbcr_write(codes, self.surface, t).expect("encoded"))
            .collect()
    }
}

/// The reference stores Metal's codes for every texel of our probes' cases, byte for byte (each
/// case's digest of Metal's planes; the first case's first texels as well).
#[test]
fn the_reference_stores_metal_s_codes_for_the_probes_cases() {
    for case in probe_cases() {
        let encoded = case.encoded();
        assert_eq!(encoded.len(), 256 * 256);
        assert_eq!(
            digest(encoded.iter().flatten().copied()),
            case.digest,
            "{:?} {:?} seed {}",
            case.format,
            case.surface,
            case.seed
        );
    }
    let first = probe_cases()[0].encoded();
    assert_eq!(
        first[..4],
        [[114, 193, 67], [122, 72, 140], [154, 150, 30], [87, 87, 76]]
    );
}

/// A surface's two planes and the texels a function wrote to a texture over them (or a pass left
/// in one): `bytes` the planes' span, `texels` each texel's `[r, g, b]` and whether it was
/// written.
struct Surface {
    planes: YCbCrPlanes,
    bytes: Vec<u8>,
    texels: Vec<([f32; 3], bool)>,
}

impl Surface {
    /// A `w x h` texture of `format` over planes of `surface`, the luma plane at byte 0 and the
    /// chroma plane (of the format's chroma size) after it, each row padded to 16 bytes and 16
    /// more; the planes' bytes and the written values from `seed`, about one texel in four
    /// unwritten.
    fn new(
        format: PixelFormat,
        surface: (YCbCrRange, YCbCrMatrix),
        [w, h]: [u32; 2],
        seed: u64,
    ) -> Surface {
        let codes = codes_of(format);
        let pad = |n: u32| n.next_multiple_of(16) + 16;
        let [sx, sy] = codes.subsampling.halved().map(u32::from);
        let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
        let luma_row = pad(w * codes.bytes);
        let chroma_row = pad(2 * cw * codes.bytes);
        let chroma_at = luma_row * h;
        let planes = YCbCrPlanes::new(
            [w, h],
            [0, luma_row],
            [chroma_at, chroma_row],
            surface,
            codes,
        );
        let mut g = Gen::new(seed);
        let bytes = (0..chroma_at + chroma_row * ch)
            .map(|_| g.next() as u8)
            .collect();
        let texels = values(seed ^ 0x5a5a, (w * h) as usize, 0, Mode::Mix)
            .into_iter()
            .map(|[red, green, blue, _]| ([red, green, blue], !g.next().is_multiple_of(4)))
            .collect();
        Surface {
            planes,
            bytes,
            texels,
        }
    }

    /// Write `code` at byte `at` of `out`: a byte, or a little-endian 16-bit word with the code
    /// in its high bits.
    fn put(&self, out: &mut [u8], at: u32, code: u16) {
        let codes = self.planes.codes;
        let at = at as usize;
        match codes.bytes {
            1 => out[at] = code as u8,
            _ => {
                let word = code << (16 - codes.bits);
                out[at..at + 2].copy_from_slice(&word.to_le_bytes());
            }
        }
    }

    /// The planes' bytes after the writes, by the reference: each written texel's luma code at
    /// its place in its row of the luma plane and its Cb and Cr codes at its pair in its row of
    /// the chroma plane; every other byte as it was.
    fn modelled(&self) -> Vec<u8> {
        let p = &self.planes;
        let codes = p.codes;
        let [w, _] = p.size;
        let mut out = self.bytes.clone();
        let mut put = |at: u32, code: u16| self.put(&mut out, at, code);
        for (i, &(texel, written)) in self.texels.iter().enumerate() {
            if !written {
                continue;
            }
            let (x, y) = (i as u32 % w, i as u32 / w);
            let [luma, cb, cr] = ycbcr_write(codes, (p.range, p.matrix), texel).expect("encoded");
            put(p.luma[0] + y * p.luma[1] + x * codes.bytes, luma);
            let pair = p.chroma[0] + y * p.chroma[1] + 2 * x * codes.bytes;
            put(pair, cb);
            put(pair + codes.bytes, cr);
        }
        out
    }

    /// The planes' bytes after a pass's store of the texels, by the reference ([`ycbcr_store`]):
    /// every texel's luma code at its place in its row of the luma plane, and each chroma pair's
    /// Cb and Cr codes of its block at its place in its row of the chroma plane, written or not;
    /// every other byte (the rows' padding) as it was.
    fn stored(&self) -> Vec<u8> {
        let p = &self.planes;
        let codes = p.codes;
        let [w, h] = p.size;
        let [cw, _] = p.chroma_size();
        let texels: Vec<[f32; 3]> = self.texels.iter().map(|&(t, _)| t).collect();
        let (luma, chroma) =
            ycbcr_store(codes, (p.range, p.matrix), &texels, p.size).expect("stored");
        let mut out = self.bytes.clone();
        for (i, &code) in luma.iter().enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            self.put(&mut out, p.luma[0] + y * p.luma[1] + x * codes.bytes, code);
        }
        assert_eq!(luma.len(), (w * h) as usize);
        for (i, &[cb, cr]) in chroma.iter().enumerate() {
            let (x, y) = (i as u32 % cw, i as u32 / cw);
            let pair = p.chroma[0] + y * p.chroma[1] + 2 * x * codes.bytes;
            self.put(&mut out, pair, cb);
            self.put(&mut out, pair + codes.bytes, cr);
        }
        out
    }

    /// The planes' bytes after the kernel's dispatch over them, both invocation orders alike,
    /// for a function's writes, or, `all`, a pass's store; the texels are only read.
    fn encoded(&self, all: bool) -> Vec<u8> {
        let m = typed_module().expect("module");
        let [w, h] = self.planes.size;
        let groups = YCbCrWriteModule::groups(&self.planes);
        let mut image = Image::new(WRITTEN, [w, h, 1], 1, 1, 1);
        image.texels = self
            .texels
            .iter()
            .map(|&([r, g, b], written)| {
                [r.to_bits(), g.to_bits(), b.to_bits(), u32::from(written)]
            })
            .collect();
        let mut runs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let part = |slot, bytes: Vec<u8>| Buffer {
                slot: Some(slot),
                address: None,
                bytes,
            };
            let shape = YCbCrWriteModule::shape(&self.planes, all);
            let res = Resources {
                buffers: vec![part(SHAPE, shape.to_vec()), part(BYTES, self.bytes.clone())],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups[0], groups[1], 1],
                ..Config::default()
            };
            let (outcome, images) =
                run_with_images(&m, &cfg, res, vec![image.clone()]).expect("run");
            assert_eq!(images[0], image, "the written texels are only read");
            assert_eq!(outcome.buffers[0], shape.to_vec(), "the shape is only read");
            runs.push(outcome.buffers[1].clone());
        }
        assert_eq!(runs[0], runs[1], "order independent");
        runs.swap_remove(0)
    }
}

/// The kernel stores the reference's codes of every written texel into both planes and keeps every
/// other byte (the unwritten texels' codes and the rows' padding), for each stored format over each
/// range and matrix, on textures of odd and even sizes whose rows end mid-word.
#[test]
fn the_kernel_stores_the_reference_s_codes_of_every_written_texel() {
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    let mut seed = 1;
    for format in STORED {
        let converted = codes_of(format).output != YCbCrOutput::Codes;
        // The code formats store the channels whatever the surface's range and matrix.
        let surfaces: &[(YCbCrRange, YCbCrMatrix)] = match converted {
            true => &[
                (Video, Bt601),
                (Video, Bt709),
                (Video, Bt2020),
                (Full, Bt601),
                (Full, Bt709),
                (Full, Bt2020),
            ],
            false => &[(Video, Bt601), (Full, Bt709)],
        };
        for &surface in surfaces {
            for size in [[7, 3], [8, 2], [1, 1], [13, 4]] {
                seed += 1;
                let s = Surface::new(format, surface, size, seed);
                assert_eq!(
                    s.encoded(false),
                    s.modelled(),
                    "{format:?} {surface:?} {size:?} seed {seed}"
                );
            }
        }
    }
}

/// The kernel stores Metal's codes of a probe case's texels: the first rows of each case (the
/// reference matched Metal over the whole texture,
/// [`the_reference_stores_metal_s_codes_for_the_probes_cases`]), every texel written.
#[test]
fn the_kernel_stores_metal_s_codes_of_the_probes_first_rows() {
    for case in probe_cases() {
        let [w, h] = [256u32, 2];
        let mut s = Surface::new(case.format, case.surface, [w, h], case.seed);
        s.texels = case
            .written()
            .into_iter()
            .take((w * h) as usize)
            .map(|t| (t, true))
            .collect();
        let encoded = s.encoded(false);
        assert_eq!(
            encoded,
            s.modelled(),
            "{:?} seed {}",
            case.format,
            case.seed
        );
        let codes = s.planes.codes;
        let code = |row: u32, index: u32| -> u16 {
            let at = (row + index * codes.bytes) as usize;
            match codes.bytes {
                1 => u16::from(encoded[at]),
                _ => u16::from_le_bytes([encoded[at], encoded[at + 1]]) >> (16 - codes.bits),
            }
        };
        let p = &s.planes;
        let stored: Vec<[u16; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let pair = p.chroma[0] + y * p.chroma[1];
                [
                    code(p.luma[0] + y * p.luma[1], x),
                    code(pair, 2 * x),
                    code(pair, 2 * x + 1),
                ]
            })
            .collect();
        assert_eq!(stored, case.encoded()[..(w * h) as usize]);
    }
}

/// The formats whose passes' stores the reference encodes: those of unpacked 8- and 10-bit codes
/// read converted or as codes, and the 4:4:4 ones read through the sRGB curve, each of which a
/// pass renders (`metal::format::ycbcr_renders`).
const RENDERED: [PixelFormat; 14] = [
    PixelFormat::YCBCR84202P,
    PixelFormat::YCBCR84222P,
    PixelFormat::YCBCR84442P,
    PixelFormat::YCBCR104202P,
    PixelFormat::YCBCR104222P,
    PixelFormat::YCBCR104442P,
    PixelFormat::YCBCR84442PSRGB,
    PixelFormat::YCBCR104442PSRGB,
    PixelFormat::RGB84202P,
    PixelFormat::RGB84222P,
    PixelFormat::RGB84442P,
    PixelFormat::RGB104202P,
    PixelFormat::RGB104222P,
    PixelFormat::RGB104442P,
];

/// The reference encodes a pass's store of the unpacked layouts read converted or as codes and the
/// 4:4:4 ones read through the sRGB curve only, of sizes the chroma blocks tile, and of a texel
/// for every texel of the size.
#[test]
fn the_store_reference_encodes_the_unpacked_layouts_of_tiled_sizes() {
    let video = (YCbCrRange::Video, YCbCrMatrix::Bt709);
    for pf in PixelFormat::ALL.iter().copied() {
        let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) else {
            continue;
        };
        let texels = [[0.25, 0.5, 0.75]; 8];
        let stored = ycbcr_store(codes, video, &texels, [4, 2]);
        assert_eq!(stored.is_some(), RENDERED.contains(&pf), "{pf:?}");
    }
    let codes = codes_of(PixelFormat::YCBCR84202P);
    assert!(ycbcr_store(codes, video, &[[0.5; 3]; 6], [3, 2]).is_none());
    assert!(ycbcr_store(codes, video, &[[0.5; 3]; 6], [2, 3]).is_none());
    assert!(ycbcr_store(codes, video, &[[0.5; 3]; 7], [4, 2]).is_none());
    let codes = codes_of(PixelFormat::YCBCR84222P);
    assert!(ycbcr_store(codes, video, &[[0.5; 3]; 6], [2, 3]).is_some());
    // A gray's chroma is the middle code; every texel's luma its own.
    let (luma, chroma) = ycbcr_store(codes, video, &[[0.0; 3], [1.0; 3]], [2, 1]).expect("stored");
    assert_eq!((luma, chroma), (vec![16, 235], vec![[128, 128]]));
}

/// The kernel stores the reference's codes of a pass's every texel into both planes, the marks not
/// read, and keeps every other byte (the rows' padding), for each unpacked layout read converted
/// or as codes over each range and matrix, on sizes whose rows end mid-word.
#[test]
fn the_kernel_stores_the_reference_s_codes_of_a_pass_s_every_texel() {
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    let mut seed = 100;
    for format in RENDERED {
        let sizes: &[[u32; 2]] = match codes_of(format).subsampling.halved() {
            [true, true] => &[[2, 2], [6, 4], [14, 2], [8, 6]],
            [true, false] => &[[2, 1], [6, 3], [14, 5]],
            _ => &[[1, 1], [7, 3], [13, 2]],
        };
        for surface in [
            (Video, Bt601),
            (Video, Bt709),
            (Video, Bt2020),
            (Full, Bt601),
            (Full, Bt709),
            (Full, Bt2020),
        ] {
            for &size in sizes {
                seed += 1;
                let s = Surface::new(format, surface, size, seed);
                assert_eq!(
                    s.encoded(true),
                    s.stored(),
                    "{format:?} {surface:?} {size:?} seed {seed}"
                );
            }
        }
    }
}

/// The values one of our probe's draws wrote, of its generator.
#[derive(Clone, Copy, Debug)]
enum PassValues {
    /// Each channel `k / 4096`, one texel in eight's mapped to `[-0.5, 1.5]`.
    Grid,
    /// Each channel beside a tie of the 1/2048 grid (up to 3 ulps either side; a quarter of them
    /// below 1/2, where a `half` rounds them), or a multiple of 2^-24.
    Ties,
}

/// One of our probes' passes into a texture of `size` and `format` over plane 0 of a CoreVideo
/// surface of `surface` (an Apple M4 under macOS 27.0.1; the consumer's run of the first on an
/// Apple M3 Max under macOS 26.7 stored the same bytes): a clear to a color from `seed`, or a
/// full-screen draw of values from it at every texel (`draw`); the planes' `plane_bytes` the
/// generator filled first, and the digest of the luma codes Metal stored, row by row, then of the
/// Cb and Cr codes of each pair, row by row.
struct PassCase {
    format: PixelFormat,
    surface: (YCbCrRange, YCbCrMatrix),
    size: [u32; 2],
    plane_bytes: usize,
    seed: u64,
    draw: Option<PassValues>,
    digest: u32,
}

impl PassCase {
    /// The texels the pass left, `[r, g, b]`: the probe's values, drawn, or its clear color
    /// (each channel `k / 4096`), which Metal takes through `float` (exactly).
    fn texels(&self) -> Vec<[f32; 3]> {
        let mut g = Gen::new(self.seed);
        for _ in 0..self.plane_bytes {
            g.next();
        }
        let n = (self.size[0] * self.size[1]) as usize;
        let values: Vec<[f32; 3]> = match self.draw {
            Some(PassValues::Ties) => (0..n)
                .map(|_| {
                    let v: [f32; 4] = std::array::from_fn(|_| match g.next() % 4 {
                        3 => (g.next() % (1 << 24)) as f32 / (1 << 24) as f32,
                        kind => {
                            let k = g.next() % if kind == 0 { 1024 } else { 2048 };
                            let t = (k as f32 + 0.5) / 2048.0;
                            let steps = (g.next() % 7) as i32 - 3;
                            f32::from_bits(t.to_bits().wrapping_add_signed(steps))
                        }
                    });
                    [v[0], v[1], v[2]]
                })
                .collect(),
            Some(PassValues::Grid) | None => (0..n)
                .map(|_| {
                    let wide = g.next().is_multiple_of(8);
                    let mut v = [0.0f32; 4];
                    for c in &mut v {
                        let q = (g.next() % 4097) as f32 / 4096.0;
                        *c = if wide { q * 2.0 - 0.5 } else { q };
                    }
                    [v[0], v[1], v[2]]
                })
                .collect(),
        };
        let clear = [0; 3].map(|_| ((g.next() % 4097) as f64 / 4096.0) as f32);
        match self.draw {
            Some(_) => values,
            None => vec![clear; n],
        }
    }
}

/// The cases: XRR2's, `420v` (no matrix, so BT.601) and `420f` under BT.709, and `422v` and
/// `444v`, each a clear and a draw, 16 x 8; and our draws of values beside the grid's ties,
/// 64 x 32, over `420v` without a matrix and with BT.709, `420f`, `422v`, `444v` and `444f`.
fn pass_cases() -> [PassCase; 14] {
    use PassValues::{Grid, Ties};
    use PixelFormat as P;
    use YCbCrMatrix::{Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    let case = |format, surface, (size, plane_bytes), seed, draw, digest| PassCase {
        format,
        surface,
        size,
        plane_bytes,
        seed,
        draw,
        digest,
    };
    let small = |bytes| ([16, 8], bytes);
    let large = |bytes| ([64, 32], bytes);
    [
        case(
            P::YCBCR84202P,
            (Video, Bt601),
            small(768),
            7101,
            None,
            0xa73b_34c5,
        ),
        case(
            P::YCBCR84202P,
            (Video, Bt601),
            small(768),
            7102,
            Some(Grid),
            0x352b_c16f,
        ),
        case(
            P::YCBCR84202P,
            (Full, Bt709),
            small(768),
            7106,
            None,
            0x7954_12c5,
        ),
        case(
            P::YCBCR84202P,
            (Full, Bt709),
            small(768),
            7107,
            Some(Grid),
            0xa2fb_bfe5,
        ),
        case(
            P::YCBCR84222P,
            (Video, Bt601),
            small(1024),
            7131,
            None,
            0x750b_1bc5,
        ),
        case(
            P::YCBCR84222P,
            (Video, Bt601),
            small(1024),
            7132,
            Some(Grid),
            0x551a_0651,
        ),
        case(
            P::YCBCR84442P,
            (Video, Bt601),
            small(1024),
            7133,
            None,
            0x609f_adc5,
        ),
        case(
            P::YCBCR84442P,
            (Video, Bt601),
            small(1024),
            7134,
            Some(Grid),
            0xd880_de27,
        ),
        case(
            P::YCBCR84202P,
            (Video, Bt601),
            large(3072),
            9101,
            Some(Ties),
            0xd25d_9b23,
        ),
        case(
            P::YCBCR84202P,
            (Full, Bt709),
            large(3072),
            9102,
            Some(Ties),
            0xa0d5_d8e0,
        ),
        case(
            P::YCBCR84222P,
            (Video, Bt601),
            large(4096),
            9103,
            Some(Ties),
            0x0823_a3bf,
        ),
        case(
            P::YCBCR84442P,
            (Video, Bt601),
            large(6144),
            9104,
            Some(Ties),
            0x48d0_d5bf,
        ),
        case(
            P::YCBCR84202P,
            (Video, Bt709),
            large(3072),
            9105,
            Some(Ties),
            0x5da6_53bb,
        ),
        case(
            P::YCBCR84442P,
            (Full, Bt709),
            large(6144),
            9106,
            Some(Ties),
            0xe8b4_97eb,
        ),
    ]
}

/// The reference stores Metal's codes of every texel of our probes' clears and draws, byte for byte
/// (each case's digest of Metal's planes), and the kernel stores the reference's.
#[test]
fn the_store_reference_and_kernel_store_metal_s_codes_of_the_probe_s_passes() {
    for case in pass_cases() {
        let what = format!("{:?} {:?} seed {}", case.format, case.surface, case.seed);
        let codes = codes_of(case.format);
        let texels = case.texels();
        let (luma, chroma) = ycbcr_store(codes, case.surface, &texels, case.size).expect("stored");
        let all = luma.iter().copied().chain(chroma.iter().flatten().copied());
        assert_eq!(digest(all), case.digest, "{what}");
        let mut s = Surface::new(case.format, case.surface, case.size, case.seed);
        s.texels = texels.into_iter().map(|t| (t, false)).collect();
        assert_eq!(s.encoded(true), s.stored(), "{what}");
    }
}

/// The eight colors of the consumer's probe of the `_sRGB` twins (an Apple M3 Max under macOS
/// 27.0.1): black, white, gray 0.5, red, green, blue, and two mixed ones.
const TWIN_COLORS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.2, 0.6, 0.9],
    [0.9, 0.3, 0.1],
];

/// The reference and the kernel store Metal's luma, Cb and Cr codes of each of [`TWIN_COLORS`]
/// into the `_sRGB` twins over surfaces without a matrix (the consumer's probe on an Apple M3 Max
/// under macOS 27.0.1: `YCBCR8_444_2P_sRGB` over `444v`, `YCBCR10_444_2P_sRGB` over `x444`, an
/// 8 x 2 texture, column `x` written with color `x` by a kernel and by a fragment function, and
/// cleared to each color by a pass): a function's writes exactly, and a pass's store exactly but
/// the 10-bit luma of (0.2, 0.6, 0.9), where Metal's pass stores 697 and its writes 696 (its value
/// by the exact curve 696.47): within a code of it.
#[test]
fn the_twins_references_and_kernel_store_metal_s_codes_of_the_probe_s_colors() {
    let surface = (YCbCrRange::Video, YCbCrMatrix::Bt601);
    for (format, metal) in [
        (
            PixelFormat::YCBCR84442PSRGB,
            [
                [16, 128, 128],
                [235, 128, 128],
                [177, 128, 128],
                [81, 90, 240],
                [145, 54, 34],
                [41, 240, 110],
                [174, 157, 90],
                [162, 88, 174],
            ],
        ),
        (
            PixelFormat::YCBCR104442PSRGB,
            [
                [64, 512, 512],
                [940, 512, 512],
                [708, 512, 512],
                [326, 361, 960],
                [578, 215, 137],
                [164, 960, 439],
                [696, 630, 360],
                [649, 351, 695],
            ],
        ),
    ] {
        let codes = codes_of(format);
        let written = TWIN_COLORS.map(|t| ycbcr_write(codes, surface, t).expect("encoded"));
        assert_eq!(written, metal, "{format:?}");
        let mut s = Surface::new(format, surface, [8, 2], 523);
        s.texels = (0..16).map(|i| (TWIN_COLORS[i % 8], true)).collect();
        assert_eq!(s.encoded(false), s.modelled(), "{format:?} written");
        for (i, &color) in TWIN_COLORS.iter().enumerate() {
            // A clear's color as the pass holds it: its `half`s.
            let cleared = color.map(|c| f16_bits_to_f32(f32_to_f16_bits(c)));
            let (luma, chroma) = ycbcr_store(codes, surface, &[cleared], [1, 1]).expect("stored");
            let [y, cb, cr] = metal[i];
            let pass_y = match (codes.bits, i) {
                (10, 6) => 697,
                _ => y,
            };
            assert!(luma[0].abs_diff(pass_y) <= 1, "{format:?} cleared to {i}");
            assert_eq!(chroma, [[cb, cr]], "{format:?} cleared to {i}");
        }
        s.texels = (0..16).map(|i| (TWIN_COLORS[i % 8], false)).collect();
        assert_eq!(s.encoded(true), s.stored(), "{format:?} stored");
    }
}

/// The values of our probe's draws into the code formats, `[r, g, b, a]` of each of `n` texels:
/// channel `k = 4 i + c` from `x = k · 2654435761 mod 2^32`, one in 16 a special (a NaN, the
/// infinities, `-0`, 1, 0, 1/2 or the least subnormal), the rest `((x >> 9) - 2^20) / 2^22`, in
/// `[-0.25, 1.75)`.
fn code_pass_values(n: usize) -> Vec<[f32; 4]> {
    const SPECIALS: [f32; 8] = [
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -0.0,
        1.0,
        0.0,
        0.5,
        1.0e-45,
    ];
    let value = |k: u32| {
        let x = k.wrapping_mul(2_654_435_761);
        match x % 16 {
            0 => SPECIALS[(x >> 4) as usize % 8],
            _ => ((x >> 9) as i32 - (1 << 20)) as f32 / (1 << 22) as f32,
        }
    };
    (0..n as u32)
        .map(|i| std::array::from_fn(|c| value(4 * i + c as u32)))
        .collect()
}

/// The reference stores Metal's codes of our probe's draws into the code formats (an Apple M3 Max
/// under macOS 27.0.1): a 64 x 64 texture of each over a CoreVideo surface (`420v`, `422f`,
/// `444v`, `x420`, `xf22` and `x444`), every texel drawn [`code_pass_values`]; each case's digest
/// of Metal's luma codes row by row, then of each pair's Cb and Cr codes. The kernel stores the
/// reference's codes of the first eight rows.
#[test]
fn the_store_reference_and_kernel_store_metal_s_codes_of_the_code_formats_passes() {
    use PixelFormat as P;
    let surface = (YCbCrRange::Video, YCbCrMatrix::Bt601);
    let values = code_pass_values(64 * 64);
    let texels: Vec<[f32; 3]> = values.iter().map(|&[r, g, b, _]| [r, g, b]).collect();
    for (format, metal) in [
        (P::RGB84202P, 0x303a_28c9),
        (P::RGB84222P, 0xd414_7cb3),
        (P::RGB84442P, 0x8d03_3f71),
        (P::RGB104202P, 0x9416_b68b),
        (P::RGB104222P, 0x6d8f_8efa),
        (P::RGB104442P, 0x7f29_8fea),
    ] {
        let codes = codes_of(format);
        let (luma, chroma) = ycbcr_store(codes, surface, &texels, [64, 64]).expect("stored");
        let all = luma.iter().copied().chain(chroma.iter().flatten().copied());
        assert_eq!(digest(all), metal, "{format:?}");
        let mut s = Surface::new(format, surface, [64, 8], 7);
        s.texels = texels[..64 * 8].iter().map(|&t| (t, false)).collect();
        assert_eq!(s.encoded(true), s.stored(), "{format:?}");
    }
}

/// What one of our probe `ten1`'s rows left in its pass's texels.
#[derive(Clone, Copy, Debug)]
enum TenValues {
    /// A draw of random channels: ties of the 1/2048, 1/4096, 1/8192 and 1/1023 grids up to 3
    /// float steps either side, `half`s of `[0, 1]`, multiples of 2^-24, and values of
    /// `[-0.25, 1.25]`; returned through a `half4` fragment where `half` (each channel rounded
    /// to nearest even).
    Random { half: bool },
    /// A draw of gray `i / 16384` at texel `i`.
    GrayFine,
    /// 64 clears, each to a color beside the ties of the 1/2048, 1/4096 and 1/1023 grids, or a
    /// multiple of 2^-24: each pass's texels the color's `half`s, as the runtime clears them.
    Clears,
}

/// One of our probe `ten1`'s rows (an Apple M4 under macOS 27.0.1): passes into
/// a texture of `size` and `format` over plane 0 of a CoreVideo surface of `surface`, the
/// planes' `plane_bytes` the generator filled first; and the digest of the codes Metal stored,
/// each pass's luma codes row by row, then its Cb and Cr codes of each pair, row by row.
struct TenCase {
    format: PixelFormat,
    surface: (YCbCrRange, YCbCrMatrix),
    size: [u32; 2],
    plane_bytes: usize,
    seed: u64,
    values: TenValues,
    digest: u32,
}

/// A channel `steps` float steps from `t` (positive).
fn nudged(t: f32, steps: i32) -> f32 {
    f32::from_bits(t.to_bits().wrapping_add_signed(steps))
}

/// `v` rounded to a `half`, to nearest even.
fn half_of(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}

impl TenCase {
    /// Each pass's texels, `[r, g, b]`, row by row.
    fn passes(&self) -> Vec<Vec<[f32; 3]>> {
        let mut g = Gen::new(self.seed);
        for _ in 0..self.plane_bytes {
            g.next();
        }
        let n = (self.size[0] * self.size[1]) as usize;
        match self.values {
            TenValues::Random { half } => {
                let mut channel = || {
                    let s = (g.next() % 7) as i32 - 3;
                    let v = match g.next() % 8 {
                        0 => nudged(((g.next() % 1024) as f32 + 0.5) / 2048.0, s),
                        1 => nudged(((g.next() % 2048) as f32 + 0.5) / 2048.0, s),
                        2 => nudged(((g.next() % 4096) as f32 + 0.5) / 4096.0, s),
                        3 => nudged(((g.next() % 8192) as f32 + 0.5) / 8192.0, s),
                        4 => nudged(((g.next() % 1023) as f32 + 0.5) / 1023.0, s),
                        5 => f16_bits_to_f32((g.next() % 0x3c01) as u16),
                        6 => (g.next() % (1 << 24)) as f32 / (1 << 24) as f32,
                        _ => (g.next() % 6145) as f32 / 4096.0 - 0.25,
                    };
                    if half {
                        half_of(v)
                    } else {
                        v
                    }
                };
                vec![(0..n).map(|_| [channel(), channel(), channel()]).collect()]
            }
            TenValues::GrayFine => vec![(0..n).map(|i| [i as f32 / 16384.0; 3]).collect()],
            TenValues::Clears => {
                let mut channel = || {
                    let s = (g.next() % 7) as i32 - 3;
                    let c = match g.next() % 4 {
                        0 => f64::from(nudged(((g.next() % 2048) as f32 + 0.5) / 2048.0, s)),
                        1 => f64::from(nudged(((g.next() % 4096) as f32 + 0.5) / 4096.0, s)),
                        2 => f64::from(nudged(((g.next() % 1023) as f32 + 0.5) / 1023.0, s)),
                        _ => (g.next() % (1 << 24)) as f64 / f64::from(1u32 << 24),
                    };
                    half_of(c as f32)
                };
                (0..64)
                    .map(|_| vec![[channel(), channel(), channel()]; n])
                    .collect()
            }
        }
    }
}

/// The cases: random draws over each 10-bit surface (`x420`, `xf20`, `x422`, `xf22`, `x444`,
/// `xf44`; no matrix, so BT.601, and BT.709), 128 x 128; a gray ramp and `half` fragments over
/// `x444`; the 8-bit control over `444v`; and 64 clears over `x420` and `xf44`, 16 x 8.
fn ten_cases() -> [TenCase; 11] {
    use PixelFormat as P;
    use TenValues::{Clears, GrayFine, Random};
    use YCbCrMatrix::{Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    let case = |format, surface, (size, plane_bytes), seed, values, digest| TenCase {
        format,
        surface,
        size,
        plane_bytes,
        seed,
        values,
        digest,
    };
    let large = |bytes| ([128, 128], bytes);
    let small = |bytes| ([16, 8], bytes);
    let float = Random { half: false };
    [
        case(
            P::YCBCR104442P,
            (Video, Bt601),
            large(98304),
            9409,
            float,
            0x936a_163e,
        ),
        case(
            P::YCBCR104442P,
            (Full, Bt709),
            large(98304),
            9417,
            float,
            0xb234_96ee,
        ),
        case(
            P::YCBCR104202P,
            (Video, Bt601),
            large(49152),
            9421,
            float,
            0x97e7_9cdd,
        ),
        case(
            P::YCBCR104202P,
            (Full, Bt709),
            large(49152),
            9424,
            float,
            0x4163_03cc,
        ),
        case(
            P::YCBCR104222P,
            (Video, Bt601),
            large(65536),
            9427,
            float,
            0x400c_8554,
        ),
        case(
            P::YCBCR104222P,
            (Full, Bt709),
            large(65536),
            9428,
            float,
            0x9a24_9e19,
        ),
        case(
            P::YCBCR104442P,
            (Video, Bt601),
            large(98304),
            9402,
            GrayFine,
            0xbb89_8db5,
        ),
        case(
            P::YCBCR104442P,
            (Video, Bt601),
            large(98304),
            9412,
            Random { half: true },
            0x2e0f_dcc1,
        ),
        case(
            P::YCBCR84442P,
            (Video, Bt601),
            large(49152),
            9431,
            float,
            0xee71_6133,
        ),
        case(
            P::YCBCR104202P,
            (Video, Bt601),
            small(768),
            9434,
            Clears,
            0x4452_7945,
        ),
        case(
            P::YCBCR104442P,
            (Full, Bt709),
            small(1024),
            9433,
            Clears,
            0x466b_1cc5,
        ),
    ]
}

/// The reference stores Metal's codes of every texel of our probe's 10-bit draws and clears (a
/// drawn `float`'s `half` toward zero, a chroma term at 2^-4 of a code) and of the 8-bit control,
/// byte for byte (each case's digest of Metal's planes); the kernel stores the reference's of each
/// pass's first 8 rows.
#[test]
fn the_store_reference_and_kernel_store_metal_s_codes_of_the_ten_bit_passes() {
    for case in ten_cases() {
        let what = format!("{:?} {:?} seed {}", case.format, case.surface, case.seed);
        let codes = codes_of(case.format);
        let passes = case.passes();
        let mut all = Vec::new();
        for texels in &passes {
            let (luma, chroma) =
                ycbcr_store(codes, case.surface, texels, case.size).expect("stored");
            all.extend(luma);
            all.extend(chroma.into_iter().flatten());
        }
        assert_eq!(digest(all), case.digest, "{what}");
        let [w, h] = case.size;
        let rows = h.min(8);
        let mut s = Surface::new(case.format, case.surface, [w, rows], case.seed);
        s.texels = passes[0][..(w * rows) as usize]
            .iter()
            .map(|&t| (t, false))
            .collect();
        assert_eq!(s.encoded(true), s.stored(), "{what}");
    }
}
