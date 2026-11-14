//! A function's writes to a 4:4:4 two-plane YCbCr texture, stored into its IOSurface's two planes
//! as Metal stores them.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1, and an Apple M3 Max under macOS 26.7, byte
//! for byte the same; our own probes over CoreVideo `444v`, `444f`, `x444` and `xf44` surfaces,
//! every texel written with fresh values: on the grid of `1 / 2048`, finer, ties of the grid and an
//! ulp either side, any `float` in `[0, 1]`, out of range, NaNs, infinities, `-0` and subnormals;
//! `float` and `half` writes; no matrix, BT.601, BT.709 and BT.2020; 2.4 million texels in all).
//! A texture of `YCBCR8_444_2P`, `YCBCR10_444_2P`, `RGB8_444_2P`, `RGB10_444_2P` or the first
//! two's `_sRGB` twins made over an IOSurface's plane 0 stores a written texel's luma code into
//! the luma plane at its place and its Cb and Cr codes into its own pair of the chroma plane,
//! alpha ignored: a byte a code (8 bits), or a little-endian 16-bit word with the code in its
//! high 10 bits and its low 6 bits 0 (10 bits). An unwritten texel's codes, and every byte
//! between the planes' rows, keep their bytes. A `half` texel writes as its value widened. Of each
//! written channel `v`:
//!
//! - **Converted** ([`YCbCrOutput::Converted`], `YCBCR*_444_2P`): `x = rint(clamp(v, 0, 1) ·
//!   2048)`, ties to even, a NaN as 0. Each code of `b` bits is
//!   `clamp(o + ((c · x + 2^25) >> 26), 0, 2^b - 1)` for the row `c` of the code and the
//!   offset `o` ([`write_rows`]): full range (`444f`, `xf44`), `c = S · K` for the matrix's
//!   15-bit row `K` (luma's sums to `2^15`, Cb's and Cr's to about 0) and the range's scale `S`,
//!   255 or 1023 of the luma, 256 or 1024 of the chroma, `o` 0 of the luma and `2^(b-1)` of the
//!   chroma; video range (`444v`, `x444`), the rows Metal stores at 18 or 17 bits (8-bit codes;
//!   10-bit ones the same at two bits fewer), `o` `16 · 2^(b-8)` of the luma and `2^(b-1)` of the
//!   chroma.
//! - **Codes** ([`YCbCrOutput::Codes`], `RGB*_444_2P`): Y the green channel, Cb the blue, Cr the
//!   red, each `round(clamp(v, 0, 1) · (2^b - 1))` of the exact product, to nearest (a NaN 0).
//!   The `float` product rounded instead misses about 1 in 14 values within 3 ulps of a tie.
//! - **sRGB** ([`YCbCrOutput::Srgb`], `YCBCR8_444_2P_sRGB` and `YCBCR10_444_2P_sRGB`; an Apple
//!   M3 Max under macOS 27.0.1, `444v` and `x444` surfaces without a matrix, black, white, gray
//!   0.5, red, green, blue, (0.2, 0.6, 0.9) and (0.9, 0.3, 0.1) written by a kernel and by a
//!   fragment function): each channel `v` clamped to `[0, 1]` (a NaN 0) goes through the sRGB
//!   curve, `s = 12.92 v` below 0.0031308 and `1.055 v^(1/2.4) - 0.055` elsewhere, and `s` is
//!   converted as a converted format's channel is. The kernel computes the curve in `float`
//!   arithmetic by the device's power (`lower::arith::pack::srgb_encode`), the reference by the
//!   same steps with the `f64` power rounded once; Metal's curve is within a code of the exact
//!   one (an Apple M4's earlier probes of random values). A step of the power moves `s · 2048` by
//!   far less than one, so a code differs by one at most, and only where `s · 2048` lies by a
//!   tie of the grid.
//!
//! The packed formats' writes are dropped (the M3 Max under macOS 27.0.1; an M4 stores them):
//! they are not encoded here ([`ycbcr_write`] answers `None`).
//!
//! **A render pass's store** (our own probes over CoreVideo `420v`, `420f`, `422v`, `444v` and
//! `444f` surfaces on an Apple M4 under macOS 27.0.1: every byte of the converted 8-bit formats'
//! clears and full-screen draws, 16 x 8, no matrix and BT.709, the consumer's Apple M3 Max under
//! macOS 26.7 storing the same bytes; and every byte of full-screen draws, 64 x 32, of floats
//! beside the ties of the 1/2048 grid and uniform ones, and of clears to colors beside them, the M3
//! Max's bytes the same; for the 10-bit formats, every byte of 128 x 128 draws over `x420`, `xf20`,
//! `x422`, `xf22`, `x444` and `xf44` of every `half` in `[0, 1]`, a 1/16384 grid and values beside
//! the ties of four grids, `float` and `half` fragments, and of 64 clears a surface). A pass whose
//! color attachment is such a texture stores every texel of it, drawn or not, into both planes.
//! Each texel's value is held as a `half`, and its `x` is the grid step of that: a drawn `float`
//! channel rounded to nearest even at 8 bits, toward zero at 10 bits; a `half` a fragment returns
//! kept; a clear's color rounded to nearest even at both (the runtime puts a clear into the pass's
//! texels as its `half`s, which either rounding keeps). Each texel's term of a row is `t = (c · x)
//! >> (20 + b - 8)`, its row counted at 2^-6 of an 8-bit code (2^-4 of a 10-bit one) and truncated;
//! and a code is `clamp(o + ((Σ t + 2^(T - 1 + k)) >> (T + k)), 0, 2^b - 1)`, `T` 6 at 8 bits and 4
//! at 10, over the block of `2^k` texels the code covers: a texel's own luma, and each chroma
//! pair's 2 x 2 texels at 4:2:0, 2 x 1 at 4:2:2, its own at 4:4:4 ([`ycbcr_store`]). A code of one
//! texel is a function's write's code of the `half`. A pass into a code format (`RGB8_420_2P` to
//! `RGB10_444_2P`, 540 to 545; our probes over `420v`, `422f`, `444v`, `x420`, `xf22` and `x444`
//! surfaces on an Apple M3 Max under macOS 27.0.1, every byte of 64 x 64 draws of values in
//! `[-0.25, 1.75)`, NaNs, infinities and zeros among them, and of draws over loads) stores each
//! texel's luma code as a write of its green channel's `half` (so rounded) stores it, and each
//! chroma pair's Cb and Cr codes the mean of its block's blue and red codes so made, rounded half
//! up: `(Σ code + 2^(k - 1)) >> k` over the block of `2^k` texels.
//!
//! A pass into a 4:4:4 `_sRGB` twin (an Apple M3 Max under macOS 27.0.1: a clear of each of the
//! eight colors above over `444v` and `x444`) stores each texel's `half` through the sRGB curve as
//! a write does, then as a pass into its converted twin stores it: the clears' codes are the
//! writes', but for the 10-bit luma of (0.2, 0.6, 0.9), 697 where the write's is 696 (its value
//! 696.47 by the exact curve), a code apart, Metal's own spread between its two paths.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`YCbCrWriteModule::LOCAL_SIZE`], [`YCbCrWriteModule::LOCAL_SIZE`], 1), no push constants,
//! and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`YCbCrWriteModule::WRITTEN_BINDING`] | the texture's written texels, a 2D storage image `Rgba32ui`, read: each the bits of its red, green and blue `float` and a mark, nonzero where a function wrote it (`StorageImageClass::YCbCrWrite`), or, for a pass's store, a pass's texels (the mark not read) |
//! | [`YCbCrWriteModule::SHAPE_BINDING`] | the planes' shape and the rows ([`YCbCrWriteModule::shape`]), a storage buffer |
//! | [`YCbCrWriteModule::BYTES_BINDING`] | the surface's bytes, 4 to a word, a storage buffer, read and written |
//!
//! Invocation `(x, y)` owns word `x` of row `y` of the luma plane (`y` below the height), or of
//! row `y - h` of the chroma plane: it replaces the codes in it of every marked texel (of every
//! texel, for a pass's store; a chroma pair's codes of its block) and keeps every other byte, so
//! no two invocations write one word. The consumer dispatches [`YCbCrWriteModule::groups`]
//! workgroups, and keeps the bytes binding within `maxStorageBufferRange`.
//!
//! **Synchronization** (consumer): the writes of the texels, the shape and the bytes →
//! `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its writes of the bytes
//! (`SHADER_STORAGE_WRITE`) → whatever reads them after.

use std::sync::Arc;

use super::ycbcr_convert::{scalar, words_buffer};
use super::{YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrRange};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::pack;
use crate::metal::format::ChromaSubsampling;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType, ImageUsage, IoDir,
    MemAccess, StorageClass, TexelMemory, Type, TypedModule, UnOp,
};

/// The 15-bit rows `K` of luma, Cb and Cr of each matrix, red, green and blue (the module docs):
/// a full-range code's row is the range's scale times its `K`.
fn rows_q15(matrix: YCbCrMatrix) -> [[i32; 3]; 3] {
    match matrix {
        YCbCrMatrix::Bt601 => [
            [9797, 19235, 3736],
            [-5537, -10846, 16383],
            [16384, -13730, -2654],
        ],
        YCbCrMatrix::Bt709 => [
            [6967, 23436, 2365],
            [-3756, -12629, 16384],
            [16384, -14884, -1501],
        ],
        YCbCrMatrix::Bt2020 => [
            [8608, 22217, 1943],
            [-4574, -11810, 16384],
            [16384, -15067, -1318],
        ],
    }
}

/// The video-range rows of luma, Cb and Cr of each matrix, red, green and blue, as Metal stores
/// 8-bit codes, each with its bits `n` (the module docs; 10-bit codes the same row at `n - 2`).
fn rows_video(matrix: YCbCrMatrix) -> [([i32; 3], u32); 3] {
    match matrix {
        YCbCrMatrix::Bt601 => [
            ([8381, 16455, 3196], 18),
            ([-4845, -9491, 14336], 18),
            ([7168, -6007, -1161], 17),
        ],
        YCbCrMatrix::Bt709 => [
            ([5960, 20049, 2023], 18),
            ([-1643, -5525, 7168], 17),
            ([14336, -13023, -1313], 18),
        ],
        YCbCrMatrix::Bt2020 => [
            ([3682, 9503, 831], 17),
            ([-4003, -10334, 14336], 18),
            ([14336, -13183, -1153], 18),
        ],
    }
}

/// The scalar `float` form `lower::arith::pack`'s sRGB curve takes.
const SCALAR: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};

/// The bits every row is counted at: a code is `(c · x + 2^25) >> 26` above its offset.
const ROW_BITS: u32 = 26;

/// The bits past a code a pass's store counts each texel's term at, truncated, of a code of
/// `bits` bits (the module docs): 6 at 8 bits, 4 at 10, the same step of an 8-bit code. A code of
/// one texel is the same as [`ROW_BITS`]' rounding of it.
fn term_bits(bits: u32) -> u32 {
    6 - (bits.clamp(8, 10) - 8)
}

/// Whether a render pass's store of a texture of `codes` takes a drawn `float` channel's `half`
/// toward zero (10-bit codes), not to nearest even (8-bit ones), as Metal does (the module
/// docs).
fn pass_rounds_toward_zero(codes: YCbCrCodes) -> bool {
    codes.bits == 10
}

/// The rows `c` (red, green, blue, counted at [`ROW_BITS`]) and offsets of the luma, Cb and Cr
/// codes of `bits` bits converted by `range` and `matrix` (the module docs).
pub(crate) fn write_rows(
    range: YCbCrRange,
    matrix: YCbCrMatrix,
    bits: u32,
) -> [([i32; 3], u32); 3] {
    let ten = bits == 10;
    let chroma = 1 << (bits - 1);
    let luma = match range {
        YCbCrRange::Video => 16 << (bits - 8),
        YCbCrRange::Full => 0,
    };
    let offsets = [luma, chroma, chroma];
    let rows: [[i32; 3]; 3] = match range {
        YCbCrRange::Full => {
            let k = rows_q15(matrix);
            let max = (1 << bits) - 1;
            let scales = [max, 1 << bits, 1 << bits];
            [0, 1, 2].map(|i| k[i].map(|c| c * scales[i]))
        }
        YCbCrRange::Video => rows_video(matrix).map(|(c, n)| {
            let n = if ten { n - 2 } else { n };
            c.map(|c| c << (ROW_BITS - n))
        }),
    };
    [0, 1, 2].map(|i| (rows[i], offsets[i]))
}

/// A pass's channel: `v` rounded to a `half` as Metal's pass holds it, to nearest even, or
/// `toward_zero` (of `v` clamped to `[0, 1]`; a NaN stays one).
fn pass_half(v: f32, toward_zero: bool) -> f32 {
    if !toward_zero {
        return f16_bits_to_f32(f32_to_f16_bits(v));
    }
    if v.is_nan() {
        return v;
    }
    // Of `v` in `[0, 1]` of exponent `E` (at least -14, a subnormal half's step), the half's
    // significand `q = floor(v · 2^(10 - E))`, its value `q · 2^(E - 10)`: each product exact.
    let v = v.clamp(0.0, 1.0);
    let exponent = (((v.to_bits() >> 23) & 0xff) as i32 - 127).max(-14);
    let scale = f64::powi(2.0, 10 - exponent);
    ((f64::from(v) * scale).floor() / scale) as f32
}

/// `v` clamped to `[0, 1]`, a NaN 0.
fn unit(v: f32) -> f32 {
    match v.is_nan() {
        true => 0.0,
        false => v.clamp(0.0, 1.0),
    }
}

/// A channel of `[0, 1]` through the sRGB curve in the kernel's steps
/// (`lower::arith::pack::srgb_encode`): `12.92 v` below 0.0031308, `1.055 v^(1/2.4) - 0.055`
/// elsewhere, each constant and step a `float`'s, the power the `f64` one rounded once.
fn srgb_encode(v: f32) -> f32 {
    let k = |x: f64| x as f32;
    match v < k(0.003_130_8) {
        true => v * k(12.92),
        false => (f64::from(v).powf(f64::from(k(1.0 / 2.4))) as f32) * k(1.055) - k(0.055),
    }
}

/// A channel's value as a converted format's conversion takes it: `v` itself, or an `_sRGB`
/// twin's (`srgb`) `v` of `[0, 1]` (a NaN 0) through the sRGB curve ([`srgb_encode`]).
fn encoded(v: f32, srgb: bool) -> f32 {
    match srgb {
        true => srgb_encode(unit(v)),
        false => v,
    }
}

/// A channel's `x = rint(clamp(v, 0, 1) · 2048)`, ties to even, a NaN 0.
fn grid_step(v: f32) -> i64 {
    if v.is_nan() {
        return 0;
    }
    let x = f64::from(v.clamp(0.0, 1.0)) * 2048.0;
    let below = x.floor();
    let rest = x - below;
    let up = rest > 0.5 || (rest == 0.5 && below % 2.0 == 1.0);
    (below as i64) + i64::from(up)
}

/// A channel's code of `max`: `round(clamp(v, 0, 1) · max)` of the exact product, to nearest (a
/// NaN 0; the product of an `f32` and a code scale is exact in an `f64`).
fn code_of(v: f32, max: u32) -> u16 {
    if v.is_nan() {
        return 0;
    }
    (f64::from(v.clamp(0.0, 1.0)) * f64::from(max) + 0.5).floor() as u16
}

/// Metal's luma, Cb and Cr codes of a texel written `[red, green, blue]` to a 4:4:4 texture of
/// `codes` over a surface of `range` and `matrix` (the module docs), each code of
/// `codes.bits` bits; `None` for a layout whose writes are not encoded (the packed formats, and
/// subsampled chroma, whose writes Metal drops).
pub fn ycbcr_write(
    codes: YCbCrCodes,
    (range, matrix): (YCbCrRange, YCbCrMatrix),
    [r, g, b]: [f32; 3],
) -> Option<[u16; 3]> {
    if codes.packed || codes.subsampling != ChromaSubsampling::Chroma444 {
        return None;
    }
    let max = (1u32 << codes.bits) - 1;
    match codes.output {
        YCbCrOutput::Codes => Some([code_of(g, max), code_of(b, max), code_of(r, max)]),
        YCbCrOutput::Converted | YCbCrOutput::Srgb => {
            let srgb = codes.output == YCbCrOutput::Srgb;
            let x = [r, g, b].map(|v| grid_step(encoded(v, srgb)));
            Some(write_rows(range, matrix, codes.bits).map(|(c, o)| {
                let t: i64 = c.iter().zip(x).map(|(&c, x)| i64::from(c) * x).sum();
                let code = i64::from(o) + ((t + (1 << (ROW_BITS - 1))) >> ROW_BITS);
                code.clamp(0, i64::from(max)) as u16
            }))
        }
    }
}

/// Metal's codes of a render pass's store of `texels` (each `[red, green, blue]`, row by row, of
/// a texture of `size` texels) into the planes of a two-plane YCbCr texture of `codes` over a
/// surface of `range` and `matrix` (the module docs): each texel's luma code, row by row, and each
/// chroma pair's Cb and Cr codes of the block of texels it covers, row by row of the chroma
/// plane. `None` for a layout whose store is not encoded (the 4:2:0 and 4:2:2 layouts read
/// through the sRGB curve, and packed ones), for a size the chroma blocks do not tile, or for
/// `texels` not `size`'s.
pub fn ycbcr_store(
    codes: YCbCrCodes,
    (range, matrix): (YCbCrRange, YCbCrMatrix),
    texels: &[[f32; 3]],
    size: [u32; 2],
) -> Option<(Vec<u16>, Vec<[u16; 2]>)> {
    let srgb = codes.output == YCbCrOutput::Srgb;
    if codes.packed || (srgb && codes.subsampling != ChromaSubsampling::Chroma444) {
        return None;
    }
    let [w, h] = size.map(|n| n as usize);
    let [sx, sy] = codes.subsampling.halved().map(usize::from);
    if texels.len() != w * h || w % (1 << sx) != 0 || h % (1 << sy) != 0 {
        return None;
    }
    let toward_zero = pass_rounds_toward_zero(codes);
    let k = (sx + sy) as u32;
    // Each chroma pair's block of texels, by their indices.
    let block = |cx: usize, cy: usize| {
        (0..1 << sy)
            .flat_map(move |dy| (0..1 << sx).map(move |dx| ((cy << sy) + dy) * w + (cx << sx) + dx))
    };
    if codes.output == YCbCrOutput::Codes {
        // Each channel's code of its `half`: luma green's, Cb blue's and Cr red's; a pair's the
        // mean of its block's, rounded half up.
        let max = (1u32 << codes.bits) - 1;
        let own: Vec<[u16; 3]> = texels
            .iter()
            .map(|&[r, g, b]| [g, b, r].map(|v| code_of(pass_half(v, toward_zero), max)))
            .collect();
        let luma = own.iter().map(|&[y, _, _]| y).collect();
        let mut chroma = Vec::with_capacity((w >> sx) * (h >> sy));
        for cy in 0..h >> sy {
            for cx in 0..w >> sx {
                let mut sums = [0u32; 2];
                for i in block(cx, cy) {
                    sums[0] += u32::from(own[i][1]);
                    sums[1] += u32::from(own[i][2]);
                }
                chroma.push(sums.map(|s| ((s + ((1 << k) >> 1)) >> k) as u16));
            }
        }
        return Some((luma, chroma));
    }
    let max = (1i64 << codes.bits) - 1;
    let rows = write_rows(range, matrix, codes.bits);
    let term_bits = term_bits(codes.bits);
    // A texel's row `c · x` counted at 2^-term_bits of a code, truncated.
    let term = |(c, _): ([i32; 3], u32), x: [i64; 3]| -> i64 {
        let t: i64 = c.iter().zip(x).map(|(&c, x)| i64::from(c) * x).sum();
        t >> (ROW_BITS - term_bits)
    };
    // A block's code of its `n = 2^k` terms' sum.
    let encode = |(_, o): ([i32; 3], u32), sum: i64, k: u32| {
        let shift = term_bits + k;
        let code = i64::from(o) + ((sum + (1 << (shift - 1))) >> shift);
        code.clamp(0, max) as u16
    };
    let steps: Vec<[i64; 3]> = texels
        .iter()
        .map(|t| t.map(|v| grid_step(encoded(pass_half(v, toward_zero), srgb))))
        .collect();
    let luma = steps
        .iter()
        .map(|&x| encode(rows[0], term(rows[0], x), 0))
        .collect();
    let mut chroma = Vec::with_capacity((w >> sx) * (h >> sy));
    for cy in 0..h >> sy {
        for cx in 0..w >> sx {
            let mut sums = [0i64; 2];
            for i in block(cx, cy) {
                sums[0] += term(rows[1], steps[i]);
                sums[1] += term(rows[2], steps[i]);
            }
            chroma.push([encode(rows[1], sums[0], k), encode(rows[2], sums[1], k)]);
        }
    }
    Some((luma, chroma))
}

/// The crate-owned YCbCr write module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YCbCrWriteModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl YCbCrWriteModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The texture's written texels: a 2D storage image `Rgba32ui`, read.
    pub const WRITTEN_BINDING: u32 = 0;
    /// The planes' shape and rows: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 1;
    /// The surface's bytes, 4 to a word: a storage buffer, read and written.
    pub const BYTES_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = word::ROWS + 3 * word::ROW_WORDS;
    /// The bits past [`ROW_BITS`]'s a code's sum is counted at: each code sums four reads of its
    /// block's texels (a luma code, and a 4:4:4 pair, four of one texel; a 4:2:2 pair, two of
    /// each of its two), so every sum is four times its block's mean.
    const READS_BITS: u32 = 2;
    /// Invocations of one workgroup along x, and along y: one word of a plane's row each.
    pub const LOCAL_SIZE: u32 = 8;

    /// The shape's bytes of `planes` (a layout [`ycbcr_write`] encodes, or, for a pass's store,
    /// `all`, one [`ycbcr_store`] encodes): with `all`, every texel's codes are stored, the marks
    /// not read (the module docs).
    pub fn shape(planes: &YCbCrPlanes, all: bool) -> [u8; Self::SHAPE_WORDS as usize * 4] {
        let codes = planes.codes;
        let [sub_x, sub_y] = codes.subsampling.halved().map(u32::from);
        let mut words = vec![
            planes.size[0],
            planes.size[1],
            planes.luma[0],
            planes.luma[1],
            planes.chroma[0],
            planes.chroma[1],
            codes.bytes,
            8 * codes.bytes - codes.bits,
            (1 << codes.bits) - 1,
            u32::from(codes.output == YCbCrOutput::Codes),
            sub_x,
            sub_y,
            u32::from(all),
        ];
        let bits = codes.bits.clamp(8, 10);
        let term_bits = term_bits(bits);
        words.push(ROW_BITS - term_bits - 12);
        words.push(u32::from(all && pass_rounds_toward_zero(codes)));
        words.push(u32::from(codes.output == YCbCrOutput::Srgb));
        let rows = write_rows(planes.range, planes.matrix, bits);
        for (c, o) in rows {
            // `c = hi · 2^12 + lo`, `lo` in `[0, 2^12)`; the offset and the rounding as
            // `(2 o + 1) · 2^(T - 1 + READS_BITS)` of the terms' `2^(26 - T)`, `T` the term's
            // bits, the sums being of four reads.
            words.extend(c.map(|c| (c >> 12) as u32));
            words.extend(c.map(|c| (c & 0xfff) as u32));
            words.push((2 * o + 1) << (term_bits - 1 + Self::READS_BITS));
        }
        debug_assert_eq!(words.len(), Self::SHAPE_WORDS as usize);
        let mut bytes = [0; Self::SHAPE_WORDS as usize * 4];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of the dispatch over `planes`: across, the words of the wider plane's row
    /// (the luma plane's a code `bytes` bytes, the chroma plane's a pair `2 · bytes`), and down,
    /// the rows of both planes.
    pub fn groups(planes: &YCbCrPlanes) -> [u32; 2] {
        let [chroma_width, chroma_height] = planes.chroma_size();
        let luma = planes.size[0] * planes.codes.bytes;
        let words = luma.max(2 * chroma_width * planes.codes.bytes).div_ceil(4);
        [words, planes.size[1] + chroma_height].map(|n| n.div_ceil(Self::LOCAL_SIZE))
    }
}

/// The shape's words, in order (written by [`YCbCrWriteModule::shape`]).
mod word {
    pub(super) const WIDTH: u32 = 0;
    pub(super) const HEIGHT: u32 = 1;
    pub(super) const LUMA_OFFSET: u32 = 2;
    pub(super) const LUMA_ROW: u32 = 3;
    pub(super) const CHROMA_OFFSET: u32 = 4;
    pub(super) const CHROMA_ROW: u32 = 5;
    /// The bytes of a code: 1 or 2.
    pub(super) const CODE_BYTES: u32 = 6;
    /// The shift that takes a code to its place in its bytes: `8 · bytes - bits`.
    pub(super) const CODE_SHIFT: u32 = 7;
    /// The largest code, `2^bits - 1`.
    pub(super) const MAX_CODE: u32 = 8;
    /// 1 where the codes are the channels' own (`YCbCrOutput::Codes`), 0 converted.
    pub(super) const CODES: u32 = 9;
    /// 1 where the chroma is halved across (4:2:0, 4:2:2), else 0.
    pub(super) const SUB_X: u32 = 10;
    /// 1 where the chroma is halved down (4:2:0), else 0.
    pub(super) const SUB_Y: u32 = 11;
    /// 1 where every texel's codes are stored (a pass's store), the marks not read.
    pub(super) const ALL: u32 = 12;
    /// The shift that takes a texel's term, `hi · x + (lo · x >> 12)`, to its bits past the
    /// code: 8 at 8 bits, 10 at 10.
    pub(super) const TERM_SHIFT: u32 = 13;
    /// 1 where a pass's store takes each channel's `half` toward zero (10-bit codes), else 0.
    pub(super) const TOWARD_ZERO: u32 = 14;
    /// 1 where each channel goes through the sRGB curve before its conversion (an `_sRGB` twin's
    /// codes, `YCbCrOutput::Srgb`), else 0.
    pub(super) const SRGB: u32 = 15;
    /// Luma's row: the high parts of red's, green's and blue's `c`, their low 12 bits, and the
    /// offset; Cb's and Cr's follow, [`ROW_WORDS`] apart.
    pub(super) const ROWS: u32 = 16;
    /// The words of a row.
    pub(super) const ROW_WORDS: u32 = 7;
}

/// The binding symbol of the written texels in the typed module.
const WRITTEN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the shape in the typed module.
const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the surface's bytes in the typed module.
const BYTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The values one invocation's codes read.
#[derive(Clone, Copy)]
struct Cx {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    zero: Value,
    one: Value,
    shape: Value,
}

impl Cx {
    fn uint(&self, f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        scalar(f, self.u32_, u64::from(v))
    }

    /// The shape's word at `index` (a `u32` value).
    fn field(&self, f: &mut FunctionBuilder<'_>, index: Value) -> BResult<Value> {
        let p = f.access_chain(self.shape, &[self.zero, index])?;
        f.load(p, MemAccess::default())
    }

    /// Channel `bits` (a `u32` of `float` bits) as a value of `[0, 1]`: a NaN 0, the rest
    /// clamped.
    fn unit(&self, f: &mut FunctionBuilder<'_>, bits: Value) -> BResult<Value> {
        let v = f.convert(ConvOp::Bitcast, bits, self.f32_)?;
        let nan = f.unary(UnOp::IsNan, v)?;
        let zero = scalar(f, self.f32_, 0)?;
        let one = scalar(f, self.f32_, u64::from(1.0f32.to_bits()))?;
        let v = f.select(nan, zero, v)?;
        f.glsl(GlslOp::FClamp, &[v, zero, one])
    }

    /// `x = rint(v · 2048)`, ties to even, of channel `bits` (a `u32`).
    fn grid_step(&self, f: &mut FunctionBuilder<'_>, bits: Value) -> BResult<Value> {
        let v = self.unit(f, bits)?;
        let scale = scalar(f, self.f32_, u64::from(2048.0f32.to_bits()))?;
        let x = f.binary(BinOp::FMul, v, scale)?;
        let x = f.glsl(GlslOp::RoundEven, &[x])?;
        f.convert(ConvOp::ConvertFToU, x, self.u32_)
    }

    /// Channel `bits` (a `u32` of `float` bits) through the sRGB curve, as its `float` bits (a
    /// `u32`): the value of `[0, 1]` (a NaN 0) by `lower::arith::pack::srgb_encode`, the device's
    /// `Pow`.
    fn srgb(&self, f: &mut FunctionBuilder<'_>, bits: Value) -> BResult<Value> {
        let v = self.unit(f, bits)?;
        let s = pack::srgb_encode(f, SCALAR, v)?;
        f.convert(ConvOp::Bitcast, s, self.u32_)
    }

    /// A pass's channel `bits` (a `u32`) as its `half`'s `float` bits (a `u32`): the value of
    /// `[0, 1]` rounded to a `half`, to nearest even, or toward zero where `toward_zero` (a
    /// `bool`). Of `v` of exponent `E` (at least -14), `q = rint(v · 2^(10 - E))` (or its floor)
    /// is the half's significand and `q · 2^(E - 10)` its value, each product of a power of two
    /// exact.
    fn pass_half(
        &self,
        f: &mut FunctionBuilder<'_>,
        bits: Value,
        toward_zero: Value,
    ) -> BResult<Value> {
        let v = self.unit(f, bits)?;
        let raw = f.convert(ConvOp::Bitcast, v, self.u32_)?;
        let twenty_three = self.uint(f, 23)?;
        let exponent = f.binary(BinOp::ShiftRightLogical, raw, twenty_three)?;
        let byte = self.uint(f, 0xff)?;
        let exponent = f.binary(BinOp::BitwiseAnd, exponent, byte)?;
        let least = self.uint(f, 127 - 14)?;
        let exponent = f.glsl(GlslOp::UMax, &[exponent, least])?;
        // `2^(10 - E)` and `2^(E - 10)` as floats: biased exponents `264 - e` and `e - 10`.
        let top = self.uint(f, 127 + 10 + 127)?;
        let up = f.binary(BinOp::ISub, top, exponent)?;
        let up = f.binary(BinOp::ShiftLeftLogical, up, twenty_three)?;
        let up = f.convert(ConvOp::Bitcast, up, self.f32_)?;
        let ten = self.uint(f, 10)?;
        let down = f.binary(BinOp::ISub, exponent, ten)?;
        let down = f.binary(BinOp::ShiftLeftLogical, down, twenty_three)?;
        let down = f.convert(ConvOp::Bitcast, down, self.f32_)?;
        let q = f.binary(BinOp::FMul, v, up)?;
        let nearest = f.glsl(GlslOp::RoundEven, &[q])?;
        let below = f.glsl(GlslOp::Floor, &[q])?;
        let q = f.select(toward_zero, below, nearest)?;
        let h = f.binary(BinOp::FMul, q, down)?;
        f.convert(ConvOp::Bitcast, h, self.u32_)
    }

    /// The code of channel `bits` (a `u32`) of `max` (a `u32`, at most 1023):
    /// `round(v · max)` of the exact product, to nearest. Of `v` in `(0, 1)`, `M · 2^-s` with
    /// `M` its significand (its implicit bit included; a subnormal's at `s = 149`) and `s` at
    /// least 24, the product `M · max = A · 2^12 + B` with `A = (M >> 12) · max + (B' >> 12)`
    /// and `B'`, `(M & 4095) · max`, below `2^22`; the code is `(A + 2^(t - 1)) >> t` for
    /// `t = s - 12`, 0 from `t` 25 on (the product is below 1/2 there).
    fn code(&self, f: &mut FunctionBuilder<'_>, bits: Value, max: Value) -> BResult<Value> {
        let v = self.unit(f, bits)?;
        let raw = f.convert(ConvOp::Bitcast, v, self.u32_)?;
        // `-0` clamps to itself: its sign bit goes.
        let magnitude = self.uint(f, 0x7fff_ffff)?;
        let raw = f.binary(BinOp::BitwiseAnd, raw, magnitude)?;
        let twenty_three = self.uint(f, 23)?;
        let exponent = f.binary(BinOp::ShiftRightLogical, raw, twenty_three)?;
        let mantissa_mask = self.uint(f, 0x7f_ffff)?;
        let fraction = f.binary(BinOp::BitwiseAnd, raw, mantissa_mask)?;
        let normal = f.binary(BinOp::INotEqual, exponent, self.zero)?;
        let implicit = self.uint(f, 0x80_0000)?;
        let lead = f.select(normal, implicit, self.zero)?;
        let m = f.binary(BinOp::BitwiseOr, fraction, lead)?;
        let exponent = f.glsl(GlslOp::UMax, &[exponent, self.one])?;
        // `t = s - 12 = 138 - exponent`, at least 12 below 1.
        let base = self.uint(f, 138)?;
        let t = f.binary(BinOp::ISub, base, exponent)?;
        let twelve = self.uint(f, 12)?;
        let low_mask = self.uint(f, 0xfff)?;
        let high = f.binary(BinOp::ShiftRightLogical, m, twelve)?;
        let high = f.binary(BinOp::IMul, high, max)?;
        let low = f.binary(BinOp::BitwiseAnd, m, low_mask)?;
        let low = f.binary(BinOp::IMul, low, max)?;
        let low = f.binary(BinOp::ShiftRightLogical, low, twelve)?;
        let a = f.binary(BinOp::IAdd, high, low)?;
        let tame = self.uint(f, 24)?;
        let small = f.binary(BinOp::UGreaterThan, t, tame)?;
        let shift = f.glsl(GlslOp::UMin, &[t, tame])?;
        let half_at = f.binary(BinOp::ISub, shift, self.one)?;
        let half = f.binary(BinOp::ShiftLeftLogical, self.one, half_at)?;
        let a = f.binary(BinOp::IAdd, a, half)?;
        let code = f.binary(BinOp::ShiftRightLogical, a, shift)?;
        let code = f.select(small, self.zero, code)?;
        // 1 itself is `max`.
        let one = scalar(f, self.f32_, u64::from(1.0f32.to_bits()))?;
        let whole = f.binary(BinOp::FOrdGreaterThanEqual, v, one)?;
        f.select(whole, max, code)
    }

    /// The code `component` (0 luma, 1 Cb, 2 Cr; a `u32`) of the four reads `texels` (each a
    /// `uvec4`) of its block: codes, the mean of the reads' own codes rounded half up,
    /// `(Σ code + 2) >> 2`, each of a pass's `half` ([`pass_half`](Self::pass_half), toward
    /// zero where `toward_zero`, a `bool`) where `pass` (a `bool`); converted, the sum of the
    /// reads' terms ([`YCbCrWriteModule::READS_BITS`]), each its row `c · x` counted at 2^-T of a
    /// code and truncated, `(Σ hi · x + (Σ lo · x >> 12)) >> S` for the shape's `term_shift`
    /// `S = 14 - T`, its `x` the [`grid_step`](Self::grid_step) of the channel, of its pass's
    /// `half` where `pass`, through the sRGB curve ([`srgb`](Self::srgb)) where `srgb` (a
    /// `bool`).
    fn code_of(
        &self,
        f: &mut FunctionBuilder<'_>,
        texels: [Value; 4],
        component: Value,
        (max, codes, pass): (Value, Value, Value),
        (term_shift, toward_zero, srgb): (Value, Value, Value),
    ) -> BResult<Value> {
        // Codes: luma the green channel, Cb the blue, Cr the red.
        let is_luma = f.binary(BinOp::IEqual, component, self.zero)?;
        let is_cb = f.binary(BinOp::IEqual, component, self.one)?;
        let mut own = self.zero;
        for &texel in &texels {
            let channels: Vec<Value> = (0..3)
                .map(|c| f.composite_extract(texel, &[c]))
                .collect::<BResult<_>>()?;
            let channel = f.select(is_cb, channels[2], channels[0])?;
            let channel = f.select(is_luma, channels[1], channel)?;
            let halved = self.pass_half(f, channel, toward_zero)?;
            let channel = f.select(pass, halved, channel)?;
            let code = self.code(f, channel, max)?;
            own = f.binary(BinOp::IAdd, own, code)?;
        }
        let two = self.uint(f, 2)?;
        let own = f.binary(BinOp::IAdd, own, two)?;
        let own = f.binary(BinOp::ShiftRightLogical, own, two)?;
        // Converted: the row of the code.
        let row_words = self.uint(f, word::ROW_WORDS)?;
        let at = f.binary(BinOp::IMul, component, row_words)?;
        let rows = self.uint(f, word::ROWS)?;
        let at = f.binary(BinOp::IAdd, at, rows)?;
        let mut row = Vec::with_capacity(3);
        for c in 0..3 {
            let index = self.uint(f, c)?;
            let i = f.binary(BinOp::IAdd, at, index)?;
            let hi = self.field(f, i)?;
            let three = self.uint(f, 3)?;
            let i = f.binary(BinOp::IAdd, i, three)?;
            let lo = self.field(f, i)?;
            row.push((hi, lo));
        }
        let twelve = self.uint(f, 12)?;
        let mut sum = scalar(f, self.i32_, 0)?;
        for &texel in &texels {
            let mut high = self.zero;
            let mut low = self.zero;
            for (c, &(hi, lo)) in row.iter().enumerate() {
                let channel = f.composite_extract(texel, &[c as u32])?;
                let halved = self.pass_half(f, channel, toward_zero)?;
                let channel = f.select(pass, halved, channel)?;
                let encoded = self.srgb(f, channel)?;
                let channel = f.select(srgb, encoded, channel)?;
                let x = self.grid_step(f, channel)?;
                let term = f.binary(BinOp::IMul, hi, x)?;
                high = f.binary(BinOp::IAdd, high, term)?;
                let term = f.binary(BinOp::IMul, lo, x)?;
                low = f.binary(BinOp::IAdd, low, term)?;
            }
            let low = f.binary(BinOp::ShiftRightLogical, low, twelve)?;
            let term = f.binary(BinOp::IAdd, high, low)?;
            let term = f.convert(ConvOp::Bitcast, term, self.i32_)?;
            let term = f.binary(BinOp::ShiftRightArithmetic, term, term_shift)?;
            sum = f.binary(BinOp::IAdd, sum, term)?;
        }
        let six = self.uint(f, 6)?;
        let i = f.binary(BinOp::IAdd, at, six)?;
        let offset = self.field(f, i)?;
        let offset = f.convert(ConvOp::Bitcast, offset, self.i32_)?;
        let sum = f.binary(BinOp::IAdd, sum, offset)?;
        // The sum's shift, `T + READS_BITS`: `ROW_BITS - 12 + READS_BITS - S`.
        let base = self.uint(f, ROW_BITS - 12 + YCbCrWriteModule::READS_BITS)?;
        let shift = f.binary(BinOp::ISub, base, term_shift)?;
        let code = f.binary(BinOp::ShiftRightArithmetic, sum, shift)?;
        let zero = scalar(f, self.i32_, 0)?;
        let top = f.convert(ConvOp::Bitcast, max, self.i32_)?;
        let code = f.glsl(GlslOp::SClamp, &[code, zero, top])?;
        let converted = f.convert(ConvOp::Bitcast, code, self.u32_)?;
        let is_codes = f.binary(BinOp::INotEqual, codes, self.zero)?;
        f.select(is_codes, own, converted)
    }
}

/// The module as a typed module, before serialization: the module [`ycbcr_write_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `WRITTEN`, `SHAPE` and
/// `BYTES` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type: u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: crate::spv::ImageFormat::Rgba32ui,
    }))?;
    let written = b.resource(WRITTEN, StorageClass::UniformConstant, image_ty)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let bytes = words_buffer(&mut b, BYTES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let one = scalar(&mut f, u32_, 1)?;
    let shape = f.module().global_value(shape)?;
    let bytes = f.module().global_value(bytes)?;
    let cx = Cx {
        u32_,
        i32_,
        f32_,
        zero,
        one,
        shape,
    };
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = cx.uint(f, index)?;
        cx.field(f, index)
    };
    let width = field(&mut f, word::WIDTH)?;
    let height = field(&mut f, word::HEIGHT)?;
    let unit = field(&mut f, word::CODE_BYTES)?;
    let code_shift = field(&mut f, word::CODE_SHIFT)?;
    let max = field(&mut f, word::MAX_CODE)?;
    let codes = field(&mut f, word::CODES)?;
    let sub_x = field(&mut f, word::SUB_X)?;
    let sub_y = field(&mut f, word::SUB_Y)?;
    let all = field(&mut f, word::ALL)?;
    let all = f.binary(BinOp::INotEqual, all, zero)?;
    let term_shift = field(&mut f, word::TERM_SHIFT)?;
    let toward_zero = field(&mut f, word::TOWARD_ZERO)?;
    let toward_zero = f.binary(BinOp::INotEqual, toward_zero, zero)?;
    let srgb = field(&mut f, word::SRGB)?;
    let srgb = f.binary(BinOp::INotEqual, srgb, zero)?;
    // Rows `[0, h)` are the luma plane's, `[h, h + ch)` the chroma plane's, `ch` its rows.
    let chroma = f.binary(BinOp::UGreaterThanEqual, y, height)?;
    let below = f.binary(BinOp::ISub, y, height)?;
    let row = f.select(chroma, below, y)?;
    let chroma_height = f.binary(BinOp::ShiftRightLogical, height, sub_y)?;
    let tall = f.binary(BinOp::IAdd, height, chroma_height)?;
    let in_rows = f.binary(BinOp::ULessThan, y, tall)?;
    // A row's codes: `w` luma codes, or `2 cw` chroma ones (a Cb and a Cr of each pair, `cw` the
    // pairs of a row).
    let chroma_width = f.binary(BinOp::ShiftRightLogical, width, sub_x)?;
    let two_w = f.binary(BinOp::IAdd, chroma_width, chroma_width)?;
    let n = f.select(chroma, two_w, width)?;
    let four = cx.uint(&mut f, 4)?;
    let per_word = f.binary(BinOp::UDiv, four, unit)?;
    let first = f.binary(BinOp::IMul, x, per_word)?;
    let in_row = f.binary(BinOp::ULessThan, first, n)?;
    let inside = f.binary(BinOp::LogicalAnd, in_rows, in_row)?;
    let luma_offset = field(&mut f, word::LUMA_OFFSET)?;
    let luma_row = field(&mut f, word::LUMA_ROW)?;
    let chroma_offset = field(&mut f, word::CHROMA_OFFSET)?;
    let chroma_row = field(&mut f, word::CHROMA_ROW)?;
    let written = f.module().global_value(written)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        let offset = f.select(chroma, chroma_offset, luma_offset)?;
        let pitch = f.select(chroma, chroma_row, luma_row)?;
        let start = f.binary(BinOp::IMul, row, pitch)?;
        let start = f.binary(BinOp::IAdd, offset, start)?;
        let two = cx.uint(f, 2)?;
        let start = f.binary(BinOp::ShiftRightLogical, start, two)?;
        let index = f.binary(BinOp::IAdd, start, x)?;
        let p = f.access_chain(bytes, &[zero, index])?;
        let old = f.load(p, MemAccess::default())?;
        let eight = cx.uint(f, 8)?;
        let bits = f.binary(BinOp::IMul, unit, eight)?;
        // A code's mask in its unit: a byte's, or a 16-bit word's.
        let ones = cx.uint(f, u32::MAX)?;
        let thirty_two = cx.uint(f, 32)?;
        let unused = f.binary(BinOp::ISub, thirty_two, bits)?;
        let mask = f.binary(BinOp::ShiftRightLogical, ones, unused)?;
        let image = f.load(written, MemAccess::default())?;
        let mut word = old;
        for k in 0..4u32 {
            let k_ = cx.uint(f, k)?;
            let i = f.binary(BinOp::IAdd, first, k_)?;
            let in_word = f.binary(BinOp::ULessThan, k_, per_word)?;
            let in_n = f.binary(BinOp::ULessThan, i, n)?;
            let live = f.binary(BinOp::LogicalAnd, in_word, in_n)?;
            // The block of code `i`: texel `i` of the luma plane's row, or the texels pair
            // `i / 2` of the chroma plane's row covers (the even codes Cb, the odd ones Cr), from
            // its first texel at `(pair << sub_x, row << sub_y)`, read four times: each read
            // `(dx & ax, dy & ay)` past the first, `ax` and `ay` the halved axes (none for luma).
            let pair = f.binary(BinOp::ShiftRightLogical, i, one)?;
            let across = f.binary(BinOp::ShiftLeftLogical, pair, sub_x)?;
            let column = f.select(chroma, across, i)?;
            let down = f.binary(BinOp::ShiftLeftLogical, row, sub_y)?;
            let first_row = f.select(chroma, down, row)?;
            let ax = f.select(chroma, sub_x, zero)?;
            let ay = f.select(chroma, sub_y, zero)?;
            let last = f.binary(BinOp::ISub, width, one)?;
            let bottom = f.binary(BinOp::ISub, height, one)?;
            let mut texels = [zero; 4];
            for (t, texel) in texels.iter_mut().enumerate() {
                let dx = cx.uint(f, (t & 1) as u32)?;
                let dy = cx.uint(f, (t >> 1) as u32)?;
                let dx = f.binary(BinOp::BitwiseAnd, dx, ax)?;
                let dy = f.binary(BinOp::BitwiseAnd, dy, ay)?;
                let c = f.binary(BinOp::IAdd, column, dx)?;
                let c = f.glsl(GlslOp::UMin, &[c, last])?;
                let r = f.binary(BinOp::IAdd, first_row, dy)?;
                let r = f.glsl(GlslOp::UMin, &[r, bottom])?;
                let coord = f.composite_construct(uvec2, &[c, r])?;
                *texel = f.read(
                    image,
                    coord,
                    &ImageOperands::default(),
                    TexelMemory::PRIVATE,
                )?;
            }
            // A function's write marks its texel; a pass's store stores every texel.
            let mark = f.composite_extract(texels[0], &[3])?;
            let marked = f.binary(BinOp::INotEqual, mark, zero)?;
            let marked = f.binary(BinOp::LogicalOr, marked, all)?;
            let live = f.binary(BinOp::LogicalAnd, live, marked)?;
            let odd = f.binary(BinOp::BitwiseAnd, i, one)?;
            let cb_or_cr = f.binary(BinOp::IAdd, odd, one)?;
            let component = f.select(chroma, cb_or_cr, zero)?;
            let code = cx.code_of(
                f,
                texels,
                component,
                (max, codes, all),
                (term_shift, toward_zero, srgb),
            )?;
            let code = f.binary(BinOp::ShiftLeftLogical, code, code_shift)?;
            let at = f.binary(BinOp::IMul, k_, bits)?;
            // A slot past the word's codes (the third and fourth of a word of two 16-bit
            // codes) is not live, and its shift stays in range: SPIR-V leaves a shift by the
            // width or more undefined.
            let thirty_one = cx.uint(f, 31)?;
            let at = f.binary(BinOp::BitwiseAnd, at, thirty_one)?;
            let placed = f.binary(BinOp::ShiftLeftLogical, code, at)?;
            let cleared = f.binary(BinOp::ShiftLeftLogical, mask, at)?;
            let kept = f.binary(BinOp::BitwiseXor, cleared, ones)?;
            let kept = f.binary(BinOp::BitwiseAnd, word, kept)?;
            let replaced = f.binary(BinOp::BitwiseOr, kept, placed)?;
            word = f.select(live, replaced, word)?;
        }
        let changed = f.binary(BinOp::INotEqual, word, old)?;
        let store: ArmFn<'_, ()> = Box::new(move |f, _| {
            f.store(p, word, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), changed, store, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, body, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: YCbCrWriteModule::LOCAL_SIZE,
            y: YCbCrWriteModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(YCbCrWriteModule::SET),
        Symbol::Binding(s) if *s == WRITTEN => Some(YCbCrWriteModule::WRITTEN_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(YCbCrWriteModule::SHAPE_BINDING),
        Symbol::Binding(s) if *s == BYTES => Some(YCbCrWriteModule::BYTES_BINDING),
        _ => None,
    }
}

/// The YCbCr write module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`YCbCrWriteModule`].
pub fn ycbcr_write_module(device: &Device) -> Result<YCbCrWriteModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the YCbCr write module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(YCbCrWriteModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
