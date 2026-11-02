//! `heapTextureSizeAndAlignWithDescriptor:` for the 26 formats this crate holds no image of
//! (`metal::format::footprint`) and ASTC's 4x2 and 8x4 formats, whose blocks the runtime holds
//! in a carrier and whose heap textures it answers Metal's model for (`heap_tiled::modelled_size`,
//! as for every ASTC texture in a carrier): Metal's size and alignment against the runtime's, over
//! two declared descriptor spaces enumerated whole, plus the rows that name each family's rule.
//!
//! **Measured** on an Apple M3 Max under macOS 27.0.1 (26A434), plain Metal with no API
//! validation layer, by a probe that forks per (format, texture type) so a descriptor Metal's
//! validation would stop costs that shape alone. Over the two spaces here and the wider sweep
//! they were cut from — every level count of ten shapes, 1 to 5 array slices, 1 to 32 depth
//! slices, both storage modes, and the `format-footprints-*` conformance recordings —
//! **193,296** answers, and the runtime matches every one.
//!
//! The spaces, enumerated in the order stated so one number stands for all of them:
//!
//! - [`SIDES`] x [`SIDES`] as a 2D texture of `ShaderRead` usage, `Private`, with
//!   `allowGPUOptimizedContents`: 1225 answers a format ([`Row::sides`]). This is what the tiles'
//!   padding, the lossless-compression threshold and a plane's packed rows show in.
//! - [`SHAPES`] in **every texture type the format takes**, under [`USAGES`] with
//!   `allowGPUOptimizedContents` both ways, `Private`: 768 answers a format, 480 for an ASTC
//!   footprint, which takes no one-dimensional or buffer descriptor ([`Row::usages`]). This is
//!   what the usage's effect on compression, the array and cube rules, and the one-plane 4:2:2
//!   even-side rule show in.
//!
//! The query is a device call and these are not device tests, so each row replays what the query
//! composes for such a format and nothing else: the descriptor's own check
//! (`TextureDescriptor::check_made`, whose `Nil` is Metal's "no texture of this shape" and
//! answers 0 bytes at [`no_texture_alignment`]) and then
//! [`footprint_answer`](crate::texture::heap::footprint_answer). `Device::heap_texture_size_and_align`
//! calls exactly those two, in that order, before it plans any image; for an ASTC 4x2 or 8x4
//! texture, whose heap texture the runtime refuses its making (`DecodedHeapTexture`), it then
//! answers [`modelled_size`](crate::texture::heap_tiled::modelled_size).

use super::super::texture::heap::{footprint_answer, no_texture_alignment};
use super::super::texture::heap_tiled::modelled_size;
use super::super::texture_descriptor::{
    effective_usage, Creator, Refused, TextureDescriptor, TextureRules,
};
use super::super::values::{ResourceOptions, StorageMode};
use crate::heap::SizeAndAlign;
use metal2vulkan::bind::AstcDecode;
use metal2vulkan::metal::{PixelFormat as P, TextureType as T, TextureUsage};

/// The device the probe ran on, as the descriptor rules see it: an Apple GPU.
const APPLE: TextureRules = TextureRules {
    sample_counts: super::super::texture_descriptor::APPLE_SAMPLE_COUNTS,
};

/// The sides the 2D sweep walks: every one from 1 to 20, then the powers of two and their
/// neighbours up to 257. A tile's padding steps at a power of two, so a side on either side of
/// one is where two layouts tell each other apart.
const SIDES: [u64; 35] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 47, 63, 64,
    65, 100, 127, 128, 129, 200, 255, 256, 257,
];

/// The shapes the usage matrix walks, chosen around the 16-texel threshold where Metal starts
/// keeping a texture losslessly compressed, with one odd-sided pair for the 4:2:2 rule.
const SHAPES: [(u64, u64); 6] = [(15, 15), (16, 16), (16, 15), (17, 17), (64, 64), (100, 70)];

/// The usages the matrix walks: `Unknown` (every usage), `ShaderRead`, `ShaderWrite`, both,
/// `RenderTarget`, `ShaderRead | RenderTarget`, `PixelFormatView` and `ShaderRead |
/// PixelFormatView`. `ShaderAtomic` is left out: Metal ends its own process on it for any format
/// but `R32Uint`.
const USAGES: [u64; 8] = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x10, 0x11];

/// The eight texture types, in the order the spaces enumerate them.
const TYPES: [T; 8] = [
    T::Type1D,
    T::Type1DArray,
    T::Type2D,
    T::Type2DArray,
    T::Cube,
    T::CubeArray,
    T::Type3D,
    T::TextureBuffer,
];

/// The five an ASTC footprint takes: Metal's validation stops a compressed format's
/// one-dimensional or buffer descriptor and ends its process, so the probe asked none.
const BLOCKY: [T; 5] = [T::Type2D, T::Type2DArray, T::Cube, T::CubeArray, T::Type3D];

/// One format's row: the answers of each space, as the probe wrote them — how many, and the
/// 64-bit FNV-1a of the `(size, align)` pairs in order, each as eight little-endian bytes.
struct Row {
    format: PixelFormat,
    /// `(answers, digest)` of the 2D side sweep.
    sides: (usize, u64),
    /// `(answers, digest)` of the usage matrix over every type the format takes.
    usages: (usize, u64),
}

type PixelFormat = P;

const fn row(format: PixelFormat, sides: (usize, u64), usages: (usize, u64)) -> Row {
    Row {
        format,
        sides,
        usages,
    }
}

/// Metal's answers, by format: ASTC's six 4x2 and 8x4 formats, the two plain stragglers, the nine
/// one-plane YCbCr formats, the ten padded 10- and 12-bit display formats, the four two-plane
/// ones and the two packed ones Metal makes no texture of.
const ROWS: &[Row] = &[
    row(
        P::ASTC4x2SRGB,
        (1225, 0x049daf078798fe57),
        (480, 0x1b8a1f9fa5c98125),
    ),
    row(
        P::ASTC4x2LDR,
        (1225, 0x049daf078798fe57),
        (480, 0x1b8a1f9fa5c98125),
    ),
    row(
        P::ASTC4x2HDR,
        (1225, 0x049daf078798fe57),
        (480, 0x1b8a1f9fa5c98125),
    ),
    row(
        P::ASTC8x4SRGB,
        (1225, 0x1891c3dd51619e4c),
        (480, 0x5bd85c333254a925),
    ),
    row(
        P::ASTC8x4LDR,
        (1225, 0x1891c3dd51619e4c),
        (480, 0x5bd85c333254a925),
    ),
    row(
        P::ASTC8x4HDR,
        (1225, 0x1891c3dd51619e4c),
        (480, 0x5bd85c333254a925),
    ),
    row(
        P::ARGB8UnormSRGB,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::YCBCR84221P,
        (1225, 0xff7620524e201368),
        (768, 0x40bd142d5d2d1ddd),
    ),
    row(
        P::YCBCR104441P,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::YCBCR84221PSRGB,
        (1225, 0xff7620524e201368),
        (768, 0x40bd142d5d2d1ddd),
    ),
    row(
        P::YCBCR104441PSRGB,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::YCBCRA84441P,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::YCBCR84221PREV,
        (1225, 0xff7620524e201368),
        (768, 0x40bd142d5d2d1ddd),
    ),
    row(
        P::YCBCR104441PPQ,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::YCBCR104441PXR,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::YCBCR104441PHLG,
        (1225, 0x90850694a55ed5f9),
        (768, 0xb0578d212e05e325),
    ),
    row(
        P::RGB10A2UnormPQ,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::R12UnormX4,
        (1225, 0xb88a6cd02f761c38),
        (768, 0x7b026068a93b714e),
    ),
    row(
        P::R12UnormX4PQ,
        (1225, 0xb88a6cd02f761c38),
        (768, 0x7b026068a93b714e),
    ),
    row(
        P::RG12UnormX8,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::R10UnormX6PQ,
        (1225, 0xb88a6cd02f761c38),
        (768, 0x7b026068a93b714e),
    ),
    row(
        P::R10UnormX6HLG,
        (1225, 0xb88a6cd02f761c38),
        (768, 0x7b026068a93b714e),
    ),
    row(
        P::R12UnormX4HLG,
        (1225, 0xb88a6cd02f761c38),
        (768, 0x7b026068a93b714e),
    ),
    row(
        P::RGB10A2UnormHLG,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::BGR10A2UnormSRGB,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::BGR10A2UnormPQ,
        (1225, 0x756637a2ca84adc5),
        (768, 0x9ec81e6bf1320ade),
    ),
    row(
        P::RGB10A82P,
        (1225, 0x67ed21710a6f652c),
        (768, 0x59dc0fc41dd6a845),
    ),
    row(
        P::RGB10A82PSRGB,
        (1225, 0x67ed21710a6f652c),
        (768, 0x59dc0fc41dd6a845),
    ),
    row(
        P::RGB10A82PPQ,
        (1225, 0x67ed21710a6f652c),
        (768, 0x59dc0fc41dd6a845),
    ),
    row(
        P::RGB10A82PHLG,
        (1225, 0x67ed21710a6f652c),
        (768, 0x59dc0fc41dd6a845),
    ),
    row(
        P::R10UnormPACKED,
        (1225, 0x72d850ede4c6d075),
        (768, 0xdd536e9534372325),
    ),
    row(
        P::RG10UnormPACKED,
        (1225, 0x72d850ede4c6d075),
        (768, 0xdd536e9534372325),
    ),
];

/// The descriptor of one row of a space.
#[allow(clippy::too_many_arguments)]
fn descriptor(
    format: PixelFormat,
    texture_type: T,
    [width, height, depth]: [u64; 3],
    levels: u64,
    slices: u64,
    usage: u64,
    optimized: bool,
) -> TextureDescriptor {
    TextureDescriptor {
        texture_type,
        pixel_format: format,
        width,
        height,
        depth,
        mipmap_level_count: levels,
        array_length: slices,
        usage: TextureUsage::try_from(u32::try_from(usage).unwrap()).unwrap(),
        resource_options: ResourceOptions::default().with_storage_mode(StorageMode::Private),
        allow_gpu_optimized_contents: optimized,
        ..TextureDescriptor::default()
    }
}

/// What `Device::heap_texture_size_and_align` answers for `d`, for a format whose footprint the
/// format's own structure decides: 0 bytes at Metal's alignment where the descriptor's check
/// answers nil, else the footprint's. Panics where the descriptor is one Metal's validation stops
/// or the footprint has no answer, neither of which a row of these spaces is.
fn answer(d: &TextureDescriptor) -> SizeAndAlign {
    match d.check_made(Creator::SizeQuery, APPLE) {
        Err(Refused::Nil(_)) => SizeAndAlign {
            size: 0,
            align: no_texture_alignment(d.pixel_format, d.texture_type),
        },
        Err(refused) => panic!("{:?} {:?}: {refused:?}", d.pixel_format, d.texture_type),
        Ok(layout) => {
            let usage = effective_usage(d.usage);
            let optimized = d.allow_gpu_optimized_contents;
            let carried = AstcDecode::of(d.pixel_format).is_some();
            footprint_answer(&layout, usage, optimized)
                .or_else(|| carried.then(|| modelled_size(&layout, usage, optimized))?)
        }
        .unwrap_or_else(|| {
            panic!(
                "{:?} {:?} {}x{}: the footprint has no answer",
                d.pixel_format, d.texture_type, d.width, d.height
            )
        }),
    }
}

/// The 64-bit FNV-1a of the answers in order, each number eight little-endian bytes: the digest
/// the probe's output is reduced by, so one constant stands for a whole space.
fn digest(answers: &[u64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in answers {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// The level-0 extent, level count and array length a type takes in the spaces: a cube is square,
/// a one-dimensional or buffer texture one row high, an array two slices.
fn shape(texture_type: T, (width, height): (u64, u64)) -> ([u64; 3], u64, u64) {
    let cube = matches!(texture_type, T::Cube | T::CubeArray);
    let one_row = matches!(texture_type, T::Type1D | T::Type1DArray | T::TextureBuffer);
    let array = matches!(texture_type, T::Type1DArray | T::Type2DArray | T::CubeArray);
    let height = if cube {
        width
    } else if one_row {
        1
    } else {
        height
    };
    ([width, height, 1], 1, if array { 2 } else { 1 })
}

/// The runtime answers Metal's size and alignment for every descriptor of both spaces, for all 32
/// formats: 1225 answers a format in the side sweep and 768 (480 for an ASTC footprint) in the
/// usage matrix, 62,048 in all.
#[test]
fn metals_footprint_answers() {
    let mut total = 0;
    for r in ROWS {
        let mut sides = Vec::new();
        for &width in &SIDES {
            for &height in &SIDES {
                let d = descriptor(r.format, T::Type2D, [width, height, 1], 1, 1, 0x1, true);
                let a = answer(&d);
                sides.push(a.size);
                sides.push(a.align);
            }
        }
        assert_eq!(sides.len() / 2, r.sides.0, "{:?}", r.format);
        assert_eq!(
            digest(&sides),
            r.sides.1,
            "{:?}: the side sweep's answers are not Metal's",
            r.format
        );
        let types: &[T] = match metal2vulkan::metal::format::block_extent(r.format) {
            Some(_) => &BLOCKY,
            None => &TYPES,
        };
        let mut usages = Vec::new();
        for &texture_type in types {
            for &pair in &SHAPES {
                let (extent, levels, slices) = shape(texture_type, pair);
                for &usage in &USAGES {
                    for optimized in [false, true] {
                        let d = descriptor(
                            r.format,
                            texture_type,
                            extent,
                            levels,
                            slices,
                            usage,
                            optimized,
                        );
                        let a = answer(&d);
                        usages.push(a.size);
                        usages.push(a.align);
                    }
                }
            }
        }
        assert_eq!(usages.len() / 2, r.usages.0, "{:?}", r.format);
        assert_eq!(
            digest(&usages),
            r.usages.1,
            "{:?}: the usage matrix's answers are not Metal's",
            r.format
        );
        total += r.sides.0 + r.usages.0;
    }
    assert_eq!(ROWS.len(), 32);
    assert_eq!(total, 62_048);
}

/// The row of each family that names its rule, so a reader meets the numbers and not only a
/// digest. Every one is the probe's answer on the M3 Max.
#[test]
fn the_rows_that_name_each_rule() {
    let read = 0x1;
    let of = |format, texture_type, extent: [u64; 3], levels, slices, usage| {
        let d = descriptor(format, texture_type, extent, levels, slices, usage, true);
        let a = answer(&d);
        (a.size, a.align)
    };
    // The tiled families: a 2-byte texel's tile is 128 x 64 elements and a 4-byte one's 64 x 64,
    // a level narrower than its tile is padded to a power of two on each side, and Metal adds
    // 8 bytes of compression metadata for each 16 x 16 tile, rounded up to 128, where the shorter
    // side is 16 texels or more and the usage is not `ShaderWrite` or `PixelFormatView`.
    assert_eq!(
        of(P::R12UnormX4, T::Type2D, [64, 64, 1], 1, 1, read),
        (8320, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::Type2D, [16, 16, 1], 1, 1, read),
        (640, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::Type2D, [16, 15, 1], 1, 1, read),
        (512, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::Type2D, [16, 17, 1], 1, 1, read),
        (1152, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::Type2D, [64, 64, 1], 1, 1, 0x2),
        (8192, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::Cube, [64, 64, 1], 1, 1, read),
        (49920, 128)
    );
    assert_eq!(
        of(P::R12UnormX4, T::TextureBuffer, [64, 1, 1], 1, 1, read),
        (128, 16)
    );
    assert_eq!(
        of(P::ARGB8UnormSRGB, T::Type2D, [64, 64, 1], 1, 1, read),
        (16512, 128)
    );
    assert_eq!(
        of(P::RG12UnormX8, T::Type2D, [64, 64, 1], 1, 1, read),
        (16512, 128)
    );
    // The ASTC footprints: a 16-byte block of 4 x 2 or 8 x 4 texels, never compressed. Their
    // height sweep is the one place the two tell each other apart — at 64 x 64 and in the whole
    // width sweep the 128-byte floor hides the block's width.
    assert_eq!(
        of(P::ASTC4x2SRGB, T::Type2D, [64, 64, 1], 1, 1, read),
        (8192, 128)
    );
    assert_eq!(
        of(P::ASTC8x4SRGB, T::Type2D, [64, 64, 1], 1, 1, read),
        (2048, 128)
    );
    assert_eq!(
        of(P::ASTC4x2SRGB, T::Type2D, [16, 4, 1], 1, 1, read),
        (128, 128)
    );
    assert_eq!(
        of(P::ASTC4x2SRGB, T::Type2D, [16, 5, 1], 1, 1, read),
        (256, 128)
    );
    assert_eq!(
        of(P::ASTC8x4SRGB, T::Type2D, [16, 16, 1], 1, 1, read),
        (128, 128)
    );
    assert_eq!(
        of(P::ASTC8x4SRGB, T::Type2D, [16, 17, 1], 1, 1, read),
        (256, 128)
    );
    // One linear plane, packed: no row padding at all, an alignment of 16, a 1D texture one row,
    // a 2D array's slices each padded to 128 at an alignment of 128, and a 3D texture's first
    // depth slice alone.
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [64, 64, 1], 1, 1, read),
        (8192, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [10, 1, 1], 1, 1, read),
        (20, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [16, 17, 1], 1, 1, read),
        (544, 16)
    );
    assert_eq!(
        of(P::YCBCR104441P, T::Type2D, [1, 2, 1], 1, 1, read),
        (8, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2DArray, [64, 64, 1], 1, 2, read),
        (16384, 128)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type3D, [64, 64, 2], 1, 1, read),
        (8192, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Cube, [64, 64, 1], 1, 1, read),
        (0, 16)
    );
    // A writable or renderable one-plane 4:2:2 texture needs both sides even; Metal makes none of
    // an odd-sided one and the query answers 0 bytes. The 4:4:4 formats take any sides.
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [15, 15, 1], 1, 1, 0x2),
        (0, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [16, 15, 1], 1, 1, 0x4),
        (0, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [16, 16, 1], 1, 1, 0x2),
        (512, 16)
    );
    assert_eq!(
        of(P::YCBCR84221P, T::Type2D, [15, 15, 1], 1, 1, read),
        (450, 16)
    );
    assert_eq!(
        of(P::YCBCR104441P, T::Type2D, [15, 15, 1], 1, 1, 0x2),
        (900, 16)
    );
    // Two planes: plane 0's 4-byte words, rows at a multiple of 16 bytes, rounded up to 16 KiB,
    // then plane 1's alpha bytes, the last row unpadded.
    assert_eq!(
        of(P::RGB10A82P, T::Type2D, [1, 1, 1], 1, 1, read),
        (16385, 16)
    );
    assert_eq!(
        of(P::RGB10A82P, T::Type2D, [16, 16, 1], 1, 1, read),
        (16640, 16)
    );
    assert_eq!(
        of(P::RGB10A82P, T::Type2D, [64, 64, 1], 1, 1, read),
        (20480, 16)
    );
    assert_eq!(of(P::RGB10A82P, T::Cube, [64, 64, 1], 1, 1, read), (0, 16));
    // The two packed formats: 0 bytes in every texture type, 128 for a 2D array.
    assert_eq!(
        of(P::R10UnormPACKED, T::Type2D, [64, 64, 1], 1, 1, read),
        (0, 16)
    );
    assert_eq!(
        of(P::R10UnormPACKED, T::Type2DArray, [64, 64, 1], 1, 2, read),
        (0, 128)
    );
    assert_eq!(
        of(P::RG10UnormPACKED, T::Type3D, [64, 64, 2], 1, 1, read),
        (0, 16)
    );
}
