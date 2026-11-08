//! `texture/heap_tiled`: Metal's layout of a heap texture's elements against the M3 probe's L rows,
//! measured on an Apple M4 under macOS 27.0.1 (an M3 Max under macOS 26.7 alike on every row of a
//! plain format): each row's answered size and alignment, its level bases and chain stride, and,
//! for a plain format, the hash of its span with every element's units holding their numbers, as
//! the probe wrote them; and the lossless compression's answers against the probe's C rows and a
//! second probe's answers.

use crate::copy_plan::block_bytes;
use crate::texture::heap::rows_lie_in_metal_tiles;
use crate::texture::heap_tiled::{
    fresh_pattern, level_bytes, lossless_answer, metal_compresses, metal_heap_layout,
    own_heap_layout, pattern_levels, sample_grid, shadow_tiles, BlockPlacement, Elements,
    FreshDepth, FreshPattern, HeapLayout, TiledLayout, TiledPart, Untiled,
};
use crate::texture_descriptor::TextureLayout;
use metal2vulkan::metal::format::byte_format;
use metal2vulkan::metal::{PixelFormat, TextureType as T, TextureUsage};
use metal2vulkan::vk::{Format, ImageAspectFlags};

/// One L row: its id, element bytes, block, type, level-0 extent, levels (0 the full chain) and
/// array length, Metal's size, alignment and chain stride (an array's, cube's or 3D texture's),
/// its level bases, and its span's hash.
struct Row {
    id: &'static str,
    unit: u64,
    block: [u64; 2],
    texture_type: T,
    extent: [u64; 3],
    levels: u32,
    array: u64,
    answer: (u64, u64, Option<u64>),
    bases: &'static [u64],
    span: u64,
}

fn row(
    id: &'static str,
    unit: u64,
    block: [u64; 2],
    texture_type: T,
    extent: [u64; 3],
    (levels, array): (u32, u64),
    answer: (u64, u64, Option<u64>),
    bases: &'static [u64],
    span: u64,
) -> Row {
    Row {
        id,
        unit,
        block,
        texture_type,
        extent,
        levels,
        array,
        answer,
        bases,
        span,
    }
}

/// The probe's L rows, as it states them (`m3probe.m`).
fn rows() -> Vec<Row> {
    vec![
        row(
            "L01",
            4,
            [1, 1],
            T::Type2D,
            [4, 2, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0x8af0436c15c63d6d,
        ),
        row(
            "L02",
            4,
            [1, 1],
            T::Type2D,
            [4, 4, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0x3a00051ff68aa595,
        ),
        row(
            "L03",
            4,
            [1, 1],
            T::Type2D,
            [4, 7, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0xdd6d80dac5721ad9,
        ),
        row(
            "L04",
            4,
            [1, 1],
            T::Type2D,
            [4, 100, 1],
            (1, 1),
            (2048, 128, None),
            &[0],
            0x855c3fdd0a0512da,
        ),
        row(
            "L05",
            4,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (109568, 128, None),
            &[0, 98304, 106496, 108544, 109056, 109184, 109312, 109440],
            0xa2fba87ea7ff6764,
        ),
        row(
            "L06",
            16,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (371584, 128, None),
            &[0, 245760, 360448, 368640, 370688, 371200, 371328, 371456],
            0x2492e1d50f408e56,
        ),
        row(
            "L07",
            1,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (44032, 128, None),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776, 43904],
            0xf5ebb0a44d4ef754,
        ),
        row(
            "L08",
            1,
            [1, 1],
            T::Type2D,
            [37, 328, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x9b11bec5ceb8c87c,
        ),
        row(
            "L09",
            1,
            [1, 1],
            T::Type2D,
            [458, 290, 1],
            (0, 1),
            (300672, 128, None),
            &[
                0, 196608, 278528, 294912, 299008, 300032, 300288, 300416, 300544,
            ],
            0x1eb0ca055317c905,
        ),
        row(
            "L10",
            2,
            [1, 1],
            T::Type2D,
            [100, 37, 1],
            (1, 1),
            (16384, 128, None),
            &[0],
            0xbce013e6e4c96e53,
        ),
        row(
            "L11",
            2,
            [1, 1],
            T::Type2D,
            [383, 256, 1],
            (4, 1),
            (300672, 128, None),
            &[0, 196608, 278528, 294912],
            0x979572030abe7044,
        ),
        row(
            "L12",
            4,
            [1, 1],
            T::Type2D,
            [100, 37, 1],
            (0, 1),
            (43904, 128, None),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0xa3513495246f96d8,
        ),
        row(
            "L13",
            8,
            [1, 1],
            T::Type2D,
            [300, 20, 1],
            (1, 1),
            (131072, 128, None),
            &[0],
            0x2c070da2053c11e3,
        ),
        row(
            "L14",
            8,
            [1, 1],
            T::Type2D,
            [129, 256, 1],
            (4, 1),
            (579200, 128, None),
            &[0, 393216, 557056, 573440],
            0x629d02556a27e882,
        ),
        row(
            "L15",
            16,
            [1, 1],
            T::Type2D,
            [33, 17, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0xe01c04b27f249279,
        ),
        row(
            "L16",
            16,
            [1, 1],
            T::Type2D,
            [576, 49, 1],
            (3, 1),
            (939648, 128, None),
            &[0, 589824, 851968],
            0x2d3542c55254fe77,
        ),
        row(
            "L17",
            4,
            [1, 1],
            T::Type2DArray,
            [100, 37, 1],
            (0, 3),
            (147456, 16384, Some(49152)),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0x05fcddb2e40428c5,
        ),
        row(
            "L18",
            1,
            [1, 1],
            T::Type2DArray,
            [561, 108, 1],
            (1, 2),
            (262144, 16384, Some(131072)),
            &[0],
            0xb8f4ce1a5624e22f,
        ),
        row(
            "L19",
            4,
            [1, 1],
            T::Cube,
            [40, 40, 1],
            (0, 1),
            (196608, 16384, Some(32768)),
            &[0, 16384, 20480, 21504, 21760, 21888],
            0x792c2854115f5b8a,
        ),
        row(
            "L20",
            4,
            [1, 1],
            T::Type3D,
            [100, 37, 9],
            (0, 1),
            (442368, 16384, Some(49152)),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0xbe53792d48aaa86c,
        ),
        row(
            "L21",
            1,
            [1, 1],
            T::Type3D,
            [103, 257, 9],
            (3, 1),
            (884736, 16384, Some(98304)),
            &[0, 65536, 81920],
            0x7d553da95cbf9f69,
        ),
        row(
            "L22",
            4,
            [1, 1],
            T::Type1D,
            [5000, 1, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x1a7c65cd83ebe0ec,
        ),
        row(
            "L23",
            4,
            [1, 1],
            T::Type2D,
            [5000, 3, 1],
            (0, 1),
            (180608, 128, None),
            &[
                0, 131072, 163840, 172032, 176128, 178176, 179200, 179712, 179968, 180096, 180224,
                180352, 180480,
            ],
            0x80c6df1ba4aa8059,
        ),
        row(
            "L24",
            1,
            [1, 1],
            T::Type2D,
            [59, 15, 1],
            (4, 1),
            (1792, 128, None),
            &[0, 1024, 1280, 1408],
            0x4b063e858ae06abc,
        ),
        row(
            "L25",
            8,
            [4, 4],
            T::Type2DArray,
            [100, 36, 1],
            (0, 3),
            (17664, 128, Some(5888)),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760],
            0x3b660bcb74cf4178,
        ),
        row(
            "L26",
            16,
            [8, 8],
            T::Type2D,
            [300, 20, 1],
            (0, 1),
            (6144, 128, None),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760, 5888, 6016],
            0x4371b8d24ef9e4f3,
        ),
        row(
            "L27",
            16,
            [4, 4],
            T::Cube,
            [64, 64, 1],
            (0, 1),
            (35328, 128, Some(5888)),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760],
            0x66e8e6b2f1f6e87d,
        ),
        row(
            "L28",
            8,
            [1, 1],
            T::Type2D,
            [64, 64, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x091e7a48780c7645,
        ),
    ]
}

/// The levels of a full chain whose largest side is `side`.
fn full(side: u64) -> u32 {
    u64::BITS - side.leading_zeros()
}

impl Row {
    /// Its slices (six a cube's).
    fn slices(&self) -> u64 {
        match self.texture_type {
            T::Cube => 6 * self.array,
            _ => self.array,
        }
    }

    /// Its levels, the full chain's for 0.
    fn levels(&self) -> u32 {
        let [w, h, d] = self.extent;
        let depth = if self.texture_type == T::Type3D { d } else { 1 };
        match self.levels {
            0 => full(w.max(h).max(depth)),
            levels => levels,
        }
    }

    /// Its layout, of its texels or blocks (a block-compressed row's answer and level bases are
    /// alike under both placements).
    fn layout(&self) -> TiledLayout {
        let (unit, block, ty) = (self.unit, self.block, self.texture_type);
        let elements = match block {
            [1, 1] => Elements::Texels,
            block => Elements::Blocks {
                block,
                placement: BlockPlacement::Halved,
            },
        };
        TiledLayout::of(
            unit,
            elements,
            ty,
            self.extent,
            self.levels(),
            self.slices(),
        )
        .unwrap_or_else(|| panic!("{}: no layout", self.id))
    }
}

/// FNV-1a over `bytes`, from `hash`.
pub(super) fn fnv(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// The probe's span hash of a texture laid out as `layout`, `size` bytes: each element of each
/// slice, level and depth slice in turn numbered from 0, its units of `min(B, 4)` bytes (unit `j`
/// of element `e` holding `e · per + j + 1`, `per` its units) written where the layout places
/// it, over passes of 8 or 16 bits for 1- and 2-byte units; the hash chained over the passes.
pub(super) fn span_hash(layout: &TiledLayout, size: u64) -> u64 {
    let unit = layout.unit;
    let u = unit.min(4);
    let per = if unit >= 4 { unit / 4 } else { 1 };
    let mut places = Vec::new();
    let mut element = 0u64;
    let slices = if layout.volume { 1 } else { layout.chains };
    for slice in 0..slices {
        for (k, level) in layout.levels.iter().enumerate() {
            for z in 0..level.depth {
                let chain = if layout.volume { z } else { slice };
                for y in 0..level.extent[1] {
                    for x in 0..level.extent[0] {
                        let at = layout.element(chain, k, x, y);
                        for j in 0..per {
                            places.push((at / u + j, element * per + j + 1));
                        }
                        element += 1;
                    }
                }
            }
        }
    }
    let bits = u64::from(u64::BITS - (element * per + 1).leading_zeros());
    let passes = if u == 4 { 1 } else { bits.div_ceil(8 * u) };
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for pass in 0..passes {
        let mut span = vec![0u8; size as usize];
        for &(place, value) in &places {
            let shifted = value >> (8 * u * pass);
            let bytes = shifted.to_le_bytes();
            let at = (place * u) as usize;
            span[at..at + u as usize].copy_from_slice(&bytes[..u as usize]);
        }
        hash = fnv(&span, hash);
    }
    hash
}

/// Every L row's answer, level bases and chain stride are the layout's.
#[test]
fn the_layout_answers_every_l_row_as_metal() {
    for row in rows() {
        let layout = row.layout();
        let (size, align, stride) = row.answer;
        assert_eq!((layout.size, layout.align), (size, align), "{}", row.id);
        let bases: Vec<u64> = layout.levels.iter().map(|level| level.base).collect();
        assert_eq!(bases, row.bases, "{}", row.id);
        if let Some(stride) = stride {
            assert_eq!(layout.stride, stride, "{}", row.id);
        }
    }
}

/// Every L row of a plain format places each element as Metal does: its span's hash is the
/// probe's.
#[test]
fn the_layout_places_every_plain_l_row_s_elements_as_metal() {
    for row in rows().iter().filter(|row| row.block == [1, 1]) {
        let layout = row.layout();
        let (size, _, _) = row.answer;
        assert_eq!(span_hash(&layout, size), row.span, "{}", row.id);
    }
}

/// The M4's answers for the probe's block-compressed rows L29 to L36: their
/// blocks' allocations are the layout's, a block an element.
#[test]
fn block_compressed_allocations_are_the_layout_s() {
    /// A row: its format, block, element bytes, extent, levels (0 the full chain), and Metal's
    /// size and level bases.
    fn block(
        pf: PixelFormat,
        unit: u64,
        [w, h]: [u64; 2],
        levels: u32,
        (size, bases): (u64, &'static [u64]),
    ) -> (PixelFormat, u64, [u64; 2], u32, u64, &'static [u64]) {
        (pf, unit, [w, h], levels, size, bases)
    }
    let long = &[0, 4096, 5120, 5376, 5504, 5632, 5760];
    let wide = &[0, 8192, 10240, 10752, 10880, 11008, 11136];
    let rows = [
        block(PixelFormat::BC1RGBA, 8, [100, 36], 0, (5888, long)),
        block(PixelFormat::BC1RGBA, 8, [64, 4], 1, (128, &[0])),
        block(PixelFormat::BC1RGBA, 8, [12, 4], 1, (128, &[0])),
        block(PixelFormat::BC1RGBA, 8, [36, 100], 0, (5888, long)),
        block(PixelFormat::BC1RGBA, 8, [4, 12], 1, (128, &[0])),
        block(PixelFormat::ASTC4x4LDR, 16, [100, 36], 0, (11264, wide)),
        block(PixelFormat::BC7RGBAUnorm, 16, [100, 36], 0, (11264, wide)),
        block(PixelFormat::BC1RGBA, 8, [100, 33], 0, (5888, long)),
    ];
    for (pf, unit, [w, h], levels, size, bases) in rows {
        let levels = if levels == 0 { full(w.max(h)) } else { levels };
        let elements = Elements::Blocks {
            block: [4, 4],
            placement: BlockPlacement::Narrow,
        };
        let layout = TiledLayout::of(unit, elements, T::Type2D, [w, h, 1], levels, 1).unwrap();
        let got: Vec<u64> = layout.levels.iter().map(|level| level.base).collect();
        assert_eq!((layout.size, layout.align), (size, 128), "{pf:?} {w}x{h}");
        assert_eq!(got, bases, "{pf:?} {w}x{h}");
    }
}

/// A level's bytes as the heap lane's Metal answers state them (the M3 Max and the
/// M4 alike; `heap_textures.rs`'s shapes, uncompressed).
#[test]
fn level_bytes_are_metal_s() {
    let rgba = [64, 64];
    assert_eq!(level_bytes(4, rgba, [64, 64], 0, Elements::Texels), 16384);
    assert_eq!(level_bytes(4, rgba, [16, 16], 0, Elements::Texels), 1024);
    assert_eq!(level_bytes(4, rgba, [17, 17], 0, Elements::Texels), 4096);
    assert_eq!(level_bytes(4, rgba, [100, 100], 0, Elements::Texels), 65536);
    assert_eq!(
        level_bytes(4, rgba, [512, 128], 0, Elements::Texels),
        262144
    );
    let chain: u64 = (0..9)
        .map(|level| level_bytes(4, rgba, [256, 256], level, Elements::Texels))
        .sum();
    assert_eq!(chain, 349824);
    assert_eq!(
        level_bytes(1, [128, 128], [64, 64], 0, Elements::Texels),
        4096
    );
    assert_eq!(
        level_bytes(2, [128, 64], [64, 64], 0, Elements::Texels),
        8192
    );
}

/// Which textures Metal compresses and what it answers for them (the probe's C rows): a colour
/// texture of neither `ShaderWrite` nor `PixelFormatView` usage with `allowGPUOptimizedContents`
/// and a level 0 of 16 texels or more on its shorter side, a depth or stencil texture of any usage;
/// its chains, then 2 bytes of metadata for each 8 x 8 texels of its level padded to powers of two,
/// rounded up to 128, for each chain.
#[test]
fn lossless_answers_are_metal_s() {
    let read = TextureUsage::SHADER_READ;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let view = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let target = TextureUsage::RENDER_TARGET;
    assert!(metal_compresses(true, read, true, [16, 16]));
    assert!(metal_compresses(true, target, true, [64, 64]));
    assert!(!metal_compresses(true, read, true, [16, 8]));
    assert!(!metal_compresses(true, read, true, [6, 43]));
    assert!(!metal_compresses(true, read, true, [2048, 5]));
    assert!(!metal_compresses(true, read, false, [64, 64]));
    assert!(!metal_compresses(true, write, true, [64, 64]));
    assert!(!metal_compresses(true, view, true, [64, 64]));
    assert!(metal_compresses(false, write, true, [64, 64]));
    assert!(!metal_compresses(false, write, true, [8, 8]));
    // (element bytes, type, extent, slices, Metal's size and alignment)
    let rows = [
        (4, T::Type2D, [64, 64, 1], 1, (16512, 128)),
        (4, T::Type2D, [16, 16, 1], 1, (1152, 128)),
        (4, T::Type2D, [100, 37, 1], 1, (33024, 128)),
        (4, T::Type2DArray, [64, 64, 1], 2, (33024, 16384)),
        (1, T::Type2D, [64, 64, 1], 1, (4224, 128)),
        (8, T::Type2D, [64, 64, 1], 1, (32896, 128)),
        (16, T::Type2D, [64, 64, 1], 1, (65664, 128)),
        (2, T::Type2D, [64, 64, 1], 1, (8320, 128)),
        (4, T::Type2D, [17, 17, 1], 1, (4224, 128)),
        (4, T::Type2D, [100, 100, 1], 1, (66048, 128)),
        (4, T::Type2DArray, [33, 33, 1], 3, (49536, 16384)),
        (4, T::Cube, [33, 33, 1], 6, (99072, 16384)),
        (4, T::Type3D, [33, 33, 33], 1, (544896, 16384)),
        (4, T::Type2D, [512, 128, 1], 1, (264192, 128)),
        (4, T::Type2D, [300, 260, 1], 1, (417792, 128)),
    ];
    for (unit, ty, extent, slices, (size, align)) in rows {
        let layout = TiledLayout::of(unit, Elements::Texels, ty, extent, 1, slices).unwrap();
        let answer = lossless_answer(&layout).unwrap();
        assert_eq!(
            (answer.size, answer.align),
            (size, align),
            "{unit} {ty:?} {extent:?}"
        );
    }
}

/// A mipmapped texture Metal compresses answers its full chain uncompressed, then for each chain
/// metadata over the full chain's levels while level 0's sides, padded to multiples of 16 and
/// halved, have a longer side of 16 or more, whatever levels it has (an Apple M4 under macOS
/// 27.0.1, and an M3 Max under 26.7, alike: the consumer's boot's `RGBA16Float` 224 x 928 and 216 x
/// 928 textures of 3 levels among them, and the 7-level 64 x 64 `R32Uint` render target MoltenVK
/// answered 0x5680 for); a 3D texture's chains, one a depth slice of level 0, and a cube array's
/// alike.
#[test]
fn mipmapped_lossless_answers_are_metal_s() {
    // (element bytes, type, extent, levels, slices, Metal's size and alignment)
    let rows = [
        (8, T::Type2D, [224, 928, 1], 3, 1, (0x279700, 128)),
        (8, T::Type2D, [216, 928, 1], 3, 1, (0x279700, 128)),
        (4, T::Type2D, [64, 64, 1], 7, 1, (0x5800, 128)),
        (4, T::Type2D, [1024, 16, 1], 10, 1, (0x15d80, 128)),
        (4, T::Type2D, [16, 1025, 1], 2, 1, (0x2b680, 128)),
        (4, T::Type2D, [513, 17, 1], 10, 1, (0x2b380, 128)),
        (4, T::Type2D, [1440, 1025, 1], 10, 1, (0x8e1680, 128)),
        (4, T::Type2D, [33, 63, 1], 2, 1, (0x5780, 128)),
        (4, T::Type2D, [500, 513, 1], 2, 1, (0x19ac80, 128)),
        (2, T::Type2D, [100, 37, 1], 4, 1, (0x5880, 128)),
        (2, T::Type2D, [129, 500, 1], 6, 1, (0x4c280, 128)),
        (4, T::Type2D, [256, 256, 1], 9, 1, (0x56200, 128)),
        (1, T::Type2D, [1025, 63, 1], 3, 1, (0x2c400, 128)),
        (8, T::Type2D, [1440, 1080, 1], 4, 1, (0x111d600, 128)),
        (4, T::Type2DArray, [100, 100, 1], 7, 2, (0x30600, 16384)),
        (4, T::Type2DArray, [16, 16, 1], 2, 2, (0xe00, 128)),
        (8, T::Type2DArray, [928, 928, 1], 10, 2, (0x135d600, 16384)),
        (4, T::Cube, [216, 216, 1], 8, 6, (0x214200, 16384)),
        (8, T::Cube, [928, 928, 1], 10, 6, (0x3a18200, 16384)),
        (2, T::Type2DArray, [513, 513, 1], 10, 6, (0x6d0200, 16384)),
        (4, T::Type3D, [64, 100, 33], 7, 1, (0x190200, 16384)),
        (4, T::Type3D, [64, 100, 33], 1, 1, (0x10a100, 16384)),
        (8, T::Type3D, [224, 17, 7], 3, 1, (0xa9180, 16384)),
        (4, T::Type3D, [513, 928, 2], 10, 1, (0x655600, 16384)),
        (4, T::Type3D, [16, 16, 16], 5, 1, (0x7000, 128)),
        (4, T::CubeArray, [100, 100, 1], 7, 12, (0x122400, 16384)),
        (8, T::CubeArray, [216, 216, 1], 8, 6, (0x3c4200, 16384)),
        (1, T::CubeArray, [17, 17, 1], 5, 12, (0x5a00, 128)),
    ];
    for (unit, ty, extent, levels, slices, (size, align)) in rows {
        let layout = TiledLayout::of(unit, Elements::Texels, ty, extent, levels, slices).unwrap();
        let answer = lossless_answer(&layout).unwrap();
        assert_eq!(
            (answer.size, answer.align),
            (size, align),
            "{unit} {ty:?} {extent:?} {levels}"
        );
    }
}

/// A one-sample `Depth32Float_Stencil8` texture answers its two planes, `Depth32Float`'s answer
/// rounded up to 16 KiB, then `Stencil8`'s, at the depth plane's alignment, compressed or not,
/// in every type it takes (an Apple M4 under macOS 27.0.1: 12330 answers; 14552 2D answers each,
/// alike).
#[test]
fn depth_stencil_answers_are_their_planes() {
    let rt = TextureUsage::RENDER_TARGET;
    let read = TextureUsage::SHADER_READ;
    // (type, extent, levels, slices, usage, allowGPUOptimizedContents, Metal's answer)
    let rows = [
        (T::Type2D, [4, 4], 1, 1, read, true, (0x4080, 128)),
        (T::Type2D, [64, 64], 7, 1, read | rt, false, (0x9700, 128)),
        (T::Type2D, [64, 64], 7, 1, read | rt, true, (0x9880, 128)),
        (T::Type2D, [100, 33], 3, 1, rt, true, (0xee00, 128)),
        (T::Type2D, [513, 224], 10, 1, read, true, (0x101800, 128)),
        (T::Type2D, [15, 17], 5, 1, read | rt, true, (0x4400, 128)),
        (
            T::Type2DArray,
            [100, 100],
            7,
            3,
            read | rt,
            true,
            (0x64900, 16384),
        ),
        (T::Cube, [64, 64], 7, 6, read | rt, false, (0x38a00, 16384)),
        (T::CubeArray, [216, 216], 8, 12, rt, true, (0x554400, 16384)),
    ];
    for (ty, [w, h], levels, slices, usage, optimized, (size, align)) in rows {
        let what = format!("{ty:?} {w}x{h} {levels} {slices} {usage:?} {optimized}");
        let ds = texture(
            ty,
            PixelFormat::Depth32FloatStencil8,
            [w, h, 1],
            (levels, slices, 1),
        );
        let format = Format::D32_SFLOAT_S8_UINT;
        let metal = metal_heap_layout(&ds, usage, optimized, format);
        let answer = metal.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
    }
}

/// A texture layout of `texture_type`, `pixel_format`, `extent`, `levels`, `layers` and
/// `samples`.
fn texture(
    texture_type: T,
    pixel_format: PixelFormat,
    extent: [u32; 3],
    (levels, layers, samples): (u32, u32, u32),
) -> TextureLayout {
    TextureLayout {
        texture_type,
        pixel_format,
        extent,
        levels,
        layers,
        samples,
    }
}

/// How the runtime lays a heap texture out over host pages, by its format, usage and shape: a
/// plain or block-compressed format whose image holds Metal's bytes tiled (PVRTC among them), and
/// an uncompressed multisample texture's samples; a compressed texture, a block-compressed one of
/// a family whose placement is not measured (ASTC 4 x 2), or one whose image holds other bytes,
/// untiled, each with its kind, and the answers Metal's where measured.
#[test]
fn heap_textures_are_tiled_where_metal_s_layout_is_measured() {
    use crate::degradation::Degradation as D;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let read = TextureUsage::SHADER_READ;
    let rgba = texture(T::Type2D, PixelFormat::RGBA8Unorm, [64, 64, 1], (1, 1, 1));
    let tiled = metal_heap_layout(&rgba, write, true, Format::R8G8B8A8_UNORM);
    let answer = tiled.answer.unwrap();
    assert_eq!((answer.size, answer.align), (16384, 128));
    assert!(matches!(tiled.layout, HeapLayout::Tiled(_)));
    // Compressed: its answer, and the lossless kind.
    let lossless = metal_heap_layout(&rgba, read, true, Format::R8G8B8A8_UNORM);
    assert_eq!(lossless.answer.map(|a| a.size), Some(16512));
    let HeapLayout::Untiled(untiled) = lossless.layout else {
        panic!("a compressed texture is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureLosslessLayout);
    // An image of other bytes.
    let other = metal_heap_layout(&rgba, write, true, Format::B8G8R8A8_UNORM);
    let HeapLayout::Untiled(untiled) = other.layout else {
        panic!("an image of other bytes is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    assert_eq!(other.answer, tiled.answer);
    // Block-compressed: Metal's answer, and its kind.
    let bc1 = texture(
        T::Type2DArray,
        PixelFormat::BC1RGBA,
        [100, 36, 1],
        (7, 3, 1),
    );
    let blocks = metal_heap_layout(&bc1, read, true, Format::BC1_RGBA_UNORM_BLOCK);
    assert_eq!(blocks.answer.map(|a| (a.size, a.align)), Some((17664, 128)));
    assert!(matches!(blocks.layout, HeapLayout::Tiled(_)));
    let pvrtc = texture(
        T::Type2D,
        PixelFormat::PVRTCRGBA4BPP,
        [64, 64, 1],
        (7, 1, 1),
    );
    // PVRTC: Metal's answer and its tiles.
    let format = Format::PVRTC1_4BPP_UNORM_BLOCK_IMG;
    let m4_pvrtc = metal_heap_layout(&pvrtc, read, true, format);
    assert_eq!(
        m4_pvrtc.answer.map(|a| (a.size, a.align)),
        Some((3200, 128))
    );
    assert!(matches!(m4_pvrtc.layout, HeapLayout::Tiled(_)));
    // ASTC 4 x 2: Metal's answer, its family's placement (the footprint rows of
    // `heap_footprint_formats`), untiled where no image holds its blocks (a carrier's do).
    let astc = texture(T::Type2D, PixelFormat::ASTC4x2LDR, [64, 64, 1], (1, 1, 1));
    let carried = metal_heap_layout(&astc, read, true, Format::UNDEFINED);
    assert_eq!(carried.answer.map(|a| (a.size, a.align)), Some((8192, 128)));
    let HeapLayout::Untiled(untiled) = carried.layout else {
        panic!("an ASTC 4 x 2 texture is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    let decoded = metal_heap_layout(&bc1, read, true, Format::R8G8B8A8_UNORM);
    assert_eq!(decoded.answer, blocks.answer);
    let HeapLayout::Untiled(untiled) = decoded.layout else {
        panic!("a block-compressed texture held decoded is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    // Multisample: Metal's answer, its samples tiled (`multisample_answers_are_metal_s`).
    let ms = texture(
        T::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [64, 64, 1],
        (1, 1, 4),
    );
    let multisample = metal_heap_layout(
        &ms,
        TextureUsage::RENDER_TARGET,
        false,
        Format::R8G8B8A8_UNORM,
    );
    assert_eq!(
        multisample.answer.map(|a| (a.size, a.align)),
        Some((65536, 128))
    );
    let HeapLayout::Tiled(tiles) = multisample.layout else {
        panic!("a multisample texture is untiled");
    };
    assert_eq!((tiles.parts[0].layout.unit, tiles.size), (16, 65536));
    // Two-aspect: its depth plane's tiles at 0, then its stencil plane's at 16 KiB; kept
    // compressed, its depth plane is, and a shadow's stencil plane lies past the depth plane's
    // compressed answer (`shadow_tiles`).
    let ds = texture(
        T::Type2D,
        PixelFormat::Depth32FloatStencil8,
        [64, 64, 1],
        (1, 1, 1),
    );
    let format = Format::D32_SFLOAT_S8_UINT;
    let two = metal_heap_layout(&ds, write, false, format);
    assert_eq!(two.answer.map(|a| (a.size, a.align)), Some((0x5000, 128)));
    let HeapLayout::Tiled(tiles) = &two.layout else {
        panic!("a two-aspect texture is untiled");
    };
    let parts: Vec<_> = (tiles.parts.iter())
        .map(|part| (part.aspect, part.at, part.layout.unit, part.layout.size))
        .collect();
    let (depth, stencil) = (
        Some(ImageAspectFlags::DEPTH),
        Some(ImageAspectFlags::STENCIL),
    );
    assert_eq!(parts, [(depth, 0, 4, 0x4000), (stencil, 0x4000, 1, 0x1000)]);
    let kept = metal_heap_layout(&ds, write, true, format);
    assert_eq!(kept.answer.map(|a| (a.size, a.align)), Some((0x9080, 128)));
    let HeapLayout::Untiled(untiled) = kept.layout else {
        panic!("a compressed two-aspect texture is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureLosslessLayout);
    let shadow = shadow_tiles(&ds, format, &kept.layout);
    let Some(HeapLayout::Uncompressed(tiles)) = shadow else {
        panic!("a compressed two-aspect texture has no shadow's tiles");
    };
    let ats: Vec<_> = tiles.parts.iter().map(|part| part.at).collect();
    assert_eq!((ats, tiles.size), (vec![0, 0x8000], 0x9000));
    let other = metal_heap_layout(&ds, write, false, Format::D32_SFLOAT);
    assert!(matches!(other.layout, HeapLayout::Untiled(_)));
    // A depth texture without GPU-optimized contents: tiled.
    let depth = texture(T::Type2D, PixelFormat::Depth32Float, [64, 64, 1], (1, 1, 1));
    let tiled_depth = metal_heap_layout(&depth, read, false, Format::D32_SFLOAT);
    assert!(matches!(tiled_depth.layout, HeapLayout::Tiled(_)));
    // A stencil view: a `Stencil8` texture's tiles and answers (the M3 Max under 27.0.1), the
    // tiles its image's stencil aspect's; kept compressed, a shadow's uncompressed tiles alike.
    let view = texture(T::Type2D, PixelFormat::X32Stencil8, [100, 70, 1], (1, 1, 1));
    let x32 = metal_heap_layout(&view, read, false, format);
    assert_eq!(x32.answer.map(|a| (a.size, a.align)), Some((0x4000, 128)));
    let HeapLayout::Tiled(tiles) = &x32.layout else {
        panic!("a stencil view is untiled");
    };
    let parts: Vec<_> = (tiles.parts.iter())
        .map(|part| (part.aspect, part.at, part.layout.unit, part.layout.size))
        .collect();
    assert_eq!(parts, [(stencil, 0, 1, 0x4000)]);
    let kept = metal_heap_layout(&view, read, true, format);
    assert_eq!(kept.answer.map(|a| (a.size, a.align)), Some((0x4200, 128)));
    let shadow = shadow_tiles(&view, format, &kept.layout);
    let Some(HeapLayout::Uncompressed(tiles)) = shadow else {
        panic!("a compressed stencil view has no shadow's tiles");
    };
    assert_eq!(tiles.parts.len(), 1);
    assert_eq!((tiles.parts[0].aspect, tiles.size), (stencil, 0x4000));
}

/// A multisample texture's answers and layout kind (on an Apple M4 under macOS 27.0.1; the M3 Max
/// under 26.7 answered our probe's descriptors alike): compressed by the single-sample rule over
/// its sample grid (`w x 2h` at 2 samples, `2w x 2h` at 4); uncompressed, its pixels elements of
/// all their samples in Metal's tiles, a multisample array's slices an array's, which a shadow
/// holds where its image holds Metal's elements; compressed, that plus metadata over the sample
/// grid for each slice, kept compressed (`Untiled::LOSSLESS_MULTISAMPLE`), a shadow holding its
/// uncompressed tiles. A `Depth32Float_Stencil8` texture's two planes alike, its stencil plane
/// past its depth plane's answer rounded up to 16 KiB.
#[test]
fn multisample_answers_are_metal_s() {
    use crate::degradation::Degradation as D;
    let rt = TextureUsage::RENDER_TARGET;
    let rw = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE;
    let view = TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW;
    // The sample grid, and the compression threshold over it.
    assert_eq!(sample_grid(2, [16, 8]), Some([16, 16]));
    assert_eq!(sample_grid(4, [8, 8]), Some([16, 16]));
    assert_eq!(sample_grid(8, [64, 64]), None);
    assert!(metal_compresses(true, rt, true, [16, 16]));
    assert!(!metal_compresses(true, rt, true, [8, 32]));
    assert!(!metal_compresses(true, view, true, [128, 128]));
    use PixelFormat as P;
    let (ms, msa) = (T::Type2DMultisample, T::Type2DMultisampleArray);
    // (format, samples, usage, allowGPUOptimizedContents, type, slices, extent, Metal's size and
    // alignment, compressed)
    let rows = [
        (
            P::R8Unorm,
            2,
            rt,
            false,
            ms,
            1,
            [300, 2],
            (2048, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            ms,
            1,
            [64, 64],
            (66048, 128),
            true,
        ),
        (
            P::RGBA8Unorm,
            4,
            rw,
            true,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            false,
            msa,
            2,
            [64, 64],
            (131072, 16384),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            msa,
            2,
            [64, 64],
            (132096, 16384),
            true,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            msa,
            3,
            [16, 16],
            (12672, 128),
            true,
        ),
        (P::RGBA8Unorm, 4, rt, true, ms, 1, [8, 8], (1152, 128), true),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            ms,
            1,
            [7, 7],
            (1024, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            2,
            rt,
            true,
            ms,
            1,
            [16, 8],
            (1152, 128),
            true,
        ),
        (
            P::RGBA8Unorm,
            2,
            rt,
            true,
            ms,
            1,
            [8, 16],
            (1024, 128),
            false,
        ),
        (
            P::RGBA16Float,
            4,
            rt,
            false,
            ms,
            1,
            [100, 37],
            (196608, 128),
            false,
        ),
        (
            P::RGBA16Float,
            4,
            rt,
            false,
            msa,
            2,
            [9, 9],
            (16384, 128),
            false,
        ),
        (
            P::RGBA32Float,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (262144, 128),
            false,
        ),
        (
            P::RGBA32Float,
            4,
            rt,
            false,
            msa,
            3,
            [17, 13],
            (98304, 16384),
            false,
        ),
        (
            P::Depth32Float,
            4,
            rw,
            true,
            ms,
            1,
            [64, 64],
            (66048, 128),
            true,
        ),
        (
            P::Depth32Float,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::Depth16Unorm,
            2,
            rt,
            true,
            msa,
            2,
            [100, 37],
            (66560, 16384),
            true,
        ),
        (P::Stencil8, 4, rt, true, ms, 1, [17, 13], (2176, 128), true),
    ];
    for (pf, samples, usage, optimized, ty, slices, [w, h], (size, align), compressed) in rows {
        let what = format!("{pf:?} x{samples} {usage:?} {optimized} {ty:?} {slices} {w}x{h}");
        let layout = texture(ty, pf, [w, h, 1], (1, slices, samples));
        // An image of other bytes than Metal's elements: its samples apart.
        let other = metal_heap_layout(&layout, usage, optimized, Format::UNDEFINED);
        let answer = other.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
        let HeapLayout::Untiled(untiled) = other.layout else {
            panic!("{what}: tiled over other bytes");
        };
        match compressed {
            true => assert_eq!(untiled, Untiled::LOSSLESS_MULTISAMPLE, "{what}"),
            false => assert_eq!(untiled.kind, D::HeapTextureUntiledLayout, "{what}"),
        }
        // Its image of Metal's elements: tiled, or its uncompressed tiles a shadow's.
        let format = image_format(pf);
        let metal = metal_heap_layout(&layout, usage, optimized, format);
        assert_eq!(metal.answer, other.answer, "{what}");
        let tiles = match (compressed, &metal.layout) {
            (false, HeapLayout::Tiled(tiles)) => tiles.clone(),
            (true, HeapLayout::Untiled(untiled)) if *untiled == Untiled::LOSSLESS_MULTISAMPLE => {
                match shadow_tiles(&layout, format, &metal.layout) {
                    Some(HeapLayout::Uncompressed(tiles)) => tiles,
                    shadow => panic!("{what}: a shadow of {shadow:?}"),
                }
            }
            (_, layout) => panic!("{what}: {layout:?}"),
        };
        let unit = u64::from(samples) * u64::from(block_bytes(pf).expect("bytes"));
        assert_eq!(tiles.parts.len(), 1, "{what}");
        assert_eq!(tiles.parts[0].layout.unit, unit, "{what}");
        if !compressed {
            assert_eq!(tiles.size, size, "{what}");
        }
    }
    // `Depth32Float_Stencil8`: its depth plane rounded up to 16 KiB, then its stencil plane, at
    // the depth plane's alignment (the consumer's 100 x 60 array of 2 samples, 98816 at 16 KiB on
    // the M3 Max), held by the rule as any depth format.
    let ds_rows = [
        (2, false, ms, 1, [8, 8], (16512, 128)),
        (2, false, msa, 1, [100, 37], (81920, 16384)),
        (2, true, ms, 1, [100, 37], (98816, 128)),
        (2, true, msa, 1, [100, 37], (98816, 16384)),
        (2, true, msa, 1, [100, 60], (98816, 16384)),
        (2, true, msa, 2, [100, 37], (181248, 16384)),
        (2, true, msa, 3, [300, 64], (661504, 16384)),
        (2, false, msa, 3, [8, 8], (16768, 128)),
        (2, true, msa, 2, [17, 64], (57600, 16384)),
        (4, false, msa, 2, [8, 8], (16896, 128)),
        (4, true, msa, 2, [8, 8], (17152, 128)),
        (4, false, msa, 3, [300, 64], (1228800, 16384)),
        (4, true, ms, 1, [300, 64], (430080, 128)),
    ];
    for (samples, optimized, ty, slices, [w, h], (size, align)) in ds_rows {
        let what = format!("Depth32Float_Stencil8 x{samples} {optimized} {ty:?} {slices} {w}x{h}");
        let ds = texture(ty, P::Depth32FloatStencil8, [w, h, 1], (1, slices, samples));
        let metal = metal_heap_layout(&ds, rt, optimized, Format::D32_SFLOAT_S8_UINT);
        let answer = metal.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
        let lossless = metal.layout == HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE);
        let grid = sample_grid(samples, [w, h].map(u64::from)).expect("grid");
        assert_eq!(lossless, optimized && grid[0].min(grid[1]) >= 16, "{what}");
        // Its two planes, the stencil plane past the depth plane's answer rounded up to 16 KiB:
        // Metal's tiles, or a compressed one's uncompressed tiles.
        let format = Format::D32_SFLOAT_S8_UINT;
        let tiles = match shadow_tiles(&ds, format, &metal.layout) {
            Some(HeapLayout::Tiled(tiles)) if !lossless => tiles,
            Some(HeapLayout::Uncompressed(tiles)) if lossless => tiles,
            shadow => panic!("{what}: a shadow of {shadow:?}"),
        };
        let units: Vec<_> = (tiles.parts.iter())
            .map(|part| (part.aspect, part.layout.unit))
            .collect();
        let (depth, stencil) = (ImageAspectFlags::DEPTH, ImageAspectFlags::STENCIL);
        let n = u64::from(samples);
        assert_eq!(units, [(Some(depth), 4 * n), (Some(stencil), n)], "{what}");
        let depth_plane = texture(ty, P::Depth32Float, [w, h, 1], (1, slices, samples));
        let depth_answer = metal_heap_layout(&depth_plane, rt, optimized, Format::D32_SFLOAT);
        let at = depth_answer
            .answer
            .expect("depth")
            .size
            .next_multiple_of(16384);
        assert_eq!(tiles.parts[1].at, at, "{what}");
        if !lossless {
            assert_eq!(tiles.size, size, "{what}");
        }
    }
    // Eight samples: not measured, no answer, refused over host pages.
    let eight = texture(ms, P::RGBA8Unorm, [64, 64, 1], (1, 1, 8));
    let metal = metal_heap_layout(&eight, rt, true, Format::R8G8B8A8_UNORM);
    assert_eq!(metal.answer, None);
    assert_ne!(
        metal.layout,
        HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE)
    );
}

/// The image format that holds `pf`'s elements as Metal's bytes: a depth or stencil format's own,
/// a colour format's byte format.
fn image_format(pf: PixelFormat) -> Format {
    match pf {
        PixelFormat::Depth32Float => Format::D32_SFLOAT,
        PixelFormat::Depth16Unorm => Format::D16_UNORM,
        PixelFormat::Stencil8 => Format::S8_UINT,
        pf => byte_format(pf).expect("a byte format"),
    }
}

/// Where a multisample texture's samples lie in Metal's tiles, uncompressed: sample `s` of pixel
/// `(x, y)` at `tiled(S·B, w, h, x, y) + s·B` from its plane's first byte, the rule an Apple M3 Max
/// under macOS 27.0.1 drew every sample of these textures at (`Depth32Float` and `Depth16Unorm`,
/// `Stencil8` and `Depth32Float_Stencil8`, 2 and 4 samples, `allowGPUOptimizedContents` off,
/// each sample's value naming it), with Metal's answer and the stencil plane's first byte.
#[test]
fn multisample_samples_lie_where_metal_draws_them() {
    use PixelFormat as P;
    // The probe's own rule: a level of `w x h` elements of `unit` bytes in tiles of the largest
    // tile's sides cut to the level's powers of two, the tiles row after row, a tile's elements
    // in Morton order, x's bit lowest.
    fn tiled(unit: u64, [w, h]: [u64; 2], [x, y]: [u64; 2]) -> u64 {
        let [tw, th] = match unit {
            1 => [128, 128],
            2 => [128, 64],
            4 => [64, 64],
            8 => [64, 32],
            16 => [32, 32],
            32 => [32, 16],
            _ => [16, 16],
        };
        let (tw, th) = (tw.min(w.next_power_of_two()), th.min(h.next_power_of_two()));
        let (bx, by) = (tw.trailing_zeros(), th.trailing_zeros());
        let (mut morton, mut bit) = (0, 0);
        for i in 0..bx.max(by) {
            if i < bx {
                morton |= (((x % tw) >> i) & 1) << bit;
                bit += 1;
            }
            if i < by {
                morton |= (((y % th) >> i) & 1) << bit;
                bit += 1;
            }
        }
        (((y / th) * w.div_ceil(tw) + x / tw) * tw * th + morton) * unit
    }
    let rt = TextureUsage::RENDER_TARGET;
    // (format, its image's, samples, extent, Metal's answer, the stencil plane's first byte)
    let rows = [
        (
            P::Depth32Float,
            Format::D32_SFLOAT,
            4,
            [100, 70],
            0x30000,
            None,
        ),
        (
            P::Depth32Float,
            Format::D32_SFLOAT,
            2,
            [37, 5],
            0x1000,
            None,
        ),
        (
            P::Depth16Unorm,
            Format::D16_UNORM,
            4,
            [64, 64],
            0x8000,
            None,
        ),
        (P::Stencil8, Format::S8_UINT, 4, [200, 130], 0x30000, None),
        (P::Stencil8, Format::S8_UINT, 2, [37, 5], 0x400, None),
        (
            P::Depth32FloatStencil8,
            Format::D32_SFLOAT_S8_UINT,
            4,
            [64, 64],
            0x14000,
            Some(0x10000),
        ),
        (
            P::Depth32FloatStencil8,
            Format::D32_SFLOAT_S8_UINT,
            2,
            [200, 130],
            0x68000,
            Some(0x50000),
        ),
        (
            P::Depth32FloatStencil8,
            Format::D32_SFLOAT_S8_UINT,
            4,
            [37, 5],
            0x4800,
            Some(0x4000),
        ),
    ];
    for (pf, format, samples, [w, h], size, stencil_at) in rows {
        let what = format!("{pf:?} x{samples} {w}x{h}");
        let layout = texture(T::Type2DMultisample, pf, [w, h, 1], (1, 1, samples));
        let metal = metal_heap_layout(&layout, rt, false, format);
        assert_eq!(metal.answer.map(|a| a.size), Some(size), "{what}");
        let HeapLayout::Tiled(tiles) = metal.layout else {
            panic!("{what}: untiled");
        };
        assert_eq!(tiles.size, size, "{what}");
        for part in &tiles.parts {
            let at = match part.aspect {
                Some(ImageAspectFlags::STENCIL) if pf == P::Depth32FloatStencil8 => {
                    stencil_at.expect("a stencil plane")
                }
                _ => 0,
            };
            assert_eq!(part.at, at, "{what}");
            for (x, y) in (0..u64::from(h)).flat_map(|y| (0..u64::from(w)).map(move |x| (x, y))) {
                let ours = part.at + part.layout.element(0, 0, x, y);
                let unit = part.layout.unit;
                let metal = at + tiled(unit, [u64::from(w), u64::from(h)], [x, y]);
                assert_eq!(ours, metal, "{what} ({x}, {y})");
            }
        }
    }
}

/// Each clear color `fresh_pattern` gives converts to its texel on every device: no rounding of
/// it is in question (`clear_value::rounds_apart`), and Metal's conversion
/// (`clear_value::tests::texel_bytes`) writes the pattern. Every pattern is of a whole number of bytes a texel, and every format with a pattern
/// but a clear color is one whose pattern holds a NaN or an extended-range code.
#[test]
fn fresh_patterns_clear_colors_write_their_texels() {
    let mut cleared = 0;
    for &pf in PixelFormat::ALL {
        match fresh_pattern(pf) {
            Some(FreshPattern::Color {
                texel,
                color: Some(color),
            }) => {
                let bytes = crate::clear_value::tests::texel_bytes(pf, color);
                assert_eq!(bytes.as_deref(), Some(texel), "{pf:?}: {color:?}");
                assert!(
                    !crate::clear_value::rounds_apart(pf, color, false),
                    "{pf:?}: {color:?}"
                );
                cleared += 1;
            }
            Some(FreshPattern::Color { texel, color: None }) => {
                assert!(
                    [2, 4, 8, 16].contains(&texel.len()),
                    "{pf:?}: a NaN or extended-range pattern of {} bytes",
                    texel.len()
                );
            }
            Some(FreshPattern::DepthStencil { depth, stencil }) => {
                assert!(depth.is_some() || stencil.is_some(), "{pf:?}");
                if let Some(FreshDepth::Value(value)) = depth {
                    assert_eq!(value, 1.0, "{pf:?}: Depth16Unorm's all ones");
                }
            }
            None => {}
        }
    }
    assert!(cleared >= 30, "{cleared} formats cleared to their pattern");
}

/// The pattern covers each level whose shorter side is 16 or more of a texture Metal compresses,
/// none of one it keeps uncompressed: the chains and sample grids our probes
/// measured (a 2x texture's grid twice as tall, a 4x one's twice as wide too).
#[test]
fn pattern_levels_are_the_levels_of_16_or_more() {
    let read = TextureUsage::SHADER_READ;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    assert_eq!(pattern_levels(true, read, true, [64, 64], 7), 3);
    assert_eq!(pattern_levels(true, read, true, [64, 64], 1), 1);
    assert_eq!(pattern_levels(true, read, true, [256, 16], 9), 1);
    assert_eq!(pattern_levels(true, read, true, [100, 40], 7), 2);
    assert_eq!(pattern_levels(true, read, true, [1024, 32], 11), 2);
    assert_eq!(pattern_levels(true, read, true, [128, 16], 8), 1);
    assert_eq!(pattern_levels(true, read, true, [16, 256], 9), 1);
    assert_eq!(pattern_levels(true, read, true, [17, 17], 5), 1);
    assert_eq!(pattern_levels(true, read, true, [16, 16], 5), 1);
    assert_eq!(pattern_levels(true, read, true, [15, 15], 4), 0);
    assert_eq!(pattern_levels(true, read, true, [4096, 1], 1), 0);
    assert_eq!(pattern_levels(true, write, true, [64, 64], 7), 0);
    assert_eq!(pattern_levels(true, read, false, [64, 64], 7), 0);
    assert_eq!(pattern_levels(false, write, true, [64, 64], 1), 1);
    let grid = sample_grid(4, [8, 8]).unwrap();
    assert_eq!(
        pattern_levels(true, TextureUsage::RENDER_TARGET, true, grid, 1),
        1
    );
    let rt = TextureUsage::RENDER_TARGET;
    for (samples, [w, h], levels) in [
        (4, [4, 4], 0),
        (4, [7, 7], 0),
        (4, [8, 4], 0),
        (4, [4, 8], 0),
        (4, [9, 8], 1),
        (4, [8, 16], 1),
        (2, [8, 8], 0),
        (2, [16, 8], 1),
        (2, [8, 16], 0),
        (2, [15, 15], 0),
        (2, [16, 16], 1),
    ] {
        let grid = sample_grid(samples, [w, h]).unwrap();
        let got = pattern_levels(true, rt, true, grid, 1);
        assert_eq!(got, levels, "{samples}x {w}x{h}");
    }
}

/// The per-texture notice of a heap texture whose bytes the runtime holds Metal's layout of
/// itself (`own_heap_layout`, `texture/heap.rs`): a multisample texture whose samples Metal lays
/// out in the heap's bytes announces nothing where its image is a shadow of those tiles, and
/// `heap-texture-untiled-layout` where it lies in memory of its own apart from them, which holds
/// none of them; the uncompressed tiles of a texture Metal keeps losslessly compressed announce
/// `heap-texture-lossless-layout`, by its sample count; a layout that is not Metal's carries its
/// own kind; and a route of its own that is measured announces nothing.
#[test]
fn a_multisample_heap_texture_apart_from_metal_s_tiles_is_announced_untiled() {
    use crate::degradation::Degradation as D;
    let read = TextureUsage::SHADER_READ;
    let format = Format::R8G8B8A8_UNORM;
    // A 4-sample texture of a sample grid of 8 x 8, under the 16 Metal compresses from: Metal
    // keeps it uncompressed and lays its samples out in the heap's bytes.
    let ms = texture(
        T::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [4, 4, 1],
        (1, 1, 4),
    );
    assert_eq!(sample_grid(4, [4, 4]), Some([8, 8]));
    let metal = metal_heap_layout(&ms, read, true, format);
    assert!(matches!(metal.layout, HeapLayout::Tiled(_)));
    // A shadow of Metal's own tiles holds the texture as Metal lays it out: no notice.
    assert_eq!(own_heap_layout(&metal.layout, 4, false), None);
    // Its image in memory of its own holds none of them: announced, as the layout Metal's tiles
    // are not measured against is (`multisample_heap_layout`'s).
    let untiled = own_heap_layout(&metal.layout, 4, true);
    assert_eq!(untiled, Some(Untiled::SAMPLES_APART));
    assert_eq!(
        untiled.map(|untiled| untiled.kind),
        Some(D::HeapTextureUntiledLayout)
    );
    // A layout that is not Metal's carries its own kind, wherever the image lies.
    assert_eq!(
        own_heap_layout(&HeapLayout::Untiled(Untiled::DRIVER), 4, false),
        Some(Untiled::DRIVER)
    );
    // A texture Metal keeps losslessly compressed: the uncompressed tiles a shadow holds it in,
    // announced lossless by its sample count, wherever its image lies.
    let big = texture(
        T::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [64, 64, 1],
        (1, 1, 4),
    );
    let compressed = metal_heap_layout(&big, read, true, format);
    let HeapLayout::Untiled(lossless) = compressed.layout else {
        panic!("Metal keeps a 64 x 64 4-sample texture compressed");
    };
    assert_eq!(lossless, Untiled::LOSSLESS_MULTISAMPLE);
    let tiles = shadow_tiles(&big, format, &HeapLayout::Untiled(lossless))
        .expect("a compressed multisample texture has a shadow's uncompressed tiles");
    assert!(matches!(tiles, HeapLayout::Uncompressed(_)));
    for apart in [false, true] {
        assert_eq!(
            own_heap_layout(&tiles, 4, apart),
            Some(Untiled::LOSSLESS_SAMPLES)
        );
        assert_eq!(own_heap_layout(&tiles, 1, apart), Some(Untiled::LOSSLESS));
        // A route of its own that is measured (a 4:2:2 texture's rows, a two-plane texture's
        // planes) lays the texture out as Metal does: nothing to announce.
        assert_eq!(own_heap_layout(&HeapLayout::Measured, 4, apart), None);
    }
}

/// Rows lie as Metal's tiles (`texture/heap.rs`'s `rows_lie_in_metal_tiles`, which decides whether
/// a linear image takes the place of a shadow of a texture Metal keeps uncompressed) only where
/// every element's byte is the same: a one-element level at any pitch, a one-row level, a level
/// at most two elements wide with rows that far apart, and slices at Metal's chain stride; not a
/// 64 x 64 or 4 x 4 level's rows, a two-wide level's rows further apart, slices elsewhere, or a
/// subresource the layouts lack.
#[test]
fn rows_lie_as_metal_s_tiles_only_where_every_element_does() {
    let part = |unit, ty, [w, h]: [u64; 2], slices| TiledPart {
        aspect: None,
        at: 0,
        layout: TiledLayout::of(unit, Elements::Texels, ty, [w, h, 1], 1, slices).unwrap(),
    };
    let rows = |offset, row_pitch| ash::vk::SubresourceLayout {
        offset,
        size: 0,
        row_pitch,
        array_pitch: 0,
        depth_pitch: 0,
    };
    for pitch in [8, 64, 256] {
        assert!(rows_lie_in_metal_tiles(
            &part(8, T::Type2D, [1, 1], 1),
            &[rows(0, pitch)],
            1
        ));
    }
    assert!(rows_lie_in_metal_tiles(
        &part(4, T::Type2D, [256, 1], 1),
        &[rows(0, 1024)],
        1
    ));
    assert!(rows_lie_in_metal_tiles(
        &part(4, T::Type1D, [100, 1], 1),
        &[rows(0, 400)],
        1
    ));
    assert!(rows_lie_in_metal_tiles(
        &part(4, T::Type2D, [2, 8], 1),
        &[rows(0, 8)],
        1
    ));
    assert!(!rows_lie_in_metal_tiles(
        &part(4, T::Type2D, [2, 8], 1),
        &[rows(0, 16)],
        1
    ));
    assert!(!rows_lie_in_metal_tiles(
        &part(4, T::Type2D, [64, 64], 1),
        &[rows(0, 256)],
        1
    ));
    assert!(!rows_lie_in_metal_tiles(
        &part(8, T::Type2D, [4, 4], 1),
        &[rows(0, 32)],
        1
    ));
    assert!(!rows_lie_in_metal_tiles(
        &part(8, T::Type2D, [4, 4], 1),
        &[],
        1
    ));
    let array = part(8, T::Type2DArray, [1, 1], 2);
    let stride = array.layout.stride;
    assert!(rows_lie_in_metal_tiles(
        &array,
        &[rows(0, 8), rows(stride, 8)],
        2
    ));
    assert!(!rows_lie_in_metal_tiles(
        &array,
        &[rows(0, 8), rows(stride + 64, 8)],
        2
    ));
    assert!(!rows_lie_in_metal_tiles(&array, &[rows(0, 8)], 2));
}
