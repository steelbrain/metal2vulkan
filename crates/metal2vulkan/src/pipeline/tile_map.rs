//! A sparse texture's tile map: the words a function reads to answer the residency of a sparse
//! texture where the device has no shader residency.
//!
//! Metal maps a sparse texture's memory a tile at a time (`updateTextureMapping:mode:region:
//! mipLevel:slice:` on an `MTLResourceStateCommandEncoder`), and a function reads whether the
//! texels it touched were mapped (`sparse_color::resident()`). A Vulkan device without
//! `shaderResourceResidency` cannot answer that from the image. The consumer then backs the
//! texture with an ordinary image and keeps, beside it, a storage buffer with a bit per tile of
//! each level and slice that is set while the tile is mapped. A module whose slot has the
//! tile-map arm (`RuntimeFact::SparseTexture { tile_map: true, .. }`) reads it at the slot's
//! `ResourceKind::TileMap` binding. The consumer builds the words, mirroring each mapping.
//!
//! **Format** (version 1): little-endian 32-bit words.
//!
//! | word | meaning |
//! |---|---|
//! | 0 | the format's version tag, `"M2T1"` read as a little-endian word |
//! | 1, 2, 3 | the tile's width, height, and depth in texels (each at least 1) |
//! | 4 | the slice count `S` (array layers; 1 for a 2D or 3D texture) |
//! | 5 | `B`, the bits of one slice |
//! | 6, 7 | 0 |
//! | `8 + 4l + 0..4` | level `l < 16`: its first bit in a slice, its tiles across x, y, z |
//! | [`BITS_WORD`]`..` | the bits: bit `b` is bit `b % 32` of word `BITS_WORD + b / 32` |
//!
//! A level in the texture's mip tail (from Metal's `firstMipmapInTail`) has one tile, the slice's
//! tail bit, which every level of the tail shares: Metal maps and unmaps the tail as one unit. An
//! entry past the texture's last level has one tile at bit 0.
//!
//! **Decoding**: the residency of texel `(x, y, z)` of level `l` and slice `s` is
//! bit `min(s, S - 1)·B + base + (t_z·n_y + t_y)·n_x + t_x` of level `l' = min(l, 15)`, where
//! `(base, n_x, n_y, n_z)` is that level's entry and `t_a = min(c_a / tile_a, n_a - 1)`. Every
//! index is clamped into the words the format states, so a function reads no word past the map
//! even for an access outside the texture (whose residency it answers from the coordinate alone).

/// Words 1, 2, 3: the tile's extent in texels.
pub const TILE_WORD: u32 = 1;
/// Word 4: the slice count.
pub const SLICES_WORD: u32 = 4;
/// Word 5: the bits of one slice.
pub const SLICE_BITS_WORD: u32 = 5;
/// The levels the map describes: a Metal texture has at most 15 (16384 texels on a side).
pub const MAX_LEVELS: u32 = 16;
/// The word of level 0's entry; level `l`'s is `LEVELS_WORD + LEVEL_WORDS·l`.
pub const LEVELS_WORD: u32 = 8;
/// The words of one level's entry: its first bit, then its tiles across x, y, and z.
pub const LEVEL_WORDS: u32 = 4;
/// The first word of the bits.
pub const BITS_WORD: u32 = LEVELS_WORD + LEVEL_WORDS * MAX_LEVELS;

/// The bytes of the tile map of a one-level sparse texture of `slices` slices of `extent` texels,
/// in tiles of `tile` texels with no mip tail, where tile `(x, y, z)` (in tiles) of slice `s` is
/// mapped exactly where `mapped(s, [x, y, z])` holds: the map a test binds as Metal's mappings.
#[cfg(test)]
pub(crate) fn one_level_map(
    extent: [u32; 3],
    slices: u32,
    tile: [u32; 3],
    mapped: impl Fn(u32, [u32; 3]) -> bool,
) -> Vec<u8> {
    let tiles = [0, 1, 2].map(|axis| extent[axis].div_ceil(tile[axis]));
    let slice_bits = tiles.iter().product::<u32>();
    let mut words = vec![0u32; (BITS_WORD + (slice_bits * slices).div_ceil(32)) as usize];
    words[0] = u32::from_le_bytes(*b"M2T1");
    words[TILE_WORD as usize..][..3].copy_from_slice(&tile);
    words[SLICES_WORD as usize] = slices;
    words[SLICE_BITS_WORD as usize] = slice_bits;
    for level in 0..MAX_LEVELS {
        // Each level's first bit is 0; a level past the first has one tile.
        let at = (LEVELS_WORD + LEVEL_WORDS * level) as usize + 1;
        words[at..at + 3].copy_from_slice(if level == 0 { &tiles } else { &[1; 3] });
    }
    let [nx, ny, nz] = tiles;
    for s in 0..slices {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if mapped(s, [x, y, z]) {
                        let bit = s * slice_bits + (z * ny + y) * nx + x;
                        words[(BITS_WORD + bit / 32) as usize] |= 1 << (bit % 32);
                    }
                }
            }
        }
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
