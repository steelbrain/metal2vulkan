//! A heap texture over host pages whose elements lie in the heap's bytes as Metal's GPU lays
//! them out: tiled (the consumer's report: a texture placed in a heap over a guest's pages is not
//! linear on Metal).
//!
//! **Metal's layout** (an Apple M4 under macOS 27.0.1, our probes and the M3 probe's L rows: a
//! `Shared` placement heap over host pages, `setAddressRanges:`; each element of each slice, level
//! and depth slice written by `replaceRegion` with values that name it, the span read through the
//! pages; an M3 Max under macOS 26.7 alike on every row of plain formats, the consumer's run of the
//! probe):
//!
//! - **Tiles.** Elements of `B` bytes lie in tiles of at most 16 KiB: 128 x 128 elements of 1
//!   byte, 128 x 64 of 2, 64 x 64 of 4, 64 x 32 of 8, 32 x 32 of 16 (`largest_tile`). A level
//!   of `w x h` elements takes tiles of `min(TW, pot(w)) x min(TH, pot(h))`, `⌈w / tw⌉` a row;
//!   the tiles lie row after row, and a tile's elements in Morton order, x's bit lowest, the
//!   longer side's extra bits on top (`TiledLayout::element`).
//! - **Levels.** Each level lies at the end of the one before. A level both of whose sides are a
//!   full tile or more takes the tiles of level 0's grid (`gx = ⌈w0 / TW⌉`, `gy = ⌈h0 / TH⌉`)
//!   halved `l` times, `⌈gx / 2^l⌉ · ⌈gy / 2^l⌉` of them, plus the whole tiles in the area the
//!   halving's fractions leave (`⌊gx · gy / 4^l − ⌊gx / 2^l⌋ · ⌊gy / 2^l⌋⌋`). The first level
//!   narrower than a tile on either side takes its own sides padded: a side whose other side
//!   is narrower than a tile to a power of two, else to a multiple of the tile; each later
//!   level halves the first one's power-of-two sides. A narrow level's bytes are rounded up to
//!   a multiple of 128 (`level_bytes`).
//! - **Chains.** Each slice of the texture, and each depth slice of a 3D one, holds a chain of
//!   the levels. One chain of a one-level texture is its level; of a mipmapped one, the full
//!   chain's bytes, whatever levels the texture has. An array's, cube's and 3D texture's chains
//!   lie at multiples of 16 KiB from a chain of 16 KiB on, the texture then aligned to 16 KiB;
//!   any other texture is aligned to 128. `heapTextureSizeAndAlignWithDescriptor:` answers the
//!   chains' bytes and that alignment: every uncompressed answer measured (4126 of every plain
//!   format, sRGB, packed, extended-range, depth and stencil formats among them, every type; the
//!   bases of 476 more).
//! - **Block-compressed formats** (an Apple M4 under macOS 27.0.1: every BC, ETC2, EAC and ASTC
//!   format, 815 textures, every element of every slice, level and depth slice;
//!   `tests/heap_tiled_blocks.rs`) lay out a block as an element. Each level's allocation counts
//!   level 0's blocks halved per level, rounding up (`⌈b0 / 2^l⌉`: its full levels' tiles, its
//!   first narrow level's sides). Its tiles count its blocks by a placement of the format's family
//!   (`BlockPlacement`), `⌈e / tw⌉` tiles a row: `Halved` counts them so too (ASTC on the M4);
//!   `Narrow` counts them so while the level before is a tile or more on that side, then halves the
//!   first narrower level's texels (its blocks' whole texels), rounding down (BC, ETC2 and EAC on
//!   the M4: level 3 of a 100 x 36 `BC1_RGBA` chain, 3 x 1 blocks of its own and 4 x 2 halved, lies
//!   in tiles of 4 x 1, where `Halved` would take 4 x 2). The M3 Max under 27.0.1 answers every
//!   texture of `tests/heap_tiled_blocks.rs` as the M4 does, its CPU's spans (`replaceRegion`,
//!   `getBytes`) and its GPU's (a blit from or into a buffer), which count a level's tiles a row
//!   alike. Each family's placement is the release's (`metal_release::heap_blocks`): the CPU
//!   copies and the kernel's moves place blocks by `TiledLevel::across`, as Metal's own two paths
//!   do. PVRTC's blocks lie so too, `Halved` (an Apple M3 Max: 320 answers, every PVRTC format
//!   square from 4 to 2048 texels, one level and full chains, one slice and three; every block of
//!   each level of six textures, written by a row-linear blit; `tests/heap_tiled_blocks.rs`). A
//!   block-compressed format of none of those families stays untiled
//!   (`heap-texture-block-compressed-layout`), its answer its device's.
//! - **Lossless compression** (14174 answers of every plain format, type, shape, usage and storage
//!   mode). Metal keeps a texture losslessly compressed where `allowGPUOptimizedContents` holds and
//!   the shorter side of its level 0 is 16 texels or more (16 x 16 compressed; 16 x 8, 6 x 43 and
//!   2048 x 5 not), a colour texture of none of `ShaderWrite`, `PixelFormatView` and `ShaderAtomic`
//!   usage and a depth or stencil texture of any (`metal_compresses`, `metal_keeps_uncompressed`).
//!   Its bytes are no layout of its texels. It answers its chains uncompressed, then each chain's
//!   metadata (`lossless_answer`, `chain_metadata`): for each of its levels, 8 bytes for each 16 x
//!   16 tile of the level, its tiles on each side padded to a power of two, rounded up to 128. A
//!   one-level texture's metadata is its level's (8538 answers, every one). A mipmapped one's
//!   covers the full chain's levels, whatever levels the texture has, while level 0's sides, each
//!   padded to a multiple of 16 and halved per level, have a longer side of 16 or more, each
//!   level's tiles counted over those halved sides (an Apple M4 under macOS 27.0.1, and an M3 Max
//!   under 26.7, alike on every one of their 410356 answers: 2D textures of 12 colour formats and
//!   `Depth16Unorm`, `Depth32Float` and `Stencil8` at every pair of 30 sides from 1 to 1440,
//!   two-slice 2D arrays and cubes of 5 colour formats at 7 sizes, 1 to 10 levels, usages 0x1, 0x5,
//!   0x7 and 0x10005, both storage modes; a 224 x 928 chain's metadata covers its first 6 levels).
//!   Every chain takes it alike: a 3D texture's, one a depth slice of level 0, and a cube array's
//!   (114921 answers on the M4: 3D textures of 3 colour formats at 12 x 12 sides and 6 depths, 2D
//!   arrays of 1, 3 and 6 slices, cubes and cube arrays of 1 and 2 of 7 colour and 4 depth and
//!   stencil formats, every level count, `allowGPUOptimizedContents` off, on and unset for the
//!   depth and stencil ones, which compress at no usage with it off).
//!   It stays untiled (`heap-texture-lossless-layout`).
//! - **Multisample textures** (the consumer's ask; on an Apple M4 under macOS 27.0.1, and the same
//!   package run on an M3 Max under 26.7 (the coordinator's run), which answered every one of its
//!   27648 descriptors and laid out every byte of its 9 textures as the M4; 18 formats, 2 and 4
//!   samples (both answer no to 8), 1 to 3 slices, six usages, `Shared` and `Private`). Metal keeps
//!   a multisample texture losslessly compressed by the rule above over its **sample grid**
//!   (`sample_grid`: `w x 2h` at 2 samples, `2w x 2h` at 4): with `allowGPUOptimizedContents`, the
//!   grid's shorter side 16 or more, and a colour texture of neither `ShaderWrite` nor
//!   `PixelFormatView` usage or a depth or stencil texture of any; the storage mode decides nothing
//!   (every one of 10152 descriptors around the threshold, on both machines). Uncompressed, a
//!   pixel's samples lie side by side as one element of `S·B` bytes (sample `s` at `s·B`), laid out
//!   as a one-level texture's elements above (tiles of 32 x 16 elements of 32 bytes and 16 x 16 of
//!   64), a multisample array's slices as an array's: every sample of 21 textures drawn at the
//!   bytes the rule names (14 on the M4, nothing past their answers written; 7 on both machines).
//!   Compressed, its bytes are no layout of its samples, and its answer is that one's plus, for
//!   each slice, its sample grid's metadata as a one-level texture's above
//!   (`multisample_heap_layout`; every answer of every format but `Depth32Float_Stencil8` measured,
//!   93422 on the M4 and 24576 on the M3 Max, compressed and not, at 128 bytes or, an array's chain
//!   of 16 KiB or more, 16 KiB, a one-slice array's too). A `Depth32Float_Stencil8` texture answers
//!   two planes: its depth plane (`Depth32Float`'s answer, compressed or not) rounded up to 16 KiB,
//!   then its stencil plane (`Stencil8`'s), at the depth plane's alignment (every one of its 4800
//!   answers on the M4 and 3072 on the M3 Max, 1 to 3 slices; before, its device's answer, whose
//!   alignment on MoltenVK was 128 where Metal's is 16 KiB for an array of a 16-KiB depth chain,
//!   the consumer's report), and so does a one-sample one, of every type and level count,
//!   compressed or not (every one of 14552 answers each on the M4 and the M3 Max, 2D, and 12330 on
//!   the M4, 2D arrays, cubes and cube arrays). The samples of depth and stencil textures lie by
//!   the same rule (an Apple M3 Max under macOS 27.0.1, `allowGPUOptimizedContents` off: every
//!   sample of `Depth32Float` textures of 8 x 8 to 100 x 70 at 2 and 4 samples, of `Depth16Unorm`
//!   ones at 4, and of `Stencil8` and `Depth32Float_Stencil8` ones of 8 x 8 to 200 x 130 at 2 and
//!   4, and 2- and 3-slice arrays of `Depth32Float` and `Stencil8` at 2 and 4, each slice a chain
//!   from the one before, drawn by a per-sample draw naming each and found at the bytes the rule
//!   names), a
//!   `Depth32Float_Stencil8` texture's stencil plane from its depth plane's answer rounded up to 16
//!   KiB (`multisample_tiles`). Over host pages, and in any heap whose memory or Metal's answer
//!   cannot hold its image, one is a shadow of its samples in Metal's tiles (a compressed one's
//!   uncompressed tiles, announced `heap-texture-lossless-layout`), where its image holds Metal's
//!   elements and the device moves its samples to and from a buffer (`SampleMoves`); elsewhere its
//!   image lies in memory of its own, its samples apart from the heap's bytes
//!   (`heap-texture-untiled-layout`, `heap.rs`).
//! - **Two aspects.** A single-sample `Depth32Float_Stencil8` texture lies in two planes, its depth
//!   elements as a `Depth32Float` texture of its shape lies, from its first byte, and its stencil
//!   elements as a `Stencil8` one lies, from the `Depth32Float` answer rounded up to 16 KiB
//!   (`TiledTexture`; an Apple M3 Max under macOS 27.0.1, `allowGPUOptimizedContents` off: every
//!   texel of each aspect of one-level 16 x 16 to 300 x 200 textures, of 7-level 64 x 64 and
//!   100 x 70 ones and of arrays of 2 and 3 slices, 1 to 4 levels, written by blits and found in
//!   the heap's bytes, and the bytes written there read back by blits). With it on Metal keeps the
//!   depth plane compressed, and a shadow holds its uncompressed tiles, the stencil plane past the
//!   compressed answer (`heap-texture-lossless-layout`).
//! - **A stencil view.** An `X32_Stencil8` texture lies as a `Stencil8` texture of its shape lies,
//!   compressed or not, and answers as one (an Apple M3 Max under macOS 27.0.1: 1720 answers, 1 to
//!   1024 texels a side, 1 to 3 levels, 1 and 3 slices, allowGPUOptimizedContents off and on; every
//!   texel of one-level 8 x 8 to 300 x 200 textures written by blits and found where a `Stencil8`
//!   texture's lie, and the bytes written there read back by blits). Its image's tiles are its
//!   stencil aspect's; the image's depth holds none of its texels.
//! - **Not measured**: a PVRTC format's layout under macOS 26.7; the runtime holds it untiled
//!   (`heap-texture-block-compressed-layout`) at its device's answer.
//!
//! **Held** as the texture's image in memory of its own, a shadow of the heap's bytes
//! (`shadow.rs`, `ShadowTarget::Tiled`): into the image, the texture's bytes copied into a
//! staging buffer, one dispatch of `bind::tiled_texels_module` moving each level's elements into
//! a tight buffer of the image's elements (`heap_plane_layout`), and that buffer copied into
//! the image; back, the heap's bytes copied into the staging buffer again, the image into the
//! tight buffer, the kernel into the staging buffer (each element's bytes; every other byte as
//! the heap held it), and each level's bytes of each chain on into the heap. Each step is
//! ordered after the one before by a full barrier, and the buffers are the texture's for its
//! life. The texture's CPU copies move each element between the heap's bytes and the caller's
//! as laid out (`TiledLayout::element`); a command that writes the bytes two ways merges as
//! any shadow's (`shadow_merge.rs`), the shadow's texels laid out tiled for it
//! (`HeapTiledCopies::record_out`).
//!
//! A **multisample** texture's image moves to and from the tight buffer, which Vulkan copies no
//! multisampled image to or from, by the interleave copies of the multisampled blits
//! (`sample_interleave.rs`, [`SampleMoves`]): each pixel's samples one element of all their bytes
//! there, as Metal's blits lay them out in a buffer and its tiles hold them; a depth or stencil
//! aspect's through the blits' own moves (maintenance8's copies, or the draws), whatever they
//! refuse refusing the shadow (tried as the texture is made, on one pixel of its format, usage
//! and sample count: `Device::sample_moves_run`). The moves are planned as an encoder records a
//! copy (`EncoderCore::sample_moves`), each way as the copy goes, their images (a band's wide and
//! scratch images, and the draws' images) taken from those the encoder's queue keeps, and each
//! band, aspect and slice of the copy going through the images of its first of its extent
//! (`copy_images.rs`); the encoder's command buffer holds them until it completes. The texture
//! holds none: before, it planned them as it was made and held their images for its life. Where
//! they write a 32-bit float depth by the `FragDepth` draw on a driver measured to change some
//! values, the texture announces `drawn-depth-copy` once, as it is made.
//!
//! A texture with a **depth** aspect, of any sample count, takes the heap's bytes into its image
//! by moves that may change a value: a copy into a depth image takes no value outside `[0, 1]`
//! without `VK_EXT_depth_range_unrestricted` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), so on
//! a device that would store a 32-bit float one clamped (`depth_clamp::clamps_depth`: ANV on the Iris Xe
//! of RPL-P) the move into the tight buffer clamps each such depth into `[0, 1]`, a NaN to 1.0
//! (`TiledLevelMove::clamps_depth`), the value the device stores: `restricted-depth-upload`,
//! announced as the texture is made. The drivers that `drawn-depth-copy` names change values
//! inside it too. So a merge's keep holds
//! the heap's bytes as the image stands for them, not the image's (`ShadowCopies::record_kept`):
//! the staging buffer as the shadow's last copy left it, the bytes the copy in read or those the
//! last copy back or merge wrote, and no move out of the image. A keep of the image would hold
//! the changed value, and the merge would keep the heap's byte in each byte a command wrote that
//! equals it; a keep of the heap's bytes as it runs would hold the CPU's writes since the copy in,
//! which the merge would undo. A byte the image holds otherwise than the heap is written back:
//! every byte a command wrote, and the values the copy in changed.

use super::super::blit_plan::BufferLayout;
use super::super::buffer::Buffer;
use super::super::copy_plan::{block_bytes, block_extent, depth_bytes, TextureRegion};
use super::super::degradation::Degradation;
use super::super::depth_copy;
use super::super::error::{internal, Error};
use super::super::heap::SizeAndAlign;
use super::super::image_plan::block_family;
use super::super::metal_release::heap_blocks;
use super::super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::super::row_copies::RowCopies;
use super::super::sample_interleave::InterleaveCopy;
use super::super::shadow::{self, ShadowStaging};
use super::super::texture_descriptor::TextureLayout;
use super::super::tracker::Barrier;
use super::super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::super::{depth_clamp, Device};
use super::copy_images::{Reused, UseMaker};
use super::heap::HeapPlane;
use super::{copy_units, texel_units, Texture, TextureShared};
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{TiledLevelMove, TiledTexelsModule};
use metal2vulkan::metal::format::{aspects, byte_format, is_depth_stencil, srgb_raw_twin};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageAspectFlags};
use std::sync::{OnceLock, Weak};

/// The bytes of Metal's largest tile, whatever its elements' size.
const TILE_BYTES: u64 = 16 << 10;
/// The multiple an array's, cube's or 3D texture's chains lie at once one holds this many bytes,
/// and its alignment then.
const PAGE: u64 = 16 << 10;
/// The multiple a narrow level's bytes are rounded up to, and any other texture's alignment.
const LINE: u64 = 128;
/// Where a shape lies in the shape buffer, per direction: past its bytes, and a multiple of
/// every `minStorageBufferOffsetAlignment`.
const SHAPE_SLOT: u64 = 1024;
/// The side of the square tile of texels whose compression metadata takes `METADATA_BYTES`.
const METADATA_TILE: u64 = 16;
/// The bytes of compression metadata a `METADATA_TILE` square takes.
const METADATA_BYTES: u64 = 8;
/// The shortest side of a level 0 Metal compresses.
const COMPRESSED_SIDE: u64 = 16;

/// Metal's largest tile of elements of `unit` bytes, elements across and down; `None` for a size
/// Metal has no tile of. Elements of 32 and 64 bytes are a multisample texture's pixels.
pub(crate) fn largest_tile(unit: u64) -> Option<[u64; 2]> {
    Some(match unit {
        1 => [128, 128],
        2 => [128, 64],
        4 => [64, 64],
        8 => [64, 32],
        16 => [32, 32],
        32 => [32, 16],
        64 => [16, 16],
        _ => return None,
    })
}

/// The power of two `v` rounds up to (1 for 0).
fn pot(v: u64) -> u64 {
    v.max(1).next_power_of_two()
}

/// The levels of a full chain whose largest side is `side`.
fn full_levels(side: u64) -> u32 {
    u64::BITS - side.max(1).leading_zeros()
}

/// The bytes a level of `[w, h]` elements of `unit` bytes takes where it is narrower than the
/// `largest` tile on a side: each side padded, to a power of two where the other side is
/// narrower than the tile, else to a multiple of its tile's side; rounded up to 128.
fn narrow_bytes(unit: u64, largest: [u64; 2], [w, h]: [u64; 2]) -> u64 {
    let tile = [largest[0].min(pot(w)), largest[1].min(pot(h))];
    let width = if h < largest[1] {
        pot(w)
    } else {
        w.next_multiple_of(tile[0])
    };
    let height = if w < largest[0] {
        pot(h)
    } else {
        h.next_multiple_of(tile[1])
    };
    (width * height * unit).next_multiple_of(LINE)
}

/// The bytes Metal lays level `level` of a chain out in, whose level 0 is `[w0, h0]` elements of
/// `unit` bytes in tiles of at most `largest`, its levels' elements as `elements` takes them (the
/// module docs).
pub(crate) fn level_bytes(
    unit: u64,
    largest: [u64; 2],
    [w0, h0]: [u64; 2],
    level: u32,
    elements: Elements,
) -> u64 {
    let sides = |k: u32| elements.allocated([w0, h0], k);
    let narrow = |[w, h]: [u64; 2]| w < largest[0] || h < largest[1];
    let here = sides(level);
    if narrow(here) {
        let first = (0..=level).find(|&k| narrow(sides(k))).unwrap_or(level);
        if first == level {
            return narrow_bytes(unit, largest, here);
        }
        let [w, h] = sides(first);
        let halved = level - first;
        return narrow_bytes(
            unit,
            largest,
            [(pot(w) >> halved).max(1), (pot(h) >> halved).max(1)],
        );
    }
    let [gx, gy] = [w0.div_ceil(largest[0]), h0.div_ceil(largest[1])];
    let step = 1u64 << level;
    let whole = gx.div_ceil(step) * gy.div_ceil(step);
    let area = gx * gy - (((gx >> level) * (gy >> level)) << (2 * level));
    (whole + (area >> (2 * level))) * TILE_BYTES
}

/// What a texture's elements are, and how Metal sizes each level's for its tiles and its
/// allocation (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elements {
    /// A plain format's texels: a level's own, `⌊w0 / 2^l⌋`, for both.
    Texels,
    /// A block-compressed format's blocks of `block` texels, placed as `placement` says.
    Blocks {
        block: [u64; 2],
        placement: BlockPlacement,
    },
}

/// How Metal places a block-compressed texture's blocks in its levels' tiles (the module
/// docs). Each level is allocated alike under both, of level 0's blocks halved per level,
/// rounding up; each lies in tiles of its elements counted as here, `⌈e / tw⌉` tiles a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockPlacement {
    /// Level 0's blocks halved per level, rounding up: `⌈b0 / 2^l⌉`.
    Halved,
    /// Level 0's blocks halved, rounding up, while the level before is a tile or more on that
    /// side; from the first level narrower than a tile, that level's texels (its blocks' whole
    /// texels) halved per level, rounding down, in blocks.
    Narrow,
}

/// The families of block-compressed formats Metal places alike (`metal_release::heap_blocks`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockFamily {
    /// BC1 to BC7, linear, sRGB, signed and unsigned.
    Bc,
    /// ETC2 and EAC, linear, sRGB and signed.
    Etc,
    /// ASTC of every block size, LDR, sRGB and HDR.
    Astc,
    /// PVRTC of 2 and 4 bits a pixel, RGB and RGBA, linear and sRGB.
    Pvrtc,
}

impl BlockFamily {
    /// The family of `pf`; `None` for a format of none (every plain format).
    pub(crate) fn of(pf: PixelFormat) -> Option<BlockFamily> {
        use PixelFormat as P;
        Some(match block_family(pf) {
            P::BC1RGBA
            | P::BC2RGBA
            | P::BC3RGBA
            | P::BC4RUnorm
            | P::BC5RGUnorm
            | P::BC6HRGBUfloat
            | P::BC7RGBAUnorm => BlockFamily::Bc,
            P::ETC2RGB8 | P::ETC2RGB8A1 | P::EACR11Unorm | P::EACRG11Unorm | P::EACRGBA8 => {
                BlockFamily::Etc
            }
            P::ASTC4x4LDR
            | P::ASTC5x4LDR
            | P::ASTC5x5LDR
            | P::ASTC6x5LDR
            | P::ASTC6x6LDR
            | P::ASTC8x5LDR
            | P::ASTC8x6LDR
            | P::ASTC8x8LDR
            | P::ASTC10x5LDR
            | P::ASTC10x6LDR
            | P::ASTC10x8LDR
            | P::ASTC10x10LDR
            | P::ASTC12x10LDR
            | P::ASTC12x12LDR
            | P::ASTC4x2LDR
            | P::ASTC8x4LDR => BlockFamily::Astc,
            P::PVRTCRGB2BPP | P::PVRTCRGB4BPP | P::PVRTCRGBA2BPP | P::PVRTCRGBA4BPP => {
                BlockFamily::Pvrtc
            }
            _ => return None,
        })
    }
}

impl Elements {
    /// The texels of an element, across and down.
    fn block(self) -> [u64; 2] {
        match self {
            Elements::Texels => [1, 1],
            Elements::Blocks { block, .. } => block.map(|side| side.max(1)),
        }
    }

    /// Level `level`'s elements as its allocation counts them, of level 0's `[w0, h0]`.
    fn allocated(self, [w0, h0]: [u64; 2], level: u32) -> [u64; 2] {
        let shift = level.min(63);
        match self {
            Elements::Texels => [(w0 >> shift).max(1), (h0 >> shift).max(1)],
            Elements::Blocks { .. } => [w0, h0].map(|side| halved(side, shift)),
        }
    }

    /// Level `level`'s elements as its tiles count them, of level 0's `[w0, h0]` in tiles of at
    /// most `largest`, its own `own`.
    fn tiled(self, [w0, h0]: [u64; 2], level: u32, largest: [u64; 2], own: [u64; 2]) -> [u64; 2] {
        let shift = level.min(63);
        match self {
            Elements::Texels => own,
            Elements::Blocks {
                placement: BlockPlacement::Halved,
                ..
            } => [w0, h0].map(|side| halved(side, shift)),
            Elements::Blocks {
                block,
                placement: BlockPlacement::Narrow,
                ..
            } => [0, 1]
                .map(|axis| narrowed([w0, h0][axis], shift, largest[axis], block[axis].max(1))),
        }
    }
}

/// `blocks` halved `shift` times, rounding up.
fn halved(blocks: u64, shift: u32) -> u64 {
    blocks.max(1).div_ceil(1 << shift).max(1)
}

/// A side of `blocks` blocks of `block` texels, at level `shift`, as `BlockPlacement::Narrow`
/// counts it in tiles of `tile` on that side.
fn narrowed(blocks: u64, shift: u32, tile: u64, block: u64) -> u64 {
    let mut side = blocks.max(1);
    let mut k = 0;
    while k < shift && side >= tile {
        side = side.div_ceil(2);
        k += 1;
    }
    ((side * block) >> (shift - k)).max(1).div_ceil(block)
}

/// The Morton index of `(x, y)` in a tile of `2^tx x 2^ty` elements: x's bit lowest, the longer
/// side's extra bits on top.
fn morton(x: u64, y: u64, tx: u32, ty: u32) -> u64 {
    let (mut m, mut k) = (0, 0);
    for i in 0..tx.max(ty) {
        if i < tx {
            m |= ((x >> i) & 1) << k;
            k += 1;
        }
        if i < ty {
            m |= ((y >> i) & 1) << k;
            k += 1;
        }
    }
    m
}

/// One level of `TiledLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TiledLevel {
    /// Its elements across and down.
    pub(crate) extent: [u64; 2],
    /// Its depth slices (a 3D level's; 1 for any other).
    pub(crate) depth: u64,
    /// Its tile, elements across and down: powers of two.
    pub(crate) tile: [u64; 2],
    /// Its tiles a row, as Metal's CPU copies and its GPU count them (`TiledLayout::element`; the
    /// kernel's moves).
    pub(crate) across: u64,
    /// Its first byte from its chain's.
    pub(crate) base: u64,
    /// The bytes Metal lays it out in.
    pub(crate) bytes: u64,
}

/// Metal's layout of a heap texture's elements (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TiledLayout {
    /// The bytes of an element.
    pub(crate) unit: u64,
    /// The texture's levels.
    pub(crate) levels: Vec<TiledLevel>,
    /// Its chains: its slices (six a cube), or a 3D texture's depth slices.
    pub(crate) chains: u64,
    /// A chain's bytes: its level's of a one-level texture, else the full chain's.
    pub(crate) chain: u64,
    /// The levels in a chain's bytes: 1 of a one-level texture, else the full chain's.
    pub(crate) chain_levels: u32,
    /// A 3D texture's: level `l` lies in the chains of its depth slices alone.
    pub(crate) volume: bool,
    /// The bytes from one chain to the next.
    pub(crate) stride: u64,
    /// The bytes Metal answers.
    pub(crate) size: u64,
    /// The alignment Metal answers.
    pub(crate) align: u64,
}

impl TiledLayout {
    /// Metal's layout of a texture of `texture_type` whose level 0 is `extent` texels (its depth
    /// a 3D texture's) in elements of `unit` bytes as `elements` says (texels, or a
    /// block-compressed format's blocks), with `levels` levels and `slices` slices (six a cube);
    /// `None` for an element size Metal has no tile of. A level's elements are its own texels'
    /// blocks; its allocation and its tiles count them as `elements` says (the module docs).
    pub(crate) fn of(
        unit: u64,
        elements: Elements,
        texture_type: TextureType,
        extent: [u64; 3],
        levels: u32,
        slices: u64,
    ) -> Option<TiledLayout> {
        use TextureType as T;
        let largest = largest_tile(unit)?;
        let one_row = matches!(texture_type, T::Type1D | T::Type1DArray);
        let volume = texture_type == T::Type3D;
        let width = extent[0].max(1);
        let height = if one_row { 1 } else { extent[1].max(1) };
        let d0 = if volume { extent[2].max(1) } else { 1 };
        let [bw, bh] = elements.block();
        let [w0, h0] = [width.div_ceil(bw), height.div_ceil(bh)];
        let bytes = |level: u32| level_bytes(unit, largest, [w0, h0], level, elements);
        let mut tiled = Vec::new();
        let mut end = 0;
        for level in 0..levels {
            let shift = level.min(63);
            let [w, h] = [
                (width >> shift).max(1).div_ceil(bw),
                (height >> shift).max(1).div_ceil(bh),
            ];
            let [tw, th] = elements.tiled([w0, h0], level, largest, [w, h]);
            let tile = [largest[0].min(pot(tw)), largest[1].min(pot(th))];
            let taken = bytes(level);
            let across = tw.div_ceil(tile[0]);
            tiled.push(TiledLevel {
                extent: [w, h],
                depth: (d0 >> shift).max(1),
                tile,
                across,
                base: end,
                bytes: taken,
            });
            end += taken;
        }
        let chain_levels = match levels {
            1 => 1,
            _ => full_levels(width.max(height).max(d0)),
        };
        let chain = match levels {
            1 => end,
            _ => (0..chain_levels).map(bytes).sum(),
        };
        let (stride, align) = match layered(texture_type) && chain >= PAGE {
            true => (chain.next_multiple_of(PAGE), PAGE),
            false => (chain, LINE),
        };
        let chains = if volume { d0 } else { slices.max(1) };
        Some(TiledLayout {
            unit,
            levels: tiled,
            chains,
            chain,
            chain_levels,
            volume,
            stride,
            size: stride * chains,
            align,
        })
    }

    /// The byte at which element `(x, y)` of level `level` of chain `chain` starts, from the
    /// texture's first byte, as Metal's CPU copies place it (`TiledLevel::across`).
    pub(crate) fn element(&self, chain: u64, level: usize, x: u64, y: u64) -> u64 {
        let l = &self.levels[level];
        let [tw, th] = l.tile;
        let tile = (y / th) * l.across + x / tw;
        let inside = morton(x % tw, y % th, tw.trailing_zeros(), th.trailing_zeros());
        chain * self.stride + l.base + (tile * tw * th + inside) * self.unit
    }
}

/// Whether Metal keeps a texture losslessly compressed (the module docs): with
/// `allowGPUOptimizedContents`, a level 0 whose shorter side (`[width, height]`, a 1D texture's
/// height 1; a multisample texture's sample grid, [`sample_grid`]) is 16 texels or more, and
/// `usage` (the effective usage: `Unknown` is every usage) Metal compresses
/// ([`metal_keeps_uncompressed`]).
pub(crate) fn metal_compresses(
    colour: bool,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    [width, height]: [u64; 2],
) -> bool {
    allow_gpu_optimized_contents
        && width.min(height) >= COMPRESSED_SIDE
        && !metal_keeps_uncompressed(colour, usage)
}

/// [`metal_compresses`] of a texture of `layout`: its level 0, a 1D texture's one row high, or a
/// multisample texture's sample grid ([`sample_grid`]; Metal keeps no other sample count's
/// texture compressed here).
pub(crate) fn metal_compresses_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
) -> bool {
    let [width, height, _] = layout.extent.map(u64::from);
    let one_row = matches!(
        layout.texture_type,
        TextureType::Type1D | TextureType::Type1DArray
    );
    let level0 = [width, if one_row { 1 } else { height }];
    let shape = match layout.samples {
        1 => Some(level0),
        samples => sample_grid(samples, level0),
    };
    let colour = !is_depth_stencil(layout.pixel_format);
    shape.is_some_and(|shape| metal_compresses(colour, usage, allow_gpu_optimized_contents, shape))
}

/// Whether Metal keeps a texture of `usage` (the effective usage: `Unknown` is every usage)
/// uncompressed whatever its size and `allowGPUOptimizedContents`: a colour texture (`colour`)
/// of `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage; never a depth or stencil texture,
/// which Metal compresses at every usage (an Apple M4 under macOS 27.0.1: every
/// one of the 256 sets of `ShaderRead`, `ShaderWrite`, `RenderTarget`, 0x8, `PixelFormatView`,
/// `NoConcurrentAccess`, `Video` and `BlockWritesOnly` and 9 sets with `ShaderAtomic`, at 13
/// formats, 15 shapes, both storage modes and the property off, on and unset, each answer the
/// texture's compressed or its plain size as this rule says, 63210 rows; the consumer's M3 Max
/// rows under 26.7 alike). Of the private usages, `NoConcurrentAccess`, `Video` and
/// `BlockWritesOnly` keep nothing uncompressed, alone too.
pub(crate) fn metal_keeps_uncompressed(colour: bool, usage: TextureUsage) -> bool {
    let plain = TextureUsage::SHADER_WRITE
        .union(TextureUsage::PIXEL_FORMAT_VIEW)
        .union(TextureUsage::SHADER_ATOMIC);
    colour && usage.intersects(plain)
}

/// What a new `Private` texture Metal keeps losslessly compressed ([`metal_compresses`]) reads
/// before anything writes it, by pixel format: not zero, but one fixed texel in every texel of each
/// level whose shorter side is 16 texels or more ([`pattern_levels`]), and every sample (an Apple
/// M4 under macOS 27.0.1: every format listed, read by blits, kernels and loads, alike in every
/// read after a written texture or buffer was released; the consumer's M3 Max alike for
/// `RGBA8Unorm`). Each storage field of the texel is all ones but the second from its lowest bit,
/// which is zero (`RGBA8Unorm` reads opaque magenta, `A1BGR5Unorm` opaque yellow), but for
/// `RG11B10Float`'s canonical NaNs and the extended-range formats' own
/// codes; a NaN in a float or depth field. `None` for a format whose new texture reads zero however
/// it is made (block-compressed, 4:2:2 and two-plane formats) or that the probes did not make.
/// An Apple silicon Mac's answer: on a device that presents a Mac keeping no texture compressed,
/// as a Mac with an Intel or AMD GPU is, a new `Private` texture holds what its memory held, and
/// nothing writes this (`Device::metal_compresses_textures`).
pub(crate) fn fresh_pattern(pf: PixelFormat) -> Option<FreshPattern> {
    use FreshPattern::{Color, DepthStencil};
    use PixelFormat as P;
    // A clear color whose conversion writes the texel exactly (`clear_value::texel_bytes`).
    const MAGENTA: Option<[f64; 4]> = Some([1.0, 0.0, 1.0, 1.0]);
    const YELLOW: Option<[f64; 4]> = Some([1.0, 1.0, 0.0, 1.0]);
    const ONES: Option<[f64; 4]> = Some([4294967295.0, 0.0, 4294967295.0, 4294967295.0]);
    const MINUS_ONE: Option<[f64; 4]> = Some([-1.0, 0.0, -1.0, -1.0]);
    const SNORM8: Option<[f64; 4]> = Some([-1.0 / 127.0, 0.0, -1.0 / 127.0, -1.0 / 127.0]);
    const SNORM16: Option<[f64; 4]> = Some([-1.0 / 32767.0, 0.0, -1.0 / 32767.0, -1.0 / 32767.0]);
    const SHARED: Option<[f64; 4]> = Some([65408.0, 0.0, 65408.0, 1.0]);
    let color = |texel: &'static [u8], color: Option<[f64; 4]>| Some(Color { texel, color });
    match pf {
        P::A8Unorm | P::R8Unorm | P::R8UnormSRGB => color(&[0xff], MAGENTA),
        P::R8Snorm => color(&[0xff], SNORM8),
        P::R8Uint => color(&[0xff], ONES),
        P::R8Sint => color(&[0xff], MINUS_ONE),
        P::RG8Unorm | P::RG8UnormSRGB => color(&[0xff, 0x00], MAGENTA),
        P::RG8Snorm => color(&[0xff, 0x00], SNORM8),
        P::RG8Uint => color(&[0xff, 0x00], ONES),
        P::RG8Sint => color(&[0xff, 0x00], MINUS_ONE),
        P::R16Unorm => color(&[0xff, 0xff], MAGENTA),
        P::R16Snorm => color(&[0xff, 0xff], SNORM16),
        P::R16Uint => color(&[0xff, 0xff], ONES),
        P::R16Sint => color(&[0xff, 0xff], MINUS_ONE),
        P::R16Float => color(&[0xff, 0xff], None),
        P::B5G6R5Unorm => color(&[0x1f, 0xf8], MAGENTA),
        P::A1BGR5Unorm => color(&[0xc1, 0xff], YELLOW),
        P::ABGR4Unorm => color(&[0x0f, 0xff], YELLOW),
        P::BGR5A1Unorm => color(&[0x1f, 0xfc], MAGENTA),
        P::RGBA8Unorm | P::RGBA8UnormSRGB | P::BGRA8Unorm | P::BGRA8UnormSRGB => {
            color(&[0xff, 0x00, 0xff, 0xff], MAGENTA)
        }
        P::RGBA8Snorm => color(&[0xff, 0x00, 0xff, 0xff], SNORM8),
        P::RGBA8Uint => color(&[0xff, 0x00, 0xff, 0xff], ONES),
        P::RGBA8Sint => color(&[0xff, 0x00, 0xff, 0xff], MINUS_ONE),
        P::RGB10A2Unorm | P::BGR10A2Unorm => color(&[0xff, 0x03, 0xf0, 0xff], MAGENTA),
        P::RGB10A2Uint => color(&[0xff, 0x03, 0xf0, 0xff], ONES),
        P::BGR10XR | P::BGR10XRSRGB => color(&[0xff, 0x03, 0xf0, 0xff], None),
        P::RG11B10Float => color(&[0xe0, 0x07, 0x00, 0xfc], None),
        P::RGB9E5Float => color(&[0xff, 0x01, 0xfc, 0xff], SHARED),
        P::RG16Unorm => color(&[0xff, 0xff, 0x00, 0x00], MAGENTA),
        P::RG16Snorm => color(&[0xff, 0xff, 0x00, 0x00], SNORM16),
        P::RG16Uint => color(&[0xff, 0xff, 0x00, 0x00], ONES),
        P::RG16Sint => color(&[0xff, 0xff, 0x00, 0x00], MINUS_ONE),
        P::RG16Float => color(&[0xff, 0xff, 0x00, 0x00], None),
        P::R32Uint => color(&[0xff; 4], ONES),
        P::R32Sint => color(&[0xff; 4], MINUS_ONE),
        P::R32Float => color(&[0xff; 4], None),
        P::RGBA16Unorm => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], MAGENTA),
        P::RGBA16Snorm => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], SNORM16),
        P::RGBA16Uint => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], ONES),
        P::RGBA16Sint => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], MINUS_ONE),
        P::RGBA16Float => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], None),
        P::RG32Uint => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], ONES),
        P::RG32Sint => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], MINUS_ONE),
        P::RG32Float => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], None),
        P::BGRA10XR | P::BGRA10XRSRGB => color(&[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf], None),
        P::RGBA32Uint => color(&RGBA32, ONES),
        P::RGBA32Sint => color(&RGBA32, MINUS_ONE),
        P::RGBA32Float => color(&RGBA32, None),
        P::Depth16Unorm => Some(DepthStencil {
            depth: Some(FreshDepth::Value(1.0)),
            stencil: None,
        }),
        P::Depth32Float => Some(DepthStencil {
            depth: Some(FreshDepth::Nan),
            stencil: None,
        }),
        P::Stencil8 => Some(DepthStencil {
            depth: None,
            stencil: Some(0xff),
        }),
        P::Depth32FloatStencil8 => Some(DepthStencil {
            depth: Some(FreshDepth::Nan),
            stencil: Some(0xff),
        }),
        _ => None,
    }
}

/// The texel of [`fresh_pattern`]'s 128-bit formats: all ones but the second 32-bit field.
const RGBA32: [u8; 16] = [
    0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

/// What every texel of a level [`fresh_pattern`] covers reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FreshPattern {
    /// A color format's texel, as Metal's blits read it, and the clear color (Metal's) whose
    /// conversion writes exactly that texel, where one does; `None` where a field is a NaN,
    /// whose bits Vulkan's clears leave to the device (`clear_value.rs`).
    Color {
        texel: &'static [u8],
        color: Option<[f64; 4]>,
    },
    /// A depth or stencil format's depth and stencil, each where the format has the aspect.
    DepthStencil {
        depth: Option<FreshDepth>,
        stencil: Option<u32>,
    },
}

/// The depth [`FreshPattern::DepthStencil`] reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FreshDepth {
    /// A depth a clear writes exactly.
    Value(f64),
    /// A NaN (all ones), which no Vulkan clear or copy writes into a depth image: a clear's depth
    /// lies in `[0, 1]`, and a copy's depth outside it is undefined.
    Nan,
}

/// The levels of a new `Private` texture whose texels read [`fresh_pattern`]: `0..n`, each level
/// whose shorter side (of `[width, height]` at level 0, a 1D texture's height 1; a multisample
/// texture's sample grid, [`sample_grid`]) is 16 texels or more, of a texture Metal compresses
/// ([`metal_compresses`]: a 64 x 64 chain's levels 64, 32 and 16, and none of
/// its levels under 16). 0 where Metal keeps it uncompressed: it reads zero throughout.
pub(crate) fn pattern_levels(
    colour: bool,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    [width, height]: [u64; 2],
    levels: u32,
) -> u32 {
    if !metal_compresses(colour, usage, allow_gpu_optimized_contents, [width, height]) {
        return 0;
    }
    (0..levels)
        .take_while(|&level| {
            let side = |axis: u64| (axis >> level).max(1);
            side(width).min(side(height)) >= COMPRESSED_SIDE
        })
        .count() as u32
}

/// Metal's answer for a texture it keeps losslessly compressed, whose texels uncompressed are
/// laid out as `layout`: its chains, then each chain's metadata over its level 0
/// ([`chain_metadata`], its full chain's levels where it is mipmapped), at the chains'
/// alignment. `None` for a layout of no level.
pub(crate) fn lossless_answer(layout: &TiledLayout) -> Option<SizeAndAlign> {
    let level = layout.levels.first()?;
    let mipmapped = layout.levels.len() > 1;
    Some(with_metadata(
        layout,
        chain_metadata(level.extent, mipmapped),
    ))
}

/// Metal's answer for a `Depth32Float_Stencil8` texture whose depth plane answers `depth`
/// (`Depth32Float`'s answer for the descriptor) and stencil plane `stencil` (`Stencil8`'s): the
/// depth plane rounded up to 16 KiB, then the stencil plane, at the depth plane's alignment
/// (the module docs).
fn depth_stencil_planes(depth: SizeAndAlign, stencil: SizeAndAlign) -> SizeAndAlign {
    SizeAndAlign {
        size: depth.size.next_multiple_of(PAGE) + stencil.size,
        align: depth.align,
    }
}

/// A single-sample `Depth32Float_Stencil8` texture of `layout`'s two planes in Metal's tiles
/// (the module docs), its image of `format`: the depth plane's elements laid out as a
/// `Depth32Float` texture of its shape is, from the texture's first byte, and the stencil plane's
/// as a `Stencil8` one's, from the depth plane's answer (kept compressed where `compressed`, its
/// metadata counted) rounded up to 16 KiB. `None` where the image holds other elements than
/// those planes' or a plane has no layout.
fn depth_stencil_tiles(
    layout: &TextureLayout,
    format: Format,
    compressed: bool,
) -> Option<TiledTexture> {
    let holds = matches!(
        copy_units(format, layout.pixel_format)[..],
        [
            (ImageAspectFlags::DEPTH, 4, [1, 1]),
            (ImageAspectFlags::STENCIL, 1, [1, 1])
        ]
    );
    if !holds || layout.samples > 1 {
        return None;
    }
    let extent = layout.extent.map(u64::from);
    let (ty, levels, slices) = (layout.texture_type, layout.levels, u64::from(layout.layers));
    let plane = |unit| TiledLayout::of(unit, Elements::Texels, ty, extent, levels, slices);
    let (depth, stencil) = (plane(4)?, plane(1)?);
    let depth_bytes = match compressed {
        true => lossless_answer(&depth)?.size,
        false => depth.size,
    };
    let at = depth_bytes.next_multiple_of(PAGE);
    Some(TiledTexture {
        size: at + stencil.size,
        parts: vec![
            TiledPart {
                aspect: Some(ImageAspectFlags::DEPTH),
                at: 0,
                layout: depth,
            },
            TiledPart {
                aspect: Some(ImageAspectFlags::STENCIL),
                at,
                layout: stencil,
            },
        ],
    })
}

/// The answer of a texture laid out uncompressed as `layout`, kept compressed with `metadata`
/// bytes for each chain.
fn with_metadata(layout: &TiledLayout, metadata: u64) -> SizeAndAlign {
    SizeAndAlign {
        size: layout.size + layout.chains * metadata,
        align: layout.align,
    }
}

/// The compression metadata of a chain whose level 0 is `[width, height]` texels (a multisample
/// texture's sample grid, [`sample_grid`]), its full chain's levels where it is `mipmapped` (the
/// module docs): for each level, 8 bytes for each 16 x 16 tile of it, its tiles on each side
/// padded to a power of two, rounded up to 128; level 0's alone, or each level of the full
/// chain while level 0's sides, each padded to a multiple of 16 and halved per level, have a
/// longer side of 16 or more, its tiles counted over those halved sides.
fn chain_metadata([width, height]: [u64; 2], mipmapped: bool) -> u64 {
    let padded = [width, height].map(|side| side.max(1).next_multiple_of(METADATA_TILE));
    let longer = padded[0].max(padded[1]);
    (0..u64::BITS)
        .take_while(|&level| level == 0 || (mipmapped && longer >> level >= METADATA_TILE))
        .map(|level| {
            let [tw, th] = padded.map(|side| pot((side >> level).max(1).div_ceil(METADATA_TILE)));
            (tw * th * METADATA_BYTES).next_multiple_of(LINE)
        })
        .sum()
}

/// The sample grid of a multisample texture of `samples` samples whose level is `[width,
/// height]` pixels, as Metal's compression counts it (the module docs): `width x 2·height` at 2
/// samples, `2·width x 2·height` at 4; `None` at any other count, which Apple's GPUs do not make
/// (an M4 and an M3 Max answer no to 8), its layout not measured.
pub(crate) fn sample_grid(samples: u32, [width, height]: [u64; 2]) -> Option<[u64; 2]> {
    match samples {
        2 => Some([width, 2 * height]),
        4 => Some([2 * width, 2 * height]),
        _ => None,
    }
}

/// Whether a texture of `texture_type` lays its chains out at multiples of 16 KiB from a chain
/// of 16 KiB on.
fn layered(texture_type: TextureType) -> bool {
    use TextureType as T;
    matches!(
        texture_type,
        T::Type1DArray
            | T::Type2DArray
            | T::Cube
            | T::CubeArray
            | T::Type3D
            | T::Type2DMultisampleArray
    )
}

/// Why a heap texture over host pages is not laid out as Metal lays it, and the kind its
/// per-texture notice names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Untiled {
    pub(crate) kind: Degradation,
    pub(crate) reason: &'static str,
}

impl Untiled {
    /// A texture Metal may keep losslessly compressed (`heap-texture-lossless-layout`).
    pub(crate) const LOSSLESS: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal may keep it losslessly compressed (allowGPUOptimizedContents, 256 texels \
                 or more, and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a \
                 depth or stencil format), whose bytes are no layout of its texels",
    };
    /// A multisample texture Metal keeps losslessly compressed (`heap-texture-lossless-layout`),
    /// held by a shadow of its samples in the tiles Metal lays it in uncompressed (`heap.rs`).
    pub(crate) const LOSSLESS_SAMPLES: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal keeps the multisample texture losslessly compressed \
                 (allowGPUOptimizedContents, a sample grid of 16 or more on its shorter side, \
                 and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a depth or \
                 stencil format), whose bytes in the heap are no layout of its samples: the \
                 heap's bytes hold its samples as Metal lays them out uncompressed",
    };
    /// A multisample texture Metal keeps losslessly compressed (`heap-texture-lossless-layout`),
    /// held in memory of its own where no shadow of its samples serves (`heap.rs`).
    pub(crate) const LOSSLESS_MULTISAMPLE: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal keeps the multisample texture losslessly compressed \
                 (allowGPUOptimizedContents, a sample grid of 16 or more on its shorter side, \
                 and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a depth or \
                 stencil format), whose bytes in the heap are no layout of its samples: its \
                 image lies in memory of its own, and the heap's bytes keep what they held",
    };
    /// A multisample texture Metal keeps losslessly compressed (`heap-texture-lossless-layout`),
    /// whose image lies in the bytes of a heap over the device's memory (`heap.rs`,
    /// `Device::device_heap_layout`).
    pub(crate) const LOSSLESS_BOUND_SAMPLES: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal keeps the multisample texture losslessly compressed \
                 (allowGPUOptimizedContents, a sample grid of 16 or more on its shorter side, \
                 and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a depth or \
                 stencil format), whose bytes in the heap are no layout of its samples: its \
                 image lies in the heap's bytes, of the runtime's usage, and need not be \
                 compressed as Metal's",
    };
    /// A texture of a heap over the device's memory whose image lies in the heap's bytes in the
    /// driver's layout (`heap-texture-untiled-layout`, `heap.rs`).
    pub(crate) const DRIVER: Untiled = Untiled::other(
        "its image lies in the heap's bytes in the device driver's layout, which no driver fact \
         measures as Metal's tiles (DriverFact::ColorImagesLieAsMetalHeapTextures)",
    );
    /// A texture of a heap over the device's memory whose image lies in the heap's bytes as a
    /// linear image, its texels in the driver's rows (`heap-texture-untiled-layout`, `heap.rs`).
    pub(crate) const LINEAR: Untiled = Untiled::other(
        "its image is linear, its texels in rows in the heap's bytes, where Metal tiles them",
    );
    /// A texture of a heap over the device's memory whose image is a shadow of the heap's bytes
    /// in rows, where Metal tiles its elements (`heap-texture-untiled-layout`, `heap.rs`).
    pub(crate) const ROWS: Untiled = Untiled::other(
        "its image is a shadow of the heap's bytes in rows, where Metal tiles its elements",
    );
    /// A multisample texture whose image lies in the heap's bytes, where Metal lays its samples
    /// out by a layout no image of the runtime's is measured to hold
    /// (`heap-texture-untiled-layout`, `heap.rs`).
    pub(crate) const MULTISAMPLE: Untiled = Untiled::other(
        "its image lies in the heap's bytes in the device driver's layout, where Metal lays the \
         multisample texture's samples out in its tiles, which no driver fact measures as Metal's \
         for it (DriverFact::MultisampleColorImagesLieAsMetalHeapTextures, of a colour one)",
    );
    /// A multisample texture Metal keeps uncompressed, whose samples it lays out in the heap's
    /// bytes, held in memory of its own apart from them (`AliasRoute::Lossless`, `heap.rs`): its
    /// image holds other bytes than Metal's samples, the device cannot move them to and from the
    /// heap's bytes, or their layout is not measured (`heap-texture-untiled-layout`).
    pub(crate) const SAMPLES_APART: Untiled = Untiled::other(
        "Metal lays the multisample texture's samples out in the heap's bytes, which its image, \
         in memory of its own, does not hold: it holds other bytes than Metal's samples, the \
         device cannot move them to and from the heap's bytes, or their layout is not measured",
    );
    /// A block-compressed texture whose blocks' placement is not measured
    /// (`heap-texture-block-compressed-layout`).
    const BLOCK_COMPRESSED: Untiled = Untiled {
        kind: Degradation::HeapTextureBlockCompressedLayout,
        reason: "its format is block-compressed, whose blocks Metal places in each level by a \
                 rule that differs by format family and by machine, not measured for its \
                 family",
    };

    /// A texture the runtime holds otherwise than tiled for `reason`
    /// (`heap-texture-untiled-layout`).
    pub(crate) const fn other(reason: &'static str) -> Untiled {
        Untiled {
            kind: Degradation::HeapTextureUntiledLayout,
            reason,
        }
    }
}

/// The reason of a texture whose format's layout in Metal's heaps is not measured.
pub(crate) const UNMEASURED_FORMAT: &str = "its format's layout in Metal's heaps is not measured";

/// A texture's elements in Metal's tiles: one layout of all of them ([`TiledTexture::one`]), or a
/// `Depth32Float_Stencil8` texture's two planes, each one aspect's elements laid out as a texture
/// of that aspect's format and the texture's shape is (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TiledTexture {
    /// Its parts, in the order of their bytes.
    pub(crate) parts: Vec<TiledPart>,
    /// The bytes from the texture's first to past its last part's last.
    pub(crate) size: u64,
}

/// One part of a [`TiledTexture`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TiledPart {
    /// The aspect of the image whose elements it lays out; `None` for each of the image's.
    pub(crate) aspect: Option<ImageAspectFlags>,
    /// Its first byte, from the texture's.
    pub(crate) at: u64,
    /// Its elements' layout, from its first byte.
    pub(crate) layout: TiledLayout,
}

impl TiledTexture {
    /// A texture whose elements lie as `layout` says, from its first byte.
    pub(crate) fn one(layout: TiledLayout) -> TiledTexture {
        TiledTexture {
            size: layout.size,
            parts: vec![TiledPart {
                aspect: None,
                at: 0,
                layout,
            }],
        }
    }

    /// The part that lays out `aspect`'s elements.
    pub(crate) fn part(&self, aspect: ImageAspectFlags) -> Option<&TiledPart> {
        (self.parts.iter()).find(|part| part.aspect.is_none_or(|own| own == aspect))
    }
}

/// What the runtime holds of Metal's layout of a heap texture's bytes (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HeapLayout {
    /// Its elements as Metal lays them: over host pages a shadow of them
    /// (`HeapTiledCopies`).
    Tiled(TiledTexture),
    /// Not laid out as Metal's: over host pages, announced once per texture.
    Untiled(Untiled),
    /// Laid out as a measured route of its own lays it (a 4:2:2 texture's rows, a two-plane
    /// texture's planes; `heap.rs`).
    Measured,
    /// A texture Metal keeps losslessly compressed, whose bytes are no layout of its texels
    /// ([`Untiled::LOSSLESS`]), held by a shadow of its elements in the tiles Metal lays it in
    /// uncompressed (`heap.rs`): announced once per texture as `heap-texture-lossless-layout`.
    Uncompressed(TiledTexture),
}

impl HeapLayout {
    /// The tiles a shadow of the texture's bytes lays its elements in: Metal's
    /// ([`HeapLayout::Tiled`]), or its uncompressed ones ([`HeapLayout::Uncompressed`]).
    pub(crate) fn tiles(&self) -> Option<&TiledTexture> {
        match self {
            HeapLayout::Tiled(tiled) | HeapLayout::Uncompressed(tiled) => Some(tiled),
            HeapLayout::Untiled(_) | HeapLayout::Measured => None,
        }
    }
}

/// The per-texture notice of a heap texture whose bytes the runtime holds Metal's layout of
/// itself (`heap.rs`): one in a heap over host pages, one whose image is a shadow of Metal's tiles
/// in the heap's bytes, and a multisample one whose image lies in memory of its own apart from
/// them (`apart`, `AliasRoute::Lossless`), where Metal lays it out as `metal` and it has `samples`
/// samples. `None` where its bytes are Metal's layout of its elements: a shadow of Metal's own
/// tiles, and a texture laid out by a measured route of its own ([`HeapLayout::Measured`]).
pub(crate) fn own_heap_layout(metal: &HeapLayout, samples: u32, apart: bool) -> Option<Untiled> {
    match metal {
        // Its image holds the uncompressed tiles of a texture Metal keeps losslessly compressed.
        HeapLayout::Uncompressed(_) => Some(match samples {
            1 => Untiled::LOSSLESS,
            _ => Untiled::LOSSLESS_SAMPLES,
        }),
        HeapLayout::Untiled(untiled) => Some(*untiled),
        // Metal lays its samples out in the heap's bytes, which its image, in memory of its own,
        // does not hold: the heap's bytes keep what they held.
        HeapLayout::Tiled(_) if apart => Some(Untiled::SAMPLES_APART),
        HeapLayout::Tiled(_) | HeapLayout::Measured => None,
    }
}

/// The layout a shadow of the heap's bytes holds a texture of `layout` (its image of `format`) in,
/// where Metal lays it out as `metal`: Metal's tiles where it tiles
/// the texture ([`HeapLayout::Tiled`]); where it keeps the texture losslessly compressed
/// ([`Untiled::LOSSLESS`], [`Untiled::LOSSLESS_MULTISAMPLE`]), the tiles it lays the texture in
/// uncompressed ([`HeapLayout::Uncompressed`]); `None` for any other (a layout not measured, or
/// an image that holds other bytes than Metal's elements).
pub(crate) fn shadow_tiles(
    layout: &TextureLayout,
    format: Format,
    metal: &HeapLayout,
) -> Option<HeapLayout> {
    match metal {
        HeapLayout::Tiled(_) => Some(metal.clone()),
        HeapLayout::Untiled(untiled) if *untiled == Untiled::LOSSLESS_MULTISAMPLE => {
            // A two-aspect texture's stencil plane past its compressed depth plane's answer.
            let [width, height, _] = layout.extent.map(u64::from);
            let grid = sample_grid(layout.samples, [width, height])?;
            let metadata = chain_metadata(grid, false);
            multisample_tiles(layout, format, Some(metadata)).map(HeapLayout::Uncompressed)
        }
        HeapLayout::Untiled(untiled) if *untiled == Untiled::LOSSLESS => {
            // A two-aspect texture's depth plane kept compressed, its stencil plane after it.
            if layout.pixel_format == PixelFormat::Depth32FloatStencil8 {
                return depth_stencil_tiles(layout, format, true).map(HeapLayout::Uncompressed);
            }
            // With `allowGPUOptimizedContents` off Metal keeps every texture uncompressed.
            let plain = metal_heap_layout(layout, TextureUsage::SHADER_READ, false, format);
            match plain.layout {
                HeapLayout::Tiled(tiled) => Some(HeapLayout::Uncompressed(tiled)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Metal's answer for a heap texture where the runtime knows it, and the runtime's layout of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalHeapLayout {
    pub(crate) answer: Option<SizeAndAlign>,
    pub(crate) layout: HeapLayout,
}

/// Metal's element bytes and texels (across and down) of a format `pf` whose image of `format`
/// holds Metal's bytes of each element in one copy unit, as its CPU copies and blits move them
/// (a colour image of its byte format or its sRGB format's raw twin's, a texel or a
/// block-compressed format's block a unit; a depth image of Metal's depth bytes and no stencil;
/// a stencil aspect's byte); `None` for any other.
/// Whether an image of `format` holding a `pf` texture holds each texel as Metal's `bytes`
/// bytes ([`image_element`]): a copy of Metal's texel's bytes writes that texel.
pub(crate) fn holds_metal_texels(pf: PixelFormat, format: Format, bytes: usize) -> bool {
    image_element(pf, format) == Some((bytes as u64, [1, 1]))
}

fn image_element(pf: PixelFormat, format: Format) -> Option<(u64, [u64; 2])> {
    let units = copy_units(format, pf);
    let element = match pf {
        // A stencil format's image holds its stencil, beside a depth it does not present where
        // the image is combined (`X32Stencil8`'s; `texel_units`, `heap_plane_layout`).
        pf if aspects(pf) == Some(ImageAspectFlags::STENCIL) => match texel_units(format, pf)[..] {
            [(ImageAspectFlags::STENCIL, bytes, block)] => Some((bytes, block)),
            _ => None,
        },
        PixelFormat::Depth16Unorm | PixelFormat::Depth32Float => match units[..] {
            [(ImageAspectFlags::DEPTH, bytes, block)] => {
                (depth_bytes(format) == Some(bytes)).then_some((bytes, block))
            }
            _ => None,
        },
        _ => {
            let bytes_of = |pf: PixelFormat| byte_format(pf).filter(|&own| own == format);
            let same = bytes_of(pf).or_else(|| srgb_raw_twin(pf).and_then(bytes_of));
            match units[..] {
                [(ImageAspectFlags::COLOR, bytes, block)]
                    if same.is_some() && block_bytes(pf) == Some(bytes) =>
                {
                    Some((bytes, block))
                }
                _ => None,
            }
        }
    };
    element.map(|(bytes, block)| (u64::from(bytes), block.map(u64::from)))
}

/// Metal's element bytes of the plain format `pf`, where its layout is measured (the module
/// docs): a colour format's texel, `Depth16Unorm`'s 2, `Depth32Float`'s 4 and `Stencil8`'s 1;
/// `None` for a two-aspect format and any other.
fn metal_element(pf: PixelFormat) -> Option<u64> {
    match pf {
        PixelFormat::Depth16Unorm => Some(2),
        PixelFormat::Depth32Float => Some(4),
        PixelFormat::Stencil8 | PixelFormat::X32Stencil8 => Some(1),
        pf if is_depth_stencil(pf) => None,
        pf => block_bytes(pf).map(u64::from),
    }
    .filter(|&unit| largest_tile(unit).is_some())
}

/// Metal's answer for a heap texture of `layout` whose elements are `unit` bytes each covering
/// `block` texels, where no image of this crate holds those elements
/// (`metal::format::footprint`): the tiles of the module's model, and a plain format's
/// lossless-compression metadata where `usage` and `allow_gpu_optimized_contents` say Metal keeps
/// the texture compressed, as for a format the crate does hold. A block-compressed format's
/// footprint is its blocks' tiles, which Metal never compresses.
///
/// This counts bytes and makes nothing: the texture itself stays refused
/// `NotImplemented(Format)` (`texture/heap.rs`), because the crate holds no image of the format.
/// A block's placement is `Halved`, which every ASTC footprint measured takes and under which
/// each level's allocation is the same either way (the module docs).
pub(crate) fn footprint_size(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    unit: u64,
    block: [u64; 2],
) -> Option<SizeAndAlign> {
    let compressed_block = block != [1, 1];
    let elements = match compressed_block {
        true => Elements::Blocks {
            block,
            placement: BlockPlacement::Halved,
        },
        false => Elements::Texels,
    };
    let tiled = TiledLayout::of(
        unit,
        elements,
        layout.texture_type,
        layout.extent.map(u64::from),
        layout.levels,
        u64::from(layout.layers),
    )?;
    // Metal keeps no block-compressed texture losslessly compressed, and a plain one by the rule
    // every plain format shares.
    if !compressed_block && metal_compresses_layout(layout, usage, allow_gpu_optimized_contents) {
        return lossless_answer(&tiled);
    }
    Some(SizeAndAlign {
        size: tiled.size,
        align: tiled.align,
    })
}

/// Metal's answer for a heap texture of `layout` (the module docs), `usage` and
/// `allow_gpu_optimized_contents` the descriptor's: [`metal_heap_layout`]'s answer,
/// which no image format decides (the format decides only how a heap over host pages holds the
/// texture, so none is given). `None` where the model has none.
pub(crate) fn modelled_size(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
) -> Option<SizeAndAlign> {
    let no_image = Format::UNDEFINED;
    metal_heap_layout(layout, usage, allow_gpu_optimized_contents, no_image).answer
}

/// Metal's answer for a heap texture of `layout` and the runtime's layout of it over host pages
/// (the module docs), its image of `format`: `usage` and `allow_gpu_optimized_contents` the
/// descriptor's, which decide whether Metal may compress it.
pub(crate) fn metal_heap_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    format: Format,
) -> MetalHeapLayout {
    let pf = layout.pixel_format;
    let [width, height, depth] = layout.extent.map(u64::from);
    let slices = u64::from(layout.layers);
    let untiled = |answer, why: Untiled| MetalHeapLayout {
        answer,
        layout: HeapLayout::Untiled(why),
    };
    // Block-compressed formats: Metal's allocations are its blocks' (every BC, ETC2, EAC, ASTC
    // and PVRTC format), placed as the release places its family's
    // (`metal_release::heap_blocks`); a format of no measured family is not modelled.
    if let Some([bw, bh]) = block_extent(pf) {
        let block = [u64::from(bw), u64::from(bh)];
        let placement = BlockFamily::of(pf).map(heap_blocks);
        let unit = block_bytes(pf).map(u64::from);
        let tiled = placement.zip(unit).and_then(|(placement, unit)| {
            let elements = Elements::Blocks { block, placement };
            let ty = layout.texture_type;
            let extent = [width, height, depth];
            TiledLayout::of(unit, elements, ty, extent, layout.levels, slices)
        });
        let answer = tiled.as_ref().map(|tiled| SizeAndAlign {
            size: tiled.size,
            align: tiled.align,
        });
        let (Some(tiled), Some(unit)) = (tiled, unit) else {
            return untiled(answer, Untiled::BLOCK_COMPRESSED);
        };
        if image_element(pf, format) != Some((unit, block)) {
            return untiled(
                answer,
                Untiled::other("its image holds other bytes than Metal's blocks"),
            );
        }
        return MetalHeapLayout {
            answer,
            layout: HeapLayout::Tiled(TiledTexture::one(tiled)),
        };
    }
    if layout.samples > 1 {
        return multisample_heap_layout(layout, usage, allow_gpu_optimized_contents, format);
    }
    let texture_type = layout.texture_type;
    let one_row = matches!(texture_type, TextureType::Type1D | TextureType::Type1DArray);
    let level0 = [width, if one_row { 1 } else { height }];
    let colour = !is_depth_stencil(pf);
    let compressed = metal_compresses(colour, usage, allow_gpu_optimized_contents, level0);
    // `Depth32Float_Stencil8`: its two planes, `Depth32Float`'s and `Stencil8`'s answers for the
    // descriptor, the depth plane rounded up to 16 KiB, each laid out as a texture of its format
    // (`depth_stencil_tiles`); Metal may keep the depth plane compressed.
    if pf == PixelFormat::Depth32FloatStencil8 {
        let plane = |pixel_format| {
            let one = TextureLayout {
                pixel_format,
                ..*layout
            };
            metal_heap_layout(&one, usage, allow_gpu_optimized_contents, format).answer
        };
        let answer = plane(PixelFormat::Depth32Float)
            .zip(plane(PixelFormat::Stencil8))
            .map(|(depth, stencil)| depth_stencil_planes(depth, stencil));
        if compressed {
            return untiled(answer, Untiled::LOSSLESS);
        }
        return match depth_stencil_tiles(layout, format, false) {
            Some(tiles) => MetalHeapLayout {
                answer,
                layout: HeapLayout::Tiled(tiles),
            },
            None => untiled(
                answer,
                Untiled::other("its image holds other bytes than Metal's elements"),
            ),
        };
    }
    let Some(unit) = metal_element(pf) else {
        return untiled(None, Untiled::other(UNMEASURED_FORMAT));
    };
    let extent = [width, height, depth];
    let tiled = TiledLayout::of(
        unit,
        Elements::Texels,
        texture_type,
        extent,
        layout.levels,
        slices,
    );
    if compressed {
        return untiled(tiled.as_ref().and_then(lossless_answer), Untiled::LOSSLESS);
    }
    let Some(tiled) = tiled else {
        return untiled(None, Untiled::other(UNMEASURED_FORMAT));
    };
    let plain = Some(SizeAndAlign {
        size: tiled.size,
        align: tiled.align,
    });
    if image_element(pf, format) != Some((unit, [1, 1])) {
        return untiled(
            plain,
            Untiled::other("its image holds other bytes than Metal's elements"),
        );
    }
    let mut tiles = TiledTexture::one(tiled);
    tiles.parts[0].aspect = held_aspect(format, pf);
    MetalHeapLayout {
        answer: plain,
        layout: HeapLayout::Tiled(tiles),
    }
}

/// Metal's answer for a multisample heap texture of `layout` (the module docs), and the runtime's
/// layout of it, its image of `format`: `usage` and `allow_gpu_optimized_contents` the
/// descriptor's. Uncompressed, its pixels lie as a one-level texture's elements of all their
/// samples (`S·B` bytes, sample `s` at `s·B`), a multisample array's slices as an array's; compressed,
/// that answer and metadata over its sample grid for each slice. `Depth32Float_Stencil8` answers
/// its two planes: the depth plane (`Depth32Float`'s answer) rounded up to 16 KiB, then the stencil
/// plane (`Stencil8`'s), at the depth plane's alignment. A sample count Apple's GPUs do not make
/// is not measured. Uncompressed, its samples in Metal's tiles ([`multisample_tiles`]) where its
/// image holds Metal's elements; compressed, [`Untiled::LOSSLESS_MULTISAMPLE`], whose uncompressed
/// tiles a shadow may hold ([`shadow_tiles`]); any other, its samples apart from the heap's bytes.
fn multisample_heap_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    format: Format,
) -> MetalHeapLayout {
    let pf = layout.pixel_format;
    let [width, height, _] = layout.extent.map(u64::from);
    let samples = layout.samples;
    let apart = |answer| MetalHeapLayout {
        answer,
        layout: HeapLayout::Untiled(Untiled::SAMPLES_APART),
    };
    let Some(grid) = sample_grid(samples, [width, height]) else {
        return apart(None);
    };
    let colour = !is_depth_stencil(pf);
    let compressed = metal_compresses(colour, usage, allow_gpu_optimized_contents, grid);
    // One plane of elements of `bytes` bytes a sample: its layout's answer, and its metadata
    // where Metal compresses it.
    let plane = |bytes: u64| {
        let unit = u64::from(samples) * bytes;
        let ty = layout.texture_type;
        let slices = u64::from(layout.layers);
        let tiled = TiledLayout::of(unit, Elements::Texels, ty, [width, height, 1], 1, slices)?;
        Some(match compressed {
            true => with_metadata(&tiled, chain_metadata(grid, false)),
            false => SizeAndAlign {
                size: tiled.size,
                align: tiled.align,
            },
        })
    };
    // `Depth32Float_Stencil8`: its depth plane (`Depth32Float`'s answer) rounded up to 16 KiB,
    // then its stencil plane (`Stencil8`'s), at the depth plane's alignment.
    let answer = match pf {
        PixelFormat::Depth32FloatStencil8 => plane(4)
            .zip(plane(1))
            .map(|(depth, stencil)| depth_stencil_planes(depth, stencil)),
        _ => metal_element(pf).and_then(plane),
    };
    if compressed {
        return MetalHeapLayout {
            answer,
            layout: HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE),
        };
    }
    match multisample_tiles(layout, format, None) {
        Some(tiles) => MetalHeapLayout {
            answer,
            layout: HeapLayout::Tiled(tiles),
        },
        None => apart(answer),
    }
}

/// A multisample texture of `layout`'s samples in Metal's tiles, laid out uncompressed (the
/// module docs), its image of `format`: each pixel's samples one element of `S·B` bytes (sample
/// `s` at `s·B`), laid out as a one-level texture's elements of that size, a multisample array's
/// slices as an array's; a `Depth32Float_Stencil8` texture's depth plane so from its first byte
/// and its stencil plane from the depth plane's answer (its `metadata` bytes for each slice
/// counted where Metal keeps it compressed) rounded up to 16 KiB, as a `Stencil8` texture's
/// samples lie (an Apple M3 Max under macOS 27.0.1: every sample of `Stencil8` and
/// `Depth32Float_Stencil8` textures of 8 x 8 to 200 x 130 at 2 and 4 samples drawn at the bytes the
/// rule names). `None` where the image holds other elements than Metal's or a plane has no
/// layout (a sample count Apple's GPUs do not make).
fn multisample_tiles(
    layout: &TextureLayout,
    format: Format,
    metadata: Option<u64>,
) -> Option<TiledTexture> {
    let pf = layout.pixel_format;
    let samples = u64::from(layout.samples);
    let [width, height, _] = layout.extent.map(u64::from);
    let (ty, slices) = (layout.texture_type, u64::from(layout.layers));
    sample_grid(layout.samples, [width, height])?;
    let plane = |bytes: u64| {
        TiledLayout::of(
            samples * bytes,
            Elements::Texels,
            ty,
            [width, height, 1],
            1,
            slices,
        )
    };
    if pf == PixelFormat::Depth32FloatStencil8 {
        let holds = matches!(
            copy_units(format, pf)[..],
            [
                (ImageAspectFlags::DEPTH, 4, [1, 1]),
                (ImageAspectFlags::STENCIL, 1, [1, 1])
            ]
        );
        if !holds {
            return None;
        }
        let (depth, stencil) = (plane(4)?, plane(1)?);
        let depth_bytes = match metadata {
            Some(metadata) => with_metadata(&depth, metadata).size,
            None => depth.size,
        };
        let at = depth_bytes.next_multiple_of(PAGE);
        return Some(TiledTexture {
            size: at + stencil.size,
            parts: vec![
                TiledPart {
                    aspect: Some(ImageAspectFlags::DEPTH),
                    at: 0,
                    layout: depth,
                },
                TiledPart {
                    aspect: Some(ImageAspectFlags::STENCIL),
                    at,
                    layout: stencil,
                },
            ],
        });
    }
    let unit = metal_element(pf)?;
    if image_element(pf, format) != Some((unit, [1, 1])) {
        return None;
    }
    let mut tiles = TiledTexture::one(plane(unit)?);
    tiles.parts[0].aspect = held_aspect(format, pf);
    Some(tiles)
}

/// The one aspect of an image of `format` whose elements a texture of `pf` lays out where the image
/// holds another it presents none of ([`texel_units`]): a stencil format's over a combined image
/// (`X32Stencil8`'s), its tiles its stencil's, the image's depth holding none of its texels; `None`
/// where the texture's tiles are each of the image's aspects.
fn held_aspect(format: Format, pf: PixelFormat) -> Option<ImageAspectFlags> {
    match texel_units(format, pf)[..] {
        [(aspect, _, _)] if copy_units(format, pf).len() > 1 => Some(aspect),
        _ => None,
    }
}

/// One direction of the kernel's moves: its shape, where it lies in the shape buffer, and the
/// dispatch's workgroups.
struct Move {
    shape: Vec<u8>,
    at: u64,
    groups: [u32; 2],
}

/// A heap texture's moves between its image and its elements tiled in the heap's bytes (the
/// module docs): the tight buffer of the image's elements, the shapes, and the kernel's two sets
/// (into the image, then back).
pub(crate) struct HeapTiledCopies {
    /// Metal's layout of the texture's elements.
    pub(crate) tiles: TiledTexture,
    /// The image's elements, each level's from its plane's offset; held for the texture's life.
    /// No command's uses name it: every copy writes the image or the heap's bytes, so the
    /// tracker orders it after every other copy of the texture, and each records a full barrier
    /// first.
    _tight: Buffer,
    tight_raw: vk::Buffer,
    /// The tight buffer's copies to and from the image, one a level: a single-sample image's
    /// copies; a multisample image's region, which its [`SampleMoves`] move.
    regions: Vec<vk::BufferImageCopy>,
    /// A multisample image's texture, known once the texture is made
    /// ([`know_samples`](Self::know_samples)); `None` for a single-sample image.
    samples: Option<OnceLock<MovedTexture>>,
    /// The tight buffer's planes and where it lies in its `VkBuffer`, which a multisample image's
    /// moves are planned over.
    planes: Vec<HeapPlane>,
    tight_at: u64,
    /// The two shapes; held for the texture's life.
    _shapes: Buffer,
    shapes_raw: vk::Buffer,
    moves: [Move; 2],
    kernel: OwnedKernel,
    sets: OwnedSets,
}

/// A multisample heap texture's moves between its image and the tight buffer, one way, which
/// Vulkan copies no multisampled image to or from: the interleave copies
/// (`sample_interleave.rs`), each pixel's samples one after another in the tight buffer as one
/// element of `S·B` bytes, as Metal's blits lay them out in a buffer and as its tiles hold them
/// (the module docs). One a slice of each aspect, planned for one copy as an encoder records it
/// ([`HeapTiledCopies::sample_moves`]), the images it goes through taken from its queue's
/// (`copy_images.rs`); the encoder's command buffer holds it until it completes, and the
/// texture's commands hold the texture.
pub(crate) struct SampleMoves {
    copies: Vec<InterleaveCopy>,
    into: bool,
}

/// A multisample heap texture whose moves are planned as its copies are recorded
/// ([`HeapTiledCopies::sample_moves`]): the texture, which holds the moves' tiles and so is held
/// weakly.
struct MovedTexture {
    texture: Weak<TextureShared>,
}

impl HeapTiledCopies {
    /// The tight buffer's copies to and from the image, one a level, each in the image's
    /// aspects, level and slices.
    pub(crate) fn regions(&self) -> &[vk::BufferImageCopy] {
        &self.regions
    }

    /// The kernel's levels of a texture of `tiles` whose image's elements lie in the tight
    /// buffer as `planes` say (`heap_plane_layout`, one plane a level of each aspect), each
    /// part's levels its aspect's planes', a depth plane's of 4-byte elements clamped into the
    /// image where `clamps` (`depth_clamp::clamps_depth`); `None` where a byte the moves name lies
    /// past `u32`, or the planes are not the levels'.
    fn levels(
        tiles: &TiledTexture,
        planes: &[HeapPlane],
        clamps: bool,
    ) -> Option<Vec<TiledLevelMove>> {
        let narrow = |v: u64| u32::try_from(v).ok();
        let mut moves = Vec::new();
        for part in &tiles.parts {
            let layout = &part.layout;
            let own: Vec<&HeapPlane> = (planes.iter())
                .filter(|plane| part.aspect.is_none_or(|aspect| aspect == plane.aspect))
                .collect();
            if own.len() != layout.levels.len() {
                return None;
            }
            for (level, plane) in layout.levels.iter().zip(own) {
                if plane.blocks != level.extent || plane.unit != layout.unit {
                    return None;
                }
                let moved = TiledLevelMove::new(
                    narrow(layout.unit)?,
                    [narrow(level.extent[0])?, narrow(level.extent[1])?],
                    narrow(plane.images)?,
                    [narrow(level.tile[0])?, narrow(level.tile[1])?],
                    narrow(level.across)?,
                    narrow(part.at + level.base)?,
                    narrow(layout.stride)?,
                    narrow(plane.offset)?,
                    narrow(level.bytes)?,
                );
                let depth = plane.aspect == ImageAspectFlags::DEPTH && layout.unit == 4;
                moves.push(match clamps && depth {
                    true => moved.clamping_depth(),
                    false => moved,
                });
            }
        }
        // Every plane is a part's.
        (moves.len() == planes.len()).then_some(moves)
    }

    /// The workgroups of a dispatch of `invocations` within `device`'s counts, or `None`.
    fn grid(device: &Device, invocations: u64) -> Option<[u32; 2]> {
        let limits = &device.properties().description.limits;
        let local = u64::from(TiledTexelsModule::LOCAL_SIZE);
        let groups = invocations.div_ceil(local).max(1);
        let across = groups.min(u64::from(limits.max_compute_work_group_count[0]));
        let down = groups.div_ceil(across.max(1));
        let fits = down <= u64::from(limits.max_compute_work_group_count[1]);
        fits.then_some([across as u32, down as u32])
    }

    /// Whether the moves of a texture of `tiles`, its tight buffer laid out as `planes` and
    /// `tight_len` bytes long, fit `device`: every byte they name within `u32`, at most
    /// `TiledTexelsModule::MAX_LEVELS` levels of all its parts, each buffer within
    /// `maxStorageBufferRange`, and each dispatch within `maxComputeWorkGroupCount`.
    pub(crate) fn fit(
        device: &Device,
        tiles: &TiledTexture,
        planes: &[HeapPlane],
        tight_len: u64,
    ) -> bool {
        let range = u64::from(
            device
                .properties()
                .description
                .limits
                .max_storage_buffer_range,
        );
        let Some(levels) = Self::levels(tiles, planes, false) else {
            return false;
        };
        let moves = [true, false].iter().all(|&into| {
            let invocations = levels
                .iter()
                .map(|level| u64::from(level.invocations(into)))
                .sum::<u64>();
            u32::try_from(invocations).is_ok() && Self::grid(device, invocations).is_some()
        });
        levels.len() <= TiledTexelsModule::MAX_LEVELS as usize
            && tiles.size <= range
            && u32::try_from(tiles.size).is_ok()
            && tight_len.next_multiple_of(4) <= range
            && u32::try_from(tight_len).is_ok()
            && moves
    }

    /// The staging buffer of a texture of `tiles` whose first byte lies at `at` of the heap's
    /// `VkBuffer`: the texture's bytes, from its first to its last, at its start, of the
    /// texture's `hazard` tracking; each level's bytes of each chain of each part copied back.
    pub(crate) fn staging(
        device: &Device,
        tiles: &TiledTexture,
        at: u64,
        hazard: HazardTrackingMode,
    ) -> Result<ShadowStaging, Error> {
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let buffer = device.own_buffer(tiles.size, options)?;
        let (_, start) = buffer.at(0);
        let back = (tiles.parts.iter())
            .flat_map(|part| {
                let layout = &part.layout;
                (0..layout.chains).flat_map(move |chain| {
                    layout.levels.iter().filter_map(move |level| {
                        // A 3D level lies in the chains of its depth slices alone.
                        let used = !layout.volume || chain < level.depth;
                        let first = part.at + chain * layout.stride + level.base;
                        used.then_some(vk::BufferCopy {
                            src_offset: start + first,
                            dst_offset: at + first,
                            size: level.bytes,
                        })
                    })
                })
            })
            .collect();
        Ok(ShadowStaging {
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start,
                size: tiles.size,
            },
            back,
            buffer,
        })
    }

    /// The moves of a texture of `tiles` through `staging` (its `staging`),
    /// whose image of `format` has its elements in its tight buffer as `planes` say, `tight_len`
    /// bytes, copied to and from the image by `regions` (offsets in that buffer), of the
    /// texture's `hazard` tracking: a 32-bit float depth clamped into `[0, 1]` on its way into
    /// the image where the device takes no other value there (`depth_clamp::clamps_depth`). Refused
    /// as `new_buffer` refuses the tight and shape buffers; the caller checked that they `fit`.
    pub(crate) fn new(
        device: &Device,
        staging: &ShadowStaging,
        (tiles, format): (TiledTexture, Format),
        (planes, tight_len, samples): (&[HeapPlane], u64, u32),
        regions: &[vk::BufferImageCopy],
        hazard: HazardTrackingMode,
    ) -> Result<HeapTiledCopies, Error> {
        let clamps = depth_clamp::clamps_depth(device, format, ImageAspectFlags::DEPTH);
        let levels = Self::levels(&tiles, planes, clamps)
            .ok_or_else(|| internal("a tiled heap texture's moves past u32"))?;
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let tight_bytes = tight_len.next_multiple_of(4);
        let tight = device.own_buffer(tight_bytes, options)?;
        let shapes = device.own_buffer(2 * SHAPE_SLOT, options)?;
        let (tight_raw, tight_at) = tight.at(0);
        let (shapes_raw, shapes_at) = shapes.at(0);
        let (staging_raw, staging_at) = (staging.raw(), staging.into.dst_offset);
        let local = TiledTexelsModule::LOCAL_SIZE;
        let mut moves = Vec::new();
        for into in [true, false] {
            let invocations = u64::from(TiledTexelsModule::invocations(&levels, into));
            let groups = Self::grid(device, invocations)
                .ok_or_else(|| internal("a tiled heap texture's moves past the device's grid"))?;
            let shape = TiledTexelsModule::shape(&levels, into, groups[0] * local)
                .ok_or_else(|| internal("a tiled heap texture of more levels than a shape"))?;
            let at = shapes_at + if into { 0 } else { SHAPE_SLOT };
            moves.push(Move { shape, at, groups });
        }
        let [into, back] = <[Move; 2]>::try_from(moves)
            .map_err(|_| internal("a tiled heap texture's moves are two"))?;
        let shape_bytes = 4 * u64::from(TiledTexelsModule::SHAPE_WORDS);
        let kernel = device.owned_kernel(OwnedModule::TiledTexels)?;
        // Each set binds the tight elements, the tiled bytes and the shape, in binding order.
        let sets = OwnedSets::write(
            device,
            &kernel,
            &[&into, &back]
                .iter()
                .map(|step| {
                    vec![
                        (tight_raw, tight_at, tight_bytes),
                        (staging_raw, staging_at, tiles.size),
                        (shapes_raw, step.at, shape_bytes),
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        let regions = regions
            .iter()
            .map(|region| vk::BufferImageCopy {
                buffer_offset: tight_at + region.buffer_offset,
                ..*region
            })
            .collect();
        Ok(HeapTiledCopies {
            tiles,
            _tight: tight,
            tight_raw,
            regions,
            samples: (samples > 1).then(OnceLock::new),
            planes: planes.to_vec(),
            tight_at,
            _shapes: shapes,
            shapes_raw,
            moves: [into, back],
            kernel,
            sets,
        })
    }

    /// Record the dispatch of the move `into` the image, or back, after writing its shape.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, with every write the dispatch
    /// reads made visible to it; the buffers and the sets are live until it completes.
    unsafe fn dispatch(&self, device: &Vk, command_buffer: vk::CommandBuffer, into: bool) {
        let (step, set) = match into {
            true => (&self.moves[0], self.sets.sets[0]),
            false => (&self.moves[1], self.sets.sets[1]),
        };
        // SAFETY: the caller's contract; the shape fits its slot, and the sets bind the
        // texture's own buffers.
        unsafe {
            device.cmd_update_buffer(command_buffer, self.shapes_raw, step.at, &step.shape);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command_buffer, step.groups[0], step.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
        }
    }

    /// Know `texture`, the multisample texture whose image it is, made on `device` by `call`, whose
    /// moves into its image change what `inexact` names (a 32-bit float depth drawn inexactly,
    /// `drawn-depth-copy`, which it announces as `call`'s), `None` where they are exact: its moves
    /// are planned as its copies are recorded ([`sample_moves`](Self::sample_moves)). Nothing for
    /// a single-sample texture, or once known.
    pub(crate) fn know_samples(
        &self,
        device: &Device,
        call: &'static str,
        texture: &Texture,
        inexact: Option<&'static str>,
    ) {
        let Some(slot) = &self.samples else {
            return;
        };
        if let Some(changes) = inexact {
            depth_copy::announce_changes(device, changes, || {
                let format = texture.pixel_format();
                format!(
                    "`{call}`'s multisampled {format:?} texture in a heap, whose image takes the \
                     depth its bytes hold by draws"
                )
            });
        }
        let known = MovedTexture {
            texture: texture.downgrade(),
        };
        // A texture is known once, as it is made, before any command records its copies.
        drop(slot.set(known));
    }

    /// A multisample texture's moves between its image and the tight buffer `into` the image or
    /// out of it, for a copy an encoder records now ([`SampleMoves`]), their images taken through
    /// `maker`, its queue's ([`UseMaker`]); `None` for a single-sample texture. Refused as the
    /// images are made; internal where the texture is not known or gone, or the device no longer
    /// moves its samples (its making tried them, `Device::sample_moves_run`).
    pub(crate) fn sample_moves(
        &self,
        maker: &UseMaker,
        into: bool,
    ) -> Result<Option<SampleMoves>, Error> {
        let Some(slot) = &self.samples else {
            return Ok(None);
        };
        let known = (slot.get())
            .ok_or_else(|| internal("a multisample heap texture's moves before it was made"))?;
        let texture = Texture::upgrade(&known.texture)
            .ok_or_else(|| internal("a multisample heap texture's moves after it went"))?;
        let reused = Reused::new(maker);
        let tight = (self.tight_raw, self.tight_at);
        let moves = SampleMoves::plan(&reused, &texture, tight, &self.planes, into)?;
        moves
            .ok_or_else(|| {
                internal("a multisample heap texture whose samples the device no longer moves")
            })
            .map(Some)
    }

    /// Whether the image is a multisample one, whose copies take moves
    /// ([`sample_moves`](Self::sample_moves)).
    pub(crate) fn moves_samples(&self) -> bool {
        self.samples.is_some()
    }

    /// Whether the moves between the image and the staging buffer give back each byte they took:
    /// a single-sample image of colour aspects alone, whose copies to and from the tight buffer
    /// move its texel blocks' bits (a depth aspect's copy takes no value outside `[0, 1]`,
    /// `VUID-vkCmdCopyBufferToImage-pRegions-07931`, which the move clamps where the device would
    /// store it clamped), and whose dispatches move words.
    /// So, right after a move in, the staging buffer holds what a move out of the image would
    /// leave there (`ShadowCopies::record_kept_from_staging`).
    pub(crate) fn moves_bits(&self) -> bool {
        self.samples.is_none()
            && (self.regions.iter())
                .all(|region| region.image_subresource.aspect_mask == vk::ImageAspectFlags::COLOR)
    }

    /// Whether the image has a depth aspect, whose moves into it may change a value the heap's
    /// bytes hold (the module docs: a 32-bit float depth outside `[0, 1]`, clamped where the
    /// device lacks `VK_EXT_depth_range_unrestricted`, and the values `drawn-depth-copy`
    /// names), so that a merge's keep copies the staging buffer, the heap's bytes as the image
    /// stands for them, and moves nothing out of the image (`ShadowCopies::record_kept`). Of a
    /// depth whose moves change none, the same bytes.
    pub(crate) fn takes_depth(&self) -> bool {
        (self.regions.iter()).any(|region| {
            (region.image_subresource.aspect_mask).contains(vk::ImageAspectFlags::DEPTH)
        })
    }

    /// Record the move of the image's elements into the tight buffer: a single-sample image's
    /// copies, or a multisample one's `moves` out of it ([`SampleMoves`]; none where its encoder
    /// planned none, failing its command buffer, `EncoderCore::sample_moves`).
    ///
    /// # Safety
    ///
    /// As `record`'s.
    unsafe fn image_out(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        moves: Option<&SampleMoves>,
    ) {
        // SAFETY: the caller's contract; the moves' images and objects are held until the
        // command buffer completes.
        unsafe {
            match &self.samples {
                None => device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    self.tight_raw,
                    &self.regions,
                ),
                Some(_) => {
                    for copy in moves
                        .filter(|moves| !moves.into)
                        .iter()
                        .flat_map(|m| &m.copies)
                    {
                        copy.record(device, command_buffer);
                    }
                }
            }
        }
    }

    /// Record the move of the tight buffer's elements into the image: a single-sample image's
    /// copies, or a multisample one's `moves` into it ([`SampleMoves`]; none where its encoder
    /// planned none, failing its command buffer, `EncoderCore::sample_moves`).
    ///
    /// # Safety
    ///
    /// As `record`'s.
    unsafe fn image_in(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        moves: Option<&SampleMoves>,
    ) {
        // SAFETY: the caller's contract; the moves' images and objects are held until the
        // command buffer completes.
        unsafe {
            match &self.samples {
                None => device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.tight_raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    &self.regions,
                ),
                Some(_) => {
                    for copy in moves
                        .filter(|moves| moves.into)
                        .iter()
                        .flat_map(|m| &m.copies)
                    {
                        copy.record(device, command_buffer);
                    }
                }
            }
        }
    }

    /// Record the move of the texture's `image` from `bytes` (the heap's `VkBuffer`) through
    /// `staging` (`into`), or back, its rows back by `back` (`shadow::record_back`), a multisample
    /// image's through `moves` that way ([`sample_moves`](Self::sample_moves)).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the image, the buffers, the sets and
    /// the moves are live until it completes, and the image rests in `GENERAL`; `back` is the
    /// copy's `ShadowCopy::staged_back`, over `bytes`.
    pub(crate) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        (bytes, staging): (vk::Buffer, &ShadowStaging),
        (into, moves): (bool, Option<&SampleMoves>),
        back: Option<&RowCopies>,
    ) {
        // SAFETY: the caller's contract; the copies lie in the buffers and the image (planned
        // when the texture was made).
        unsafe {
            Barrier::everything().record(device, command_buffer);
            // Both ways the staging buffer starts as the heap's bytes: back, so the bytes
            // between the elements are the heap's when they are copied back.
            device.cmd_copy_buffer(command_buffer, bytes, staging.raw(), &[staging.into]);
            if !into {
                self.image_out(device, command_buffer, image, moves);
            }
            self.dispatch(device, command_buffer, into);
            if into {
                self.image_in(device, command_buffer, image, moves);
            } else {
                shadow::record_back(device, command_buffer, (staging, bytes), back);
            }
        }
    }

    /// Record a copy of the texture's `image` into `to` from `at`, laid out as in the heap's
    /// bytes from the texture's first (a merge's kept or changed part, `shadow_merge.rs`): the
    /// move back into `staging`, a multisample image's through `moves` out of it, then its bytes
    /// on. A byte no element lies in is the staging buffer's, the heap's as the last move left it,
    /// alike in each copy of one merge.
    ///
    /// # Safety
    ///
    /// As `record`; `to` holds the texture's bytes from `at`, a multiple of 4.
    pub(crate) unsafe fn record_out(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (image, moves): (vk::Image, Option<&SampleMoves>),
        staging: &ShadowStaging,
        (to, at): (vk::Buffer, u64),
    ) {
        // SAFETY: the caller's contract, as `record`'s.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            self.image_out(device, command_buffer, image, moves);
            self.dispatch(device, command_buffer, false);
            let region = vk::BufferCopy {
                src_offset: staging.into.dst_offset,
                dst_offset: at,
                size: staging.into.size,
            };
            device.cmd_copy_buffer(command_buffer, staging.raw(), to, &[region]);
        }
    }
}

impl HeapTiledCopies {
    /// Whether `device` runs the moves of `texture`'s multisample image to and from a tight
    /// buffer laid out as `planes` ([`SampleMoves`]), planned each way over no buffer and dropped:
    /// `None` where it does not, else what the moves into the image change
    /// (`drawn-depth-copy`), `None` where they are exact.
    pub(crate) fn samples_run(
        device: &Device,
        texture: &Texture,
        planes: &[HeapPlane],
    ) -> Result<Option<Option<&'static str>>, Error> {
        let none = (vk::Buffer::null(), 0);
        let reused = Reused::new(device);
        let Some(into) = SampleMoves::plan(&reused, texture, none, planes, true)? else {
            return Ok(None);
        };
        let out = SampleMoves::plan(&reused, texture, none, planes, false)?;
        let inexact = into.copies.iter().find_map(InterleaveCopy::inexact_depth);
        Ok(out.map(|_| inexact))
    }
}

impl SampleMoves {
    /// The moves of `texture`'s multisample image to or from (`into` the image) the tight buffer
    /// (`tight`, its `VkBuffer` and where it lies there) laid out as `planes` (one an aspect, its
    /// images the texture's slices, each pixel's samples one element), each slice of each aspect
    /// a step of `maker`, taking the images an earlier one took ([`Reused`]). `None` where the
    /// device cannot run one, as the blits' copies of such a texture refuse
    /// (`sample_interleave.rs`).
    fn plan(
        maker: &Reused,
        texture: &Texture,
        (tight, tight_at): (vk::Buffer, u64),
        planes: &[HeapPlane],
        into: bool,
    ) -> Result<Option<SampleMoves>, Error> {
        let samples = texture.record().samples;
        let mut copies = Vec::new();
        for plane in planes {
            let [width, height, _] = plane.extent;
            for slice in 0..plane.images {
                let layout = BufferLayout {
                    offset: plane.offset + slice * plane.image_pitch,
                    bytes_per_row: plane.pitch,
                    bytes_per_image: plane.image_pitch,
                };
                let region = TextureRegion {
                    level: plane.level,
                    slice: u32::try_from(slice)
                        .map_err(|_| internal("a multisample heap texture's slice past u32"))?,
                    origin: [0; 3],
                    size: [width, height, 1],
                };
                // The tight buffer's rows of this slice end within its image pitch.
                let planned = InterleaveCopy::plan(
                    "a heap multisample texture's sample copy",
                    maker,
                    tight,
                    tight_at,
                    layout.offset + plane.image_pitch,
                    &layout,
                    texture,
                    &region,
                    plane.aspect,
                    samples,
                    into,
                )?;
                let Some(copy) = planned else {
                    return Ok(None);
                };
                copies.push(copy);
                maker.next();
            }
        }
        Ok(Some(SampleMoves { copies, into }))
    }
}
