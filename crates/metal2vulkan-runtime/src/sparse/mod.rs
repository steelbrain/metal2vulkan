//! Sparse heaps and their textures (`MTLHeapTypeSparse`).
//!
//! - **Metal's answers** (this module). The tile a sparse texture maps in
//!   ([`Device::sparse_tile_size`]), its bytes, a texture's `firstMipmapInTail` and
//!   `tailSizeInBytes`, and the region conversions are Metal's rules as an M4 answered them, on
//!   every device: they make nothing.
//! - **The device** (`device.rs`). What the device has of sparse binding and residency, enabled
//!   as the runtime's own need, or what it lacks, which a sparse heap or texture is refused
//!   naming (`Feature::SparseTextures`) where a function may read the image's residency.
//! - **The textures** (`texture.rs`, `mapping.rs`). A sparse heap's texture is a sparse resident
//!   `VkImage` whose memory is bound by Vulkan unit (a sparse block or a mip tail) while any of
//!   Metal's tiles it holds texels of is mapped, the unit's unmapped tiles kept at zero; or, where
//!   the device makes no sparse image of it and no function reads an image's own residency
//!   (MoltenVK), an ordinary image whose unmapped tiles the runtime keeps at zero (`emulated.rs`).
//!   Either keeps a tile map where its type has one, from which every function reads the residency
//!   of Metal's tiles.

pub(super) mod bind;
pub(super) mod device;
pub(super) mod emulated;
pub(super) mod encoder;
pub(super) mod heap;
pub(super) mod indirect;
pub(super) mod mapping;
pub(super) mod texture;

use super::copy_plan::block_extent;
use super::error::{invalid, Error, Problem};
use super::values::{Region, Size};
use super::Device;
use metal2vulkan::metal::format::{byte_format, is_depth_stencil};
use metal2vulkan::metal::{PixelFormat, TextureType};

/// The page of a sparse heap's tiles with no page named: `MTLSparsePageSize16`, the default of a
/// heap descriptor and of the queries without a page.
pub(super) const DEFAULT_PAGE: SparsePageSize = SparsePageSize::Size16;

/// `MTLSparsePageSize`, with Metal's raw values: the bytes of one tile of a sparse heap.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SparsePageSize {
    /// `MTLSparsePageSize16`: 16 KiB.
    #[default]
    Size16,
    /// `MTLSparsePageSize64`: 64 KiB.
    Size64,
    /// `MTLSparsePageSize256`: 256 KiB.
    Size256,
}

impl SparsePageSize {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            SparsePageSize::Size16 => 101,
            SparsePageSize::Size64 => 102,
            SparsePageSize::Size256 => 103,
        }
    }

    /// The page of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<SparsePageSize, Error> {
        Ok(match raw {
            101 => SparsePageSize::Size16,
            102 => SparsePageSize::Size64,
            103 => SparsePageSize::Size256,
            raw => return Err(undefined("MTLSparsePageSize", raw)),
        })
    }

    /// The page's bytes.
    pub fn bytes(self) -> u64 {
        match self {
            SparsePageSize::Size16 => 16 << 10,
            SparsePageSize::Size64 => 64 << 10,
            SparsePageSize::Size256 => 256 << 10,
        }
    }
}

/// `MTLSparseTextureRegionAlignmentMode`, with Metal's raw values.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SparseTextureRegionAlignmentMode {
    /// `MTLSparseTextureRegionAlignmentModeOutward`: the tiles that cover the region.
    #[default]
    Outward,
    /// `MTLSparseTextureRegionAlignmentModeInward`: the tiles inside the region.
    Inward,
}

impl SparseTextureRegionAlignmentMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            SparseTextureRegionAlignmentMode::Outward => 0,
            SparseTextureRegionAlignmentMode::Inward => 1,
        }
    }

    /// The mode of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<SparseTextureRegionAlignmentMode, Error> {
        Ok(match raw {
            0 => SparseTextureRegionAlignmentMode::Outward,
            1 => SparseTextureRegionAlignmentMode::Inward,
            raw => return Err(undefined("MTLSparseTextureRegionAlignmentMode", raw)),
        })
    }
}

fn undefined(ty: &'static str, raw: u64) -> Error {
    invalid("from_raw", Problem::UndefinedValue { ty, raw })
}

/// What a sparse heap's or texture's call did wrong (`Problem::Sparse`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseProblem {
    /// A texture type a sparse heap makes no texture of: 1D, 1D array and texture buffers, for
    /// which Metal answers nil (an [`Error::Nil`](crate::Error::Nil)).
    TextureType {
        /// The descriptor's texture type.
        texture_type: TextureType,
    },
    /// A dual-plane format (depth and stencil), which validation stops: what mapping its stencil
    /// does without validation is not measured.
    DualPlane {
        /// The descriptor's pixel format.
        pixel_format: PixelFormat,
    },
    /// A move between a sparse texture and one that is not (not measured; a mapping update of a
    /// texture that is not sparse, and a move between two such, record nothing, as plain Metal
    /// does).
    NotSparse,
    /// Lists of regions, levels and slices of unequal lengths (`update_texture_mappings`): Metal's
    /// one `numRegions` cannot state them, so no Metal call reaches this.
    Lengths,
    /// An indirect update's count, `bytes` of it at `offset`, past the buffer's `length`, where it
    /// is encoded.
    IndirectArguments {
        /// The arguments' offset.
        offset: u64,
        /// The arguments' bytes.
        bytes: u64,
        /// The buffer's length.
        length: u64,
    },
}

/// The bytes of one block of `pf` as a sparse tile counts it, and the block's texels: a color
/// format's texel block, and a depth or stencil format's depth plane (4 bytes for
/// `Depth32Float_Stencil8`, measured), or its stencil's where it has only one.
fn tile_block(pf: PixelFormat) -> Option<(u64, [u64; 2])> {
    if is_depth_stencil(pf) {
        let bytes = match pf {
            PixelFormat::Depth16Unorm => 2,
            PixelFormat::Stencil8 => 1,
            PixelFormat::Depth32Float
            | PixelFormat::Depth24UnormStencil8
            | PixelFormat::Depth32FloatStencil8 => 4,
            _ => return None,
        };
        return Some((bytes, [1, 1]));
    }
    let bytes = u64::from(byte_format(pf)?.block()?.size);
    let [width, height] = block_extent(pf).unwrap_or([1, 1]);
    Some((bytes, [u64::from(width), u64::from(height)]))
}

/// Metal's sparse tile, in texels, for `pf` at `samples` samples in pages of `page` bytes: the
/// same for every texture type (measured on an M4). With `b` bytes a block and `s` samples, `n =
/// ceil(log2(page / (b * s)))` blocks, `2^ceil(n/2)` blocks wide and `2^floor(n/2)` high, one deep;
/// 1x1x1 for a format of no block bytes, or 0 samples.
pub(super) fn tile_texels(pf: PixelFormat, samples: u64, page: u64) -> [u64; 3] {
    let Some((bytes, [block_width, block_height])) = tile_block(pf) else {
        return [1, 1, 1];
    };
    if samples == 0 {
        return [1, 1, 1];
    }
    let unit = u128::from(bytes) * u128::from(samples);
    // The least `n` with `2^n * unit >= page`.
    let mut n = 0u32;
    while n < 62 && (unit << n) < u128::from(page) {
        n += 1;
    }
    let width = 1u64 << n.div_ceil(2);
    let height = 1u64 << (n / 2);
    [width * block_width, height * block_height, 1]
}

/// The extent of level `level` of a texture whose level 0 is `extent` (each axis halved,
/// at least 1).
pub(super) fn level_extent(extent: [u64; 3], level: u64) -> [u64; 3] {
    let shift = level.min(63) as u32;
    extent.map(|axis| (axis >> shift).max(1))
}

/// The levels Metal's tail rule runs over for a texture of `levels` levels whose level 0 is
/// `extent`: a texture of one level its one level, a mipmapped texture its full chain of
/// `floor(log2(largest axis)) + 1` levels whatever its own count (measured).
fn chain_levels(extent: [u64; 3], levels: u64) -> u64 {
    if levels <= 1 {
        return levels;
    }
    let largest = extent.into_iter().max().unwrap_or(1).max(1);
    u64::from(largest.ilog2()) + 1
}

/// Metal's `firstMipmapInTail` of a sparse texture whose level 0 is `extent`, of `levels`
/// levels, in tiles of `tile` texels: the first level of its chain ([`chain_levels`]) whose width
/// is at most half the tile's or whose height is at most half the tile's, else the chain's
/// length. A mipmapped texture's may be past its own levels (measured).
pub(super) fn first_mipmap_in_tail(extent: [u64; 3], levels: u64, tile: [u64; 3]) -> u64 {
    let chain = chain_levels(extent, levels);
    (0..chain)
        .find(|&level| {
            let [width, height, _] = level_extent(extent, level);
            width <= tile[0] / 2 || height <= tile[1] / 2
        })
        .unwrap_or(chain)
}

/// Metal's `tailSizeInBytes` of a sparse texture of `texture_type` and `pf` at `samples`
/// samples whose level 0 is `extent`, of `levels` levels, with its tail from `first` in pages of
/// `page` bytes, over its chain ([`chain_levels`]): per slice, the bytes of the tail's levels in
/// whole blocks, rounded up to whole pages; for a 3D texture, for each depth slice `d` of the first
/// tail level, the bytes of slice `d` of each tail level that has one, rounded up to whole pages,
/// summed; 0 without a tail (measured).
pub(super) fn tail_size_in_bytes(
    texture_type: TextureType,
    pf: PixelFormat,
    samples: u64,
    extent: [u64; 3],
    levels: u64,
    first: u64,
    page: u64,
) -> u64 {
    let chain = chain_levels(extent, levels);
    if first >= chain || page == 0 {
        return 0;
    }
    let Some((bytes, [block_width, block_height])) = tile_block(pf) else {
        return 0;
    };
    let slice_bytes = |level: u64| {
        let [width, height, _] = level_extent(extent, level);
        width.div_ceil(block_width) * height.div_ceil(block_height) * bytes * samples.max(1)
    };
    let pages = |sum: u64| sum.div_ceil(page) * page;
    if texture_type != TextureType::Type3D {
        return pages((first..chain).map(slice_bytes).sum());
    }
    (0..level_extent(extent, first)[2])
        .map(|slice| {
            pages(
                (first..chain)
                    .filter(|&level| slice < level_extent(extent, level)[2])
                    .map(slice_bytes)
                    .sum(),
            )
        })
        .sum()
}

/// `convertSparsePixelRegions:...` of one region: the tiles of `tile` texels that cover it
/// (`Outward`), or that lie inside it (`Inward`, whose size Metal computes in unsigned texels, so
/// a region inside one tile wraps: measured). `tile` has no axis of 0.
pub(super) fn pixel_region_tiles(
    region: &Region,
    tile: [u64; 3],
    mode: SparseTextureRegionAlignmentMode,
) -> Region {
    let origin = [region.origin.x, region.origin.y, region.origin.z];
    let size = [region.size.width, region.size.height, region.size.depth];
    let axis = |index: usize| -> (u64, u64) {
        let (start, end) = (origin[index], origin[index].wrapping_add(size[index]));
        let t = tile[index];
        match mode {
            SparseTextureRegionAlignmentMode::Outward => {
                let first = start / t;
                (first, end.div_ceil(t).wrapping_sub(first))
            }
            SparseTextureRegionAlignmentMode::Inward => {
                let first = start.div_ceil(t);
                let texels = (end / t * t).wrapping_sub(first.wrapping_mul(t));
                (first, texels / t)
            }
        }
    };
    let ((x, width), (y, height), (z, depth)) = (axis(0), axis(1), axis(2));
    Region::new_3d(x, y, z, width, height, depth)
}

/// `convertSparsePixelRegions:...` of each of `pixels` into the region of `tiles` at its place,
/// by [`pixel_region_tiles`]; a `tile` with an axis of 0 writes nothing, as plain Metal writes
/// nothing (an M4 and an M3 Max).
pub(super) fn convert_pixel_regions(
    pixels: &[Region],
    tiles: &mut [Region],
    tile: [u64; 3],
    mode: SparseTextureRegionAlignmentMode,
) {
    if tile.contains(&0) {
        return;
    }
    for (pixels, tiles) in pixels.iter().zip(tiles) {
        *tiles = pixel_region_tiles(pixels, tile, mode);
    }
}

impl Device {
    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`: the tile of a sparse texture of
    /// `pixel_format` at `sample_count` samples in 16-KiB pages, in texels, the same for every
    /// texture type, as Metal's rule answers on every device.
    pub fn sparse_tile_size(
        &self,
        texture_type: TextureType,
        pixel_format: PixelFormat,
        sample_count: u64,
    ) -> Size {
        self.sparse_tile_size_for_page(texture_type, pixel_format, sample_count, DEFAULT_PAGE)
    }

    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:`: as
    /// [`sparse_tile_size`](Self::sparse_tile_size), in pages of `page`.
    pub fn sparse_tile_size_for_page(
        &self,
        _texture_type: TextureType,
        pixel_format: PixelFormat,
        sample_count: u64,
        page: SparsePageSize,
    ) -> Size {
        let [width, height, depth] = tile_texels(pixel_format, sample_count, page.bytes());
        Size::new(width, height, depth)
    }

    /// `sparseTileSizeInBytes`: 16384, the bytes of a 16-KiB page's tile.
    pub fn sparse_tile_size_in_bytes(&self) -> u64 {
        DEFAULT_PAGE.bytes()
    }

    /// `sparseTileSizeInBytesForSparsePageSize:`: the page's bytes.
    pub fn sparse_tile_size_in_bytes_for_page(&self, page: SparsePageSize) -> u64 {
        page.bytes()
    }

    /// `convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:`: each
    /// region of texels, `pixel_regions[i]`, as tiles of `tile_size` into `tile_regions[i]` (see
    /// [`SparseTextureRegionAlignmentMode`]); Metal's `numRegions` is the slices' one length.
    ///
    /// A tile size with an axis of 0 writes nothing, as plain Metal writes nothing, and its
    /// validation takes it (an M4 under macOS 27.0.1 and an M3 Max under 26.7, each axis and all
    /// three at 0, both alignment modes: the caller's regions keep what they held).
    ///
    /// Refused (`Invalid(LengthMismatch)`): slices of different lengths, which Metal's one count
    /// cannot state; a refusal writes nothing.
    pub fn convert_sparse_pixel_regions(
        &self,
        pixel_regions: &[Region],
        tile_regions: &mut [Region],
        tile_size: Size,
        mode: SparseTextureRegionAlignmentMode,
    ) -> Result<(), Error> {
        if pixel_regions.len() != tile_regions.len() {
            return Err(invalid(
                "convert_sparse_pixel_regions",
                Problem::LengthMismatch,
            ));
        }
        let tile = [tile_size.width, tile_size.height, tile_size.depth];
        convert_pixel_regions(pixel_regions, tile_regions, tile, mode);
        Ok(())
    }

    /// `convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:`: each region of tiles
    /// of `tile_size` as texels.
    pub fn convert_sparse_tile_regions(
        &self,
        tile_regions: &[Region],
        tile_size: Size,
    ) -> Vec<Region> {
        tile_regions
            .iter()
            .map(|region| {
                Region::new_3d(
                    region.origin.x.wrapping_mul(tile_size.width),
                    region.origin.y.wrapping_mul(tile_size.height),
                    region.origin.z.wrapping_mul(tile_size.depth),
                    region.size.width.wrapping_mul(tile_size.width),
                    region.size.height.wrapping_mul(tile_size.height),
                    region.size.depth.wrapping_mul(tile_size.depth),
                )
            })
            .collect()
    }
}
