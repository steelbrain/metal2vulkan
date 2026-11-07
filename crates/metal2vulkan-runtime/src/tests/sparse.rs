//! `sparse`: Metal's sparse answers that need no device, each against what an M4 answered: tiles,
//! mip tails, conversions, the page values, and what a device's sparse support refuses; and how
//! Metal's units lie in the device's, which texels a batch zeroes and a move copies.

use crate::error::{Error, RuntimeGap};
use crate::sparse::device::{wanted, SparseDevice};
use crate::sparse::mapping::{MetalUnit, MoveSide, TexelBox, UnitGrid, VkUnit};
use crate::sparse::texture::{sparse_refusal, SparseShape, VkSparseLayout};
use crate::sparse::{
    convert_pixel_regions, first_mipmap_in_tail, pixel_region_tiles, tail_size_in_bytes,
    tile_texels, SparsePageSize, SparseTextureRegionAlignmentMode,
};
use crate::{HeapDescriptor, Origin, Region, Size};
use ash::vk;
use metal2vulkan::error::{Feature, SparseNeed, Unsupported};
use metal2vulkan::metal::{PixelFormat, TextureType};
use std::collections::{BTreeSet, HashSet};

const K16: u64 = 16 << 10;
const K64: u64 = 64 << 10;
const K256: u64 = 256 << 10;

#[test]
fn the_page_values_are_metal_s() {
    for (page, raw, bytes) in [
        (SparsePageSize::Size16, 101, K16),
        (SparsePageSize::Size64, 102, K64),
        (SparsePageSize::Size256, 103, K256),
    ] {
        assert_eq!(page.raw(), raw);
        assert_eq!(SparsePageSize::from_raw(raw), Ok(page));
        assert_eq!(page.bytes(), bytes);
    }
    for raw in [0, 100, 104, 200] {
        assert!(SparsePageSize::from_raw(raw).is_err(), "{raw}");
    }
    assert_eq!(
        HeapDescriptor::default().sparse_page_size,
        SparsePageSize::Size16
    );
    assert_eq!(SparseTextureRegionAlignmentMode::Outward.raw(), 0);
    assert_eq!(SparseTextureRegionAlignmentMode::Inward.raw(), 1);
}

/// The tiles an M4 answered, by format, sample count and page.
#[test]
fn tiles_are_metal_s_rule() {
    use PixelFormat as P;
    let cases: &[(PixelFormat, u64, u64, [u64; 3])] = &[
        (P::R8Unorm, 1, K16, [128, 128, 1]),
        (P::R8Unorm, 1, K64, [256, 256, 1]),
        (P::R8Unorm, 1, K256, [512, 512, 1]),
        (P::R8Unorm, 2, K16, [128, 64, 1]),
        (P::R8Unorm, 4, K64, [128, 128, 1]),
        (P::R8Unorm, 8, K256, [256, 128, 1]),
        (P::RG8Unorm, 1, K16, [128, 64, 1]),
        (P::RG8Unorm, 8, K16, [32, 32, 1]),
        (P::RGBA8Unorm, 1, K16, [64, 64, 1]),
        (P::RGBA8Unorm, 1, K64, [128, 128, 1]),
        (P::RGBA8Unorm, 1, K256, [256, 256, 1]),
        (P::RGBA8Unorm, 2, K16, [64, 32, 1]),
        (P::RGBA8Unorm, 4, K16, [32, 32, 1]),
        (P::RGBA8Unorm, 8, K16, [32, 16, 1]),
        (P::RGBA8Unorm, 8, K256, [128, 64, 1]),
        (P::RGBA8UnormSRGB, 1, K16, [64, 64, 1]),
        (P::R32Float, 1, K16, [64, 64, 1]),
        (P::RGBA16Float, 1, K16, [64, 32, 1]),
        (P::RGBA32Float, 1, K16, [32, 32, 1]),
        (P::BC1RGBA, 1, K16, [256, 128, 1]),
        (P::BC7RGBAUnorm, 1, K16, [128, 128, 1]),
        (P::ASTC4x4LDR, 1, K16, [128, 128, 1]),
        (P::Depth32Float, 1, K16, [64, 64, 1]),
        (P::Depth16Unorm, 1, K16, [128, 64, 1]),
        (P::Stencil8, 1, K16, [128, 128, 1]),
        (P::Depth32FloatStencil8, 1, K16, [64, 64, 1]),
        // Odd and zero sample counts, and no format.
        (P::RGBA8Unorm, 3, K16, [64, 32, 1]),
        (P::RGBA8Unorm, 16, K16, [16, 16, 1]),
        (P::RGBA8Unorm, 0, K16, [1, 1, 1]),
        (P::Invalid, 1, K16, [1, 1, 1]),
    ];
    for &(pf, samples, page, tile) in cases {
        assert_eq!(
            tile_texels(pf, samples, page),
            tile,
            "{pf:?} x{samples} {page}"
        );
    }
}

/// `firstMipmapInTail` and `tailSizeInBytes` as an M4 answered them (each page size: textures of a
/// few levels, whose rule runs over the full chain): (type, format, extent, levels, page, first,
/// tail bytes), one sample.
#[test]
fn mip_tails_are_metal_s_rule() {
    use PixelFormat as P;
    use TextureType as T;
    #[rustfmt::skip]
    let cases: &[(TextureType, PixelFormat, [u64; 3], u64, u64, u64, u64)] = &[
        (T::Type2D, P::R32Float, [256, 128, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [16, 16, 1], 1, K16, 0, 16384),
        (T::Type2D, P::R32Float, [64, 64, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [63, 64, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [64, 63, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [32, 256, 1], 1, K16, 0, 32768),
        (T::Type2D, P::R32Float, [256, 32, 1], 1, K16, 0, 32768),
        (T::Type2D, P::R32Float, [128, 128, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [128, 128, 1], 8, K16, 2, 16384),
        (T::Type2D, P::R32Float, [256, 256, 1], 9, K16, 3, 16384),
        (T::Type2D, P::R32Float, [1024, 1024, 1], 11, K16, 5, 16384),
        (T::Type2D, P::R32Float, [1000, 700, 1], 10, K16, 5, 16384),
        (T::Type2D, P::R32Float, [4096, 64, 1], 13, K16, 1, 360448),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K16, 3, 16384),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 10, K16, 4, 16384),
        (T::Type2D, P::R8Unorm, [1024, 1024, 1], 11, K16, 4, 16384),
        (T::Type2D, P::RGBA16Float, [256, 256, 1], 9, K16, 3, 16384),
        (T::Type2D, P::RGBA32Float, [256, 256, 1], 9, K16, 4, 16384),
        (T::Type2D, P::BC1RGBA, [512, 512, 1], 10, K16, 2, 16384),
        (T::Type2D, P::BC7RGBAUnorm, [512, 512, 1], 10, K16, 3, 16384),
        (T::Type2D, P::ASTC4x4LDR, [512, 512, 1], 10, K16, 3, 16384),
        (T::Type2D, P::Depth32Float, [256, 256, 1], 1, K16, 1, 0),
        (T::Type2D, P::Depth16Unorm, [256, 256, 1], 1, K16, 1, 0),
        (T::Type2D, P::Stencil8, [256, 256, 1], 1, K16, 1, 0),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 9, K16, 3, 16384),
        (T::Type2DArray, P::R32Float, [16, 16, 1], 1, K16, 0, 16384),
        (T::Type3D, P::R32Float, [64, 32, 32], 7, K16, 0, 524288),
        (T::Type3D, P::R32Float, [256, 256, 16], 9, K16, 3, 32768),
        (T::Type3D, P::R32Float, [64, 64, 32], 1, K16, 1, 0),
        (T::Type3D, P::RGBA8Unorm, [128, 128, 128], 8, K16, 2, 524288),
        (T::Cube, P::R32Float, [256, 256, 1], 9, K16, 3, 16384),
        (T::CubeArray, P::R32Float, [256, 256, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [256, 128, 1], 1, K64, 1, 0),
        (T::Type2D, P::R32Float, [16, 16, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [64, 64, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [63, 64, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [64, 63, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [32, 256, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [256, 32, 1], 1, K64, 0, 65536),
        (T::Type2D, P::R32Float, [128, 128, 1], 1, K64, 1, 0),
        (T::Type2D, P::R32Float, [128, 128, 1], 8, K64, 1, 65536),
        (T::Type2D, P::R32Float, [256, 256, 1], 9, K64, 2, 65536),
        (T::Type2D, P::R32Float, [1024, 1024, 1], 11, K64, 4, 65536),
        (T::Type2D, P::R32Float, [1000, 700, 1], 10, K64, 4, 65536),
        (T::Type2D, P::R32Float, [4096, 64, 1], 13, K64, 0, 1441792),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K64, 2, 65536),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 10, K64, 3, 65536),
        (T::Type2D, P::R8Unorm, [1024, 1024, 1], 11, K64, 3, 65536),
        (T::Type2D, P::RGBA16Float, [256, 256, 1], 9, K64, 2, 65536),
        (T::Type2D, P::RGBA32Float, [256, 256, 1], 9, K64, 3, 65536),
        (T::Type2D, P::BC1RGBA, [512, 512, 1], 10, K64, 1, 65536),
        (T::Type2D, P::BC7RGBAUnorm, [512, 512, 1], 10, K64, 2, 65536),
        (T::Type2D, P::ASTC4x4LDR, [512, 512, 1], 10, K64, 2, 65536),
        (T::Type2D, P::Depth32Float, [256, 256, 1], 1, K64, 1, 0),
        (T::Type2D, P::Depth16Unorm, [256, 256, 1], 1, K64, 1, 0),
        (T::Type2D, P::Stencil8, [256, 256, 1], 1, K64, 1, 0),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 9, K64, 2, 65536),
        (T::Type2DArray, P::R32Float, [16, 16, 1], 1, K64, 0, 65536),
        (T::Type3D, P::R32Float, [64, 32, 32], 7, K64, 0, 2097152),
        (T::Type3D, P::R32Float, [256, 256, 16], 9, K64, 2, 262144),
        (T::Type3D, P::R32Float, [64, 64, 32], 1, K64, 0, 2097152),
        (T::Type3D, P::RGBA8Unorm, [128, 128, 128], 8, K64, 1, 4194304),
        (T::Cube, P::R32Float, [256, 256, 1], 9, K64, 2, 65536),
        (T::CubeArray, P::R32Float, [256, 256, 1], 1, K64, 1, 0),
        (T::Type2D, P::R32Float, [256, 128, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [16, 16, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [64, 64, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [63, 64, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [64, 63, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [32, 256, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [256, 32, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [128, 128, 1], 1, K256, 0, 262144),
        (T::Type2D, P::R32Float, [128, 128, 1], 8, K256, 0, 262144),
        (T::Type2D, P::R32Float, [256, 256, 1], 9, K256, 1, 262144),
        (T::Type2D, P::R32Float, [1024, 1024, 1], 11, K256, 3, 262144),
        (T::Type2D, P::R32Float, [1000, 700, 1], 10, K256, 3, 262144),
        (T::Type2D, P::R32Float, [4096, 64, 1], 13, K256, 0, 1572864),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K256, 1, 262144),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 10, K256, 2, 262144),
        (T::Type2D, P::R8Unorm, [1024, 1024, 1], 11, K256, 2, 262144),
        (T::Type2D, P::RGBA16Float, [256, 256, 1], 9, K256, 1, 262144),
        (T::Type2D, P::RGBA32Float, [256, 256, 1], 9, K256, 2, 262144),
        (T::Type2D, P::BC1RGBA, [512, 512, 1], 10, K256, 0, 262144),
        (T::Type2D, P::BC7RGBAUnorm, [512, 512, 1], 10, K256, 1, 262144),
        (T::Type2D, P::ASTC4x4LDR, [512, 512, 1], 10, K256, 1, 262144),
        (T::Type2D, P::Depth32Float, [256, 256, 1], 1, K256, 1, 0),
        (T::Type2D, P::Depth16Unorm, [256, 256, 1], 1, K256, 0, 262144),
        (T::Type2D, P::Stencil8, [256, 256, 1], 1, K256, 0, 262144),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 9, K256, 1, 262144),
        (T::Type2DArray, P::R32Float, [16, 16, 1], 1, K256, 0, 262144),
        (T::Type3D, P::R32Float, [64, 32, 32], 7, K256, 0, 8388608),
        (T::Type3D, P::R32Float, [256, 256, 16], 9, K256, 1, 2097152),
        (T::Type3D, P::R32Float, [64, 64, 32], 1, K256, 0, 8388608),
        (T::Type3D, P::RGBA8Unorm, [128, 128, 128], 8, K256, 0, 33554432),
        (T::Cube, P::R32Float, [256, 256, 1], 9, K256, 1, 262144),
        (T::CubeArray, P::R32Float, [256, 256, 1], 1, K256, 1, 0),
        (T::Type2D, P::R32Float, [8192, 128, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [8192, 128, 1], 2, K16, 2, 360448),
        (T::Type2D, P::R32Float, [8192, 128, 1], 3, K16, 2, 360448),
        (T::Type2D, P::R32Float, [8192, 128, 1], 4, K16, 2, 360448),
        (T::Type2D, P::R32Float, [8192, 128, 1], 14, K16, 2, 360448),
        (T::Type2D, P::R32Float, [4096, 64, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [4096, 64, 1], 2, K16, 1, 360448),
        (T::Type2D, P::R32Float, [256, 256, 1], 1, K16, 1, 0),
        (T::Type2D, P::R32Float, [256, 256, 1], 2, K16, 3, 16384),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K16, 3, 16384),
        (T::Type2D, P::R32Float, [256, 256, 1], 4, K16, 3, 16384),
        (T::Type2D, P::R32Float, [128, 128, 1], 2, K16, 2, 16384),
        (T::Type2D, P::R32Float, [128, 128, 1], 3, K16, 2, 16384),
        (T::Type2D, P::R32Float, [256, 128, 1], 2, K16, 2, 16384),
        (T::Type2D, P::R32Float, [64, 64, 1], 2, K16, 1, 16384),
        (T::Type2D, P::R32Float, [16, 16, 1], 2, K16, 0, 16384),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 4, K16, 4, 16384),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 5, K16, 4, 16384),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 3, K16, 3, 16384),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 2, K16, 2, 360448),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 3, K16, 2, 360448),
        (T::Type3D, P::R32Float, [256, 256, 16], 2, K16, 3, 32768),
        (T::Type3D, P::R32Float, [256, 256, 16], 3, K16, 3, 32768),
        (T::Type3D, P::R32Float, [256, 256, 16], 4, K16, 3, 32768),
        (T::Type3D, P::R32Float, [1024, 64, 8], 2, K16, 1, 311296),
        (T::Type3D, P::R32Float, [1024, 64, 8], 11, K16, 1, 311296),
        (T::Cube, P::R32Float, [256, 256, 1], 3, K16, 3, 16384),
        (T::Cube, P::R32Float, [256, 256, 1], 2, K16, 3, 16384),
        (T::Type2D, P::R32Float, [8192, 128, 1], 1, K64, 1, 0),
        (T::Type2D, P::R32Float, [8192, 128, 1], 2, K64, 1, 1441792),
        (T::Type2D, P::R32Float, [8192, 128, 1], 3, K64, 1, 1441792),
        (T::Type2D, P::R32Float, [8192, 128, 1], 4, K64, 1, 1441792),
        (T::Type2D, P::R32Float, [8192, 128, 1], 14, K64, 1, 1441792),
        (T::Type2D, P::R32Float, [4096, 64, 1], 1, K64, 0, 1048576),
        (T::Type2D, P::R32Float, [4096, 64, 1], 2, K64, 0, 1441792),
        (T::Type2D, P::R32Float, [256, 256, 1], 1, K64, 1, 0),
        (T::Type2D, P::R32Float, [256, 256, 1], 2, K64, 2, 65536),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K64, 2, 65536),
        (T::Type2D, P::R32Float, [256, 256, 1], 4, K64, 2, 65536),
        (T::Type2D, P::R32Float, [128, 128, 1], 2, K64, 1, 65536),
        (T::Type2D, P::R32Float, [128, 128, 1], 3, K64, 1, 65536),
        (T::Type2D, P::R32Float, [256, 128, 1], 2, K64, 1, 65536),
        (T::Type2D, P::R32Float, [64, 64, 1], 2, K64, 0, 65536),
        (T::Type2D, P::R32Float, [16, 16, 1], 2, K64, 0, 65536),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 4, K64, 3, 65536),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 5, K64, 3, 65536),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 3, K64, 2, 65536),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 2, K64, 1, 1441792),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 3, K64, 1, 1441792),
        (T::Type3D, P::R32Float, [256, 256, 16], 2, K64, 2, 262144),
        (T::Type3D, P::R32Float, [256, 256, 16], 3, K64, 2, 262144),
        (T::Type3D, P::R32Float, [256, 256, 16], 4, K64, 2, 262144),
        (T::Type3D, P::R32Float, [1024, 64, 8], 2, K64, 0, 2490368),
        (T::Type3D, P::R32Float, [1024, 64, 8], 11, K64, 0, 2490368),
        (T::Cube, P::R32Float, [256, 256, 1], 3, K64, 2, 65536),
        (T::Cube, P::R32Float, [256, 256, 1], 2, K64, 2, 65536),
        (T::Type2D, P::R32Float, [8192, 128, 1], 1, K256, 0, 4194304),
        (T::Type2D, P::R32Float, [8192, 128, 1], 2, K256, 0, 5767168),
        (T::Type2D, P::R32Float, [8192, 128, 1], 3, K256, 0, 5767168),
        (T::Type2D, P::R32Float, [8192, 128, 1], 4, K256, 0, 5767168),
        (T::Type2D, P::R32Float, [8192, 128, 1], 14, K256, 0, 5767168),
        (T::Type2D, P::R32Float, [4096, 64, 1], 1, K256, 0, 1048576),
        (T::Type2D, P::R32Float, [4096, 64, 1], 2, K256, 0, 1572864),
        (T::Type2D, P::R32Float, [256, 256, 1], 1, K256, 1, 0),
        (T::Type2D, P::R32Float, [256, 256, 1], 2, K256, 1, 262144),
        (T::Type2D, P::R32Float, [256, 256, 1], 3, K256, 1, 262144),
        (T::Type2D, P::R32Float, [256, 256, 1], 4, K256, 1, 262144),
        (T::Type2D, P::R32Float, [128, 128, 1], 2, K256, 0, 262144),
        (T::Type2D, P::R32Float, [128, 128, 1], 3, K256, 0, 262144),
        (T::Type2D, P::R32Float, [256, 128, 1], 2, K256, 0, 262144),
        (T::Type2D, P::R32Float, [64, 64, 1], 2, K256, 0, 262144),
        (T::Type2D, P::R32Float, [16, 16, 1], 2, K256, 0, 262144),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 4, K256, 2, 262144),
        (T::Type2D, P::RGBA8Unorm, [512, 512, 1], 5, K256, 2, 262144),
        (T::Type2DArray, P::R32Float, [256, 256, 1], 3, K256, 1, 262144),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 2, K256, 0, 5767168),
        (T::Type2DArray, P::R32Float, [8192, 128, 1], 3, K256, 0, 5767168),
        (T::Type3D, P::R32Float, [256, 256, 16], 2, K256, 1, 2097152),
        (T::Type3D, P::R32Float, [256, 256, 16], 3, K256, 1, 2097152),
        (T::Type3D, P::R32Float, [256, 256, 16], 4, K256, 1, 2097152),
        (T::Type3D, P::R32Float, [1024, 64, 8], 2, K256, 0, 3145728),
        (T::Type3D, P::R32Float, [1024, 64, 8], 11, K256, 0, 3145728),
        (T::Cube, P::R32Float, [256, 256, 1], 3, K256, 1, 262144),
        (T::Cube, P::R32Float, [256, 256, 1], 2, K256, 1, 262144),
    ];
    for &(texture_type, pf, extent, levels, page, first, tail) in cases {
        let tile = tile_texels(pf, 1, page);
        let what = format!("{texture_type:?} {pf:?} {extent:?} {levels} levels, {page}-byte pages");
        assert_eq!(first_mipmap_in_tail(extent, levels, tile), first, "{what}");
        assert_eq!(
            tail_size_in_bytes(texture_type, pf, 1, extent, levels, first, page),
            tail,
            "{what}"
        );
    }
    // Multisampled textures, of one level (16-KiB pages).
    for (pf, extent) in [(P::R32Float, [256, 256, 1]), (P::RGBA8Unorm, [256, 256, 1])] {
        let tile = tile_texels(pf, 4, K16);
        assert_eq!(first_mipmap_in_tail(extent, 1, tile), 1);
        let ms = T::Type2DMultisample;
        assert_eq!(tail_size_in_bytes(ms, pf, 4, extent, 1, 1, K16), 0);
    }
}

/// **Every tail the sparse-tail cases recorded on Metal**, in 16-KiB pages: each shape that
/// answers a tail of its own, with the `firstMipmapInTail` and the `tailSizeInBytes` Metal gave
/// it.
///
/// The rows come from the committed recordings of `sparse-tails-by-shape`,
/// `sparse-tails-by-texture-type` and `sparse-tails-wide-and-short` (an Apple M3 Max, macOS
/// 27.0.1), one per texture type, format and shape that has a non-zero tail, so the table is the
/// oracle's own answers rather than a restatement of the rule.
///
/// **The thirteen marked rows are what the model used to get wrong**, every one of them short by
/// whole pages: they are the extreme aspect ratios, where a tail holds many levels a few texels
/// wide, and they are the rows that need a level's axes padded to powers of two and a level
/// floored at one tile row. The rest are unchanged by both and are here as the guard: a padding or
/// a floor more eager than Metal's moves one of them and fails here.
#[test]
fn every_recorded_mip_tail_is_metals() {
    use metal2vulkan::metal::PixelFormat as P;
    use metal2vulkan::metal::TextureType as T;
    let cases: &[(T, P, u64, [u64; 3], u64, u64, u64)] = &[
        (T::Cube, P::R32Float, 1, [256, 256, 1], 4, 3, 16384),
        (T::Cube, P::R32Float, 1, [256, 256, 1], 9, 3, 16384),
        (T::Cube, P::RGBA8Unorm, 1, [256, 256, 1], 4, 3, 16384),
        (T::Cube, P::RGBA8Unorm, 1, [256, 256, 1], 9, 3, 16384),
        (T::CubeArray, P::R32Float, 1, [256, 256, 1], 4, 3, 16384),
        (T::CubeArray, P::R32Float, 1, [256, 256, 1], 9, 3, 16384),
        (T::CubeArray, P::RGBA8Unorm, 1, [256, 256, 1], 4, 3, 16384),
        (T::CubeArray, P::RGBA8Unorm, 1, [256, 256, 1], 9, 3, 16384),
        (T::Type2D, P::BGR10XR, 1, [1, 1, 1], 1, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [3, 1000, 1], 1, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [3, 1000, 1], 4, 0, 32768), // padded, or floored
        (T::Type2D, P::BGR10XR, 1, [100, 37, 1], 4, 1, 16384),
        (T::Type2D, P::BGR10XR, 1, [512, 2, 1], 1, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [512, 2, 1], 4, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [1000, 1, 1], 1, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [1000, 1, 1], 4, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [1000, 3, 1], 1, 0, 16384),
        (T::Type2D, P::BGR10XR, 1, [1000, 3, 1], 4, 0, 32768), // padded, or floored
        (T::Type2D, P::BGR10XR, 1, [2048, 4, 1], 1, 0, 32768),
        (T::Type2D, P::BGR10XR, 1, [2048, 4, 1], 4, 0, 49152),
        (T::Type2D, P::BGRA10XR, 1, [1, 1, 1], 1, 0, 16384),
        (T::Type2D, P::BGRA10XR, 1, [3, 1000, 1], 1, 0, 32768),
        (T::Type2D, P::BGRA10XR, 1, [3, 1000, 1], 4, 0, 49152), // padded, or floored
        (T::Type2D, P::BGRA10XR, 1, [100, 37, 1], 4, 2, 16384),
        (T::Type2D, P::BGRA10XR, 1, [512, 2, 1], 1, 0, 16384),
        (T::Type2D, P::BGRA10XR, 1, [512, 2, 1], 4, 0, 16384),
        (T::Type2D, P::BGRA10XR, 1, [1000, 1, 1], 1, 0, 16384),
        (T::Type2D, P::BGRA10XR, 1, [1000, 1, 1], 4, 0, 32768), // padded, or floored
        (T::Type2D, P::BGRA10XR, 1, [1000, 3, 1], 1, 0, 32768),
        (T::Type2D, P::BGRA10XR, 1, [1000, 3, 1], 4, 0, 49152), // padded, or floored
        (T::Type2D, P::BGRA10XR, 1, [2048, 4, 1], 1, 0, 65536),
        (T::Type2D, P::BGRA10XR, 1, [2048, 4, 1], 4, 0, 98304),
        (T::Type2D, P::R32Float, 1, [1, 1, 1], 1, 0, 16384),
        (T::Type2D, P::R32Float, 1, [3, 1000, 1], 1, 0, 16384),
        (T::Type2D, P::R32Float, 1, [3, 1000, 1], 4, 0, 32768), // padded, or floored
        (T::Type2D, P::R32Float, 1, [100, 37, 1], 4, 1, 16384),
        (T::Type2D, P::R32Float, 1, [512, 2, 1], 1, 0, 16384),
        (T::Type2D, P::R32Float, 1, [512, 2, 1], 4, 0, 16384),
        (T::Type2D, P::R32Float, 1, [1000, 1, 1], 1, 0, 16384),
        (T::Type2D, P::R32Float, 1, [1000, 1, 1], 4, 0, 16384),
        (T::Type2D, P::R32Float, 1, [1000, 3, 1], 1, 0, 16384),
        (T::Type2D, P::R32Float, 1, [1000, 3, 1], 4, 0, 32768), // padded, or floored
        (T::Type2D, P::R32Float, 1, [2048, 4, 1], 1, 0, 32768),
        (T::Type2D, P::R32Float, 1, [2048, 4, 1], 4, 0, 49152),
        (T::Type2D, P::R8Unorm, 1, [1, 1, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [3, 1000, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [3, 1000, 1], 4, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [100, 37, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [100, 37, 1], 4, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [512, 2, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [512, 2, 1], 4, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [1000, 1, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [1000, 1, 1], 4, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [1000, 3, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [1000, 3, 1], 4, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [2048, 4, 1], 1, 0, 16384),
        (T::Type2D, P::R8Unorm, 1, [2048, 4, 1], 4, 0, 16384),
        (T::Type2D, P::RGBA32Float, 1, [1, 1, 1], 1, 0, 16384),
        (T::Type2D, P::RGBA32Float, 1, [3, 1000, 1], 1, 0, 65536), // padded, or floored
        (T::Type2D, P::RGBA32Float, 1, [3, 1000, 1], 4, 0, 98304), // padded, or floored
        (T::Type2D, P::RGBA32Float, 1, [100, 37, 1], 4, 2, 16384),
        (T::Type2D, P::RGBA32Float, 1, [512, 2, 1], 1, 0, 16384),
        (T::Type2D, P::RGBA32Float, 1, [512, 2, 1], 4, 0, 32768),
        (T::Type2D, P::RGBA32Float, 1, [1000, 1, 1], 1, 0, 16384),
        (T::Type2D, P::RGBA32Float, 1, [1000, 1, 1], 4, 0, 49152), // padded, or floored
        (T::Type2D, P::RGBA32Float, 1, [1000, 3, 1], 1, 0, 65536), // padded, or floored
        (T::Type2D, P::RGBA32Float, 1, [1000, 3, 1], 4, 0, 98304), // padded, or floored
        (T::Type2D, P::RGBA32Float, 1, [2048, 4, 1], 1, 0, 131072),
        (T::Type2D, P::RGBA32Float, 1, [2048, 4, 1], 4, 0, 196608), // padded, or floored
        (T::Type2D, P::RGBA8Unorm, 1, [256, 256, 1], 4, 3, 16384),
        (T::Type2DArray, P::R32Float, 1, [256, 256, 1], 4, 3, 16384),
        (T::Type2DArray, P::R32Float, 1, [256, 256, 1], 9, 3, 16384),
        (T::Type2DArray, P::RGBA8Unorm, 1, [256, 256, 1], 4, 3, 16384),
        (T::Type2DArray, P::RGBA8Unorm, 1, [256, 256, 1], 9, 3, 16384),
        (T::Type3D, P::R32Float, 1, [256, 256, 64], 4, 3, 131072),
        (T::Type3D, P::R32Float, 1, [256, 256, 64], 9, 3, 131072),
        (T::Type3D, P::RGBA8Unorm, 1, [256, 256, 64], 4, 3, 131072),
        (T::Type3D, P::RGBA8Unorm, 1, [256, 256, 64], 9, 3, 131072),
    ];
    for &(texture_type, pf, samples, extent, levels, first, tail) in cases {
        let tile = tile_texels(pf, samples, K16);
        let what = format!("{texture_type:?} {pf:?} {extent:?} {levels} levels, {samples} samples");
        assert_eq!(first_mipmap_in_tail(extent, levels, tile), first, "{what}");
        assert_eq!(
            tail_size_in_bytes(texture_type, pf, samples, extent, levels, first, K16),
            tail,
            "{what}"
        );
    }
}

/// The conversions an M4 answered (64x64 tiles): outward covers, inward
/// takes the tiles inside with a size that wraps when none is.
#[test]
fn conversions_are_metal_s() {
    use SparseTextureRegionAlignmentMode::{Inward, Outward};
    let tile = [64, 64, 1];
    let cases = [
        (
            Region::new_2d(0, 0, 64, 64),
            Outward,
            Region::new_3d(0, 0, 0, 1, 1, 1),
        ),
        (
            Region::new_2d(10, 70, 100, 30),
            Outward,
            Region::new_3d(0, 1, 0, 2, 1, 1),
        ),
        (
            Region::new_2d(64, 64, 1, 1),
            Outward,
            Region::new_3d(1, 1, 0, 1, 1, 1),
        ),
        (
            Region::new_2d(0, 0, 64, 64),
            Inward,
            Region::new_3d(0, 0, 0, 1, 1, 1),
        ),
        (
            Region::new_2d(10, 70, 100, 30),
            Inward,
            Region::new_3d(1, 2, 0, 0, 288230376151711743, 1),
        ),
        (
            Region::new_2d(64, 64, 1, 1),
            Inward,
            Region::new_3d(1, 1, 0, 0, 0, 1),
        ),
    ];
    for (pixels, mode, tiles) in cases {
        assert_eq!(
            pixel_region_tiles(&pixels, tile, mode),
            tiles,
            "{pixels:?} {mode:?}"
        );
    }
}

/// A tile with an axis of 0 writes nothing into the caller's regions, as plain Metal writes
/// nothing (an M4 under macOS 27.0.1 and an M3 Max under 26.7, the region 5, 7 by 20, 30 into
/// regions filled with `0xab` bytes); a 16 by 16 tile converts it as Metal did.
#[test]
fn a_tile_with_an_axis_of_0_converts_nothing() {
    use SparseTextureRegionAlignmentMode::{Inward, Outward};
    let pixels = [Region::new_2d(5, 7, 20, 30)];
    let filled = 0xabab_abab_abab_abab_u64;
    let held = Region::new_3d(filled, filled, filled, filled, filled, filled);
    for tile in [[0, 16, 1], [16, 0, 1], [16, 16, 0], [0, 0, 0]] {
        for mode in [Outward, Inward] {
            let mut tiles = [held];
            convert_pixel_regions(&pixels, &mut tiles, tile, mode);
            assert_eq!(tiles, [held], "{tile:?} {mode:?}");
        }
    }
    let mut tiles = [held];
    convert_pixel_regions(&pixels, &mut tiles, [16, 16, 1], Outward);
    assert_eq!(tiles, [Region::new_3d(0, 0, 0, 2, 3, 1)]);
    convert_pixel_regions(&pixels, &mut tiles, [16, 16, 1], Inward);
    assert_eq!(tiles, [Region::new_3d(1, 1, 0, 0, 1, 1)]);
}

/// A device's sparse support refuses, in order, what a sparse heap and each image lack.
#[test]
fn sparse_support_names_what_is_missing() {
    let all = vk::PhysicalDeviceFeatures {
        sparse_binding: vk::TRUE,
        sparse_residency_image2_d: vk::TRUE,
        sparse_residency_image3_d: vk::TRUE,
        sparse_residency4_samples: vk::TRUE,
        robust_buffer_access: vk::TRUE,
        ..Default::default()
    };
    let enabled = wanted(&all);
    assert_eq!(enabled.robust_buffer_access, vk::FALSE);
    assert_eq!(enabled.sparse_residency4_samples, vk::TRUE);
    let strict = vk::PhysicalDeviceSparseProperties {
        residency_non_resident_strict: vk::TRUE,
        ..Default::default()
    };
    let queue = vk::QueueFlags::GRAPHICS | vk::QueueFlags::SPARSE_BINDING;
    let device = SparseDevice::of(&enabled, queue, &strict);
    assert!(device.binds() && device.strict);
    assert_eq!(device.heaps(), Ok(()));
    let one = vk::SampleCountFlags::TYPE_1;
    assert_eq!(device.image(vk::ImageType::TYPE_2D, one), Ok(()));
    assert_eq!(device.image(vk::ImageType::TYPE_3D, one), Ok(()));
    assert_eq!(
        device.image(vk::ImageType::TYPE_2D, vk::SampleCountFlags::TYPE_4),
        Ok(())
    );
    assert_eq!(
        device.image(vk::ImageType::TYPE_2D, vk::SampleCountFlags::TYPE_2),
        Err(SparseNeed::SparseResidencySamples(2))
    );
    let no_queue = SparseDevice::of(&enabled, vk::QueueFlags::GRAPHICS, &strict);
    assert_eq!(no_queue.heaps(), Err(SparseNeed::SparseBindingQueue));
    let none = SparseDevice::of(&vk::PhysicalDeviceFeatures::default(), queue, &strict);
    assert_eq!(none.heaps(), Err(SparseNeed::SparseBinding));
    let only_2d = vk::PhysicalDeviceFeatures {
        sparse_residency_image3_d: vk::FALSE,
        ..enabled
    };
    let device = SparseDevice::of(&only_2d, queue, &Default::default());
    assert!(!device.strict);
    assert_eq!(
        device.image(vk::ImageType::TYPE_3D, one),
        Err(SparseNeed::SparseResidencyImage3D)
    );
}

/// A sparse need the runtime has not built is the runtime's gap, and one the device simply does
/// not have is the device's, named by the Vulkan capability it is.
///
/// The two must not be confused in either direction, because a consumer reads the difference and
/// acts on it, and the conformance suite scores it: `Needs::NotRepresentable` used to cover every
/// need past sparse binding, so a device without `sparseResidency2Samples` made our refusal read
/// as unbuilt work of ours. `Needs::DeviceCapability` is the one word that says "the device lacks
/// it" for a capability the compile layer's vocabulary does not name -- which the sparse ones are
/// not, on purpose (`super::super::sparse::device`), since only the runtime needs them.
#[test]
fn a_sparse_need_is_the_runtime_gap_or_the_devices_and_never_the_wrong_one() {
    // Ours: the exact path exists and is not built. Sparse binding and its queue family take the
    // tile map's ordinary images; a metadata aspect is a binding the runtime does not do.
    for need in [
        SparseNeed::SparseBinding,
        SparseNeed::SparseBindingQueue,
        SparseNeed::SparseMetadata,
    ] {
        assert_eq!(
            sparse_refusal(need),
            Error::NotImplemented(RuntimeGap::SparseTileMaps),
            "{need:?}"
        );
    }
    // The device's: a Vulkan 1.0 capability it either has or does not, named as the device would
    // name it, so the refusal reads as the device's gap and not as our unbuilt work.
    for (need, capability) in [
        (SparseNeed::SparseResidencyImage2D, "sparseResidencyImage2D"),
        (SparseNeed::SparseResidencyImage3D, "sparseResidencyImage3D"),
        (
            SparseNeed::SparseResidencySamples(2),
            "sparseResidency2Samples",
        ),
        (
            SparseNeed::SparseResidencySamples(4),
            "sparseResidency4Samples",
        ),
        (
            SparseNeed::SparseResidencySamples(8),
            "sparseResidency8Samples",
        ),
        (
            SparseNeed::SparseResidencySamples(16),
            "sparseResidency16Samples",
        ),
    ] {
        assert_eq!(
            sparse_refusal(need),
            Error::Unsupported(Unsupported::device_capability(
                Feature::SparseTextures { missing: need },
                capability
            )),
            "{need:?}"
        );
    }
    // A need Vulkan has no one member for is still the device's, named in words rather than by a
    // member that does not exist.
    for need in [
        SparseNeed::SparseResidencySamples(32),
        SparseNeed::SparseImageFormat,
    ] {
        let Error::Unsupported(unsupported) = sparse_refusal(need) else {
            panic!("{need:?}");
        };
        let metal2vulkan::error::Needs::DeviceCapability(words) = unsupported.needs else {
            panic!("{need:?}: {:?}", unsupported.needs);
        };
        assert!(words.contains("sparse residency"), "{need:?}: {words}");
    }
}

/// A 2D `R32Float` texture of `extent` and `levels` levels, Metal's 16-KiB tiles (64x64 texels),
/// over a device whose blocks are 64 KiB (128x128 texels) and whose mip tail starts at
/// `vk_tail`.
fn grid(extent: [u64; 3], levels: u64, layers: u32, vk_tail: u32, single: bool) -> UnitGrid {
    let shape = SparseShape {
        texture_type: if layers > 1 {
            TextureType::Type2DArray
        } else {
            TextureType::Type2D
        },
        pixel_format: PixelFormat::R32Float,
        samples: 1,
        extent,
        levels,
    };
    let tile = tile_texels(PixelFormat::R32Float, 1, K16);
    UnitGrid {
        shape,
        layers,
        tile,
        first_mipmap_in_tail: first_mipmap_in_tail(extent, levels, tile),
        layout: VkSparseLayout {
            aspect: vk::ImageAspectFlags::COLOR,
            granularity: [128, 128, 1],
            block_bytes: K64,
            memory_type_bits: 1,
            mip_tail_first_lod: vk_tail,
            mip_tail_size: K64,
            mip_tail_offset: 0,
            mip_tail_stride: K64,
            single_mip_tail: single,
        },
    }
}

fn tile(level: u32, slice: u32, at: [u32; 2]) -> MetalUnit {
    MetalUnit::Tile {
        level,
        slice,
        at: [at[0], at[1], 0],
    }
}

fn block(level: u32, layer: u32, at: [u32; 2]) -> VkUnit {
    VkUnit::Block {
        level,
        layer,
        at: [at[0], at[1], 0],
    }
}

fn texels(level: u32, layer: u32, origin: [u32; 2], size: [u32; 2]) -> TexelBox {
    TexelBox {
        level,
        layer,
        origin: [origin[0], origin[1], 0],
        size: [size[0], size[1], 1],
    }
}

/// Metal's bounds rule for a mapping update, which the direct and plural selectors, the indirect
/// one and the move all share ([`UnitGrid::clipped_region`]): plain Metal **takes** an argument
/// past the texture's own and maps the tiles inside it, where its own API validation layer stops
/// the call.
///
/// Measured by `metal2vulkan-conformance/probes/sparse_mapping_past_the_bounds.m` on an Apple M3
/// Max under macOS 27.0.1, through the direct
/// `updateTextureMapping:mode:region:mipLevel:slice:` over this very shape — a 256 by 256
/// nine-level `R32Float` texture whose level 0 holds 4 by 4 of Metal's 16-KiB tiles — the heap's
/// `usedSize` 0 before each shape and the command buffer `Completed` with no error after. A
/// **slice** past the texture's took one page there: it names its tile at that slice, which the
/// heap counts and nothing holds.
#[test]
fn a_mapping_update_past_the_bounds_clips_as_plain_metal_does() {
    let grid = grid([256, 256, 1], 9, 1, 3, false);
    assert_eq!(grid.first_mipmap_in_tail, 3, "the tail of a 256 by 256");
    let one = Region::new_2d(0, 0, 1, 1);
    // The control, which says the readings below are not all empty by construction: one tile of
    // level 0 maps one tile, where Metal took one 16-KiB page.
    assert_eq!(grid.clipped_region(0, 0, &one), vec![tile(0, 0, [0, 0])]);
    // A region of 8 by 8 tiles where the level holds 4 by 4 maps the level's own tiles and no
    // more, which is what Metal's 16 pages were.
    let whole = grid.clipped_region(0, 0, &Region::new_2d(0, 0, 4, 4));
    assert_eq!(whole.len(), 16, "level 0 holds 4 by 4 tiles");
    assert_eq!(
        grid.clipped_region(0, 0, &Region::new_2d(0, 0, 8, 8)),
        whole
    );
    // A level past the texture's levels and an origin past the level's last tile each map
    // nothing, where Metal's `usedSize` stayed at 0; a slice past its slices names its tile there,
    // past the texture's slices, where Metal took one page.
    assert!(grid.clipped_region(9, 0, &one).is_empty(), "level 9 of 9");
    let past = grid.clipped_region(0, 1, &one);
    assert_eq!(past, vec![tile(0, 1, [0, 0])], "slice 1 of 1");
    assert!(grid.past_slices(past[0]));
    assert!(!grid.past_slices(tile(0, 0, [0, 0])));
    assert!(
        grid.clipped_region(0, 0, &Region::new_2d(4, 4, 1, 1))
            .is_empty(),
        "the tile past (3, 3)"
    );
}

/// **Which tiles a sparse move moves, past the bounds and inside them**, against plain Metal's own
/// readings.
///
/// A move changes no `usedSize`, so nothing a counter can read says any of this; these are the
/// texels either side of the move on an Apple M3 Max under macOS 27.0.1, plain Metal
/// (`AGXG15CDevice`, read back off the device), one shape per process, each tile's verdict off all
/// 4,096 of its texels, over this very shape — a 256 by 256 nine-level texture whose level 0 holds
/// 4 by 4 of Metal's 16-KiB tiles, every tile of the source filled with a word naming itself
/// (`metal2vulkan-conformance/probes/sparse_move_past_the_bounds.m`).
///
/// Each assertion below names its probe shape and what Metal read for it. A destination tile
/// **took** a source tile where the probe read that source tile's word as the destination tile's
/// own **first texel**: the tile is mixed rather than that word throughout, and the probe reads no
/// share of it (its `differing>=` count is guarded on the outer loop, so it is a floor). The first
/// assertion is the control: an entirely in-bounds move whose reading **must** change, or none of
/// the rest is a measurement of anything.
#[test]
fn a_move_is_metals_run_of_destination_tiles() {
    let source_grid = grid([256, 256, 1], 9, 1, 3, false);
    let other = grid([256, 256, 1], 9, 1, 3, false);
    let at = |level: u64, slice: u64, x: u64, y: u64| MoveSide {
        level,
        slice,
        origin: Origin::new(x, y, 0),
    };
    let four_wide = Size::new(4, 1, 1);
    let four = Size::new(4, 4, 1);
    let two = Size::new(2, 2, 1);
    // `control`: one tile at (0, 0) onto (3, 3). B(3, 3) took A(0, 0), read off B(3, 3)'s own
    // first texel, with A(0, 0) reading zero and every other tile of both textures as it was.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 0, 0), Size::new(1, 1, 1), &other, at(0, 0, 3, 3)),
        vec![(tile(0, 0, [0, 0]), tile(0, 0, [3, 3]))]
    );
    // `dest-2d`: 2 by 2 tiles at (0, 0) onto (1, 1), both sides inside the level. B(1,1), B(2,1),
    // B(1,2), B(2,2) took A(0,0), A(1,0), A(0,1), A(1,1), each off the destination tile's own first
    // texel -- the tile's own offset within the region, which is what an ordinary move looks like.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 0, 0), two, &other, at(0, 0, 1, 1)),
        vec![
            (tile(0, 0, [0, 0]), tile(0, 0, [1, 1])),
            (tile(0, 0, [1, 0]), tile(0, 0, [2, 1])),
            (tile(0, 0, [0, 1]), tile(0, 0, [1, 2])),
            (tile(0, 0, [1, 1]), tile(0, 0, [2, 2])),
        ]
    );
    // `region-open`: the SOURCE overhangs -- 4 by 1 tiles at (2, 0) of a level holding 4 by 4 --
    // onto (0, 0). Metal moved the two tiles inside and nothing anywhere for the two outside:
    // B(0,0) and B(1,0) took A(2,0) and A(3,0), each read off the destination tile's own first
    // texel, B(2,0) and B(3,0) read zero, and A's other rows were untouched, so nothing wrapped.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 2, 0), four_wide, &other, at(0, 0, 0, 0)),
        vec![
            (tile(0, 0, [2, 0]), tile(0, 0, [0, 0])),
            (tile(0, 0, [3, 0]), tile(0, 0, [1, 0])),
        ]
    );
    // `region-dest`: the DESTINATION overhangs -- 4 by 1 tiles at (0, 0) onto (2, 0), two tiles
    // past the row. Metal moved ALL FOUR: B(2,0), B(3,0), B(2,1), B(3,1) took A(0,0), A(1,0),
    // A(2,0), A(3,0), each read off the destination tile's own first texel, and A's whole row 0
    // read zero. The destination's width is clipped to the two tiles right of the origin and its
    // height is not, so the run takes the next row.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 0, 0), four_wide, &other, at(0, 0, 2, 0)),
        vec![
            (tile(0, 0, [0, 0]), tile(0, 0, [2, 0])),
            (tile(0, 0, [1, 0]), tile(0, 0, [3, 0])),
            (tile(0, 0, [2, 0]), tile(0, 0, [2, 1])),
            (tile(0, 0, [3, 0]), tile(0, 0, [3, 1])),
        ]
    );
    // `dest-row`: the same overhang a row down, onto (2, 1). Metal read B(2,1), B(3,1), B(2,2),
    // B(3,2), so the run's row is the origin's own column onwards and not the level's left edge.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 0, 0), four_wide, &other, at(0, 0, 2, 1)),
        vec![
            (tile(0, 0, [0, 0]), tile(0, 0, [2, 1])),
            (tile(0, 0, [1, 0]), tile(0, 0, [3, 1])),
            (tile(0, 0, [2, 0]), tile(0, 0, [2, 2])),
            (tile(0, 0, [3, 0]), tile(0, 0, [3, 2])),
        ]
    );
    // `src-clip-2d`: both sides overhang, and the reading that says the run's row is the width the
    // move STATES rather than the clipped source's. 4 by 4 tiles at (2, 2) clip to 2 by 2, onto
    // (1, 1) where three tiles lie to the right: B(1,1), B(2,1), B(3,1), B(1,2) took A(2,2),
    // A(3,2), A(2,3), A(3,3), each read off the destination tile's own first texel. A row of the
    // clipped two would have put the third at (1, 2).
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 2, 2), four, &other, at(0, 0, 1, 1)),
        vec![
            (tile(0, 0, [2, 2]), tile(0, 0, [1, 1])),
            (tile(0, 0, [3, 2]), tile(0, 0, [2, 1])),
            (tile(0, 0, [2, 3]), tile(0, 0, [3, 1])),
            (tile(0, 0, [3, 3]), tile(0, 0, [1, 2])),
        ]
    );
    // `level` and `slice`: a source level or slice past the texture's own moved nothing at all --
    // both textures' every tile as it was and `usedSize` where it started. The destination side of
    // each, and an origin past the level's last tile, move nothing here too; Metal's own reading
    // for those is the update's rather than the move's.
    for (source, into) in [
        (at(9, 0, 0, 0), at(0, 0, 0, 0)),
        (at(0, 0, 0, 0), at(9, 0, 0, 0)),
        (at(0, 1, 0, 0), at(0, 0, 0, 0)),
        (at(0, 0, 0, 0), at(0, 1, 0, 0)),
        (at(0, 0, 4, 4), at(0, 0, 0, 0)),
        (at(0, 0, 0, 0), at(0, 0, 4, 4)),
        (at(4, 0, 0, 0), at(0, 0, 0, 0)),
    ] {
        assert!(
            source_grid.move_pairs(source, two, &other, into).is_empty(),
            "{source:?} -> {into:?}"
        );
    }
    // A size of no tiles on an axis moves nothing, as it maps nothing.
    assert!(source_grid
        .move_pairs(at(0, 0, 0, 0), Size::new(0, 2, 1), &other, at(0, 0, 0, 0))
        .is_empty());
    // The tail is one unit however large the region: a tail onto a tiled level's tile, and a tile
    // onto a whole tail, each one pair, which is what Metal's validation takes and Metal moves.
    let tail = MetalUnit::Tail { slice: 0 };
    assert_eq!(
        source_grid.move_pairs(at(3, 0, 0, 0), four, &other, at(0, 0, 1, 1)),
        vec![(tail, tile(0, 0, [1, 1]))]
    );
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 1, 1), four, &other, at(3, 0, 0, 0)),
        vec![(tile(0, 0, [1, 1]), tail)]
    );
    // A run that would pass the level's last row drops the tiles past it: unmeasured, so nothing
    // moves there rather than moving somewhere nobody has read. 4 by 1 tiles onto (2, 3), the
    // level's last row, places two and drops two.
    assert_eq!(
        source_grid.move_pairs(at(0, 0, 0, 0), four_wide, &other, at(0, 0, 2, 3)),
        vec![
            (tile(0, 0, [0, 0]), tile(0, 0, [2, 3])),
            (tile(0, 0, [1, 0]), tile(0, 0, [3, 3])),
        ]
    );
}

/// Four of Metal's 16-KiB tiles lie in one 64-KiB block, which holds memory while any of them is
/// mapped: a tile maps alone, and the block is freed with its last.
#[test]
fn a_block_is_bound_while_any_of_its_tiles_is_mapped() {
    let grid = grid([256, 256, 1], 1, 1, 1, false);
    let quad = [[0, 0], [1, 0], [0, 1], [1, 1]].map(|at| tile(0, 0, at));
    for unit in quad {
        assert_eq!(grid.vk_units_of(unit), [block(0, 0, [0, 0])]);
    }
    assert_eq!(grid.metal_units_of(block(0, 0, [0, 0])), quad);
    assert_eq!(grid.vk_units_of(tile(0, 0, [2, 3])), [block(0, 0, [1, 1])]);
    let mut mapped = HashSet::new();
    assert!(!grid.wanted(block(0, 0, [0, 0]), &mapped));
    mapped.insert(tile(0, 0, [1, 0]));
    assert!(grid.wanted(block(0, 0, [0, 0]), &mapped));
    assert!(!grid.wanted(block(0, 0, [1, 0]), &mapped));
    mapped.insert(tile(0, 0, [0, 1]));
    mapped.remove(&tile(0, 0, [1, 0]));
    assert!(grid.wanted(block(0, 0, [0, 0]), &mapped));
    mapped.clear();
    assert!(!grid.wanted(block(0, 0, [0, 0]), &mapped));
}

/// A block bound to new memory is zeroed whole; an unmapped tile is zeroed inside the blocks
/// that hold memory, and nowhere else; a tile at a level's edge holds its level's texels alone.
#[test]
fn the_texels_zeroed_are_the_new_blocks_and_the_unmapped_tiles_in_held_ones() {
    let grid = grid([256, 256, 1], 1, 1, 1, false);
    let bound = BTreeSet::from([block(0, 0, [1, 0])]);
    assert_eq!(
        grid.bound_boxes(&bound),
        BTreeSet::from([texels(0, 0, [128, 0], [128, 128])])
    );
    let unmapped = [tile(0, 0, [0, 0]), tile(0, 0, [3, 3])];
    let held = |unit: VkUnit| unit == block(0, 0, [0, 0]);
    assert_eq!(
        grid.unmapped_boxes(unmapped, held),
        BTreeSet::from([texels(0, 0, [0, 0], [64, 64])])
    );
    assert!(grid.unmapped_boxes(unmapped, |_| false).is_empty());
    // 100x100: the edge tile is 36x36, inside the one block.
    let edge = self::grid([100, 100, 1], 1, 1, 1, false);
    assert_eq!(
        edge.metal_boxes(tile(0, 0, [1, 1])),
        [texels(0, 0, [64, 64], [36, 36])]
    );
    assert_eq!(
        edge.vk_boxes(block(0, 0, [0, 0])),
        [texels(0, 0, [0, 0], [100, 100])]
    );
    assert_eq!(
        edge.unmapped_boxes([tile(0, 0, [1, 1])], |_| true),
        BTreeSet::from([texels(0, 0, [64, 64], [36, 36])])
    );
}

/// A move copies a tile's texels to its destination tile, level by level for a tail; between
/// textures of other formats it copies nothing (the destination reads zero).
#[test]
fn a_move_copies_its_tile_s_texels() {
    let grid = grid([256, 256, 1], 1, 1, 1, false);
    assert_eq!(
        grid.move_boxes(tile(0, 0, [0, 0]), &grid, tile(0, 0, [3, 2])),
        [(
            texels(0, 0, [0, 0], [64, 64]),
            texels(0, 0, [192, 128], [64, 64])
        )]
    );
    let edge = self::grid([100, 100, 1], 1, 1, 1, false);
    assert_eq!(
        grid.move_boxes(tile(0, 0, [0, 0]), &edge, tile(0, 0, [1, 1])),
        [(
            texels(0, 0, [0, 0], [36, 36]),
            texels(0, 0, [64, 64], [36, 36])
        )]
    );
    let mut other = grid;
    other.shape.pixel_format = PixelFormat::RGBA8Unorm;
    assert!(grid
        .move_boxes(tile(0, 0, [0, 0]), &other, tile(0, 0, [0, 0]))
        .is_empty());
    // A tail moves each of its levels (256x256 of 9 levels: Metal's tail from level 3).
    let mips = self::grid([256, 256, 1], 9, 2, 2, false);
    let tail = mips.move_boxes(
        MetalUnit::Tail { slice: 0 },
        &mips,
        MetalUnit::Tail { slice: 1 },
    );
    assert_eq!(tail.len(), 6);
    assert_eq!(
        tail[0],
        (
            texels(3, 0, [0, 0], [32, 32]),
            texels(3, 1, [0, 0], [32, 32])
        )
    );
}

/// The device's mip tail holds Metal's tiles of the levels in it and Metal's tail, per layer or,
/// for one tail of every layer (`SINGLE_MIPTAIL`), of every slice.
#[test]
fn the_device_s_tail_holds_metal_s_tiles_and_tail() {
    let mips = grid([256, 256, 1], 9, 2, 2, false);
    assert_eq!(mips.first_tail(), 3);
    assert_eq!(
        mips.vk_units_of(tile(2, 1, [0, 0])),
        [VkUnit::Tail { layer: 1 }]
    );
    assert_eq!(
        mips.vk_units_of(MetalUnit::Tail { slice: 1 }),
        [VkUnit::Tail { layer: 1 }]
    );
    assert_eq!(
        mips.metal_units_of(VkUnit::Tail { layer: 1 }),
        [tile(2, 1, [0, 0]), MetalUnit::Tail { slice: 1 }]
    );
    let single = grid([256, 256, 1], 9, 2, 2, true);
    assert_eq!(
        single.vk_units_of(MetalUnit::Tail { slice: 1 }),
        [VkUnit::Tail { layer: 0 }]
    );
    assert_eq!(
        single.metal_units_of(VkUnit::Tail { layer: 0 }),
        [
            tile(2, 0, [0, 0]),
            MetalUnit::Tail { slice: 0 },
            tile(2, 1, [0, 0]),
            MetalUnit::Tail { slice: 1 },
        ]
    );
    // Slice 1's tail mapped keeps the shared tail bound; slice 0's unmapped tail is zeroed in it.
    let mapped = HashSet::from([MetalUnit::Tail { slice: 1 }]);
    assert!(single.wanted(VkUnit::Tail { layer: 0 }, &mapped));
    let zeroed = single.unmapped_boxes([MetalUnit::Tail { slice: 0 }], |_| true);
    assert_eq!(zeroed.len(), 6);
    assert!(zeroed.iter().all(|b| b.layer == 0 && b.level >= 3));
}

/// A box meets another in their common texels of one level and layer, and splits into pieces of
/// at most a tile from its origin.
#[test]
fn texel_boxes_intersect_and_split() {
    let a = texels(0, 0, [0, 0], [128, 128]);
    assert_eq!(
        a.intersect(&texels(0, 0, [64, 96], [128, 128])),
        Some(texels(0, 0, [64, 96], [64, 32]))
    );
    assert_eq!(a.intersect(&texels(0, 1, [0, 0], [128, 128])), None);
    assert_eq!(a.intersect(&texels(0, 0, [128, 0], [8, 8])), None);
    let pieces = texels(0, 0, [0, 0], [100, 70]).pieces([64, 64, 1]);
    assert_eq!(
        pieces,
        [
            texels(0, 0, [0, 0], [64, 64]),
            texels(0, 0, [64, 0], [36, 64]),
            texels(0, 0, [0, 64], [64, 6]),
            texels(0, 0, [64, 64], [36, 6]),
        ]
    );
}

/// **The slice is the one axis plain Metal does not bound in a mapping update, and what that
/// costs the heap's count** — against Metal's own `usedSize` readings for the very same argument
/// lists.
///
/// `encoder.rs` and `clipped_region` had carried three claims about the INDIRECT selector's count
/// for a region whose slice is past the texture's — a tile more beside other regions, none alone,
/// and an unmap of it taking the count below zero — and no probe in the tree ran the indirect
/// selector at all. **All three are false.** The readings are
/// `metal2vulkan-conformance/probes/sparse_indirect_usedsize.m` on an Apple M3 Max (the name read
/// back off the device) under macOS 27.0.1, plain Metal with the layer off, one shape per process,
/// `usedSize` read twice at each step and every shape run twice, over a 2-slice 256 by 256
/// nine-level `R32Float` texture whose level 0 holds 4 by 4 of Metal's 16-KiB tiles.
///
/// Metal's rule is the plain one: it bounds the level and the origin and **does not bound the
/// slice**, charging a region of a slice past the texture's exactly what the same region on an
/// in-bounds slice costs — the region clipped to the level's tiles — and that charge then behaves
/// like any mapping, idempotent and given back on an unmap. The runtime's heap counts the same
/// tiles, which it maps nowhere (`UnitGrid::past_slices`): `metal` below is what Metal read, and
/// the runtime counts the units the list names.
///
/// Each row is a probe shape. The in-bounds rows are the controls: their counts **must** move, or
/// no row here is a measurement of anything.
#[test]
fn a_slice_past_the_textures_is_the_one_axis_metal_does_not_bound() {
    // The probe's texture, as a 2-slice array, so slice 2 is past its own slices.
    let array = grid([256, 256, 1], 9, 2, 3, false);
    assert_eq!(array.layers, 2, "the probe's texture holds two slices");
    assert_eq!(array.first_mipmap_in_tail, 3, "the tail of a 256 by 256");
    let one = |x: u64, y: u64| Region::new_2d(x, y, 1, 1);

    // The tiles one indirect argument list states, in `indirect::resolve`'s own order: each
    // region clipped, all of them one update. Every unit of a tiled level is one tile of the
    // heap's count, so the list's length is the pages the runtime counts for it.
    let ours = |regions: &[(u64, u64, Region)]| {
        regions
            .iter()
            .flat_map(|(level, slice, region)| array.clipped_region(*level, *slice, region))
            .count()
    };

    // shape, the argument list, Metal's pages for it.
    let rows: &[(&str, &[(u64, u64, Region)], usize)] = &[
        // The controls, through the indirect selector: one in-bounds tile is one page, two are
        // two, so the reading follows the list rather than being fixed by construction.
        ("indirect-control", &[(0, 0, one(0, 0))], 1),
        (
            "indirect-two-inbounds",
            &[(0, 0, one(0, 0)), (0, 0, one(1, 0))],
            2,
        ),
        // 1. ALONE. Metal charges one page for the out-of-bounds slice; the direct selector read
        //    the same one page for it (`direct-slice-alone`, which is the committed reading off
        //    `sparse_mapping_past_the_bounds.m`). So "none alone" is false, for either selector.
        ("indirect-slice-alone", &[(0, 2, one(0, 0))], 1),
        // 2. BESIDE ANOTHER REGION. Two pages, where the in-bounds region alone is one: the
        //    out-of-bounds region added exactly the one page it adds alone, never a tile more.
        (
            "indirect-slice-beside",
            &[(0, 0, one(0, 0)), (0, 2, one(0, 0))],
            2,
        ),
        // And with four pages already mapped it went 4 -> 5, again exactly one, so the charge is
        // the same in every configuration rather than varying with what else is mapped.
        ("indirect-slice-after-four", &[(0, 2, one(0, 0))], 1),
        // The charge follows the REGION, clipped to the level: 8 by 8 tiles on a level holding
        // 4 by 4 cost 16 pages on the out-of-bounds slice, as they do on an in-bounds one.
        (
            "indirect-slice-big-region",
            &[(0, 2, Region::new_2d(0, 0, 8, 8))],
            16,
        ),
        // One page per out-of-bounds region, not one per update: slices 2 and 3 cost two.
        (
            "indirect-two-oob-slices",
            &[(0, 2, one(0, 0)), (0, 3, one(0, 0))],
            2,
        ),
        // The level and the origin ARE bounded, and cost nothing — which is what makes the slice
        // the one unbounded axis rather than a general "Metal does not check".
        ("indirect-level-alone", &[(9, 0, one(0, 0))], 0),
        ("indirect-origin-alone", &[(0, 0, one(4, 4))], 0),
    ];
    for (shape, regions, metal) in rows {
        assert_eq!(ours(regions), *metal, "what the runtime counts for {shape}");
    }

    // 3. AN UNMAP OF IT, in every configuration measured. Metal's charge is reversible and
    //    floors: an unmap of the out-of-bounds region takes off what that region holds and
    //    nothing more, so the count never passed below where it started.
    //    `indirect-unmap-slice-alone` 0 -> 0; `-after-one` 1 -> 1; `-after-four` 4 -> 4 (the
    //    configuration another reading had already found did not lower the count);
    //    `indirect-map-then-unmap-slice` 0 -> 1 -> 0, the page given back exactly;
    //    `indirect-slice-unmap-twice` 0 -> 1 -> 0 -> 0, a second unmap taking nothing more;
    //    `indirect-slice-twice` 0 -> 1 -> 1, the charge idempotent per region.
    //    So "below zero" is false. The runtime gives a unit's page back only where its map took
    //    it (`SparseState::charged`, `sparse/bind.rs`'s `apply`), and `SparseHeap::give`
    //    saturates besides.
    let past = array.clipped_region(0, 2, &one(0, 0));
    assert_eq!(past, vec![tile(0, 2, [0, 0])]);
    assert!(
        array.past_slices(past[0]),
        "slice 2 of 2 is past the texture's"
    );

    // A slice past a 1-slice texture's own, which is the shape the suite's non-array sparse
    // cases use, clips the same way.
    let plain = grid([256, 256, 1], 9, 1, 3, false);
    assert_eq!(
        plain.clipped_region(0, 1, &one(0, 0)),
        vec![tile(0, 1, [0, 0])],
        "slice 1 of 1"
    );
}

/// **A slice past the texture's on its other levels and its tail costs what the same region
/// inside it costs**, as on level 0 above, against plain Metal's own `usedSize` readings.
///
/// The readings are `metal2vulkan-conformance/probes/sparse_slice_past_levels_usedsize.m` on an
/// Apple M3 Max (the name read back off the device) under macOS 27.0.1, plain Metal with the layer
/// off, one shape a process and every shape run twice, each command buffer `Completed` with no
/// error, through all three forms a mapping update takes: the direct selector, the indirect one,
/// and the direct one through a view of the texture's slice 1, which names the texture's slice
/// 2. The three forms read alike in every shape and run. The texture is the probe's 2-slice 256
/// by 256 nine-level `R32Float` texture, whose level 1 holds 2 by 2 of Metal's 16-KiB tiles,
/// level 2 one, and whose tail starts at level 3 (`firstMipmapInTail`) and is one page a slice
/// (`tailSizeInBytes` 16384).
///
/// Each row is a probe shape on slice 2, past the texture's, with its control: the same region on
/// an in-bounds slice, read after it, whose count **must** move where the rule maps anything, or
/// the row measured nothing. The runtime counts the units [`UnitGrid::clipped_region`] names,
/// a tile one page and a slice's tail its tail's pages (`SparseImage::heap_tiles`), and maps
/// those past the slices nowhere (`SparseState::charged`).
#[test]
fn a_slice_past_the_textures_costs_every_level_and_the_tail_what_it_costs_inside() {
    let array = grid([256, 256, 1], 9, 2, 3, false);
    assert_eq!(
        array.first_mipmap_in_tail, 3,
        "the probe's firstMipmapInTail"
    );
    let tail_bytes = tail_size_in_bytes(
        TextureType::Type2DArray,
        PixelFormat::R32Float,
        1,
        [256, 256, 1],
        9,
        3,
        K16,
    );
    assert_eq!(tail_bytes, K16, "the probe's tailSizeInBytes");
    let pages = |units: &[MetalUnit]| {
        (units.iter())
            .map(|unit| match unit.is_tail() {
                true => tail_bytes / K16,
                false => 1,
            })
            .sum::<u64>()
    };
    let one = Region::new_2d(0, 0, 1, 1);
    // shape, level, region, Metal's pages for slice 2 and for its control on slice 0.
    let rows = [
        // One tile of level 1, and a region of 8 by 8 tiles where the level holds 2 by 2: the
        // region clipped to the level's tiles, one page each.
        ("l1", 1, one, 1, 1),
        ("l1-big", 1, Region::new_2d(0, 0, 8, 8), 4, 4),
        // Level 2's one tile.
        ("l2", 2, one, 1, 1),
        // A tile of level 3, `firstMipmapInTail`: the slice's whole tail, one page here.
        ("tail", 3, one, 1, 1),
        // Level 5, inside the tail past its first level: nothing, slice 2's and the control's
        // alike.
        ("tail-inner", 5, one, 0, 0),
    ];
    for (shape, level, region, metal, control) in rows {
        let past = array.clipped_region(level, 2, &region);
        assert_eq!(pages(&past), metal, "{shape}: slice 2 of 2");
        assert!(
            past.iter().all(|&unit| array.past_slices(unit)),
            "{shape}: every unit past the texture's slices"
        );
        let inside = array.clipped_region(level, 0, &region);
        assert_eq!(pages(&inside), control, "{shape}: the control on slice 0");
        assert!(
            !inside.iter().any(|&unit| array.past_slices(unit)),
            "{shape}"
        );
    }
    assert!(
        array
            .clipped_region(3, 2, &one)
            .iter()
            .all(|unit| unit.is_tail()),
        "level 3's tile is the slice's tail"
    );
    // `l1-unmap`: a map of level 1's tile on slice 2 took one page and an unmap of it gave the
    // page back (16384 -> 0), as the control's did (16384 -> 0) and as level 0's did through the
    // indirect selector above. The runtime gives back a unit's page only where its map took it
    // (`sparse/bind.rs`), and an unmap names the same unit the map did.
    assert_eq!(
        array.clipped_region(1, 2, &one),
        vec![tile(1, 2, [0, 0])],
        "the unit the map and the unmap name"
    );
}
