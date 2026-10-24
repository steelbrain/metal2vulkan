//! `sparse`: Metal's sparse answers that need no device, each against what an M4 answered: tiles,
//! mip tails, conversions, the page values, and what a device's sparse support refuses; and how
//! Metal's units lie in the device's, which texels a batch zeroes and a move copies.

use crate::error::{Error, RuntimeGap};
use crate::sparse::device::{wanted, SparseDevice};
use crate::sparse::mapping::{MetalUnit, TexelBox, UnitGrid, VkUnit};
use crate::sparse::texture::{sparse_refusal, SparseShape, VkSparseLayout};
use crate::sparse::{
    convert_pixel_regions, first_mipmap_in_tail, pixel_region_tiles, tail_size_in_bytes,
    tile_texels, SparsePageSize, SparseTextureRegionAlignmentMode,
};
use crate::{HeapDescriptor, Region};
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
/// `usedSize` 0 before each shape and the command buffer `Completed` with no error after. The one
/// reading that is not this rule's is a **slice** past the texture's, which took one page there and
/// maps none here.
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
    // A level past the texture's levels, a slice past its slices, and an origin past the level's
    // last tile each map nothing, where Metal's `usedSize` stayed at 0.
    assert!(grid.clipped_region(9, 0, &one).is_empty(), "level 9 of 9");
    assert!(grid.clipped_region(0, 1, &one).is_empty(), "slice 1 of 1");
    assert!(
        grid.clipped_region(0, 0, &Region::new_2d(4, 4, 1, 1))
            .is_empty(),
        "the tile past (3, 3)"
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
