//! The texture object's pure parts: image aspects, Metal regions in Vulkan's counts, the copy
//! regions of CPU copies, and the device's texture state.

use super::super::copy_plan::ImageRegion;
use super::super::texture::{
    buffer_image_copy, image_aspects, narrow_range, texture_region, TextureState,
};
use super::super::values::Region;
use super::image_plan::committed;
use ash::vk;
use metal2vulkan::vk::{Format, ImageAspectFlags};

#[test]
fn images_carry_their_formats_aspects() {
    use ImageAspectFlags as A;
    assert_eq!(image_aspects(Format::R8G8B8A8_UNORM), A::COLOR);
    assert_eq!(image_aspects(Format::BC1_RGBA_UNORM_BLOCK), A::COLOR);
    assert_eq!(image_aspects(Format::D32_SFLOAT), A::DEPTH);
    assert_eq!(image_aspects(Format::S8_UINT), A::STENCIL);
    assert_eq!(
        image_aspects(Format::D32_SFLOAT_S8_UINT),
        A::DEPTH | A::STENCIL
    );
}

#[test]
fn metal_regions_narrow_to_vulkans_counts_or_refuse() {
    let r = texture_region(&Region::new_3d(1, 2, 3, 4, 5, 6), 7, 8).unwrap();
    assert_eq!(
        (r.level, r.slice, r.origin, r.size),
        (7, 8, [1, 2, 3], [4, 5, 6])
    );
    assert!(texture_region(&Region::new_2d(1 << 32, 0, 1, 1), 0, 0).is_none());
    assert!(texture_region(&Region::new_1d(0, 1), 0, 1 << 40).is_none());
    assert_eq!(narrow_range(&(2..5)), Some(2..5));
    assert_eq!(narrow_range(&(0..(1 << 33))), None);
}

#[test]
fn cpu_copies_read_a_tightly_packed_buffer() {
    let copy = buffer_image_copy(&ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 2,
        layer: 5,
        layers: 1,
        origin: [1, 2, 0],
        size: [3, 4, 1],
    });
    assert_eq!(
        (
            copy.buffer_offset,
            copy.buffer_row_length,
            copy.buffer_image_height
        ),
        (0, 0, 0)
    );
    let s = copy.image_subresource;
    assert_eq!(
        (
            s.aspect_mask.as_raw(),
            s.mip_level,
            s.base_array_layer,
            s.layer_count
        ),
        (vk::ImageAspectFlags::COLOR.as_raw(), 2, 5, 1)
    );
    assert_eq!(
        (
            copy.image_offset.x,
            copy.image_offset.y,
            copy.image_offset.z
        ),
        (1, 2, 0)
    );
    assert_eq!(
        (
            copy.image_extent.width,
            copy.image_extent.height,
            copy.image_extent.depth
        ),
        (3, 4, 1)
    );
}

#[test]
fn the_granularity_is_a_whole_power_of_two() {
    let (_, desc, _) = committed().into_iter().next().unwrap();
    assert_eq!(TextureState::new(&desc, 0, false, false).granularity(), 1);
    assert_eq!(
        TextureState::new(&desc, 1024, false, false).granularity(),
        1024
    );
    assert_eq!(
        TextureState::new(&desc, 1000, false, false).granularity(),
        1024
    );
}

#[test]
fn copy_units_are_the_images_aspects_blocks() {
    use super::super::texture::copy_units;
    use metal2vulkan::metal::PixelFormat;
    use ImageAspectFlags as A;
    assert_eq!(
        copy_units(Format::R8G8B8A8_UNORM, PixelFormat::RGBA8Unorm),
        vec![(A::COLOR, 4, [1, 1])]
    );
    assert_eq!(
        copy_units(Format::BC1_RGBA_UNORM_BLOCK, PixelFormat::BC1RGBA),
        vec![(A::COLOR, 8, [4, 4])]
    );
    assert_eq!(
        copy_units(
            Format::D32_SFLOAT_S8_UINT,
            PixelFormat::Depth32FloatStencil8
        ),
        vec![(A::DEPTH, 4, [1, 1]), (A::STENCIL, 1, [1, 1])]
    );
    assert_eq!(
        copy_units(Format::D16_UNORM, PixelFormat::Depth16Unorm),
        vec![(A::DEPTH, 2, [1, 1])]
    );
}

/// Host copies reach a block format's image only where the driver's copies of such an image are
/// its own: the usage puts the image in host-visible memory, and MoltenVK copies the whole image
/// out of that memory as the memory is bound, through a pitch its own check then rejects
/// (`RuntimeDriverDefect::HostTransferImagesCopiedAtBind`; `texture/host_copy.rs` and
/// `driver_defects.rs` carry the backtrace). Every other driver keeps the fast path.
///
/// The row the image's own format gives is a row of **blocks**, which is the pitch that driver
/// calls short: the bytes a row of a 16 x 16 `BC1_RGBA` level holds are its 4 blocks of 8 bytes,
/// 32, not its 16 texels of 8, 128. The second half of this test holds the copy to that row, so a
/// pitch that ever came out of the texels instead would fail here with no device.
#[test]
fn host_copies_reach_a_block_format_only_off_the_bind_copy_defect() {
    use super::super::texture::host_copy::reaches_format as reaches;
    use super::super::texture::{copy_units, zero_copies};
    use metal2vulkan::error::RuntimeDriverDefect as D;
    use metal2vulkan::metal::PixelFormat as P;
    let none = std::collections::BTreeSet::new();
    // A driver with the defect, and one with another defect, so the test fails if the set is
    // consulted for the wrong member.
    let bind: std::collections::BTreeSet<D> = [D::HostTransferImagesCopiedAtBind].into();
    let other: std::collections::BTreeSet<D> = [D::LazilyAllocatedSubresourceViews].into();
    // Every format whose block holds one texel, including the uncompressed carriers that hold a
    // compressed format's blocks in their texels: the image is the carrier's, and so is its row.
    for format in [
        Format::R8_UNORM,
        Format::R8G8B8A8_UNORM,
        Format::B8G8R8A8_SRGB,
        Format::R32G32B32A32_UINT,
        Format::R16G16B16A16_SFLOAT,
        Format::D16_UNORM,
        Format::D32_SFLOAT,
        Format::S8_UINT,
        Format::D32_SFLOAT_S8_UINT,
        Format::A2B10G10R10_UNORM_PACK32,
        // The registry records the IMG PVRTC formats as one texel in a block of 8 bytes, which no
        // other compressed format is, so the rule here lets them through. Nothing reaches it: a
        // PVRTC texture's image never takes host-transfer usage (`Device::new_image_starting`
        // leaves it out by `is_pvrtc`), and on a device without the format its blocks lie in an
        // uncompressed carrier whose own block is one texel.
        Format::PVRTC1_4BPP_UNORM_BLOCK_IMG,
        Format::PVRTC2_2BPP_SRGB_BLOCK_IMG,
    ] {
        let block = format.block().expect("an image format has a block");
        assert_eq!(block.texels, 1, "{format:?}");
        for defects in [&none, &bind, &other] {
            assert!(reaches(format, defects), "{format:?}");
        }
    }
    // Every format whose block holds more than one texel, by family: BC, BC6H and BC7, ETC2 and
    // EAC, ASTC. Each reaches host copies but on the one driver that copies such an image out of
    // its memory as it is bound.
    for format in [
        Format::BC1_RGBA_UNORM_BLOCK,
        Format::BC1_RGBA_SRGB_BLOCK,
        Format::BC2_UNORM_BLOCK,
        Format::BC3_SRGB_BLOCK,
        Format::BC4_UNORM_BLOCK,
        Format::BC4_SNORM_BLOCK,
        Format::BC5_UNORM_BLOCK,
        Format::BC5_SNORM_BLOCK,
        Format::BC6H_SFLOAT_BLOCK,
        Format::BC7_UNORM_BLOCK,
        Format::ETC2_R8G8B8_UNORM_BLOCK,
        Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        Format::EAC_R11_UNORM_BLOCK,
        Format::EAC_R11G11_SNORM_BLOCK,
        Format::ASTC_4X4_UNORM_BLOCK,
        Format::ASTC_12X12_SRGB_BLOCK,
    ] {
        let block = format.block().expect("an image format has a block");
        assert!(block.texels > 1, "{format:?}");
        assert!(reaches(format, &none), "{format:?}");
        assert!(reaches(format, &other), "{format:?}");
        assert!(!reaches(format, &bind), "{format:?}");
    }
    // A format no image has takes no host copies on any driver.
    for defects in [&none, &bind, &other] {
        assert!(!reaches(Format::UNDEFINED, defects));
    }
    // The row of a block format's level is its blocks, not its texels: the copy a host write of a
    // 16 x 16 `BC1_RGBA` level reads is 4 blocks of 8 bytes a row, and the whole level 128 bytes.
    let units = copy_units(Format::BC1_RGBA_UNORM_BLOCK, P::BC1RGBA);
    assert_eq!(units, vec![(ImageAspectFlags::COLOR, 8, [4, 4])]);
    let (length, copies) = zero_copies([16, 16, 1], 1, 1, false, &units, 4 << 20);
    assert_eq!(copies.len(), 1);
    assert_eq!(copies[0].size, [16, 16, 1]);
    let row_blocks = 16u64.div_ceil(u64::from(units[0].2[0]));
    assert_eq!(row_blocks * u64::from(units[0].1), 32);
    assert_eq!(length, 32 * 4);
    assert_ne!(length, 16 * 8 * 16);
    // The copy is tightly packed, so the row length Vulkan is given is the level's own.
    let copy = buffer_image_copy(&copies[0]);
    assert_eq!((copy.buffer_row_length, copy.buffer_image_height), (0, 0));
}

#[test]
fn zero_copies_cover_every_texel_once_in_bands() {
    use super::super::texture::zero_copies;
    // A 2D array of 3 layers, 9x7 with 3 levels, of 4-byte texels, in bands of at most 2 rows
    // (2 rows of 36 bytes fit 80 bytes, 3 do not).
    let unit = [(ImageAspectFlags::COLOR, 4, [1, 1])];
    let (length, copies) = zero_copies([9, 7, 1], 3, 3, false, &unit, 80);
    assert_eq!(length, 72);
    let mut covered = std::collections::BTreeMap::new();
    for c in &copies {
        assert_eq!(
            (c.layers, c.origin[0], c.origin[2], c.size[2]),
            (1, 0, 0, 1)
        );
        let [w, h, _] = [9u32, 7, 1].map(|a| (a >> c.level).max(1));
        assert_eq!(c.size[0], w);
        assert!(u64::from(c.size[0]) * 4 * u64::from(c.size[1]) <= length);
        for y in c.origin[1]..c.origin[1] + c.size[1] {
            assert!(y < h);
            *covered.entry((c.level, c.layer, y)).or_insert(0) += 1;
        }
    }
    let rows: u32 = (0..3).map(|l| (7u32 >> l).max(1)).sum();
    assert_eq!(covered.len() as u32, rows * 3);
    assert!(covered.values().all(|&n| n == 1));
    // Compressed blocks: bands of block rows, a partial last block row reaching the edge.
    let bc = [(ImageAspectFlags::COLOR, 8, [4, 4])];
    let (length, copies) = zero_copies([10, 10, 1], 1, 1, false, &bc, 24);
    assert_eq!(length, 24);
    let bands: Vec<(u32, u32)> = copies.iter().map(|c| (c.origin[1], c.size[1])).collect();
    assert_eq!(bands, vec![(0, 4), (4, 4), (8, 2)]);
    // A 3D level: each depth slice; a row larger than the budget is one band of one row.
    let (length, copies) = zero_copies([64, 2, 3], 1, 1, true, &unit, 16);
    assert_eq!(length, 256);
    assert_eq!(copies.len(), 6);
    assert!(copies.iter().all(|c| c.size[1] == 1));
}
