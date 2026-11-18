//! Blit copies between textures, and between buffers and textures; buffer fills; synchronizes.
//!
//! - The rules: a texture copies into one of its own pixel format and sample count; two formats
//!   or two sample counts are `Refusal::TextureCopyMismatch`, as Metal's validation refuses them;
//!   what one copy command in the floor does not make, one pixel format held in two Vulkan formats
//!   among it, is `Gap::TextureCopy`; a level, slice or region outside a texture or
//!   off its block grid is `Refusal::TextureRegion`; a size of 0 on an axis a texture lacks copies
//!   one texel there, and on another copies nothing; a 3D texture's image copies to and from a 2D
//!   slice.
//! - Under a device cfg, on every device of the cfg's kind: the bytes of a level of a 2D texture
//!   land in a slice of a 2D array, a cube's face in an
//!   image of a 3D texture, an image of a 3D texture in a 2D texture, a slice of an array in
//!   another slice of it, and a block-compressed region in another texture where the device makes
//!   one, everything around each region kept. One copy alone is one submission, two barriers and
//!   the segment's end, and copies no byte of the runtime's own; a barrier an encoder owed is paid
//!   by the copy's first. Every gauge comes back to zero with the device. The bytes each copy
//!   must leave are plain Metal's: their FNV-1a 64 hashes are those of probe `blit_copy_layouts.m`
//!   in the conformance crate (an Apple M3 Max, macOS 27.0.1, job m3-max-43464, plain and under
//!   the validation layer).
//! - Between a buffer and a texture: each image of the region lies at its own offset, its rows a
//!   count of texels apart; what Metal's validation stops is `Refusal::BufferCopyLayout`, and an
//!   offset or row stride a copy command cannot address `Gap::BufferCopyAlignment`. Under a device
//!   cfg: bytes from a buffer land in a mipmap level, a slice of a 1D array and BC1 blocks, and a
//!   slice's and a 3D region's bytes land in buffers at padded strides, everything around each
//!   region kept and each result plain Metal's (job m3-max-43555); one copy into a Shared buffer
//!   alone is one submission and three barriers, the segment's end reaching the host, and copies
//!   no byte of the runtime's own.
//! - A copy between buffers takes ranges inside both buffers (`Refusal::BufferCopyRange`), both
//!   offsets and the size multiples of 4 (`Refusal::BufferCopyUnaligned`), and a source and a
//!   destination that share no byte (`Refusal::BufferCopyOverlap`), one buffer's ranges or two
//!   buffers' in one block; a size of 0 copies nothing. Under a device cfg its bytes land where
//!   it names, between two buffers of one block and within one buffer, everything around them
//!   kept, at one submission and three barriers alone, copying no byte of the runtime's own; a
//!   copy into a Managed buffer lands in the bytes shaders reach, which a synchronize brings to
//!   the host.
//! - A fill takes a range of whole words inside the buffer: an empty range or one past the end is
//!   `Refusal::FillRange`, one of no whole words `Gap::FillAlignment`. Under a device cfg its
//!   bytes are plain Metal's (probe `blit_fill_rules.m`, job m3-max-43571), at one submission and
//!   three barriers alone.
//! - Under a device cfg: a synchronize brings the host's bytes of a Managed buffer up to date with
//!   a fill before it, to plain Metal's bytes (probe `blit_synchronize_rules.m`, job
//!   m3-max-43632): by one copy between two barriers where the buffer has a device copy, which
//!   the fill alone does not reach; by nothing where it has none. Each is counted and is no
//!   submission of its own; one of a texture without named bytes records nothing. A synchronize
//!   of a Managed texture copies each level of its image into its place in the named bytes,
//!   level 0 plain Metal's getBytes after a synchronize (job m3-max-43667), between one pair of
//!   barriers. A blit's segment ends reaching the host exactly where it wrote Shared memory or
//!   synchronized a device copy or a Managed texture.

use crate::blit::{
    buffer_copy, buffer_to_buffer, fill_range, texture_copy, BufferCopy, BufferLayout, ImageCopy,
    Lies, Side, Span, TexturePlace,
};
use crate::error::{Error, Gap, Refusal};
use crate::texture::{CopyUnit, Placed};
use crate::{PixelFormat, TextureDescriptor, TextureType};
use ash::vk;

/// A side of a copy: a texture of `d` held in `format`, of `unit`.
fn side(d: &TextureDescriptor, format: vk::Format, unit: Option<CopyUnit>) -> Side<'_> {
    Side {
        descriptor: d,
        format,
        unit,
        base: [0, 0],
    }
}

fn color(bytes: u32) -> Option<CopyUnit> {
    Some(CopyUnit {
        aspect: vk::ImageAspectFlags::COLOR,
        bytes,
        block: [1, 1],
    })
}

fn descriptor(
    texture_type: TextureType,
    format: PixelFormat,
    extent: [u32; 3],
) -> TextureDescriptor {
    let mut d = TextureDescriptor::new(texture_type, format, extent[0], extent[1]);
    d.depth = extent[2];
    d
}

fn at(slice: u32, level: u32, origin: [u32; 3]) -> TexturePlace {
    TexturePlace {
        slice,
        level,
        origin,
    }
}

fn placed(level: u32, layer: u32, origin: [u32; 3], size: [u32; 3]) -> Placed {
    Placed {
        level,
        layer,
        origin,
        size,
    }
}

/// The copy `result` checked, which copies something.
fn copied(result: Result<Option<ImageCopy>, Error>) -> ImageCopy {
    match result {
        Ok(Some(copy)) => copy,
        Ok(None) => panic!("it copies nothing"),
        Err(error) => panic!("{error}"),
    }
}

fn refused<T>(result: Result<Option<T>, Error>) -> Refusal {
    match result {
        Err(Error::Refused(refusal)) => refusal,
        Err(other) => panic!("not a refusal: {other}"),
        Ok(_) => panic!("not a refusal: it was checked"),
    }
}

fn gap<T>(result: Result<Option<T>, Error>) -> Gap {
    match result {
        Err(Error::Unsupported(gap)) => gap,
        Err(other) => panic!("not a gap: {other}"),
        Ok(_) => panic!("not a gap: it was checked"),
    }
}

#[test]
fn a_copy_places_both_regions() {
    let mut rgba = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [13, 7, 1]);
    rgba.mipmap_level_count = 3;
    let mut array = descriptor(TextureType::Type2DArray, PixelFormat::RGBA8Unorm, [8, 8, 1]);
    array.array_length = 3;
    let source = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    let destination = side(&array, vk::Format::R8G8B8A8_UNORM, color(4));
    let copy = copied(texture_copy(
        &source,
        at(0, 1, [1, 0, 0]),
        [4, 3, 1],
        &destination,
        at(2, 0, [2, 4, 0]),
        false,
    ));
    assert_eq!(copy.from, placed(1, 0, [1, 0, 0], [4, 3, 1]));
    assert_eq!(copy.to, placed(0, 2, [2, 4, 0], [4, 3, 1]));
    assert!(copy.aspect == vk::ImageAspectFlags::COLOR);
    // A depth of 0 on 2D textures copies one image; a width of 0 copies nothing.
    let flat = copied(texture_copy(
        &source,
        at(0, 0, [0; 3]),
        [2, 2, 0],
        &destination,
        at(0, 0, [0; 3]),
        false,
    ));
    assert_eq!(flat.from.size, [2, 2, 1]);
    assert!(matches!(
        texture_copy(
            &source,
            at(0, 0, [0; 3]),
            [0, 2, 1],
            &destination,
            at(0, 0, [0; 3]),
            false
        ),
        Ok(None)
    ));
    // A level, a slice or a region past either texture.
    for (from, to, size) in [
        (at(0, 3, [0; 3]), at(0, 0, [0; 3]), [1, 1, 1]),
        (at(0, 0, [0; 3]), at(3, 0, [0; 3]), [1, 1, 1]),
        (at(0, 1, [3, 0, 0]), at(0, 0, [0; 3]), [4, 1, 1]),
        (at(0, 0, [0; 3]), at(0, 0, [6, 0, 0]), [4, 1, 1]),
    ] {
        assert!(
            matches!(
                refused(texture_copy(&source, from, size, &destination, to, false)),
                Refusal::TextureRegion { .. }
            ),
            "{from:?} {to:?} {size:?}"
        );
    }
}

#[test]
fn a_3d_image_copies_to_and_from_a_2d_slice_and_a_cube_face_is_its_slice() {
    let volume = descriptor(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 3]);
    let plane = descriptor(TextureType::Type2D, PixelFormat::R8Unorm, [4, 4, 1]);
    let cube = descriptor(TextureType::Cube, PixelFormat::R8Unorm, [4, 4, 1]);
    let (v, p, c) = (
        side(&volume, vk::Format::R8_UNORM, color(1)),
        side(&plane, vk::Format::R8_UNORM, color(1)),
        side(&cube, vk::Format::R8_UNORM, color(1)),
    );
    let copy = texture_copy(
        &v,
        at(0, 0, [1, 1, 2]),
        [2, 2, 1],
        &p,
        at(0, 0, [2, 0, 0]),
        false,
    )
    .expect("a copy")
    .expect("one image");
    assert_eq!(copy.from, placed(0, 0, [1, 1, 2], [2, 2, 1]));
    assert_eq!(copy.to, placed(0, 0, [2, 0, 0], [2, 2, 1]));
    let copy = texture_copy(
        &c,
        at(4, 0, [0; 3]),
        [4, 4, 1],
        &v,
        at(0, 0, [0, 0, 1]),
        false,
    )
    .expect("a copy")
    .expect("a face");
    assert_eq!(copy.from, placed(0, 4, [0; 3], [4, 4, 1]));
    assert_eq!(copy.to, placed(0, 0, [0, 0, 1], [4, 4, 1]));
    // Two images of a 3D texture do not fit one 2D slice.
    assert!(matches!(
        refused(texture_copy(
            &v,
            at(0, 0, [0; 3]),
            [4, 4, 2],
            &p,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
    // One texture's two slices copy; one slice into itself is not built.
    let mut array = descriptor(TextureType::Type2DArray, PixelFormat::R8Unorm, [4, 4, 1]);
    array.array_length = 2;
    let a = side(&array, vk::Format::R8_UNORM, color(1));
    assert!(texture_copy(
        &a,
        at(0, 0, [0; 3]),
        [2, 2, 1],
        &a,
        at(1, 0, [2, 2, 0]),
        true
    )
    .expect("a copy")
    .is_some());
    assert_eq!(
        gap(texture_copy(
            &a,
            at(0, 0, [0; 3]),
            [2, 2, 1],
            &a,
            at(0, 0, [2, 2, 0]),
            true
        )),
        Gap::TextureCopy {
            source: PixelFormat::R8Unorm,
            destination: PixelFormat::R8Unorm
        }
    );
}

#[test]
fn what_metal_refuses_is_refused_and_what_one_command_does_not_make_is_a_gap() {
    let d = |format, extent| descriptor(TextureType::Type2D, format, extent);
    let rgba = d(PixelFormat::RGBA8Unorm, [8, 8, 1]);
    let srgb = d(PixelFormat::RGBA8UnormSRGB, [8, 8, 1]);
    let word = d(PixelFormat::R32Uint, [8, 8, 1]);
    let rg = d(PixelFormat::RG8Unorm, [8, 8, 1]);
    let bc1 = d(PixelFormat::BC1RGBA, [8, 8, 1]);
    let rg32 = d(PixelFormat::RG32Uint, [8, 8, 1]);
    let depth = d(PixelFormat::Depth32Float, [8, 8, 1]);
    let r32 = d(PixelFormat::R32Float, [8, 8, 1]);
    let alpha = d(PixelFormat::A8Unorm, [8, 8, 1]);
    let r8 = d(PixelFormat::R8Unorm, [8, 8, 1]);
    let line = descriptor(TextureType::Type1D, PixelFormat::R8Unorm, [8, 1, 1]);
    let mut msaa = descriptor(
        TextureType::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [8, 8, 1],
    );
    msaa.sample_count = 4;
    let whole = |source: &Side<'_>, destination: &Side<'_>| {
        texture_copy(
            source,
            at(0, 0, [0; 3]),
            [8, 1, 1],
            destination,
            at(0, 0, [0; 3]),
            false,
        )
    };
    let rgba = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    let bc1 = side(
        &bc1,
        vk::Format::BC1_RGBA_UNORM_BLOCK,
        Some(CopyUnit {
            aspect: vk::ImageAspectFlags::COLOR,
            bytes: 8,
            block: [4, 4],
        }),
    );
    let depth = side(
        &depth,
        vk::Format::D32_SFLOAT,
        Some(CopyUnit {
            aspect: vk::ImageAspectFlags::DEPTH,
            bytes: 4,
            block: [1, 1],
        }),
    );
    let mismatch = |source, destination| Refusal::TextureCopyMismatch {
        source,
        destination,
    };
    let not_built = |source, destination| Gap::TextureCopy {
        source,
        destination,
    };
    // Two formats, of one unit's bytes or not, as Metal's validation layer refuses them.
    assert_eq!(
        refused(whole(&rgba, &side(&rg, vk::Format::R8G8_UNORM, color(2)))),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::RG8Unorm)
    );
    assert_eq!(
        refused(whole(&rgba, &side(&word, vk::Format::R32_UINT, color(4)))),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::R32Uint)
    );
    assert_eq!(
        refused(whole(
            &rgba,
            &side(&srgb, vk::Format::R8G8B8A8_SRGB, color(4))
        )),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8UnormSRGB)
    );
    assert_eq!(
        refused(whole(&side(&msaa, vk::Format::R8G8B8A8_UNORM, None), &rgba)),
        mismatch(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Unorm)
    );
    let msaa = side(&msaa, vk::Format::R8G8B8A8_UNORM, None);
    assert_eq!(
        gap(whole(&msaa, &msaa)),
        not_built(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Unorm)
    );
    assert_eq!(
        refused(whole(&bc1, &side(&rg32, vk::Format::R32G32_UINT, color(8)))),
        mismatch(PixelFormat::BC1RGBA, PixelFormat::RG32Uint)
    );
    assert_eq!(
        refused(whole(&depth, &side(&r32, vk::Format::R32_SFLOAT, color(4)))),
        mismatch(PixelFormat::Depth32Float, PixelFormat::R32Float)
    );
    assert!(whole(&depth, &depth).expect("a copy").is_some());
    // `A8Unorm` held in `A8_UNORM` by one texture's usage and in `R8_UNORM` by the other's.
    let (alpha, alpha_in_red) = (
        side(&alpha, vk::Format::A8_UNORM_KHR, color(1)),
        side(&alpha, vk::Format::R8_UNORM, color(1)),
    );
    assert_eq!(
        gap(whole(&alpha, &alpha_in_red)),
        not_built(PixelFormat::A8Unorm, PixelFormat::A8Unorm)
    );
    assert!(whole(&alpha_in_red, &alpha_in_red)
        .expect("a copy")
        .is_some());
    assert_eq!(
        gap(whole(
            &side(&line, vk::Format::R8_UNORM, color(1)),
            &side(&r8, vk::Format::R8_UNORM, color(1))
        )),
        not_built(PixelFormat::R8Unorm, PixelFormat::R8Unorm)
    );
    // A compressed region is whole blocks or reaches the level's edge on both sides.
    assert!(texture_copy(
        &bc1,
        at(0, 0, [4, 0, 0]),
        [4, 4, 1],
        &bc1,
        at(0, 0, [0, 4, 0]),
        false
    )
    .expect("a copy")
    .is_some());
    assert!(matches!(
        refused(texture_copy(
            &bc1,
            at(0, 0, [0; 3]),
            [2, 4, 1],
            &bc1,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
    let mut chain = descriptor(TextureType::Type2D, PixelFormat::BC1RGBA, [8, 8, 1]);
    chain.mipmap_level_count = 4;
    let chain = side(&chain, vk::Format::BC1_RGBA_UNORM_BLOCK, bc1.unit);
    // Level 2 is 2 x 2 texels, inside one block: its edge, but not the destination's.
    assert!(texture_copy(
        &chain,
        at(0, 2, [0; 3]),
        [2, 2, 1],
        &chain,
        at(0, 3, [0; 3]),
        false
    )
    .is_err());
    assert!(matches!(
        refused(texture_copy(
            &chain,
            at(0, 2, [0; 3]),
            [2, 2, 1],
            &bc1,
            at(0, 0, [0; 3]),
            false
        )),
        Refusal::TextureRegion { .. }
    ));
}

/// A buffer at `base` in its block, `len` bytes long, and a region in it at `offset`, rows
/// `bytes_per_row` and images `bytes_per_image` apart.
fn lies(base: u64, len: u64, offset: u64, bytes_per_row: u64, bytes_per_image: u64) -> Lies {
    Lies {
        base,
        len,
        layout: BufferLayout {
            offset,
            bytes_per_row,
            bytes_per_image,
        },
    }
}

/// The buffer copy `result` checked, which copies something.
fn buffer_copied(result: Result<Option<BufferCopy>, Error>) -> BufferCopy {
    match result {
        Ok(Some(copy)) => copy,
        Ok(None) => panic!("it copies nothing"),
        Err(error) => panic!("{error}"),
    }
}

/// Each region's offset in the block, image offset and extent.
fn regions(copy: &BufferCopy) -> Vec<(u64, [i32; 3], [u32; 3])> {
    copy.regions()
        .map(|region| {
            let (o, e) = (region.image_offset, region.image_extent);
            (
                region.buffer_offset,
                [o.x, o.y, o.z],
                [e.width, e.height, e.depth],
            )
        })
        .collect()
}

#[test]
fn a_buffer_copy_places_each_image_at_its_own_offset() {
    let mut rgba = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [13, 7, 1]);
    rgba.mipmap_level_count = 3;
    let rgba = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    // Rows 32 bytes apart are 8 texels apart, from the buffer's offset in its block on.
    let copy = buffer_copied(buffer_copy(
        &rgba,
        at(0, 1, [1, 1, 0]),
        [4, 2, 1],
        lies(256, 256, 16, 32, 0),
        true,
    ));
    assert_eq!(copy.placed, placed(1, 0, [1, 1, 0], [4, 2, 1]));
    assert_eq!(copy.row_length, 8);
    assert_eq!(regions(&copy), [(272, [1, 1, 0], [4, 2, 1])]);
    // One row reads no row stride.
    let row = buffer_copied(buffer_copy(
        &rgba,
        at(0, 0, [0; 3]),
        [4, 1, 1],
        lies(0, 256, 0, 0, 0),
        false,
    ));
    assert_eq!(row.row_length, 0);
    // A 3D region's images each at their own offset, 24 bytes apart: no whole number of rows.
    let volume = descriptor(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 3]);
    let volume = side(&volume, vk::Format::R8_UNORM, color(1));
    let copy = buffer_copied(buffer_copy(
        &volume,
        at(0, 0, [1, 0, 0]),
        [3, 4, 3],
        lies(0, 128, 8, 5, 24),
        false,
    ));
    assert_eq!(copy.row_length, 5);
    assert_eq!(
        regions(&copy),
        [
            (8, [1, 0, 0], [3, 4, 1]),
            (32, [1, 0, 1], [3, 4, 1]),
            (56, [1, 0, 2], [3, 4, 1])
        ]
    );
    // A 1D array's slice, a height and depth of 0 one texel; its image stride is not read.
    let mut line = descriptor(TextureType::Type1DArray, PixelFormat::R32Float, [16, 1, 1]);
    line.array_length = 2;
    let line = side(&line, vk::Format::R32_SFLOAT, color(4));
    let copy = buffer_copied(buffer_copy(
        &line,
        at(1, 0, [3, 0, 0]),
        [8, 0, 0],
        lies(0, 256, 16, 32, 40),
        true,
    ));
    assert_eq!(copy.placed, placed(0, 1, [3, 0, 0], [8, 1, 1]));
    assert_eq!(regions(&copy), [(16, [3, 0, 0], [8, 1, 1])]);
    // BC1 blocks: a row of blocks is 8 bytes.
    let bc1 = descriptor(TextureType::Type2D, PixelFormat::BC1RGBA, [8, 8, 1]);
    let block = Some(CopyUnit {
        aspect: vk::ImageAspectFlags::COLOR,
        bytes: 8,
        block: [4, 4],
    });
    let bc1 = side(&bc1, vk::Format::BC1_RGBA_UNORM_BLOCK, block);
    let copy = buffer_copied(buffer_copy(
        &bc1,
        at(0, 0, [0; 3]),
        [8, 8, 1],
        lies(0, 64, 8, 24, 0),
        true,
    ));
    assert_eq!(copy.row_length, 12);
    assert!(matches!(
        buffer_copy(
            &bc1,
            at(0, 0, [0; 3]),
            [0, 8, 1],
            lies(0, 64, 0, 16, 0),
            true
        ),
        Ok(None)
    ));
}

#[test]
fn a_layout_metal_stops_is_refused_and_one_a_copy_command_cannot_address_is_a_gap() {
    let rgba = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [8, 8, 1]);
    let rgba = side(&rgba, vk::Format::R8G8B8A8_UNORM, color(4));
    let two_rows =
        |lies, into_image| buffer_copy(&rgba, at(0, 0, [0; 3]), [4, 2, 1], lies, into_image);
    let layout = |offset, bytes_per_row, bytes_per_image, len| Refusal::BufferCopyLayout {
        offset,
        bytes_per_row,
        bytes_per_image,
        len,
    };
    // Metal's validation: an offset or a row stride of no whole texels, rows closer than the
    // region's, and a region past the buffer's end.
    assert_eq!(
        refused(two_rows(lies(0, 4096, 2, 16, 0), true)),
        layout(2, 16, 0, 4096)
    );
    assert_eq!(
        refused(two_rows(lies(0, 4096, 0, 18, 0), false)),
        layout(0, 18, 0, 4096)
    );
    assert_eq!(
        refused(two_rows(lies(0, 4096, 0, 12, 0), true)),
        layout(0, 12, 0, 4096)
    );
    assert_eq!(
        refused(two_rows(lies(0, 4096, 4080, 16, 0), true)),
        layout(4080, 16, 0, 4096)
    );
    assert!(two_rows(lies(0, 4096, 4064, 16, 0), true).is_ok());
    let volume = descriptor(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 2]);
    let volume = side(&volume, vk::Format::R8_UNORM, color(1));
    let deep = |bytes_per_image| {
        buffer_copy(
            &volume,
            at(0, 0, [0; 3]),
            [4, 4, 2],
            lies(0, 4096, 0, 4, bytes_per_image),
            true,
        )
    };
    assert_eq!(refused(deep(12)), layout(0, 4, 12, 4096));
    assert!(deep(18).is_ok());
    // Metal takes an R8Unorm offset of 2.
    let r8 = descriptor(TextureType::Type2D, PixelFormat::R8Unorm, [8, 8, 1]);
    let r8 = side(&r8, vk::Format::R8_UNORM, color(1));
    assert!(buffer_copy(
        &r8,
        at(0, 0, [0; 3]),
        [4, 2, 1],
        lies(0, 4096, 2, 8, 0),
        true
    )
    .is_ok());
    // A copy command: a depth offset a multiple of 4, every image's offset whole texel blocks,
    // rows whole blocks.
    let alignment =
        |pixel_format, offset, bytes_per_row, bytes_per_image| Gap::BufferCopyAlignment {
            pixel_format,
            offset,
            bytes_per_row,
            bytes_per_image,
        };
    let depth = descriptor(TextureType::Type2D, PixelFormat::Depth16Unorm, [8, 8, 1]);
    let unit = |aspect, bytes, block| {
        Some(CopyUnit {
            aspect,
            bytes,
            block,
        })
    };
    let depth = side(
        &depth,
        vk::Format::D16_UNORM,
        unit(vk::ImageAspectFlags::DEPTH, 2, [1, 1]),
    );
    let out_of_depth = |offset| {
        buffer_copy(
            &depth,
            at(0, 0, [0; 3]),
            [4, 2, 1],
            lies(0, 4096, offset, 8, 0),
            false,
        )
    };
    assert_eq!(
        gap(out_of_depth(2)),
        alignment(PixelFormat::Depth16Unorm, 2, 8, 0)
    );
    assert!(out_of_depth(4).is_ok());
    let bc1 = descriptor(TextureType::Type2D, PixelFormat::BC1RGBA, [8, 8, 1]);
    let bc1 = side(
        &bc1,
        vk::Format::BC1_RGBA_UNORM_BLOCK,
        unit(vk::ImageAspectFlags::COLOR, 8, [4, 4]),
    );
    let blocks = |offset, bytes_per_row| {
        buffer_copy(
            &bc1,
            at(0, 0, [0; 3]),
            [8, 8, 1],
            lies(0, 4096, offset, bytes_per_row, 0),
            true,
        )
    };
    assert_eq!(
        gap(blocks(0, 20)),
        alignment(PixelFormat::BC1RGBA, 0, 20, 0)
    );
    assert_eq!(
        gap(blocks(4, 16)),
        alignment(PixelFormat::BC1RGBA, 4, 16, 0)
    );
    // The buffer's own offset in its block counts.
    assert_eq!(
        gap(two_rows(lies(2, 4096, 0, 16, 0), true)),
        alignment(PixelFormat::RGBA8Unorm, 0, 16, 0)
    );
    let words = descriptor(TextureType::Type3D, PixelFormat::RGBA8Unorm, [4, 4, 2]);
    let words = side(&words, vk::Format::R8G8B8A8_UNORM, color(4));
    assert_eq!(
        gap(buffer_copy(
            &words,
            at(0, 0, [0; 3]),
            [4, 4, 2],
            lies(0, 4096, 0, 16, 66),
            true
        )),
        alignment(PixelFormat::RGBA8Unorm, 0, 16, 66)
    );
    // No plain copy of the image's bytes, and no upload of a 32-bit float depth.
    let mut msaa = descriptor(
        TextureType::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [8, 8, 1],
    );
    msaa.sample_count = 4;
    let msaa = side(&msaa, vk::Format::R8G8B8A8_UNORM, None);
    assert_eq!(
        gap(buffer_copy(
            &msaa,
            at(0, 0, [0; 3]),
            [4, 2, 1],
            lies(0, 4096, 0, 16, 0),
            false
        )),
        Gap::TextureBytes {
            pixel_format: PixelFormat::RGBA8Unorm
        }
    );
    let float = descriptor(TextureType::Type2D, PixelFormat::Depth32Float, [8, 8, 1]);
    let float = side(
        &float,
        vk::Format::D32_SFLOAT,
        unit(vk::ImageAspectFlags::DEPTH, 4, [1, 1]),
    );
    let float_copy = |into_image| {
        buffer_copy(
            &float,
            at(0, 0, [0; 3]),
            [4, 2, 1],
            lies(0, 4096, 0, 16, 0),
            into_image,
        )
    };
    assert_eq!(
        gap(float_copy(true)),
        Gap::FloatDepthUpload {
            pixel_format: PixelFormat::Depth32Float
        }
    );
    assert!(float_copy(false).is_ok());
    assert!(matches!(
        refused(buffer_copy(
            &rgba,
            at(1, 0, [0; 3]),
            [4, 2, 1],
            lies(0, 4096, 0, 16, 0),
            true
        )),
        Refusal::TextureRegion { level: 0, slice: 1 }
    ));
}

#[test]
fn a_copy_between_buffers_takes_words_inside_both_that_share_no_byte() {
    use ash::vk::Handle;
    let one = vk::Buffer::from_raw(1);
    let span = |buffer, base, len| Span { buffer, base, len };
    // Offsets in the device's buffer are each buffer's in its block plus its own.
    let region = buffer_to_buffer(
        span(one, 256, 64),
        8,
        span(vk::Buffer::from_raw(2), 0, 32),
        4,
        16,
    )
    .expect("a copy")
    .expect("a region");
    assert_eq!(
        (region.src_offset, region.dst_offset, region.size),
        (264, 4, 16)
    );
    // Two buffers of one block, side by side, and one buffer's two ranges, apart.
    assert!(
        buffer_to_buffer(span(one, 0, 64), 0, span(one, 64, 64), 0, 64).is_ok_and(|r| r.is_some())
    );
    assert!(
        buffer_to_buffer(span(one, 0, 64), 0, span(one, 0, 64), 32, 32).is_ok_and(|r| r.is_some())
    );
    // A size of 0 copies nothing, wherever it is.
    assert!(matches!(
        buffer_to_buffer(span(one, 0, 64), 64, span(one, 0, 64), 0, 0),
        Ok(None)
    ));
    let refused = |result: Result<Option<vk::BufferCopy>, Error>| match result {
        Err(Error::Refused(refusal)) => refusal,
        other => panic!("{:?}", other.map(|r| r.is_some())),
    };
    for (offset, size) in [(60, 8), (68, 0), (u64::MAX - 3, 8)] {
        assert_eq!(
            refused(buffer_to_buffer(
                span(one, 0, 64),
                offset,
                span(one, 128, 256),
                0,
                size
            )),
            Refusal::BufferCopyRange {
                offset,
                size,
                len: 64
            },
            "{offset} {size}"
        );
    }
    assert_eq!(
        refused(buffer_to_buffer(
            span(one, 0, 256),
            0,
            span(one, 0, 64),
            60,
            8
        )),
        Refusal::BufferCopyRange {
            offset: 60,
            size: 8,
            len: 64
        },
        "the destination"
    );
    for (from, to, size) in [(2, 0, 4), (0, 2, 4), (0, 0, 6)] {
        assert_eq!(
            refused(buffer_to_buffer(
                span(one, 0, 64),
                from,
                span(vk::Buffer::from_raw(2), 0, 64),
                to,
                size
            )),
            Refusal::BufferCopyUnaligned {
                source_offset: from,
                destination_offset: to,
                size
            },
            "{from} {to} {size}"
        );
    }
    // One buffer's overlapping ranges, and two buffers of one block that overlap.
    assert_eq!(
        refused(buffer_to_buffer(
            span(one, 0, 64),
            0,
            span(one, 0, 64),
            16,
            32
        )),
        Refusal::BufferCopyOverlap {
            source_offset: 0,
            destination_offset: 16,
            size: 32
        }
    );
    assert_eq!(
        refused(buffer_to_buffer(
            span(one, 32, 64),
            0,
            span(one, 0, 64),
            0,
            64
        )),
        Refusal::BufferCopyOverlap {
            source_offset: 0,
            destination_offset: 0,
            size: 64
        }
    );
}

#[test]
fn a_fill_takes_whole_words_of_a_range_inside_the_buffer() {
    // The range's offset is the buffer's in its block plus its own.
    assert!(matches!(fill_range(256, 64, 4..12), Ok((260, 8))));
    for (start, end) in [(8, 8), (60, 68), (12, 4)] {
        assert!(
            matches!(
                fill_range(0, 64, start..end),
                Err(Error::Refused(Refusal::FillRange { len: 64, .. }))
            ),
            "{start}..{end}"
        );
    }
    for (base, start, end) in [(0, 1, 9), (0, 4, 10), (2, 4, 12)] {
        assert!(
            matches!(
                fill_range(base, 64, start..end),
                Err(Error::Unsupported(Gap::FillAlignment { .. }))
            ),
            "{base}: {start}..{end}"
        );
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::{
        BlitCommandEncoder, Buffer, BufferLayout, BytesPlace, CommandBuffer, CommandBufferStatus,
        Device, DispatchType, Error, Gap, IndexClass, MemoryClass, PixelFormat, Refusal, Region,
        Resource, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// A byte pattern of `len` bytes, different at every offset of one call and between calls.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// FNV-1a 64 of `bytes`.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// `bytes`, which are plain Metal's answer: their FNV-1a 64 is `hash`, the probe's.
    fn metal(bytes: Vec<u8>, hash: u64) -> Vec<u8> {
        assert_eq!(fnv(&bytes), hash, "plain Metal's bytes");
        bytes
    }

    /// One level and slice of a texture, whole, in elements of `unit` bytes covering `block`
    /// texels square.
    struct Level<'a> {
        texture: &'a Texture,
        slice: u32,
        level: u32,
        unit: usize,
        block: u32,
    }

    impl Level<'_> {
        fn extent(&self) -> [u32; 3] {
            let d = self.texture.descriptor();
            [d.width, d.height, d.depth].map(|e| (e >> self.level).max(1))
        }

        /// Elements across, rows and images.
        fn grid(&self) -> [usize; 3] {
            let [w, h, d] = self.extent();
            [
                w.div_ceil(self.block) as usize,
                h.div_ceil(self.block) as usize,
                d as usize,
            ]
        }

        fn layout(&self) -> (Region, u64, u64, usize) {
            let [across, rows, images] = self.grid();
            let row = across * self.unit;
            let region = Region {
                origin: [0; 3],
                size: self.extent(),
            };
            (region, row as u64, (row * rows) as u64, row * rows * images)
        }

        /// Writes a pattern of `seed` over the whole level and slice, and returns it.
        fn write(&self, seed: u8) -> Vec<u8> {
            let (region, row, image, len) = self.layout();
            let bytes = pattern(len, seed);
            self.texture
                .replace_region(region, self.level, self.slice, &bytes, row, image)
                .expect("replace_region");
            bytes
        }

        fn read(&self) -> Vec<u8> {
            let (region, row, image, len) = self.layout();
            let mut bytes = vec![0; len];
            self.texture
                .get_bytes(&mut bytes, row, image, region, self.level, self.slice)
                .expect("get_bytes");
            bytes
        }

        /// The place of element `at` in Metal's terms.
        fn place(&self, at: [usize; 3]) -> TexturePlace {
            let b = self.block as usize;
            TexturePlace {
                slice: self.slice,
                level: self.level,
                origin: [(at[0] * b) as u32, (at[1] * b) as u32, at[2] as u32],
            }
        }
    }

    /// One copy of `size` elements, from `at` in `from`, whose bytes are `source`, to `to_at` in
    /// `to`, whose bytes are `before`: the bytes `to` must hold after it.
    struct Expect<'a> {
        from: &'a Level<'a>,
        source: &'a [u8],
        at: [usize; 3],
        to: &'a Level<'a>,
        before: Vec<u8>,
        to_at: [usize; 3],
        size: [usize; 3],
    }

    impl Expect<'_> {
        fn record(&self, name: &str, command_buffer: &CommandBuffer) {
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let b = self.from.block as usize;
            let texels = [self.size[0] * b, self.size[1] * b, self.size[2]].map(|n| n as u32);
            // A compressed level smaller than a block copies the texels it has.
            let extent = self.from.extent();
            let texels = std::array::from_fn(|axis| texels[axis].min(extent[axis]));
            encoder
                .copy_from_texture(
                    self.from.texture,
                    self.from.place(self.at),
                    texels,
                    self.to.texture,
                    self.to.place(self.to_at),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
        }

        /// The destination's bytes after the copy.
        fn after(&self) -> Vec<u8> {
            let (from, to) = (self.from.grid(), self.to.grid());
            let unit = self.from.unit;
            let mut bytes = self.before.clone();
            for z in 0..self.size[2] {
                for y in 0..self.size[1] {
                    let element = |grid: [usize; 3], at: [usize; 3]| {
                        ((at[2] + z) * grid[1] + at[1] + y) * grid[0] + at[0]
                    };
                    let (s, d) = (
                        element(from, self.at) * unit,
                        element(to, self.to_at) * unit,
                    );
                    let len = self.size[0] * unit;
                    bytes[d..d + len].copy_from_slice(&self.source[s..s + len]);
                }
            }
            bytes
        }
    }

    fn texture(
        device: &Device,
        texture_type: TextureType,
        format: PixelFormat,
        extent: [u32; 3],
        usage: TextureUsage,
        index: u32,
    ) -> Result<Texture, Error> {
        let mut d = TextureDescriptor::new(texture_type, format, extent[0], extent[1]);
        d.depth = extent[2];
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        if matches!(texture_type, TextureType::Type2DArray) {
            d.array_length = 3;
        }
        if texture_type == TextureType::Type2D && format == PixelFormat::RGBA8Unorm {
            d.mipmap_level_count = 3;
        }
        device.new_texture(&d, Some(index))
    }

    /// Bytes land where each copy names, everything around them kept; one copy alone costs one
    /// submission and three barriers and copies nothing of the runtime's own; an owed barrier is
    /// paid by the copy's first.
    #[test]
    fn every_device_copies_between_textures() {
        each_device("blit_textures", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .expect("a grant");
            device
                .grant_indices(IndexClass::StorageTexture, 1, 16)
                .expect("a grant");
            let read = TextureUsage::SHADER_READ;
            let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let make = |texture_type, format, extent, usage, index| {
                texture(&device, texture_type, format, extent, usage, index)
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"))
            };
            let rgba = make(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [13, 7, 1],
                read,
                4096,
            );
            let words = make(
                TextureType::Type2DArray,
                PixelFormat::RGBA8Unorm,
                [8, 8, 1],
                write,
                1,
            );
            let cube = make(
                TextureType::Cube,
                PixelFormat::R8Unorm,
                [4, 4, 1],
                read,
                4097,
            );
            let volume = make(
                TextureType::Type3D,
                PixelFormat::R8Unorm,
                [4, 4, 3],
                read,
                4098,
            );
            let plane = make(
                TextureType::Type2D,
                PixelFormat::R8Unorm,
                [4, 4, 1],
                read,
                4099,
            );
            let level = |texture, slice, level, unit| Level {
                texture,
                slice,
                level,
                unit,
                block: 1,
            };
            let (rgba1, words2) = (level(&rgba, 0, 1, 4), level(&words, 2, 0, 4));
            let (words0, words1) = (level(&words, 0, 0, 4), level(&words, 1, 0, 4));
            let (face, image) = (level(&cube, 4, 0, 1), level(&volume, 0, 0, 1));
            let flat = level(&plane, 0, 0, 1);
            let sources = [rgba1.write(1), face.write(2), words0.write(3)];
            let befores = [
                words2.write(4),
                image.write(5),
                flat.write(6),
                words1.write(7),
            ];
            let queue = device.new_command_queue(4).expect("a queue");
            // The textures' uploads go with a command buffer of their own, one with commands: one
            // with nothing for the GPU takes no submission.
            let first = queue.command_buffer().expect("a command buffer");
            first
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            first.commit().expect("a commit");
            finished(name, &first);

            // A level of a 2D texture into a slice of an array.
            let into_array = Expect {
                from: &rgba1,
                source: &sources[0],
                at: [1, 0, 0],
                to: &words2,
                before: befores[0].clone(),
                to_at: [2, 4, 0],
                size: [4, 3, 1],
            };
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            into_array.record(name, &command_buffer);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.bytes_copied - before.bytes_copied,
                    after.passes - before.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            assert_eq!(
                words2.read(),
                metal(into_array.after(), 0xfbcf_3538_7a97_4105),
                "{name}: into the array"
            );

            // A cube's face into a 3D image, a 3D image into a 2D texture, and a slice of an
            // array into another of it, in one encoder; then a barrier an encoder owed, paid by
            // a copy's first.
            let into_volume = Expect {
                from: &face,
                source: &sources[1],
                at: [0, 0, 0],
                to: &image,
                before: befores[1].clone(),
                to_at: [0, 0, 1],
                size: [4, 4, 1],
            };
            let volume_after = metal(into_volume.after(), 0xf814_3509_403c_ef55);
            let into_plane = Expect {
                from: &image,
                source: &volume_after,
                at: [1, 1, 1],
                to: &flat,
                before: befores[2].clone(),
                to_at: [2, 0, 0],
                size: [2, 2, 1],
            };
            let across = Expect {
                from: &words0,
                source: &sources[2],
                at: [0, 0, 0],
                to: &words1,
                before: befores[3].clone(),
                to_at: [3, 3, 0],
                size: [5, 5, 1],
            };
            let command_buffer = queue.command_buffer().expect("a command buffer");
            for copy in [&into_volume, &into_plane, &across] {
                copy.record(name, &command_buffer);
            }
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(image.read(), volume_after, "{name}: into the volume");
            assert_eq!(
                flat.read(),
                metal(into_plane.after(), 0x52ff_1814_9f17_740d),
                "{name}: into the plane"
            );
            assert_eq!(
                words1.read(),
                metal(across.after(), 0x3735_fdb1_9af0_1a15),
                "{name}: across slices"
            );

            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut compute = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            compute.memory_barrier();
            assert!(matches!(
                command_buffer.blit_command_encoder(),
                Err(Error::Refused(Refusal::EncoderOpen))
            ));
            compute.end_encoding();
            into_array.record(name, &command_buffer);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                after.barriers - before.barriers,
                3,
                "{name}: the owed barrier"
            );

            // A block-compressed region, where the device makes the format.
            let bc = |index| {
                texture(
                    &device,
                    TextureType::Type2D,
                    PixelFormat::BC1RGBA,
                    [8, 8, 1],
                    read,
                    index,
                )
            };
            match (bc(4100), bc(4101)) {
                (Ok(a), Ok(b)) => {
                    let (from, to) = (
                        Level {
                            texture: &a,
                            slice: 0,
                            level: 0,
                            unit: 8,
                            block: 4,
                        },
                        Level {
                            texture: &b,
                            slice: 0,
                            level: 0,
                            unit: 8,
                            block: 4,
                        },
                    );
                    let source = from.write(8);
                    let blocks = Expect {
                        from: &from,
                        source: &source,
                        at: [1, 0, 0],
                        to: &to,
                        before: to.write(9),
                        to_at: [0, 1, 0],
                        size: [1, 1, 1],
                    };
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    blocks.record(name, &command_buffer);
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(
                        to.read(),
                        metal(blocks.after(), 0xc00d_3bc2_2f3d_14c5),
                        "{name}: BC1 blocks"
                    );
                }
                (Err(Error::Unsupported(Gap::TextureFormat { .. })), _) => {
                    println!("{name}: no BC1 texture on this device")
                }
                (a, b) => panic!("{name}: BC1: {:?} {:?}", a.err(), b.err()),
            }

            // A slice past the array is refused, and the encoder goes on.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let past = TexturePlace {
                slice: 3,
                level: 0,
                origin: [0; 3],
            };
            assert!(matches!(
                encoder.copy_from_texture(&words, past, [1, 1, 1], &rgba, TexturePlace::default()),
                Err(Error::Refused(Refusal::TextureRegion {
                    level: 0,
                    slice: 3
                }))
            ));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert!(matches!(
                command_buffer.blit_command_encoder(),
                Err(Error::Refused(Refusal::Committed))
            ));
        });
    }

    /// Writes `bytes` over the start of a Shared buffer.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("a Shared buffer");
        assert!(bytes.len() as u64 <= buffer.length());
        // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    }

    /// A Shared buffer's bytes, as the host reads them.
    fn read(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// `into` with `rows` (an offset in `from`, one in `into`) of `len` bytes each copied over.
    fn moved(
        mut into: Vec<u8>,
        from: &[u8],
        rows: impl Iterator<Item = (usize, usize)>,
        len: usize,
    ) -> Vec<u8> {
        for (from_at, into_at) in rows {
            into[into_at..into_at + len].copy_from_slice(&from[from_at..from_at + len]);
        }
        into
    }

    /// Bytes from buffers land in a mipmap level, a 1D array's slice and BC1 blocks, and a
    /// slice's and a 3D region's bytes land in buffers at padded strides, as plain Metal leaves
    /// them; one copy into a Shared buffer alone costs one submission and three barriers, the
    /// segment's end reaching the host, and copies nothing of the runtime's own.
    #[test]
    fn every_device_copies_between_buffers_and_textures() {
        each_device("blit_buffers", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .expect("a grant");
            device
                .grant_indices(IndexClass::StorageTexture, 1, 16)
                .expect("a grant");
            let read_usage = TextureUsage::SHADER_READ;
            let make = |texture_type, format, extent, index| {
                texture(&device, texture_type, format, extent, read_usage, index)
                    .unwrap_or_else(|error| panic!("{name}: {format:?}: {error}"))
            };
            let levels = make(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [13, 7, 1],
                4096,
            );
            let slices = make(
                TextureType::Type2DArray,
                PixelFormat::RGBA8Unorm,
                [8, 8, 1],
                4097,
            );
            let volume = make(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 3], 4098);
            let mut line =
                TextureDescriptor::new(TextureType::Type1DArray, PixelFormat::R32Float, 16, 1);
            line.array_length = 2;
            line.usage = read_usage;
            line.storage_mode = StorageMode::Shared;
            let lines = (device.new_texture(&line, Some(4099)))
                .unwrap_or_else(|error| panic!("{name}: a 1D array: {error}"));
            let level = |texture, slice, level, unit| Level {
                texture,
                slice,
                level,
                unit,
                block: 1,
            };
            let (level1, slice2) = (level(&levels, 0, 1, 4), level(&slices, 2, 0, 4));
            let (volume0, line1) = (level(&volume, 0, 0, 1), level(&lines, 1, 0, 4));
            let level_before = level1.write(12);
            let slice_bytes = slice2.write(13);
            let volume_bytes = volume0.write(15);
            let line_before = line1.write(19);
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let mut next = 0;
            let mut buffer = |len: u64, seed: u8| {
                let buffer = (device.new_buffer(block.id, next, len, Some(0))).expect("a buffer");
                next += len.next_multiple_of(256);
                let bytes = pattern(len as usize, seed);
                write(&buffer, &bytes);
                (buffer, bytes)
            };
            let (into_level, level_source) = buffer(256, 11);
            let (from_slice, slice_before) = buffer(1024, 14);
            let (from_volume, volume_before) = buffer(128, 16);
            let (into_line, line_source) = buffer(256, 18);
            let (into_blocks, block_source) = buffer(64, 17);
            let layout = |offset, bytes_per_row, bytes_per_image| BufferLayout {
                offset,
                bytes_per_row,
                bytes_per_image,
            };
            let place = |slice, level, origin| TexturePlace {
                slice,
                level,
                origin,
            };
            let queue = device.new_command_queue(4).expect("a queue");
            // The textures' uploads go with a command buffer of their own, one with commands: one
            // with nothing for the GPU takes no submission.
            let first = queue.command_buffer().expect("a command buffer");
            first
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            first.commit().expect("a commit");
            finished(name, &first);

            // A slice's region into a buffer at 64, rows 48 apart, alone.
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_texture_to_buffer(
                    &slices,
                    place(2, 0, [2, 3, 0]),
                    [5, 4, 1],
                    &from_slice,
                    layout(64, 48, 0),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.bytes_copied - before.bytes_copied,
                    after.passes - before.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            let rows = (0..4).map(|y| (((3 + y) * 8 + 2) * 4, 64 + y * 48));
            assert_eq!(
                read(&from_slice),
                metal(
                    moved(slice_before, &slice_bytes, rows, 20),
                    0x186e_870a_23ed_18a5
                ),
                "{name}: a slice into a buffer"
            );

            // Into a level at 16, rows 32 apart; a 3D region out at 8, rows 5 and images 24
            // apart; into a 1D array's slice; and BC1 blocks where the device makes them.
            let bc1 = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::BC1RGBA,
                [8, 8, 1],
                read_usage,
                4100,
            );
            let bc1 = match bc1 {
                Ok(texture) => Some(texture),
                Err(Error::Unsupported(Gap::TextureFormat { .. })) => {
                    println!("{name}: no BC1 texture on this device");
                    None
                }
                Err(error) => panic!("{name}: BC1: {error}"),
            };
            let blocks0 = bc1.as_ref().map(|texture| Level {
                texture,
                slice: 0,
                level: 0,
                unit: 8,
                block: 4,
            });
            let blocks_before = blocks0.as_ref().map(|level| level.write(20));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let copied = [
                encoder.copy_from_buffer(
                    &into_level,
                    layout(16, 32, 0),
                    [4, 2, 1],
                    &levels,
                    place(0, 1, [1, 1, 0]),
                ),
                encoder.copy_from_texture_to_buffer(
                    &volume,
                    place(0, 0, [1, 0, 0]),
                    [3, 4, 3],
                    &from_volume,
                    layout(8, 5, 24),
                ),
                encoder.copy_from_buffer(
                    &into_line,
                    layout(16, 32, 40),
                    [8, 1, 1],
                    &lines,
                    place(1, 0, [3, 0, 0]),
                ),
            ];
            for result in copied {
                result.unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            if let Some(bc1) = &bc1 {
                encoder
                    .copy_from_buffer(
                        &into_blocks,
                        layout(8, 16, 0),
                        [4, 4, 1],
                        bc1,
                        place(0, 0, [4, 4, 0]),
                    )
                    .unwrap_or_else(|error| panic!("{name}: BC1: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let rows = (0..2).map(|y| (16 + y * 32, ((1 + y) * 6 + 1) * 4));
            assert_eq!(
                level1.read(),
                metal(
                    moved(level_before, &level_source, rows, 16),
                    0xa5b4_4bf7_f3be_f955
                ),
                "{name}: a buffer into a level"
            );
            let rows =
                (0..3).flat_map(|z| (0..4).map(move |y| (z * 16 + y * 4 + 1, 8 + z * 24 + y * 5)));
            assert_eq!(
                read(&from_volume),
                metal(
                    moved(volume_before, &volume_bytes, rows, 3),
                    0xf57f_e637_5f8f_f12d
                ),
                "{name}: a 3D region into a buffer"
            );
            assert_eq!(
                line1.read(),
                metal(
                    moved(line_before, &line_source, [(16, 12)].into_iter(), 32),
                    0x8ec4_0174_4810_bfc5
                ),
                "{name}: a buffer into a 1D array's slice"
            );
            if let (Some(blocks0), Some(blocks_before)) = (blocks0, blocks_before) {
                assert_eq!(
                    blocks0.read(),
                    metal(
                        moved(blocks_before, &block_source, [(8, 24)].into_iter(), 8),
                        0x2be2_b29b_0bcd_00a5
                    ),
                    "{name}: BC1 blocks from a buffer"
                );
            }
        });
    }

    /// A fill's bytes are plain Metal's, everything around the range kept, at one submission and
    /// three barriers alone, the segment's end reaching the host.
    #[test]
    fn every_device_fills_a_range_of_a_buffer() {
        each_device("blit_fill", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let buffer = (device.new_buffer(block.id, 256, 64, Some(0))).expect("a buffer");
            let before = pattern(64, 21);
            write(&buffer, &before);
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .fill_buffer(&buffer, 4..12, 0xab)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.bytes_copied - metrics.bytes_copied,
                    after.passes - metrics.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            let mut want = before;
            want[4..12].fill(0xab);
            assert_eq!(
                read(&buffer),
                metal(want, 0x5d28_bc84_b408_472d),
                "{name}: the fill"
            );
        });
    }

    /// A copy between two buffers of one block and one within a buffer land where they name,
    /// everything around them kept, at one submission and three barriers alone, copying nothing
    /// of the runtime's own; a copy into a Managed buffer lands in the bytes shaders reach, which a
    /// synchronize brings to the host.
    #[test]
    fn every_device_copies_between_buffers() {
        each_device("blit_buffer_to_buffer", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let source = (device.new_buffer(block.id, 0, 256, Some(0))).expect("a buffer");
            let destination = (device.new_buffer(block.id, 256, 256, Some(0))).expect("a buffer");
            let (from, into) = (pattern(256, 31), pattern(256, 47));
            write(&source, &from);
            write(&destination, &into);
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();

            // Between two buffers of one block, alone.
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_buffer_to_buffer(&source, 16, &destination, 100, 64)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.bytes_copied - metrics.bytes_copied,
                    after.passes - metrics.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            let mut want = into.clone();
            want[100..164].copy_from_slice(&from[16..80]);
            assert_eq!(read(&destination), want, "{name}: between two buffers");
            assert_eq!(read(&source), from, "{name}: the source kept");

            // Within one buffer, two ranges apart; a size of 0 records nothing more.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_buffer_to_buffer(&source, 0, &source, 128, 32)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder
                .copy_from_buffer_to_buffer(&source, 0, &source, 0, 0)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let mut want = from.clone();
            want[128..160].copy_from_slice(&from[0..32]);
            assert_eq!(read(&source), want, "{name}: within one buffer");

            // Into a Managed buffer: its bytes shaders reach, brought to the host by a
            // synchronize.
            let copied = !device.core().unified_memory();
            let managed = (device.new_managed_buffer(block.id, 512, 64, 64))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_buffer_to_buffer(&destination, 0, &managed, 8, 32)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder
                .synchronize_resource(Resource::Buffer(&managed))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let landed = read(&managed);
            println!("{name}: a copy into a Managed buffer: device copy {copied}");
            assert_eq!(
                landed[8..40],
                read(&destination)[0..32],
                "{name}: into a Managed buffer (device copy: {copied})"
            );
        });
    }

    /// Returns once the block-maker thread has nothing to make (it makes an allocation of a
    /// class whose free bytes run low, which a count of a test's own work must not take in), and
    /// the reaper has made the pipeline cache, which sets its gauge.
    pub(crate) fn settled(device: &Device) {
        within("8 MiB free in each class", || {
            let pool = &device.core().pool;
            pool.stocked(MemoryClass::Private) && pool.stocked(MemoryClass::Shared)
        });
        within("the pipeline cache", || device.core().caches.is_made());
    }

    /// `command_buffer` committed and complete with no error.
    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// A synchronize brings the host's bytes of a Managed buffer up to date with a fill before it
    /// in the command buffer, to plain Metal's bytes. Where the buffer has a device copy the fill
    /// alone leaves the host's bytes as they were, and the synchronize is one copy between two
    /// barriers; where it has none the host reads the fill's bytes at once and the synchronize
    /// records nothing. Either way it is counted, is no submission of its own, and the segment's
    /// end reaches the host. A synchronize of a texture without named bytes records nothing.
    #[test]
    fn every_device_synchronizes_a_managed_buffer() {
        each_device("blit_synchronize", |name, _, device| {
            let copied = !device.core().unified_memory();
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let before = pattern(64, 21);
            let start = (device.new_buffer(block.id, 256, 64, None)).expect("a buffer");
            write(&start, &before);
            drop(start);
            let managed = (device.new_managed_buffer(block.id, 256, 64, 64))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Made before the first command buffer, whose upload moves it home.
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .expect("a grant");
            let usage = TextureUsage::SHADER_READ;
            let texture = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::R8Unorm,
                [4, 4, 1],
                usage,
                4096,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();

            // The fill alone: the host reads it only where the buffer has no device copy.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .fill_buffer(&managed, 4..8, 0xab)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let mut want = before.clone();
            if !copied {
                want[4..8].fill(0xab);
            }
            assert_eq!(
                read(&managed),
                want,
                "{name}: the fill alone (device copy: {copied})"
            );

            // The fill and a synchronize.
            settled(&device);
            let metrics = counters.metrics();
            let reaches = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .fill_buffer(&managed, 4..12, 0xab)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder
                .synchronize_resource(Resource::Buffer(&managed))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.bytes_copied - metrics.bytes_copied,
                    after.synchronizations - metrics.synchronizations,
                    counters.host_reaches() - reaches,
                ),
                (1, if copied { 5 } else { 3 }, 0, 1, 1),
                "{name}: submissions, barriers, bytes copied, synchronizations, host reaches \
                 (device copy: {copied})"
            );
            let mut want = before;
            want[4..12].fill(0xab);
            assert_eq!(
                read(&managed),
                metal(want, 0x5d28_bc84_b408_472d),
                "{name}: the synchronized fill (device copy: {copied})"
            );

            // A synchronize of a texture without named bytes records nothing: the command buffer
            // has nothing for the GPU, and takes no submission and no barrier.
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .synchronize_resource(Resource::Texture(&texture))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            let ran = Arc::new(AtomicBool::new(false));
            let handled = Arc::clone(&ran);
            command_buffer
                .add_completed_handler(move |done| {
                    let completed = done.status() == CommandBufferStatus::Completed;
                    handled.store(completed, Ordering::SeqCst);
                })
                .expect("a completed handler");
            run(name, &command_buffer);
            let after = counters.metrics();
            within("the completed handler", || ran.load(Ordering::SeqCst));
            let (start, end) = (
                command_buffer.gpu_start_time(),
                command_buffer.gpu_end_time(),
            );
            assert!(
                start == end && start > 0,
                "{name}: GPU times {start}, {end}"
            );
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.synchronizations - metrics.synchronizations,
                ),
                (0, 0, 1),
                "{name}: a texture's submissions, barriers and synchronizations"
            );
        });
    }

    /// A blit's segment ends reaching the host where it wrote a Shared buffer, by a fill or a
    /// copy from a texture, and not where it wrote only Private memory or a texture: the
    /// invariant a synchronize with nothing to copy rests on, as the compute encoder's test holds
    /// it for dispatches.
    #[test]
    fn every_device_reaches_the_host_only_after_a_blit_writes_shared_memory() {
        each_device("blit_host_reach", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .expect("a grant");
            let shared = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let private = (device.take_block(MemoryClass::Private, 4096)).expect("a Private block");
            let shared = (device.new_buffer(shared.id, 0, 256, Some(0))).expect("a Shared buffer");
            let private =
                (device.new_buffer(private.id, 0, 256, Some(0))).expect("a Private buffer");
            let texture = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::R8Unorm,
                [16, 16, 1],
                TextureUsage::SHADER_READ,
                4096,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: 16,
                bytes_per_image: 0,
            };
            let at = TexturePlace::default();
            type Blit<'a> = &'a dyn Fn(&mut BlitCommandEncoder) -> Result<(), Error>;
            let fill_private: Blit =
                &|encoder: &mut BlitCommandEncoder| encoder.fill_buffer(&private, 0..256, 1);
            let into_texture: Blit = &|encoder: &mut BlitCommandEncoder| {
                encoder.copy_from_buffer(&private, layout, [16, 16, 1], &texture, at)
            };
            let fill_shared: Blit =
                &|encoder: &mut BlitCommandEncoder| encoder.fill_buffer(&shared, 0..256, 2);
            let out_of_texture: Blit = &|encoder: &mut BlitCommandEncoder| {
                encoder.copy_from_texture_to_buffer(&texture, at, [16, 16, 1], &shared, layout)
            };
            for (what, blit, reaches) in [
                ("a fill of a Private buffer", fill_private, 0),
                ("a copy into a texture", into_texture, 0),
                ("a fill of a Shared buffer", fill_shared, 1),
                ("a copy into a Shared buffer", out_of_texture, 1),
            ] {
                let before = counters.host_reaches();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                blit(&mut encoder).unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                assert_eq!(counters.host_reaches() - before, reaches, "{name}: {what}");
            }
            // The copy out of the texture brought the Private buffer's fill to the host.
            assert_eq!(read(&shared), vec![1; 256], "{name}: the bytes");
        });
    }

    /// A synchronize of a Managed texture copies each level of its image into its place in the
    /// consumer's named bytes, rows 256 bytes apart, everything between them kept, level 0 plain
    /// Metal's: one submission, two barriers and the segment's end reaching the host, no byte of
    /// the runtime's own. Copies into the texture alone leave the named bytes as they were.
    #[test]
    fn every_device_synchronizes_a_managed_texture_into_its_named_bytes() {
        each_device("blit_synchronize_texture", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 8192)).expect("a Shared block");
            let named = (device.new_buffer(block.id, 0, 2048, Some(0))).expect("a buffer");
            let source = (device.new_buffer(block.id, 4096, 128, Some(0))).expect("a buffer");
            let (level0, level1) = (pattern(64, 21), pattern(16, 99));
            write(&source, &[level0.clone(), level1.clone()].concat());
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::RENDER_TARGET;
            let place = |offset| BytesPlace {
                offset,
                bytes_per_row: 256,
                bytes_per_image: 0,
            };
            let texture = (device.new_managed_texture(&d, None, &named, &[place(0), place(1024)]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();

            // The copies into the image alone.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            for (offset, bytes_per_row, side, level) in [(0, 16, 4, 0), (64, 8, 2, 1)] {
                let layout = BufferLayout {
                    offset,
                    bytes_per_row,
                    bytes_per_image: 0,
                };
                let at = TexturePlace {
                    level,
                    ..TexturePlace::default()
                };
                encoder
                    .copy_from_buffer(&source, layout, [side, side, 1], &texture, at)
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            assert_eq!(read(&named), vec![0; 2048], "{name}: the copies alone");

            // The synchronize.
            settled(&device);
            let metrics = counters.metrics();
            let reaches = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .synchronize_resource(Resource::Texture(&texture))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.bytes_copied - metrics.bytes_copied,
                    after.synchronizations - metrics.synchronizations,
                    counters.host_reaches() - reaches,
                ),
                (1, 3, 0, 1, 1),
                "{name}: submissions, barriers, bytes copied, synchronizations, host reaches"
            );
            let got = read(&named);
            let rows = |at: usize, row: usize, count: usize| -> Vec<u8> {
                (0..count)
                    .flat_map(|y| got[at + y * 256..at + y * 256 + row].to_vec())
                    .collect()
            };
            assert_eq!(
                rows(0, 16, 4),
                metal(level0, 0xa918_a78f_c82c_bba5),
                "{name}: level 0"
            );
            assert_eq!(rows(1024, 8, 2), level1, "{name}: level 1");
            let mut between = got.clone();
            for (at, row, count) in [(0, 16, 4), (1024, 8, 2)] {
                for y in 0..count {
                    between[at + y * 256..at + y * 256 + row].fill(0);
                }
            }
            assert_eq!(between, vec![0; 2048], "{name}: the bytes between the rows");
        });
    }
}
