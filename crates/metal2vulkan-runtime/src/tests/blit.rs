//! Blit copies between textures, and between buffers and textures; buffer fills; synchronizes.
//!
//! - The rules: a texture copies into one of its own pixel format and sample count; two formats
//!   or two sample counts are `Refusal::TextureCopyMismatch`, as Metal's validation refuses them;
//!   what one copy command in the floor does not make, one pixel format held in two Vulkan formats
//!   and a multisample texture among it, is `Gap::TextureCopy`; a format of depth and stencil
//!   both copies both aspects in each region, overlapping regions of one texture too; within one level and slice of one image (one texture, or a
//!   texture and its view), regions apart copy in the general layout, the same region copies
//!   nothing, and overlapping regions, which Metal leaves undefined, copy in slabs across the
//!   axis of fewest, each no thicker than the shift there, the far end of the shift first; so do
//!   regions of two textures in place over one buffer whose bytes meet, each shift that from a
//!   destination row to a source row on its bytes, and where no axis cuts them so they are
//!   `Gap::OverlappingCopy`, while rows that interleave or other bytes copy whole; so are two
//!   textures whose memory meets where one is placed in a heap, though not two slices of one; a
//!   level, slice or region outside a texture or off its block grid is `Refusal::TextureRegion`;
//!   a size of 0 on an axis a texture lacks copies one texel there, and on another copies
//!   nothing; a 3D texture's image copies to and from a 2D slice.
//! - Under a device cfg, on every device of the cfg's kind: the bytes of a level of a 2D texture
//!   land in a slice of a 2D array, a cube's face in an
//!   image of a 3D texture, an image of a 3D texture in a 2D texture, a slice of an array in
//!   another slice of it, and a block-compressed region in another texture where the device makes
//!   one, everything around each region kept; a texture's region into another of the same level
//!   and slice, from the texture or from its view, the same. One copy alone is one submission, two barriers and
//!   the segment's end, and copies no byte of the runtime's own; a barrier an encoder owed is paid
//!   by the copy's first. Every gauge comes back to zero with the device. The bytes each copy
//!   must leave are plain Metal's: their FNV-1a 64 hashes are those of probe `blit_copy_layouts.m`
//!   in the conformance crate (an Apple M3 Max, macOS 27.0.1, job m3-max-43464, plain and under
//!   the validation layer). Overlapping regions of one level and slice leave the source's bytes
//!   from before the copy, toward higher and lower coordinates on each axis, through a view, and
//!   in BC1 blocks where the device makes them, at one barrier more a slab after the first. So do
//!   regions of two textures in place over one buffer whose bytes meet, a row on and within the
//!   rows, each way, every other byte of the buffer kept; whole rows each across the end of its
//!   own source row and the start of the next are refused, counted, and record nothing.
//! - Between a buffer and a texture: each image of the region lies at its own offset, its rows a
//!   count of texels apart; what Metal's validation stops is `Refusal::BufferCopyLayout`, and an
//!   offset or row stride a copy command cannot address `Gap::BufferCopyAlignment`. A region whose
//!   bytes meet those of a texture in place over the buffer copies in slabs as two textures over
//!   one buffer do, either way, its own texels copying nothing, and where no axis cuts them so,
//!   or where a texture placed in a heap holds bytes of the region, is `Gap::OverlappingCopy`.
//!   Under a device cfg: such regions a row on and back and within the rows, into the texture
//!   and out of it, leave the source's bytes from before the copy at one barrier more a slab
//!   after the first, and the refused shape records nothing and is counted; so are copies between
//!   heap textures, and a heap buffer and a texture, whose memory meets, while ones apart run.
//!   Bytes from a buffer land in a mipmap level, a slice of a 1D array and BC1 blocks, and a
//!   slice's and a 3D region's bytes land in buffers at padded strides, everything around each
//!   region kept and each result plain Metal's (job m3-max-43555); one copy into a Shared buffer
//!   alone is one submission and three barriers, the segment's end reaching the host, and copies
//!   no byte of the runtime's own.
//! - A copy between buffers takes ranges inside both buffers (`Refusal::BufferCopyRange`), at any
//!   offsets and of any size, and a source and a destination that share no byte
//!   (`Refusal::BufferCopyOverlap`), one buffer's ranges or two buffers' in one block; a size of 0
//!   copies nothing. Under a device cfg its bytes land where it names, between two buffers of one
//!   block and within one buffer, everything around them kept, at one submission and three
//!   barriers alone, copying no byte of the runtime's own; at offsets and sizes of no whole words
//!   too, each byte of the range and no other, as plain Metal's copy (probe
//!   `blit_buffer_copy_unaligned.m`, jobs m3-max-69613 and m3-max-69614); a copy into a Managed
//!   buffer lands in the bytes shaders reach, which a synchronize brings to the host.
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
    Lies, Side, Slabs, Span, TexturePlace,
};
use crate::error::{Error, Gap, Refusal};
use crate::texture::{CopyUnit, Placed, Rows};
use crate::{PixelFormat, TextureDescriptor, TextureType};
use ash::vk::{self, Handle};

/// A side of a copy: a texture of `d` held in `format`, of `unit`.
fn side(d: &TextureDescriptor, format: vk::Format, unit: Option<CopyUnit>) -> Side<'_> {
    Side {
        descriptor: d,
        format,
        unit,
        base: [0, 0],
        rows: None,
        bound: None,
    }
}

/// A side of an RGBA8 texture of `d` in place over a buffer: its first texel at `offset` in the
/// `VkBuffer` numbered `buffer`, its rows `pitch` apart.
fn over(d: &TextureDescriptor, buffer: u64, offset: u64, pitch: u64) -> Side<'_> {
    Side {
        rows: Some(Rows {
            buffer: vk::Buffer::from_raw(buffer),
            offset,
            pitch,
        }),
        ..side(d, vk::Format::R8G8B8A8_UNORM, color(4))
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
    // One texture's two slices copy; within one slice, regions apart copy in the general layout,
    // the same region copies nothing, and overlapping regions copy in slabs.
    let mut array = descriptor(TextureType::Type2DArray, PixelFormat::R8Unorm, [4, 4, 1]);
    array.array_length = 2;
    let a = side(&array, vk::Format::R8_UNORM, color(1));
    let within = |to, size| texture_copy(&a, at(0, 0, [0; 3]), size, &a, to, true);
    let across = copied(within(at(1, 0, [0; 3]), [2, 2, 1]));
    assert!(!across.shared, "two slices");
    for (to, size) in [
        ([2, 2, 0], [2, 2, 1]),
        ([2, 0, 0], [2, 4, 1]),
        ([0, 3, 0], [4, 1, 1]),
    ] {
        let apart = copied(within(at(0, 0, to), size));
        assert!(apart.shared, "{to:?} {size:?}");
        assert_eq!(apart.to, placed(0, 0, to, size));
    }
    assert!(matches!(within(at(0, 0, [0; 3]), [3, 3, 1]), Ok(None)));
    // Overlapping regions copy in slabs, each one's regions apart.
    for (to, size, slabs) in [
        ([1, 1, 0], [2, 2, 1], (0, 1, true)),
        ([1, 0, 0], [2, 1, 1], (0, 1, true)),
        ([0, 1, 0], [4, 2, 1], (1, 1, true)),
    ] {
        let copy = copied(within(at(0, 0, to), size));
        assert!(copy.shared, "{to:?} {size:?}");
        let (axis, thickness, reversed) = slabs;
        assert_eq!(
            copy.slabs,
            Some(Slabs {
                axis,
                thickness,
                reversed
            }),
            "{to:?} {size:?}"
        );
    }
}

/// The regions of `copy`'s commands, in order: each one's source and destination origins and its
/// extent.
fn slab_regions(copy: &ImageCopy) -> Vec<([i32; 2], [i32; 2], [u32; 2])> {
    (copy.regions())
        .map(|r| {
            (
                [r.src_offset.x, r.src_offset.y],
                [r.dst_offset.x, r.dst_offset.y],
                [r.extent.width, r.extent.height],
            )
        })
        .collect()
}

#[test]
fn overlapping_regions_copy_in_slabs_the_far_end_of_the_shift_first() {
    let d = descriptor(TextureType::Type2D, PixelFormat::RG8Unorm, [256, 352, 1]);
    let s = side(&d, vk::Format::R8G8_UNORM, color(2));
    let within = |from, size, to| {
        copied(texture_copy(
            &s,
            at(0, 0, from),
            size,
            &s,
            at(0, 0, to),
            true,
        ))
    };
    // Columns 0 to 20 of a row onto columns 20 to 40: the last column first, then the rest.
    assert_eq!(
        slab_regions(&within([0, 0, 0], [21, 1, 0], [20, 0, 0])),
        [([20, 0], [40, 0], [1, 1]), ([0, 0], [20, 0], [20, 1])]
    );
    // Toward lower coordinates, the first slab first; the last slab what is left.
    assert_eq!(
        slab_regions(&within([0, 5, 0], [3, 5, 1], [0, 3, 0])),
        [
            ([0, 5], [0, 3], [3, 2]),
            ([0, 7], [0, 5], [3, 2]),
            ([0, 9], [0, 7], [3, 1]),
        ]
    );
    // Shifted along both axes, the axis of fewer slabs: three across y against eight across x.
    let both = within([4, 4, 0], [8, 6, 1], [5, 2, 0]);
    assert_eq!(
        both.slabs,
        Some(Slabs {
            axis: 1,
            thickness: 2,
            reversed: false
        })
    );
    assert_eq!(
        slab_regions(&both),
        [
            ([4, 4], [5, 2], [8, 2]),
            ([4, 6], [5, 4], [8, 2]),
            ([4, 8], [5, 6], [8, 2]),
        ]
    );
    // Regions apart are one slab, the whole region.
    assert_eq!(
        slab_regions(&within([0, 0, 0], [4, 4, 1], [4, 0, 0])),
        [([0, 0], [4, 0], [4, 4])]
    );
}

#[test]
fn regions_of_two_textures_over_one_buffer_copy_in_slabs_where_their_bytes_meet() {
    // Rows of 16 texels of 4 bytes, 64 bytes apart.
    let d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [16, 4, 1]);
    let first = over(&d, 1, 0, 64);
    let copy = |from: &Side<'_>, to: &Side<'_>, size| {
        texture_copy(from, at(0, 0, [0; 3]), size, to, at(0, 0, [0; 3]), false)
    };
    let slabs = |from: &Side<'_>, to: &Side<'_>, size| copied(copy(from, to, size)).slabs;
    let cut = |axis, thickness, reversed| {
        Some(Slabs {
            axis,
            thickness,
            reversed,
        })
    };
    let whole = [16, 4, 1];
    // A texture a row on: slabs of a row, the end the shift points to first.
    let row_on = over(&d, 1, 64, 64);
    let forward = copied(copy(&first, &row_on, whole));
    assert!(!forward.shared, "two images");
    assert_eq!(
        slab_regions(&forward),
        [
            ([0, 3], [0, 3], [16, 1]),
            ([0, 2], [0, 2], [16, 1]),
            ([0, 1], [0, 1], [16, 1]),
            ([0, 0], [0, 0], [16, 1]),
        ]
    );
    assert_eq!(slabs(&row_on, &first, whole), cut(1, 1, false));
    assert_eq!(slabs(&first, &over(&d, 1, 128, 64), whole), cut(1, 2, true));
    // Two texels on, a region that stays inside the other's rows: slabs of two texels across.
    let within = over(&d, 1, 8, 64);
    assert_eq!(slabs(&first, &within, [12, 4, 1]), cut(0, 2, true));
    assert_eq!(slabs(&within, &first, [12, 4, 1]), cut(0, 2, false));
    // A row and two texels on, whole rows: each destination row across the end of one source
    // row and the start of the next, neither its own, so slabs of a row.
    assert_eq!(slabs(&first, &over(&d, 1, 72, 64), whole), cut(1, 1, true));
    // Two texels on, whole rows: each destination row across the end of its own source row and
    // the start of the next. No cut along one axis keeps the order, which the encoder refuses.
    assert_eq!(slabs(&first, &within, whole), None);
    // Rows of two pitches that meet at their starts: no cut either.
    let wider = over(&d, 1, 0, 128);
    assert_eq!(slabs(&first, &wider, [8, 4, 1]), None);
    // Regions whose rows interleave, touching, of one pitch or two, and textures over other bytes,
    // share no byte: one slab, the whole region.
    let (left, interleaved) = (over(&d, 1, 0, 128), over(&d, 1, 64, 128));
    assert_eq!(slabs(&left, &interleaved, whole), cut(0, 16, false));
    assert_eq!(
        slabs(&first, &over(&d, 1, 32, 128), [8, 4, 1]),
        cut(0, 8, false)
    );
    assert_eq!(slabs(&first, &over(&d, 2, 0, 64), whole), cut(0, 16, false));
    // The same bytes: nothing to copy.
    assert!(matches!(
        texture_copy(
            &first,
            at(0, 0, [0, 1, 0]),
            [16, 3, 1],
            &row_on,
            at(0, 0, [0; 3]),
            false
        ),
        Ok(None)
    ));
}

/// A side of an RGBA8 texture of `d` whose memory, `len` bytes at `base` in the `VkBuffer`
/// numbered 1, other resources may share: placed in a heap, laid out as the device chooses.
fn in_heap(d: &TextureDescriptor, base: u64, len: u64) -> Side<'_> {
    Side {
        bound: Some(Span {
            buffer: vk::Buffer::from_raw(1),
            base,
            len,
        }),
        ..side(d, vk::Format::R8G8B8A8_UNORM, color(4))
    }
}

#[test]
fn textures_whose_memory_meets_where_one_is_laid_out_by_the_device_copy_in_no_order() {
    let d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [16, 4, 1]);
    let slabs = |from: &Side<'_>, to: &Side<'_>| {
        let size = [16, 4, 1];
        let origin = at(0, 0, [0; 3]);
        copied(texture_copy(from, origin, size, to, origin, false)).slabs
    };
    let whole = Some(Slabs {
        axis: 0,
        thickness: 16,
        reversed: false,
    });
    let (a, b) = (in_heap(&d, 0, 4096), in_heap(&d, 2048, 4096));
    // Two heap textures whose ranges meet, either way, and a texture in place whose image's
    // memory lies in a heap texture's: no order, which the encoder refuses.
    assert_eq!(slabs(&a, &b), None);
    assert_eq!(slabs(&b, &a), None);
    let laid = Side {
        bound: Some(Span {
            buffer: vk::Buffer::from_raw(1),
            base: 3072,
            len: 256,
        }),
        ..over(&d, 1, 3072, 64)
    };
    assert_eq!(slabs(&laid, &b), None);
    // Ranges that touch, ranges in another buffer, and a texture whose memory no other resource
    // shares: one slab.
    assert_eq!(slabs(&a, &in_heap(&d, 4096, 4096)), whole);
    let elsewhere = Side {
        bound: Some(Span {
            buffer: vk::Buffer::from_raw(2),
            base: 0,
            len: 4096,
        }),
        ..in_heap(&d, 0, 4096)
    };
    assert_eq!(slabs(&a, &elsewhere), whole);
    let own = side(&d, vk::Format::R8G8B8A8_UNORM, color(4));
    assert_eq!(slabs(&a, &own), whole);
    // Two slices of one heap texture are one image's: one slab, two subresources.
    let mut array = descriptor(
        TextureType::Type2DArray,
        PixelFormat::RGBA8Unorm,
        [16, 4, 1],
    );
    array.array_length = 2;
    let slices = in_heap(&array, 0, 8192);
    let copy = copied(texture_copy(
        &slices,
        at(0, 0, [0; 3]),
        [16, 4, 1],
        &slices,
        at(1, 0, [0; 3]),
        true,
    ));
    assert!(!copy.shared);
    assert_eq!(copy.slabs, whole);
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

#[test]
fn a_format_of_depth_and_stencil_both_copies_both_aspects_in_each_region() {
    use vk::ImageAspectFlags as A;
    // Each region's source and destination aspects, as numbers.
    let both = [(A::DEPTH | A::STENCIL).as_raw(); 2];
    let held = [
        (
            PixelFormat::Depth32FloatStencil8,
            vk::Format::D32_SFLOAT_S8_UINT,
        ),
        (
            PixelFormat::Depth24UnormStencil8,
            vk::Format::D24_UNORM_S8_UINT,
        ),
    ];
    for (pf, format) in held {
        let d = descriptor(TextureType::Type2D, pf, [8, 8, 1]);
        let (a, b) = (side(&d, format, None), side(&d, format, None));
        let aspects = |copy: &ImageCopy| {
            (copy.regions())
                .map(|r| {
                    let masks = [r.src_subresource.aspect_mask, r.dst_subresource.aspect_mask];
                    masks.map(A::as_raw)
                })
                .collect::<Vec<_>>()
        };
        let copy = copied(texture_copy(
            &a,
            at(0, 0, [1, 2, 0]),
            [3, 4, 1],
            &b,
            at(0, 0, [4, 0, 0]),
            false,
        ));
        let raw = format.as_raw();
        assert_eq!(aspects(&copy), [both], "{pf:?} in {raw}");
        assert_eq!(slab_regions(&copy), [([1, 2], [4, 0], [3, 4])]);
        // Within one texture: regions apart are one region, overlapping ones slabs that share no
        // texel, the far end of the shift first, each of both aspects.
        let within = |from, to| {
            copied(texture_copy(
                &a,
                at(0, 0, from),
                [4, 2, 1],
                &a,
                at(0, 0, to),
                true,
            ))
        };
        let apart = within([0, 0, 0], [4, 4, 0]);
        assert!(apart.shared);
        assert_eq!(slab_regions(&apart), [([0, 0], [4, 4], [4, 2])]);
        let overlapping = within([0, 0, 0], [0, 1, 0]);
        assert!(overlapping.shared);
        assert_eq!(
            slab_regions(&overlapping),
            [([0, 1], [0, 2], [4, 1]), ([0, 0], [0, 1], [4, 1])]
        );
        assert_eq!(aspects(&overlapping), [both; 2]);
    }
    // A multisample texture of both stays unbuilt.
    let mut msaa = descriptor(
        TextureType::Type2DMultisample,
        PixelFormat::Depth32FloatStencil8,
        [8, 8, 1],
    );
    msaa.sample_count = 4;
    let msaa = side(&msaa, vk::Format::D32_SFLOAT_S8_UINT, None);
    assert_eq!(
        gap(texture_copy(
            &msaa,
            at(0, 0, [0; 3]),
            [8, 8, 1],
            &msaa,
            at(0, 0, [0; 3]),
            false
        )),
        Gap::TextureCopy {
            source: PixelFormat::Depth32FloatStencil8,
            destination: PixelFormat::Depth32FloatStencil8,
        }
    );
}

/// A buffer at `base` in its block, `len` bytes long, and a region in it at `offset`, rows
/// `bytes_per_row` and images `bytes_per_image` apart.
fn lies(base: u64, len: u64, offset: u64, bytes_per_row: u64, bytes_per_image: u64) -> Lies {
    Lies {
        buffer: vk::Buffer::from_raw(1),
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
    copy.slabs()
        .flatten()
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
fn a_buffer_copy_over_its_texture_s_own_bytes_copies_in_slabs_or_in_no_order() {
    // A texture in place at the first byte of the `VkBuffer` numbered 1: 16 texels of 4 bytes a
    // row, rows 64 bytes apart.
    let d = descriptor(TextureType::Type2D, PixelFormat::RGBA8Unorm, [16, 4, 1]);
    let texture = over(&d, 1, 0, 64);
    let copy =
        |size, lies, into_image| buffer_copy(&texture, at(0, 0, [0; 3]), size, lies, into_image);
    let slabs = |size, lies, into_image| buffer_copied(copy(size, lies, into_image)).slabs;
    let cut = |axis, thickness, reversed| {
        Some(Slabs {
            axis,
            thickness,
            reversed,
        })
    };
    let whole = [16, 4, 1];
    // The buffer's region a row on, into the texture: destination row 0 first, each row read
    // before it is written; out of the texture: the last row first.
    let row_on = lies(0, 1024, 64, 64, 0);
    assert_eq!(
        regions(&buffer_copied(copy(whole, row_on, true))),
        [
            (64, [0, 0, 0], [16, 1, 1]),
            (128, [0, 1, 0], [16, 1, 1]),
            (192, [0, 2, 0], [16, 1, 1]),
            (256, [0, 3, 0], [16, 1, 1]),
        ]
    );
    assert_eq!(
        regions(&buffer_copied(copy(whole, row_on, false))),
        [
            (256, [0, 3, 0], [16, 1, 1]),
            (192, [0, 2, 0], [16, 1, 1]),
            (128, [0, 1, 0], [16, 1, 1]),
            (64, [0, 0, 0], [16, 1, 1]),
        ]
    );
    // Two texels on, a region inside the texture's rows: slabs of two texels across, toward the
    // shift's far end first; one row high, its row stride unread.
    let within = lies(0, 1024, 8, 64, 0);
    assert_eq!(slabs([12, 4, 1], within, true), cut(0, 2, false));
    assert_eq!(slabs([12, 4, 1], within, false), cut(0, 2, true));
    assert_eq!(
        regions(&buffer_copied(copy(
            [5, 1, 1],
            lies(0, 1024, 8, 0, 0),
            true
        ))),
        [
            (8, [0, 0, 0], [2, 1, 1]),
            (16, [2, 0, 0], [2, 1, 1]),
            (24, [4, 0, 0], [1, 1, 1]),
        ]
    );
    // Whole rows two texels on: each region row across the end of a texture row and the start
    // of the next. No cut along one axis keeps the order, which the encoder refuses.
    assert_eq!(slabs(whole, within, true), None);
    assert_eq!(slabs(whole, within, false), None);
    // A region after the texture's rows, and one in another buffer: one slab.
    assert_eq!(
        slabs(whole, lies(0, 1024, 256, 64, 0), true),
        cut(0, 16, false)
    );
    let other = Lies {
        buffer: vk::Buffer::from_raw(2),
        ..row_on
    };
    assert_eq!(slabs(whole, other, false), cut(0, 16, false));
    // The texture's own bytes: nothing to copy.
    assert!(matches!(
        copy(whole, lies(0, 1024, 0, 64, 0), true),
        Ok(None)
    ));
    // A texture placed in a heap: a region whose bytes meet its memory copies in no order, one
    // whose bytes do not in one slab.
    let heap = in_heap(&d, 512, 256);
    let placed = |offset| {
        let layout = lies(0, 1024, offset, 64, 0);
        buffer_copied(buffer_copy(&heap, at(0, 0, [0; 3]), whole, layout, true)).slabs
    };
    assert_eq!(placed(400), None);
    assert_eq!(placed(256), cut(0, 16, false));
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
    // No plain copy of the image's bytes; a 32-bit float depth copies both ways.
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
    for into_image in [true, false] {
        let copy = buffer_copied(float_copy(into_image));
        assert_eq!(
            copy.unit.aspect.as_raw(),
            vk::ImageAspectFlags::DEPTH.as_raw(),
            "{into_image}"
        );
    }
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
fn a_copy_between_buffers_takes_ranges_inside_both_that_share_no_byte() {
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
    // Any offsets and any size: the region is the copy's own bytes.
    for (from, to, size) in [(2, 0, 4), (0, 2, 4), (0, 0, 6), (3, 5, 1)] {
        let region = buffer_to_buffer(
            span(one, 256, 64),
            from,
            span(vk::Buffer::from_raw(2), 0, 64),
            to,
            size,
        )
        .expect("a copy")
        .expect("a region");
        assert_eq!(
            (region.src_offset, region.dst_offset, region.size),
            (256 + from, to, size),
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
        Device, DispatchType, Error, Gap, HeapRange, IndexClass, MemoryClass, PixelFormat, Refusal,
        Region, Resource, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
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
                .replace_region(
                    region,
                    self.level,
                    self.slice,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    image,
                )
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
                .grant_indices(IndexClass::Texture, 1, 16)
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
            let closings = counters.closings();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            into_array.record(name, &command_buffer);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            // The copy's last barrier holds one global memory barrier beside its image barriers:
            // what it wrote is visible through any resource over the same bytes.
            let closed = counters.closings();
            assert_eq!(
                (closed.0 - closings.0, closed.1 - closings.1),
                (1, 1),
                "{name}: the copy's closing barrier and its global memory barrier"
            );
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

    /// A copy within one subresource of one image, from a texture's view or from the texture
    /// itself into the texture: disjoint regions copy, one submission and three barriers as any
    /// copy, everything around the region kept; the same region copies nothing and records no
    /// command.
    #[test]
    fn every_device_copies_within_one_subresource_of_a_texture_and_its_view() {
        each_device("blit_within", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let texture = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [8, 8, 1],
                TextureUsage::SHADER_READ,
                4096,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let view = (texture.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                Some(4097),
            ))
            .unwrap_or_else(|error| panic!("{name}: the view: {error}"));
            let level = |texture| Level {
                texture,
                slice: 0,
                level: 0,
                unit: 4,
                block: 1,
            };
            let (whole, through_view) = (level(&texture), level(&view));
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let mut source = whole.write(11);
            // The upload goes with a command buffer of its own, one with commands.
            let first = queue.command_buffer().expect("a command buffer");
            first
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            first.commit().expect("a commit");
            finished(name, &first);
            for (case, from) in [("from the view", &through_view), ("from itself", &whole)] {
                let copy = Expect {
                    from,
                    source: &source,
                    at: [0, 0, 0],
                    to: &whole,
                    before: source.clone(),
                    to_at: [4, 3, 0],
                    size: [4, 2, 1],
                };
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                copy.record(name, &command_buffer);
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: {case}");
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.submissions - before.submissions,
                        after.barriers - before.barriers,
                        after.bytes_copied - before.bytes_copied,
                    ),
                    (1, 3, 0),
                    "{name}: {case}: submissions, barriers, bytes copied"
                );
                let expected = copy.after();
                assert_eq!(whole.read(), expected, "{name}: {case}");
                source = expected;
            }

            // The same region: nothing to copy, no command, every byte kept.
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let at = |origin| TexturePlace {
                slice: 0,
                level: 0,
                origin,
            };
            (encoder.copy_from_texture(&view, at([2, 1, 0]), [3, 3, 1], &texture, at([2, 1, 0])))
                .unwrap_or_else(|error| panic!("{name}: the same region: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers
                ),
                (0, 0),
                "{name}: the same region records nothing"
            );
            assert_eq!(
                whole.read(),
                source,
                "{name}: the same region keeps every byte"
            );
            println!("{name}: copies within one subresource, through a view and of itself");
        });
    }

    /// Overlapping regions of one level and slice, which Metal leaves undefined, leave each
    /// destination texel holding what the source held before the copy, the answer an Apple M3
    /// Max gives (the conformance crate's `buffer_texture_blits.rs`), everything around them
    /// kept: shifted toward higher and toward lower coordinates along x and along y, from a
    /// texture's view into the texture, along the depth of a 3D texture, and by whole blocks of
    /// BC1 where the device makes it. Each copy is one submission and copies no byte of the
    /// runtime's own; each slab after the first costs one barrier more than a copy's three.
    #[test]
    fn every_device_copies_overlapping_regions_from_the_source_before_the_copy() {
        each_device("blit_overlapping", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let made = |texture_type, format, extent, index| {
                let read = TextureUsage::SHADER_READ;
                texture(&device, texture_type, format, extent, read, index)
            };
            let plane = (made(
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [8, 8, 1],
                4096,
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let view = (plane.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                Some(4097),
            ))
            .unwrap_or_else(|error| panic!("{name}: the view: {error}"));
            let volume = (made(TextureType::Type3D, PixelFormat::R8Unorm, [4, 4, 4], 4098))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let blocks = made(TextureType::Type2D, PixelFormat::BC1RGBA, [16, 16, 1], 4099);
            let level = |texture, unit, block| Level {
                texture,
                slice: 0,
                level: 0,
                unit,
                block,
            };
            let (whole, through_view) = (level(&plane, 4, 1), level(&view, 4, 1));
            let images = level(&volume, 1, 1);
            let bc1 = match &blocks {
                Ok(texture) => Some(level(texture, 8, 4)),
                Err(Error::Unsupported(Gap::TextureFormat { .. })) => {
                    println!("{name}: no BC1 texture on this device");
                    None
                }
                Err(error) => panic!("{name}: BC1: {error}"),
            };
            // Each case: the source and where in it, the destination and where in it, the size
            // in elements, and the slabs the copy takes.
            let mut cases = vec![
                (
                    "toward higher x",
                    &whole,
                    [0, 1, 0],
                    &whole,
                    [3, 1, 0],
                    [5, 2, 1],
                    2,
                ),
                (
                    "toward lower x",
                    &whole,
                    [2, 0, 0],
                    &whole,
                    [0, 0, 0],
                    [6, 3, 1],
                    3,
                ),
                (
                    "toward higher y",
                    &whole,
                    [1, 0, 0],
                    &whole,
                    [1, 1, 0],
                    [3, 6, 1],
                    6,
                ),
                (
                    "toward lower y",
                    &whole,
                    [0, 3, 0],
                    &whole,
                    [0, 1, 0],
                    [8, 5, 1],
                    3,
                ),
                (
                    "from the view",
                    &through_view,
                    [0; 3],
                    &whole,
                    [2, 2, 0],
                    [4, 4, 1],
                    2,
                ),
                (
                    "along the depth",
                    &images,
                    [0; 3],
                    &images,
                    [0, 0, 1],
                    [4, 4, 3],
                    3,
                ),
            ];
            if let Some(bc1) = &bc1 {
                cases.push((
                    "by whole BC1 blocks",
                    bc1,
                    [0; 3],
                    bc1,
                    [1, 0, 0],
                    [3, 2, 1],
                    3,
                ));
            }
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            for (seed, (case, from, at, to, to_at, size, slabs)) in (1..).zip(cases) {
                let source = to.write(seed);
                // The upload goes with a command buffer of its own, one with commands.
                let upload = queue.command_buffer().expect("a command buffer");
                upload
                    .record_for_test(&[], |_, _| {})
                    .expect("a command buffer's commands");
                upload.commit().expect("a commit");
                finished(name, &upload);
                let copy = Expect {
                    from,
                    source: &source,
                    at,
                    to,
                    before: source.clone(),
                    to_at,
                    size,
                };
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                copy.record(name, &command_buffer);
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: {case}");
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.submissions - before.submissions,
                        after.barriers - before.barriers,
                        after.bytes_copied - before.bytes_copied,
                    ),
                    (1, 2 + slabs, 0),
                    "{name}: {case}: submissions, barriers, bytes copied"
                );
                assert_eq!(to.read(), copy.after(), "{name}: {case}");
            }
            println!("{name}: overlapping copies, each from the source before the copy");
        });
    }

    /// Two textures in place over one Shared buffer, rows the device's own pitch apart, whose
    /// regions' bytes meet, which Metal leaves undefined, leave each destination texel holding
    /// what the source held before the copy, the answer an Apple M3 Max gives (the conformance
    /// crate's `buffer_texture_blits.rs`, its rows a row on), every other byte of the buffer kept:
    /// a texture a row on from the other, onto it and from it, in slabs of a row; one a memory
    /// alignment on, a region within the other's rows, onto it and from it, in slabs of that many
    /// bytes across; regions apart in one command. Each copy is one submission and copies no byte
    /// of the runtime's own; each slab after the first costs one barrier more than a copy's three.
    /// Whole rows a memory alignment on, each destination row across the end of its own source
    /// row and the start of the next, are `Gap::OverlappingCopy`, counted, and record nothing.
    #[test]
    fn every_device_copies_between_textures_over_one_buffer_from_the_source_before_the_copy() {
        use crate::texture::TextureObject;
        const HEIGHT: u32 = 4;
        each_device("blit_over_one_buffer", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let format = PixelFormat::RGBA8Unorm;
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, 64, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            // A place in the buffer's bytes that a linear image takes: a memory alignment on, a
            // texel at least.
            let memory = (device.linear_memory_alignment(&d)).expect("a linear image");
            let step = memory.max(4);
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Rows of four steps at least, at the device's own pitch: a texture a step on starts
            // inside the other's first row, and a region a step narrower than a row lies inside
            // the other's rows.
            let pitch = [256, 4 * step, align].into_iter().max().expect("a pitch");
            d.width = u32::try_from(pitch / 4).expect("a width");
            let len = (2 * u64::from(HEIGHT) + 1) * pitch;
            let block = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut indices = 4096..;
            let mut over = |offset: u64| {
                let texture = (buffer.new_texture(&d, indices.next(), offset, pitch))
                    .unwrap_or_else(|error| panic!("{name}: at {offset}: {error}"));
                assert!(
                    texture.copied_over().is_none(),
                    "{name}: at {offset}: in place"
                );
                (texture, offset)
            };
            let first = over(0);
            let row_on = over(pitch);
            let within = over(step);
            let apart = over(u64::from(HEIGHT) * pitch);
            let queue = device.new_command_queue(4).expect("a queue");
            // The textures' moves home go with a command buffer of their own.
            let homes = queue.command_buffer().expect("a command buffer");
            homes
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            homes.commit().expect("a commit");
            finished(name, &homes);
            let counters = device.counters();
            let width = d.width;
            let shift = u32::try_from(step / 4).expect("a shift");
            let across = (width - shift).div_ceil(shift);
            // Each case: the source and the destination with their offsets in the buffer, the
            // region's width (its height is the textures'), and the slabs the copy takes.
            let cases = [
                ("a row on", &first, &row_on, width, HEIGHT),
                ("a row back", &row_on, &first, width, HEIGHT),
                (
                    "within the rows, on",
                    &first,
                    &within,
                    width - shift,
                    across,
                ),
                (
                    "within the rows, back",
                    &within,
                    &first,
                    width - shift,
                    across,
                ),
                ("apart", &first, &apart, width, 1),
            ];
            let at = TexturePlace::default();
            for (seed, (case, (from, from_at), (to, to_at), columns, slabs)) in (1..).zip(cases) {
                let before = pattern(len as usize, seed);
                write(&buffer, &before);
                settled(&device);
                let metrics = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                (encoder.copy_from_texture(from, at, [columns, HEIGHT, 1], to, at))
                    .unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                let rows = (0..u64::from(HEIGHT)).map(|row| {
                    (
                        (from_at + row * pitch) as usize,
                        (to_at + row * pitch) as usize,
                    )
                });
                let want = moved(before.clone(), &before, rows, 4 * columns as usize);
                assert_eq!(read(&buffer), want, "{name}: {case}: the buffer's bytes");
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.submissions - metrics.submissions,
                        after.barriers - metrics.barriers,
                        after.bytes_copied - metrics.bytes_copied,
                    ),
                    (1, 2 + u64::from(slabs), 0),
                    "{name}: {case}: submissions, barriers, bytes copied"
                );
            }

            // Whole rows a memory alignment on: refused, nothing recorded, every byte kept.
            let before = read(&buffer);
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let refused =
                encoder.copy_from_texture(&first.0, at, [width, HEIGHT, 1], &within.0, at);
            assert!(
                matches!(
                    refused,
                    Err(Error::Unsupported(Gap::OverlappingCopy { .. }))
                ),
                "{name}: whole rows a memory alignment on: {refused:?}"
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.overlapping_copies_refused - metrics.overlapping_copies_refused,
                ),
                (0, 0, 1),
                "{name}: the refused copy records nothing and is counted"
            );
            assert_eq!(
                read(&buffer),
                before,
                "{name}: the refused copy keeps every byte"
            );
            println!(
                "{name}: copies between textures over one buffer from the source before the copy, \
                 rows {pitch} bytes apart, a memory alignment of {memory}"
            );
        });
    }

    /// Between a buffer and a texture in place over its bytes, on every device: the buffer's
    /// region a row on from the texture and a row back, into the texture and out of it, in slabs
    /// of a row; a region a memory alignment on within the texture's rows, into it and out of it,
    /// in slabs of that many bytes across; a region apart in one command. Each leaves the whole
    /// buffer holding the source's bytes from before the copy, every other byte kept, at one
    /// submission and no byte copied of the runtime's own, each slab after the first costing one
    /// barrier more than a copy's three. Whole rows a memory alignment on are
    /// `Gap::OverlappingCopy`, counted, and record nothing.
    #[test]
    fn every_device_copies_between_a_buffer_and_a_texture_over_it_from_the_source_before_the_copy()
    {
        use crate::texture::TextureObject;
        const HEIGHT: u32 = 4;
        each_device("blit_buffer_under_texture", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .expect("a grant");
            let format = PixelFormat::RGBA8Unorm;
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, 64, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let memory = (device.linear_memory_alignment(&d)).expect("a linear image");
            let step = memory.max(4);
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = [256, 4 * step, align].into_iter().max().expect("a pitch");
            d.width = u32::try_from(pitch / 4).expect("a width");
            // The texture a row in, so that a region may start a row before it, and rows after
            // it for a region a row on and one apart.
            let place = pitch;
            let len = (2 * u64::from(HEIGHT) + 2) * pitch;
            let block = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = (buffer.new_texture(&d, Some(4096), place, pitch))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(texture.copied_over().is_none(), "{name}: in place");
            let queue = device.new_command_queue(4).expect("a queue");
            let homes = queue.command_buffer().expect("a command buffer");
            homes
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            homes.commit().expect("a commit");
            finished(name, &homes);
            let counters = device.counters();
            let width = d.width;
            let shift = u32::try_from(step / 4).expect("a shift");
            let across = (width - shift).div_ceil(shift);
            let at = TexturePlace::default();
            // Each case: into the texture or out of it, the region's offset in the buffer, its
            // width (its height is the texture's), and the slabs the copy takes.
            let cases = [
                (
                    "into the texture, a row on",
                    true,
                    place + pitch,
                    width,
                    HEIGHT,
                ),
                ("into the texture, a row back", true, 0, width, HEIGHT),
                (
                    "out of the texture, a row on",
                    false,
                    place + pitch,
                    width,
                    HEIGHT,
                ),
                ("out of the texture, a row back", false, 0, width, HEIGHT),
                (
                    "into the texture, within its rows",
                    true,
                    place + step,
                    width - shift,
                    across,
                ),
                (
                    "out of the texture, within its rows",
                    false,
                    place + step,
                    width - shift,
                    across,
                ),
                ("apart", true, place + u64::from(HEIGHT) * pitch, width, 1),
            ];
            for (seed, (case, into, offset, columns, slabs)) in (1..).zip(cases) {
                let before = pattern(len as usize, seed);
                write(&buffer, &before);
                settled(&device);
                let metrics = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                let layout = BufferLayout {
                    offset,
                    bytes_per_row: pitch,
                    bytes_per_image: 0,
                };
                let size = [columns, HEIGHT, 1];
                let copied = match into {
                    true => encoder.copy_from_buffer(&buffer, layout, size, &texture, at),
                    false => {
                        encoder.copy_from_texture_to_buffer(&texture, at, size, &buffer, layout)
                    }
                };
                copied.unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                let rows = (0..u64::from(HEIGHT)).map(|row| {
                    let (region, image) = (offset + row * pitch, place + row * pitch);
                    match into {
                        true => (region as usize, image as usize),
                        false => (image as usize, region as usize),
                    }
                });
                let want = moved(before.clone(), &before, rows, 4 * columns as usize);
                assert_eq!(read(&buffer), want, "{name}: {case}: the buffer's bytes");
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.submissions - metrics.submissions,
                        after.barriers - metrics.barriers,
                        after.bytes_copied - metrics.bytes_copied,
                        after.overlapping_copies_refused - metrics.overlapping_copies_refused,
                    ),
                    (1, 2 + u64::from(slabs), 0, 0),
                    "{name}: {case}: submissions, barriers, bytes copied, refusals"
                );
            }

            // Whole rows a memory alignment on: refused and counted, nothing recorded, every byte
            // kept.
            let before = read(&buffer);
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let layout = BufferLayout {
                offset: place + step,
                bytes_per_row: pitch,
                bytes_per_image: 0,
            };
            let refused =
                encoder.copy_from_buffer(&buffer, layout, [width, HEIGHT, 1], &texture, at);
            assert!(
                matches!(
                    refused,
                    Err(Error::Unsupported(Gap::OverlappingCopy { .. }))
                ),
                "{name}: whole rows a memory alignment on: {refused:?}"
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.barriers - metrics.barriers,
                    after.overlapping_copies_refused - metrics.overlapping_copies_refused,
                ),
                (0, 0, 1),
                "{name}: the refused copy records nothing and is counted"
            );
            assert_eq!(
                read(&buffer),
                before,
                "{name}: the refused copy keeps every byte"
            );
            println!(
                "{name}: copies between a buffer and a texture over it from the source before \
                 the copy, rows {pitch} bytes apart, a memory alignment of {memory}"
            );
        });
    }

    /// Textures placed in one heap, on every device: a copy between two whose ranges meet, and
    /// between a heap buffer and a texture whose ranges meet, either way, is
    /// `Gap::OverlappingCopy`, counted, and records nothing, the device laying the texels out as
    /// it chooses. Copies between a texture and another, and a heap buffer, whose ranges do not
    /// meet run: bytes from a Shared buffer through both textures and the heap buffer come back
    /// as they went in, at one submission.
    #[test]
    fn every_device_refuses_a_copy_between_heap_resources_whose_memory_meets() {
        const SIDE: u32 = 16;
        each_device("blit_heap_aliases", |name, _, device| {
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.storage_mode = StorageMode::Private;
            let fit = (device.heap_texture_size_and_align(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let bytes = u64::from(SIDE * SIDE * 4);
            // Two textures apart, then a buffer after both; an alias of the first texture, and a
            // buffer over its bytes.
            let second = fit.size.next_multiple_of(fit.align);
            let after = (second + fit.size).next_multiple_of(256);
            let heap_len = after + bytes;
            let block = (device.take_block(MemoryClass::Private, heap_len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let heap = HeapRange {
                block: block.id,
                offset: 0,
                len: heap_len,
            };
            let texture = |offset| {
                (device.new_heap_texture_for_views(&d, heap, offset))
                    .unwrap_or_else(|error| panic!("{name}: at {offset}: {error}"))
            };
            let heap_buffer = |offset| {
                (device.new_heap_buffer(heap, offset, bytes))
                    .unwrap_or_else(|error| panic!("{name}: at {offset}: {error}"))
            };
            let (first, apart, alias) = (texture(0), texture(second), texture(0));
            let (beyond, under) = (heap_buffer(after), heap_buffer(0));
            let shared = (device.take_block(MemoryClass::Shared, 2 * bytes))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let source = (device.new_buffer(shared.id, 0, bytes, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(shared.id, bytes, bytes, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let homes = queue.command_buffer().expect("a command buffer");
            homes
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            homes.commit().expect("a commit");
            finished(name, &homes);
            let counters = device.counters();
            let at = TexturePlace::default();
            let size = [SIDE, SIDE, 1];
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: u64::from(SIDE * 4),
                bytes_per_image: 0,
            };

            // Ranges that meet: each copy refused, counted, and nothing recorded.
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            let refusals = [
                (
                    "texture onto its alias",
                    encoder.copy_from_texture(&first, at, size, &alias, at),
                ),
                (
                    "texture into a buffer over its bytes",
                    encoder.copy_from_texture_to_buffer(&first, at, size, &under, layout),
                ),
                (
                    "buffer into a texture over its bytes",
                    encoder.copy_from_buffer(&under, layout, size, &first, at),
                ),
            ];
            for (case, refused) in refusals {
                assert!(
                    matches!(
                        refused,
                        Err(Error::Unsupported(Gap::OverlappingCopy { .. }))
                    ),
                    "{name}: {case}: {refused:?}"
                );
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            let after_refusals = counters.metrics();
            assert_eq!(
                (
                    after_refusals.submissions - metrics.submissions,
                    after_refusals.barriers - metrics.barriers,
                    after_refusals.overlapping_copies_refused - metrics.overlapping_copies_refused,
                ),
                (0, 0, 3),
                "{name}: the refused copies record nothing and are counted"
            );

            // Ranges apart: bytes through both textures and the heap buffer, as they went in.
            let want = pattern(bytes as usize, 7);
            write(&source, &want);
            settled(&device);
            let metrics = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            (encoder.copy_from_buffer(&source, layout, size, &first, at))
                .unwrap_or_else(|error| panic!("{name}: into the first texture: {error}"));
            (encoder.copy_from_texture(&first, at, size, &apart, at))
                .unwrap_or_else(|error| panic!("{name}: into the texture apart: {error}"));
            (encoder.copy_from_texture_to_buffer(&apart, at, size, &beyond, layout))
                .unwrap_or_else(|error| panic!("{name}: into the heap buffer: {error}"));
            (encoder.copy_from_buffer_to_buffer(&beyond, 0, &out, 0, bytes))
                .unwrap_or_else(|error| panic!("{name}: out of the heap buffer: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            assert_eq!(read(&out), want, "{name}: the bytes through the heap");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - metrics.submissions,
                    after.overlapping_copies_refused - metrics.overlapping_copies_refused,
                ),
                (1, 0),
                "{name}: submissions, refusals"
            );
            println!(
                "{name}: copies between heap resources whose memory meets refused, apart run, \
                 a texture of {} bytes at an alignment of {}",
                fit.size, fit.align
            );
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
                .grant_indices(IndexClass::Texture, 1, 16)
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
            let closings = device.counters().closings();
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
            // Each copy, into a texture or out of one, ends on one barrier that holds one global
            // memory barrier beside its image barriers.
            let copies = 3 + u64::from(bc1.is_some());
            let closed = device.counters().closings();
            assert_eq!(
                (closed.0 - closings.0, closed.1 - closings.1),
                (copies, copies),
                "{name}: the copies' closing barriers and their global memory barriers"
            );
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
            let managed =
                (crate::tests::managed::on_device::new_managed(&device, block.id, 512, 64, 64))
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

    /// A copy between buffers at offsets and of sizes of no whole words lands each byte of its
    /// range and no other, as plain Metal's copy does on macOS (probe
    /// `blit_buffer_copy_unaligned.m` in the conformance crate: each of these copies exact on an
    /// Apple M3 Max, macOS 27.0.1, job m3-max-69613, and on an AMD Radeon Pro 5300M and an Intel
    /// UHD Graphics 630, macOS 26.7.1, job m3-max-69614).
    #[test]
    fn every_device_copies_between_buffers_at_any_offset_and_size() {
        const LEN: u64 = 65_536;
        each_device("blit_buffer_any_offset", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 2 * LEN)).expect("a Shared block");
            let source = (device.new_buffer(block.id, 0, LEN, Some(0))).expect("a buffer");
            let destination = (device.new_buffer(block.id, LEN, LEN, Some(0))).expect("a buffer");
            let (from, into) = (pattern(LEN as usize, 31), pattern(LEN as usize, 47));
            let queue = device.new_command_queue(4).expect("a queue");
            for (at, to, size) in [
                (0, 0, 62_302),
                (0, 0, 1),
                (0, 0, 3),
                (1, 0, 8),
                (0, 2, 8),
                (3, 5, 4097),
            ] {
                write(&source, &from);
                write(&destination, &into);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                encoder
                    .copy_from_buffer_to_buffer(&source, at, &destination, to, size)
                    .unwrap_or_else(|error| panic!("{name}: {at} {to} {size}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                let (at, to, size) = (at as usize, to as usize, size as usize);
                let mut want = into.clone();
                want[to..to + size].copy_from_slice(&from[at..at + size]);
                let got = read(&destination);
                let differs = got.iter().zip(&want).position(|(got, want)| got != want);
                assert_eq!(
                    differs, None,
                    "{name}: {at} {to} {size}: a byte that differs"
                );
                assert!(
                    read(&source) == from,
                    "{name}: {at} {to} {size}: the source"
                );
                println!("{name}: {size} bytes from {at} to {to}: each byte of the range copied");
            }
        });
    }

    /// Returns once the block-maker thread has nothing to make (it makes an allocation of a
    /// class that holds no wholly free unit, which a count of a test's own work must not take
    /// in), and the reaper has made the pipeline cache, which sets its gauge.
    pub(crate) fn settled(device: &Device) {
        within("a wholly free unit in each class", || {
            let pool = &device.core().pool;
            pool.stocked(MemoryClass::Private) && pool.stocked(MemoryClass::Shared)
        });
        within("the pipeline cache", || device.core().caches.is_seeded());
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
            let managed =
                (crate::tests::managed::on_device::new_managed(&device, block.id, 256, 64, 64))
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
            // A colour render target takes an index.
            (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
            let places = [place(0), place(1024)];
            let texture = (device.new_managed_texture(&d, Some(4096), &named, &places))
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
