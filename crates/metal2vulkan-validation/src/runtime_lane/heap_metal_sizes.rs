//! Heap textures whose size and alignment Metal answers otherwise than as its GPU's image of
//! them (`texture/heap.rs`, `metal_heap_answer`), on the device the cfg selects, under the
//! validation layer:
//!
//! - A 4:2:2 texture (`GBGR422`, `BGRG422`) answers its rows of 2-byte texels at a multiple of
//!   16 bytes, the last unpadded, at an alignment of 16, as Metal does (an M4 under macOS
//!   27.0.1: 44 bytes for 6 x 3, 148 for 10 x 5, 6000 for 1000 x 3; an M3 Max under 26.7 alike, the
//!   consumer's report), and its texels lie in a `Shared` heap's bytes where Metal's do: a buffer
//!   over them reads each row at that pitch, the bytes between untouched. An `X32_Stencil8` texture
//!   answers what a `Stencil8` texture of the descriptor answers, as Metal sizes it (the M4, in
//!   every shape measured); `Unspecialized` answers 0 bytes at its texture type's alignment (128
//!   for 1D and 2D, 16 for a texture buffer, 16 KiB for the arrays, cubes and 3D textures: the M3
//!   Max under macOS 26.7), and its texture is refused as Metal's nil (`Error::Nil`,
//!   `TextureProblem::Unspecialized`); one of a multisample type or of `ShaderAtomic` usage as
//!   Metal stops it, plain too.
//! - A two-plane extended-range texture (`RGB10A8_2P_XR10` and its sRGB twin) answers Metal's
//!   size and alignment in every texture type Metal makes (an M4 under macOS 27.0.1: 20480 at 16
//!   for 64 x 64, its first level and slice alone for a mipmapped or 3D one, 128 for a 2D array),
//!   and a cube or cube array 0 bytes at 16 and no texture: Metal's nil from the device
//!   (`Error::Nil`), and from a placement heap, where Metal makes one, a refusal as not
//!   implemented.
//! - Apple's private usage bits (an M4 under macOS 27.0.1, plain and validated): a texture made
//!   with `NoConcurrentAccess` (0x4000), `Video` (0x8000) and `BlockWritesOnly` (0x10000) by the
//!   device, a heap, a buffer or as a shared texture answers them in its usage, as its views and
//!   shared opens do, and its heap answer is the one without them; usage bit 0x8 is taken and
//!   dropped from the texture's usage (a texture of it alone answers none), while the heap answer
//!   of a descriptor of it alone is a usage's, not `Unknown`'s.
//! - Two such textures placed side by side at the answers in a placement heap keep their own
//!   texels on every device: each blitted from a buffer, both read back after. Where the
//!   device's image needs more than Metal's answer, the runtime holds the texture as a shadow of
//!   the heap's bytes in memory of its own, so the answer stays Metal's; a two-plane texture's
//!   image lies apart from the heap's bytes, in memory of its own.

use super::at_plan;
use super::degrade_notices::{self as notices, counts};
use super::host_memory::on_device_with;
use super::host_memory::{on_device, run};
use super::Messages;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Degradation, Device, Error, HeapDescriptor, HeapType,
    IdPlacement, Invalid, Nil, Origin, Problem, Region, ResourceOptions, Size, SizeAndAlign,
    Texture, TextureDescriptor, TextureProblem,
};

/// The alignment Metal's heap query answers, with 0 bytes, for an array, a cube, a cube array, a 3D
/// texture or a multisample array of a format it makes no texture of and that has no planes
/// (`Unspecialized`, `RGBA16Float_XR`, `AnyDepth`, `AnyStencil`): 128, as for a 1D or 2D texture
/// (an Apple M3 Max under macOS 27.0.1, an M4 alike).
pub(super) const STACKED_NIL_ALIGNMENT: u64 = 128;

/// A Shared buffer holding `bytes`.
fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// `bytes`, rows of `row` bytes, blitted into level 0 and slice 0 of `texture`, `width` x
/// `height` texels.
fn upload(queue: &CommandQueue, texture: &Texture, bytes: &[u8], row: u64, extent: [u64; 2]) {
    let staging = staged(queue.device(), bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &staging,
        0,
        row,
        bytes.len() as u64,
        Size::new(extent[0], extent[1], 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// Level 0 and slice 0 of `texture`, `width` x `height` texels in rows of `row` bytes, read back
/// through a blit.
fn download(queue: &CommandQueue, texture: &Texture, row: u64, extent: [u64; 2]) -> Vec<u8> {
    let bytes = row * extent[1];
    let copy = queue
        .device()
        .new_buffer(bytes, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(extent[0], extent[1], 1),
        &copy,
        0,
        row,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    // SAFETY: the buffer holds `bytes` bytes the completed copy wrote.
    unsafe { std::slice::from_raw_parts(copy.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// `n` bytes from `seed`, each different from its neighbours'.
fn pattern(n: u64, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// Two textures of `descriptor` (2D, `extent`, `texel` bytes a texel), side by side at its
/// `answer` in a placement heap: each keeps its own texels, the second's written first, reading
/// back what one texture of the descriptor alone reads back after the same blits (its bytes, but
/// where Metal's blits address other rows: a two-plane texture's at a width off 4,
/// `two_plane_copies.rs`).
fn side_by_side(
    device: &Device,
    descriptor: &TextureDescriptor,
    answer: SizeAndAlign,
    extent: [u64; 2],
    texel: u64,
    what: &str,
) {
    let queue = device.new_command_queue().unwrap();
    let next = answer.size.next_multiple_of(answer.align);
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Placement,
            (next + answer.size).max(1),
            descriptor.resource_options,
        ))
        .unwrap();
    let first = heap.new_texture_at_offset(descriptor, 0).unwrap();
    let second = heap.new_texture_at_offset(descriptor, next).unwrap();
    let row = extent[0] * texel;
    let (a, b) = (pattern(row * extent[1], 3), pattern(row * extent[1], 101));
    upload(&queue, &second, &b, row, extent);
    upload(&queue, &first, &a, row, extent);
    let alone = |bytes: &[u8]| {
        let texture = device.new_texture(descriptor).unwrap();
        upload(&queue, &texture, bytes, row, extent);
        download(&queue, &texture, row, extent)
    };
    assert_eq!(
        download(&queue, &first, row, extent),
        alone(&a),
        "{what}: the first"
    );
    assert_eq!(
        download(&queue, &second, row, extent),
        alone(&b),
        "{what}: the second"
    );
}

/// A 4:2:2 texture of `descriptor` (`Shared`, `extent`) at offset 0 of a placement heap, written
/// by `replace_region`: a buffer over its `answer` reads each row at a multiple of 16 bytes, the
/// bytes between untouched, as a buffer over Metal's reads them.
fn metal_s_rows(
    device: &Device,
    descriptor: &TextureDescriptor,
    answer: SizeAndAlign,
    extent: [u64; 2],
    what: &str,
) {
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Placement,
            answer.size,
            descriptor.resource_options,
        ))
        .unwrap();
    let over = heap
        .new_buffer_at_offset(answer.size, descriptor.resource_options, 0)
        .unwrap();
    let contents = over.contents().unwrap();
    // SAFETY: the buffer holds `answer.size` bytes, and no command uses them.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xEE, answer.size as usize) };
    let texture = heap.new_texture_at_offset(descriptor, 0).unwrap();
    let row = extent[0] * 2;
    let texels = pattern(row * extent[1], 3);
    texture
        .replace_region(
            Region::new_2d(0, 0, extent[0], extent[1]),
            0,
            0,
            &texels,
            row,
            0,
        )
        .unwrap();
    // SAFETY: as above; the write is the CPU's, done.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), answer.size as usize) };
    let pitch = row.next_multiple_of(16);
    for y in 0..extent[1] {
        let at = (y * pitch) as usize;
        let want = &texels[(y * row) as usize..((y + 1) * row) as usize];
        assert_eq!(&bytes[at..at + row as usize], want, "{what}: row {y}");
        let gap = (at + row as usize)..(at + pitch as usize).min(bytes.len());
        assert!(
            bytes[gap].iter().all(|&b| b == 0xEE),
            "{what}: after row {y}"
        );
    }
}

/// Metal's answer for a 4:2:2 texture on every device; two side by side in a `Private` and a
/// `Shared` placement heap keep their texels, and a `Shared` heap's bytes hold them where
/// Metal's do.
#[test]
fn four_two_two_heap_textures_take_metal_s_size_and_layout() {
    on_device_with(at_plan, |device| {
        // Metal's sizes.
        let shapes = [
            ([64, 32], 4096),
            ([8, 8], 128),
            ([6, 3], 44),
            ([10, 5], 148),
            ([1000, 3], 6000),
        ];
        for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
            for ([width, height], size) in shapes {
                for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                    let what = format!("{pf:?} {width}x{height} {options:?}");
                    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
                    descriptor.usage = TextureUsage::SHADER_READ;
                    descriptor.resource_options = options;
                    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                    assert_eq!((answer.size, answer.align), (size, 16), "{what}");
                    side_by_side(device, &descriptor, answer, [width, height], 2, &what);
                    if options == ResourceOptions::shared() {
                        metal_s_rows(device, &descriptor, answer, [width, height], &what);
                    }
                }
            }
        }
    });
}

/// An `X32_Stencil8` texture answers a `Stencil8` one's size and alignment on every device; two
/// side by side keep their stencil.
#[test]
fn x32_stencil8_heap_textures_take_a_stencil8_texture_s_size() {
    on_device_with(at_plan, |device| {
        let shapes = [
            (TextureType::Type2D, [64, 32], 1, 1),
            (TextureType::Type2D, [6, 3], 1, 1),
            (TextureType::Type2D, [1000, 3], 1, 1),
            (TextureType::Type2D, [64, 64], 7, 1),
            (TextureType::Type2DArray, [64, 64], 1, 3),
            (TextureType::Cube, [32, 32], 6, 1),
        ];
        for (texture_type, [width, height], levels, array_length) in shapes {
            let of = |pf| {
                let mut d = TextureDescriptor::texture_2d(pf, width, height, levels > 1);
                d.texture_type = texture_type;
                d.mipmap_level_count = levels;
                d.array_length = array_length;
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::private();
                d
            };
            let what = format!("{texture_type:?} {width}x{height} levels {levels}");
            let x32 = of(PixelFormat::X32Stencil8);
            let answer = device.heap_texture_size_and_align(&x32).unwrap();
            let stencil8 = device
                .heap_texture_size_and_align(&of(PixelFormat::Stencil8))
                .unwrap();
            assert_eq!(answer, stencil8, "{what}");
            eprintln!(
                "HEAPTEX x32 {what}: size {} align {}",
                answer.size, answer.align
            );
            if texture_type == TextureType::Type2D && levels == 1 {
                side_by_side(device, &x32, answer, [width, height], 1, &what);
            }
        }
    });
}

/// A two-plane extended-range texture answers Metal's size and alignment on every device, in each
/// storage mode; two side by side keep their color words, which blits move (`two_planes.rs`); a
/// cube and a cube array answer 0 bytes at 16, and no texture is made of either, on the device or
/// in a heap.
#[test]
fn two_plane_heap_textures_take_metal_s_size() {
    on_device_with(at_plan, |device| {
        // (type, width, height, depth, levels, slices, Metal's size and alignment).
        let shapes = [
            (TextureType::Type2D, [64, 64, 1], 1, 1, (20480, 16)),
            (TextureType::Type2D, [1, 1, 1], 1, 1, (16385, 16)),
            (TextureType::Type2D, [7, 3, 1], 1, 1, (16423, 16)),
            (TextureType::Type2D, [1000, 3, 1], 1, 1, (19400, 16)),
            (TextureType::Type2D, [64, 65, 1], 1, 1, (36928, 16)),
            (TextureType::Type2D, [7, 600, 1], 1, 1, (42359, 16)),
            (TextureType::Type2D, [1920, 1080, 1], 1, 1, (10380288, 16)),
            (TextureType::Type2D, [64, 64, 1], 7, 1, (20480, 16)),
            (TextureType::Type2D, [17, 9, 1], 5, 1, (16657, 16)),
            (TextureType::Type2DArray, [64, 64, 1], 1, 3, (61440, 128)),
            (TextureType::Type2DArray, [7, 3, 1], 1, 5, (17024, 128)),
            (TextureType::Type2DArray, [64, 65, 1], 1, 2, (57600, 128)),
            (TextureType::Type2DArray, [17, 9, 1], 3, 2, (17152, 128)),
            (TextureType::Type1D, [64, 1, 1], 1, 1, (16448, 16)),
            (TextureType::Type1D, [5000, 1, 1], 1, 1, (37768, 16)),
            (TextureType::Type1DArray, [7, 1, 1], 1, 5, (17024, 16)),
            (TextureType::Type1DArray, [5000, 1, 1], 1, 4, (102400, 16)),
            (TextureType::Type3D, [64, 64, 4], 1, 1, (20480, 16)),
            (TextureType::Type3D, [32, 32, 20], 1, 1, (17408, 16)),
            (TextureType::Type3D, [64, 64, 4], 3, 1, (20480, 16)),
        ];
        for pf in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            for (texture_type, [width, height, depth], levels, slices, (size, align)) in shapes {
                for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                    let what = format!(
                        "{pf:?} {texture_type:?} {width}x{height}x{depth} levels {levels} \
                         slices {slices} {options:?}"
                    );
                    let mut d = TextureDescriptor::texture_2d(pf, width, height, levels > 1);
                    d.texture_type = texture_type;
                    d.depth = depth;
                    d.mipmap_level_count = levels;
                    d.array_length = slices;
                    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    d.resource_options = options;
                    let answer = device
                        .heap_texture_size_and_align(&d)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!((answer.size, answer.align), (size, align), "{what}");
                    let small = width * height <= 4096;
                    if texture_type == TextureType::Type2D && levels == 1 && small {
                        side_by_side(device, &d, answer, [width, height], 4, &what);
                    }
                }
            }
            for (texture_type, slices) in [(TextureType::Cube, 1), (TextureType::CubeArray, 2)] {
                let what = format!("{pf:?} {texture_type:?}");
                let mut d = TextureDescriptor::texture_2d(pf, 32, 32, false);
                d.texture_type = texture_type;
                d.array_length = slices;
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::private();
                let answer = device.heap_texture_size_and_align(&d).unwrap();
                assert_eq!((answer.size, answer.align), (0, 16), "{what}");
                let nil = |call| {
                    Some(Error::Nil(Nil::new(
                        call,
                        Problem::Texture(TextureProblem::TypeForFormat),
                    )))
                };
                assert_eq!(device.new_texture(&d).err(), nil("new_texture"), "{what}");
                let heap = device
                    .new_heap(&HeapDescriptor::new(
                        HeapType::Placement,
                        1 << 16,
                        ResourceOptions::private(),
                    ))
                    .unwrap();
                // A placement heap's Metal makes, a gap of the runtime's.
                assert!(
                    matches!(
                        heap.new_texture_at_offset(&d, 0),
                        Err(Error::Unsupported(u))
                            if u.feature == Feature::Format && u.needs == Needs::NotImplemented
                    ),
                    "{what}"
                );
            }
        }
    });
}

/// `Unspecialized`: Metal answers 0 bytes from the heap query, at its texture type's alignment
/// ([`STACKED_NIL_ALIGNMENT`] for the arrays, cubes and 3D textures), and nil for the texture, a
/// heap's (automatic and placement) and a buffer's (`newTextureWithDescriptor:offset:bytesPerRow:` and the linear form), in every storage
/// mode and usage; it stops a multisample one and one of `ShaderAtomic` usage, plain too, in the
/// query and the calls alike.
#[test]
fn unspecialized_answers_metal_s_nothing() {
    on_device(|device| {
        let nil = |call| {
            Error::Nil(Nil::new(
                call,
                Problem::Texture(TextureProblem::Unspecialized),
            ))
        };
        let of = |texture_type: TextureType, usage: TextureUsage, options: ResourceOptions| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::Unspecialized, 16, 16, false);
            d.texture_type = texture_type;
            match texture_type {
                TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer => {
                    d.height = 1
                }
                TextureType::Type3D => d.depth = 2,
                _ => {}
            }
            if matches!(
                texture_type,
                TextureType::Type1DArray
                    | TextureType::Type2DArray
                    | TextureType::CubeArray
                    | TextureType::Type2DMultisampleArray
            ) {
                d.array_length = 2;
            }
            if matches!(
                texture_type,
                TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
            ) {
                d.sample_count = 4;
            }
            d.usage = usage;
            d.resource_options = options;
            d
        };
        let stacked = STACKED_NIL_ALIGNMENT;
        let types = [
            (TextureType::Type1D, 128),
            (TextureType::Type1DArray, stacked),
            (TextureType::Type2D, 128),
            (TextureType::Type2DArray, stacked),
            (TextureType::Cube, stacked),
            (TextureType::CubeArray, stacked),
            (TextureType::Type3D, stacked),
            (TextureType::TextureBuffer, 16),
        ];
        let usages = [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TextureUsage::RENDER_TARGET,
        ];
        for options in [ResourceOptions::private(), ResourceOptions::shared()] {
            for (texture_type, align) in types {
                for usage in usages {
                    let what = format!("{texture_type:?} {usage:?} {options:?}");
                    let d = of(texture_type, usage, options);
                    let answer = device
                        .heap_texture_size_and_align(&d)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!((answer.size, answer.align), (0, align), "{what}");
                    assert_eq!(
                        device.new_texture(&d).err(),
                        Some(nil("new_texture")),
                        "{what}"
                    );
                }
                // Metal stops `ShaderAtomic` usage, plain too.
                let atomic = TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE
                    | TextureUsage::SHADER_ATOMIC;
                let d = of(texture_type, atomic, options);
                let stopped = |call| {
                    Some(Error::Invalid(Invalid::new(
                        call,
                        Problem::Texture(TextureProblem::UsageForFormat),
                    )))
                };
                assert_eq!(
                    device.heap_texture_size_and_align(&d).err(),
                    stopped("heap_texture_size_and_align"),
                    "{texture_type:?} atomic {options:?}"
                );
                assert_eq!(
                    device.new_texture(&d).err(),
                    stopped("new_texture"),
                    "{texture_type:?} atomic {options:?}"
                );
            }
            // And a multisample one, whatever its sample count.
            for texture_type in [
                TextureType::Type2DMultisample,
                TextureType::Type2DMultisampleArray,
            ] {
                let d = of(texture_type, TextureUsage::RENDER_TARGET, options);
                assert_eq!(
                    device.heap_texture_size_and_align(&d).err(),
                    Some(Error::Invalid(Invalid::new(
                        "heap_texture_size_and_align",
                        Problem::Texture(TextureProblem::TypeForFormat),
                    ))),
                    "{texture_type:?} {options:?}"
                );
            }
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::Unspecialized, 64, 32, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = options;
            let automatic = device
                .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 20, options))
                .unwrap();
            assert_eq!(
                automatic.new_texture(&descriptor).err(),
                Some(nil("new_texture")),
                "{options:?}"
            );
            let placement = device
                .new_heap(&HeapDescriptor::new(HeapType::Placement, 1 << 20, options))
                .unwrap();
            assert_eq!(
                placement.new_texture_at_offset(&descriptor, 0).err(),
                Some(nil("new_texture_at_offset")),
                "{options:?}"
            );
            let buffer = device.new_buffer(1 << 16, options).unwrap();
            assert_eq!(
                buffer.new_texture(&descriptor, 0, 256).err(),
                Some(nil("new_texture")),
                "{options:?}"
            );
            assert_eq!(
                buffer.new_linear_texture(&descriptor, 0, 256, 0).err(),
                Some(nil("new_linear_texture")),
                "{options:?}"
            );
        }
    });
}

/// The formats of which Metal makes no texture alone (`RGBA16Float_XR`, `AnyDepth`,
/// `AnyStencil` and the two-plane YCbCr formats, `YCBCR8_420_2P` here): every rule Metal's
/// validation stops comes first, from every creator and the heap query alike (`Error::Invalid`),
/// and then Metal's nil (`Error::Nil`) from the device, either heap, a buffer and its linear
/// form, a placement heap whose storage is not the descriptor's too; the query answers 0 bytes at
/// the type's alignment, the M3 Max's ([`STACKED_NIL_ALIGNMENT`] for the arrays, cubes and 3D
/// textures).
#[test]
fn formats_without_textures_answer_metal_s_stops_then_its_nil() {
    use TextureType as T;
    on_device(|device| {
        let stopped = |call, problem| Some(Error::Invalid(Invalid::new(call, problem)));
        let nil = |call| {
            Some(Error::Nil(Nil::new(
                call,
                Problem::Texture(TextureProblem::NoTextureOfFormat),
            )))
        };
        let query = "heap_texture_size_and_align";
        for options in [ResourceOptions::private(), ResourceOptions::shared()] {
            let heap = |heap_type| {
                device
                    .new_heap(&HeapDescriptor::new(heap_type, 1 << 20, options))
                    .unwrap()
            };
            let (placement, automatic) = (heap(HeapType::Placement), heap(HeapType::Automatic));
            let buffer = device.new_buffer(1 << 16, options).unwrap();
            // Every creator's answer for `d`, with its call.
            let answers = |d: &TextureDescriptor| {
                [
                    ("new_texture", device.new_texture(d).err()),
                    ("new_texture", automatic.new_texture(d).err()),
                    (
                        "new_texture_at_offset",
                        placement.new_texture_at_offset(d, 0).err(),
                    ),
                    ("new_texture", buffer.new_texture(d, 0, 1024).err()),
                    (
                        "new_linear_texture",
                        buffer.new_linear_texture(d, 0, 1024, 0).err(),
                    ),
                ]
            };
            for pf in [
                PixelFormat::RGBA16FloatXR,
                PixelFormat::AnyDepth,
                PixelFormat::AnyStencil,
                PixelFormat::YCBCR84202P,
            ] {
                let ycbcr = pf == PixelFormat::YCBCR84202P;
                let any_depth = matches!(pf, PixelFormat::AnyDepth | PixelFormat::AnyStencil);
                let of = |texture_type: TextureType| {
                    let mut d = TextureDescriptor::texture_2d(pf, 64, 64, false);
                    d.texture_type = texture_type;
                    d.usage = TextureUsage::SHADER_READ;
                    d.resource_options = options;
                    match texture_type {
                        T::Type1D | T::Type1DArray | T::TextureBuffer => d.height = 1,
                        T::Type3D => d.depth = 2,
                        _ => {}
                    }
                    if matches!(
                        texture_type,
                        T::Type1DArray | T::Type2DArray | T::CubeArray | T::Type2DMultisampleArray
                    ) {
                        d.array_length = 2;
                    }
                    if matches!(
                        texture_type,
                        T::Type2DMultisample | T::Type2DMultisampleArray
                    ) {
                        d.sample_count = 2;
                        d.usage = TextureUsage::RENDER_TARGET;
                    }
                    d
                };
                // Metal's nil from every creator, and the query's 0 bytes at the type's
                // alignment, for each type Metal does not stop.
                let stacked = STACKED_NIL_ALIGNMENT;
                for (texture_type, align, ycbcr_align) in [
                    (T::Type1D, 128, 16),
                    (T::Type1DArray, stacked, 16),
                    (T::Type2D, 128, 16),
                    (T::Type2DArray, stacked, 128),
                    (T::Cube, stacked, 16),
                    (T::CubeArray, stacked, 16),
                    (T::Type3D, stacked, 16),
                    (T::TextureBuffer, 16, 16),
                    (T::Type2DMultisample, 128, 0),
                    (T::Type2DMultisampleArray, stacked, 0),
                ] {
                    let what = format!("{pf:?} {texture_type:?} {options:?}");
                    let d = of(texture_type);
                    let one_dimensional = matches!(texture_type, T::Type1D | T::Type1DArray);
                    let multisample = matches!(
                        texture_type,
                        T::Type2DMultisample | T::Type2DMultisampleArray
                    );
                    let stops = (any_depth
                        && (one_dimensional
                            || multisample
                            || matches!(texture_type, T::Type3D | T::TextureBuffer)))
                        || (ycbcr && multisample);
                    if stops {
                        let problem = Problem::Texture(TextureProblem::TypeForFormat);
                        assert_eq!(
                            device.heap_texture_size_and_align(&d).err(),
                            stopped(query, problem.clone()),
                            "{what}"
                        );
                        for (call, answer) in answers(&d) {
                            assert_eq!(answer, stopped(call, problem.clone()), "{what} {call}");
                        }
                        continue;
                    }
                    let answer = device
                        .heap_texture_size_and_align(&d)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let align = if ycbcr { ycbcr_align } else { align };
                    assert_eq!((answer.size, answer.align), (0, align), "{what}");
                    for (call, answer) in answers(&d) {
                        assert_eq!(answer, nil(call), "{what} {call}");
                    }
                }
                // Each rule Metal stops comes first: an extent of 0, a side past 16384.
                let what = format!("{pf:?} {options:?}");
                let mut d = of(T::Type2D);
                d.width = 0;
                let zero = Problem::Texture(TextureProblem::ZeroExtent);
                assert_eq!(
                    device.heap_texture_size_and_align(&d).err(),
                    stopped(query, zero.clone()),
                    "{what}"
                );
                for (call, answer) in answers(&d) {
                    assert_eq!(answer, stopped(call, zero.clone()), "{what} {call}");
                }
                d.width = 16385;
                let past = Problem::Texture(TextureProblem::PastLargestSide { max: 16384 });
                assert_eq!(
                    device.heap_texture_size_and_align(&d).err(),
                    stopped(query, past.clone()),
                    "{what}"
                );
                for (call, answer) in answers(&d) {
                    assert_eq!(answer, stopped(call, past.clone()), "{what} {call}");
                }
                // A placement heap whose storage is not the descriptor's answers Metal's nil
                // first, where it stops a texture of a format.
                let mut other = of(T::Type2D);
                other.resource_options = if options == ResourceOptions::private() {
                    ResourceOptions::shared()
                } else {
                    ResourceOptions::private()
                };
                assert_eq!(
                    placement.new_texture_at_offset(&other, 0).err(),
                    nil("new_texture_at_offset"),
                    "{what}"
                );
            }
        }
    });
}

/// A texture of Apple's private `BlockWritesOnly` usage (0x10000) answers what the descriptor
/// without the bit answers, as Metal does (an M3 Max: a 16x16 `BGRA8Unorm` `Private` texture of
/// `ShaderRead | RenderTarget | BlockWritesOnly` answers 1152, as without the bit), in 2D,
/// mipmapped and array shapes, `Private` and `Shared`, though its image
/// takes the storage usage, which MoltenVK sizes as a `ShaderWrite` texture's. Two such textures
/// side by side at the answer keep their own texels.
#[test]
fn a_block_writes_only_texture_answers_the_size_without_the_bit() {
    on_device_with(at_plan, |device| {
        let shapes = [
            (PixelFormat::BGRA8Unorm, TextureType::Type2D, [16, 16], 1, 1),
            (
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                [100, 37],
                1,
                1,
            ),
            (
                PixelFormat::RGBA16Float,
                TextureType::Type2D,
                [256, 256],
                9,
                1,
            ),
            (
                PixelFormat::BGRA8Unorm,
                TextureType::Type2DArray,
                [64, 64],
                1,
                4,
            ),
        ];
        let bases = [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        ];
        for (pf, texture_type, [width, height], levels, array_length) in shapes {
            for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                for base in bases {
                    let of = |usage| {
                        let mut d = TextureDescriptor::texture_2d(pf, width, height, levels > 1);
                        d.texture_type = texture_type;
                        d.mipmap_level_count = levels;
                        d.array_length = array_length;
                        d.usage = usage;
                        d.resource_options = options;
                        d
                    };
                    let what = format!(
                        "{pf:?} {texture_type:?} {width}x{height} levels {levels} x{array_length} \
                         {options:?} {base:?}"
                    );
                    let blocked = of(base | TextureUsage::BLOCK_WRITES_ONLY);
                    let answer = device.heap_texture_size_and_align(&blocked).unwrap();
                    let plain = device.heap_texture_size_and_align(&of(base)).unwrap();
                    assert_eq!(answer, plain, "{what}");
                    eprintln!(
                        "HEAPTEX bwo {what}: size {} align {}",
                        answer.size, answer.align
                    );
                    if texture_type == TextureType::Type2D && levels == 1 {
                        side_by_side(device, &blocked, answer, [width, height], 4, &what);
                    }
                }
            }
        }
    });
}

/// Apple's private usage bits stay in the usage of a texture every creator makes, of its views
/// and of its shared open, and change no heap answer; 0x8 is dropped from the texture's usage, as
/// Metal drops it (an M4, plain and validated alike), while the descriptor keeps it, so a
/// descriptor of 0x8 alone answers the compressed size Metal answers, no smaller than `Unknown`'s,
/// and its texture answers no usage, as Metal's does.
#[test]
fn private_usage_bits_stay_as_metal_keeps_them() {
    on_device(|device| {
        let private = TextureUsage::NO_CONCURRENT_ACCESS
            | TextureUsage::VIDEO
            | TextureUsage::BLOCK_WRITES_ONLY;
        // The guest's 0x10005, and every bit with 0x8, which a descriptor keeps and Metal's
        // texture drops.
        assert_eq!(
            TextureUsage::try_from(0x1_0005).ok(),
            Some(
                TextureUsage::SHADER_READ
                    | TextureUsage::RENDER_TARGET
                    | TextureUsage::BLOCK_WRITES_ONLY
            )
        );
        let described = TextureUsage::try_from(0x1_c00d).unwrap();
        assert_eq!(described.as_raw(), 0x1_c00d);
        let usage = described.without_dropped();
        assert_eq!(
            usage,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET | private
        );
        let of = |usage: TextureUsage, options: ResourceOptions| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
            d.usage = usage;
            d.resource_options = options;
            d
        };
        let viewed = usage | TextureUsage::PIXEL_FORMAT_VIEW;
        for options in [ResourceOptions::private(), ResourceOptions::shared()] {
            let what = format!("{options:?}");
            let plain = device
                .heap_texture_size_and_align(&of(TextureUsage::SHADER_READ, options))
                .unwrap();
            let with = device
                .heap_texture_size_and_align(&of(TextureUsage::SHADER_READ | private, options))
                .unwrap();
            assert_eq!((with.size, with.align), (plain.size, plain.align), "{what}");
            let texture = device.new_texture(&of(viewed, options)).unwrap();
            assert_eq!(texture.usage(), viewed, "{what}: the device's");
            for view in [
                texture.new_texture_view(PixelFormat::RGBA8Unorm).unwrap(),
                texture
                    .new_texture_view(PixelFormat::RGBA8UnormSRGB)
                    .unwrap(),
            ] {
                assert_eq!(view.usage(), viewed, "{what}: a view");
            }
            let heap = device
                .new_heap(&HeapDescriptor::new(HeapType::Placement, 1 << 20, options))
                .unwrap();
            let placed = heap
                .new_texture_at_offset(&of(described, options), 0)
                .unwrap();
            assert_eq!(placed.usage(), usage, "{what}: a heap's");
            let buffer = device.new_buffer(1 << 16, options).unwrap();
            let over = buffer.new_texture(&of(described, options), 0, 256).unwrap();
            assert_eq!(over.usage(), usage, "{what}: a buffer's");
            // 0x8 alone: the texture answers no usage, where one of `Unknown` usage answers
            // `ShaderRead | ShaderWrite | RenderTarget | PixelFormatView`, and
            // the heap query at least the compressed size Metal answers, where `Unknown`'s is
            // Metal's plain size (the same image, so no smaller anywhere).
            let alone = TextureUsage::try_from(0x8).unwrap();
            let answered = device
                .heap_texture_size_and_align(&of(alone, options))
                .unwrap();
            let empty = of(TextureUsage::empty(), options);
            let unknown = device.heap_texture_size_and_align(&empty).unwrap();
            assert!(
                answered.size >= unknown.size,
                "{what}: {answered:?} {unknown:?}"
            );
            eprintln!("HEAPSIZE 0x8 alone {what}: {answered:?}, Unknown {unknown:?}");
            let made = device.new_texture(&of(alone, options)).unwrap();
            assert_eq!(made.usage(), TextureUsage::empty(), "{what}: 0x8 alone");
            let all = device.new_texture(&empty).unwrap();
            assert_eq!(all.usage().as_raw(), 0x17, "{what}: Unknown");
            let read = TextureUsage::try_from(0x9).unwrap();
            let made = device.new_texture(&of(read, options)).unwrap();
            assert_eq!(made.usage(), TextureUsage::SHADER_READ, "{what}: 0x9");
        }
        let shared = device
            .new_shared_texture(&of(described, ResourceOptions::private()))
            .unwrap();
        assert_eq!(shared.usage(), usage, "a shared texture's");
        let handle = shared.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        assert_eq!(opened.usage(), usage, "its open's");
    });
}

/// The kind a heap texture whose Metal layout the runtime does not model takes
/// ([`an_unmodelled_heap_size_is_announced_once`]).
pub(super) const UNMODELLED_SIZES: [Degradation; 1] = [Degradation::UnmeasuredHeapTextureSize];

/// A heap texture whose layout in Metal's heaps the runtime does not model answers its device's
/// size and announces `unmeasured-heap-texture-size` once a device: an 8-sample `RGBA8Unorm`
/// texture, where the device takes 8 samples (Apple's GPUs take 2 and 4, so Metal's layout of 8 is
/// not measured); a device without announces nothing. A PVRTC texture, where the device has the
/// format, answers Metal's 2048 bytes for 64x64 `PVRTC_RGBA_4BPP`, announcing nothing, as do
/// one-level and mipmapped textures Metal
/// compresses announce nothing, their answers Metal's model's (the consumer's boot's three-level
/// 224 x 928 `RGBA16Float` texture of `ShaderRead | RenderTarget | BlockWritesOnly` usage and a
/// seven-level 64 x 64 x 64 3D one).
#[test]
fn an_unmodelled_heap_size_is_announced_once() {
    notices::on_device(|device, messages| {
        messages.take_notices();
        let private = |mut d: TextureDescriptor| {
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            d
        };
        let kinds = |messages: &Messages| -> Vec<Degradation> {
            let notices = messages.take_notices();
            counts(&notices);
            notices
                .into_iter()
                .filter_map(|message| message.degradation)
                .map(|notice| notice.degradation)
                .filter(|kind| UNMODELLED_SIZES.contains(kind))
                .collect()
        };
        let mut boot = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 224, 928, true);
        boot.mipmap_level_count = 3;
        boot.usage = TextureUsage::try_from(0x10005).unwrap();
        boot.resource_options = ResourceOptions::private();
        let mut volume = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, true);
        volume.texture_type = TextureType::Type3D;
        volume.depth = 64;
        let one = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        for (what, d, least) in [
            ("one level", private(one), 0x4080),
            ("the boot's", boot, 0x279700),
            ("a mipmapped 3D", private(volume), 0x10_0000),
        ] {
            let answer = device.heap_texture_size_and_align(&d).unwrap();
            assert_eq!(kinds(messages), [], "{what}");
            assert!(answer.size >= least, "{what}: {answer:?}: at least Metal's");
        }
        let mut eight = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        eight.texture_type = TextureType::Type2DMultisample;
        eight.sample_count = 8;
        let pvrtc = TextureDescriptor::texture_2d(PixelFormat::PVRTCRGBA4BPP, 64, 64, false);
        if let Ok(answer) = device.heap_texture_size_and_align(&private(pvrtc)) {
            assert_eq!(kinds(messages), [], "PVRTC");
            assert!(answer.size >= 2048, "PVRTC: {answer:?}: at least Metal's");
        }
        match device.heap_texture_size_and_align(&private(eight)) {
            Ok(answer) => {
                println!("UNMODELLED 8 samples: {answer:?}");
                assert_eq!(kinds(messages), UNMODELLED_SIZES, "8 samples: announced");
                let again = device.heap_texture_size_and_align(&private(eight)).unwrap();
                assert_eq!(again, answer, "8 samples, again");
                assert_eq!(kinds(messages), [], "8 samples: once a device");
            }
            Err(_) => println!("UNMODELLED 8 samples: refused by the device"),
        }
    });
}

/// The kind a heap texture whose image takes more of a heap than Metal's modelled answer takes
/// ([`a_heap_size_past_metal_s_is_announced_once`]).
pub(super) const PAST_METAL_SIZES: [Degradation; 1] = [Degradation::HeapTextureLargerThanMetal];

/// A heap texture whose image on the device takes more bytes, or a coarser alignment, than
/// Metal's answer, which the runtime models, answers the device's and announces
/// `heap-texture-larger-than-metal` once a device, exactly where some answer is not Metal's:
/// `Private` 2D textures whose answers the consumer measured on its M3 Max under macOS 26.7 (each
/// at 128; the runtime's model alike), among them a 4 x 4 `Depth32Float` render target of 128
/// bytes, which the consumer saw answer 1024 on NVIDIA. Each answer is printed (`PAST-METAL`).
#[test]
fn a_heap_size_past_metal_s_is_announced_once() {
    notices::on_device(|device, messages| {
        messages.take_notices();
        let kinds = |messages: &Messages| -> Vec<Degradation> {
            let notices = messages.take_notices();
            counts(&notices);
            notices
                .into_iter()
                .filter_map(|message| message.degradation)
                .map(|notice| notice.degradation)
                .filter(|kind| PAST_METAL_SIZES.contains(kind))
                .collect()
        };
        // (format, width, height, levels, usage, Metal's size at an alignment of 128)
        let rows = [
            (PixelFormat::Depth32Float, 4, 4, 1, 0x4, 0x80),
            (PixelFormat::Depth16Unorm, 4, 4, 1, 0x4, 0x80),
            (PixelFormat::Stencil8, 4, 4, 1, 0x4, 0x80),
            (PixelFormat::RGBA8Unorm, 32, 64, 1, 0x3, 0x2000),
            (PixelFormat::RGBA8Unorm, 63, 63, 1, 0x3, 0x4000),
            (PixelFormat::RGBA8Unorm, 64, 64, 1, 0x1, 0x4080),
            (PixelFormat::RGBA8Unorm, 64, 64, 1, 0x4, 0x4080),
            (PixelFormat::RGBA8Unorm, 64, 64, 7, 0x3, 0x5680),
            (PixelFormat::RGBA8Unorm, 128, 128, 1, 0x1, 0x1_0200),
            (PixelFormat::RGBA8Unorm, 256, 256, 1, 0x1, 0x4_0800),
        ];
        let mut past = 0;
        for (pf, width, height, levels, usage, size) in rows {
            let mut d = TextureDescriptor::texture_2d(pf, width, height, false);
            d.mipmap_level_count = levels;
            d.usage = TextureUsage::try_from(usage).unwrap();
            d.resource_options = ResourceOptions::private();
            let what = format!("{pf:?} {width}x{height} {levels} levels {usage:#x}");
            let answer = device.heap_texture_size_and_align(&d).unwrap();
            println!(
                "PAST-METAL {what}: {:#x} at {:#x}, Metal's {size:#x} at 0x80",
                answer.size, answer.align
            );
            assert!(
                answer.size >= size && answer.align >= 0x80,
                "{what}: {answer:?}"
            );
            let larger = (answer.size, answer.align) != (size, 0x80);
            past += usize::from(larger);
            let want = match larger && past == 1 {
                true => PAST_METAL_SIZES.to_vec(),
                false => Vec::new(),
            };
            assert_eq!(kinds(messages), want, "{what}: once a device");
        }
        println!("PAST-METAL {past} of {} answers past Metal's", rows.len());
    });
}

/// `Device::metal_heap_texture_size_and_align` answers Metal's own heap size and alignment, the
/// same on every device, whatever the device's answer: 68 shapes of our probes (their rows on an
/// Apple M4 under macOS 27.0.1 and an M3 Max under 26.7, which answered alike: mipmapped 2D
/// textures that Metal compresses or not, the consumer's boot's two among them, 2D arrays and
/// cubes, `Shared` ones; and the M4's alone: 3D textures, cube arrays, arrays of 6, depth
/// and stencil textures with `allowGPUOptimizedContents` off, `Depth32Float_Stencil8` ones). The
/// device's own answer, where it makes one, is at least Metal's; each pair is printed
/// (`METAL-MODEL`), with the count of shapes whose device answer is coarser
/// (every one on NVIDIA). No model answers an 8-sample texture (or the device's rules stop it), nor
/// a PVRTC one at macOS 26.7's level (at 27.0.1's, Metal's 2048 bytes for 64x64
/// `PVRTC_RGBA_4BPP`); a texture buffer and a format plain Metal makes no texture of answer as the size
/// query does.
#[test]
fn metal_s_heap_answer_is_the_same_on_every_device() {
    use PixelFormat as P;
    use TextureType as T;
    on_device(|device| {
        // (format, type, extent, levels, array length, usage, Private, allowGPUOptimizedContents,
        // Metal's size and alignment)
        #[rustfmt::skip]
        let rows = [
            (P::RGBA16Float, T::Type2D, [224, 928, 1], 3, 1, 0x10005, true, true, (0x279700, 0x80)),
            (P::RGBA16Float, T::Type2D, [216, 928, 1], 3, 1, 0x10005, true, true, (0x279700, 0x80)),
            (P::RGBA16Float, T::Type2D, [224, 928, 1], 3, 1, 0x7, true, true, (0x276b80, 0x80)),
            (P::R32Float, T::Type2D, [64, 64, 1], 7, 1, 0x5, true, true, (0x5800, 0x80)),
            (P::R32Float, T::Type2D, [1024, 16, 1], 10, 1, 0x5, true, true, (0x15d80, 0x80)),
            (P::RGBA8Unorm, T::Type2D, [1440, 1025, 1], 10, 1, 0x5, true, true, (0x8e1680, 0x80)),
            (P::RGBA16Float, T::Type2D, [1440, 1080, 1], 4, 1, 0x10005, false, true, (0x111d600, 0x80)),
            (P::R16Float, T::Type2D, [100, 37, 1], 4, 1, 0x1, false, true, (0x5880, 0x80)),
            (P::Depth32FloatStencil8, T::Type2D, [4, 4, 1], 1, 1, 0x1, true, true, (0x4080, 0x80)),
            (P::RGBA16Float, T::Type2D, [32, 257, 1], 9, 1, 0x7, true, true, (0x2ac00, 0x80)),
            (P::RGBA16Float, T::Type2D, [65, 1024, 1], 6, 1, 0x1, true, true, (0x12c380, 0x80)),
            (P::BGRA8Unorm, T::Type2D, [1440, 128, 1], 9, 1, 0x1, true, true, (0x100300, 0x80)),
            (P::BGRA8Unorm, T::Type2D, [1440, 257, 1], 2, 1, 0x1, true, true, (0x2c1780, 0x80)),
            (P::RGBA8Unorm, T::Type2D, [513, 1024, 1], 8, 1, 0x10005, true, true, (0x341780, 0x80)),
            (P::RGBA8Unorm, T::Type2D, [17, 500, 1], 6, 1, 0x5, true, true, (0x15b80, 0x80)),
            (P::R16Float, T::Type2D, [129, 31, 1], 8, 1, 0x7, true, true, (0x5700, 0x80)),
            (P::R16Float, T::Type2D, [24, 17, 1], 5, 1, 0x7, true, true, (0xb80, 0x80)),
            (P::RG16Float, T::Type2D, [33, 24, 1], 2, 1, 0x7, true, true, (0x2b80, 0x80)),
            (P::RG16Float, T::Type2D, [256, 16, 1], 4, 1, 0x1, true, true, (0x5a00, 0x80)),
            (P::R8Unorm, T::Type2D, [129, 1024, 1], 9, 1, 0x10005, true, true, (0x4d900, 0x80)),
            (P::R8Unorm, T::Type2D, [1024, 928, 1], 9, 1, 0x7, true, true, (0x155700, 0x80)),
            (P::R32Float, T::Type2D, [255, 63, 1], 5, 1, 0x5, true, true, (0x15a80, 0x80)),
            (P::R32Float, T::Type2D, [224, 31, 1], 4, 1, 0x1, true, true, (0xae80, 0x80)),
            (P::RGBA32Float, T::Type2D, [128, 1025, 1], 5, 1, 0x10005, true, true, (0x2cd800, 0x80)),
            (P::RGBA32Float, T::Type2D, [129, 17, 1], 4, 1, 0x5, true, true, (0x2ae00, 0x80)),
            (P::R32Uint, T::Type2D, [32, 16, 1], 3, 1, 0x5, true, true, (0xd00, 0x80)),
            (P::R32Uint, T::Type2D, [257, 1024, 1], 9, 1, 0x10005, true, true, (0x1ded80, 0x80)),
            (P::RG8Unorm, T::Type2D, [513, 65, 1], 6, 1, 0x1, true, true, (0x2ee00, 0x80)),
            (P::RG8Unorm, T::Type2D, [33, 16, 1], 6, 1, 0x10005, true, true, (0xd00, 0x80)),
            (P::RG32Float, T::Type2D, [33, 100, 1], 6, 1, 0x7, true, true, (0x15600, 0x80)),
            (P::RG32Float, T::Type2D, [17, 500, 1], 8, 1, 0x1, true, true, (0x2b080, 0x80)),
            (P::RGBA16Unorm, T::Type2D, [256, 17, 1], 3, 1, 0x5, true, true, (0x15a00, 0x80)),
            (P::RGBA16Unorm, T::Type2D, [224, 63, 1], 6, 1, 0x1, true, true, (0x2af00, 0x80)),
            (P::Depth16Unorm, T::Type2D, [128, 257, 1], 4, 1, 0x5, true, true, (0x1a280, 0x80)),
            (P::Depth16Unorm, T::Type2D, [1440, 32, 1], 3, 1, 0x5, true, true, (0x2ba00, 0x80)),
            (P::Depth32Float, T::Type2D, [500, 127, 1], 6, 1, 0x5, true, true, (0x56280, 0x80)),
            (P::Depth32Float, T::Type2D, [1025, 24, 1], 7, 1, 0x5, true, true, (0x56480, 0x80)),
            (P::Stencil8, T::Type2D, [33, 63, 1], 2, 1, 0x5, true, true, (0x1800, 0x80)),
            (P::Stencil8, T::Type2D, [224, 32, 1], 8, 1, 0x5, true, true, (0x2f00, 0x80)),
            (P::Depth32FloatStencil8, T::Type2D, [24, 257, 1], 9, 1, 0x5, true, true, (0x1db80, 0x80)),
            (P::Depth32FloatStencil8, T::Type2D, [1025, 256, 1], 5, 1, 0x5, true, true, (0x1e2e00, 0x80)),
            (P::RG16Float, T::Cube, [928, 928, 1], 2, 1, 0x7, true, true, (0x1e78000, 0x4000)),
            (P::RGBA16Float, T::Type2DArray, [64, 64, 1], 7, 2, 0x10005, true, true, (0x18300, 0x4000)),
            (P::RGBA16Float, T::Cube, [256, 256, 1], 7, 1, 0x1, true, true, (0x40c500, 0x4000)),
            (P::R16Float, T::Type2DArray, [224, 224, 1], 7, 2, 0x1, true, true, (0x59600, 0x4000)),
            (P::RGBA8Unorm, T::Cube, [64, 64, 1], 5, 1, 0x5, true, true, (0x30900, 0x4000)),
            (P::RGBA8Unorm, T::Cube, [64, 64, 1], 6, 1, 0x5, true, true, (0x30900, 0x4000)),
            (P::RGBA8Unorm, T::Type3D, [33, 33, 33], 4, 1, 0x5, true, true, (0x10a100, 0x4000)),
            (P::RGBA16Float, T::Type3D, [100, 224, 3], 7, 1, 0x7, true, true, (0xf0000, 0x4000)),
            (P::BGRA8Unorm, T::Type3D, [129, 224, 2], 7, 1, 0x5, true, true, (0x91600, 0x4000)),
            (P::BGRA8Unorm, T::Type3D, [33, 224, 2], 3, 1, 0x7, true, true, (0x30000, 0x4000)),
            (P::RGBA8Unorm, T::Type3D, [16, 224, 3], 7, 1, 0x5, true, true, (0x18600, 0x4000)),
            (P::RGBA8Unorm, T::Type3D, [16, 224, 16], 6, 1, 0x10005, true, true, (0x82000, 0x4000)),
            (P::Depth16Unorm, T::CubeArray, [513, 513, 1], 8, 1, 0x1, true, true, (0x6d0200, 0x4000)),
            (P::Depth16Unorm, T::CubeArray, [513, 513, 1], 3, 1, 0x1, true, false, (0x690000, 0x4000)),
            (P::Depth16Unorm, T::CubeArray, [64, 64, 1], 6, 1, 0x5, true, true, (0x11100, 0x80)),
            (P::Depth16Unorm, T::CubeArray, [216, 216, 1], 4, 1, 0x5, true, true, (0x10c200, 0x4000)),
            (P::Depth32FloatStencil8, T::Type2DArray, [4, 4, 1], 3, 6, 0x5, true, true, (0x4900, 0x80)),
            (P::Depth32FloatStencil8, T::Type2DArray, [100, 100, 1], 7, 6, 0x1, true, false, (0xc0000, 0x4000)),
            (P::Depth16Unorm, T::Type2D, [16, 17, 1], 4, 1, 0x5, true, false, (0x680, 0x80)),
            (P::Depth32Float, T::Type2D, [16, 928, 1], 2, 1, 0x1, true, false, (0x15780, 0x80)),
            (P::Depth32Float, T::Type2D, [129, 513, 1], 5, 1, 0x1, true, false, (0xa1700, 0x80)),
            (P::Depth32FloatStencil8, T::Type2D, [129, 928, 1], 2, 1, 0x5, true, true, (0x161800, 0x80)),
            (P::Depth32FloatStencil8, T::CubeArray, [216, 216, 1], 6, 2, 0x4, true, true, (0x554400, 0x4000)),
            (P::Depth32FloatStencil8, T::Type2D, [1, 100, 1], 5, 1, 0x1, true, true, (0x4380, 0x80)),
            (P::Depth32FloatStencil8, T::Type2D, [928, 15, 1], 5, 1, 0x4, true, false, (0x1d800, 0x80)),
            (P::Depth32FloatStencil8, T::CubeArray, [513, 513, 1], 8, 1, 0x1, true, false, (0xf90000, 0x4000)),
            (P::Depth32FloatStencil8, T::Type2D, [33, 513, 1], 7, 1, 0x4, true, true, (0x6e380, 0x80)),
        ];
        let (mut coarser, mut answered) = (0, 0);
        for (pf, ty, [width, height, depth], levels, length, usage, private, optimized, want) in
            rows
        {
            let mut d = TextureDescriptor::texture_2d(pf, width, height, levels > 1);
            d.texture_type = ty;
            d.depth = depth;
            d.mipmap_level_count = levels;
            d.array_length = length;
            d.usage = TextureUsage::try_from(usage).unwrap();
            d.resource_options = match private {
                true => ResourceOptions::private(),
                false => ResourceOptions::shared(),
            };
            d.allow_gpu_optimized_contents = optimized;
            let what = format!(
                "{pf:?} {ty:?} {width}x{height}x{depth} {levels} levels x{length} {usage:#x} \
                 private={private} optimized={optimized}"
            );
            let metal = device
                .metal_heap_texture_size_and_align(&d)
                .unwrap_or_else(|e| panic!("{what}: {e}"))
                .unwrap_or_else(|| panic!("{what}: no model"));
            assert_eq!((metal.size, metal.align), want, "{what}");
            match device.heap_texture_size_and_align(&d) {
                Ok(own) => {
                    answered += 1;
                    let past = own.size > metal.size || own.align > metal.align;
                    coarser += usize::from(past);
                    println!(
                        "METAL-MODEL {what}: Metal's {:#x} at {:#x}, the device's {:#x} at {:#x}",
                        metal.size, metal.align, own.size, own.align
                    );
                    assert!(
                        own.size >= metal.size && own.align >= metal.align,
                        "{what}: {own:?} under Metal's {metal:?}"
                    );
                }
                Err(error) => println!("METAL-MODEL {what}: the device refused it: {error:?}"),
            }
        }
        println!(
            "METAL-MODEL {coarser} of {answered} device answers coarser than Metal's, of {} shapes",
            rows.len()
        );
        // A PVRTC texture, modelled; no model: an 8-sample one, where the device's rules take it.
        let private = |mut d: TextureDescriptor| {
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            d
        };
        let pvrtc = TextureDescriptor::texture_2d(PixelFormat::PVRTCRGBA4BPP, 64, 64, false);
        let answer = device.metal_heap_texture_size_and_align(&private(pvrtc));
        let answer = answer.ok().map(|answer| answer.map(|a| (a.size, a.align)));
        assert_eq!(answer, Some(Some((2048, 128))), "PVRTC");
        let mut eight = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        eight.texture_type = TextureType::Type2DMultisample;
        eight.sample_count = 8;
        // Where the device's rules refuse 8 samples, both queries refuse the descriptor.
        let eight = private(eight);
        match device.metal_heap_texture_size_and_align(&eight) {
            Ok(answer) => assert_eq!(answer, None, "8 samples"),
            Err(error) => {
                println!("METAL-MODEL 8 samples: refused: {error:?}");
                let own = device.heap_texture_size_and_align(&eight);
                assert!(own.is_err(), "8 samples: the size query answers {own:?}");
            }
        }
        // As the size query: a texture buffer, and a format of which plain Metal makes none.
        let mut buffer = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 100, 1, false);
        buffer.texture_type = TextureType::TextureBuffer;
        let unspecialized =
            TextureDescriptor::texture_2d(PixelFormat::Unspecialized, 16, 16, false);
        for (what, d) in [("texture buffer", buffer), ("Unspecialized", unspecialized)] {
            let d = private(d);
            let metal = device.metal_heap_texture_size_and_align(&d).unwrap();
            let own = device.heap_texture_size_and_align(&d).unwrap();
            assert_eq!(metal, Some(own), "{what}");
        }
    });
}
