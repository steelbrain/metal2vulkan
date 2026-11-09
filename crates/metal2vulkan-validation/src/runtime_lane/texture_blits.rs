//! Device tests of the blit encoder's texture commands on the device the cfg selects, under the
//! validation layer: copies between buffers and textures (on the layouts Vulkan states and on the
//! ones that go through a scratch buffer) and between textures, Metal's refusals, 32-bit float
//! depth outside `[0, 1]` uploaded as Metal holds it with the form each device stores it in
//! printed, what each driver's depth images keep of the `float`s Metal's hold, against its driver
//! facts, and `generate_mipmaps` with each format's way printed.

use super::degrade_notices::signs_depth_tests;
use super::degrade_notices::twins_carried;
use super::degrade_notices::{assert_made_once, assert_once, forced, loses_depth_denormals};
use super::degrade_notices::{decodes_blocks_its_way, shared_in_system_memory};
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::MipmapFilterChain;
use metal2vulkan::device::VkExtension;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::format::{is_depth_stencil, x_channel};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CopyProblem, CpuCacheMode, Degradation,
    Device, DriverFact, Error, HazardTrackingMode, Invalid, Origin, Problem, Region,
    ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `f` of `buffer`'s contents.
fn with_bytes<R>(buffer: &Buffer, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer, and no
    // command buffer of these tests uses them while `f` runs.
    f(unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) })
}

/// A Shared buffer whose byte `i` holds `i % 251`.
fn counting(device: &Device, length: u64) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    with_bytes(&buffer, |bytes| {
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
    });
    buffer
}

fn counted(start: usize, n: usize) -> Vec<u8> {
    (start..start + n).map(|i| (i % 251) as u8).collect()
}

fn texture(
    device: &Device,
    pf: PixelFormat,
    ty: TextureType,
    extent: [u64; 3],
    levels: u64,
    slices: u64,
    storage: StorageMode,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, extent[0], extent[1], false);
    d.texture_type = ty;
    d.depth = extent[2];
    d.mipmap_level_count = levels;
    d.array_length = slices;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_texture(&d)
}

/// `bytes_per_row * rows` bytes of `level` and `slice` of a texture, rows tightly packed.
fn level_bytes(t: &Texture, level: u64, slice: u64, w: u64, h: u64, texel: u64) -> Vec<u8> {
    let mut b = vec![0u8; (w * h * texel) as usize];
    t.get_bytes(
        &mut b,
        w * texel,
        0,
        Region::new_2d(0, 0, w, h),
        level,
        slice,
    )
    .unwrap();
    b
}

/// Copies between buffers and textures and between textures land where Metal puts them, in one
/// encoder in encoded order; layouts Vulkan cannot state copy through a scratch buffer exactly;
/// Metal's refusals are typed.
#[test]
fn texture_blits_copy_where_metal_puts_them() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let d2 = TextureType::Type2D;
        let private = texture(&device, rgba, d2, [8, 8, 1], 2, 1, StorageMode::Private).unwrap();
        let shared = texture(&device, rgba, d2, [8, 8, 1], 2, 1, StorageMode::Shared).unwrap();
        let source = counting(&device, 1024);
        let out = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        with_bytes(&out, |b| b.fill(0xee));

        // Into the private texture (rows 40 bytes apart, from offset 4; level 1 tight), all of
        // it into the shared one, then a region of it into a buffer and that buffer's bytes on.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let none = BlitOption::None;
        blit.copy_from_buffer_to_texture(
            &source,
            4,
            40,
            0,
            Size::new(8, 8, 1),
            &private,
            0,
            0,
            Origin::new(0, 0, 0),
            none,
        )
        .unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            512,
            16,
            0,
            Size::new(4, 4, 1),
            &private,
            0,
            1,
            Origin::new(0, 0, 0),
            none,
        )
        .unwrap();
        blit.copy_texture(&private, &shared).unwrap();
        blit.copy_from_texture_to_buffer(
            &private,
            0,
            0,
            Origin::new(2, 3, 0),
            Size::new(4, 2, 1),
            &out,
            8,
            20,
            0,
            none,
        )
        .unwrap();
        blit.copy_from_buffer_to_buffer(&out, 8, &out, 128, 40)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let level0: Vec<u8> = (0..8).flat_map(|y| counted(4 + 40 * y, 32)).collect();
        assert_eq!(level_bytes(&shared, 0, 0, 8, 8, 4), level0);
        assert_eq!(level_bytes(&shared, 1, 0, 4, 4, 4), counted(512, 64));
        with_bytes(&out, |b| {
            let rows: Vec<u8> = (0..2)
                .flat_map(|y| counted(4 + 40 * (3 + y) + 8, 16))
                .collect();
            assert_eq!(b[8..24], rows[..16]);
            assert_eq!(b[24..28], [0xee; 4], "the bytes between rows are left");
            assert_eq!(b[28..44], rows[16..]);
            assert_eq!(
                b[128..168],
                b[8..48],
                "the later copy saw the texture's bytes"
            );
        });

        // A 3D texture whose images are 18 bytes apart (a multiple of no texel), in and out
        // through a scratch buffer: each row at its offset rounded down to 4 bytes, the second
        // image's at 16 and 24, as plain Metal places them; a region of one
        // texture into another format of one unit.
        let volume = texture(
            &device,
            rgba,
            TextureType::Type3D,
            [2, 2, 2],
            1,
            1,
            StorageMode::Shared,
        )
        .unwrap();
        let back = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        with_bytes(&back, |b| b.fill(0xee));
        let words = texture(
            &device,
            PixelFormat::R32Uint,
            d2,
            [4, 4, 1],
            1,
            1,
            StorageMode::Shared,
        )
        .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let size = Size::new(2, 2, 2);
        let origin = Origin::new(0, 0, 0);
        blit.copy_from_buffer_to_texture(&source, 0, 8, 18, size, &volume, 0, 0, origin, none)
            .unwrap();
        blit.copy_from_texture_to_buffer(&volume, 0, 0, origin, size, &back, 4, 8, 18, none)
            .unwrap();
        blit.copy_from_texture_to_texture(
            &private,
            0,
            0,
            Origin::new(1, 1, 0),
            Size::new(3, 2, 1),
            &words,
            0,
            0,
            Origin::new(0, 2, 0),
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut images = vec![0u8; 32];
        volume
            .get_bytes(&mut images, 8, 16, Region::new_3d(0, 0, 0, 2, 2, 2), 0, 0)
            .unwrap();
        let want: Vec<u8> = [0usize, 8, 16, 24]
            .iter()
            .flat_map(|&at| counted(at, 8))
            .collect();
        assert_eq!(images, want);
        with_bytes(&back, |b| {
            // From offset 4: rows at 4, 12, 22 and 30, rounded down to 20 and 28.
            for (i, &at) in [4usize, 12, 20, 28].iter().enumerate() {
                assert_eq!(b[at..at + 8], want[8 * i..8 * i + 8], "row {i}");
            }
            assert_eq!(b[..4], [0xee; 4]);
            assert_eq!(b[36..], [0xee; 28], "past the images");
        });
        let rows: Vec<u8> = (0..4)
            .flat_map(|y| match y {
                2 | 3 => {
                    let mut row = counted(4 + 40 * (y - 1) + 4, 12);
                    row.extend([0; 4]);
                    row
                }
                _ => vec![0; 16],
            })
            .collect();
        assert_eq!(level_bytes(&words, 0, 0, 4, 4, 4), rows);

        // Slices and levels of one array into another.
        let arrays = [0u8; 2].map(|_| {
            texture(
                &device,
                rgba,
                TextureType::Type2DArray,
                [4, 4, 1],
                2,
                3,
                StorageMode::Shared,
            )
            .unwrap()
        });
        for slice in 0..3u64 {
            for level in 0..2u64 {
                let side = 4 >> level;
                let bytes = counted(
                    (slice * 32 + level * 8) as usize,
                    (side * side * 4) as usize,
                );
                arrays[0]
                    .replace_region(
                        Region::new_2d(0, 0, side, side),
                        level,
                        slice,
                        &bytes,
                        side * 4,
                        0,
                    )
                    .unwrap();
            }
        }
        let [from, to] = &arrays;
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture_slices(from, 1, 0, to, 0, 0, 2, 2)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        for slice in 0..2u64 {
            for level in 0..2u64 {
                let side = 4 >> level;
                let want = counted(
                    ((slice + 1) * 32 + level * 8) as usize,
                    (side * side * 4) as usize,
                );
                assert_eq!(level_bytes(to, level, slice, side, side, 4), want);
            }
        }
        let fresh = level_bytes(to, 0, 2, 4, 4, 4);
        assert!(fresh.iter().all(|&b| b == 0), "slice 2 untouched");

        // Depth: 16-bit from an offset Vulkan does not take (through a scratch buffer); 32-bit
        // float as Metal holds it, on every device (these floats lie in [0, 1]).
        let depth16 = texture(
            &device,
            PixelFormat::Depth16Unorm,
            d2,
            [2, 2, 1],
            1,
            1,
            StorageMode::Shared,
        );
        let depth32 = texture(
            &device,
            PixelFormat::Depth32Float,
            d2,
            [2, 2, 1],
            1,
            1,
            StorageMode::Shared,
        );
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let square = Size::new(2, 2, 1);
        if let Ok(depth16) = &depth16 {
            blit.copy_from_buffer_to_texture(&source, 2, 4, 0, square, depth16, 0, 0, origin, none)
                .unwrap();
        }
        if let Ok(depth32) = &depth32 {
            blit.copy_from_buffer_to_texture(&source, 0, 8, 0, square, depth32, 0, 0, origin, none)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        if let Ok(depth16) = &depth16 {
            assert_eq!(level_bytes(depth16, 0, 0, 2, 2, 2), counted(2, 8));
        }
        if let Ok(depth32) = &depth32 {
            assert_eq!(level_bytes(depth32, 0, 0, 2, 2, 4), counted(0, 16));
        }

        // Metal's refusals, and the hints.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        assert_eq!(
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                32,
                0,
                Size::new(8, 8, 1),
                &shared,
                0,
                0,
                Origin::new(1, 0, 0),
                none
            ),
            Err(invalid(
                "copy_from_buffer_to_texture",
                Problem::Copy(CopyProblem::Region)
            ))
        );
        // An overlapping copy within one texture is no refusal: it moves as `memmove` does
        // (`format_copies` checks its texels).
        blit.copy_from_texture_to_texture(
            &shared,
            0,
            0,
            origin,
            Size::new(4, 4, 1),
            &shared,
            0,
            0,
            Origin::new(2, 2, 0),
        )
        .unwrap();
        // Plain Metal does nothing for a texture of another storage mode, whatever the level.
        blit.synchronize_texture(&shared, 0, 0).unwrap();
        blit.synchronize_texture(&shared, 0, 7).unwrap();
        blit.synchronize_resource(shared.as_resource()).unwrap();
        let managed = texture(&device, rgba, d2, [4, 4, 1], 1, 1, StorageMode::Managed).unwrap();
        blit.synchronize_texture(&managed, 0, 0).unwrap();
        blit.synchronize_resource(managed.as_resource()).unwrap();
        assert_eq!(
            blit.synchronize_texture(&managed, 1, 0),
            Err(invalid(
                "synchronize_texture",
                Problem::Copy(CopyProblem::Region)
            ))
        );
        blit.optimize_contents_for_gpu_access(&shared).unwrap();
        blit.optimize_contents_for_cpu_access_slice_level(&shared, 0, 1)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A 1D texture made after a small 2D one keeps what a command buffer writing both writes into
/// it: ANV on Intel's Xe-LP GPUs compresses the 2D texture and answers its size short of the
/// 64-byte line its compression data is written back in, and the 1D texture placed right after it
/// lost its first texels to the command buffer's end, until the runtime rounded such an image's
/// size up to whole 256-byte chunks of compression data there. Every device runs it.
#[test]
fn a_texture_beside_a_compressed_one_keeps_its_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let shared = StorageMode::Shared;
        let (at, none) = (Origin::new(0, 0, 0), BlitOption::None);
        let flat = texture(&device, rgba, TextureType::Type2D, [4, 4, 1], 1, 1, shared).unwrap();
        let line = texture(&device, rgba, TextureType::Type1D, [4, 1, 1], 1, 1, shared).unwrap();
        let after = texture(&device, rgba, TextureType::Type2D, [4, 1, 1], 1, 1, shared).unwrap();
        let source = counting(&device, 256);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            16,
            0,
            Size::new(4, 4, 1),
            &flat,
            0,
            0,
            at,
            none,
        )
        .unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            64,
            16,
            0,
            Size::new(4, 1, 1),
            &line,
            0,
            0,
            at,
            none,
        )
        .unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            128,
            16,
            0,
            Size::new(4, 1, 1),
            &after,
            0,
            0,
            at,
            none,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(level_bytes(&flat, 0, 0, 4, 4, 4), counted(0, 64));
        assert_eq!(level_bytes(&line, 0, 0, 4, 1, 4), counted(64, 16));
        assert_eq!(level_bytes(&after, 0, 0, 4, 1, 4), counted(128, 16));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A size of 0 copies what plain Metal copies (an M4): on an axis a texture of the copy lacks, one
/// texel there (the depth of a 2D texture, the height of a 1D one); on another axis, or of no
/// slices, nothing, while the encoder's fills before and after run; and a CPU copy of an empty
/// region, nothing.
#[test]
fn empty_sizes_copy_what_plain_metal_copies() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let shared = StorageMode::Shared;
        let make = |ty, extent| texture(&device, rgba, ty, extent, 1, 1, shared).unwrap();
        let flat = make(TextureType::Type2D, [4, 4, 1]);
        let other = make(TextureType::Type2D, [4, 4, 1]);
        let line = make(TextureType::Type1D, [4, 1, 1]);
        let volume = make(TextureType::Type3D, [4, 4, 2]);
        let source = counting(&device, 256);
        let out = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        with_bytes(&out, |b| b.fill(0xee));
        let fills = device.new_buffer(8, ResourceOptions::shared()).unwrap();
        let (at, none) = (Origin::new(0, 0, 0), BlitOption::None);

        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&fills, 0..4, 0xaa).unwrap();
        // One texel on the lacking axis: a 2D texture's depth, a 1D texture's height.
        let one_image = Size::new(4, 4, 0);
        blit.copy_from_buffer_to_texture(&source, 0, 16, 0, one_image, &flat, 0, 0, at, none)
            .unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            64,
            16,
            0,
            Size::new(4, 0, 1),
            &line,
            0,
            0,
            at,
            none,
        )
        .unwrap();
        blit.copy_from_texture_to_buffer(&flat, 0, 0, at, one_image, &out, 0, 16, 0, none)
            .unwrap();
        // Nothing: a width of 0, a 3D texture's depth of 0, a 2D texture's height of 0, no
        // slices.
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            16,
            0,
            Size::new(0, 4, 1),
            &other,
            0,
            0,
            at,
            none,
        )
        .unwrap();
        blit.copy_from_buffer_to_texture(&source, 0, 16, 64, one_image, &volume, 0, 0, at, none)
            .unwrap();
        blit.copy_from_texture_to_texture(&flat, 0, 0, at, Size::new(4, 0, 1), &other, 0, 0, at)
            .unwrap();
        blit.copy_texture_slices(&flat, 0, 0, &other, 0, 0, 0, 1)
            .unwrap();
        blit.fill_buffer(&fills, 4..8, 0xbb).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);

        assert_eq!(level_bytes(&flat, 0, 0, 4, 4, 4), counted(0, 64));
        assert_eq!(level_bytes(&line, 0, 0, 4, 1, 4), counted(64, 16));
        let mut expected = counted(0, 64);
        expected.resize(256, 0xee);
        assert_eq!(with_bytes(&out, |b| b.to_vec()), expected);
        assert_eq!(level_bytes(&other, 0, 0, 4, 4, 4), vec![0; 64]);
        let mut whole = vec![0xeeu8; 128];
        volume
            .get_bytes(&mut whole, 16, 64, Region::new_3d(0, 0, 0, 4, 4, 2), 0, 0)
            .unwrap();
        assert_eq!(whole, vec![0; 128]);
        assert_eq!(
            with_bytes(&fills, |b| b.to_vec()),
            [[0xaa; 4], [0xbb; 4]].concat()
        );

        // The CPU's copies of an empty region, of a 2D texture's depth too, write nothing.
        let empty = Region::new_3d(0, 0, 0, 4, 4, 0);
        flat.replace_region(empty, 0, 0, &[0x55; 64], 16, 64)
            .unwrap();
        flat.replace_region(Region::new_2d(0, 0, 0, 4), 0, 0, &[0x55; 64], 16, 0)
            .unwrap();
        assert_eq!(level_bytes(&flat, 0, 0, 4, 4, 4), counted(0, 64));
        let mut read = vec![0x77u8; 64];
        flat.get_bytes(&mut read, 16, 64, empty, 0, 0).unwrap();
        flat.get_bytes(&mut read, 16, 0, Region::new_2d(0, 0, 4, 0), 0, 0)
            .unwrap();
        assert_eq!(read, vec![0x77; 64]);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A 4x4 `RGBA8Unorm` texture's bytes when each row was copied from `rows` of a
/// [`counting`] buffer.
fn rows_from(rows: [usize; 4]) -> Vec<u8> {
    rows.iter().flat_map(|&row| counted(row, 16)).collect()
}

/// Copies off Metal's rules land where plain Metal puts them (an M4 under macOS 27.0.1; its
/// validation stops each): offsets and `bytesPerRow` off the copy unit, blit options a format does
/// not take, compressed regions off their blocks, and the CPU's pitches; what plain Metal leaves
/// undefined stays refused.
#[test]
fn copies_off_metals_rules_land_where_plain_metal_puts_them() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let (rgba, d2) = (PixelFormat::RGBA8Unorm, TextureType::Type2D);
        let (shared, private) = (StorageMode::Shared, StorageMode::Private);
        let rgba8 = || texture(&device, rgba, d2, [4, 4, 1], 1, 1, shared).unwrap();
        let source = counting(&device, 1024);
        let (at, none, four) = (Origin::new(0, 0, 0), BlitOption::None, Size::new(4, 4, 1));

        // Offsets and pitches off the copy unit (`sw_rgba8_4x4`): each row at its offset
        // rounded down to 4 bytes, and Metal's aligned path from offset 0.
        let cases: [(u64, u64, [usize; 4]); 4] = [
            (5, 16, [4, 20, 36, 52]),
            (1, 18, [0, 16, 36, 52]),
            (0, 18, [0, 16, 32, 48]),
            (2, 6, [0, 8, 12, 20]),
        ];
        let targets: Vec<Texture> = cases.iter().map(|_| rgba8()).collect();
        let out = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        with_bytes(&out, |b| b.fill(0xee));
        let overlapped = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        with_bytes(&overlapped, |b| b.fill(0xee));
        // Blit options a format does not take (`bo_*`): a combined depth/stencil texture
        // without one copies its depth, and an option naming an aspect a format lacks is
        // ignored.
        let depth: Vec<u8> = (0..16u16)
            .flat_map(|i| (f32::from(i) / 16.0).to_le_bytes())
            .collect();
        let depth_in = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        with_bytes(&depth_in, |b| b.copy_from_slice(&depth));
        let depth_out = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let combined = texture(
            &device,
            PixelFormat::Depth32FloatStencil8,
            d2,
            [4, 4, 1],
            1,
            1,
            private,
        );
        let colour = rgba8();
        // A compressed region off its blocks (`ba_*`): the blocks it starts in.
        let bc1 = texture(&device, PixelFormat::BC1RGBA, d2, [16, 16, 1], 1, 1, shared);
        if let Ok(bc1) = &bc1 {
            bc1.replace_region(Region::new_2d(0, 0, 16, 16), 0, 0, &[0xee; 128], 32, 0)
                .unwrap();
        }

        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for ((offset, bpr, _), t) in cases.iter().zip(&targets) {
            blit.copy_from_buffer_to_texture(&source, *offset, *bpr, 0, four, t, 0, 0, at, none)
                .unwrap();
        }
        blit.copy_from_texture_to_buffer(&targets[2], 0, 0, at, four, &out, 0, 18, 0, none)
            .unwrap();
        // Out of a texture at a pitch below its row: each row at `y x bytesPerRow`, written in
        // turn, the later row's bytes standing (one of the orders plain Metal writes them in).
        blit.copy_from_texture_to_buffer(&targets[0], 0, 0, at, four, &overlapped, 0, 8, 0, none)
            .unwrap();
        // Refused: a pitch of 0 on the aligned path (Metal reads garbage).
        assert_eq!(
            blit.copy_from_buffer_to_texture(&source, 0, 0, 0, four, &targets[0], 0, 0, at, none),
            Err(invalid(
                "copy_from_buffer_to_texture",
                Problem::Copy(CopyProblem::BytesPerRow)
            ))
        );
        if let Ok(combined) = &combined {
            blit.copy_from_buffer_to_texture(&depth_in, 0, 16, 0, four, combined, 0, 0, at, none)
                .unwrap();
            let option = BlitOption::DepthFromDepthStencil;
            blit.copy_from_texture_to_buffer(
                combined, 0, 0, at, four, &depth_out, 0, 16, 0, option,
            )
            .unwrap();
        }
        let stencil = BlitOption::StencilFromDepthStencil;
        blit.copy_from_buffer_to_texture(&source, 0, 16, 0, four, &colour, 0, 0, at, stencil)
            .unwrap();
        if let Ok(bc1) = &bc1 {
            let block = Size::new(4, 4, 1);
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                32,
                0,
                block,
                bc1,
                0,
                0,
                Origin::new(5, 5, 0),
                none,
            )
            .unwrap();
            let wide = Size::new(5, 4, 1);
            blit.copy_from_buffer_to_texture(
                &source,
                64,
                32,
                0,
                wide,
                bc1,
                0,
                0,
                Origin::new(0, 8, 0),
                none,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);

        for ((_, _, rows), t) in cases.iter().zip(&targets) {
            assert_eq!(level_bytes(t, 0, 0, 4, 4, 4), rows_from(*rows), "{rows:?}");
        }
        let mut written = vec![0xeeu8; 256];
        for (row, at) in [0, 16, 36, 52].into_iter().enumerate() {
            written[at..at + 16].copy_from_slice(&counted(16 * row, 16));
        }
        assert_eq!(with_bytes(&out, |b| b.to_vec()), written);
        // `targets[0]` holds the rows from 4, 20, 36 and 52; each but the last keeps its first 8
        // bytes, the rest under the next row's.
        let mut standing = vec![0xeeu8; 64];
        for (row, from) in [4, 20, 36].into_iter().enumerate() {
            standing[8 * row..8 * row + 8].copy_from_slice(&counted(from, 8));
        }
        standing[24..40].copy_from_slice(&counted(52, 16));
        assert_eq!(with_bytes(&overlapped, |b| b.to_vec()), standing);
        if combined.is_ok() {
            assert_eq!(with_bytes(&depth_out, |b| b.to_vec()), depth);
        }
        assert_eq!(level_bytes(&colour, 0, 0, 4, 4, 4), counted(0, 64));
        if let Ok(bc1) = &bc1 {
            // Off its blocks on the CPU too: the block from (0, 0).
            bc1.replace_region(Region::new_2d(1, 1, 3, 2), 0, 0, &[0x11; 8], 32, 0)
                .unwrap();
            let mut blocks = vec![0u8; 128];
            bc1.get_bytes(&mut blocks, 32, 0, Region::new_2d(0, 0, 16, 16), 0, 0)
                .unwrap();
            let mut want = vec![0xeeu8; 128];
            want[..8].fill(0x11);
            want[40..48].copy_from_slice(&counted(0, 8));
            want[64..80].copy_from_slice(&counted(64, 16));
            assert_eq!(blocks, want);
        }

        // The CPU's pitches off Metal's rules (`host`, `bpr_rr_*`, `bpr_gb_*`): a pitch of a row
        // or more exactly, 0 every row at the first byte, and one below a row nothing.
        let host = rgba8();
        let bytes = counted(0, 256);
        let whole = Region::new_2d(0, 0, 4, 4);
        host.replace_region(whole, 0, 0, &bytes, 18, 0).unwrap();
        assert_eq!(
            level_bytes(&host, 0, 0, 4, 4, 4),
            rows_from([0, 18, 36, 54])
        );
        host.replace_region(whole, 0, 0, &bytes[100..], 8, 0)
            .unwrap();
        assert_eq!(
            level_bytes(&host, 0, 0, 4, 4, 4),
            rows_from([0, 18, 36, 54])
        );
        let mut last = vec![0xeeu8; 16];
        host.get_bytes(&mut last, 0, 0, whole, 0, 0).unwrap();
        assert_eq!(last, counted(54, 16));
        let mut untouched = vec![0xeeu8; 64];
        host.get_bytes(&mut untouched, 12, 0, whole, 0, 0).unwrap();
        assert_eq!(untouched, vec![0xee; 64]);
        host.replace_region(whole, 0, 0, &bytes[64..], 0, 0)
            .unwrap();
        assert_eq!(level_bytes(&host, 0, 0, 4, 4, 4), rows_from([64; 4]));
    }
    layer_reported_nothing_at_all(&messages);
}

/// Block `i` of `level` of a `BC3_RGBA` texture of [`seeded_bc3`] `seed` (below 4): 16 bytes, each
/// unique to the block.
fn bc3_block(seed: u64, level: u64, i: u64) -> Vec<u8> {
    let tag = (seed * 64 + level * 8 + i) as u8;
    (0..16u8).map(|b| b.wrapping_mul(17) ^ tag).collect()
}

/// The blocks of `level` of a `BC3_RGBA` texture of [`seeded_bc3`] `seed`, `extent` large at level
/// 0, rows tightly packed: their texel extent, and their bytes.
fn seeded_level(seed: u64, extent: [u64; 2], level: u64) -> ([u64; 2], Vec<u8>) {
    let [w, h] = extent.map(|e| (e >> level).max(1));
    let blocks = w.div_ceil(4) * h.div_ceil(4);
    let bytes = (0..blocks)
        .flat_map(|i| bc3_block(seed, level, i))
        .collect();
    ([w, h], bytes)
}

/// A Shared `BC3_RGBA` texture `extent` large, of `levels` levels, each block of each level
/// [`bc3_block`] of `seed`.
fn seeded_bc3(device: &Device, seed: u64, extent: [u64; 2], levels: u64) -> Texture {
    let (bc3, d2, shared) = (
        PixelFormat::BC3RGBA,
        TextureType::Type2D,
        StorageMode::Shared,
    );
    let t = texture(
        device,
        bc3,
        d2,
        [extent[0], extent[1], 1],
        levels,
        1,
        shared,
    )
    .unwrap();
    for level in 0..levels {
        let ([w, h], bytes) = seeded_level(seed, extent, level);
        let row = w.div_ceil(4) * 16;
        t.replace_region(Region::new_2d(0, 0, w, h), level, 0, &bytes, row, 0)
            .unwrap();
    }
    t
}

/// The blocks of `level` of a `BC3_RGBA` texture `extent` large at level 0, rows tightly packed.
fn bc3_blocks(t: &Texture, extent: [u64; 2], level: u64) -> Vec<u8> {
    let [w, h] = extent.map(|e| (e >> level).max(1));
    let row = w.div_ceil(4) * 16;
    let mut b = vec![0u8; (row * h.div_ceil(4)) as usize];
    t.get_bytes(&mut b, row, 0, Region::new_2d(0, 0, w, h), level, 0)
        .unwrap();
    b
}

/// Whole blocks of compressed levels smaller than a block copy as plain Metal copies them (an
/// Apple M3 Max under macOS 26.7, a probe; its validation takes all but the copy into a texture
/// from a buffer and the 4 x 2 regions): a 4 x 4 region of a 4 x 2, 2 x 1 or 1 x 1 level of a
/// `BC3_RGBA` texture out of it, into it, between levels of one size, into a whole block of a
/// larger level and back (whose Vulkan extents differ: through a scratch buffer), and on the CPU,
/// each moving the block's 16 bytes and nothing else.
#[test]
fn whole_blocks_of_levels_smaller_than_a_block_copy_whole() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        // Levels 16 x 8, 8 x 4, 4 x 2, 2 x 1 and 1 x 1; and one 8 x 8 level of 2 x 2 blocks.
        let (small, large) = ([16, 8], [8, 8]);
        let source = seeded_bc3(&device, 0, small, 5);
        let same = seeded_bc3(&device, 1, small, 5);
        let other = seeded_bc3(&device, 2, small, 5);
        let big = seeded_bc3(&device, 3, large, 1);
        let out = device.new_buffer(96, ResourceOptions::shared()).unwrap();
        with_bytes(&out, |b| b.fill(0xee));
        let written: Vec<u8> = (0..16u8).map(|b| 0xa5 ^ b).collect();
        let input = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        with_bytes(&input, |b| b.copy_from_slice(&written));
        let (zero, none) = (Origin::new(0, 0, 0), BlitOption::None);
        let (block, half) = (Size::new(4, 4, 1), Size::new(4, 2, 1));

        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        // Out of the texture (`t2b_l9_4x4`, `t2b_l11_4x4`, `t2b_l9_o2_4x2`), 32 bytes apart:
        // a second row of blocks would land in the 16 bytes after each.
        blit.copy_from_texture_to_buffer(&source, 0, 2, zero, block, &out, 0, 16, 0, none)
            .unwrap();
        blit.copy_from_texture_to_buffer(&source, 0, 4, zero, block, &out, 32, 16, 0, none)
            .unwrap();
        let below = Origin::new(0, 2, 0);
        blit.copy_from_texture_to_buffer(&source, 0, 2, below, half, &out, 64, 16, 0, none)
            .unwrap();
        // Into it (`b2t_l9_4x4`), at the 2 x 1 level.
        blit.copy_from_buffer_to_texture(&input, 0, 16, 0, block, &same, 0, 3, zero, none)
            .unwrap();
        // Between levels of one size (`t2t_same_l9_4x4`).
        blit.copy_from_texture_to_texture(&source, 0, 2, zero, block, &same, 0, 2, zero)
            .unwrap();
        // Into a whole block of a larger level, and back (`t2t_small_to_big_*`,
        // `t2t_big_to_small_*`).
        let corner = Origin::new(4, 4, 0);
        blit.copy_from_texture_to_texture(&source, 0, 2, zero, block, &big, 0, 0, corner)
            .unwrap();
        blit.copy_from_texture_to_texture(&source, 0, 3, zero, half, &big, 0, 0, zero)
            .unwrap();
        let right = Origin::new(4, 0, 0);
        blit.copy_from_texture_to_texture(&big, 0, 0, right, block, &other, 0, 2, zero)
            .unwrap();
        let low = Origin::new(0, 4, 0);
        blit.copy_from_texture_to_texture(&big, 0, 0, low, half, &other, 0, 4, zero)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);

        let seeded = |seed, extent, level| seeded_level(seed, extent, level).1;
        let copied = with_bytes(&out, |b| b.to_vec());
        for (at, want) in [
            (0, bc3_block(0, 2, 0)),
            (32, bc3_block(0, 4, 0)),
            (64, bc3_block(0, 2, 0)),
        ] {
            assert_eq!(copied[at..at + 16], want, "{at}");
            assert_eq!(copied[at + 16..at + 32], [0xee; 16], "{at}");
        }
        assert_eq!(bc3_blocks(&same, small, 3), written);
        assert_eq!(bc3_blocks(&same, small, 2), bc3_block(0, 2, 0));
        let mut want = seeded(3, large, 0);
        want[48..].copy_from_slice(&bc3_block(0, 2, 0));
        want[..16].copy_from_slice(&bc3_block(0, 3, 0));
        assert_eq!(bc3_blocks(&big, large, 0), want);
        assert_eq!(bc3_blocks(&other, small, 2), bc3_block(3, 0, 1));
        assert_eq!(bc3_blocks(&other, small, 4), bc3_block(3, 0, 2));
        for (t, seed, kept) in [(&same, 1, [0, 1, 4]), (&other, 2, [0, 1, 3])] {
            for l in kept {
                assert_eq!(
                    bc3_blocks(t, small, l),
                    seeded(seed, small, l),
                    "{seed} {l}"
                );
            }
        }

        // On the CPU (`cpu_get_l9_4x4`, `cpu_replace_l9_4x4`).
        let mut got = vec![0xeeu8; 32];
        source
            .get_bytes(&mut got, 16, 0, Region::new_2d(0, 0, 4, 4), 2, 0)
            .unwrap();
        assert_eq!(got[..16], bc3_block(0, 2, 0));
        assert_eq!(got[16..], [0xee; 16]);
        source
            .replace_region(Region::new_2d(0, 0, 4, 4), 4, 0, &written, 16, 0)
            .unwrap();
        assert_eq!(bc3_blocks(&source, small, 4), written);
        assert_eq!(bc3_blocks(&source, small, 3), seeded(0, small, 3));
    }
    layer_reported_nothing_at_all(&messages);
}

/// Copies through the encoder's scratch buffer each start where Vulkan takes a depth aspect's
/// bytes, at a multiple of 4: both levels of a 3 x 3 `Depth16Unorm` texture into an `R16Unorm`
/// one, level 0's 18 bytes before level 1's.
#[test]
fn depth_copies_through_the_scratch_buffer_start_at_multiples_of_four() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let d2 = TextureType::Type2D;
        let (shared, private) = (StorageMode::Shared, StorageMode::Private);
        let depth = texture(
            &device,
            PixelFormat::Depth16Unorm,
            d2,
            [3, 3, 1],
            2,
            1,
            private,
        );
        let colour = texture(&device, PixelFormat::R16Unorm, d2, [3, 3, 1], 2, 1, shared);
        let (depth, colour) = (depth.unwrap(), colour.unwrap());
        let source = counting(&device, 64);
        let (zero, none) = (Origin::new(0, 0, 0), BlitOption::None);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let three = Size::new(3, 3, 1);
        blit.copy_from_buffer_to_texture(&source, 0, 6, 0, three, &depth, 0, 0, zero, none)
            .unwrap();
        let one = Size::new(1, 1, 1);
        blit.copy_from_buffer_to_texture(&source, 32, 2, 0, one, &depth, 0, 1, zero, none)
            .unwrap();
        blit.copy_texture(&depth, &colour).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(level_bytes(&colour, 0, 0, 3, 3, 2), counted(0, 18));
        assert_eq!(level_bytes(&colour, 1, 0, 1, 1, 2), counted(32, 2));
    }
    layer_reported_nothing_at_all(&messages);
}

/// Copies between textures of different sizes copy what plain Metal copies (an M4 under macOS
/// 27.0.1; its validation stops the level copies): a level copy the region common to both levels
/// from their origins, the rest of the larger kept (`sz_slices_8x4_to_4x8`); a whole copy the
/// levels paired by size (`sz_whole_8l4_to_4l3`, `sz_whole_4l3_to_8l4`); and none where no levels
/// are of one size (`sz_whole_8_to_3`, which aborts plain Metal).
#[test]
fn copies_between_textures_of_different_sizes_copy_what_metal_copies() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let made = |w: u64, h: u64, levels: u64| {
            texture(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                [w, h, 1],
                levels,
                1,
                StorageMode::Shared,
            )
            .unwrap()
        };
        let side = |extent: u64, level: u64| (extent >> level).max(1);
        // Level `l` of a source counts from `0x40 * l`; every destination byte is 0xee.
        let fill = |t: &Texture, source: bool| {
            for level in 0..t.mipmap_level_count() {
                let (w, h) = (side(t.width(), level), side(t.height(), level));
                let n = (4 * w * h) as usize;
                let bytes = if source {
                    counted(0x40 * level as usize, n)
                } else {
                    vec![0xee; n]
                };
                t.replace_region(Region::new_2d(0, 0, w, h), level, 0, &bytes, 4 * w, 0)
                    .unwrap();
            }
        };
        let blit = |body: &dyn Fn(&mut BlitCommandEncoder) -> Result<(), Error>| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            let done = body(&mut encoder);
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            done
        };
        // An 8x4 level into a 4x8 one: their common 4x4.
        let (wide, tall) = (made(8, 4, 1), made(4, 8, 1));
        fill(&wide, true);
        fill(&tall, false);
        blit(&|b| b.copy_texture_slices(&wide, 0, 0, &tall, 0, 0, 1, 1)).unwrap();
        let source = counted(0, 4 * 8 * 4);
        let mut want: Vec<u8> = (0..4)
            .flat_map(|y| source[32 * y..32 * y + 16].to_vec())
            .collect();
        want.extend(vec![0xee; 4 * 4 * 4]);
        assert_eq!(
            level_bytes(&tall, 0, 0, 4, 8, 4),
            want,
            "sz_slices_8x4_to_4x8"
        );
        // Whole copies pair levels by size, either way.
        for (from, to, offset) in [((8, 4), (4, 3), 1), ((4, 3), (8, 4), -1)] {
            let (a, b) = (made(from.0, from.0, from.1), made(to.0, to.0, to.1));
            fill(&a, true);
            fill(&b, false);
            blit(&|e| e.copy_texture(&a, &b)).unwrap();
            for level in 0..to.1 {
                let w = side(to.0, level);
                let got = level_bytes(&b, level, 0, w, w, 4);
                let want = match u64::try_from(level as i64 + offset) {
                    Ok(src) if src < from.1 => counted(0x40 * src as usize, (4 * w * w) as usize),
                    _ => vec![0xee; (4 * w * w) as usize],
                };
                assert_eq!(got, want, "{from:?} into {to:?}, level {level}");
            }
        }
        // No levels of one size.
        let (eight, three) = (made(8, 8, 4), made(3, 3, 2));
        assert_eq!(
            blit(&|e| e.copy_texture(&eight, &three)),
            Err(invalid("copy_texture", Problem::Copy(CopyProblem::Size)))
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// `generate_mipmaps` leaves a texture of one level, of any format, and a depth or stencil
/// texture of any levels, as it is, as plain Metal does (an M4 under macOS 27.0.1; its validation
/// stops both).
#[test]
fn mipmaps_leave_single_levels_and_depth_as_they_are() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let d2 = TextureType::Type2D;
        let one = texture(
            &device,
            PixelFormat::RGBA8Uint,
            d2,
            [4, 4, 1],
            1,
            1,
            StorageMode::Shared,
        )
        .unwrap();
        one.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &counted(0, 64), 16, 0)
            .unwrap();
        let depth = texture(
            &device,
            PixelFormat::Depth32Float,
            d2,
            [4, 4, 1],
            3,
            1,
            StorageMode::Private,
        )
        .unwrap();
        let values: Vec<u8> = (0..21u16)
            .flat_map(|i| (f32::from(i) / 32.0).to_le_bytes())
            .collect();
        let levels = device.new_buffer(84, ResourceOptions::shared()).unwrap();
        with_bytes(&levels, |b| b.copy_from_slice(&values));
        let back = device.new_buffer(84, ResourceOptions::shared()).unwrap();
        // Levels 4x4, 2x2 and 1x1 at bytes 0, 64 and 80.
        let at = Origin::new(0, 0, 0);
        let shapes = [(0, 16, 4), (64, 8, 2), (80, 4, 1)];
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (level, (offset, bpr, side)) in shapes.into_iter().enumerate() {
            let size = Size::new(side, side, 1);
            let level = level as u64;
            blit.copy_from_buffer_to_texture(
                &levels,
                offset,
                bpr,
                0,
                size,
                &depth,
                0,
                level,
                at,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.generate_mipmaps(&one).unwrap();
        blit.generate_mipmaps(&depth).unwrap();
        for (level, (offset, bpr, side)) in shapes.into_iter().enumerate() {
            let size = Size::new(side, side, 1);
            let level = level as u64;
            blit.copy_from_texture_to_buffer(
                &depth,
                0,
                level,
                at,
                size,
                &back,
                offset,
                bpr,
                0,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(level_bytes(&one, 0, 0, 4, 4, 4), counted(0, 64));
        assert_eq!(with_bytes(&back, |b| b.to_vec()), values);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind `generate_mipmaps` takes where a filtered format's chain ran, the device's blits or
/// the compute filter: in [`mipmaps_of_every_format_run_or_are_refused_typed`] and
/// `mipmap_chains`.
pub(super) const MIPMAPS: [Degradation; 1] = [Degradation::DeviceFilteredMipmaps];

/// Whether plain Metal mipmaps `pf` by picking texels, which the runtime copies exactly on every
/// device: a color format a shader reads as integers, which an extended-range format is not
/// (`mipmap_chains::integer_mipmaps_pick_the_texels_plain_metal_picks`).
fn picked(pf: PixelFormat) -> bool {
    use metal2vulkan::pipeline::SampledType;
    !is_depth_stencil(pf)
        && matches!(
            metal2vulkan::metal::format::sampled_type(pf),
            Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
        )
}

/// The kind [`depth_outside_the_unit_range_uploads_as_metal_holds_it`] takes when it uploads into
/// a depth image on a device whose depth image may clamp ([`depth_images`]).
pub(super) const UPLOADS: [Degradation; 1] = [Degradation::RestrictedDepthUpload];

/// 32-bit float depth outside `[0, 1]` uploads into a depth image, through `replace_region` and
/// through a blit from a buffer, on every device.
///
/// A depth image, a read-only texture's and a render target's alike, keeps each value where the
/// device enabled `VK_EXT_depth_range_unrestricted` or its driver is measured to keep one without
/// it; elsewhere Vulkan leaves such a value undefined
/// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`) and a device keeps it or stores it clamped to
/// `[0, 1]`, which its first upload announces once ([`UPLOADS`]): each value reads back as one of
/// the two, and the form the device took is printed; where it keeps them, the upload of a negative
/// value into a render target, whose depth a test reads, announces `signed-depth-test` where the
/// device's depth test compares the value as it is (`degrade_notices::signs_depth_tests`). A copy
/// between two depth images moves the source's words, and a copy into a render target announces the
/// same.
#[test]
fn depth_outside_the_unit_range_uploads_as_metal_holds_it() {
    const VALUES: [f32; 4] = [-0.5, 2.0, 1000.0, 0.25];
    let words: Vec<u8> = VALUES.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name();
        let (unrestricted, clamps, _) = depth_images(&device);
        let queue = device.new_command_queue().unwrap();
        let made = |usage: TextureUsage| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 2, 2, false);
            d.usage = usage;
            d.resource_options = ResourceOptions::new(
                StorageMode::Shared,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            device.new_texture(&d).unwrap()
        };
        let read_only = TextureUsage::SHADER_READ;
        let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let floats = |texture: &Texture| -> Vec<f32> {
            level_bytes(texture, 0, 0, 2, 2, 4)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
                .collect()
        };
        // Each value as written, or (where the device's depth image may clamp) each clamped.
        let form = |how: &str, read: &[f32]| {
            let kept = read == VALUES;
            let clamped = read == VALUES.map(|value| value.clamp(0.0, 1.0));
            assert!(kept || (clamps && clamped), "{how} on {name}: {read:?}");
            eprintln!(
                "32-bit float depth outside [0, 1] through {how} on {name} \
                 (VK_EXT_depth_range_unrestricted {}): {}",
                if unrestricted { "enabled" } else { "absent" },
                if kept { "kept" } else { "clamped" }
            );
        };

        // What the device announced before its first texture.
        messages.take_notices();
        let replaced = made(read_only);
        let blitted = made(read_only);
        let held = made(target);
        assert_once(&messages.take_notices(), &[]);
        replaced
            .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &words, 8, 0)
            .unwrap();
        let clamped = forced(&[(Degradation::RestrictedDepthUpload, clamps)]);
        let uploaded = assert_made_once(&messages, &[], &clamped);
        for kind in UPLOADS {
            assert_eq!(
                uploaded.contains_key(&kind),
                clamps,
                "{kind:?} on {name} exactly where the device may clamp: {uploaded:?}"
            );
        }
        held.replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &words, 8, 0)
            .unwrap();
        let signed = forced(&[(Degradation::SignedDepthTest, signs_depth_tests(&device))]);
        assert_made_once(&messages, &[], &signed);
        let source = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        with_bytes(&source, |bytes| bytes.copy_from_slice(&words));
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            8,
            0,
            Size::new(2, 2, 1),
            &blitted,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        form("replace_region", &floats(&replaced));
        form("a blit", &floats(&blitted));
        form("replace_region into a render target", &floats(&held));

        // A copy between two read-only textures, out of one into a render target, and back.
        let (copied, into_target, back) = (made(read_only), made(target), made(read_only));
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture(&replaced, &copied).unwrap();
        blit.copy_texture(&replaced, &into_target).unwrap();
        blit.copy_texture(&held, &back).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let replaced = floats(&replaced);
        assert_eq!(
            floats(&copied),
            replaced,
            "a copy between depth images on {name}"
        );
        assert_eq!(
            floats(&into_target),
            replaced,
            "a copy into a render target on {name}"
        );
        assert_eq!(
            floats(&back),
            floats(&held),
            "a copy out of a render target on {name}"
        );
        // The blit from the `Shared` buffer writes a depth image on the GPU, which announces what
        // a float depth's may store; every other copy into a depth image copies one depth
        // image's bits into another.
        let first = forced(&[
            (
                Degradation::DepthDenormals,
                loses_depth_denormals(&device, false),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        let later = assert_made_once(&messages, &[], &first);
        for kind in &UPLOADS {
            assert!(
                !later.contains_key(kind),
                "{kind:?} announced again on {name}: {later:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// What `device`'s depth images keep, as the runtime plans them: whether it enabled
/// `VK_EXT_depth_range_unrestricted`; whether its uploads clamp a value outside `[0, 1]`
/// (without the extension, `degrade_notices::clamps_depth_uploads`); and whether it keeps every
/// `float` (it does not clamp, and its driver has `DriverFact::DepthImageKeepsDenormals`).
pub(super) fn depth_images(device: &Device) -> (bool, bool, bool) {
    let unrestricted = device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let facts = &device.properties().driver_facts;
    let clamps = super::degrade_notices::clamps_depth_uploads(device);
    let keeps_floats = !clamps && facts.contains(&DriverFact::DepthImageKeepsDenormals);
    (unrestricted, clamps, keeps_floats)
}

/// The denormals [`depth_images_keep_what_their_drivers_claim`] writes: both signs, the smallest,
/// the largest and values between (`0x000116c2` is `1e-40`).
const DENORMALS: [u32; 11] = [
    0x0000_0001,
    0x8000_0001,
    0x0000_0100,
    0x0001_16c2,
    0x8001_16c2,
    0x0010_0000,
    0x0040_0000,
    0x8040_0000,
    0x0055_5555,
    0x007f_ffff,
    0x807f_ffff,
];

/// The values outside `[0, 1]` it writes.
const OUTSIDE: [f32; 5] = [-0.5, 2.0, 1000.0, -1000.0, 1.5];

/// What a `D32_SFLOAT` image keeps of the `float`s Metal's `Depth32Float` texel holds, against
/// the driver's denormal fact (`DriverFact::DepthImageKeepsDenormals`), which this lane measures
/// and checks at each gate, and the upload's clamp.
///
/// A render target's `Depth32Float` texture, which has a depth image on every device, takes 11
/// denormals and 5 values outside `[0, 1]` through `replace_region` and reads them back through
/// `get_bytes`. Each count kept is printed with the driver's `driverID` and `driverVersion`. A
/// driver with the fact keeps every denormal the upload gives it; every device keeps the values
/// outside `[0, 1]` where it enabled `VK_EXT_depth_range_unrestricted`, and elsewhere its upload
/// clamps each of them, and the negative denormals, as `bind::clamped_depth` does
/// (`clamps_depth_uploads`). The shader's reads and comparisons of such a texture are
/// `depth_compares`' (Metal's answers there).
#[test]
fn depth_images_keep_what_their_drivers_claim() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let (unrestricted, _, _) = depth_images(&device);
        let properties = device.properties();
        // The constructor checked the creation notices against `creation_kinds`.
        assert_once(&messages.take_notices(), &[]);
        let facts = &properties.driver_facts;
        let description = &properties.description;
        let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 4, 4, false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let texture = device.new_texture(&d).unwrap();
        let words: Vec<u32> = DENORMALS
            .iter()
            .copied()
            .chain(OUTSIDE.iter().map(|value| value.to_bits()))
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        texture
            .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &bytes, 16, 0)
            .unwrap();
        let read: Vec<u32> = level_bytes(&texture, 0, 0, 4, 4, 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let kept = |range: std::ops::Range<usize>| range.filter(|&i| read[i] == words[i]).count();
        let (denormals, outside) = (kept(0..11), kept(11..16));
        eprintln!(
            "depth image on {name} (driverID {:?}, driverVersion {:#x}, \
             VK_EXT_depth_range_unrestricted {}): {denormals} of 11 denormals kept, {outside} of \
             5 values outside [0, 1] kept; read {read:08x?}",
            description.driver_id,
            description.driver_version,
            if unrestricted { "enabled" } else { "absent" },
        );
        // What the upload gives the image: each value, or its clamp where uploads clamp.
        let clamps = super::degrade_notices::clamps_depth_uploads(&device);
        let given = |i: usize| match clamps {
            true => metal2vulkan::bind::clamped_depth(words[i]),
            false => words[i],
        };
        if facts.contains(&DriverFact::DepthImageKeepsDenormals) {
            let lost: Vec<usize> = (0..11).filter(|&i| read[i] != given(i)).collect();
            assert!(
                lost.is_empty(),
                "the driver's denormal fact on {name}: {lost:?} of {read:08x?}"
            );
        }
        if unrestricted {
            assert_eq!(outside, 5, "values outside [0, 1] on {name}: {read:08x?}");
        }
        let off: Vec<usize> = (11..16).filter(|&i| read[i] != given(i)).collect();
        assert!(
            unrestricted || off.is_empty(),
            "values outside [0, 1] on {name}, clamped as uploaded: {off:?} of {read:08x?}"
        );
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}

/// `generate_mipmaps` of every pixel format in 2D (levels that halve exactly and levels that do
/// not) and 3D: refused as Metal refuses it, left as it is for a depth or stencil texture, an
/// integer format's texels picked, and every other format's levels made by the device's blit
/// chain or, where its blits cannot filter the format, by the compute filter, as the work counts
/// say; each format's way printed. A filtered chain announces `device-filtered-mipmaps` once
/// ([`MIPMAPS`]). Refused `NotImplemented(MipmapGeneration)` only for a format the compute filter
/// does not compute (the RGBX formats among them), never for one it does.
#[test]
fn mipmaps_of_every_format_run_or_are_refused_typed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // What the device announced before its first mipmaps.
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (mut blitted, mut filtered, mut picks, mut metal, mut gaps) = (0, 0, 0, 0, 0);
        let mut lines = Vec::new();
        // The kinds the textures' makings announce.
        let mut first = BTreeSet::new();
        for &pf in PixelFormat::ALL {
            let mut outcome = Vec::new();
            // Chains that halve exactly in 2D and 3D, and one that does not.
            for (ty, extent, levels) in [
                (TextureType::Type2D, [8, 8, 1], 4),
                (TextureType::Type2D, [7, 5, 1], 3),
                (TextureType::Type3D, [4, 4, 4], 3),
            ] {
                let Ok(t) = texture(&device, pf, ty, extent, levels, 1, StorageMode::Private)
                else {
                    outcome.push("no texture");
                    continue;
                };
                // A carried texture's decoded twin (mipmaps read no twin, so its precision is
                // never announced here), and a BC texture's device decode.
                first.extend(forced(&[
                    (Degradation::CarriedDecodedTwin, twins_carried(&device, pf)),
                    (
                        Degradation::CompressedTexelDecode,
                        decodes_blocks_its_way(&device, pf),
                    ),
                ]));
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                let before = device.work_counts();
                let result = blit.generate_mipmaps(&t);
                let after = device.work_counts();
                let blits = after.image_blits_recorded - before.image_blits_recorded;
                let filter = after.mipmaps_filtered - before.mipmaps_filtered;
                match result {
                    // Plain Metal leaves a depth or stencil texture as it is.
                    Ok(()) if is_depth_stencil(pf) => outcome.push("left as it is"),
                    Ok(()) if picked(pf) => {
                        assert_eq!((blits, filter), (0, 0), "{pf:?} {ty:?}: picks");
                        picks += 1;
                        outcome.push("picked");
                    }
                    Ok(()) if filter == 1 => {
                        assert_eq!(blits, 0, "{pf:?} {ty:?}: the filter alone");
                        filtered += 1;
                        outcome.push("compute filter");
                    }
                    Ok(()) => {
                        assert_eq!(blits, levels - 1, "{pf:?} {ty:?}: a blit a level");
                        blitted += 1;
                        outcome.push("blits");
                    }
                    Err(Error::Invalid(Invalid {
                        problem: Problem::Mipmap(_),
                        ..
                    })) => {
                        metal += 1;
                        outcome.push("Metal refuses");
                    }
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(u.feature, Feature::MipmapGeneration, "{pf:?}: {u:?}");
                        assert!(
                            x_channel(pf) || MipmapFilterChain::texel_bytes(pf).is_none(),
                            "{pf:?} {ty:?}: refused where the compute filter computes it: {u:?}"
                        );
                        gaps += 1;
                        outcome.push("refused");
                    }
                    Err(other) => panic!("{pf:?}: {other}"),
                }
                blit.end_encoding().unwrap();
                run(&command_buffer);
            }
            if outcome
                .iter()
                .any(|o| !matches!(*o, "Metal refuses" | "no texture" | "left as it is"))
            {
                lines.push(format!(
                    "  {}: 2D 8x8 {}, 2D 7x5 {}, 3D 4x4x4 {}",
                    pf.name(),
                    outcome[0],
                    outcome[1],
                    outcome[2],
                ));
            }
        }
        eprintln!(
            "mipmaps on {}: {blitted} by the device's blits, {filtered} by the compute filter \
             (both the device's filter, not Apple's: a near-equivalent), {picks} picked (plain \
             Metal's texels), {metal} refused as Metal refuses them, {gaps} refused (not \
             implemented)",
            device.name()
        );
        for line in lines {
            eprintln!("{line}");
        }
        assert!(metal > 0);
        assert!(blitted > 0);
        let ran = blitted + filtered > 0;
        first.extend(forced(&[(Degradation::DeviceFilteredMipmaps, ran)]));
        let announced = assert_made_once(&messages, &[], &first);
        for kind in MIPMAPS {
            assert_eq!(
                announced.contains_key(&kind),
                ran,
                "{kind:?} exactly where a chain ran: {announced:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
