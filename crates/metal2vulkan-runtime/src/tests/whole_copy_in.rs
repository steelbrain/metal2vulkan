//! A copy from a buffer that writes a whole subresource moves that level and layer from undefined
//! (`TextureObject::away_into`), and every other copy into a texture moves it from home.
//!
//! - The move's old layout: undefined for a region from the level's first texel to its edge on
//!   every axis of every aspect, at the first level and past it, a 3D level's depth whole; home
//!   for a region short of the level on any axis (a 3D level's depth among them), one aspect of a
//!   depth and stencil format, and a texture in place over a buffer, whose bytes the buffer keeps.
//!   The move's range and its source scope are the old form's either way.
//! - Whole refills through rt: an R8Unorm 1920 x 1080 texture of ShaderRead and RenderTarget,
//!   refilled [`REFILLS`] times on one queue by a copy from a Shared staging buffer into a Private
//!   one and from that into the texture, then read back whole to Shared memory, each read
//!   compared byte for byte with what the host wrote; every byte of a refill differs from the one
//!   before it. With the move from home, a Radeon 780M under RADV (Mesa 26.2.6) left one aligned
//!   4 x 2 cell of the previous refill in about 2 of 1000 such refills (7 of 4000 in a pure Vulkan
//!   repro of the same calls), and none with the move from undefined.

use crate::tests::command::on_device::{each_device, finished};
use crate::texture::{Placed, TextureObject};
use crate::{
    BufferLayout, CommandBufferStatus, Device, IndexClass, MemoryClass, PixelFormat, StorageMode,
    Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
};
use ash::vk;

/// The refills of the repeat test.
const REFILLS: u64 = 1000;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn private(pf: PixelFormat, [w, h, d]: [u32; 3], usage: TextureUsage) -> TextureDescriptor {
    let kind = match d {
        1 => TextureType::Type2D,
        _ => TextureType::Type3D,
    };
    let mut descriptor = TextureDescriptor::new(kind, pf, w, h);
    descriptor.depth = d;
    descriptor.usage = usage;
    descriptor.storage_mode = StorageMode::Private;
    descriptor
}

/// The old layout of the move before a copy of `aspect` into `origin` and `size` of `level` of
/// `texture`'s first layer.
fn from(
    texture: &Texture,
    aspect: vk::ImageAspectFlags,
    level: u32,
    (origin, size): ([u32; 3], [u32; 3]),
) -> vk::ImageLayout {
    let placed = Placed {
        level,
        layer: 0,
        origin,
        size,
    };
    texture.away_into(&placed, aspect).old_layout
}

#[test]
fn the_move_before_a_copy_in_starts_from_undefined_for_a_whole_subresource() {
    each_device("whole_copy_in_moves", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 8)).expect("indices");
        let read = TextureUsage::SHADER_READ;
        let make = |d: &TextureDescriptor, index: u32| {
            (device.new_texture(d, Some(index)))
                .unwrap_or_else(|error| panic!("{name}: a texture: {error}"))
        };
        let colour = vk::ImageAspectFlags::COLOR;
        let undefined = vk::ImageLayout::UNDEFINED;
        // R8, 64 x 32, three levels: level 1 is 32 x 16.
        let mut d = private(PixelFormat::R8Unorm, [64, 32, 1], read);
        d.mipmap_level_count = 3;
        let flat = make(&d, 1);
        assert_eq!(
            from(&flat, colour, 0, ([0; 3], [64, 32, 1])),
            undefined,
            "{name}"
        );
        assert_eq!(
            from(&flat, colour, 1, ([0; 3], [32, 16, 1])),
            undefined,
            "{name}"
        );
        for short in [
            ([0, 0, 0], [63, 32, 1]),
            ([0, 0, 0], [64, 31, 1]),
            ([1, 0, 0], [63, 32, 1]),
            ([0, 1, 0], [64, 31, 1]),
            ([0, 0, 0], [64, 16, 1]),
        ] {
            assert_ne!(
                from(&flat, colour, 0, short),
                undefined,
                "{name}: {short:?}"
            );
        }
        assert_ne!(
            from(&flat, colour, 1, ([0; 3], [64, 32, 1])),
            undefined,
            "{name}"
        );
        // A 3D level: its depth whole, and short.
        let volume = make(&private(PixelFormat::R8Unorm, [16, 16, 8], read), 2);
        assert_eq!(
            from(&volume, colour, 0, ([0; 3], [16, 16, 8])),
            undefined,
            "{name}"
        );
        assert_ne!(
            from(&volume, colour, 0, ([0; 3], [16, 16, 7])),
            undefined,
            "{name}"
        );
        assert_ne!(
            from(&volume, colour, 0, ([0, 0, 1], [16, 16, 7])),
            undefined,
            "{name}"
        );
        // One aspect of depth and stencil, and both.
        let both = private(
            PixelFormat::Depth32FloatStencil8,
            [64, 32, 1],
            TextureUsage::RENDER_TARGET,
        );
        if let Ok(both) = device.new_texture(&both, Some(3)) {
            let whole = ([0; 3], [64, 32, 1]);
            let all = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
            assert_eq!(from(&both, all, 0, whole), undefined, "{name}");
            for one in [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL] {
                assert_ne!(from(&both, one, 0, whole), undefined, "{name}: {one:?}");
            }
        }
        // A texture in place over a buffer keeps the buffer's bytes.
        let align = (device.minimum_linear_texture_alignment(PixelFormat::R8Unorm))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let pitch = align.max(256);
        let block = (device.take_block(MemoryClass::Shared, 64 * pitch)).expect("a block");
        let buffer = (device.new_buffer(block.id, 0, 64 * pitch, None)).expect("a buffer");
        let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::R8Unorm, 256, 64);
        d.usage = read;
        let over = (buffer.new_texture(&d, Some(4), 0, pitch)).expect("a texture over the buffer");
        if over.copied_over().is_none() {
            assert_ne!(
                from(&over, colour, 0, ([0; 3], [256, 64, 1])),
                undefined,
                "{name}"
            );
        }
    });
}

/// The byte at column `x` of row `y` in refill `r`: 37 more each refill.
fn pattern(r: u64, x: u32, y: u32) -> u8 {
    (u64::from((x * 3) ^ (y * 5)) + r * 37) as u8
}

#[test]
fn whole_refills_of_a_render_target_each_read_back_whole() {
    each_device("whole_copy_in_refills", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 1)).expect("indices");
        let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let d = private(PixelFormat::R8Unorm, [WIDTH, HEIGHT, 1], usage);
        let texture = (device.new_texture(&d, Some(1))).expect("the texture");
        refill(name, &device, &texture);
    });
}

fn refill(name: &str, device: &Device, texture: &Texture) {
    let bytes = u64::from(WIDTH * HEIGHT);
    let room = bytes.next_multiple_of(1 << 20);
    let shared = (device.take_block(MemoryClass::Shared, 2 * room)).expect("a Shared block");
    let private = (device.take_block(MemoryClass::Private, room)).expect("a Private block");
    let buffer = |block: &crate::Block, offset: u64| {
        (device.new_buffer(block.id, offset, bytes, None)).expect("a buffer")
    };
    let (staging, back, local) = (
        buffer(&shared, 0),
        buffer(&shared, room),
        buffer(&private, 0),
    );
    let host = |buffer: &crate::Buffer| buffer.contents().expect("a Shared buffer's contents");
    let queue = device.new_command_queue(4).expect("a queue");
    let layout = BufferLayout {
        offset: 0,
        bytes_per_row: u64::from(WIDTH),
        bytes_per_image: bytes,
    };
    let place = TexturePlace {
        slice: 0,
        level: 0,
        origin: [0; 3],
    };
    let mut want = vec![0u8; bytes as usize];
    let mut stale = Vec::new();
    for r in 1..=REFILLS {
        for (at, byte) in want.iter_mut().enumerate() {
            let (x, y) = (at as u32 % WIDTH, at as u32 / WIDTH);
            *byte = pattern(r, x, y);
        }
        // SAFETY: the staging buffer is mapped for `bytes`, and no work uses it now.
        unsafe {
            std::ptr::copy_nonoverlapping(want.as_ptr(), host(&staging).as_ptr(), want.len())
        };
        let fill = queue.command_buffer().expect("a command buffer");
        let mut e = fill.blit_command_encoder().expect("an encoder");
        (e.copy_from_buffer_to_buffer(&staging, 0, &local, 0, bytes)).expect("the staging copy");
        (e.copy_from_buffer(&local, layout, [WIDTH, HEIGHT, 1], texture, place))
            .expect("the copy in");
        e.end_encoding();
        fill.commit().expect("a commit");
        let read = queue.command_buffer().expect("a command buffer");
        let mut e = read.blit_command_encoder().expect("an encoder");
        (e.copy_from_texture_to_buffer(texture, place, [WIDTH, HEIGHT, 1], &back, layout))
            .expect("the read");
        e.end_encoding();
        read.commit().expect("a commit");
        for (what, cb) in [("read", &read), ("fill", &fill)] {
            finished(name, cb);
            assert_eq!(
                cb.status(),
                CommandBufferStatus::Completed,
                "{name}: the {what}"
            );
        }
        // SAFETY: the read-back is mapped for `bytes`, and its work has completed.
        let got = unsafe { std::slice::from_raw_parts(host(&back).as_ptr(), bytes as usize) };
        if got != want.as_slice() {
            let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            let first = got.iter().zip(&want).position(|(g, w)| g != w).unwrap_or(0);
            let (x, y) = (first as u32 % WIDTH, first as u32 / WIDTH);
            stale.push(format!(
                "refill {r}: {wrong} bytes, the first at ({x}, {y})"
            ));
        }
    }
    assert!(
        stale.is_empty(),
        "{name}: refills whose read-back differs from what the host wrote, of {REFILLS}: {stale:?}"
    );
}
