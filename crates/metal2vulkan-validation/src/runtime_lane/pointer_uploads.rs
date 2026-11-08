//! Uploads from bytes at a pointer (`HostBytes`), on the device the cfg selects:
//! `Texture::replace_region_with_host_bytes` and `BufferContents::CopyHost` write exactly what their
//! slice forms write: a texture with an image (its rows packed straight from the pointer), one
//! over a buffer (the bytes read once into a copy first), and Shared and Private buffers. Refused
//! as the slice forms are.

use super::host_memory::{gpu_read, on_device};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BufferContents, BufferDescriptor, CopyProblem, Device, Error, HostBytes, Problem, Region,
    ResourceOptions, Texture, TextureDescriptor,
};
use std::ptr::NonNull;

/// The texture's side, in texels.
const SIDE: u64 = 16;

/// Host bytes over `bytes`, as a caller with memory of its own makes them.
fn over(bytes: &[u8]) -> HostBytes<'_> {
    // SAFETY: `bytes` is a live slice, readable, which nothing writes while a call reads it.
    unsafe { HostBytes::new(NonNull::from(bytes).cast(), bytes.len()) }
}

/// A Shared RGBA8 texture of `SIDE` by `SIDE`: with an image of its own, or over a buffer.
fn texture(device: &Device, over_buffer: bool) -> Texture {
    let format = PixelFormat::RGBA8Unorm;
    let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    if !over_buffer {
        return device.new_texture(&d).unwrap();
    }
    let align = device
        .minimum_linear_texture_alignment_for_pixel_format(format)
        .unwrap();
    let row = (SIDE * 4).next_multiple_of(align.max(1));
    let buffer = device
        .new_buffer(row * SIDE, ResourceOptions::shared())
        .unwrap();
    buffer.new_texture(&d, 0, row).unwrap()
}

/// Every texel of `texture`, tightly packed.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut all = vec![0; (SIDE * SIDE * 4) as usize];
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .get_bytes(&mut all, SIDE * 4, 0, whole, 0, 0)
        .unwrap();
    all
}

/// A region written from host bytes holds what the slice form writes, rows pitched past the
/// region's: the texture with an image, and the one over a buffer. Bytes short of the rows are
/// refused as the slice form refuses them.
#[test]
fn a_texture_replaced_from_host_bytes_holds_what_its_slice_form_writes() {
    on_device(|device| {
        let region = Region::new_2d(2, 3, 8, 5);
        let pitch = 8 * 4 + 12;
        let bytes: Vec<u8> = (0..pitch * 5).map(|i| (i * 13 + 7) as u8).collect();
        for over_buffer in [false, true] {
            let (from_slice, from_host) =
                (texture(device, over_buffer), texture(device, over_buffer));
            from_slice
                .replace_region(region, 0, 0, &bytes, pitch, 0)
                .unwrap();
            from_host
                .replace_region_with_host_bytes(region, 0, 0, over(&bytes), pitch, 0)
                .unwrap();
            let written = texels(&from_host);
            assert!(
                written == texels(&from_slice),
                "over a buffer: {over_buffer}"
            );
            let first = ((3 * SIDE + 2) * 4) as usize;
            assert_eq!(&written[first..first + 32], &bytes[..32], "{over_buffer}");
            // One byte short of what the rows read: the last row ends a row past its pitch's start.
            let short = &bytes[..(4 * pitch + 8 * 4 - 1) as usize];
            let refused =
                from_host.replace_region_with_host_bytes(region, 0, 0, over(short), pitch, 0);
            let Err(Error::Invalid(invalid)) = refused else {
                panic!("short bytes: {refused:?}");
            };
            assert_eq!(invalid.problem, Problem::Copy(CopyProblem::BufferRange));
        }
        eprintln!("HOST-BYTES texture: as the slice form, with an image and over a buffer");
    });
}

/// A buffer made from host bytes holds them, Shared and Private, as one made from the slice does;
/// a length other than the descriptor's is refused as the slice form's is.
#[test]
fn a_buffer_made_from_host_bytes_holds_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 31 + 5) as u8).collect();
        let length = bytes.len() as u64;
        for options in [ResourceOptions::shared(), ResourceOptions::private()] {
            let contents = BufferContents::CopyHost(over(&bytes));
            let descriptor = BufferDescriptor::new(length, options, contents);
            let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
            assert!(gpu_read(&queue, &buffer, 0..length) == bytes, "{options:?}");
        }
        let contents = BufferContents::CopyHost(over(&bytes));
        let descriptor = BufferDescriptor::new(length + 4, ResourceOptions::shared(), contents);
        let refused = device.new_buffer_with_descriptor(descriptor);
        let Err(Error::Invalid(invalid)) = refused else {
            panic!("a length mismatch: {refused:?}");
        };
        assert_eq!(invalid.problem, Problem::LengthMismatch);
        eprintln!("HOST-BYTES buffer: Shared and Private hold the bytes");
    });
}
