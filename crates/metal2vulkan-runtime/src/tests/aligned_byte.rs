//! A buffer of the runtime's own memory that lays a byte on a multiple of 256 bytes of its memory
//! (`BufferDescriptor::aligned_byte`), as a consumer's device cache of a decoded frame whose
//! planes CoreVideo lays 32 bytes into their surface:
//!
//! - **The planes in place.** Over `Shared` and `Private` buffers laying byte `0x20` so, an 8-bit
//!   frame's planes (`R8Unorm` 3840x2160 at `0x20` and `RG8Unorm` 1920x1080 at `0x20 + 0x7E9000`,
//!   rows 3840 bytes apart) and a 10-bit frame's (`R16Unorm` at `0x20` and `RG16Unorm` at
//!   `0x20 + 0xFD2000`, rows 7680 bytes apart), each sampled, written by a shader and rendered
//!   into, are each the linear image aliasing the buffer's memory (no `shadowed_textures`) wherever
//!   the same texture at offset 0 of a buffer made without the option is one; at least one is. Over
//!   a buffer made without it, the same textures at `0x20` are shadows wherever the device's
//!   linear-image alignment (`Device::minimum_linear_texture_alignment_for_pixel_format`) is past
//!   32 bytes.
//! - **The texels.** A blit of each sampled plane into a buffer reads the bytes written at the
//!   plane's offset: through `contents` for a `Shared` buffer, by a blit from a staging buffer
//!   for a `Private` one.
//! - **Metal's contract.** The buffer's length is the descriptor's, its `gpuAddress` lies on
//!   Metal's 16-KiB page as every range the runtime picks, and a `Shared` one's contents put byte
//!   `0x20` on a multiple of 64 bytes (`minMemoryMapAlignment` is at least 64), 32 bytes past one.
//! - **Over the caller's memory**, the option is refused: `Invalid` (`AlignedByteNoCopy`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`ALIGNED-BYTE`): each plane's
//! paths with and without the option.

use super::blit_holds::{device, Host, Scratch};
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, Error, Invalid, IoSurfacePlane,
    Origin, Problem, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::Duration;

/// The byte the buffers lay on a multiple of 256: CoreVideo's first plane's.
const K: u64 = 0x20;

/// The longest a command buffer may take here.
const WAIT: Duration = Duration::from_secs(20);

/// A plane of a frame: its format, size, bytes a row and offset in the surface.
#[derive(Clone, Copy, Debug)]
struct Plane {
    format: PixelFormat,
    size: [u64; 2],
    row: u64,
    at: u64,
}

/// The 8-bit and 10-bit frames' planes, each 3840x2160 luma and 1920x1080 chroma, 32 bytes into
/// the surface and right after the luma rows.
const PLANES: [Plane; 4] = [
    Plane {
        format: PixelFormat::R8Unorm,
        size: [3840, 2160],
        row: 3840,
        at: K,
    },
    Plane {
        format: PixelFormat::RG8Unorm,
        size: [1920, 1080],
        row: 3840,
        at: K + 3840 * 2160,
    },
    Plane {
        format: PixelFormat::R16Unorm,
        size: [3840, 2160],
        row: 7680,
        at: K,
    },
    Plane {
        format: PixelFormat::RG16Unorm,
        size: [1920, 1080],
        row: 7680,
        at: K + 7680 * 2160,
    },
];

/// The surface's bytes: the 10-bit frame's planes, the longer.
const LENGTH: u64 = K + 7680 * 2160 * 3 / 2;

/// The usages each plane is made with: sampled, written by a shader, rendered into.
const USAGES: [TextureUsage; 3] = [
    TextureUsage::SHADER_READ,
    TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
    TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
];

/// The byte the surface holds at `offset`.
fn surface_byte(offset: u64) -> u8 {
    (offset * 7 + offset / 3851 + 5) as u8
}

/// A buffer of the runtime's own memory of [`LENGTH`] bytes of `options`, laying byte `aligned`
/// on a multiple of 256 where given, holding [`surface_byte`]s.
fn surface(device: &Device, options: ResourceOptions, aligned: Option<u64>) -> Buffer {
    let bytes: Vec<u8> = (0..LENGTH).map(surface_byte).collect();
    let descriptor = BufferDescriptor::new(LENGTH, options, BufferContents::Copy(&bytes));
    let descriptor = match aligned {
        Some(byte) => descriptor.with_aligned_byte(byte),
        None => descriptor,
    };
    device.new_buffer_with_descriptor(descriptor).unwrap()
}

/// `plane`'s texture of `usage` over `buffer` at `at` (the consumer's IOSurface plane), and
/// whether it is the linear image aliasing the buffer's memory: no shadow counted as it is made.
fn plane_texture(
    device: &Device,
    buffer: &Buffer,
    (plane, at): (Plane, u64),
    usage: TextureUsage,
) -> (Texture, bool) {
    let [w, h] = plane.size;
    let mut descriptor = TextureDescriptor::texture_2d(plane.format, w, h, false);
    descriptor.usage = usage;
    descriptor.resource_options = buffer.resource_options();
    let before = device.work_counts().shadowed_textures;
    let texture = buffer
        .new_texture_with_iosurface_plane(&descriptor, IoSurfacePlane::new(at, plane.row, true))
        .unwrap();
    let aliases = device.work_counts().shadowed_textures == before;
    (texture, aliases)
}

/// The texels of `plane`'s `texture` as a blit copies them into a buffer, rows one after another.
fn blitted(device: &Device, texture: &Texture, plane: Plane) -> Vec<u8> {
    let [w, h] = plane.size;
    let bytes = plane.row * h;
    let read = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &read,
        0,
        plane.row,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    // SAFETY: the read buffer's mapped bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

#[test]
fn a_plane_at_the_aligned_byte_aliases_its_buffer_where_one_at_offset_0_does() {
    let scratch = Scratch::new("aligned-byte");
    let device = device(&scratch.0);
    let mut aliased = 0;
    for options in [ResourceOptions::shared(), ResourceOptions::private()] {
        let laid = surface(&device, options, Some(K));
        let plain = surface(&device, options, None);
        assert_eq!(
            laid.length(),
            LENGTH,
            "{options:?}: the descriptor's length"
        );
        assert_eq!(
            laid.gpu_address().0 % (16 << 10),
            0,
            "{options:?}: gpuAddress on Metal's page"
        );
        if let Some(contents) = laid.contents() {
            let at = contents.as_ptr() as u64;
            assert_eq!(
                (at + K) % 64,
                0,
                "{options:?}: byte {K:#x} on a multiple of 64"
            );
            assert_eq!(
                at % 64,
                32,
                "{options:?}: contents 32 bytes past a multiple of 64"
            );
        }
        for plane in PLANES {
            let alignment = device
                .minimum_linear_texture_alignment_for_pixel_format(plane.format)
                .unwrap();
            for usage in USAGES {
                let (texture, at_k) = plane_texture(&device, &laid, (plane, plane.at), usage);
                let (_, at_0) = plane_texture(&device, &plain, (plane, plane.at - K), usage);
                let (_, unlaid) = plane_texture(&device, &plain, (plane, plane.at), usage);
                println!(
                    "ALIGNED-BYTE {} {options:?} {plane:?} {usage:?}: laid at {:#x} aliases {at_k}, \
                     at {:#x} without the option {at_0}, at {:#x} without it {unlaid} (linear \
                     alignment {alignment})",
                    device.name(),
                    plane.at,
                    plane.at - K,
                    plane.at,
                );
                assert_eq!(
                    at_k, at_0,
                    "{options:?} {plane:?} {usage:?}: as at offset 0"
                );
                if alignment > K {
                    assert!(
                        !unlaid,
                        "{options:?} {plane:?} {usage:?}: off the alignment without the option"
                    );
                }
                aliased += u64::from(at_k);
                if usage == TextureUsage::SHADER_READ {
                    let want: Vec<u8> = (plane.at..plane.at + plane.row * plane.size[1])
                        .map(surface_byte)
                        .collect();
                    let got = blitted(&device, &texture, plane);
                    assert!(
                        got == want,
                        "{options:?} {plane:?}: the texels at the plane's offset"
                    );
                }
            }
        }
    }
    assert!(
        aliased > 0,
        "no plane aliases its buffer on {}",
        device.name()
    );
}

#[test]
fn aligned_byte_is_refused_over_the_callers_memory() {
    let scratch = Scratch::new("aligned-byte-no-copy");
    let device = device(&scratch.0);
    let host = Host::new();
    // SAFETY: the memory outlives the call, which refuses it before any import.
    let memory = unsafe { host.memory() };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: None,
    };
    let descriptor =
        BufferDescriptor::new(1 << 16, ResourceOptions::shared(), contents).with_aligned_byte(K);
    let made = device.new_buffer_with_descriptor(descriptor);
    assert!(
        matches!(
            made,
            Err(Error::Invalid(Invalid {
                problem: Problem::AlignedByteNoCopy,
                ..
            }))
        ),
        "the option over the caller's memory: {:?}",
        made.err()
    );
}
