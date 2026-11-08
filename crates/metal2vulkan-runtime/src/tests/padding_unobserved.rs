//! A buffer of the runtime's own memory whose maker declares the bytes between a texture's rows
//! unobserved (`BufferDescriptor::padding_unobserved`):
//!
//! - **Declared.** A `BGRA8Unorm` texture of 101x48 texels (404 bytes a row) over it at pitches
//!   past its rows (416, 448 and 512 bytes, the next multiples of 16, 64 and 256 that drivers lay
//!   linear rows at: MoltenVK, llvmpipe and ANV, RADV and NVIDIA) answers `NotHostPages`, and its
//!   making takes a linear image aliasing the buffer wherever the device lays one at the pitch,
//!   counted in `WorkCounts::unobserved_padding_aliases` and not in `shadowed_textures`, else a
//!   shadow; at least one pitch takes the image. Through such an image, a blit of the texture into
//!   a buffer reads the texels the CPU wrote into the declared buffer's rows, at its pitch.
//! - **Undeclared.** The same texture over a buffer made without the declaration answers
//!   `Shadow(BytesBetweenRows)` and is a shadow (`shadowed_textures`), at every pitch.
//! - **Over the caller's memory**, the declaration is refused: `Invalid`
//!   (`PaddingUnobservedNoCopy`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`UNOBSERVED-PADDING`): each
//! pitch's answers and paths.

use super::blit_holds::{device, Host, Scratch};
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, Device, Error, Invalid, Origin, Problem,
    ResourceOptions, Size, Texture, TextureDescriptor, TexturePlacement, TextureShadowReason,
    WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::Duration;

/// The texture's texels a row, and its rows.
const WIDTH: u64 = 101;
const HEIGHT: u64 = 48;

/// The pitches past its 404-byte rows the texture is made at.
const PITCHES: [u64; 3] = [416, 448, 512];

/// The longest a command buffer may take here.
const WAIT: Duration = Duration::from_secs(20);

/// A buffer of the device's own memory of `length` bytes, declaring its padding unobserved where
/// `declared`.
fn own_buffer(device: &Device, length: u64, declared: bool) -> Buffer {
    let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero);
    let descriptor = match declared {
        true => descriptor.with_padding_unobserved(),
        false => descriptor,
    };
    device.new_buffer_with_descriptor(descriptor).unwrap()
}

/// The texture's descriptor: `ShaderRead`, `Shared`.
fn texture_descriptor() -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    descriptor
}

/// The texture `buffer` makes at `pitch`, the counts before and after its making.
fn made(device: &Device, buffer: &Buffer, pitch: u64) -> (Texture, WorkCounts, WorkCounts) {
    let before = device.work_counts();
    let texture = buffer.new_texture(&texture_descriptor(), 0, pitch).unwrap();
    (texture, before, device.work_counts())
}

/// Byte `x` of row `y` the CPU writes into the texels.
fn texel_byte(x: u64, y: u64) -> u8 {
    (x * 7 + y * 31 + 3) as u8
}

/// The texels of `texture` as a blit copies them into a buffer of rows one after another.
fn blitted(device: &Device, texture: &Texture) -> Vec<u8> {
    let row = WIDTH * 4;
    let read = device
        .new_buffer(row * HEIGHT, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(WIDTH, HEIGHT, 1),
        &read,
        0,
        row,
        row * HEIGHT,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    // SAFETY: the read buffer's mapped bytes, which no command uses now.
    unsafe {
        std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), (row * HEIGHT) as usize)
    }
    .to_vec()
}

#[test]
fn a_texture_over_unobserved_padding_aliases_its_buffer_and_over_observed_padding_is_a_shadow() {
    let scratch = Scratch::new("padding-unobserved");
    let device = device(&scratch.0);
    let length = PITCHES[2] * HEIGHT;
    let descriptor = texture_descriptor();
    let mut aliased = Vec::new();
    for pitch in PITCHES {
        let observed = own_buffer(&device, length, false);
        let answer = observed.texture_placement(&descriptor, 0, pitch).unwrap();
        assert_eq!(
            answer,
            TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows),
            "undeclared at {pitch}: the answer"
        );
        let (texture, before, after) = made(&device, &observed, pitch);
        assert!(
            texture.shadow().is_some() && !texture.aliases_buffer(),
            "undeclared at {pitch}: a shadow"
        );
        assert_eq!(after.shadowed_textures - before.shadowed_textures, 1);
        assert_eq!(
            after.unobserved_padding_aliases,
            before.unobserved_padding_aliases
        );

        let declared = own_buffer(&device, length, true);
        assert!(
            !declared.in_fresh_pages(),
            "a short buffer lies in no fresh pages"
        );
        let answer = declared.texture_placement(&descriptor, 0, pitch).unwrap();
        assert_eq!(
            answer,
            TexturePlacement::NotHostPages,
            "declared at {pitch}: the answer"
        );
        // SAFETY: the declared buffer's mapped bytes, which no command uses yet.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(declared.contents().unwrap().as_ptr(), length as usize)
        };
        for y in 0..HEIGHT {
            for x in 0..WIDTH * 4 {
                bytes[(y * pitch + x) as usize] = texel_byte(x, y);
            }
        }
        let (texture, before, after) = made(&device, &declared, pitch);
        let aliases = texture.aliases_buffer();
        assert_eq!(
            aliases,
            texture.shadow().is_none(),
            "declared at {pitch}: an aliasing image or a shadow"
        );
        assert_eq!(
            after.unobserved_padding_aliases - before.unobserved_padding_aliases,
            u64::from(aliases),
            "declared at {pitch}: counted as an aliasing image over unobserved padding"
        );
        assert_eq!(
            after.shadowed_textures - before.shadowed_textures,
            u64::from(!aliases),
            "declared at {pitch}: counted as a shadow"
        );
        let read = blitted(&device, &texture);
        for y in 0..HEIGHT {
            for x in 0..WIDTH * 4 {
                assert_eq!(
                    read[(y * WIDTH * 4 + x) as usize],
                    texel_byte(x, y),
                    "declared at {pitch}: byte {x} of row {y}"
                );
            }
        }
        if aliases {
            aliased.push(pitch);
        }
        eprintln!(
            "UNOBSERVED-PADDING on {}: {WIDTH}x{HEIGHT} BGRA8Unorm at {pitch}: undeclared a \
             shadow ({:?}); declared {} ({:?})",
            device.name(),
            TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows),
            match aliases {
                true => "an aliasing image",
                false => "a shadow (the device lays no linear image at the pitch)",
            },
            TexturePlacement::NotHostPages,
        );
    }
    assert!(
        !aliased.is_empty(),
        "no pitch of {PITCHES:?} took an aliasing image on {}",
        device.name()
    );
}

#[test]
fn padding_unobserved_is_refused_over_the_callers_memory() {
    let scratch = Scratch::new("padding-unobserved-no-copy");
    let device = device(&scratch.0);
    let host = Host::new();
    // SAFETY: the memory outlives the call, which refuses it before any import.
    let memory = unsafe { host.memory() };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: None,
        keep_alive: None,
    };
    let descriptor = BufferDescriptor::new(1 << 16, ResourceOptions::shared(), contents)
        .with_padding_unobserved();
    let made = device.new_buffer_with_descriptor(descriptor);
    assert!(
        matches!(
            made,
            Err(Error::Invalid(Invalid {
                problem: Problem::PaddingUnobservedNoCopy,
                ..
            }))
        ),
        "the declaration over the caller's memory: {:?}",
        made.err()
    );
}
