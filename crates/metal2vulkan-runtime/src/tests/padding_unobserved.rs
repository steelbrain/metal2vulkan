//! A buffer of the runtime's own memory whose maker declares the bytes between a texture's rows
//! unobserved (`BufferDescriptor::padding_unobserved`):
//!
//! - **Declared.** A `BGRA8Unorm` texture of 101x48 texels (404 bytes a row) over it at pitches
//!   past its rows (416, 448 and 512 bytes, the next multiples of 16, 64 and 256 that drivers lay
//!   linear rows at: MoltenVK, llvmpipe and ANV, RADV and NVIDIA), once the device's check
//!   settled, takes a linear image aliasing the buffer wherever the device lays one at the pitch,
//!   or a `DRM_FORMAT_MOD_LINEAR` image at the pitch where its check passed for its pooled memory
//!   (`WorkCounts::pooled_pitched_images`), counted in `WorkCounts::unobserved_padding_aliases`
//!   and not in `shadowed_textures`, and answers `NotHostPages`; else a shadow, which answers its
//!   cause; at least one pitch takes the image. Through such an image, a blit of the texture into
//!   a buffer reads the texels the CPU wrote into the declared buffer's rows, at its pitch.
//! - **Rendered at a pitch the linear image misses.** A texture that renders, over such a buffer
//!   at a 64-aligned pitch whose rows end inside a 64-byte line (the guests' device caches), is an
//!   aliasing image exactly where the device makes `DRM_FORMAT_MOD_LINEAR` images and its check
//!   of its pooled memory passed for renders over padding declared unobserved
//!   (`in_place::Access::UnobservedRenders`: its rows, each to its pitch, share no line with bytes
//!   past them), else a shadow; through the image, the texels the CPU wrote read back, a pass's
//!   clear reaches the buffer's rows, and the bytes after the last row keep the CPU's.
//! - **Large, over a `Private` buffer.** A compositor's texture that renders and samples, 3840x2400
//!   at a pitch no device's own linear image lays its rows at (37 MB), over a tracked `Private`
//!   buffer, lies as its first three rows alone do at the same pitch (46,848 bytes; one row alone
//!   is no control: llvmpipe makes no image of one row at a padded pitch): a
//!   `DRM_FORMAT_MOD_LINEAR` image in place where the device makes one and its check passed,
//!   whatever the texture's bytes, else the same shadow. Each frame uploads the texels into the buffer, reads them through the texture,
//!   clears it through a pass and reads the buffer back: the texture reads the upload, the buffer
//!   holds the clear, and in place no shadow texel is copied (printed: `LARGE-POOLED`).
//! - **Made before the check answered.** Such a texture of the guests' most common cache shape
//!   (172x67 at a 704-byte pitch), made on a device whose check has not run, is a shadow made
//!   before its pooled verdict where the query answered `Pending`; a pass's clear before the
//!   check reaches the buffer's rows through it. Once the check settled, the next command buffer
//!   (a blit that reads it, committed while no other is open) moves it to the image a texture
//!   made then takes, exactly where that one is a `DRM_FORMAT_MOD_LINEAR` image at the pitch
//!   (`WorkCounts::pooled_textures_moved_in_place`, `pooled_pitched_images`), its shadow's image
//!   freed once that command buffer completed (`pooled_texture_shadows_freed_after_move`); the
//!   blit reads the clear, a later one reads it with no shadow copy, and a later clear reaches
//!   the buffer's rows, the bytes after the last row keeping the CPU's.
//! - **Undeclared.** The same texture over a buffer made without the declaration answers
//!   `Shadow(BytesBetweenRows)` and is a shadow (`shadowed_textures`), at every pitch; where the
//!   buffer lies in a block of the runtime's own pages (`memory.rs`'s `Holds::PageBuffers`), it
//!   takes the route over fresh pages instead: in place where the device's check passed for
//!   them, else a shadow it decided, as answered.
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

/// The longest the device's check may take to settle.
const CHECK_WAIT: Duration = Duration::from_secs(120);

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
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    // Its submission checks answered too, so that where the device takes blocks of its own pages,
    // every pitch's undeclared buffer lies in one (`memory.rs`'s `Holds::PageBuffers`).
    let _settled = device.settle_submit_checks(CHECK_WAIT);
    eprintln!(
        "UNOBSERVED-PADDING on {}: a held import costs submissions nothing {}, blocks of the \
         runtime's own pages taken {}",
        device.name(),
        device.fresh_pages_pay(),
        device.page_blocks_taken()
    );
    let length = PITCHES[2] * HEIGHT;
    let descriptor = texture_descriptor();
    let mut aliased = Vec::new();
    for pitch in PITCHES {
        let observed = own_buffer(&device, length, false);
        let answer = observed.texture_placement(&descriptor, 0, pitch).unwrap();
        let undeclared = (answer, observed.in_page_block());
        let (texture, before, after) = made(&device, &observed, pitch);
        if observed.in_page_block() {
            // In a block of the runtime's own pages: as over fresh pages, in place where the
            // check passed for them and the device makes the image over them at the pitch, else
            // a shadow the check or the making decided (`texture/route.rs`).
            assert!(
                matches!(
                    answer,
                    TexturePlacement::InPlace
                        | TexturePlacement::Shadow(
                            TextureShadowReason::Unchecked { .. }
                                | TextureShadowReason::CheckFailed { .. }
                                | TextureShadowReason::Refused(_)
                        )
                ),
                "undeclared in the runtime's own pages at {pitch}: the answer {answer:?}"
            );
            assert_eq!(
                texture.shadow().is_none(),
                answer == TexturePlacement::InPlace,
                "undeclared in the runtime's own pages at {pitch}: the answer is the making's"
            );
        } else {
            assert_eq!(
                answer,
                TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows),
                "undeclared at {pitch}: the answer"
            );
            assert!(
                texture.shadow().is_some() && !texture.aliases_buffer(),
                "undeclared at {pitch}: a shadow"
            );
            assert_eq!(after.shadowed_textures - before.shadowed_textures, 1);
        }
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
            answer == TexturePlacement::NotHostPages,
            aliases,
            "declared at {pitch}: the answer {answer:?} is the making's"
        );
        assert!(
            aliases || matches!(answer, TexturePlacement::Shadow(_)),
            "declared at {pitch}: a shadow's answer names its cause, not {answer:?}"
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
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
        eprintln!(
            "UNOBSERVED-PADDING on {}: {WIDTH}x{HEIGHT} BGRA8Unorm at {pitch}: undeclared {:?} \
             in {}; declared {} ({answer:?})",
            device.name(),
            undeclared.0,
            match undeclared.1 {
                true => "a block of the runtime's own pages",
                false => "a plain block",
            },
            match (aliases, pitched) {
                (true, 0) => "the device's linear image",
                (true, _) => "a DRM_FORMAT_MOD_LINEAR image at the pitch",
                (false, _) => "a shadow",
            },
        );
    }
    assert!(
        !aliased.is_empty(),
        "no pitch of {PITCHES:?} took an aliasing image on {}",
        device.name()
    );
}

/// The guests' device caches' shapes: `BGRA8Unorm`, rows at a 64-aligned pitch that end inside a
/// 64-byte line: (width, height, pitch).
const CACHES: [(u64, u64, u64); 3] = [(61, 37, 256), (250, 25, 1024), (3662, 3, 14656)];

#[test]
fn a_rendered_texture_at_a_pitch_the_linear_image_misses_takes_a_pitched_image_where_the_pooled_check_passed(
) {
    use crate::in_place::{Access, Padding, PagesClass, RowEnds, Verdict};
    let scratch = Scratch::new("padding-unobserved-pitched");
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    let queue = device.new_command_queue().unwrap();
    for (width, height, pitch) in CACHES {
        let row = width * 4;
        let length = pitch * height + 4096;
        let buffer = own_buffer(&device, length, true);
        // SAFETY: the buffer's mapped bytes, which no command uses yet.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().unwrap().as_ptr(), length as usize)
        };
        for (at, byte) in bytes.iter_mut().enumerate() {
            *byte = texel_byte(at as u64 % pitch, at as u64 / pitch);
        }
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let answer = buffer.texture_placement(&descriptor, 0, pitch).unwrap();
        let before = device.work_counts();
        let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
        let after = device.work_counts();
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
        let aliases = texture.aliases_buffer();
        let layout = crate::linear_plan::BufferLayout {
            offset: 0,
            row_pitch: pitch,
            slice_pitch: pitch * height,
            row_bytes: row,
            rows: height,
            slices: 1,
        };
        // The check the texture asks: of renders over padding declared unobserved, where its rows
        // to their pitch share no line with bytes past them, as the buffer's last row's do here.
        let line = device.shared().in_place.line();
        let start = buffer.contents().unwrap().as_ptr().addr() as u64;
        let rows = RowEnds::of(start, &layout, line);
        let padding = Padding::of(start, &layout, line, true).declared(true);
        assert_eq!(
            (rows, padding),
            (RowEnds::InLines, Padding::Unobserved),
            "{width}x{height} at {pitch}"
        );
        let verdicts = [Access::Renders, Access::UnobservedRenders].map(|access| {
            device
                .shared()
                .in_place
                .verdict(PagesClass::Pooled, access, rows)
        });
        let passed = verdicts[1] == Verdict::Passed;
        assert_eq!(
            answer == TexturePlacement::NotHostPages,
            aliases,
            "{width}x{height} at {pitch}: the answer {answer:?} is the making's"
        );
        assert!(
            pitched == 0 || (aliases && passed),
            "{width}x{height} at {pitch}: a pitched image only where the pooled check passed \
             ({verdicts:?})"
        );
        // Through the image (or the shadow), the CPU's texels.
        let read = device
            .new_buffer(row * height, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &read,
            0,
            row,
            row * height,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        // SAFETY: the read buffer's mapped bytes, which no command uses now.
        let read = unsafe {
            std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), (row * height) as usize)
        };
        for y in 0..height {
            for x in 0..row {
                assert_eq!(
                    read[(y * row + x) as usize],
                    texel_byte(x, y),
                    "{width}x{height} at {pitch}: byte {x} of row {y} read through the texture"
                );
            }
        }
        // A pass's clear reaches the buffer's rows; the bytes after the last row keep the CPU's.
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = crate::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = crate::LoadAction::Clear;
        color.clear_color = crate::ClearColor::new(0.0, 0.0, 1.0, 1.0);
        color.store_action = crate::StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        for y in 0..height {
            for x in 0..width {
                let at = (y * pitch + x * 4) as usize;
                assert_eq!(
                    bytes[at..at + 4],
                    [0xff, 0, 0, 0xff],
                    "{width}x{height} at {pitch}: texel {x} of row {y} after the clear"
                );
            }
        }
        for at in pitch * height..length {
            assert_eq!(
                bytes[at as usize],
                texel_byte(at % pitch, at / pitch),
                "{width}x{height} at {pitch}: byte {at} after the last row"
            );
        }
        eprintln!(
            "POOLED-PITCHED on {}: {width}x{height} BGRA8Unorm at {pitch} rendered: pooled \
             verdicts {verdicts:?} ({rows:?}); answer {answer:?}; {}",
            device.name(),
            match (aliases, pitched) {
                (true, 0) => "the device's linear image",
                (true, _) => "a DRM_FORMAT_MOD_LINEAR image at the pitch",
                (false, _) => "a shadow",
            },
        );
    }
}

/// A compositor's texture: 3840x2400 `BGRA8Unorm` at rows of 15,616 bytes (256 past its 15,360
/// bytes of texels, a multiple of 256, at which no device's own linear image lays them): 37,478,400
/// bytes to their pitch.
const LARGE: (u64, u64, u64) = (3840, 2400, 15_616);

#[test]
fn a_large_texture_over_a_private_buffer_lies_as_its_first_rows_do_and_reads_right() {
    let scratch = Scratch::new("padding-unobserved-large");
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    let (width, height, pitch) = LARGE;
    let (row, length) = (width * 4, pitch * height);
    let private = |length| {
        let descriptor =
            BufferDescriptor::new(length, ResourceOptions::private(), BufferContents::Zero)
                .with_padding_unobserved();
        device.new_buffer_with_descriptor(descriptor).unwrap()
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    // The control: the texture's first three rows alone over a buffer of their own, at the same
    // pitch.
    let mut first_rows = descriptor;
    first_rows.height = 3;
    let control = private(3 * pitch).texture_placement(&first_rows, 0, pitch);
    let control = control.unwrap();
    let buffer = private(length);
    let answer = buffer.texture_placement(&descriptor, 0, pitch).unwrap();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
    let made = device.work_counts();
    let pitched = made.pooled_pitched_images - before.pooled_pitched_images;
    let shadows = made.shadowed_textures - before.shadowed_textures;
    // The CPU's texels at the pitch, in a `Shared` buffer each frame uploads into the texture's.
    let staging = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    // SAFETY: the staging buffer's mapped bytes, which no command uses yet.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(staging.contents().unwrap().as_ptr(), length as usize)
    };
    for (at, byte) in bytes.iter_mut().enumerate() {
        *byte = texel_byte(at as u64 % pitch, at as u64 / pitch);
    }
    let bytes: &[u8] = bytes;
    let read = device
        .new_buffer(row * height, ResourceOptions::shared())
        .unwrap();
    let held = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    // A frame, one command buffer of four encoders: the texels uploaded into the texture's buffer,
    // read through the texture, a pass's clear through the texture, and the buffer's bytes read
    // back. Its shadow texels copied in and back.
    let frame = || {
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, length)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &read,
            0,
            row,
            row * height,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        let mut pass = crate::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = crate::LoadAction::Clear;
        color.clear_color = crate::ClearColor::new(0.0, 0.0, 1.0, 1.0);
        color.store_action = crate::StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 0, &held, 0, length)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        // SAFETY: the read buffers' mapped bytes, which no command uses now.
        let (read, held) = unsafe {
            (
                std::slice::from_raw_parts(
                    read.contents().unwrap().as_ptr(),
                    (row * height) as usize,
                ),
                std::slice::from_raw_parts(held.contents().unwrap().as_ptr(), length as usize),
            )
        };
        for y in 0..height {
            let (at, to) = ((y * row) as usize, (y * pitch) as usize);
            assert!(
                read[at..at + row as usize] == bytes[to..to + row as usize],
                "row {y} read through the texture holds the bytes uploaded"
            );
        }
        assert!(
            rows_hold(held, LARGE, [0xff, 0, 0, 0xff]),
            "the clear through the texture reached the buffer's rows"
        );
        let after = device.work_counts();
        (
            after.shadow_texels_copied_in - before.shadow_texels_copied_in,
            after.shadow_texels_copied_back - before.shadow_texels_copied_back,
        )
    };
    let first = frame();
    let next = frame();
    eprintln!(
        "LARGE-POOLED on {}: {width}x{height} BGRA8Unorm at {pitch} ({length} bytes) over a \
         Private buffer: its first three rows answer {control:?}, the texture {answer:?}; pitched \
         {pitched}, shadows {shadows}; shadow texels copied in and back, the first frame \
         {first:?}, the next {next:?}",
        device.name(),
    );
    assert_eq!(
        answer, control,
        "the texture lies as its first three rows do, whatever its bytes"
    );
    match answer {
        TexturePlacement::NotHostPages => {
            assert!(
                texture.aliases_buffer() && texture.shadow().is_none(),
                "in place"
            );
            assert_eq!(shadows, 0, "counted as no shadow");
            assert_eq!((first, next), ((0, 0), (0, 0)), "no shadow texel copied");
        }
        _ => {
            assert!(
                texture.shadow().is_some(),
                "a shadow where the device makes no image"
            );
            assert_eq!(shadows, 1, "counted as a shadow");
        }
    }
}

/// The guests' most common device cache's shape, which the Iris Xe's linear image lays out
/// tight (688 bytes of texels a row) and so takes an image at its pitch: (width, height, pitch).
const MOVED: (u64, u64, u64) = (172, 67, 704);

/// A pass that clears `texture` to `color` and stores it, committed on `queue`, completed.
fn cleared(queue: &crate::CommandQueue, texture: &Texture, color: crate::ClearColor) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = crate::RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = crate::LoadAction::Clear;
    attachment.clear_color = color;
    attachment.store_action = crate::StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Whether every texel of `bytes`' rows (`width` by `height` at `pitch`) is `texel`.
fn rows_hold(bytes: &[u8], (width, height, pitch): (u64, u64, u64), texel: [u8; 4]) -> bool {
    (0..height).all(|y| {
        let at = (y * pitch) as usize;
        bytes[at..at + (width * 4) as usize]
            .chunks(4)
            .all(|t| t == texel)
    })
}

#[test]
fn a_rendered_texture_made_before_the_pooled_check_moves_to_its_pitched_image_once_it_passes() {
    let scratch = Scratch::new("padding-unobserved-move");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let (width, height, pitch) = MOVED;
    let row = width * 4;
    let length = pitch * height + 4096;
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let buffer = own_buffer(&device, length, true);
    // SAFETY: the buffer's mapped bytes, which no command uses while they are written or read
    // here: each command buffer below completed before.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().unwrap().as_ptr(), length as usize)
    };
    for (at, byte) in bytes.iter_mut().enumerate() {
        *byte = texel_byte(at as u64 % pitch, at as u64 / pitch);
    }
    let past = bytes[(pitch * height) as usize..].to_vec();
    // The query starts the check, which has not run on this device: `Pending` wherever the
    // pooled class is asked.
    let answer = buffer.texture_placement(&descriptor, 0, pitch).unwrap();
    let before = device.work_counts();
    let early = buffer.new_texture(&descriptor, 0, pitch).unwrap();
    let made = device.work_counts();
    let shadow = early.shadow().is_some();
    let pending = answer == TexturePlacement::Pending && shadow;
    assert_eq!(
        made.pooled_pitched_images, before.pooled_pitched_images,
        "made before the check: no image at the pitch"
    );
    // A pass's clear before the check reaches the buffer's rows.
    let blue = crate::ClearColor::new(0.0, 0.0, 1.0, 1.0);
    cleared(&queue, &early, blue);
    let blue = [0xff, 0, 0, 0xff];
    assert!(
        rows_hold(bytes, MOVED, blue),
        "the clear before the check is in the rows"
    );
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    // A texture made now of the same shape, over another buffer: the image the first moves to.
    let other = own_buffer(&device, length, true);
    let late_before = device.work_counts();
    let late = other.new_texture(&descriptor, 0, pitch).unwrap();
    let late_after = device.work_counts();
    let late_pitched = late.aliases_buffer()
        && late_after.pooled_pitched_images - late_before.pooled_pitched_images == 1;
    // The next command buffer, committed while no other is open, takes the move.
    let read = |texture: &Texture| -> Vec<u8> {
        let out = device
            .new_buffer(row * height, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, height, 1),
            &out,
            0,
            row,
            row * height,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        // SAFETY: the read buffer's mapped bytes, which no command uses now.
        unsafe {
            std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), (row * height) as usize)
        }
        .to_vec()
    };
    let at_move = read(&early);
    let after = device.work_counts();
    let moved = after.pooled_textures_moved_in_place - before.pooled_textures_moved_in_place;
    let counted = after.pooled_pitched_images - late_after.pooled_pitched_images;
    let expected = u64::from(pending && late_pitched);
    eprintln!(
        "POOLED-MOVE on {}: {width}x{height} BGRA8Unorm at {pitch} rendered: answered {answer:?} \
         before the check, a shadow {shadow}; one made after it {}; after the next commit: \
         moved {moved}, counted at its pitch {counted}",
        device.name(),
        match late_pitched {
            true => "a DRM_FORMAT_MOD_LINEAR image at the pitch",
            false => "no image at the pitch",
        },
    );
    assert_eq!(
        (moved, counted),
        (expected, expected),
        "made before the check ({answer:?}): it moves exactly where one made after takes the \
         image at the pitch ({late_pitched})"
    );
    // Moved, an image over the buffer; else as it was made.
    let aliases = match pending {
        true => expected == 1,
        false => !shadow,
    };
    assert_eq!(
        early.aliases_buffer(),
        aliases,
        "after the next commit: an image over the buffer"
    );
    assert!(
        rows_hold(&at_move, (width, height, row), blue),
        "the read at the move reads the clear"
    );
    // Its shadow's image goes once the command buffer that took the move completed, on the
    // thread that lets that command buffer go.
    let started = std::time::Instant::now();
    let freed = loop {
        let counts = device.work_counts();
        let freed = counts.pooled_texture_shadows_freed_after_move
            - before.pooled_texture_shadows_freed_after_move;
        if freed >= moved || started.elapsed() > WAIT {
            break freed;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(freed, moved, "a moved texture's shadow image goes");
    // After the move: a read with no shadow copy, and a clear in the buffer's rows.
    let before = device.work_counts();
    let later = read(&early);
    let copies = device.work_counts().shadow_copies - before.shadow_copies;
    assert!(
        rows_hold(&later, (width, height, row), blue),
        "a later read reads the clear"
    );
    assert!(
        moved == 0 || copies == 0,
        "in place, a read took {copies} shadow copies"
    );
    let green = crate::ClearColor::new(0.0, 1.0, 0.0, 1.0);
    cleared(&queue, &early, green);
    assert!(
        rows_hold(bytes, MOVED, [0, 0xff, 0, 0xff]),
        "a later clear is in the rows"
    );
    assert_eq!(
        bytes[(pitch * height) as usize..],
        past[..],
        "the bytes after the last row keep the CPU's"
    );
    drop((early, late));
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
