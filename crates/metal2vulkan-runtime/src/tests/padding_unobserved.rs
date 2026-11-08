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
//! - **Past the pooled bound** (`in_place::POOLED_PITCHED_BYTES`). A texture whose rows to their
//!   pitch take more bytes, that renders or only samples, over a tracked `Private` buffer takes no
//!   pitched image and, where one inside the bound took one, answers `Shadow(LinearCost)`; over a
//!   `Shared` buffer it never answers `Shadow(LinearCost)`, and takes the pitched image where one
//!   inside the bound took one and the device's check of its access over its pooled memory
//!   passed.
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
            "UNOBSERVED-PADDING on {}: {WIDTH}x{HEIGHT} BGRA8Unorm at {pitch}: undeclared a \
             shadow ({:?}); declared {} ({answer:?})",
            device.name(),
            TexturePlacement::Shadow(TextureShadowReason::BytesBetweenRows),
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
/// 64-byte line, each inside the pooled bound (`in_place::POOLED_PITCHED_BYTES`): (width, height,
/// pitch).
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

/// The guests' most common device cache, 172x67 `BGRA8Unorm` at 704-byte rows (47,168 bytes),
/// inside the pooled bound, and one of 164x82 at the same pitch (57,728 bytes), past it.
const BOUNDED: [(u64, u64, u64); 2] = [(172, 67, 704), (164, 82, 704)];

#[test]
fn a_texture_past_the_pooled_bound_is_a_shadow_of_its_linear_cost_over_private_memory_alone() {
    use crate::in_place::{Access, Padding, PagesClass, RowEnds, Verdict, POOLED_PITCHED_BYTES};
    let scratch = Scratch::new("padding-unobserved-bound");
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    let [inside, (width, height, pitch)] = BOUNDED;
    let bytes = pitch * height;
    assert!(inside.2 * inside.1 <= POOLED_PITCHED_BYTES && bytes > POOLED_PITCHED_BYTES);
    // The texture of `usage` made over a new buffer of `options` that declares its padding
    // unobserved: the buffer, its answer, and whether it took a pitched image and was counted a
    // shadow.
    let made = |(width, height, pitch): (u64, u64, u64), options, usage| {
        let descriptor = BufferDescriptor::new(pitch * height, options, BufferContents::Zero)
            .with_padding_unobserved();
        let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        descriptor.usage = usage;
        descriptor.resource_options = buffer.resource_options();
        let answer = buffer.texture_placement(&descriptor, 0, pitch).unwrap();
        let before = device.work_counts();
        let texture = buffer.new_texture(&descriptor, 0, pitch).unwrap();
        let after = device.work_counts();
        let pitched = after.pooled_pitched_images - before.pooled_pitched_images == 1;
        let shadow = after.shadowed_textures - before.shadowed_textures == 1;
        eprintln!(
            "POOLED-BOUND on {}: {width}x{height} BGRA8Unorm at {pitch} ({} bytes, the bound \
             {POOLED_PITCHED_BYTES}) of usage {usage:?} over a {:?} buffer: answer {answer:?}; \
             pitched {pitched}, shadow {shadow}, aliases {}",
            device.name(),
            pitch * height,
            options.storage_mode(),
            texture.aliases_buffer(),
        );
        (buffer, answer, pitched, shadow)
    };
    let rendered = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    // Whether the device takes a pitched image for the cache inside the bound.
    let (_, _, takes, _) = made(inside, ResourceOptions::shared(), rendered);
    let linear_cost = TexturePlacement::Shadow(TextureShadowReason::LinearCost { bytes });
    for usage in [rendered, TextureUsage::SHADER_READ] {
        // Over a `Private` buffer, tracked, past the bound: a shadow of its linear cost where the
        // device makes such images.
        let (_, answer, pitched, shadow) =
            made((width, height, pitch), ResourceOptions::private(), usage);
        assert!(
            !pitched,
            "{usage:?} over the Private buffer: no pitched image"
        );
        if takes {
            assert_eq!(answer, linear_cost, "{usage:?} over the Private buffer");
            assert!(
                shadow,
                "{usage:?} over the Private buffer: counted as a shadow"
            );
        }
        // Over a `Shared` buffer, which the CPU writes: never its linear cost, and the pitched
        // image where the device takes one and its check of the texture's access over its pooled
        // memory, of rows that lie as its own, passed.
        let (buffer, answer, pitched, _) =
            made((width, height, pitch), ResourceOptions::shared(), usage);
        assert_ne!(answer, linear_cost, "{usage:?} over the Shared buffer");
        let layout = crate::linear_plan::BufferLayout {
            offset: 0,
            row_pitch: pitch,
            slice_pitch: pitch * height,
            row_bytes: width * 4,
            rows: height,
            slices: 1,
        };
        let line = device.shared().in_place.line();
        let start = buffer.contents().unwrap().as_ptr().addr() as u64;
        let rows = RowEnds::of(start, &layout, line);
        let access = match usage.contains(TextureUsage::RENDER_TARGET) {
            true => Access::rendered(Padding::of(start, &layout, line, true).declared(true)),
            false => Access::Reads,
        };
        let verdict = device
            .shared()
            .in_place
            .verdict(PagesClass::Pooled, access, rows);
        eprintln!("POOLED-BOUND: {usage:?} over the Shared buffer asks {access:?} ({rows:?}): {verdict:?}");
        if takes && verdict == Verdict::Passed {
            assert!(pitched, "{usage:?} over the Shared buffer: a pitched image");
            assert_eq!(
                answer,
                TexturePlacement::NotHostPages,
                "{usage:?} over the Shared buffer"
            );
        }
    }
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
