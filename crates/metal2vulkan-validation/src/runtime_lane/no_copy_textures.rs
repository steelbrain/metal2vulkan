//! Device tests of 2D textures (and 2D arrays) over no-copy buffers on the device the cfg selects.
//! A texture over the caller's host pages is a shadow image the encoders copy the texels into and
//! back (`shadowed-texture`), or under `HostPageTexturePolicy::InPlace` an image aliasing the
//! pages where the device's check passed for each of its accesses (one that reads, renders, or a
//! shader writes of texels of 4, 8 or 16 bytes): either reads and writes the caller's bytes as
//! Metal's texture over the buffer does.
//!
//! - [`textures_over_no_copy_buffers_read_and_write_the_caller_s_pages`], under the validation
//!   layer: the consumer's surface formats (and `BGR10_XR`, an extended-range format held in an
//!   integer image) at 1920x1080 with packed rows, and with rows padded by 64 bytes as a guest
//!   browser's surfaces have them (the `_padded` tests), as render targets the shaders read (and
//!   write), and as textures the shaders only read (and write): the bytes the caller wrote before
//!   the texture was made are what a blit reads out of it; a render pass's clear (a blit's write,
//!   without render target usage) lands in the pages, as the next blit reads the texture;
//!   the caller's later writes are what the next blit reads; and a blit's write through the texture
//!   read through the buffer by a blit, and the other way round, in one blit encoder and in two of
//!   one command buffer, read the write (the texture and its buffer are one resource to hazard
//!   tracking, which on MoltenVK leaves blits to Metal's tracking). Padded rows' bytes between
//!   the rows are never written. Each runs under both settings of `host_page_textures`: under
//!   `Shadow` every texture is a shadow, which `vk_image` answers no image for, counted
//!   (`WorkCounts::shadowed_textures`) and made without zeroing its image (`shadows_zeroed`,
//!   `images_started_on_the_host`: its copy in writes every texel first), and making it, and
//!   making it again, asks the driver no format's modifier features (`modifier_feature_queries`);
//!   under `InPlace`, once the device's check answered, each is in place (`vk_image` answers its
//!   image, `host_textures_in_place`) or a shadow counted by why, but one with `ShaderWrite` usage
//!   of 1- or 2-byte texels, which no check covers, a shadow. The second making takes the same
//!   form. Each texture's form is printed (`NOCOPY-TEXTURE`).
//! - [`two_slice_arrays_over_no_copy_buffers_read_and_write_the_caller_s_pages`], under the
//!   validation layer: two-slice `RGBA8Unorm` 2D arrays made by the linear call, a 4x4 sampled one
//!   and an 8x8 render target, read and written a slice at a time at Metal's slice places
//!   (`NOCOPY-ARRAY`: the form, and a shadow's reason).
//! - [`passes_blend_load_and_store_as_private_textures`], under the validation layer, one process
//!   per format and row pitch (packed, and padded by 64 bytes): render passes over the texture that
//!   load, blend and store, and clear and blend, match a private texture of the same bytes byte
//!   for byte, read through the texture and through the buffer, the bytes between padded rows
//!   untouched (`NOCOPY-BLEND`). But a clear's channel, and a blend over it, may stand a code
//!   from the private texture's on a device whose clear is not Metal's conversion, as Vulkan
//!   leaves a clear's rounding to the device, which may round it by the image
//!   ([`clear_codes_apart`]), where each image's clear announces `device-color-conversion`.
//! - [`surface_frame_costs`], without the validation layer, report only (built in release for the
//!   numbers): the consumer's display surface, a 1920x1080 `RGBA16Float` or `BGRA8Unorm` texture
//!   of an IOSurface plane over a no-copy buffer of the surface's pages, written each frame by a
//!   render pass (a clear, or a load, each with or without a blended rectangle over its left
//!   half) and read by a blit into a private texture, through the same texture or a second one
//!   over a second no-copy buffer of the same pages (the host display's); and each step alone
//!   (`None` for a pass or blit left out). Printed per case (`SURFACE-COST`): the form, the median
//!   and fastest GPU time of a frame's command buffer and of the whole frame on the host (encoding
//!   to completion), and what the device announced.

use super::degrade_notices::{clear_conversions, counts};
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::{
    layer_reported_nothing_at_all, validated_device, validated_device_with, Messages, ScratchCache,
    WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, HostPageTexturePolicy};
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, IoSurfacePlane, LoadAction,
    Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::path::Path;
use std::time::Instant;

/// The consumer's display surface's size.
const W: u64 = 1920;
const H: u64 = 1080;

/// A no-copy buffer of `len` bytes over `mapping`, of default options as the consumer makes its
/// surfaces' buffers; `None`, printed, where the device imports no host memory.
fn over(device: &Device, mapping: &Mapping, len: usize) -> Option<Buffer> {
    // SAFETY: each test drops the mapping after every buffer over it.
    let made = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("NOCOPY-TEXTURE: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// Whether `texture`'s texels are its own linear image's: an aliasing image, not a shadow.
fn aliased(texture: &Texture) -> bool {
    // SAFETY: the handle is only compared, while the texture lives.
    unsafe { texture.vk_image() }.is_some()
}

fn form(aliased: bool) -> &'static str {
    if aliased {
        "aliased"
    } else {
        "shadowed"
    }
}

fn whole() -> (Origin, Size) {
    (Origin::new(0, 0, 0), Size::new(W, H, 1))
}

/// `texture`'s texels, rows `bpr` bytes apart, blitted out into a new `Shared` buffer.
fn blit_out(queue: &CommandQueue, texture: &Texture, bpr: u64) -> Vec<u8> {
    let len = bpr * H;
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let (origin, size) = whole();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        origin,
        size,
        &out,
        0,
        bpr,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of `len` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
}

/// A render pass that clears `texture` to `color` and stores it, in a command buffer of its own
/// that completes.
fn clear(queue: &CommandQueue, texture: &Texture, color: [f64; 4]) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(color[0], color[1], color[2], color[3]);
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A test of each of the consumer's surface formats, with its texels' bytes and the bytes its rows
/// are padded by, each in a process of its own: `$test(pf, texel, pad)` as `$name`.
macro_rules! per_format {
    ($test:ident: $($name:ident => $pf:ident, $texel:expr, $pad:expr;)*) => {
        $(
            #[test]
            fn $name() {
                $test(PixelFormat::$pf, $texel, $pad);
            }
        )*
    };
}

/// The texels of `bytes` laid out in rows `bpr` apart, each row's `row` bytes of texels, of the
/// [`H`] rows: what a texture over them reads and writes.
fn texel_rows(bytes: &[u8], bpr: u64, row: u64) -> Vec<u8> {
    bytes
        .chunks(bpr as usize)
        .take(H as usize)
        .flat_map(|r| &r[..row as usize])
        .copied()
        .collect()
}

/// The bytes of `bytes` between rows `bpr` apart of `row` bytes of texels, of the [`H`] rows: what
/// no texture over them writes.
fn padding(bytes: &[u8], bpr: u64, row: u64) -> Vec<u8> {
    bytes
        .chunks(bpr as usize)
        .take(H as usize)
        .flat_map(|r| &r[(row as usize).min(r.len())..])
        .copied()
        .collect()
}

/// A `Shared` buffer holding `bytes`.
pub(super) fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: a new `Shared` buffer of `bytes.len()` bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// The bytes of a `Shared` buffer, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A write of `texture` (rows `bpr` bytes apart, `row` bytes of texels each) through a blit, then
/// a read of `under`, the no-copy buffer it lies over, by a blit, and the other way round (a write
/// of `under`, then a read of `texture`), each pair in one blit encoder when `one_encoder`, else in
/// two encoders of one command buffer: each read sees its write's texels, as one resource's
/// accesses are ordered.
fn ordered(
    queue: &CommandQueue,
    texture: &Texture,
    under: &Buffer,
    (bpr, row): (u64, u64),
    one_encoder: bool,
) {
    let device = queue.device();
    let n = bpr * H;
    let (origin, size) = whole();
    for (round, texture_first) in [(5usize, true), (9, false)] {
        let pattern = counting(n as usize, round);
        let source = staged(device, &pattern);
        let out = device.new_buffer(n, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        if texture_first {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                bpr,
                n,
                size,
                texture,
                0,
                0,
                origin,
                BlitOption::None,
            )
            .unwrap();
        } else {
            blit.copy_from_buffer_to_buffer(&source, 0, under, 0, n)
                .unwrap();
        }
        if !one_encoder {
            blit.end_encoding().unwrap();
            blit = command_buffer.blit_command_encoder().unwrap();
        }
        if texture_first {
            blit.copy_from_buffer_to_buffer(under, 0, &out, 0, n)
                .unwrap();
        } else {
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                0,
                origin,
                size,
                &out,
                0,
                bpr,
                n,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        assert!(
            texel_rows(&bytes_of(&out), bpr, row) == texel_rows(&pattern, bpr, row),
            "a {} write read through the {} in {}: stale bytes",
            if texture_first { "texture" } else { "buffer" },
            if texture_first { "buffer" } else { "texture" },
            if one_encoder {
                "one encoder"
            } else {
                "two encoders"
            },
        );
    }
}

/// A texture over a no-copy buffer, its rows padded by `pad` bytes, reads the caller's bytes and
/// writes into them as Metal's does, aliased or shadowed (the module docs), as a render target or
/// not, never the bytes between its rows; a write through it read through the buffer, and the
/// other way round, in one blit encoder and in two, each see the write. A shadow is made without
/// zeroing its image; on ANV a padded texture is an image at Metal's row pitch. The form of each
/// is printed. One test a format and pitch, each its own process.
fn read_and_write_the_caller_s_pages(pf: PixelFormat, texel: u64, pad: u64) {
    for policy in [
        HostPageTexturePolicy::Shadow,
        HostPageTexturePolicy::InPlace,
    ] {
        read_and_write_the_caller_s_pages_under(pf, texel, pad, policy);
    }
}

/// [`read_and_write_the_caller_s_pages`] on a device of `policy`.
fn read_and_write_the_caller_s_pages_under(
    pf: PixelFormat,
    texel: u64,
    pad: u64,
    policy: HostPageTexturePolicy,
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        if policy == HostPageTexturePolicy::InPlace {
            assert!(device.settle_in_place_checks(CHECK_WAIT));
        }
        let queue = device.new_command_queue().unwrap();
        let usages = [
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ];
        for usage in usages {
            let row = W * texel;
            let bpr = row + pad;
            let rows = |bytes: &[u8]| texel_rows(bytes, bpr, row);
            let n = (bpr * H) as usize;
            let len = n.next_multiple_of(GUEST_PAGE);
            let mapping = Mapping::anonymous(len);
            mapping.write(0, &counting(len, 7));
            let Some(buffer) = over(&device, &mapping, len) else {
                return;
            };
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let before = device.work_counts();
            let texture = buffer.new_texture(&d, 0, bpr).unwrap();
            let counted = device.work_counts();
            let made = aliased(&texture);
            let what = format!(
                "{policy:?}, {} usage {usage:?}, rows of {bpr} bytes",
                pf.name()
            );
            let shadowed = counted.shadowed_textures - before.shadowed_textures;
            let in_place = counted.host_textures_in_place - before.host_textures_in_place;
            // Under `Shadow`, and for a texture that writes through a shader whose texels are of
            // a width the device's check stores none of (1 or 2 bytes), a shadow; under `InPlace`
            // every other may be in place, counted so.
            let writes = usage.contains(TextureUsage::SHADER_WRITE);
            let may_alias = policy == HostPageTexturePolicy::InPlace
                && (!writes || [4, 8, 16].contains(&texel));
            assert!(
                !made || may_alias,
                "{what}: a texture over host pages aliased"
            );
            assert_eq!(
                (shadowed, in_place),
                match made {
                    true => (0, 1),
                    false => (1, 0),
                },
                "{what}: its form not counted"
            );
            // A shadow's copy in writes every texel before any command reads it: no zeros first.
            assert_eq!(
                counted.shadows_zeroed - before.shadows_zeroed,
                0,
                "{what}: the shadow's image was zeroed"
            );
            assert_eq!(
                counted.images_started_on_the_host - before.images_started_on_the_host,
                0,
                "{what}: the shadow's image was zeroed on the host"
            );
            // Making it asks the driver no format's modifier features, nor does making it again,
            // which takes the same form.
            let again = buffer.new_texture(&d, 0, bpr).unwrap();
            let asked = device.work_counts();
            if !may_alias {
                assert_eq!(
                    asked.modifier_feature_queries - before.modifier_feature_queries,
                    0,
                    "{what}: the texture's making asked the driver's modifier features"
                );
            }
            assert_eq!(aliased(&again), made, "{what}: made again in another form");
            drop(again);
            let between = padding(&mapping.read(0..n), bpr, row);
            // The caller's bytes from before the texture was made.
            assert!(
                rows(&blit_out(&queue, &texture, bpr)) == rows(&mapping.read(0..n)),
                "{what}: a blit does not read the caller's bytes"
            );
            if usage.contains(TextureUsage::RENDER_TARGET) {
                // A render pass's clear, in the pages.
                clear(&queue, &texture, [0.25, 0.5, 0.75, 1.0]);
                let cleared = rows(&blit_out(&queue, &texture, bpr));
                assert!(
                    cleared == rows(&mapping.read(0..n)),
                    "{what}: the clear is not in the pages"
                );
                let first = &cleared[..texel as usize];
                assert!(
                    cleared.chunks_exact(texel as usize).all(|t| t == first),
                    "{what}: the clear wrote unlike texels"
                );
                assert_ne!(
                    first,
                    &counting(len, 7)[..texel as usize],
                    "{what}: the clear wrote nothing"
                );
            } else {
                // A blit's write, in the pages.
                let written = counting(n, 3);
                let source = staged(&device, &written);
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                let (origin, size) = whole();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    bpr,
                    n as u64,
                    size,
                    &texture,
                    0,
                    0,
                    origin,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                assert!(
                    rows(&mapping.read(0..n)) == rows(&written),
                    "{what}: the blit's write is not in the pages"
                );
            }
            assert!(
                padding(&mapping.read(0..n), bpr, row) == between,
                "{what}: a write through the texture reached the bytes between its rows"
            );
            // The caller's later bytes.
            mapping.write(0, &counting(len, 11));
            assert!(
                rows(&blit_out(&queue, &texture, bpr)) == rows(&mapping.read(0..n)),
                "{what}: a blit does not read the caller's later bytes"
            );
            // Writes through the texture and the buffer, read the other way.
            ordered(&queue, &texture, &buffer, (bpr, row), true);
            ordered(&queue, &texture, &buffer, (bpr, row), false);
            eprintln!(
                "NOCOPY-TEXTURE on {}: {what} {W}x{H}: {} (shadows {shadowed})",
                device.name(),
                form(made)
            );
            drop((texture, buffer));
            drop(mapping);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

per_format! {
    read_and_write_the_caller_s_pages:
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_bgra8 => BGRA8Unorm, 4, 0;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgba16f => RGBA16Float, 8, 0;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgb10a2 => RGB10A2Unorm, 4, 0;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgba8 => RGBA8Unorm, 4, 0;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_bgr10_xr => BGR10XR, 4, 0;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_bgra8_padded => BGRA8Unorm, 4, 64;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgba16f_padded => RGBA16Float, 8, 64;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgb10a2_padded => RGB10A2Unorm, 4, 64;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_rgba8_padded => RGBA8Unorm, 4, 64;
    textures_over_no_copy_buffers_read_and_write_the_caller_s_pages_bgr10_xr_padded => BGR10XR, 4, 64;
}

/// `texture`'s slice `slice`, `w`x`h` texels with rows `bpr` bytes apart, blitted out into a new
/// `Shared` buffer.
fn slice_out(
    queue: &CommandQueue,
    texture: &Texture,
    slice: u64,
    (w, h, bpr): (u64, u64, u64),
) -> Vec<u8> {
    let len = bpr * h;
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        slice,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &out,
        0,
        bpr,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    bytes_of(&out)
}

/// Two-slice `RGBA8Unorm` 2D arrays over no-copy buffers, made by the linear call (slices
/// `bytes_per_row` × height rounded up to 128 bytes apart, as Metal lays them out), read and
/// written a slice at a time as Metal's: a 4x4 array with rows of 16 bytes (slice 1 at byte 128,
/// past a gap of 64 bytes), sampled and blitted; and an 8x8 render target with rows of 32 bytes
/// (slice 1 at byte 256). Each slice reads the caller's bytes at its place; a blit's write into
/// slice 1 lands there and nowhere else; a render target's clear of slice 1 lands there and
/// leaves slice 0; the caller's later bytes are what the next blit reads. Printed per array
/// (`NOCOPY-ARRAY`): its form, and for a shadow the `shadowed-texture` notice's reason (each
/// array on a device of its own, as the notice comes once a device).
#[test]
fn two_slice_arrays_over_no_copy_buffers_read_and_write_the_caller_s_pages() {
    let shapes = [
        (4u64, 4u64, TextureUsage::SHADER_READ),
        (
            8,
            8,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        ),
    ];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    for (w, h, usage) in shapes {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let bpr = w * 4;
        let image = (bpr * h) as usize;
        let pitch = image.next_multiple_of(128);
        let mapping = Mapping::anonymous(GUEST_PAGE);
        mapping.write(0, &counting(GUEST_PAGE, 7));
        let Some(buffer) = over(&device, &mapping, GUEST_PAGE) else {
            return;
        };
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 2;
        d.usage = usage;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_linear_texture(&d, 0, bpr, 0).unwrap();
        let made = aliased(&texture);
        let what = format!("RGBA8Unorm {w}x{h}, two slices, usage {usage:?}");
        let shape = (w, h, bpr);
        // The caller's bytes from before the texture was made, each slice at its place.
        for slice in 0..2 {
            let at = slice as usize * pitch;
            assert!(
                slice_out(&queue, &texture, slice, shape) == mapping.read(at..at + image),
                "{what}: slice {slice} does not read the caller's bytes"
            );
        }
        // A blit's write into slice 1, there and nowhere else.
        let before = mapping.read(0..GUEST_PAGE);
        let written = counting(image, 3);
        let source = staged(&device, &written);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            bpr,
            image as u64,
            Size::new(w, h, 1),
            &texture,
            1,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let mut expected = before;
        expected[pitch..pitch + image].copy_from_slice(&written);
        assert!(
            mapping.read(0..GUEST_PAGE) == expected,
            "{what}: the blit into slice 1 is not exactly at its bytes"
        );
        if usage.contains(TextureUsage::RENDER_TARGET) {
            // A render pass's clear of slice 1, there, leaving slice 0.
            let command_buffer = queue.command_buffer().unwrap();
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(texture.clone());
            attachment.slice = 1;
            attachment.load_action = LoadAction::Clear;
            attachment.store_action = StoreAction::Store;
            attachment.clear_color = ClearColor::new(0.0, 1.0, 0.0, 1.0);
            command_buffer
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let green = [0u8, 255, 0, 255].repeat(image / 4);
            assert!(
                mapping.read(pitch..pitch + image) == green,
                "{what}: the clear of slice 1 is not in its bytes"
            );
            assert!(
                mapping.read(0..pitch) == expected[..pitch],
                "{what}: the clear of slice 1 reached slice 0's bytes"
            );
        }
        // The caller's later bytes.
        mapping.write(0, &counting(GUEST_PAGE, 11));
        assert!(
            slice_out(&queue, &texture, 1, shape) == mapping.read(pitch..pitch + image),
            "{what}: slice 1 does not read the caller's later bytes"
        );
        let reason = counted_reason(&messages);
        eprintln!(
            "NOCOPY-ARRAY on {}: {what}, rows of {bpr} bytes, slice 1 at byte {pitch}: {}{reason}",
            device.name(),
            form(made)
        );
        drop((texture, buffer, queue, device));
        drop(mapping);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The `shadowed-texture` notice's reason among the notices delivered so far (removed), as
/// `" (<reason>)"`, or empty where none came.
fn counted_reason(messages: &Messages) -> String {
    messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .find(|notice| notice.degradation == Degradation::ShadowedTexture)
        .map(|notice| format!(" ({})", notice.reason))
        .unwrap_or_default()
}

/// Texels of `pf` (`texel` bytes each) for `n` bytes, varied by `k`, every one a value a blend
/// keeps finite: `RGBA16Float` halves in [2^-7, 1), other formats any bits.
fn texels_of(texel: u64, n: usize, k: usize) -> Vec<u8> {
    if texel == 8 {
        (0..n / 2)
            .flat_map(|i| (0x2000u16 + ((i * 97 + k * 13) % 0x1c00) as u16).to_le_bytes())
            .collect()
    } else {
        counting(n, k)
    }
}

/// The owned fixture's rectangle and solid color (`tob_renders`, as `view_attachments` draws),
/// into one attachment of `pf`, blending the source over the destination by its alpha.
pub(super) fn blending(device: &Device, pf: PixelFormat) -> RenderPipelineState {
    rects(device, pf, true)
}

/// [`blending`]'s pipeline, blending where `blends`, else writing the color over what is there.
pub(super) fn rects(device: &Device, pf: PixelFormat, blends: bool) -> RenderPipelineState {
    rects_beside(device, pf, blends, None)
}

/// [`rects`]'s pipeline for a pass with a depth or stencil attachment of `beside`'s format
/// (`Depth32Float` a depth one, any other a stencil one), which it does not test.
pub(super) fn rects_beside(
    device: &Device,
    pf: PixelFormat,
    blends: bool,
    beside: Option<PixelFormat>,
) -> RenderPipelineState {
    let mut d = rects_descriptor(device, pf, blends);
    match beside {
        Some(PixelFormat::Depth32Float) => d.depth_format = beside,
        Some(_) => d.stencil_format = beside,
        None => {}
    }
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// [`blending`]'s pipeline for a pass whose color 1 is a scratch of `scratch`'s format, which it
/// leaves alone under an empty write mask.
pub(super) fn blending_beside_scratch(
    device: &Device,
    pf: PixelFormat,
    scratch: PixelFormat,
) -> RenderPipelineState {
    let mut d = rects_descriptor(device, pf, true);
    let mut color = ColorAttachmentDescriptor::new(scratch);
    color.write_mask = ColorWriteMask::NONE;
    d.color_attachments[1] = Some(color);
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// [`rects`]'s pipeline descriptor.
fn rects_descriptor(device: &Device, pf: PixelFormat, blends: bool) -> RenderPipelineDescriptor {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
    d.fragment = Some(function("tob_solid"));
    let mut color = ColorAttachmentDescriptor::new(pf);
    color.blending_enabled = blends;
    color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::Zero;
    d.color_attachments[0] = Some(color);
    d
}

/// The bytes of four `f32`s, as a shader reads them.
pub(super) fn floats(values: [f32; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The left and right halves, as the rectangle's corners.
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];
const RIGHT: [f32; 4] = [0.0, -1.0, 1.0, 1.0];

/// What a round of [`passes_blend_load_and_store_as_private_textures`] writes first, in its
/// command buffer.
enum First<'a> {
    /// Nothing.
    Nothing,
    /// A blit of these bytes into the buffer under the texture, or into the texture itself (the
    /// private reference).
    Bytes(&'a [u8], Option<&'a Buffer>),
}

/// A round's render pass: it loads the texture, or clears it to (0.1, 0.2, 0.3, 0.4) where
/// `clear`, blends `color` over `corners`, and stores.
#[derive(Clone, Copy)]
struct Blend {
    clear: bool,
    corners: [f32; 4],
    color: [f32; 4],
}

/// One command buffer: `first`; `blend`'s pass over `texture`; then a blit of the texture into a new
/// buffer and, where `under` is given, one of `under`'s bytes into another. Their bytes.
fn round(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    (texture, under): (&Texture, Option<&Buffer>),
    first: First<'_>,
    blend: Blend,
    bpr: u64,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let Blend {
        clear,
        corners,
        color,
    } = blend;
    let device = queue.device();
    let n = bpr * H;
    let corners = staged(device, &floats(corners));
    let color = staged(device, &floats(color));
    let out = device.new_buffer(n, ResourceOptions::shared()).unwrap();
    let out_under = device.new_buffer(n, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let (origin, size) = whole();
    let mut held = Vec::new();
    if let First::Bytes(bytes, into) = first {
        let source = staged(device, bytes);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        match into {
            Some(buffer) => blit
                .copy_from_buffer_to_buffer(&source, 0, buffer, 0, n)
                .unwrap(),
            None => blit
                .copy_from_buffer_to_texture(
                    &source,
                    0,
                    bpr,
                    n,
                    size,
                    texture,
                    0,
                    0,
                    origin,
                    BlitOption::None,
                )
                .unwrap(),
        }
        blit.end_encoding().unwrap();
        held.push(source);
    }
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = if clear {
        LoadAction::Clear
    } else {
        LoadAction::Load
    };
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.1, 0.2, 0.3, 0.4);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        origin,
        size,
        &out,
        0,
        bpr,
        n,
        BlitOption::None,
    )
    .unwrap();
    if let Some(under) = under {
        blit.copy_from_buffer_to_buffer(under, 0, &out_under, 0, n)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
    (bytes_of(&out), under.map(|_| bytes_of(&out_under)))
}

/// `bytes` blitted into `texture`, rows `bpr` bytes apart, in a command buffer that completes.
fn upload(queue: &CommandQueue, texture: &Texture, bytes: &[u8], bpr: u64) {
    let source = staged(queue.device(), bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let (origin, size) = whole();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        bpr,
        bytes.len() as u64,
        size,
        texture,
        0,
        0,
        origin,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The codes a clear's channel, and a blend over it, may stand from another image's clear of the
/// same value on `device`: none where its clear is Metal's conversion
/// (`DriverFact::ColorClearIsMetals`), else one. Vulkan lets a clear store either code nearest a
/// normalized channel's value and leaves a float's rounding to the device, which ANV on the Iris
/// Xe of RPL-P takes by the image: (0.1, 0.2, 0.3, 0.4) clears an `RGBA8Unorm` or `BGRA8Unorm`
/// channel's 0.1 (25.5 codes) to `0x1a` in a private optimal image and to `0x19` in an image in
/// place over the pages or in a shadow that takes its word view
/// (`DriverFact::ShadowCopyBackDispatchFaster`), and an `RGBA16Float` blend of 0.6 at alpha 0.5
/// over its clear's 0.3 stores `0x3733` in the private image and `0x3732` in place.
pub(super) fn clear_codes_apart(device: &Device) -> u32 {
    let facts = &device.properties().driver_facts;
    u32::from(!facts.contains(&DriverFact::ColorClearIsMetals))
}

/// The texels of `a` and `b` (`texel` bytes each, of four channels: a byte each, or a
/// little-endian 16-bit code) with a channel more than `codes` codes apart, and the first.
pub(super) fn differing_by(a: &[u8], b: &[u8], texel: u64, codes: u32) -> (usize, Option<usize>) {
    let width = (texel / 4).max(1) as usize;
    let code = |c: &[u8]| {
        c.iter()
            .rev()
            .fold(0u32, |v, &byte| v << 8 | u32::from(byte))
    };
    let mut count = 0;
    let mut first = None;
    for (i, (x, y)) in a
        .chunks_exact(texel as usize)
        .zip(b.chunks_exact(texel as usize))
        .enumerate()
    {
        let apart = x
            .chunks_exact(width)
            .zip(y.chunks_exact(width))
            .any(|(x, y)| code(x).abs_diff(code(y)) > codes);
        if apart {
            count += 1;
            first.get_or_insert(i);
        }
    }
    (count, first)
}

/// A render pass over a texture over a no-copy buffer (a surface's render target), its rows padded
/// by `pad` bytes, loads, blends and stores as one over a private texture of the same bytes does,
/// byte for byte on the same device, aliased or shadowed: a pass that loads the caller's bytes and
/// blends over the left half, read through the texture and through the buffer in the same command
/// buffer; a blit's write of the buffer, then a pass that loads it and blends over the right half,
/// in one command buffer; and a pass that clears and blends over the left half. The pages hold the
/// reference's texels after each, and the bytes between the rows stay the caller's. Under each
/// setting of `host_page_textures`: under `InPlace`, once the device's check answered, the render
/// target is in place where its check passed for an image that renders. The form is printed
/// (`NOCOPY-BLEND`). One test a format and pitch, each its own process.
fn passes_blend_load_and_store_as_private_textures(pf: PixelFormat, texel: u64, pad: u64) {
    for policy in [
        HostPageTexturePolicy::Shadow,
        HostPageTexturePolicy::InPlace,
    ] {
        passes_blend_load_and_store_under(pf, texel, pad, policy);
    }
}

/// [`passes_blend_load_and_store_as_private_textures`] on a device of `policy`.
fn passes_blend_load_and_store_under(
    pf: PixelFormat,
    texel: u64,
    pad: u64,
    policy: HostPageTexturePolicy,
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.host_page_textures = policy;
            options.announce_each = [Degradation::DeviceColorConversion].into();
        })
        .unwrap();
        if policy == HostPageTexturePolicy::InPlace {
            assert!(device.settle_in_place_checks(CHECK_WAIT));
        }
        let queue = device.new_command_queue().unwrap();
        let pipeline = blending(&device, pf);
        let row = W * texel;
        let bpr = row + pad;
        let n = (bpr * H) as usize;
        let len = n.next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        let initial = texels_of(texel, n, 1);
        mapping.write(0, &initial);
        let Some(buffer) = over(&device, &mapping, len) else {
            return;
        };
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, bpr).unwrap();
        d.resource_options = ResourceOptions::private();
        let private = device.new_texture(&d).unwrap();
        let what = format!(
            "{policy:?}, {} {W}x{H}, rows of {bpr} bytes ({})",
            pf.name(),
            form(aliased(&texture))
        );
        assert!(
            policy == HostPageTexturePolicy::InPlace || !aliased(&texture),
            "{what}: aliased under the shadow's setting"
        );
        let check_by = |round: &str, ours: &[u8], reference: &[u8], codes: u32| {
            let (ours, reference) = (texel_rows(ours, bpr, row), texel_rows(reference, bpr, row));
            let (count, first) = differing_by(&ours, &reference, texel, codes);
            assert!(
                count == 0,
                "{what}, {round}: {count} texels differ from the private texture's by more than \
                 {codes} codes, the first at texel {first:?}"
            );
        };
        let check =
            |round: &str, ours: &[u8], reference: &[u8]| check_by(round, ours, reference, 0);
        // The bytes between the rows, which no pass or blit through the texture writes.
        let between = |round: &str, expected: &[u8]| {
            assert!(
                padding(&mapping.read(0..n), bpr, row) == padding(expected, bpr, row),
                "{what}, {round}: the bytes between the rows changed"
            );
        };
        let second = texels_of(texel, n, 2);
        let blue = [0.2, 0.4, 0.6, 0.5];
        let amber = [0.9, 0.6, 0.1, 0.25];
        // The reference starts from the caller's bytes.
        upload(&queue, &private, &initial, bpr);
        let (reference, _) = round(
            &queue,
            &pipeline,
            (&private, None),
            First::Nothing,
            Blend {
                clear: false,
                corners: LEFT,
                color: blue,
            },
            bpr,
        );
        // A pass over the caller's bytes as they were before the texture was made.
        let (ours, through_buffer) = round(
            &queue,
            &pipeline,
            (&texture, Some(&buffer)),
            First::Nothing,
            Blend {
                clear: false,
                corners: LEFT,
                color: blue,
            },
            bpr,
        );
        check(
            "a pass that loads and blends, read through the texture",
            &ours,
            &reference,
        );
        check(
            "the same pass read through the buffer in its command buffer",
            &through_buffer.unwrap(),
            &reference,
        );
        check(
            "the same pass, in the pages",
            &mapping.read(0..n),
            &reference,
        );
        between("the first pass", &initial);
        // A blit's write of the buffer, then a pass that loads it, in one command buffer.
        let (reference, _) = round(
            &queue,
            &pipeline,
            (&private, None),
            First::Bytes(&second, None),
            Blend {
                clear: false,
                corners: RIGHT,
                color: amber,
            },
            bpr,
        );
        let (ours, _) = round(
            &queue,
            &pipeline,
            (&texture, None),
            First::Bytes(&second, Some(&buffer)),
            Blend {
                clear: false,
                corners: RIGHT,
                color: amber,
            },
            bpr,
        );
        check(
            "a blit's write of the buffer, then a pass that loads it",
            &ours,
            &reference,
        );
        check("that pass, in the pages", &mapping.read(0..n), &reference);
        between("the blit's write and the pass", &second);
        // A pass that clears and blends, each image's clear announced, within the codes its
        // conversion may stand apart (`clear_codes_apart`).
        messages.take_notices();
        let (reference, _) = round(
            &queue,
            &pipeline,
            (&private, None),
            First::Nothing,
            Blend {
                clear: true,
                corners: LEFT,
                color: blue,
            },
            bpr,
        );
        let private_cleared = clear_conversions(&messages.take_notices());
        let (ours, _) = round(
            &queue,
            &pipeline,
            (&texture, None),
            First::Nothing,
            Blend {
                clear: true,
                corners: LEFT,
                color: blue,
            },
            bpr,
        );
        let ours_cleared = clear_conversions(&messages.take_notices());
        let codes = clear_codes_apart(&device);
        assert!(
            codes == 0 || (private_cleared > 0 && ours_cleared > 0),
            "{what}: the clears' device-color-conversion notices: the private texture's \
             {private_cleared}, ours {ours_cleared}"
        );
        check_by("a pass that clears and blends", &ours, &reference, codes);
        check_by(
            "that pass, in the pages",
            &mapping.read(0..n),
            &reference,
            codes,
        );
        between("the pass that clears", &second);
        eprintln!(
            "NOCOPY-BLEND on {}: {what}: every round as a private texture's",
            device.name()
        );
        drop((texture, private, buffer));
        drop(mapping);
    }
    layer_reported_nothing_at_all(&messages);
}

per_format! {
    passes_blend_load_and_store_as_private_textures:
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_bgra8 => BGRA8Unorm, 4, 0;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgba16f => RGBA16Float, 8, 0;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgb10a2 => RGB10A2Unorm, 4, 0;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgba8 => RGBA8Unorm, 4, 0;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_bgra8_padded => BGRA8Unorm, 4, 64;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgba16f_padded => RGBA16Float, 8, 64;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgb10a2_padded => RGB10A2Unorm, 4, 64;
    passes_over_no_copy_textures_blend_load_and_store_as_private_ones_rgba8_padded => RGBA8Unorm, 4, 64;
}

/// How a frame's render pass writes the surface.
#[derive(Clone, Copy, Debug)]
enum Pass {
    /// No pass.
    None,
    /// Clears it and stores.
    Clear,
    /// Loads it and stores.
    Load,
    /// Clears it, draws a blended rectangle over its left half, and stores.
    ClearDraw,
    /// Loads it, draws a blended rectangle over its left half, and stores.
    LoadDraw,
}

/// What a frame's blit reads the surface through.
#[derive(Clone, Copy, Debug)]
enum Reader {
    /// No blit.
    None,
    /// The texture the pass wrote.
    Same,
    /// A second texture of the surface's plane, over a second no-copy buffer of its pages.
    SecondImport,
}

/// The frames [`surface_frame_costs`] times: each step alone and the consumer's combinations.
const FRAMES_TIMED: [(Pass, Reader); 10] = [
    (Pass::Clear, Reader::None),
    (Pass::ClearDraw, Reader::None),
    (Pass::None, Reader::Same),
    (Pass::None, Reader::SecondImport),
    (Pass::Clear, Reader::Same),
    (Pass::Clear, Reader::SecondImport),
    (Pass::Load, Reader::Same),
    (Pass::Load, Reader::SecondImport),
    (Pass::ClearDraw, Reader::Same),
    (Pass::LoadDraw, Reader::Same),
];

/// The frames timed per case, after [`WARM`] untimed ones.
const FRAMES: usize = 30;
const WARM: usize = 3;

/// The median and the fastest of `times`.
fn median_and_fastest(mut times: Vec<f64>) -> (f64, f64) {
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0])
}

/// The per-frame cost of the consumer's display surface (the module docs), printed.
#[test]
fn surface_frame_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let queue = device.new_command_queue().unwrap();
    for (pf, texel) in [
        (PixelFormat::RGBA16Float, 8u64),
        (PixelFormat::BGRA8Unorm, 4),
    ] {
        let bpr = W * texel;
        let len = ((bpr * H) as usize).next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        let (Some(first), Some(second)) =
            (over(&device, &mapping, len), over(&device, &mapping, len))
        else {
            return;
        };
        let plane = IoSurfacePlane::new(0, bpr, false);
        let surface = |buffer: &Buffer| {
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            buffer.new_texture_with_iosurface_plane(&d, plane).unwrap()
        };
        let writer = surface(&first);
        let second_reader = surface(&second);
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let display = device.new_texture(&d).unwrap();
        let pipeline = blending(&device, pf);
        let corners = staged(&device, &floats(LEFT));
        let color = staged(&device, &floats([0.9, 0.3, 0.1, 0.5]));
        for (pass, read) in FRAMES_TIMED {
            let reader = match read {
                Reader::None => None,
                Reader::Same => Some(&writer),
                Reader::SecondImport => Some(&second_reader),
            };
            let frame = |k: usize| -> (f64, f64) {
                let began = Instant::now();
                let command_buffer = queue.command_buffer().unwrap();
                let (load, draw) = match pass {
                    Pass::None => (None, false),
                    Pass::Clear => (Some(LoadAction::Clear), false),
                    Pass::Load => (Some(LoadAction::Load), false),
                    Pass::ClearDraw => (Some(LoadAction::Clear), true),
                    Pass::LoadDraw => (Some(LoadAction::Load), true),
                };
                if let Some(load) = load {
                    let mut descriptor = RenderPassDescriptor::new();
                    let attachment = &mut descriptor.color_attachments[0];
                    attachment.texture = Some(writer.clone());
                    attachment.load_action = load;
                    attachment.store_action = StoreAction::Store;
                    let shade = (k % 64) as f64 / 64.0;
                    attachment.clear_color = ClearColor::new(shade, 0.5, 1.0 - shade, 1.0);
                    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                    if draw {
                        encoder.set_render_pipeline_state(&pipeline).unwrap();
                        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                        encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
                        encoder
                            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                    }
                    encoder.end_encoding().unwrap();
                }
                if let Some(reader) = reader {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    let (origin, size) = whole();
                    blit.copy_from_texture_to_texture(
                        reader, 0, 0, origin, size, &display, 0, 0, origin,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                }
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                let host = began.elapsed().as_secs_f64() * 1e6;
                assert_eq!(command_buffer.error(), None);
                let start = command_buffer.gpu_start_time().unwrap();
                let end = command_buffer.gpu_end_time().unwrap();
                ((end - start) * 1e6, host)
            };
            for k in 0..WARM {
                frame(k);
            }
            let (gpu, host): (Vec<f64>, Vec<f64>) = (WARM..WARM + FRAMES).map(frame).unzip();
            let (gpu_median, gpu_fastest) = median_and_fastest(gpu);
            let (host_median, host_fastest) = median_and_fastest(host);
            let notices = counts(&messages.take_notices());
            let forms = (
                form(aliased(&writer)),
                reader.map_or("no", |reader| form(aliased(reader))),
            );
            eprintln!(
                "SURFACE-COST on {}: {} {W}x{H}, a {pass:?} pass and a blit through the \
                 {read:?} texture ({} writer, {} reader): GPU median {gpu_median:.0} us \
                 (fastest {gpu_fastest:.0}), frame on the host median {host_median:.0} us \
                 (fastest {host_fastest:.0}); announced {notices:?}",
                device.name(),
                pf.name(),
                forms.0,
                forms.1,
            );
        }
        drop((writer, second_reader, display, first, second));
        drop(mapping);
    }
}
