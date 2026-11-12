//! What a device cache's draws cost as a shadow and in place at its pitch, on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the pooled
//! bound (`in_place::POOLED_PITCHED_BYTES`, `TextureShadowReason::LinearCost`) against how much
//! a frame draws into each cache.
//!
//! The textures are a guest's device caches, `BGRA8Unorm` with render target and shader read
//! usage over `Private` buffers of the runtime's own, [`TEXTURES`] of each of [`SHAPES`]: over
//! buffers that declare the bytes between rows unobserved (`BufferDescriptor::padding_unobserved`),
//! which lie in place at their pitch as a `DRM_FORMAT_MOD_LINEAR` image where the device takes one
//! and the bound allows, and over ones that do not, which are shadows. Each frame is one command
//! buffer: for each cache a pass that clears it and stores it, drawing each of [`DRAWS`] times a
//! rectangle over the whole cache blended premultiplied (`tob_rect`, `tob_solid`), then for each
//! cache a draw into a private target that reads it whole (`tob_read`), as
//! `in_place_textures::pooled_pitched_costs` does with no draw. [`WARM`] frames run untimed, then
//! [`TIMED`] frames are printed per shape, draws and form (`POOLED-DRAW-COST`): the caches in
//! place at their pitch and as shadows, the median and fastest GPU time of a frame, and the shadow
//! copies a frame.
//!
//! [`pooled_surface_costs`] weighs a guest window server's desktop device cache, past the bound,
//! against a private texture: frames of many passes into it, each its own encoder.
//!
//! [`pooled_sample_costs`] weighs caches a frame only samples, as a window server samples the
//! surfaces an app drew, over `Shared` buffers the CPU writes each frame and over `Private` ones:
//! at a padded pitch (a shadow past the bound) and at their tight one (the device's own linear
//! image, in place at any size).
//!
//! [`pooled_crossover_costs`] weighs one texture over a `Private` buffer, from 16 KiB to 4 MiB,
//! in place at its pitch against a shadow, as a compositor samples it each frame, the texture
//! changing every frame or every 60th: the pooled bound's crossover on the device.
//!
//! Ignored by default: measurements, not checks. Run them by name with `--run-ignored all`.

use super::attachment_reads::{floats, function};
use super::first_use_writes::CHECK_WAIT;
use super::in_place_textures::in_place;
use super::no_copy_textures::staged;
use super::pitched_surfaces::reading;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BlendFactor, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    Buffer, BufferContents, BufferDescriptor, ClearColor, Device, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, StoreAction, Texture,
    TextureDescriptor,
};

/// The caches' shapes, (width, rows, row bytes): the consumer's three most common device caches
/// on the Iris Xe and the shapes of the sweep behind the pooled bound, from 23 KB to 1.36 MB.
const SHAPES: [(u64, u64, u64); 9] = [
    (104, 52, 448),
    (136, 68, 576),
    (172, 67, 704),
    (250, 125, 1024),
    (200, 200, 832),
    (250, 250, 1024),
    (360, 360, 1472),
    (540, 270, 2176),
    (820, 410, 3328),
];
/// The caches of each shape a frame draws.
const TEXTURES: usize = 4;
/// The draws into each cache a frame.
const DRAWS: [usize; 4] = [0, 1, 4, 16];
/// The untimed and the timed frames of each case.
const WARM: usize = 5;
const TIMED: usize = 40;

/// A device of the in-place policy, without the validation layer, its in-place check settled.
fn settled_device(cache: &ScratchCache, messages: &Messages) -> Device {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        in_place(options);
    })
    .unwrap();
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    device
}

/// The `tob_rect` and `tob_solid` state into a color of `pf`, blended premultiplied.
fn blending(device: &Device, pf: PixelFormat) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, "tob_solid", "tob_renders.ll"));
    let mut color = ColorAttachmentDescriptor::new(pf);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    d.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// A `BGRA8Unorm` texture of render target and shader read usage, `w` by `h`, over a new `Private`
/// buffer of rows `bpr` apart, which declares the bytes between them unobserved where `declared`.
fn cache_texture(
    device: &Device,
    (w, h, bpr): (u64, u64, u64),
    declared: bool,
) -> (Buffer, Texture) {
    let descriptor =
        BufferDescriptor::new(bpr * h, ResourceOptions::private(), BufferContents::Zero);
    let descriptor = match declared {
        true => descriptor.with_padding_unobserved(),
        false => descriptor,
    };
    let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, bpr).unwrap();
    (buffer, texture)
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pooled_draw_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = settled_device(&cache, &messages);
    let queue = device.new_command_queue().unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let reads = reading(&device, bgra, "tob_read");
    let blends = blending(&device, bgra);
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let half = floats([0.25, 0.125, 0.0625, 0.5]);
    let mut d = TextureDescriptor::texture_2d(bgra, 1640, 820, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    for (w, h, bpr) in SHAPES {
        for declared in [false, true] {
            let before = device.work_counts();
            let caches: Vec<_> = (0..TEXTURES)
                .map(|_| cache_texture(&device, (w, h, bpr), declared))
                .collect();
            let after = device.work_counts();
            let pitched = after.pooled_pitched_images - before.pooled_pitched_images;
            let shadows = after.shadowed_textures - before.shadowed_textures;
            for draws in DRAWS {
                let frame = |n: usize| {
                    let command_buffer = queue.command_buffer().unwrap();
                    for (_, texture) in &caches {
                        let mut pass = RenderPassDescriptor::new();
                        let color = &mut pass.color_attachments[0];
                        color.texture = Some(texture.clone());
                        color.load_action = LoadAction::Clear;
                        color.clear_color = ClearColor::new(n as f64 / 64.0, 0.5, 0.25, 1.0);
                        color.store_action = StoreAction::Store;
                        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                        if draws > 0 {
                            encoder.set_render_pipeline_state(&blends).unwrap();
                            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                            encoder.set_fragment_bytes(&half, 0).unwrap();
                        }
                        for _ in 0..draws {
                            encoder
                                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                                .unwrap();
                        }
                        encoder.end_encoding().unwrap();
                    }
                    for (_, texture) in &caches {
                        let mut pass = RenderPassDescriptor::new();
                        let color = &mut pass.color_attachments[0];
                        color.texture = Some(target.clone());
                        color.load_action = LoadAction::DontCare;
                        color.store_action = StoreAction::Store;
                        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                        encoder.set_render_pipeline_state(&reads).unwrap();
                        encoder
                            .set_scissor_rect(ScissorRect::new(0, 0, w, h))
                            .unwrap();
                        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                        encoder.set_fragment_texture(Some(texture), 0).unwrap();
                        encoder
                            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                        encoder.end_encoding().unwrap();
                    }
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None);
                    command_buffer.gpu_end_time().unwrap()
                        - command_buffer.gpu_start_time().unwrap()
                };
                for n in 0..WARM {
                    frame(n);
                }
                let copies = device.work_counts().shadow_copies;
                let mut times: Vec<f64> = (WARM..WARM + TIMED).map(&frame).collect();
                let copies = device.work_counts().shadow_copies - copies;
                times.sort_by(f64::total_cmp);
                println!(
                    "POOLED-DRAW-COST {} {TEXTURES} caches {w}x{h} at {bpr} ({} KB) {draws} draws \
                     each, declared {declared}: {pitched} in place at their pitch, {shadows} \
                     shadows; a frame's GPU time median {:.1} us, fastest {:.1} us; shadow copies \
                     {:.1} a frame",
                    device.name(),
                    bpr * h / 1000,
                    times[TIMED / 2] * 1e6,
                    times[0] * 1e6,
                    copies as f64 / TIMED as f64,
                );
            }
        }
    }
    messages.take_notices();
}

/// The `pooled_surface_costs` caches, (width, rows, row bytes, declared): the guest window server's
/// desktop device cache at its 3662-texel width and 14848-byte rows, over a buffer that declares
/// its padding unobserved (past the pooled bound: `LinearCost`) and over one that does not; and the
/// 3648x2038 one of a guest whose modes are a multiple of 16 texels wide, at its 14592-byte rows.
const SURFACES: [(u64, u64, u64, bool); 3] = [
    (3662, 2038, 14848, true),
    (3662, 2038, 14848, false),
    (3648, 2038, 14592, true),
];
/// The passes of a `pooled_surface_costs` frame, and the draws of each.
const SURFACE_PASSES: [usize; 3] = [1, 8, 30];
const SURFACE_DRAWS: usize = 3;

/// What a guest's desktop device cache costs a frame of render passes: over a private texture and
/// over each of [`SURFACES`], one command buffer a frame of each count of [`SURFACE_PASSES`]
/// passes, each an encoder of its own that loads and stores the cache and draws [`SURFACE_DRAWS`]
/// blended 600x400 rectangles at places of their own, through no scissor or through a scissor of
/// each rectangle; then a draw that reads the whole cache into a private target, as the
/// consumer's present samples it. Printed per case (`POOLED-SURFACE-COST`): the cache's form, the
/// median GPU time of a frame of [`TIMED`] after [`WARM`], and per frame the shadow copies and the
/// texels they copy in and back.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pooled_surface_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = settled_device(&cache, &messages);
    let queue = device.new_command_queue().unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let reads = reading(&device, bgra, "tob_read");
    let blends = blending(&device, bgra);
    let whole = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let half = floats([0.25, 0.125, 0.0625, 0.5]);
    let mut d = TextureDescriptor::texture_2d(bgra, 3662, 2038, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let presented = device.new_texture(&d).unwrap();
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let private = device.new_texture(&d).unwrap();
    let mut caches = vec![(private, "private".to_owned(), None)];
    for (w, h, bpr, declared) in SURFACES {
        let before = device.work_counts();
        let (buffer, texture) = cache_texture(&device, (w, h, bpr), declared);
        let after = device.work_counts();
        let form = match after.shadowed_textures - before.shadowed_textures {
            0 => "in place",
            _ => "a shadow",
        };
        caches.push((
            texture,
            format!("{w}x{h} at {bpr}, declared {declared}, {form}"),
            Some(buffer),
        ));
    }
    for (texture, kind, _) in &caches {
        let (w, h) = (texture.width(), texture.height());
        let rect = |n: usize| {
            let n = n as u64;
            let (x, y) = (n * 1373 % (w - 600), (n * 3 + 1) * 787 % (h - 400));
            [x, y, x + 600, y + 400]
        };
        let corners = |[x0, y0, x1, y1]: [u64; 4]| {
            let x = |p: u64| p as f32 / w as f32 * 2.0 - 1.0;
            let y = |p: u64| 1.0 - p as f32 / h as f32 * 2.0;
            floats([x(x0), y(y1), x(x1), y(y0)])
        };
        for passes in SURFACE_PASSES {
            for scissored in [false, true] {
                let frame = |n: usize| {
                    let command_buffer = queue.command_buffer().unwrap();
                    for pass in 0..passes {
                        let mut d = RenderPassDescriptor::new();
                        let color = &mut d.color_attachments[0];
                        color.texture = Some(texture.clone());
                        color.load_action = LoadAction::Load;
                        color.store_action = StoreAction::Store;
                        let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
                        encoder.set_render_pipeline_state(&blends).unwrap();
                        encoder.set_fragment_bytes(&half, 0).unwrap();
                        for draw in 0..SURFACE_DRAWS {
                            let r @ [x0, y0, x1, y1] = rect(n + pass * SURFACE_DRAWS + draw);
                            if scissored {
                                encoder
                                    .set_scissor_rect(ScissorRect::new(x0, y0, x1 - x0, y1 - y0))
                                    .unwrap();
                            }
                            encoder.set_vertex_bytes(&corners(r), 0).unwrap();
                            encoder
                                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                                .unwrap();
                        }
                        encoder.end_encoding().unwrap();
                    }
                    let mut d = RenderPassDescriptor::new();
                    let color = &mut d.color_attachments[0];
                    color.texture = Some(presented.clone());
                    color.load_action = LoadAction::DontCare;
                    color.store_action = StoreAction::Store;
                    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
                    encoder.set_render_pipeline_state(&reads).unwrap();
                    encoder.set_vertex_buffer(Some(&whole), 0, 0).unwrap();
                    encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None);
                    command_buffer.gpu_end_time().unwrap()
                        - command_buffer.gpu_start_time().unwrap()
                };
                for n in 0..WARM {
                    frame(n);
                }
                let before = device.work_counts();
                let mut times: Vec<f64> = (WARM..WARM + TIMED).map(&frame).collect();
                let after = device.work_counts();
                times.sort_by(f64::total_cmp);
                let per = |a: u64, b: u64| (b - a) as f64 / TIMED as f64;
                println!(
                    "POOLED-SURFACE-COST {} {kind}: {passes} passes of {SURFACE_DRAWS} draws, \
                     scissored {scissored}: GPU {:.3} ms/frame (median); per frame: shadow copies \
                     {:.1}, texels copied in {:.0}, back {:.0}",
                    device.name(),
                    times[TIMED / 2] * 1e3,
                    per(before.shadow_copies, after.shadow_copies),
                    per(
                        before.shadow_texels_copied_in,
                        after.shadow_texels_copied_in
                    ),
                    per(
                        before.shadow_texels_copied_back,
                        after.shadow_texels_copied_back
                    ),
                );
            }
        }
    }
    drop(caches);
    messages.take_notices();
}

/// The `pooled_sample_costs` caches, (width, rows, padded row bytes): one of the consumer's common
/// small device caches, a browser's tile, a browser's window and the desktop; each also at its
/// tight rows, a multiple of the 16 bytes Metal's pitch takes here (each width a multiple of 4).
const SAMPLED: [(u64, u64, u64); 4] = [
    (172, 67, 704),
    (1308, 512, 5248),
    (1324, 940, 5312),
    (3648, 2038, 14848),
];

/// What a frame that samples each cache whole costs: [`TEXTURES`] `BGRA8Unorm` textures of shader
/// read usage alone of each of [`SAMPLED`] at its padded pitch over a buffer that declares its
/// padding unobserved and at its tight one, over `Shared` buffers whose first texels the CPU
/// writes before each frame's commit and over `Private` ones; each frame one command buffer of a
/// draw into a private target that reads each texture whole (`tob_read`). Printed per case
/// (`POOLED-SAMPLE-COST`): the textures in place and as shadows, the median GPU time of a frame of
/// [`TIMED`] after [`WARM`], and per frame the shadow copies and the texels they copy in.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pooled_sample_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = settled_device(&cache, &messages);
    let queue = device.new_command_queue().unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let reads = reading(&device, bgra, "tob_read");
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let mut d = TextureDescriptor::texture_2d(bgra, 3648, 2038, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    for (w, h, padded) in SAMPLED {
        for options in [ResourceOptions::shared(), ResourceOptions::private()] {
            for bpr in [padded, 4 * w] {
                let before = device.work_counts();
                let textures: Vec<_> = (0..TEXTURES)
                    .map(|_| {
                        let descriptor =
                            BufferDescriptor::new(bpr * h, options, BufferContents::Zero)
                                .with_padding_unobserved();
                        let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
                        let mut d = TextureDescriptor::texture_2d(bgra, w, h, false);
                        d.usage = TextureUsage::SHADER_READ;
                        d.resource_options = buffer.resource_options();
                        let texture = buffer.new_texture(&d, 0, bpr).unwrap();
                        (buffer, texture)
                    })
                    .collect();
                let after = device.work_counts();
                let shadows = after.shadowed_textures - before.shadowed_textures;
                let frame = |n: usize| {
                    for (buffer, _) in &textures {
                        if let Some(contents) = buffer.contents() {
                            let texel = [n as u8, 0x40, 0x80, 0xff];
                            // SAFETY: a `Shared` buffer of at least a texel, which no command
                            // uses now (the frame before was waited for).
                            unsafe {
                                std::ptr::copy_nonoverlapping(texel.as_ptr(), contents.as_ptr(), 4)
                            };
                        }
                    }
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut pass = RenderPassDescriptor::new();
                    let color = &mut pass.color_attachments[0];
                    color.texture = Some(target.clone());
                    color.load_action = LoadAction::DontCare;
                    color.store_action = StoreAction::Store;
                    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder.set_render_pipeline_state(&reads).unwrap();
                    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                    encoder
                        .set_scissor_rect(ScissorRect::new(0, 0, w, h))
                        .unwrap();
                    for (_, texture) in &textures {
                        encoder.set_fragment_texture(Some(texture), 0).unwrap();
                        encoder
                            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                    }
                    encoder.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None);
                    command_buffer.gpu_end_time().unwrap()
                        - command_buffer.gpu_start_time().unwrap()
                };
                for n in 0..WARM {
                    frame(n);
                }
                let before = device.work_counts();
                let mut times: Vec<f64> = (WARM..WARM + TIMED).map(&frame).collect();
                let after = device.work_counts();
                times.sort_by(f64::total_cmp);
                let per = |a: u64, b: u64| (b - a) as f64 / TIMED as f64;
                println!(
                    "POOLED-SAMPLE-COST {} {TEXTURES} textures {w}x{h} at {bpr} ({} KB) over {:?} \
                     buffers: {} in place, {shadows} shadows; a frame's GPU time median {:.1} us; \
                     per frame: shadow copies {:.1}, texels copied in {:.0}",
                    device.name(),
                    bpr * h / 1000,
                    options.storage_mode(),
                    TEXTURES as u64 - shadows,
                    times[TIMED / 2] * 1e6,
                    per(before.shadow_copies, after.shadow_copies),
                    per(
                        before.shadow_texels_copied_in,
                        after.shadow_texels_copied_in
                    ),
                );
            }
        }
    }
    messages.take_notices();
}

/// The `pooled_crossover_costs` textures, (bytes of a row's texels, rows, row bytes): from 16 KiB
/// to 4 MiB to their pitch, each pitch a multiple of 256 bytes at least 256 past the multiple of
/// 256 its texels' bytes round up to, so that no device's own linear image lays rows there; and,
/// of `BGRA8Unorm`, the shape of a guest's most common shadow of its linear cost on an RTX 2080
/// (723x625 at 3072-byte rows), which a device that lays linear rows on 256 bytes takes as its
/// own linear image.
const CROSSOVER: [(u64, u64, u64); 10] = [
    (240, 32, 512),
    (240, 64, 512),
    (720, 64, 1024),
    (720, 128, 1024),
    (1760, 128, 2048),
    (1760, 256, 2048),
    (1760, 512, 2048),
    (2892, 625, 3072),
    (3840, 512, 4096),
    (3840, 1024, 4096),
];
/// The frames from one change of a `pooled_crossover_costs` texture to the next.
const PERIODS: [usize; 2] = [1, 60];
/// The timed frames of each `pooled_crossover_costs` case: two changes at every 60th frame.
const CROSSOVER_TIMED: usize = 120;

/// How a `pooled_crossover_costs` texture changes: through a pass that clears it and draws a
/// blended rectangle over it whole (a shadow's image copied back after it), or through a blit that
/// copies its buffer's bytes whole from another buffer (a shadow's image copied in at its next
/// use).
#[derive(Clone, Copy, Debug)]
enum Change {
    Render,
    Blit,
}

/// What one texture over a `Private` buffer costs a frame in place at its pitch and as a shadow:
/// of `BGRA8Unorm` and `RGBA16Float`, render target and shader read usage, each of [`CROSSOVER`],
/// in place over a buffer that declares its padding unobserved, over which a one-row texture was
/// made first (so the measured one takes the `DRM_FORMAT_MOD_LINEAR` image at any size,
/// `texture_spans.rs`), and a shadow over one that does not declare it. Each frame is one command
/// buffer: every [`PERIODS`]th frame the texture changes ([`Change`]), then a draw samples it
/// whole, one texel a pixel through a linear filter (`tob_sample`), into a private target, as a
/// compositor draws a window's surface. [`WARM`] frames run untimed, then [`CROSSOVER_TIMED`]
/// frames; printed per case (`POOLED-CROSSOVER`): each form's mean GPU time and wall time (encode,
/// commit and wait) of a frame and the shadow's copies a frame. `POOLED-CROSSOVER-FORMS` names
/// each texture's placement as the query answers it and whether it was made in that form.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pooled_crossover_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = settled_device(&cache, &messages);
    let queue = device.new_command_queue().unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let samples = reading(&device, bgra, "tob_sample");
    let mut sd = SamplerDescriptor::new();
    sd.min_filter = SamplerMinMagFilter::Linear;
    sd.mag_filter = SamplerMinMagFilter::Linear;
    let sampler = device.new_sampler_state(&sd).unwrap();
    let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let half = floats([0.25, 0.125, 0.0625, 0.5]);
    let mut d = TextureDescriptor::texture_2d(bgra, 1024, 1024, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let private = ResourceOptions::private();
    for (pf, texel) in [(bgra, 4), (PixelFormat::RGBA16Float, 8)] {
        let blends = blending(&device, pf);
        for (row, h, bpr) in CROSSOVER {
            let (w, bytes) = (row / texel, bpr * h);
            let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            d.resource_options = private;
            let declared = BufferDescriptor::new(bytes, private, BufferContents::Zero)
                .with_padding_unobserved();
            let aliased = device.new_buffer_with_descriptor(declared).unwrap();
            let mut first = d;
            first.height = 1;
            let first = aliased.new_texture(&first, 0, bpr).unwrap();
            let placed_in = aliased.texture_placement(&d, 0, bpr).unwrap();
            let before = device.work_counts();
            let placed = aliased.new_texture(&d, 0, bpr).unwrap();
            let after = device.work_counts();
            let in_place_made = after.shadowed_textures == before.shadowed_textures
                && after.pooled_pitched_images > before.pooled_pitched_images;
            let plain = BufferDescriptor::new(bytes, private, BufferContents::Zero);
            let plain = device.new_buffer_with_descriptor(plain).unwrap();
            let placed_shadow = plain.texture_placement(&d, 0, bpr).unwrap();
            let before = device.work_counts();
            let shadow = plain.new_texture(&d, 0, bpr).unwrap();
            let shadow_made = device.work_counts().shadowed_textures > before.shadowed_textures;
            println!(
                "POOLED-CROSSOVER-FORMS {} {pf:?} {w}x{h} at {bpr} ({bytes} bytes): in place \
                 {placed_in:?}, made {in_place_made}; shadow {placed_shadow:?}, made {shadow_made}",
                device.name(),
            );
            let source = BufferDescriptor::new(bytes, private, BufferContents::Zero);
            let source = device.new_buffer_with_descriptor(source).unwrap();
            let forms = [
                ("in place", in_place_made, &aliased, &placed),
                ("shadow", shadow_made, &plain, &shadow),
            ];
            for change in [Change::Render, Change::Blit] {
                for period in PERIODS {
                    let mut costs = Vec::new();
                    for (form, made, buffer, texture) in forms {
                        if !made {
                            costs.push(format!("{form} not made"));
                            continue;
                        }
                        let frame = |n: usize| {
                            let start = std::time::Instant::now();
                            let command_buffer = queue.command_buffer().unwrap();
                            if n % period == 0 {
                                match change {
                                    Change::Render => {
                                        let mut pass = RenderPassDescriptor::new();
                                        let color = &mut pass.color_attachments[0];
                                        color.texture = Some(texture.clone());
                                        color.load_action = LoadAction::Clear;
                                        color.clear_color =
                                            ClearColor::new(n as f64 / 256.0, 0.5, 0.25, 1.0);
                                        color.store_action = StoreAction::Store;
                                        let mut encoder =
                                            command_buffer.render_command_encoder(&pass).unwrap();
                                        encoder.set_render_pipeline_state(&blends).unwrap();
                                        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                                        encoder.set_fragment_bytes(&half, 0).unwrap();
                                        encoder
                                            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                                            .unwrap();
                                        encoder.end_encoding().unwrap();
                                    }
                                    Change::Blit => {
                                        let mut encoder =
                                            command_buffer.blit_command_encoder().unwrap();
                                        encoder
                                            .copy_from_buffer_to_buffer(
                                                &source, 0, buffer, 0, bytes,
                                            )
                                            .unwrap();
                                        encoder.end_encoding().unwrap();
                                    }
                                }
                            }
                            let mut pass = RenderPassDescriptor::new();
                            let color = &mut pass.color_attachments[0];
                            color.texture = Some(target.clone());
                            color.load_action = LoadAction::DontCare;
                            color.store_action = StoreAction::Store;
                            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                            encoder.set_render_pipeline_state(&samples).unwrap();
                            encoder
                                .set_scissor_rect(ScissorRect::new(0, 0, w, h))
                                .unwrap();
                            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                            encoder.set_fragment_texture(Some(texture), 0).unwrap();
                            encoder
                                .set_fragment_sampler_state(Some(&sampler), 0)
                                .unwrap();
                            encoder
                                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                                .unwrap();
                            encoder.end_encoding().unwrap();
                            command_buffer.commit().unwrap();
                            command_buffer.wait_until_completed_for(WAIT).unwrap();
                            assert_eq!(command_buffer.error(), None);
                            let wall = start.elapsed().as_secs_f64();
                            let gpu = command_buffer.gpu_end_time().unwrap()
                                - command_buffer.gpu_start_time().unwrap();
                            (gpu, wall)
                        };
                        for n in 0..WARM {
                            frame(n);
                        }
                        let copies = device.work_counts().shadow_copies;
                        let times: Vec<(f64, f64)> =
                            (WARM..WARM + CROSSOVER_TIMED).map(&frame).collect();
                        let copies = device.work_counts().shadow_copies - copies;
                        let mean = |f: fn(&(f64, f64)) -> f64| {
                            times.iter().map(f).sum::<f64>() / CROSSOVER_TIMED as f64 * 1e6
                        };
                        costs.push(format!(
                            "{form} GPU {:.1} us wall {:.1} us copies {:.2}",
                            mean(|t| t.0),
                            mean(|t| t.1),
                            copies as f64 / CROSSOVER_TIMED as f64,
                        ));
                    }
                    println!(
                        "POOLED-CROSSOVER {} {pf:?} {w}x{h} at {bpr} ({bytes} bytes) changes by \
                         {change:?} every {period}: {}",
                        device.name(),
                        costs.join("; "),
                    );
                }
            }
            drop(first);
        }
    }
    messages.take_notices();
}
