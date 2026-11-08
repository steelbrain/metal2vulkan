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
//! Ignored by default: measurements, not checks. Run them by name with `--run-ignored all`.

use super::attachment_reads::{floats, function};
use super::first_use_writes::CHECK_WAIT;
use super::in_place_textures::in_place;
use super::no_copy_textures::staged;
use super::pitched_surfaces::reading;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
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

/// The `tob_rect` and `tob_solid` state into a `BGRA8Unorm` color, blended premultiplied.
fn blending(device: &Device) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, "tob_solid", "tob_renders.ll"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
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
    let blends = blending(&device);
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
    let blends = blending(&device);
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
