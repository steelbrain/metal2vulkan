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
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

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
    BufferContents, BufferDescriptor, ClearColor, LoadAction, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, ScissorRect, StoreAction, TextureDescriptor,
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

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pooled_draw_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        in_place(options);
    })
    .unwrap();
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    let queue = device.new_command_queue().unwrap();
    let bgra = PixelFormat::BGRA8Unorm;
    let reads = reading(&device, bgra, "tob_read");
    let mut d = RenderPipelineDescriptor::new(function(&device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(&device, "tob_solid", "tob_renders.ll"));
    let mut color = ColorAttachmentDescriptor::new(bgra);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    d.color_attachments[0] = Some(color);
    let blends = device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap();
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
                .map(|_| {
                    let descriptor = BufferDescriptor::new(
                        bpr * h,
                        ResourceOptions::private(),
                        BufferContents::Zero,
                    );
                    let descriptor = match declared {
                        true => descriptor.with_padding_unobserved(),
                        false => descriptor,
                    };
                    let buffer = device.new_buffer_with_descriptor(descriptor).unwrap();
                    let mut d = TextureDescriptor::texture_2d(bgra, w, h, false);
                    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
                    d.resource_options = buffer.resource_options();
                    let texture = buffer.new_texture(&d, 0, bpr).unwrap();
                    (buffer, texture)
                })
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
