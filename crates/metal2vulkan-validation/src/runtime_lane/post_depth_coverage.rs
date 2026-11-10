//! Device tests of `[[sample_mask, post_depth_coverage]]` on the device the cfg selects, under the
//! validation layer, over the owned fixture `pdc_post_depth_coverage`.
//!
//! An 8x8 pass at 4 samples (a 4-sample `R32Float` color resolved into a 1-sample one, and a
//! 4-sample `Depth32Float` cleared to 1.0) draws a triangle that writes depth 0.25 over part of the
//! target, then the triangle over the whole target at depth 0.5 under `Less`, without depth
//! writes, through `pdc_after` (the popcount / 4 of the coverage after the early depth test) or
//! `pdc_plain` (1.0). Each runs once per pixel and writes the samples that pass, so where `k` of a
//! pixel's samples pass, `pdc_plain` resolves to k / 4 and `pdc_after` to (k / 4)²: Metal's answer
//! on an M3 Max, at every pixel, over 12 partly covered pixels of this pass. A device without
//! `VK_EXT_post_depth_coverage` refuses `pdc_after` by that need; a device whose driver splits a
//! clipped triangle's pixel-rate fragments (lavapipe) may refuse it for that first.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverWorkaround, VkExtension};
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, Error, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};
use std::path::Path;

/// The pass's side in pixels.
const SIDE: u64 = 8;

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine/pdc_post_depth_coverage.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// A pipeline of `pdc_vertex` and `fragment` at 4 samples over an `R32Float` color, written or
/// not, and a `Depth32Float` depth.
fn pipeline(
    device: &Device,
    library: &Library,
    fragment: &str,
    written: bool,
) -> Result<RenderPipelineState, Error> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("pdc_vertex"));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::R32Float);
    if !written {
        color.write_mask = ColorWriteMask::NONE;
    }
    descriptor.color_attachments[0] = Some(color);
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor.raster_sample_count = 4;
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A texture of `pf`, SIDE wide and high, of `samples`, a render target, shared or private.
fn texture(device: &Device, pf: PixelFormat, samples: u64, shared: bool) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
    }
    descriptor.sample_count = samples;
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = if shared {
        ResourceOptions::shared()
    } else {
        ResourceOptions::private()
    };
    device.new_texture(&descriptor).unwrap()
}

/// The resolved red of each pixel after the pass draws the depth triangle through `depth`, then
/// the whole target through `drawn` (see the module docs).
fn drawn(device: &Device, depth: &RenderPipelineState, drawn: &RenderPipelineState) -> Vec<f32> {
    let color = texture(device, PixelFormat::R32Float, 4, false);
    let resolved = texture(device, PixelFormat::R32Float, 1, true);
    let depth_texture = texture(device, PixelFormat::Depth32Float, 4, false);
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachments[0].texture = Some(color);
    pass.color_attachments[0].load_action = LoadAction::Clear;
    pass.color_attachments[0].store_action = StoreAction::MultisampleResolve;
    pass.color_attachments[0].resolve_texture = Some(resolved.clone());
    pass.depth_attachment.texture = Some(depth_texture);
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    let state = |compare, write| {
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = compare;
        state.depth_write_enabled = write;
        device.new_depth_stencil_state(&state).unwrap()
    };
    let (always, less) = (
        state(CompareFunction::Always, true),
        state(CompareFunction::Less, false),
    );
    let floats =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let near = buffer_with(device, &floats(&[-1.0, -1.0, 1.0, -0.3, -1.0, 1.0])).unwrap();
    let whole = buffer_with(device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0])).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    for (pipeline, state, corners, z) in [
        (depth, &always, &near, 0.25f32),
        (drawn, &less, &whole, 0.5),
    ] {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder.set_depth_stencil_state(state).unwrap();
        encoder.set_vertex_buffer(Some(corners), 0, 0).unwrap();
        encoder.set_vertex_bytes(&z.to_le_bytes(), 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
    resolved
        .get_bytes(
            &mut bytes,
            4 * SIDE,
            4 * SIDE * SIDE,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The coverage after the early depth test reads as Metal's: `pdc_after` resolves to the square
/// of `pdc_plain` at every pixel, partly covered ones among them.
#[test]
fn the_coverage_after_the_depth_test_reads_as_metal_s() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let library = library(&device);
        let after = match pipeline(&device, &library, "pdc_after", true) {
            Ok(after) => after,
            Err(Error::Unsupported(u)) => {
                let ext = VkExtension::ExtPostDepthCoverage;
                let lacks = u.feature == Feature::PostDepthCoverage
                    && u.needs == Needs::AnyOf(vec![vec![Capability::Extension(ext)]])
                    && !device
                        .compile_device()
                        .profile()
                        .enabled()
                        .extensions
                        .contains(&ext);
                let split = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
                let splits = u.feature == Feature::PixelCoverage
                    && u.needs == Needs::DriverDefect(split)
                    && device
                        .compile_device()
                        .profile()
                        .workarounds()
                        .contains(&split);
                assert!(lacks || splits, "pdc_after on {name}: {u:?}");
                eprintln!("post-depth coverage on {name}: refused, {u:?}");
                return;
            }
            Err(error) => panic!("pdc_after on {name}: {error}"),
        };
        let plain = pipeline(&device, &library, "pdc_plain", true).unwrap();
        let depth = pipeline(&device, &library, "pdc_plain", false).unwrap();
        let reds = drawn(&device, &depth, &plain);
        let squares = drawn(&device, &depth, &after);
        let partial = reds.iter().filter(|&&r| r > 0.0 && r < 1.0).count();
        assert!(partial > 0, "no pixel partly covered on {name}: {reds:?}");
        assert_eq!(
            squares,
            reds.iter().map(|r| r * r).collect::<Vec<_>>(),
            "pdc_after against pdc_plain squared on {name}"
        );
        eprintln!("post-depth coverage on {name}: Metal's, {partial} pixels partly covered");
    }
    layer_reported_nothing_at_all(&messages);
}
