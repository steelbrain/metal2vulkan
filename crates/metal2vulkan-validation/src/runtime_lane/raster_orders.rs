//! A raster order group on a color output the function does not read on the device the cfg selects,
//! under the validation layer.
//!
//! The owned fixture `fragment_rog_order` draws Metal's fullscreen triangle eight times over, its
//! triangle `k` writing `(v, 1 - v, v / 2, 1 / 2)` with `v = (k + 1) / 8` into a color output in
//! group 0 that the function does not read. The pipeline blends every channel `d' = s + d / 2`
//! (factors `One` and `SourceAlpha`), which does not commute, so the 4² `RGBA16Float` target,
//! cleared to (0.5, 0.25, 0.125, 1), holds the answer of the triangles' order alone, every step
//! exact in `half`: the eight in one draw, in eight draws, and in two of four. Metal's answer, as
//! an M4 draws it, is that order's in each.
//!
//! The compile layer gives such a group no interlock: Vulkan blends a sample's fragments in
//! rasterization order, the draws' order and within a draw its primitives'. So every device gives
//! Metal's answer, whether or not it has the pixel interlock (which is printed).

use super::resolves::{lane, quiet, Lane, ONE};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    StoreAction,
};

/// The triangles drawn, each three vertices.
const TRIANGLES: u32 = 8;

/// The clear color.
const CLEAR: [f32; 4] = [0.5, 0.25, 0.125, 1.0];

/// The fixture's pipeline on `lane`'s device: `rog_order_layers` and `rog_order_output`, drawing
/// an `RGBA16Float` target blended `d' = s + d / 2`.
fn pipeline(lane: &Lane) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("fragment_rog_order.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("rog_order_layers")));
    descriptor.fragment = Some(StageSource::Function(function("rog_order_output")));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::SourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::SourceAlpha;
    descriptor.color_attachments[0] = Some(color);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What every texel holds after the triangles blend in their order: `d' = s + d / 2` from the
/// clear, triangle `k`'s `s` being `(v, 1 - v, v / 2, 1 / 2)` with `v = (k + 1) / 8`.
fn ordered() -> [f32; 4] {
    let mut d = CLEAR;
    for k in 0..TRIANGLES {
        let v = (k + 1) as f32 / 8.0;
        for (d, s) in d.iter_mut().zip([v, 1.0 - v, v / 2.0, 0.5]) {
            *d = s + *d / 2.0;
        }
    }
    d
}

/// The value of the normal or zero `half` of `bits` (every value here is one).
fn half(bits: u16) -> f32 {
    let sign = if bits >> 15 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    assert!(exponent != 0x1f, "not a finite half: {bits:#06x}");
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-24)
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

/// The texels the target holds after `draws` (each a first vertex and a vertex count).
fn drawn(lane: &Lane, pipeline: &RenderPipelineState, draws: &[(u32, u32)]) -> Vec<[f32; 4]> {
    let target = lane.texture(PixelFormat::RGBA16Float, ONE, TextureUsage::RENDER_TARGET);
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    let [r, g, b, a] = CLEAR.map(f64::from);
    attachment.clear_color = ClearColor::new(r, g, b, a);
    attachment.store_action = StoreAction::Store;
    lane.run(&descriptor, |_, encoder, _| {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        for &(start, count) in draws {
            encoder
                .draw_primitives(PrimitiveType::Triangle, start, count, 1, 0)
                .unwrap();
        }
    })
    .unwrap();
    lane.read(&target, (0, 0), 8, BlitOption::None)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|texel| {
            let h = |i: usize| half(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            [h(0), h(1), h(2), h(3)]
        })
        .collect()
}

/// Overlapping triangles in a group on a color output the function does not read blend in their
/// order on every device, in one draw and across draws: Metal's answer.
#[test]
fn a_group_on_an_unread_color_output_blends_in_primitive_order() {
    let (messages, _cache, lane) = lane();
    let pipeline = pipeline(&lane);
    let expected = ordered();
    let name = lane.device.name();
    let interlock = lane
        .device
        .compile_device()
        .profile()
        .fragment()
        .pixel_interlock;
    for (what, draws) in [
        ("one draw", vec![(0, 3 * TRIANGLES)]),
        ("eight draws", (0..TRIANGLES).map(|k| (3 * k, 3)).collect()),
        (
            "two draws of four",
            vec![
                (0, 3 * TRIANGLES / 2),
                (3 * TRIANGLES / 2, 3 * TRIANGLES / 2),
            ],
        ),
    ] {
        let texels = drawn(&lane, &pipeline, &draws);
        assert_eq!(texels.len(), 16);
        for (index, texel) in texels.iter().enumerate() {
            assert_eq!(*texel, expected, "{what} on {name}: texel {index}");
        }
        eprintln!(
            "raster order {what} on {name} (pixel interlock {interlock}): {expected:?}, the \
             triangles' order"
        );
    }
    quiet(lane, &messages);
}
