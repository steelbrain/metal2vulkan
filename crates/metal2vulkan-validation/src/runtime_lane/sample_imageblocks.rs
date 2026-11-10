//! Device tests of a fragment imageblock in passes of one and four samples, and beside a depth
//! attachment, on the device the cfg selects, under the validation layer, over the owned fixture
//! `ibs_sample_imageblocks`: each pass is 8x8, an `R32Float` color (resolved at four samples) and
//! the cell's float, which `ibs_read` returns as red. Against Metal on the M3 Max (macOS 27.0.1,
//! a probe of the same functions):
//!
//! - At four samples the imageblock is per sample: after a write of 1 over the whole target, an
//!   add of 2 through a triangle that covers part of some pixels, and a read, a pixel resolves to
//!   `1 + 2c`, `c` the share of its samples the triangle covers (`ibs_plain`'s resolve); and
//!   `11 + 2c` after a further add of 10 over the whole target. A write returning the sample mask
//!   0b0101 keeps samples 0 and 2: 4 everywhere.
//! - A fragment its depth test fails drops its write; one that passes writes: after a write of 1,
//!   a write of 7 failing over the whole target and a write of 5 passing through the triangle,
//!   `1 + 4c`, at one sample and four, and so for a function under `[[early_fragment_tests]]`
//!   (whose draws test in the shader at four samples on a driver that drops an early-tests
//!   shader's image writes there). An add that discards column 7, failing with 2 and passing with
//!   4, leaves 5 but 1 in column 7.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::{Feature, Needs};
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

/// The triangle over the whole target, and the one that covers part of some pixels.
const WHOLE: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
const PART: [f32; 6] = [-1.0, -1.0, 1.0, -0.3, -1.0, 1.0];

/// One draw: the fragment function, the value in its buffer 0, its triangle, its depth, and
/// whether it tests depth `Less` (else `Always`); no draw writes depth.
struct Draw {
    fragment: &'static str,
    value: f32,
    corners: [f32; 6],
    z: f32,
    less: bool,
}

const fn draw(fragment: &'static str, value: f32, corners: [f32; 6], z: f32, less: bool) -> Draw {
    Draw {
        fragment,
        value,
        corners,
        z,
        less,
    }
}

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine/ibs_sample_imageblocks.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// A pipeline of `ibs_vertex` and `fragment` at `samples` over the `R32Float` color (written by
/// the functions that return a color) and, `depth`, a `Depth32Float` depth.
fn pipeline(
    device: &Device,
    library: &Library,
    fragment: &str,
    samples: u32,
    depth: bool,
) -> Result<RenderPipelineState, Error> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("ibs_vertex"));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::R32Float);
    if !matches!(fragment, "ibs_read" | "ibs_plain") {
        color.write_mask = ColorWriteMask::NONE;
    }
    descriptor.color_attachments[0] = Some(color);
    if depth {
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
    }
    descriptor.raster_sample_count = samples;
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A render target of `pf`, SIDE wide and high, of `samples`, shared where `shared`.
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

/// The red of each pixel after a pass at `samples` (beside a depth cleared to 0.5 where `depth`)
/// runs `draws` through `pipelines` (one per draw), resolved at more than one sample.
fn run(
    device: &Device,
    pipelines: &[RenderPipelineState],
    draws: &[Draw],
    samples: u64,
    depth: bool,
) -> Vec<f32> {
    let resolved = texture(device, PixelFormat::R32Float, 1, true);
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.load_action = LoadAction::Clear;
    if samples > 1 {
        color.texture = Some(texture(device, PixelFormat::R32Float, samples, false));
        color.store_action = StoreAction::MultisampleResolve;
        color.resolve_texture = Some(resolved.clone());
    } else {
        color.texture = Some(resolved.clone());
        color.store_action = StoreAction::Store;
    }
    if depth {
        pass.depth_attachment.texture =
            Some(texture(device, PixelFormat::Depth32Float, samples, false));
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.5;
        pass.depth_attachment.store_action = StoreAction::DontCare;
    }
    pass.tile_width = 16;
    pass.tile_height = 16;
    pass.imageblock_sample_length = 32;
    let state = |less: bool| {
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = match less {
            true => CompareFunction::Less,
            false => CompareFunction::Always,
        };
        state.depth_write_enabled = false;
        device.new_depth_stencil_state(&state).unwrap()
    };
    let (always, less) = (state(false), state(true));
    let floats =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    let mut held = Vec::new();
    for (draw, pipeline) in draws.iter().zip(pipelines) {
        let corners = buffer_with(device, &floats(&draw.corners)).unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        if depth {
            let state = if draw.less { &less } else { &always };
            encoder.set_depth_stencil_state(state).unwrap();
        }
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_vertex_bytes(&draw.z.to_le_bytes(), 1).unwrap();
        encoder
            .set_fragment_bytes(&draw.value.to_le_bytes(), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.push(corners);
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drop(held);
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

/// Metal's answers of the module docs, at one sample and four, against each pixel's coverage by
/// the partial triangle.
#[test]
fn an_imageblock_is_per_sample_and_tested_as_metal_s() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let library = library(&device);
        let mut answers = Vec::new();
        for samples in [1u64, 4] {
            // The share of each pixel's samples the partial triangle covers.
            let plain = [draw("ibs_plain", 0.0, PART, 0.25, false)];
            let made = pipeline(&device, &library, "ibs_plain", samples as u32, false).unwrap();
            let cover = run(&device, &[made], &plain, samples, false);
            let partial = cover.iter().filter(|&&c| c > 0.0 && c < 1.0).count();
            assert!(
                samples == 1 || partial > 0,
                "no partly covered pixel on {name}"
            );
            let per_pixel = |f: &dyn Fn(usize, f32) -> f32| -> Vec<f32> {
                cover.iter().enumerate().map(|(i, &c)| f(i, c)).collect()
            };
            let mut cases: Vec<(&str, bool, Vec<Draw>, Vec<f32>)> = vec![
                (
                    "a partial add",
                    false,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_add", 2.0, PART, 0.25, false),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    per_pixel(&|_, c| 1.0 + 2.0 * c),
                ),
                (
                    "a partial add, then a whole one",
                    false,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_add", 2.0, PART, 0.25, false),
                        draw("ibs_add", 10.0, WHOLE, 0.25, false),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    per_pixel(&|_, c| 11.0 + 2.0 * c),
                ),
                (
                    "a write under a failing and a passing depth test",
                    true,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_write", 7.0, WHOLE, 0.75, true),
                        draw("ibs_write", 5.0, PART, 0.25, true),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    per_pixel(&|_, c| 1.0 + 4.0 * c),
                ),
                (
                    "a discarding add under a failing and a passing depth test",
                    true,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_add_discard", 2.0, WHOLE, 0.75, true),
                        draw("ibs_add_discard", 4.0, WHOLE, 0.25, true),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    per_pixel(&|i, _| if i as u64 % SIDE == 7 { 1.0 } else { 5.0 }),
                ),
                (
                    "an early-tests write under a failing and a passing depth test",
                    true,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_write_early", 7.0, PART, 0.75, true),
                        draw("ibs_write_early", 5.0, PART, 0.25, true),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    per_pixel(&|_, c| 1.0 + 4.0 * c),
                ),
            ];
            if samples > 1 {
                cases.push((
                    "a write returning the mask 0b0101",
                    false,
                    vec![
                        draw("ibs_write", 1.0, WHOLE, 0.25, false),
                        draw("ibs_write_masked", 7.0, WHOLE, 0.25, false),
                        draw("ibs_read", 0.0, WHOLE, 0.25, false),
                    ],
                    vec![4.0; (SIDE * SIDE) as usize],
                ));
            }
            for (what, depth, draws, want) in cases {
                let mut pipelines = Vec::new();
                let mut refused = None;
                for d in &draws {
                    match pipeline(&device, &library, d.fragment, samples as u32, depth) {
                        Ok(p) => pipelines.push(p),
                        Err(Error::Unsupported(u)) => {
                            refused = Some((d.fragment, u));
                            break;
                        }
                        Err(error) => panic!("{what} at {samples}x on {name}: {error}"),
                    }
                }
                if let Some((fragment, u)) = refused {
                    let lacks =
                        u.feature == Feature::Imageblocks && matches!(u.needs, Needs::AnyOf(_));
                    assert!(lacks, "{what} at {samples}x on {name}: {fragment}: {u:?}");
                    answers.push(format!("{what} {samples}x refused, {u:?}"));
                    continue;
                }
                let got = run(&device, &pipelines, &draws, samples, depth);
                assert_eq!(got, want, "{what} at {samples}x on {name}");
                answers.push(format!("{what} {samples}x Metal's"));
            }
        }
        eprintln!("sample imageblocks on {name}: {}", answers.join("; "));
    }
    layer_reported_nothing_at_all(&messages);
}
