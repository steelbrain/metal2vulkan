//! A render state's creation predicts the `VkPipeline` its first draw takes
//! (`pipeline/predicted.rs`), and three shapes of draw take it where they made their own:
//!
//! - **A draw in a pass that resolves.** A creation cannot know whether its draws' passes resolve,
//!   and render pass compatibility ignores the resolves in a pass of one subpass, so a key holds
//!   none (`PassShape::keyed`): a draw in a four-sample pass that resolves its color takes the
//!   pipeline the creation made, against a pass that resolves nothing.
//! - **A draw of points under the point class.** Such a draw takes Metal's guard band where the
//!   device takes it (`render_state::guard_points`), a variant of its own, which the creation now
//!   predicts.
//! - **A multisampled blend into `BGRA10_XR`** in a pass that clears it, which holds the color's
//!   `half` shadow from that draw on (`render/extended_shadow.rs`): the variant that reads and
//!   writes the shadow, which the creation now predicts. A device whose pass fetches through
//!   input attachments holds no shadow, and its creation predicts none.
//!
//! Each reads one creation's makings and the draw's takings as deltas on the test's own device
//! (`WorkCounts::creation_pipelines_made` and `creation_pipelines_drawn`), once the device's
//! pipeline pool holds nothing, and asserts the creation made at least one, so a creation that
//! predicts nothing cannot pass by taking nothing.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/rt6_render_state.ll` (`llvm-as`): `rt6_full`, a
//! fullscreen triangle, `rt6_point`, a point of 4 pixels that writes `[[point_size]]`, and
//! `rt6_white`, each reading its depth from vertex buffer 0.

use super::blit_holds::{device, Scratch, WAIT};
use super::uniform_bind_offsets::idle;
use crate::{
    Device, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{
    BlendFactor, PixelFormat, PrimitiveTopologyClass, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The target's side, in pixels.
const SIDE: u64 = 8;

/// The fixture's render state of `vertex` and `rt6_white` into one color of `color`, at
/// `samples`, of the topology class `class`.
fn state(
    device: &Device,
    vertex: &str,
    color: ColorAttachmentDescriptor,
    samples: u32,
    class: PrimitiveTopologyClass,
) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/rt6_render_state.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    let vertex = library.new_function_with_name(vertex).unwrap();
    let fragment = library.new_function_with_name("rt6_white").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(color);
    descriptor.raster_sample_count = samples;
    descriptor.input_primitive_topology = class;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A render target of `format` at `samples`, or `None` where the device makes none.
fn target(device: &Device, format: PixelFormat, samples: u64) -> Option<Texture> {
    let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    if samples > 1 {
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples;
    }
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).ok()
}

/// Draw `primitive` once with `state` in `pass`, its vertex buffer 0 the depth 0.5, and wait.
fn draw(
    device: &Device,
    state: &RenderPipelineState,
    pass: &RenderPassDescriptor,
    primitive: PrimitiveType,
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    let depth: Vec<u8> = [0.5f32, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    encoder.set_vertex_bytes(&depth, 0).unwrap();
    let vertices = match primitive {
        PrimitiveType::Point => 1,
        _ => 3,
    };
    encoder
        .draw_primitives(primitive, 0, vertices, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The pipelines a creation made and its draw took, as `[made, drawn]`, from the counts before
/// the creation, after it and after the draw.
fn reach(before: &WorkCounts, created: &WorkCounts, drawn: &WorkCounts) -> [u64; 2] {
    [
        created.creation_pipelines_made - before.creation_pipelines_made,
        drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
    ]
}

/// Assert `what`'s creation made a pipeline and its draw took every one it made.
fn taken(what: &str, [made, drawn]: [u64; 2]) {
    assert!(made > 0, "{what}: the creation made no predicted pipeline");
    assert_eq!(
        drawn, made,
        "{what}: the draw took {drawn} of the {made} pipeline(s) its state's creation made, so \
         the creation predicted a key its first draw does not take"
    );
}

/// A four-sample draw in a pass that resolves its color takes the creation's pipeline: a key
/// holds no pass's resolves.
#[test]
fn a_draw_in_a_pass_that_resolves_takes_the_creation_s_pipeline() {
    let scratch = Scratch::new("creation-reach-resolve");
    let device = device(&scratch.0);
    let (Some(samples), Some(resolve)) = (
        target(&device, PixelFormat::RGBA8Unorm, 4),
        target(&device, PixelFormat::RGBA8Unorm, 1),
    ) else {
        eprintln!("SKIPPED: no four-sample RGBA8Unorm target here");
        return;
    };
    let before = idle(&device);
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    let state = state(
        &device,
        "rt6_full",
        color,
        4,
        PrimitiveTopologyClass::Unspecified,
    );
    let created = idle(&device);
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(samples);
    attachment.resolve_texture = Some(resolve);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::MultisampleResolve;
    draw(&device, &state, &pass, PrimitiveType::Triangle);
    let drawn = idle(&device);
    taken(
        "a draw in a pass that resolves",
        reach(&before, &created, &drawn),
    );
}

/// A point draw of a state of the point class takes the creation's pipeline, which is of the
/// variant that takes Metal's guard band where the device takes it.
#[test]
fn a_point_draw_of_the_point_class_takes_the_creation_s_pipeline() {
    let scratch = Scratch::new("creation-reach-points");
    let device = device(&scratch.0);
    let Some(target) = target(&device, PixelFormat::RGBA8Unorm, 1) else {
        eprintln!("SKIPPED: no RGBA8Unorm target here");
        return;
    };
    let before = idle(&device);
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    let state = state(
        &device,
        "rt6_point",
        color,
        1,
        PrimitiveTopologyClass::Point,
    );
    let created = idle(&device);
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    draw(&device, &state, &pass, PrimitiveType::Point);
    let drawn = idle(&device);
    taken(
        "a point draw of the point class",
        reach(&before, &created, &drawn),
    );
}

/// A four-sample source-over blend into a `BGRA10_XR` color the pass clears takes the creation's
/// pipeline, which is of the variant that holds the color's shadow where the device's pass holds
/// one.
#[test]
fn a_multisampled_extended_range_blend_takes_the_creation_s_pipeline() {
    let scratch = Scratch::new("creation-reach-xr");
    let device = device(&scratch.0);
    let Some(target) = target(&device, PixelFormat::BGRA10XR, 4) else {
        eprintln!("SKIPPED: no four-sample BGRA10_XR target here");
        return;
    };
    let before = idle(&device);
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA10XR);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let state = state(
        &device,
        "rt6_full",
        color,
        4,
        PrimitiveTopologyClass::Unspecified,
    );
    let created = idle(&device);
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    draw(&device, &state, &pass, PrimitiveType::Triangle);
    let drawn = idle(&device);
    taken(
        "a multisampled blend into BGRA10_XR",
        reach(&before, &created, &drawn),
    );
}
