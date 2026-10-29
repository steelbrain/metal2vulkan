//! A render state's first draw's `VkPipeline`, made inside the state's creation where Metal
//! compiles, on the device the cfg selects, under the validation layer: a state's creation makes
//! the pipeline of the key its first draw takes under Metal's defaults, so that draw makes none and
//! hands none to the pool, and draws as one made at the draw does; a draw of another key makes its
//! own; and the specialized notices of a pipeline made at creation wait for its first draw, which
//! announces them once. Nothing is kept from earlier draws: a draw of a key the state alone does
//! not predict makes its own. The pipelines are the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`, and owned functions reading their
//! vertices through a vertex descriptor (`LAYOUT_FUNCTIONS`).

use super::degrade_notices::{counts, on_device};
use super::pixel_samplers::metal_sampler;
use super::{
    dynamic_input_assembly, layer_reported_nothing_at_all, pipelines_settled,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::{llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBuffer, CommandBufferStatus, Degradation, Device, Library,
    LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeSet;
use std::path::Path;

/// The render target's side, and the color `colored_fragment` draws with its `RGBA8Unorm` bytes.
const SIDE: u64 = 8;
const DRAWN: [f32; 4] = [1.0, 0.2, 0.0, 1.0];
const DRAWN_TEXEL: [u8; 4] = [255, 51, 0, 255];
/// [`DRAWN`] as an `RGBA16Float` texel's bytes.
const HALF_DRAWN: [u8; 8] = [0x00, 0x3c, 0x66, 0x32, 0x00, 0x00, 0x00, 0x3c];

/// The owned fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/examples/runtime_examples.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The render pipeline state of the fixture's `colored_vertex` and `fragment`, one
/// `RGBA8Unorm` target.
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    state_with(device, fragment, |_| {})
}

/// The kinds of the precision notes [`state`]'s stages keep under some specialization only
/// (`StageModule::specialized_notes`), as the device's compile layer compiles its descriptor.
fn specialized_kinds(device: &Device, fragment: &str) -> BTreeSet<Degradation> {
    let compiled = device
        .compile_device()
        .compile_render_pipeline(&descriptor(device, fragment, |_| {}), &Budget::standard())
        .unwrap();
    compiled
        .stages
        .iter()
        .flat_map(|stage| &stage.specialized_notes)
        .map(|s| Degradation::from_precision_note(s.note))
        .collect()
}

/// [`state`], its target's attachment as `adjust` changes it.
fn state_with(
    device: &Device,
    fragment: &str,
    adjust: impl FnOnce(&mut ColorAttachmentDescriptor),
) -> RenderPipelineState {
    device
        .new_render_pipeline_state(
            &descriptor(device, fragment, adjust),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// [`state_with`]'s descriptor.
fn descriptor(
    device: &Device,
    fragment: &str,
    adjust: impl FnOnce(&mut ColorAttachmentDescriptor),
) -> RenderPipelineDescriptor {
    let library = library(device);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    adjust(&mut color);
    descriptor.color_attachments[0] = Some(color);
    descriptor
}

/// The pipelines made, and those handed to the pool, since `before`.
fn made_since(device: &Device, before: &WorkCounts) -> (u64, u64) {
    let now = device.work_counts();
    (
        now.variant_pipelines_made - before.variant_pipelines_made,
        now.pipelines_pooled - before.pipelines_pooled,
    )
}

/// The graphics pipeline libraries made, the pipelines linked from them, the pipelines made, and
/// those handed to the pool, since `before`.
fn linked_since(device: &Device, before: &WorkCounts) -> (u64, u64, u64, u64) {
    let now = device.work_counts();
    (
        now.pipeline_libraries_made - before.pipeline_libraries_made,
        now.pipelines_linked - before.pipelines_linked,
        now.variant_pipelines_made - before.variant_pipelines_made,
        now.pipelines_pooled - before.pipelines_pooled,
    )
}

/// Commit `command_buffer` and wait for it to complete.
fn finish(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Draw six vertices over the whole of a new `RGBA8Unorm` target as `primitive` with `state`,
/// binding no texture, and wait; the target's first texel.
fn draw(device: &Device, state: &RenderPipelineState, primitive: PrimitiveType) -> [u8; 4] {
    draw_with(device, state, primitive, None)
}

/// A shared 2D texture of `format`, `SIDE` texels a side, each texel `texel`'s bytes.
fn filled(device: &Device, format: PixelFormat, texel: &[u8]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = texel.repeat((SIDE * SIDE) as usize);
    let row = texel.len() as u64 * SIDE;
    texture
        .replace_region(
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
            &bytes,
            row,
            row * SIDE,
        )
        .unwrap();
    texture
}

/// [`draw`], binding `texture`, where given, at fragment texture 0.
fn draw_with(
    device: &Device,
    state: &RenderPipelineState,
    primitive: PrimitiveType,
    texture: Option<&Texture>,
) -> [u8; 4] {
    draw_bound(device, state, primitive, (texture, None))
}

/// [`draw_with`], binding the sampler, where given, at fragment sampler 0 too.
fn draw_bound(
    device: &Device,
    state: &RenderPipelineState,
    primitive: PrimitiveType,
    bound: (Option<&Texture>, Option<&SamplerState>),
) -> [u8; 4] {
    let texel = draw_into(
        device,
        state,
        primitive,
        bound,
        (PixelFormat::RGBA8Unorm, 4),
    );
    texel.try_into().unwrap()
}

/// [`draw_bound`] into a new target of `format`, of `texel` bytes a texel; the target's first
/// texel's bytes.
fn draw_into(
    device: &Device,
    state: &RenderPipelineState,
    primitive: PrimitiveType,
    (texture, sampler): (Option<&Texture>, Option<&SamplerState>),
    (format, texel): (PixelFormat, u64),
) -> Vec<u8> {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    // A list's two triangles, or a strip's covering triangles, over the whole target.
    let corners: [[f32; 2]; 6] = [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ];
    let positions: Vec<u8> = corners
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let colors: Vec<u8> = (0..corners.len())
        .flat_map(|_| DRAWN)
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let positions = device
        .new_buffer_with_bytes(&positions, ResourceOptions::shared())
        .unwrap();
    let colors = device
        .new_buffer_with_bytes(&colors, ResourceOptions::shared())
        .unwrap();
    let readback = device.new_buffer(texel, ResourceOptions::shared()).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    if let Some(texture) = texture {
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
    }
    if let Some(sampler) = sampler {
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
    }
    encoder.draw_primitives(primitive, 0, 6, 1, 0).unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(1, 1, 1),
        &readback,
        0,
        texel,
        texel,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    finish(&command_buffer);
    let contents = readback.contents().unwrap();
    // SAFETY: the buffer's contents hold its `texel` bytes, and the command buffer that wrote
    // them completed.
    unsafe { std::slice::from_raw_parts(contents.cast::<u8>().as_ptr(), texel as usize) }.to_vec()
}

/// A state's creation makes its first draw's pipeline: the draw makes none and hands none to the
/// pool, and draws what a pipeline made at the draw draws; a draw of another topology class,
/// another key, makes its own at the draw. A strip is of the triangles' class: where the device
/// makes the input assembly dynamic it takes the pipeline made at creation; elsewhere it is another
/// key, made at its draw.
#[test]
fn a_state_s_first_draw_takes_the_pipeline_made_at_its_creation() {
    on_device(|device, _| {
        let before = device.work_counts();
        let state = state(device, "colored_fragment");
        assert_eq!(made_since(device, &before), (1, 0), "made at creation");
        let created = device.work_counts();
        assert_eq!(draw(device, &state, PrimitiveType::Triangle), DRAWN_TEXEL);
        assert_eq!(
            made_since(device, &created),
            (0, 0),
            "the first draw found its pipeline made"
        );
        let drawn = device.work_counts();
        assert_eq!(
            draw(device, &state, PrimitiveType::TriangleStrip),
            DRAWN_TEXEL
        );
        let dynamic = dynamic_input_assembly(device);
        assert_eq!(
            made_since(device, &drawn).0,
            u64::from(!dynamic),
            "a strip's pipeline, made at its draw unless the topology is dynamic ({dynamic})"
        );
    });
}

/// The specialized notices of a pipeline made at its state's creation wait for its first draw:
/// the creation announces none of them, the first draw (which makes no pipeline) announces them,
/// each once, and a second draw announces nothing more. The pipeline is the predicted one: a draw
/// of `sampled_fragment`, whose `half` sample's NaN is the device's decode where the format bound
/// there may hold one (`PrecisionNote::DeviceNanEncoding`, a test of the bound format, which no
/// pipeline's key carries). Each draw binds an `RGBA16Float` texture, which takes the pipeline
/// made at creation (`a_draw_binding_a_texture_takes_the_pipeline_made_at_creation`); every kind
/// the first draw announces is one the stage specializes ([`specialized_kinds`]). On a device that
/// samples through Metal's own texture unit the creation makes the predicted key's general form,
/// and hands the predicted key itself to the pool right after (`pipeline/render/residual.rs`):
/// both counted once the pool settles, and neither announcing anything before a draw.
#[test]
fn a_pipeline_made_at_creation_announces_at_its_first_draw() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let nans = Degradation::from_precision_note(PrecisionNote::DeviceNanEncoding);
        let waits = specialized_kinds(&device, "sampled_fragment");
        assert!(
            waits.contains(&nans),
            "the stage's specialized notes: {waits:?}"
        );
        let wide = filled(&device, PixelFormat::RGBA16Float, &HALF_DRAWN);
        drop(messages.take_notices());
        let before = device.work_counts();
        let state = state(&device, "sampled_fragment");
        pipelines_settled(&device);
        let metal = metal_sampler(&device);
        assert_eq!(
            made_since(&device, &before),
            (1 + u64::from(metal), 0),
            "the predicted key's pipeline, made at creation, and its general form where the \
             device samples through Metal's texture unit ({metal})"
        );
        let at_creation = counts(&messages.take_notices());
        let created = device.work_counts();
        draw_with(&device, &state, PrimitiveType::Triangle, Some(&wide));
        assert_eq!(
            made_since(&device, &created),
            (0, 0),
            "the first draw found its pipeline made"
        );
        let at_draw = counts(&messages.take_notices());
        assert!(
            at_draw.contains_key(&nans),
            "the first draw announced the pipeline's notices: {at_draw:?} (creation: \
             {at_creation:?})"
        );
        let unspecialized: Vec<_> = at_draw
            .keys()
            .filter(|kind| !waits.contains(kind))
            .collect();
        assert!(
            unspecialized.is_empty(),
            "the first draw announced what no specialization keeps: {unspecialized:?}"
        );
        let early: Vec<_> = at_draw
            .keys()
            .filter(|kind| at_creation.contains_key(kind))
            .collect();
        assert!(early.is_empty(), "announced at creation too: {early:?}");
        assert!(
            at_draw.values().all(|&n| n == 1),
            "announced more than once: {at_draw:?}"
        );
        draw_with(&device, &state, PrimitiveType::Triangle, Some(&wide));
        assert_eq!(
            counts(&messages.take_notices()),
            Default::default(),
            "a second draw announced again"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A state's creation predicts the draw that binds no texture: nothing it sees names a texture a
/// draw binds. A draw of `sampled_fragment` binding nothing takes the pipeline made at creation,
/// and so do draws binding an `RGBA8Unorm` or an `RGBA16Float` texture, each sampling its texels:
/// the class of the format the slot binds reaches the function as a push field, not as the
/// pipeline's specialization. On a device that samples through Metal's own texture unit the
/// creation makes the predicted key's general form too (`pipeline/render/residual.rs`), and the
/// predicted key, made on the pool right after, is the one these draws take once the pool settles:
/// each class is plain and measured, as the unbound slot the prediction binds.
#[test]
fn a_draw_binding_a_texture_takes_the_pipeline_made_at_creation() {
    on_device(|device, _| {
        let metal = metal_sampler(device);
        let before = device.work_counts();
        let state = state(device, "sampled_fragment");
        pipelines_settled(device);
        assert_eq!(
            made_since(device, &before),
            (1 + u64::from(metal), 0),
            "made at creation, with the general form where the device samples through Metal's \
             texture unit ({metal})"
        );
        let created = device.work_counts();
        draw(device, &state, PrimitiveType::Triangle);
        assert_eq!(
            made_since(device, &created),
            (0, 0),
            "the draw binding nothing found its pipeline made"
        );
        let source = filled(device, PixelFormat::RGBA8Unorm, &DRAWN_TEXEL);
        let drawn = device.work_counts();
        assert_eq!(
            draw_with(device, &state, PrimitiveType::Triangle, Some(&source)),
            DRAWN_TEXEL
        );
        let wide = filled(device, PixelFormat::RGBA16Float, &HALF_DRAWN);
        assert_eq!(
            draw_with(device, &state, PrimitiveType::Triangle, Some(&wide)),
            DRAWN_TEXEL
        );
        assert_eq!(
            made_since(device, &drawn).0,
            0,
            "draws binding an 8-bit RGBA and a half-float texture"
        );
        let after = device.work_counts();
        assert_eq!(
            after.creation_pipelines_drawn - before.creation_pipelines_drawn,
            1,
            "the pipeline made at creation drew"
        );
        assert_eq!(
            after.residual_form_draws, before.residual_form_draws,
            "no draw took the general form"
        );
    });
}

/// A `BC4RUnorm` texture of `side` texels square, every block of it `block`'s 8 bytes.
fn bc4_filled(device: &Device, side: u64, block: [u8; 8]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BC4RUnorm, side, side, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let blocks = side / 4;
    let bytes = block.repeat((blocks * blocks) as usize);
    texture
        .replace_region(
            Region::new_2d(0, 0, side, side),
            0,
            0,
            &bytes,
            8 * blocks,
            0,
        )
        .unwrap();
    texture
}

/// A draw binding a texture of a class whose `half` conversion is not measured, a `BC4RUnorm` one,
/// draws at once with the pipeline its state's creation made. On a device that samples through
/// Metal's own texture unit the class keys apart from the prediction (`SpecKey::UnmeasuredTexels`
/// 1, where the unbound slot the prediction binds has 0), so the draw takes its key's general form,
/// the one the creation made (`pipeline/render/residual.rs`), and hands its own key's making to
/// the pool, which nothing waits for; once made, a second draw takes it. Elsewhere the class is a
/// push field alone, and both draws take the predicted key. Each draws the blocks' red, 1.
#[test]
fn a_draw_of_an_unmeasured_class_takes_the_general_form_its_creation_made() {
    on_device(|device, _| {
        let metal = metal_sampler(device);
        let before = device.work_counts();
        let state = state(device, "sampled_fragment");
        let created = pipelines_settled(device);
        assert_eq!(
            made_since(device, &before),
            (1 + u64::from(metal), 0),
            "made at creation, with the general form where the device samples through Metal's \
             texture unit ({metal})"
        );
        // Both endpoints 255, every index 0: red 1, and Metal's green 0, blue 0 and alpha 1.
        let red = bc4_filled(device, SIDE, [0xff, 0xff, 0, 0, 0, 0, 0, 0]);
        const RED: [u8; 4] = [255, 0, 0, 255];
        assert_eq!(
            draw_with(device, &state, PrimitiveType::Triangle, Some(&red)),
            RED
        );
        let drawn = pipelines_settled(device);
        let metal_count = u64::from(metal);
        assert_eq!(
            (
                drawn.residual_form_draws - created.residual_form_draws,
                drawn.residual_form_makings - created.residual_form_makings,
                drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
            ),
            (metal_count, metal_count, 1),
            "the draw took the general form, made at creation, and began its own key's making \
             where the device samples through Metal's texture unit ({metal}); elsewhere the \
             predicted key"
        );
        assert_eq!(
            made_since(device, &created),
            (metal_count, metal_count),
            "its own key, made on the pool while the draw went on, where the class keys apart"
        );
        assert_eq!(
            draw_with(device, &state, PrimitiveType::Triangle, Some(&red)),
            RED
        );
        let again = device.work_counts();
        assert_eq!(
            (
                again.residual_form_draws - drawn.residual_form_draws,
                made_since(device, &drawn),
            ),
            (0, (0, 0)),
            "the second draw took its own key, made"
        );
    });
}

/// A fragment function of the fixture's `colored_vertex` outputs that samples `texture2d<float>`
/// slot 0 at its position over 8, at its implicit level, through runtime sampler slot 0.
const RUNTIME_SAMPLED_FRAGMENT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @runtime_sampled_fragment(<4 x float> %pos, <4 x float> %color, ptr addrspace(1) %t, ptr addrspace(2) %s) {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, <float 1.250000e-01, float 1.250000e-01>
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %v
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!0 = !{ptr @runtime_sampled_fragment, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.fragment_input", !"generated(5colorDv4_f)", !"air.flat", !"air.arg_type_name", !"float4", !"air.arg_name", !"color", !"air.arg_unused"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"source"}
!7 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;

/// The render pipeline state of the fixture's `colored_vertex` and
/// [`RUNTIME_SAMPLED_FRAGMENT`], one target of `attachment`.
fn runtime_sampled_state(device: &Device, attachment: PixelFormat) -> RenderPipelineState {
    let vertex = library(device)
        .new_function_with_name("colored_vertex")
        .unwrap();
    let fragment = device
        .new_library_with_air(&llvm_assemble_text(RUNTIME_SAMPLED_FRAGMENT).unwrap())
        .unwrap()
        .new_function_with_name("runtime_sampled_fragment")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A creation of a state with a runtime sampler slot predicts the draw that binds Metal's
/// default sampler there and nothing else: it makes that one pipeline, and a draw binding the
/// default sampler and no texture makes none and takes it.
#[test]
fn a_creation_makes_the_default_sampler_s_key() {
    on_device(|device, _| {
        let before = device.work_counts();
        let state = runtime_sampled_state(device, PixelFormat::RGBA8Unorm);
        assert_eq!(made_since(device, &before), (1, 0), "made at creation");
        let default = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let created = device.work_counts();
        draw_bound(
            device,
            &state,
            PrimitiveType::Triangle,
            (None, Some(&default)),
        );
        assert_eq!(
            made_since(device, &created),
            (0, 0),
            "the draw found its pipeline made at creation"
        );
        let drawn = device.work_counts();
        assert_eq!(
            drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
            1,
            "the pipeline made at creation drew"
        );
    });
}

/// Where the device links its render pipelines from graphics pipeline libraries
/// (`pipeline/libraries.rs`), a state's creation makes the four libraries of its first draw's
/// pipeline and links it; a second state of the same functions and pass whose attachment blends
/// with factors one and zero, which store the output as the unblended attachment does (the
/// fragment rounds it, and link gives such an attachment no Vulkan blend), makes no library and
/// links its pipeline from the first state's: its fragment function is not compiled again. Both
/// draw what the fragment computes. A device that makes its pipelines whole makes no library and
/// links none.
#[test]
fn states_of_one_fragment_and_pass_share_its_fragment_library() {
    on_device(|device, _| {
        let before = device.work_counts();
        let first = state(device, "colored_fragment");
        let (libraries, linked, made, pooled) = linked_since(device, &before);
        assert_eq!((made, pooled), (1, 0), "made at creation");
        let created = device.work_counts();
        let second = state_with(device, "colored_fragment", |color| {
            color.blending_enabled = true;
        });
        let again = linked_since(device, &created);
        match linked {
            0 => {
                assert_eq!(libraries, 0, "a device that makes its pipelines whole");
                assert_eq!(again, (0, 0, 1, 0), "the second state's, made whole");
            }
            _ => {
                assert_eq!(
                    (libraries, linked),
                    (4, 1),
                    "the first state's four libraries"
                );
                assert_eq!(
                    again,
                    (0, 1, 1, 0),
                    "the second state's pipeline linked from the first state's libraries"
                );
            }
        }
        assert_eq!(draw(device, &first, PrimitiveType::Triangle), DRAWN_TEXEL);
        assert_eq!(draw(device, &second, PrimitiveType::Triangle), DRAWN_TEXEL);
    });
}

/// A draw of another topology class takes another key, made at the draw; where the device links
/// from libraries it is linked from the state's fragment shader and fragment output libraries,
/// making only the libraries of the parts the class changes: its vertex input (the topology) and
/// its pre-rasterization (Metal biases depth for triangles only, `metal::state::depth_bias`). A
/// device that makes its pipelines whole makes it whole.
#[test]
fn a_draw_of_another_class_links_from_its_state_s_fragment_libraries() {
    on_device(|device, _| {
        let before = device.work_counts();
        let state = state(device, "colored_fragment");
        let links = device.work_counts().pipelines_linked > before.pipelines_linked;
        let created = device.work_counts();
        // Three lines over the target; what they cover is not what this test reads.
        draw(device, &state, PrimitiveType::Line);
        let (libraries, linked, made, pooled) = linked_since(device, &created);
        assert_eq!(
            (made, pooled),
            (1, 1),
            "the lines' pipeline, made at their draw"
        );
        match links {
            false => assert_eq!((libraries, linked), (0, 0), "made whole"),
            true => assert_eq!(
                (libraries, linked),
                (2, 1),
                "its vertex input and pre-rasterization libraries, the fragment's the state's"
            ),
        }
    });
}

/// Two functions of the shape a compositor's state takes: a vertex function reading a `float2`
/// position (attribute 0) and a `float4` color (attribute 1) through `[[stage_in]]`, writing the
/// position at depth 0 and the color, and a fragment function returning that color.
const LAYOUT_FUNCTIONS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float> }> @layout_vertex(<2 x float> %position, <4 x float> %color) {
  %extended = shufflevector <2 x float> %position, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %extended, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %with_position = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %clip, 0
  %output = insertvalue <{ <4 x float>, <4 x float> }> %with_position, <4 x float> %color, 1
  ret <{ <4 x float>, <4 x float> }> %output
}

define <4 x float> @layout_fragment(<4 x float> %position, <4 x float> %color) {
  ret <4 x float> %color
}

!air.vertex = !{!0}
!air.fragment = !{!7}
!0 = !{ptr @layout_vertex, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(5colorDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"position"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!7 = !{ptr @layout_fragment, !8, !10}
!8 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!10 = !{!11, !12}
!11 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!12 = !{i32 1, !"air.fragment_input", !"generated(5colorDv4_f)", !"air.flat", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
"#;

/// The bytes of one vertex of [`LAYOUT_FUNCTIONS`]' layout: its position, then its color.
const LAYOUT_STRIDE: u32 = 24;

/// [`DRAWN_TEXEL`] as a `BGRA8Unorm` texel's bytes.
const DRAWN_BGRA: [u8; 4] = [0, 51, 255, 255];

/// The render pipeline state of [`LAYOUT_FUNCTIONS`] drawing into one `BGRA8Unorm` target, both
/// attributes in buffer 0's layout, of `stride` bytes (`None`: `MTLBufferLayoutStrideDynamic`).
fn layout_state(device: &Device, stride: Option<u32>) -> RenderPipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(LAYOUT_FUNCTIONS).unwrap())
        .unwrap();
    let vertex = library.new_function_with_name("layout_vertex").unwrap();
    let fragment = library.new_function_with_name("layout_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let mut layouts = VertexDescriptor::new();
    layouts
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(stride));
    layouts.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
    );
    layouts.attributes.insert(
        1,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 8, 0),
    );
    descriptor.vertex_descriptor = Some(layouts);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// How a draw of [`layout_state`] binds its vertices at buffer 0.
#[derive(Clone, Copy, Debug)]
enum Vertices {
    /// A buffer at offset 0, without a stride of its own.
    Buffer,
    /// Inline bytes.
    Bytes,
    /// A buffer at offset 0 at this stride.
    Strided(u64),
}

/// Draw a list's two triangles over the whole of a new `BGRA8Unorm` target with `state`, each
/// vertex's color [`DRAWN`], the vertices bound as `vertices`, and wait; the target's first
/// texel's bytes.
fn draw_layout(device: &Device, state: &RenderPipelineState, vertices: Vertices) -> [u8; 4] {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let corners: [[f32; 2]; 6] = [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ];
    let bytes: Vec<u8> = corners
        .iter()
        .flat_map(|corner| corner.iter().chain(&DRAWN))
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(bytes.len(), corners.len() * LAYOUT_STRIDE as usize);
    let buffer = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let readback = device.new_buffer(4, ResourceOptions::shared()).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    match vertices {
        Vertices::Buffer => encoder.set_vertex_buffer(Some(&buffer), 0, 0).unwrap(),
        Vertices::Bytes => encoder.set_vertex_bytes(&bytes, 0).unwrap(),
        Vertices::Strided(stride) => encoder
            .set_vertex_buffer_with_stride(Some(&buffer), 0, stride, 0)
            .unwrap(),
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(1, 1, 1),
        &readback,
        0,
        4,
        4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    finish(&command_buffer);
    let contents = readback.contents().unwrap();
    // SAFETY: the buffer's contents hold its 4 bytes, and the command buffer that wrote them
    // completed.
    unsafe { std::slice::from_raw_parts(contents.cast::<u8>().as_ptr(), 4) }
        .try_into()
        .unwrap()
}

/// A state whose vertex function reads its vertices through a vertex descriptor's layout makes
/// its first draw's pipeline at its creation: every draw binds the layout's buffer, so the draw
/// predicted binds it, at an offset every attribute's format aligns to. A first draw binding a
/// buffer at offset 0 makes no pipeline and hands none to the pool, takes the one made at
/// creation and draws what the function computes; a draw binding the vertices as inline bytes
/// takes it too.
#[test]
fn a_state_with_a_vertex_descriptor_makes_its_first_draw_s_pipeline_at_creation() {
    on_device(|device, _| {
        let before = device.work_counts();
        let state = layout_state(device, Some(LAYOUT_STRIDE));
        assert_eq!(made_since(device, &before), (1, 0), "made at creation");
        let created = device.work_counts();
        assert_eq!(draw_layout(device, &state, Vertices::Buffer), DRAWN_BGRA);
        assert_eq!(
            made_since(device, &created),
            (0, 0),
            "the first draw found its pipeline made"
        );
        assert_eq!(
            device.work_counts().creation_pipelines_drawn - created.creation_pipelines_drawn,
            1,
            "the pipeline made at creation drew"
        );
        let drawn = device.work_counts();
        assert_eq!(draw_layout(device, &state, Vertices::Bytes), DRAWN_BGRA);
        assert_eq!(
            made_since(device, &drawn),
            (0, 0),
            "a draw of inline bytes took it too"
        );
    });
}

/// A layout of `MTLBufferLayoutStrideDynamic` is predicted bound at the stride that packs its
/// attributes: a first draw binding the buffer at that stride makes no pipeline and takes the
/// one made at creation, whether the device makes strides dynamic state or keys its pipelines on
/// them.
#[test]
fn a_dynamic_layout_s_packed_stride_takes_the_pipeline_made_at_creation() {
    on_device(|device, _| {
        let before = device.work_counts();
        let state = layout_state(device, None);
        assert_eq!(made_since(device, &before), (1, 0), "made at creation");
        let created = device.work_counts();
        assert_eq!(
            draw_layout(device, &state, Vertices::Strided(LAYOUT_STRIDE.into())),
            DRAWN_BGRA
        );
        assert_eq!(
            made_since(device, &created),
            (0, 0),
            "the first draw at the packed stride found its pipeline made"
        );
        assert_eq!(
            device.work_counts().creation_pipelines_drawn - created.creation_pipelines_drawn,
            1,
            "the pipeline made at creation drew"
        );
    });
}
