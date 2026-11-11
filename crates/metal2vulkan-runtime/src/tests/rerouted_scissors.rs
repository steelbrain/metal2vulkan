//! A draw recorded again into another render pass instance draws through the scissor rectangles
//! it was drawn through, not the encoder's latest (`render/reroute.rs`'s `Held`, which keeps them
//! beside the rest of the encoder's state, `ScissorRects` living apart from it).
//!
//! A pass of a `BGRA10_XR` color it clears keeps its instance's draws for recording again. A draw
//! that blends into the color an earlier draw of the instance wrote starts the color's `half`
//! shadow there, and the instance's draws record again first into an instance that holds it, in
//! its place (`render/extended_shadow.rs`). Here the pass's first draw writes white through a
//! small scissor rectangle ([`FIRST`]); the rectangle is then set to the render area, and the
//! second draw blends into the color with the blend color, 0.5, as both factors, which starts the
//! shadow and so records the first draw again. The target then holds 1.0 inside the first
//! rectangle and 0.5 elsewhere (codes 894 and 639, every channel); a first draw recorded again
//! through the encoder's latest rectangle writes white everywhere, so the target holds 1.0
//! everywhere.
//!
//! That the reroute ran is in the counts: the pass leaves one more instance unrun than the same
//! pass whose second draw writes without blending (`WorkCounts::unrun_instances_ended`), and the
//! device announces no `extended-range-blends-stored-codes`, which a shadow that did not start
//! would (the blend then reads the codes, which hold the same values here). A device whose pass
//! holds no such shadow (`framebuffer_fetch::holds_extended_shadows`), or that has no framebuffer
//! fetch form and refuses the blending state as `FramebufferFetch`, prints `SKIPPED` and stops.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/rt6_render_state.ll` (`llvm-as`): `rt6_full`, a
//! fullscreen triangle reading its depth from vertex buffer 0, and `rt6_white`.

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use super::encoder_holds::{bytes_of, library};
use super::uniform_bind_offsets::idle;
use crate::framebuffer_fetch;
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, Error, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The target's side, in pixels.
const SIDE: u64 = 16;

/// The first draw's scissor rectangle: x, y, width, height.
const FIRST: [u64; 4] = [2, 3, 5, 4];

/// A `BGRA10_XR` texel's bytes: four 16-bit fields.
const TEXEL_BYTES: u64 = 8;

/// The `BGRA10_XR` codes of 1.0 and 0.5 (`510 v + 384`).
const ONE: u16 = 894;
const HALF: u16 = 639;

/// The fixture's state of `rt6_full` and `rt6_white` into one `BGRA10_XR` color, blending into
/// it with the blend color as both factors where `blends`, or the device's refusal of it.
fn state(device: &Device, blends: bool) -> Result<RenderPipelineState, Error> {
    let library = library(device, "engine/rt6_render_state.ll");
    let function = |name| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rt6_full"));
    descriptor.fragment = Some(function("rt6_white"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA10XR);
    if blends {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::BlendColor;
        color.source_alpha_blend_factor = BlendFactor::BlendColor;
        color.destination_rgb_blend_factor = BlendFactor::BlendColor;
        color.destination_alpha_blend_factor = BlendFactor::BlendColor;
    }
    descriptor.color_attachments[0] = Some(color);
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A private `SIDE`x`SIDE` `BGRA10_XR` render target.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA10XR, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// One pass that clears `target` to zero and draws `first` through the [`FIRST`] rectangle, then
/// `second` through the render area, the blend color 0.5, and a blit of the target into `read`:
/// the codes each texel holds after it (blue, green, red, alpha), row after row.
fn run(
    queue: &CommandQueue,
    (first, second): (&RenderPipelineState, &RenderPipelineState),
    target: &Texture,
    read: &Buffer,
) -> Vec<[u16; 4]> {
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    let depth = bytes_of(&[0.5f32, 0.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&depth, 0).unwrap();
    encoder.set_blend_color(0.5, 0.5, 0.5, 0.5);
    let [x, y, width, height] = FIRST;
    encoder.set_render_pipeline_state(first).unwrap();
    encoder
        .set_scissor_rect(ScissorRect::new(x, y, width, height))
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder
        .set_scissor_rect(ScissorRect::new(0, 0, SIDE, SIDE))
        .unwrap();
    encoder.set_render_pipeline_state(second).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        read,
        0,
        SIDE * TEXEL_BYTES,
        SIDE * SIDE * TEXEL_BYTES,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    // SAFETY: the read buffer's mapped bytes, which no command uses now.
    let bytes = unsafe {
        std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), read.length() as usize)
    };
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|texel| {
            std::array::from_fn(|c| u16::from_le_bytes([texel[2 * c], texel[2 * c + 1]]) >> 6)
        })
        .collect()
}

/// Whether pixel `(x, y)` lies inside the [`FIRST`] rectangle.
fn inside_first(x: u64, y: u64) -> bool {
    let [fx, fy, width, height] = FIRST;
    (fx..fx + width).contains(&x) && (fy..fy + height).contains(&y)
}

/// The instances a run left unrun, from the counts before it and after it.
fn unrun(before: &WorkCounts, after: &WorkCounts) -> u64 {
    after.unrun_instances_ended - before.unrun_instances_ended
}

/// A pass's first draw, recorded again into the instance that holds its color's shadow when the
/// second draw starts it, draws through its own scissor rectangle, not the one the encoder set
/// after it.
#[test]
fn a_draw_recorded_again_draws_through_its_own_scissor_rectangle() {
    let scratch = Scratch::new("rerouted-scissors");
    let heard = Heard::default();
    let device = device(
        &scratch.0,
        &[Degradation::ExtendedRangeBlendsStoredCodes],
        &heard,
    );
    let profile = device.compile_device().profile();
    let fragment = profile.fragment();
    let fetches = fragment.rasterization_order_color_access || fragment.pixel_interlock;
    let writes = state(&device, false).unwrap();
    let blends = match state(&device, true) {
        Ok(state) => state,
        Err(Error::Unsupported(refusal))
            if !fetches && refusal.feature == Feature::FramebufferFetch =>
        {
            eprintln!(
                "SKIPPED on {}: no framebuffer fetch form, the blending state refused: {refusal}",
                device.name()
            );
            return;
        }
        Err(error) => panic!("the BGRA10_XR blend's state was refused: {error}"),
    };
    let inputs = framebuffer_fetch::through_inputs(profile);
    if !framebuffer_fetch::holds_extended_shadows(profile, inputs, 1) {
        eprintln!(
            "SKIPPED on {}: its pass holds no extended-range shadow, so no draw records again",
            device.name()
        );
        return;
    }
    let target = target(&device);
    let read = device
        .new_buffer(SIDE * SIDE * TEXEL_BYTES, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    // A first pass makes every pipeline the passes measured draw with: a draw whose pipeline the
    // device's pool is making records into a piece of its own, and an instance of such pieces is
    // not counted when a reroute leaves it.
    run(&queue, (&writes, &blends), &target, &read);
    let before = idle(&device);
    run(&queue, (&writes, &writes), &target, &read);
    let between = idle(&device);
    heard.lock().unwrap().clear();
    let codes = run(&queue, (&writes, &blends), &target, &read);
    let after = idle(&device);
    let stored: Vec<String> = (heard.lock().unwrap().iter())
        .filter(|notice| notice.degradation == Degradation::ExtendedRangeBlendsStoredCodes)
        .map(|notice| notice.reason.clone())
        .collect();
    let [plain, rerouted] = [unrun(&before, &between), unrun(&between, &after)];
    eprintln!(
        "on {}: instances left unrun: {plain} by the pass that writes twice, {rerouted} by the \
         pass that blends; extended-range-blends-stored-codes announced {} time(s)",
        device.name(),
        stored.len()
    );
    assert!(
        stored.is_empty(),
        "the blend read the stored codes, so its color's shadow did not start: {stored:?}"
    );
    assert_eq!(
        rerouted,
        plain + 1,
        "the pass that blends left {rerouted} instance(s) unrun and the one that writes twice \
         {plain}: the blend's shadow start did not record the first draw again into an instance \
         in the current one's place"
    );
    let mut leaked = Vec::new();
    let mut wrong = Vec::new();
    for (at, texel) in (0u64..).zip(&codes) {
        let (x, y) = (at % SIDE, at / SIDE);
        let first = inside_first(x, y);
        let expected = match first {
            true => ONE,
            false => HALF,
        };
        if texel.iter().all(|&code| code == expected) {
            continue;
        }
        match (first, texel.iter().all(|&code| code == ONE)) {
            (false, true) => leaked.push((x, y)),
            _ => wrong.push(((x, y), *texel)),
        }
    }
    assert!(
        leaked.is_empty(),
        "the first draw, recorded again, wrote {} pixel(s) outside its scissor rectangle {FIRST:?} \
         (x, y, width, height), as if through the encoder's later one: {:?}",
        leaked.len(),
        &leaked[..leaked.len().min(8)]
    );
    assert!(
        wrong.is_empty(),
        "{} texel(s) hold other codes than {ONE} inside {FIRST:?} and {HALF} outside it \
         (blue, green, red, alpha): {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)]
    );
}
