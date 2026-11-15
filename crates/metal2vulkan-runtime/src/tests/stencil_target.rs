//! A `Stencil8` render target drawn into and read back, on every device: on one without `S8_UINT`
//! its image is a combined depth/stencil one (`FormatEmulation::StencilInCombined`: Intel's UHD
//! 630 under its Windows driver), whose depth aspect no texture of the program's holds, loaded and
//! stored `DONT_CARE` (`render_target.rs`).
//!
//! - **A draw stores its stencil.** A pass clears a 16 x 16 `Stencil8` texture to 0x5a and draws a
//!   triangle over the whole target, scissored to its left half, whose stencil operation replaces
//!   the stencil with the reference 0xa5, and stores it: a copy out of the texture reads 0xa5 in
//!   each texel of the left half and 0x5a in each of the right, a byte a texel.
//! - **A later pass keeps it.** A pass that loads the stencil and stores it, drawing nothing,
//!   leaves the same bytes: the hidden depth aspect's `DONT_CARE` disturbs no stencil byte.
//!
//! Printed (`STENCIL-TARGET`): the device and the format of the texture's image.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, LoadAction,
    Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The texture's side.
const SIDE: u64 = 16;
/// The stencil the pass clears to, and the one its draw writes.
const CLEARED: u8 = 0x5a;
const DRAWN: u8 = 0xa5;

/// Commit `command_buffer` and wait for it to complete.
fn commit(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A new `Private` `SIDE` square texture of `format` that renders.
fn target(device: &Device, format: PixelFormat) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A state that draws a triangle over the whole target into `RGBA8Unorm` beside a `Stencil8`
/// attachment.
fn draw_state(device: &Device) -> RenderPipelineState {
    let scout = library(device, "engine/format_scout.ll");
    let stage = |name: &str| {
        let function = scout.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(stage("scout_vertex"));
    descriptor.fragment = Some(stage("scout_fragment_f"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.stencil_format = Some(PixelFormat::Stencil8);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The bytes of `stencil`, a byte a texel, copied out by a command buffer of `queue`.
fn stencil_bytes(queue: &CommandQueue, stencil: &Texture) -> Vec<u8> {
    let out: Buffer = (queue.device())
        .new_buffer(SIDE * SIDE, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        stencil,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE,
        SIDE * SIDE,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    commit(&command_buffer);
    let at = out.contents().unwrap();
    // SAFETY: a `Shared` buffer's contents span its length, and no command uses them now.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), (SIDE * SIDE) as usize) }.to_vec()
}

#[test]
fn a_stencil8_target_stores_what_a_draw_writes_and_a_later_pass_keeps_it() {
    let scratch = Scratch::new("stencil-target");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let stencil = target(&device, PixelFormat::Stencil8);
    let color = target(&device, PixelFormat::RGBA8Unorm);
    let format = stencil.image_plan().unwrap().format;
    println!(
        "STENCIL-TARGET on {}: Stencil8 held in {format:?}",
        device.name()
    );
    let mut replace = DepthStencilDescriptor::default();
    for face in [
        &mut replace.front_face_stencil,
        &mut replace.back_face_stencil,
    ] {
        face.stencil_compare_function = CompareFunction::Always;
        face.depth_stencil_pass_operation = StencilOperation::Replace;
    }
    let replace = device.new_depth_stencil_state(&replace).unwrap();
    let state = draw_state(&device);
    // Cleared, drawn into over the left half, and stored.
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(color.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::DontCare;
    let attachment = &mut pass.stencil_attachment;
    attachment.texture = Some(stencil.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_stencil = u32::from(CLEARED);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_depth_stencil_state(&replace).unwrap();
    encoder.set_stencil_reference_value(u32::from(DRAWN));
    let half = ScissorRect {
        x: 0,
        y: 0,
        width: SIDE / 2,
        height: SIDE,
    };
    encoder.set_scissor_rect(half).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    commit(&command_buffer);
    let expected: Vec<u8> = (0..SIDE * SIDE)
        .map(|at| match at % SIDE < SIDE / 2 {
            true => DRAWN,
            false => CLEARED,
        })
        .collect();
    assert!(
        stencil_bytes(&queue, &stencil) == expected,
        "the drawn half and the cleared half"
    );
    // Loaded and stored with no draw: the same bytes.
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.stencil_attachment;
    attachment.texture = Some(stencil.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.render_command_encoder(&pass).unwrap())
        .end_encoding()
        .unwrap();
    commit(&command_buffer);
    assert!(
        stencil_bytes(&queue, &stencil) == expected,
        "a later pass keeps them"
    );
}
