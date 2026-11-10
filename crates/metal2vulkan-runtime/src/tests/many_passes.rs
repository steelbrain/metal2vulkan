//! One command buffer of many render passes, each loading and storing a color target and blending
//! a rectangle over it beside a `Depth32Float` depth attachment it clears and stores (the shape of
//! the consumer's checked frames, `linear_runs.rs`), as Metal runs it, in Vulkan command buffers
//! that grow with the passes by one each: the pass's first instance records inline in the
//! encoder's own primary command buffer, beside its depth attachment (`render/inline_first.rs`),
//! not in a secondary command buffer the primary executes; and past the first
//! `PLANNED_BARRIERS` encoders that one begins with a barrier from everything before, where the
//! submission would begin a command buffer of barriers before each (`encoder.rs`).
//!
//! On ANV below Xe-HP (Mesa 26.2.6, `anv_device.c`) each Vulkan command buffer begun takes a
//! 64 KiB block of the device's 1 GiB binding-table pool until it is begun again, reset or
//! freed: at most 16,384 at once (`spares.rs`). On ms01's Iris Xe, with a primary and a
//! secondary a pass, one command buffer of such passes failed with `OutOfMemory(Device)` at the
//! end of its 8,087th pass; with the pass inline and a command buffer of barriers before each,
//! one of 8,300 passes encoded and its submission, past the pool, crashed in the driver as it
//! recorded the targets' first clears (`texture/first_move.rs`), where one of 8,000 ran.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_solid` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::encoder_holds::bytes_of;
use super::merge_scratch::{blending, blit, counts_alone, drawn, read, settled, H, TEXEL};
use crate::encoder::PLANNED_BARRIERS;
use crate::{
    BlitOption, CommandBuffer, CommandQueue, DepthStencilState, Device, LoadAction, Origin,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::time::Duration;

/// The passes after the clear: past the 8,086 such passes a command buffer held on ms01's Iris Xe
/// with a primary and a secondary each, and the 8,000 it held with a command buffer of barriers
/// before each, within the 16,384 command buffers its binding-table pool holds with one each
/// and at most [`PLANNED_BARRIERS`] of barriers.
const PASSES: usize = 10_000;

/// The passes each command buffer of the reference holds ([`reference`]).
const PER: usize = 500;

/// The targets' width; their height is [`H`].
const WIDTH: u64 = 64;

/// The command buffers this many passes may begin past one each and [`PLANNED_BARRIERS`] of
/// barriers: the clear's, and a few the commit's segments and first work may take.
const SLACK: u64 = 64;

/// The longest the command buffer of every pass may take: a debug build on lavapipe encodes and
/// runs [`PASSES`] passes in seconds.
const WAIT: Duration = Duration::from_secs(300);

/// What the passes draw with: the blending state beside a `Depth32Float` depth, and a
/// depth/stencil state that writes the depth.
struct States {
    blending: RenderPipelineState,
    depth: DepthStencilState,
}

impl States {
    fn new(device: &Device) -> States {
        let mut descriptor = metal2vulkan::metal::state::DepthStencilDescriptor::default();
        descriptor.depth_write_enabled = true;
        States {
            blending: blending(device),
            depth: device.new_depth_stencil_state(&descriptor).unwrap(),
        }
    }
}

/// A `Private` texture of `format`, [`WIDTH`] x [`H`], with `usage`.
fn private_texture(device: &Device, format: PixelFormat, usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, WIDTH, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A color target and its depth.
fn targets(device: &Device) -> (Texture, Texture) {
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    (
        private_texture(device, PixelFormat::RGBA8Unorm, usage),
        private_texture(
            device,
            PixelFormat::Depth32Float,
            TextureUsage::RENDER_TARGET,
        ),
    )
}

/// The descriptor of a pass over `color`, loaded (`load`) and stored, beside `depth`, cleared and
/// stored.
fn descriptor((color, depth): (&Texture, &Texture), load: LoadAction) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(color.clone());
    attachment.load_action = load;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = crate::ClearColor::new(0.0, 0.0, 0.0, 1.0);
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    descriptor
}

/// Encode into `command_buffer` the pass that clears `targets`, no draw.
fn clear(command_buffer: &CommandBuffer, targets: (&Texture, &Texture)) {
    let encoder = command_buffer
        .render_command_encoder(&descriptor(targets, LoadAction::Clear))
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// Encode into `command_buffer` pass `i`, which blends rectangle `i` (`drawn`) over `targets`'
/// color, loaded, beside their depth.
fn pass(command_buffer: &CommandBuffer, targets: (&Texture, &Texture), states: &States, i: usize) {
    let mut encoder = command_buffer
        .render_command_encoder(&descriptor(targets, LoadAction::Load))
        .unwrap();
    encoder.set_render_pipeline_state(&states.blending).unwrap();
    encoder.set_depth_stencil_state(&states.depth).unwrap();
    let (corners, rgba) = drawn(i);
    (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
    (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// Commit `command_buffer` and wait for it to complete, with no error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The texels of `texture`, packed: read through a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let packed = WIDTH * TEXEL;
    let out = (queue.device())
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, H, 1),
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, packed * H)
}

/// The texels of the clear and the [`PASSES`] passes over targets of their own, in command
/// buffers of [`PER`] passes each.
fn reference(queue: &CommandQueue, states: &States) -> Vec<u8> {
    let (color, depth) = targets(queue.device());
    let command_buffer = queue.command_buffer().unwrap();
    clear(&command_buffer, (&color, &depth));
    run(&command_buffer);
    for start in (0..PASSES).step_by(PER) {
        let command_buffer = queue.command_buffer().unwrap();
        for i in start..(start + PER).min(PASSES) {
            pass(&command_buffer, (&color, &depth), states, i);
        }
        run(&command_buffer);
    }
    texels_of(queue, &color)
}

/// One command buffer of a clear and [`PASSES`] passes, each beside a depth it clears and stores,
/// runs, as Metal runs it, and leaves the texels the same passes leave across command buffers of
/// [`PER`] passes. Each pass's first instance records inline: no secondary command buffer, and
/// at most one command buffer its encoder begins a pass; and the encoders past the first
/// [`PLANNED_BARRIERS`] begin with a barrier from everything before, so the submission begins at
/// most that many command buffers of barriers.
#[test]
fn one_command_buffer_of_ten_thousand_passes_beside_a_cleared_depth_runs() {
    let cache = Scratch::new("many-passes");
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let states = States::new(&device);
    let (color, depth) = targets(&device);
    let before = counts_alone(&device);
    let command_buffer = queue.command_buffer().unwrap();
    clear(&command_buffer, (&color, &depth));
    for i in 0..PASSES {
        pass(&command_buffer, (&color, &depth), &states, i);
    }
    run(&command_buffer);
    let after = device.work_counts();
    let begun = after.command_buffers_begun - before.command_buffers_begun;
    let encoders = after.encoder_command_buffers_begun - before.encoder_command_buffers_begun;
    let secondaries =
        after.command_buffers_begun_secondaries - before.command_buffers_begun_secondaries;
    let barriers =
        after.command_buffers_begun_batch_barriers - before.command_buffers_begun_batch_barriers;
    let kept = after.barrier_buffers_kept - before.barrier_buffers_kept;
    let inline = after.inline_passes - before.inline_passes;
    let barriered = after.encoders_barriered_at_open - before.encoders_barriered_at_open;
    println!(
        "MANY-PASSES {}: {PASSES} passes beside a cleared depth in one command buffer began \
         {begun} command buffers: {encoders} as its encoders recorded, {secondaries} of them \
         secondaries, and {barriers} of barriers as it was submitted, {kept} more kept; \
         {inline} passes inline, {barriered} encoders barriered as they opened",
        device.name()
    );
    assert_eq!(secondaries, 0, "the passes began secondaries");
    assert_eq!(inline, PASSES as u64, "the passes recorded inline");
    assert!(
        encoders <= PASSES as u64 + SLACK,
        "the passes' encoders began {encoders} command buffers"
    );
    let planned = PLANNED_BARRIERS as u64;
    assert_eq!(
        barriered,
        PASSES as u64 + 1 - planned,
        "the encoders past the planned barriers began with one"
    );
    assert!(
        begun <= PASSES as u64 + planned + SLACK,
        "the command buffer began {begun} command buffers, {barriers} of barriers"
    );
    let texels = texels_of(&queue, &color);
    assert!(
        texels == reference(&queue, &states),
        "the passes in one command buffer left other texels than across command buffers"
    );
}
