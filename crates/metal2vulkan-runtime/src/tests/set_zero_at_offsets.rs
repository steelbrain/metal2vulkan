//! Draws that bind one buffer at other offsets, at `device` slots read through storage views of
//! their whole windows, write their set 0 once: each such slot's descriptor range runs from the
//! buffer's space start to its padded end whatever the bind offset, which its residual push field
//! holds (`metal2vulkan`'s `bind::instantiate::planned_range`). Where set 0 is allocated and
//! bound, they bind the set 0 their encoder's first draw wrote
//! (`WorkCounts::set_zero_sets_reused`), once. Where their variant pushes it
//! (`WorkCounts::draws_pushing_set_zero`: a device of Vulkan 1.4 or `VK_KHR_push_descriptor`
//! whose set 0 a push descriptor set layout holds), the first draw's push carries its writes and
//! no later draw pushes, since set 0 holds them (`render/bound.rs`). They draw what the same draws
//! draw from buffers of their own each, which write a set each.
//!
//! Device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The functions
//! are `colored_vertex` and `colored_fragment` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::draw_again::warm;
use super::encoder_holds::{bytes_of, pass, render_state, words_of};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, Device, Origin, RenderCommandEncoder,
    RenderPipelineState, ResourceOptions, Size, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};

/// The side of the target, and the columns of it the draws fill: one each.
const SIDE: u64 = 16;

/// The draws of a run.
const DRAWS: u64 = 9;

/// A draw's vertices: a rectangle of two triangles.
const VERTICES: u32 = 6;

/// The bytes of a draw's positions (`float2` each) and of its colours (`float4` each).
const POSITION_BYTES: u64 = VERTICES as u64 * 8;
const COLOR_BYTES: u64 = VERTICES as u64 * 16;

/// The colour of draw `at`, in `RGBA8Unorm` codes, each exact in `float`.
fn codes(at: u64) -> [u8; 4] {
    let red = 16 * (at as u8 + 1);
    [red, 128, 255 - red, 255]
}

/// Draw `at`'s positions: the rectangle over column `at` of the target, top to bottom.
fn positions(at: u64) -> Vec<f32> {
    let x = |column: u64| -1.0 + 2.0 * column as f32 / SIDE as f32;
    let (left, right) = (x(at), x(at + 1));
    [
        [left, -1.0],
        [right, -1.0],
        [left, 1.0],
        [left, 1.0],
        [right, -1.0],
        [right, 1.0],
    ]
    .concat()
}

/// Draw `at`'s colours: its colour at each vertex.
fn colors(at: u64) -> Vec<f32> {
    let color = codes(at).map(|code| f32::from(code) / 255.0);
    color.repeat(VERTICES as usize)
}

/// A shared buffer of `values`.
fn buffer_of(device: &Device, values: &[f32]) -> Buffer {
    let bytes = bytes_of(values, f32::to_le_bytes);
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// What a run drew and counted: the target's words, its `set_zero_sets_reused`,
/// `draws_rebound` and `draws_pushing_set_zero`, and its `vkCmdBindDescriptorSets`,
/// `vkCmdPushDescriptorSet` and `vkCmdDraw` calls.
struct Ran {
    words: Vec<u32>,
    reused: u64,
    rebound: u64,
    pushing: u64,
    binds: u64,
    pushes: u64,
    draws: u64,
}

/// Encode `encode` with `state` set in one pass into a new cleared `RGBA8Unorm` target, commit and
/// wait: what it drew, and what its encoder counted.
fn run(
    device: &Device,
    state: &RenderPipelineState,
    encode: impl Fn(&mut RenderCommandEncoder),
) -> Ran {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let grew = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Ran {
        words: words_of(&back),
        reused: grew(|counts| counts.set_zero_sets_reused),
        rebound: grew(|counts| counts.draws_rebound),
        pushing: grew(|counts| counts.draws_pushing_set_zero),
        binds: grew(|counts| counts.commands_recorded.cmd_bind_descriptor_sets),
        pushes: grew(|counts| counts.commands_recorded.cmd_push_descriptor_set),
        draws: grew(|counts| counts.commands_recorded.cmd_draw),
    }
}

/// The draw of the rectangle at the bound offsets.
fn rectangle(encoder: &mut RenderCommandEncoder) {
    let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, VERTICES, 1, 0);
    assert_eq!(drawn, Ok(()));
}

/// Whether the draws of `runs` pushed their set 0 (`WorkCounts::draws_pushing_set_zero`): one
/// variant's draws push it all or none.
fn pushes_set_zero(runs: &[&Ran]) -> bool {
    let pushing: Vec<u64> = runs.iter().map(|ran| ran.pushing).collect();
    assert!(
        pushing.iter().all(|&p| p == 0) || pushing.iter().all(|&p| p == DRAWS),
        "draws pushing set 0 in each run: {pushing:?} of {DRAWS}; one variant's draws push it \
         all or none"
    );
    pushing[0] == DRAWS
}

/// How a run recorded set 0: the sets it bound again (`set_zero_sets_reused`), and its
/// `vkCmdBindDescriptorSets` and `vkCmdPushDescriptorSet` calls.
fn set_zero(ran: &Ran) -> (u64, u64, u64) {
    (ran.reused, ran.binds, ran.pushes)
}

/// The state of `colored_vertex` and `colored_fragment`, and one buffer of every draw's positions
/// and one of their colours, one after another.
fn state_and_buffers(device: &Device) -> (RenderPipelineState, (Buffer, Buffer)) {
    let state = render_state(
        device,
        "examples/runtime_examples.ll",
        "colored_vertex",
        "colored_fragment",
    );
    let (all_positions, all_colors): (Vec<f32>, Vec<f32>) =
        (0..DRAWS).fold(Default::default(), |(mut p, mut c), at| {
            p.extend(positions(at));
            c.extend(colors(at));
            (p, c)
        });
    let shared = (
        buffer_of(device, &all_positions),
        buffer_of(device, &all_colors),
    );
    (state, shared)
}

#[test]
fn draws_that_move_only_a_device_buffer_s_offset_bind_one_set_zero() {
    let scratch = Scratch::new("set-zero-at-offsets");
    let device = device(&scratch.0);
    let (state, shared) = state_and_buffers(&device);
    let own: Vec<(Buffer, Buffer)> = (0..DRAWS)
        .map(|at| {
            (
                buffer_of(&device, &positions(at)),
                buffer_of(&device, &colors(at)),
            )
        })
        .collect();
    // One buffer each for the positions and the colours, the draws' offsets set between them.
    let at_offsets = run(&device, &state, |encoder| {
        encoder.set_vertex_buffer(Some(&shared.0), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&shared.1), 0, 1).unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                encoder
                    .set_vertex_buffer_offset(POSITION_BYTES * at, 0)
                    .unwrap();
                encoder
                    .set_vertex_buffer_offset(COLOR_BYTES * at, 1)
                    .unwrap();
            }
            rectangle(encoder);
        }
    });
    // Buffers of their own each, at offset 0.
    let own_buffers = run(&device, &state, |encoder| {
        for (positions, colors) in &own {
            encoder.set_vertex_buffer(Some(positions), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(colors), 0, 1).unwrap();
            rectangle(encoder);
        }
    });
    let expected: Vec<u32> = (0..SIDE * SIDE)
        .map(|texel| match texel % SIDE {
            column if column < DRAWS => u32::from_le_bytes(codes(column)),
            _ => 0,
        })
        .collect();
    assert_eq!(at_offsets.words, expected, "the draws at offsets");
    assert_eq!(
        own_buffers.words, expected,
        "the draws of their own buffers"
    );
    assert_eq!((at_offsets.draws, own_buffers.draws), (DRAWS, DRAWS));
    let (once, each) = match pushes_set_zero(&[&at_offsets, &own_buffers]) {
        false => ((DRAWS - 1, 1, 0), (0, DRAWS, 0)),
        true => ((0, 0, 1), (0, 0, DRAWS)),
    };
    assert_eq!(
        set_zero(&at_offsets),
        once,
        "the draws at offsets record their first draw's set 0, once"
    );
    assert_eq!(
        set_zero(&own_buffers),
        each,
        "the draws of their own buffers write and record a set 0 each"
    );
}

/// Draws that move the second buffer's offset alone, the first bound as before, are rebound at
/// that slot alone (`WorkCounts::draws_rebound`): its descriptor range found at its place among
/// the bound ones from the plan, the unchanged slot before it not looked at
/// (`rebind_draw_slots_into`), as every step finds it, which every second draw rebound checks in
/// builds with debug assertions (`render/draw_again/`). Each draws draw 0's rectangle in its own
/// colour, the last one's left.
#[test]
fn draws_that_move_the_second_buffer_s_offset_alone_are_rebound() {
    let scratch = Scratch::new("set-zero-second-offset");
    let device = device(&scratch.0);
    let (state, shared) = state_and_buffers(&device);
    let draws = |encoder: &mut RenderCommandEncoder| {
        encoder.set_vertex_buffer(Some(&shared.0), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&shared.1), 0, 1).unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                encoder
                    .set_vertex_buffer_offset(COLOR_BYTES * at, 1)
                    .unwrap();
            }
            rectangle(encoder);
        }
    };
    // Each draw's own pipeline made, as a draw kept to draw again finds it.
    warm(&device, |encoder| {
        encoder.set_render_pipeline_state(&state).unwrap();
        draws(encoder);
    });
    let ran = run(&device, &state, draws);
    let expected: Vec<u32> = (0..SIDE * SIDE)
        .map(|texel| match texel % SIDE {
            0 => u32::from_le_bytes(codes(DRAWS - 1)),
            _ => 0,
        })
        .collect();
    assert_eq!(ran.words, expected);
    let once = match pushes_set_zero(&[&ran]) {
        false => (DRAWS - 1, 1, 0),
        true => (0, 0, 1),
    };
    assert_eq!(
        set_zero(&ran),
        once,
        "the draws record their first draw's set 0, once"
    );
    assert!(
        ran.rebound > 0,
        "no draw rebound at the colours' slot alone"
    );
}
