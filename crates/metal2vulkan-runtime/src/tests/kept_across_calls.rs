//! A render encoder's kept draws kept across the calls that change nothing they read or recorded
//! (`render/draw_again.rs`, `WorkCounts::kept_draws_kept_across_*`): `update_fence`,
//! `wait_for_fence` where the encoder used no shadowed texture, `use_resource`,
//! `use_resources`, `use_heap` and `use_heaps` of no memoryless texture, and
//! `memory_barrier_with_scope` over the render targets alone inside the pass, which lets go the
//! kept draws that read the pass's colors past snapshots of their own. The draw after each
//! records the same commands and pixels as the same draw after the same call and a call that
//! forgets (`set_tessellation_factor_scale`, which no draw of the state reads), which takes every
//! step: drawn again where the call changed nothing, and taking every step itself where a use
//! changed the encoder's residency (`WorkCounts::draws_unlike_kept_under`). The calls that forget
//! are counted by kind (`WorkCounts::kept_draws_let_go_*`), and so is an encoder whose last draw
//! drew nothing (`WorkCounts::encoders_last_draw_empty`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll`'s and `wlk_readers.ll`'s (`llvm-as`).

use super::blit_holds::{until, Scratch, WAIT};
use super::draw_again::{adding_state, device, rectangle, run, set_rectangle, warm, Heard, DRAWS};
use super::encoder_holds::{bytes_of, library, words_of};
use crate::{
    Buffer, ClearColor, CommandBufferStatus, Device, Fence, Heap, HeapDescriptor, HeapType,
    Library, LoadAction, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{
    BarrierScope, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// What the calls between the draws name.
struct Named {
    updated: Fence,
    waited: Fence,
    buffer: Buffer,
    heap: Heap,
}

impl Named {
    fn new(device: &Device) -> Named {
        let heap = HeapDescriptor::new(HeapType::Automatic, 1 << 16, ResourceOptions::private());
        Named {
            updated: device.new_fence(),
            waited: device.new_fence(),
            buffer: device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            heap: device.new_heap(&heap).unwrap(),
        }
    }

    /// The call before the draw `at` (from 1 to `DRAWS - 1`): a fence's update, a wait for a
    /// fence no encoder updated, a use of a buffer the encoder did not use (which changes its
    /// residency), the same use again and as a list, a use of a heap (which changes it), the same
    /// as a list, and the update again.
    fn call(&self, encoder: &mut RenderCommandEncoder, at: u64) {
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let buffer = self.buffer.as_resource();
        let called = match at {
            1 | 8 => encoder.update_fence(&self.updated, RenderStages::FRAGMENT),
            2 => encoder.wait_for_fence(&self.waited, RenderStages::VERTEX),
            3 | 4 => encoder.use_resource(buffer, ResourceUsage::READ, stages),
            5 => encoder.use_resources(&[buffer], ResourceUsage::READ, stages),
            6 => encoder.use_heap(&self.heap, stages),
            _ => encoder.use_heaps(&[&self.heap], stages),
        };
        assert_eq!(called, Ok(()));
    }
}

/// [`DRAWS`] draws of the rectangle, each but the first after its call of `named`
/// ([`Named::call`]), and a call that forgets after it where `forget`.
fn draws(
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
    named: &Named,
    forget: bool,
) {
    set_rectangle(encoder, state);
    for at in 0..DRAWS {
        if at > 0 {
            named.call(encoder, at);
            if forget {
                encoder.set_tessellation_factor_scale(1.0);
            }
        }
        rectangle(encoder);
    }
}

/// The draws after the fence and use calls draw again where the calls changed nothing, with the
/// same commands and pixels as the same draws after the same calls and a call that forgets each,
/// which take every step; after a use that changed the residency the draw takes every step.
#[test]
fn draws_after_fences_and_uses_draw_again_as_every_step_draws() {
    let scratch = Scratch::new("kept-across-calls");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let named = Named::new(&device);
    warm(&device, |encoder| draws(encoder, &state, &named, true));
    let kept = run(&device, |encoder| draws(encoder, &state, &named, false));
    let apart = run(&device, |encoder| draws(encoder, &state, &named, true));
    assert_eq!(kept.words, apart.words, "the pixels of the draws");
    assert_eq!(kept.commands, apart.commands);
    assert_eq!(kept.commands["cmd_draw"], DRAWS);
    // Each call kept the kept draw: three fence calls and five uses, in both runs; the forgetting
    // call let it go after each in the second.
    let across_fence = |counts: &WorkCounts| counts.kept_draws_kept_across_fence;
    let across_use = |counts: &WorkCounts| counts.kept_draws_kept_across_use;
    let let_go_other = |counts: &WorkCounts| counts.kept_draws_let_go_other;
    for ran in [&kept, &apart] {
        assert_eq!(ran.counted(across_fence), 3);
        assert_eq!(ran.counted(across_use), 5);
    }
    assert_eq!(kept.counted(let_go_other), 0);
    assert_eq!(apart.counted(let_go_other), DRAWS - 1);
    // The draws after the first use of the buffer and of the heap took every step, the residency
    // changed; the others after the calls drew again: every second one from the first after a
    // draw that took every step in a build with debug assertions, which takes every step instead.
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    let under = |counts: &WorkCounts| counts.draws_unlike_kept_under;
    let alone = match cfg!(debug_assertions) {
        true => 3,
        false => 6,
    };
    assert_eq!(kept.counted(as_before), alone);
    assert_eq!(kept.counted(under), 2);
    assert_eq!(apart.counted(as_before), 0);
    let last_empty = |counts: &WorkCounts| counts.encoders_last_draw_empty;
    assert_eq!(kept.counted(last_empty), 0);
}

/// A barrier between draws lets the kept draw go, counted as a barrier's; an encoder whose last
/// draw drew nothing is counted.
#[test]
fn a_barrier_lets_go_and_an_empty_last_draw_counts() {
    let scratch = Scratch::new("kept-across-barrier");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let encode = |encoder: &mut RenderCommandEncoder| {
        set_rectangle(encoder, &state);
        rectangle(encoder);
        let stages = (RenderStages::FRAGMENT, RenderStages::VERTEX);
        encoder.memory_barrier_with_scope(BarrierScope::BUFFERS, stages.0, stages.1);
        rectangle(encoder);
        let empty = encoder.draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0);
        assert_eq!(empty, Ok(()));
    };
    warm(&device, encode);
    let ran = run(&device, encode);
    assert_eq!(ran.counted(|counts| counts.kept_draws_let_go_barrier), 1);
    assert_eq!(ran.counted(|counts| counts.kept_draws_let_go_other), 0);
    assert_eq!(ran.counted(|counts| counts.encoders_last_draw_empty), 1);
}

/// The draws after barriers over the render targets inside the pass, fragment to fragment, draw
/// again, with the same commands and pixels as the same draws after the same barriers and a call
/// that forgets each, which take every step: each barrier keeps the kept draw
/// (`WorkCounts::kept_draws_kept_across_barrier`), in both runs.
#[test]
fn draws_after_barriers_inside_the_pass_draw_again_as_every_step_draws() {
    let scratch = Scratch::new("kept-across-barriers");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let fragment = RenderStages::FRAGMENT;
    let draws = |encoder: &mut RenderCommandEncoder, forget: bool| {
        set_rectangle(encoder, &state);
        for at in 0..DRAWS {
            if at > 0 {
                encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
                if forget {
                    encoder.set_tessellation_factor_scale(1.0);
                }
            }
            rectangle(encoder);
        }
    };
    warm(&device, |encoder| draws(encoder, true));
    let kept = run(&device, |encoder| draws(encoder, false));
    let apart = run(&device, |encoder| draws(encoder, true));
    assert_eq!(kept.words, apart.words, "the pixels of the draws");
    assert_eq!(kept.commands, apart.commands);
    assert_eq!(kept.commands["cmd_draw"], DRAWS);
    let across = |counts: &WorkCounts| counts.kept_draws_kept_across_barrier;
    let let_go = |counts: &WorkCounts| counts.kept_draws_let_go_barrier;
    for ran in [&kept, &apart] {
        assert_eq!(ran.counted(across), DRAWS - 1);
        assert_eq!(ran.counted(let_go), 0);
    }
    // Every draw after a barrier drew again: every second one from the first after a draw that
    // took every step in a build with debug assertions, which takes every step instead.
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    let alone = match cfg!(debug_assertions) {
        true => (DRAWS - 1) / 2,
        false => DRAWS - 1,
    };
    assert_eq!(kept.counted(as_before), alone);
    assert_eq!(apart.counted(as_before), 0);
}

/// The side of the target [`read_after_barrier`] draws into.
const READ_SIDE: u64 = 16;

/// The state of `tob_rect` and `tob_read`, which writes under each fragment its fragment texture
/// 0's texel at the fragment's pixel, into `RGBA8Unorm`, unblended.
fn reading_state(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_read"));
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The state of `tob_rect` and `wlk_read_branch`, which writes under each fragment its fragment
/// texture 0's texel at the fragment's pixel scaled toward a quarter once for each quarter of its
/// alpha, into `RGBA8Unorm`, unblended: each draw of it over a texel of a half changes what the
/// next reads.
fn shading_state(device: &Device) -> RenderPipelineState {
    let renders = library(device, "engine/tob_renders.ll");
    let readers = library(device, "engine/wlk_readers.ll");
    let function = |library: &Library, name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(&renders, "tob_rect"));
    descriptor.fragment = Some(function(&readers, "wlk_read_branch"));
    let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Into a new `RGBA8Unorm` target that a first pass clears to a half, a second pass loading it,
/// encoded by `draws` (given the target). The target's words and the work counts' change over the
/// second pass.
fn second_pass(
    device: &Device,
    draws: impl Fn(&mut RenderCommandEncoder, &Texture),
) -> (Vec<u32>, WorkCounts, WorkCounts) {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, READ_SIDE, READ_SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let target: Texture = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(READ_SIDE * READ_SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let pass = |load_action| {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = load_action;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.5, 0.5, 0.5, 0.5);
        pass
    };
    let command_buffer = queue.command_buffer().unwrap();
    let cleared = command_buffer.render_command_encoder(&pass(LoadAction::Clear));
    cleared.unwrap().end_encoding().unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(LoadAction::Load))
        .unwrap();
    draws(&mut encoder, &target);
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = crate::Size::new(READ_SIDE, READ_SIDE, 1);
    let row = READ_SIDE * 4;
    let origin = crate::Origin::new(0, 0, 0);
    let copied = blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        origin,
        size,
        &back,
        0,
        row,
        row * READ_SIDE,
        crate::BlitOption::None,
    );
    copied.unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    (words_of(&back), before, after)
}

/// Sets `state` to read `target` (fragment texture 0) over the rectangle.
fn set_reading(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState, target: &Texture) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    encoder.set_fragment_texture(Some(target), 0).unwrap();
}

/// Until every key's own pipeline that `encode` draws with is made and the pool idle, as
/// `draw_again::warm` waits: a draw recorded against a making keeps nothing.
fn warm_reads(device: &Device, encode: impl Fn()) {
    until("every key's own pipeline made", || {
        let residual = device.work_counts().residual_form_draws;
        encode();
        device.work_counts().residual_form_draws == residual
    });
    until("the pipeline pool idle", || {
        device.work_counts().pipeline_pool_busy == 0
    });
}

/// A kept draw that reads its pass's color through its texture's copy, set aside by a draw of
/// another state that writes the color, is let go at a barrier over the render targets inside the
/// pass, so its next draw takes every step and reads what that draw wrote, as the same draws after
/// a call that forgets read it; the barrier keeps the writing draw's kept draw. A reader through
/// snapshots, where the device reads so, stays kept and reads it behind its own barrier.
#[test]
fn a_barrier_inside_the_pass_lets_go_kept_readers_of_its_colors() {
    let scratch = Scratch::new("kept-across-barrier-readers");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let (read, add) = (reading_state(&device), adding_state(&device));
    let fragment = RenderStages::FRAGMENT;
    // A read of the target, a draw that adds to it, the barrier and the read again, which reads
    // what the draw added (Metal's answer after the barrier); a call that forgets after the
    // barrier where `forget`.
    let read_after_barrier = |forget: bool| {
        second_pass(&device, |encoder, target| {
            set_reading(encoder, &read, target);
            rectangle(encoder);
            set_rectangle(encoder, &add);
            rectangle(encoder);
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
            if forget {
                encoder.set_tessellation_factor_scale(1.0);
            }
            set_reading(encoder, &read, target);
            rectangle(encoder);
        })
    };
    warm_reads(&device, || {
        read_after_barrier(true);
    });
    let (kept, before, after) = read_after_barrier(false);
    let (apart, _, _) = read_after_barrier(true);
    assert_eq!(kept, apart, "the pixels of the reads");
    let counted = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    assert_eq!(counted(|counts| counts.kept_draws_kept_across_barrier), 1);
    assert_eq!(counted(|counts| counts.kept_draws_let_go_barrier), 0);
}

/// The last direct draw's kept draw, where it reads its pass's color through its texture's copy
/// (a read the snapshots refuse, which decides a branch), is let go at a barrier over the render
/// targets inside the pass, so the same draw after the barrier takes every step and reads what the
/// draw before the barrier wrote, as the same draws after a call that forgets read it: drawn
/// again, it would read the copy the draw before read. The draw after the barrier splits its
/// instance, so it keeps nothing.
#[test]
fn a_barrier_inside_the_pass_lets_go_the_last_draw_s_kept_reader_of_its_colors() {
    let scratch = Scratch::new("kept-across-barrier-last-reader");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = shading_state(&device);
    let fragment = RenderStages::FRAGMENT;
    // Two reads of the target, the barrier between them, and a call that forgets after it where
    // `forget`.
    let reads = |forget: bool| {
        second_pass(&device, |encoder, target| {
            set_reading(encoder, &state, target);
            rectangle(encoder);
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
            if forget {
                encoder.set_tessellation_factor_scale(1.0);
            }
            rectangle(encoder);
        })
    };
    warm_reads(&device, || {
        reads(true);
    });
    let (kept, before, after) = reads(false);
    let (apart, _, _) = reads(true);
    assert_eq!(kept, apart, "the pixels of the reads");
    let counted = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    assert_eq!(counted(|counts| counts.attachment_reads_through_copies), 2);
    assert_eq!(counted(|counts| counts.kept_draws_let_go_barrier), 1);
    assert_eq!(counted(|counts| counts.kept_draws_kept_across_barrier), 0);
}
