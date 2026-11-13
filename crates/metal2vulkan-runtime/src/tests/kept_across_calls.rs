//! A render encoder's kept draws kept across the calls that change nothing they read or recorded
//! (`render/draw_again.rs`, `WorkCounts::kept_draws_kept_across_*`): `update_fence`,
//! `wait_for_fence` where the encoder used no shadowed texture, and `use_resource`,
//! `use_resources`, `use_heap` and `use_heaps` of no memoryless texture. The draw after each
//! records the same commands and pixels as the same draw after the same call and a call that
//! forgets (`set_tessellation_factor_scale`, which no draw of the state reads), which takes every
//! step: drawn again where the call changed nothing, and taking every step itself where a use
//! changed the encoder's residency (`WorkCounts::draws_unlike_kept_under`). The calls that forget
//! are counted by kind (`WorkCounts::kept_draws_let_go_*`), and so is an encoder whose last draw
//! drew nothing (`WorkCounts::encoders_last_draw_empty`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll`'s (`llvm-as`).

use super::blit_holds::Scratch;
use super::draw_again::{adding_state, device, rectangle, run, set_rectangle, warm, Heard, DRAWS};
use crate::{
    Buffer, Device, Fence, Heap, HeapDescriptor, HeapType, RenderCommandEncoder,
    RenderPipelineState, ResourceOptions, WorkCounts,
};
use metal2vulkan::metal::{BarrierScope, PrimitiveType, RenderStages, ResourceUsage};

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
