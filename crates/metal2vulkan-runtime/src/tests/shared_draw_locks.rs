//! A render encoder's draws of one pipeline state and shape take no lock of what draws on other
//! threads share, past its first draw: its settle is the state's kept one (`live_bind.rs`), a
//! state that made no visible function table lists none without the device's registry
//! (`GraphicsState::function_tables`), and a variant that tests no slot's format tells its
//! formats' notes quiet without its lock (`VariantObjects::degrade_bound_formats`). Counted by
//! `device::shared_draw_lock`, in builds with debug assertions.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::Scratch;
use super::draw_again::{
    adding_state, device, rectangle, rectangles_from_columns, run, set_rectangle, warm, Heard,
    DRAWS,
};
use super::encoder_holds::bytes_of;
use crate::command_buffer::state_locks;
use crate::device::shared_draw_locks;
use crate::RenderCommandEncoder;
use std::cell::{Cell, RefCell};

/// Draws of the rectangle, each after its vertex bytes are set again (a call, so each takes every
/// step, its shape kept), take shared locks at their encoder's first draw, and none at any later
/// one.
#[test]
fn draws_of_a_kept_shape_take_no_shared_lock_past_the_first() {
    let scratch = Scratch::new("shared-draw-locks");
    let device = device(&scratch.0, &[], &Heard::default());
    let state = adding_state(&device);
    let (first, later) = (Cell::new(0), Cell::new(0));
    let encode = |encoder: &mut RenderCommandEncoder| {
        let start = shared_draw_locks();
        set_rectangle(encoder, &state);
        rectangle(encoder);
        let drawn = shared_draw_locks();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        for _ in 1..DRAWS {
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            rectangle(encoder);
        }
        first.set(drawn - start);
        later.set(shared_draw_locks() - drawn);
    };
    warm(&device, encode);
    run(&device, encode);
    assert!(
        first.get() > 0,
        "the encoder's first draw settles through its state"
    );
    assert_eq!(later.get(), 0, "shared locks taken past the first draw");
}

/// Draws rebound at another offset of their vertex buffer (`WorkCounts::draws_rebound`) lock no
/// command buffer's state, which its encoders on other threads share: of each draw after the
/// offset changed, the draws it counted rebound and the states it locked, every draw counted
/// rebound locking none. A draw the debug check takes every step of (every second) may lock it.
#[test]
fn draws_rebound_at_another_offset_lock_no_command_buffer_state() {
    let scratch = Scratch::new("rebound-state-locks");
    let device = device(&scratch.0, &[], &Heard::default());
    let state = adding_state(&device);
    let columns = rectangles_from_columns(&device);
    let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
    let drawn = RefCell::new(Vec::new());
    let encode = |encoder: &mut RenderCommandEncoder| {
        drawn.borrow_mut().clear();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&columns), 0, 0).unwrap();
        encoder.set_fragment_bytes(&color, 0).unwrap();
        rectangle(encoder);
        for at in 1..DRAWS {
            encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
            let before = (device.work_counts().draws_rebound, state_locks());
            rectangle(encoder);
            let after = (device.work_counts().draws_rebound, state_locks());
            drawn
                .borrow_mut()
                .push((after.0 - before.0, after.1 - before.1));
        }
    };
    warm(&device, encode);
    run(&device, encode);
    let drawn = drawn.into_inner();
    let rebound: Vec<_> = drawn.iter().filter(|(rebound, _)| *rebound == 1).collect();
    assert!(
        rebound.len() as u64 >= (DRAWS - 1) / 2 - 1,
        "draws rebound: {drawn:?}"
    );
    assert!(
        rebound.iter().all(|(_, locked)| *locked == 0),
        "command buffer states locked by draws rebound (rebound, locked): {drawn:?}"
    );
}
