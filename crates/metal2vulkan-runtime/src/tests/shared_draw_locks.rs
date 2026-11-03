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
use super::draw_again::{adding_state, device, rectangle, run, set_rectangle, warm, Heard, DRAWS};
use super::encoder_holds::bytes_of;
use crate::device::shared_draw_locks;
use crate::RenderCommandEncoder;
use std::cell::Cell;

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
