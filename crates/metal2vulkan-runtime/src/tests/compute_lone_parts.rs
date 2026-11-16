//! A compute encoder of one dispatch, as a guest's encoders are, summarizes its hazards from the
//! dispatch's own uses (`tracker.rs`, `EncoderUses`'s `lone`): its part builds no map of them, so
//! a command buffer of such encoders notes no command in one (`tracker::part_notes`), and one of
//! encoders of two dispatches notes both of each.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects, with `compute_parked.rs`'s kernel.

use super::blit_holds::Scratch;
use super::compute_parked::{written, Kernel, WORDS};
use super::encoder_holds::words_of;
use crate::tracker::part_notes;

/// The compute encoders of a command buffer, a guest's.
const ENCODERS: u64 = 5;

/// A command buffer of [`ENCODERS`] compute encoders of one dispatch each, each writing its own
/// words: no dispatch notes its uses in a part's map, and every word is what it wrote.
#[test]
fn encoders_of_one_dispatch_note_no_part_map() {
    let scratch = Scratch::new("lone-parts");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(ENCODERS);
    let command_buffer = queue.command_buffer().unwrap();
    let before = part_notes();
    for k in 0..ENCODERS {
        kernel.dispatch(&command_buffer, (&output, k * WORDS * 4));
    }
    let noted = part_notes() - before;
    kernel.run(&command_buffer);
    assert_eq!(words_of(&output), written(ENCODERS), "the words");
    assert_eq!(
        noted, 0,
        "commands noted in a part's map, {ENCODERS} encoders"
    );
}

/// The same command buffer with each encoder dispatching twice over its words: each encoder's
/// part notes both its dispatches in its map, the first as the second comes.
#[test]
fn encoders_of_two_dispatches_note_both() {
    let scratch = Scratch::new("lone-parts-two");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(ENCODERS);
    let command_buffer = queue.command_buffer().unwrap();
    let before = part_notes();
    for k in 0..ENCODERS {
        kernel.dispatches(&command_buffer, (&output, k * WORDS * 4), 2);
    }
    let noted = part_notes() - before;
    kernel.run(&command_buffer);
    assert_eq!(words_of(&output), written(ENCODERS), "the words");
    assert_eq!(noted, 2 * ENCODERS, "commands noted in a part's map");
}
