//! A compute encoder's bind state sets aside the states of the pipeline states it dispatched before
//! its last dispatch's (`compute/live.rs`'s **One per pipeline state lately dispatched**), and they
//! go on with it to the command buffer's next compute encoder and, past the commit, to the device's
//! next: a dispatch of a pipeline state after dispatches of others starts from its own state's
//! (`WorkCounts::dispatches_from_set_aside`), whose commands it takes while no value they may read
//! moved (`set_aside_commands_kept`), and instantiates again where one did
//! (`set_aside_commands_instantiated`). Every dispatch is the only one of its compute encoder, as a
//! guest's compute workloads encode them, and switches pipeline state:
//!
//! - **The same bindings**: three states in turn, each writing its own output, instantiate once
//!   each in the first command buffer and never in the next, and every word is what they wrote.
//! - **An output that moves**: each state's output bound at another offset at each of its
//!   dispatches instantiates again, and writes there.
//! - **Bytes that change**: each state's scale set as bytes (`set_bytes`) of another value at each
//!   command buffer keeps its commands, which read no value of bytes, and writes with the new one.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), its input at buffer 0 (`i`), its output at 1 and its
//! scale at 2; each of the three pipeline states is a creation of its own of that function.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::words_of;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, ResourceOptions,
    Size, WorkCounts,
};

/// The words a dispatch writes, one threadgroup's.
const WORDS: u64 = 64;

/// The pipeline states dispatched in turn.
const STATES: usize = 3;

/// The dispatches of each state in a command buffer.
const ROUNDS: u64 = 4;

/// The device, the three states, the kernel's input `0, 1, 2, ...` as floats and its scale of 1.
struct Kernels {
    device: Device,
    states: Vec<ComputePipelineState>,
    input: Buffer,
    scale: Buffer,
}

/// How a dispatch's scale is bound: the buffer of 1, or bytes of a value.
#[derive(Clone, Copy)]
enum Scale {
    One,
    Bytes(f32),
}

impl Kernels {
    fn new(scratch: &Scratch) -> Kernels {
        let device = device(&scratch.0);
        let states = (0..STATES).map(|_| state(&device)).collect();
        let input: Vec<u8> = (0..WORDS).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let input = (device.new_buffer_with_bytes(&input, ResourceOptions::shared())).unwrap();
        let scale =
            (device.new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())).unwrap();
        Kernels {
            device,
            states,
            input,
            scale,
        }
    }

    /// A `Shared` buffer of `count` dispatches' words of zeros.
    fn output(&self, count: u64) -> Buffer {
        let bytes = vec![0; (count * WORDS * 4) as usize];
        (self
            .device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared()))
        .unwrap()
    }

    /// A compute encoder of `command_buffer` of one dispatch of state `k` that writes `output`
    /// from byte `offset` on with `scale`, ended.
    fn dispatch(
        &self,
        command_buffer: &CommandBuffer,
        k: usize,
        (output, offset): (&Buffer, u64),
        scale: Scale,
    ) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.states[k]).unwrap();
        encoder.set_buffer(Some(&self.input), 0, 0).unwrap();
        encoder.set_buffer(Some(output), offset, 1).unwrap();
        match scale {
            Scale::One => encoder.set_buffer(Some(&self.scale), 0, 2).unwrap(),
            Scale::Bytes(value) => encoder.set_bytes(&value.to_le_bytes(), 2).unwrap(),
        }
        encoder
            .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(WORDS, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Commit `command_buffer` and wait for it: it completes with no error.
    fn run(&self, command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?}",
            command_buffer.error()
        );
    }
}

/// What moved of the counts the set-aside states name, from `before` to `after`.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// `dispatches_instantiated`.
    instantiated: u64,
    /// `dispatches_from_set_aside`.
    from_aside: u64,
    /// `set_aside_commands_kept`.
    kept: u64,
    /// `set_aside_commands_instantiated`.
    again: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        Moved {
            instantiated: after.dispatches_instantiated - before.dispatches_instantiated,
            from_aside: after.dispatches_from_set_aside - before.dispatches_from_set_aside,
            kept: after.set_aside_commands_kept - before.set_aside_commands_kept,
            again: after.set_aside_commands_instantiated - before.set_aside_commands_instantiated,
        }
    }
}

/// The words `(scale + 1) * i` of `count` dispatches, as floats' bits.
fn written(count: u64, scale: f32) -> Vec<u32> {
    (0..count)
        .flat_map(|_| (0..WORDS).map(move |i| ((scale + 1.0) * i as f32).to_bits()))
        .collect()
}

/// Zero `buffer`, which no command buffer uses now.
fn zero(buffer: &Buffer) {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command buffer reads or writes now.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, buffer.length() as usize) };
}

/// Two command buffers of [`ROUNDS`] rounds of the three states in turn, an encoder each, each
/// state writing its own output with the same bindings at every dispatch: in the first, each
/// state's first dispatch instantiates and each later one starts from its state set aside and
/// takes its commands; in the second, which starts from the states the first gave the device,
/// every dispatch does. Every output holds what its state wrote.
#[test]
fn states_in_turn_of_the_same_bindings_instantiate_once_each() {
    let scratch = Scratch::new("aside-same");
    let kernels = Kernels::new(&scratch);
    let queue = kernels.device.new_command_queue().unwrap();
    let outputs: Vec<Buffer> = (0..STATES).map(|_| kernels.output(1)).collect();
    let dispatches = ROUNDS * STATES as u64;
    let wants = [
        Moved {
            instantiated: STATES as u64,
            from_aside: dispatches - STATES as u64,
            kept: dispatches - STATES as u64,
            again: 0,
        },
        Moved {
            instantiated: 0,
            from_aside: dispatches,
            kept: dispatches,
            again: 0,
        },
    ];
    for (round, want) in wants.into_iter().enumerate() {
        outputs.iter().for_each(zero);
        let before = kernels.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..ROUNDS {
            for (k, output) in outputs.iter().enumerate() {
                kernels.dispatch(&command_buffer, k, (output, 0), Scale::One);
            }
        }
        let moved = Moved::of(&before, &kernels.device.work_counts());
        kernels.run(&command_buffer);
        for (k, output) in outputs.iter().enumerate() {
            assert_eq!(
                words_of(output),
                written(1, 1.0),
                "{round}: state {k}'s words"
            );
        }
        assert_eq!(moved, want, "command buffer {round}: the counts");
    }
}

/// A command buffer of [`ROUNDS`] rounds of the three states in turn, an encoder each, each state
/// binding its output at the next dispatch's offset at each of its dispatches: each dispatch
/// instantiates, those that start from their state set aside again, and every dispatch's words are
/// what it wrote.
#[test]
fn a_state_whose_output_moves_instantiates_again() {
    let scratch = Scratch::new("aside-moved");
    let kernels = Kernels::new(&scratch);
    let queue = kernels.device.new_command_queue().unwrap();
    let outputs: Vec<Buffer> = (0..STATES).map(|_| kernels.output(ROUNDS)).collect();
    let before = kernels.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for round in 0..ROUNDS {
        for (k, output) in outputs.iter().enumerate() {
            let at = (output, round * WORDS * 4);
            kernels.dispatch(&command_buffer, k, at, Scale::One);
        }
    }
    let moved = Moved::of(&before, &kernels.device.work_counts());
    kernels.run(&command_buffer);
    for (k, output) in outputs.iter().enumerate() {
        assert_eq!(words_of(output), written(ROUNDS, 1.0), "state {k}'s words");
    }
    let dispatches = ROUNDS * STATES as u64;
    let want = Moved {
        instantiated: dispatches,
        from_aside: dispatches - STATES as u64,
        kept: 0,
        again: dispatches - STATES as u64,
    };
    assert_eq!(moved, want, "the counts");
}

/// [`ROUNDS`] command buffers of the three states in turn, an encoder each, each state's scale set
/// as bytes of the command buffer's own value: every dispatch after the first command buffer's
/// starts from its state set aside and takes its commands, which read no value of the bytes, and
/// each command buffer's words are those of its own scale.
#[test]
fn a_state_whose_bytes_change_keeps_its_commands() {
    let scratch = Scratch::new("aside-bytes");
    let kernels = Kernels::new(&scratch);
    let queue = kernels.device.new_command_queue().unwrap();
    let outputs: Vec<Buffer> = (0..STATES).map(|_| kernels.output(1)).collect();
    for round in 0..ROUNDS {
        let scale = 2.0 + round as f32;
        outputs.iter().for_each(zero);
        let before = kernels.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        for (k, output) in outputs.iter().enumerate() {
            kernels.dispatch(&command_buffer, k, (output, 0), Scale::Bytes(scale));
        }
        let moved = Moved::of(&before, &kernels.device.work_counts());
        kernels.run(&command_buffer);
        for (k, output) in outputs.iter().enumerate() {
            let want = written(1, scale);
            assert_eq!(words_of(output), want, "{round}: state {k}'s words");
        }
        let states = STATES as u64;
        let want = match round {
            0 => Moved {
                instantiated: states,
                from_aside: 0,
                kept: 0,
                again: 0,
            },
            _ => Moved {
                instantiated: 0,
                from_aside: states,
                kept: states,
                again: 0,
            },
        };
        assert_eq!(moved, want, "command buffer {round}: the counts");
    }
}

/// A state set aside takes at most 2 KiB beside its bind state's slots and lists: eight bind
/// states (the one in place and seven set aside) at most 16 KiB of an encoder's box.
#[test]
fn a_state_set_aside_is_bounded() {
    let bytes = crate::compute::SET_ASIDE_BYTES;
    eprintln!("SET-ASIDE-BYTES {bytes}");
    assert!(bytes <= 2048, "a state set aside of {bytes} bytes");
}
