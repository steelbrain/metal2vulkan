//! An encoder holds none of the shared objects its command buffer held lately again
//! (`retained.rs`'s **Once each, through the command buffer**): encoders that name the same
//! objects add one holds entry to their command buffer, not one each, and every object lives until
//! the whole command buffer completed, where a part that held it completed first and where an
//! encoder that held it through its command buffer is encoded again apart.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), with a scale of 1.

use super::blit_holds::{device, until, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::words_of;
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommitFailure,
    ComputePipelineState, Device, EventRef, FailedEncoders, ResourceOptions, Size,
};
use std::any::Any;
use std::collections::BTreeSet;
use std::sync::{Arc, Weak};
use std::time::Duration;

/// The words of each buffer: one threadgroup.
const WORDS: u64 = 64;

/// The device, the kernel's state, its input `0, 1, 2, ...` and its scale of 1.
struct Kernel {
    device: Device,
    state: ComputePipelineState,
    input: Buffer,
    scale: Buffer,
}

impl Kernel {
    /// The kernel, its pipeline made by a command buffer that ran it.
    fn new(scratch: &Scratch) -> Kernel {
        let device = device(&scratch.0);
        let state = state(&device);
        let floats = |value: &dyn Fn(u64) -> f32| {
            let bytes: Vec<u8> = (0..WORDS).flat_map(|i| value(i).to_le_bytes()).collect();
            device
                .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
                .unwrap()
        };
        let input = floats(&|i| i as f32);
        let scale = floats(&|_| 1.0);
        let kernel = Kernel {
            device,
            state,
            input,
            scale,
        };
        let warm = kernel.output();
        let queue = kernel.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        kernel.dispatch(&command_buffer, &warm);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        kernel
    }

    /// A `Shared` buffer of [`WORDS`] zero words.
    fn output(&self) -> Buffer {
        self.device
            .new_buffer(WORDS * 4, ResourceOptions::shared())
            .unwrap()
    }

    /// A compute encoder of `command_buffer` of one dispatch that writes `output`, ended.
    fn dispatch(&self, command_buffer: &CommandBuffer, output: &Buffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&self.input), 0, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(WORDS, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// Watch the input and the scale, and let go of them: the command buffers that use them
    /// hold them from now on.
    fn release_buffers(self) -> (Device, ComputePipelineState, Watched) {
        let watched = Watched(vec![watch(self.input.held()), watch(self.scale.held())]);
        (self.device, self.state, watched)
    }
}

/// Another reference to `held` that keeps it from nothing.
fn watch(held: Arc<dyn Any + Send + Sync>) -> Weak<dyn Any + Send + Sync> {
    Arc::downgrade(&held)
}

/// The kernel's input and scale, released by the test.
struct Watched(Vec<Weak<dyn Any + Send + Sync>>);

impl Watched {
    /// Whether both still live.
    fn live(&self) -> bool {
        self.0.iter().all(|weak| weak.strong_count() > 0)
    }

    /// Wait until neither lives: the command buffer that held them completed.
    fn gone(&self) {
        until("the released buffers outlived their command buffer", || {
            self.0.iter().all(|weak| weak.strong_count() == 0)
        });
    }
}

/// The words of what the kernel writes: `2 * i`.
fn written() -> Vec<u32> {
    (0..WORDS)
        .map(|i| (2 * i) as f32)
        .map(f32::to_bits)
        .collect()
}

/// The command buffer's holds entries now.
fn entries(command_buffer: &CommandBuffer) -> usize {
    command_buffer.state().retained.len()
}

/// Eight encoders of one command buffer, each of a dispatch of the same state over the same
/// buffers, add one holds entry: the first's. The seven after it hold none of those objects again.
#[test]
fn encoders_of_the_same_objects_hold_them_once_a_command_buffer() {
    let scratch = Scratch::new("held-lately-once");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output();
    let command_buffer = queue.command_buffer().unwrap();
    let before = entries(&command_buffer);
    kernel.dispatch(&command_buffer, &output);
    let first = entries(&command_buffer);
    assert_eq!(first, before + 1, "the first encoder's holds");
    for _ in 1..8 {
        kernel.dispatch(&command_buffer, &output);
    }
    assert_eq!(
        entries(&command_buffer),
        first,
        "an encoder held again what its command buffer held lately"
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(words_of(&output), written());
}

/// An encoder behind an event wait only the host signals holds nothing anew: what it uses the
/// encoder before the wait holds, in the command buffer's earlier part. That part runs and
/// completes first, and its output is then cleared; the input and scale, released at the commit,
/// live until the later part completed, which reads them and writes the output again, and go
/// after.
#[test]
fn an_object_an_earlier_part_holds_lives_until_the_later_part_completed() {
    let scratch = Scratch::new("held-lately-parts");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output();
    let ran = kernel.device.new_shared_event().unwrap();
    let gate = kernel.device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, &output);
    command_buffer
        .encode_signal_event(EventRef::Shared(&ran), 1)
        .unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let before = entries(&command_buffer);
    kernel.dispatch(&command_buffer, &output);
    assert_eq!(
        entries(&command_buffer),
        before,
        "the encoder behind the wait held again what the one before held"
    );
    command_buffer.commit().unwrap();
    let (_device, _state, watched) = kernel.release_buffers();
    until("the part before the wait never ran", || {
        ran.signaled_value() == 1
    });
    // Time for the part's completion to be processed, which lets go of nothing it held.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(words_of(&output), written());
    assert_ne!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(
        watched.live(),
        "a buffer the later part uses went as the earlier part completed"
    );
    let contents = output.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command reads or writes until the gate opens.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, output.length() as usize) };
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    assert_eq!(words_of(&output), written(), "the later part's write");
    watched.gone();
}

/// Three encoders of one group behind an event wait only the host signals, the third of the
/// objects the first names, so holding nothing anew; a commit that fails the second encodes the
/// first and the third again apart. The input and scale, released at the commit, live until the
/// command buffer completed, the third's output is written, and they go after.
#[test]
fn an_encoder_encoded_again_apart_keeps_what_its_command_buffer_held_for_it() {
    let scratch = Scratch::new("held-lately-again");
    let kernel = Kernel::new(&scratch);
    let device = kernel.device.clone();
    let queue = device.new_command_queue().unwrap();
    let outputs = [kernel.output(), kernel.output()];
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let counts = device.work_counts();
    kernel.dispatch(&command_buffer, &outputs[0]);
    kernel.dispatch(&command_buffer, &outputs[1]);
    let before = entries(&command_buffer);
    kernel.dispatch(&command_buffer, &outputs[0]);
    assert_eq!(
        entries(&command_buffer),
        before,
        "the third encoder held again what the first held"
    );
    let folded = device.work_counts().encoders_folded - counts.encoders_folded;
    assert_eq!(folded, 2, "the three encoders are one group");
    let failure = CommitFailure {
        code: 2,
        encoders: FailedEncoders::Set(BTreeSet::from([1])),
        signals: true,
    };
    command_buffer.commit_failed(failure).unwrap();
    let (_device, _state, watched) = kernel.release_buffers();
    assert_ne!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(
        watched.live(),
        "a buffer the encoders encoded again use went before they ran"
    );
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(2))
    );
    let reencoded = device.work_counts().encoders_reencoded - counts.encoders_reencoded;
    assert_eq!(reencoded, 2, "the first and third encoded again apart");
    assert_eq!(words_of(&outputs[0]), written());
    assert!(words_of(&outputs[1]).iter().all(|&word| word == 0));
    watched.gone();
}
