//! The locks of its command buffer's state an encoder's thread takes as the encoder opens and as
//! it ends (`EncoderCore::open`, `EncoderCore::end_with_replay`): one each, for a compute or a
//! blit encoder opened after a compute encoder of the same command buffer. Counted by
//! `command_buffer::state_locks`, in builds with debug assertions.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`).

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use crate::command_buffer::state_locks;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState,
    Device, ResourceOptions, Size,
};

/// The words of each buffer: one threadgroup.
const WORDS: u64 = 64;

/// The kernel's state and its input, output and scale.
struct Kernel {
    device: Device,
    state: ComputePipelineState,
    buffers: [Buffer; 3],
}

impl Kernel {
    /// The kernel, its pipeline made by a command buffer that ran it.
    fn new(scratch: &Scratch) -> Kernel {
        let device = device(&scratch.0);
        let state = state(&device);
        let buffer = || {
            device
                .new_buffer(WORDS * 4, ResourceOptions::shared())
                .unwrap()
        };
        let buffers = [buffer(), buffer(), buffer()];
        let kernel = Kernel {
            device,
            state,
            buffers,
        };
        let queue = kernel.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        kernel.dispatch(&command_buffer);
        run(&command_buffer);
        kernel
    }

    /// Bind the state and buffers on `encoder` and dispatch the kernel once.
    fn encode(&self, encoder: &mut ComputeCommandEncoder) {
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        for (index, buffer) in (0..).zip(&self.buffers) {
            encoder.set_buffer(Some(buffer), 0, index).unwrap();
        }
        encoder
            .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(WORDS, 1, 1))
            .unwrap();
    }

    /// A compute encoder of `command_buffer` of one dispatch, ended.
    fn dispatch(&self, command_buffer: &CommandBuffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        self.encode(&mut encoder);
        encoder.end_encoding().unwrap();
    }
}

/// Commit `command_buffer` and wait for it: it completes with no error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// A compute encoder opened after another compute encoder locks the state once as it opens and
/// once as it ends.
#[test]
fn a_compute_encoder_after_another_locks_the_state_once_at_its_open_and_once_at_its_end() {
    let scratch = Scratch::new("state-locks-compute");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer);
    let before = state_locks();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let open = state_locks() - before;
    kernel.encode(&mut encoder);
    let before = state_locks();
    encoder.end_encoding().unwrap();
    let end = state_locks() - before;
    assert_eq!(
        (open, end),
        (1, 1),
        "the state locks at its open and its end"
    );
    run(&command_buffer);
}

/// A blit encoder opened after a compute encoder locks the state once as it opens and once as it
/// ends.
#[test]
fn a_blit_encoder_after_a_compute_encoder_locks_the_state_once_at_its_open_and_once_at_its_end() {
    let scratch = Scratch::new("state-locks-blit");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer);
    let before = state_locks();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let open = state_locks() - before;
    blit.fill_buffer(&kernel.buffers[1], 0..WORDS * 4, 0x5a)
        .unwrap();
    let before = state_locks();
    blit.end_encoding().unwrap();
    let end = state_locks() - before;
    assert_eq!(
        (open, end),
        (1, 1),
        "the state locks at its open and its end"
    );
    run(&command_buffer);
}
