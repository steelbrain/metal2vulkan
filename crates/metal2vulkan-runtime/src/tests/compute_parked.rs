//! A compute encoder's end parks the box of its core and its bind state, with what its last
//! dispatch settled, in its command buffer, and the command buffer's next compute encoder opens on
//! them (`compute/live.rs`'s **Parked in its command buffer**): a dispatch whose new table brings in
//! the very state the last compute encoder's dispatch bound takes what that one settled
//! (`WorkCounts::dispatches_settled_across_encoders`), and one that binds otherwise settles again.
//! The commit gives them back to the device, so each command buffer's first compute encoder opens on
//! the device's spares (`WorkCounts::parked_compute_taken` counts the others).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), with a scale of 1 and an input of `i`, so each word it
//! writes is `2 * i`.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::words_of;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, ResourceOptions,
    Size, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;

/// The words a dispatch writes, one threadgroup's.
pub(super) const WORDS: u64 = 64;

/// The compute encoders of a command buffer.
const ENCODERS: u64 = 8;

/// The byte a blit encoder fills with.
const FILL: u8 = 0x5a;

/// The device, the kernel's state, its scale of 1 and its input `0, 1, 2, ...` as floats.
pub(super) struct Kernel {
    pub(super) device: Device,
    state: ComputePipelineState,
    scale: Buffer,
    input: Buffer,
}

impl Kernel {
    pub(super) fn new(scratch: &Scratch) -> Kernel {
        let device = device(&scratch.0);
        let state = state(&device);
        let scale =
            (device.new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())).unwrap();
        let input: Vec<u8> = (0..WORDS).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let input = (device.new_buffer_with_bytes(&input, ResourceOptions::shared())).unwrap();
        Kernel {
            device,
            state,
            scale,
            input,
        }
    }

    /// A `Shared` buffer of `count` dispatches' words of zeros.
    pub(super) fn output(&self, count: u64) -> Buffer {
        let bytes = vec![0; (count * WORDS * 4) as usize];
        (self
            .device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared()))
        .unwrap()
    }

    /// A compute encoder of `command_buffer` of one dispatch that writes `output` from byte
    /// `offset` on, ended.
    pub(super) fn dispatch(&self, command_buffer: &CommandBuffer, output: (&Buffer, u64)) {
        self.dispatches(command_buffer, output, 1);
    }

    /// [`dispatch`](Self::dispatch) with `count` dispatches of the same words in the encoder.
    pub(super) fn dispatches(
        &self,
        command_buffer: &CommandBuffer,
        (output, offset): (&Buffer, u64),
        count: u32,
    ) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&self.input), 0, 0).unwrap();
        encoder.set_buffer(Some(output), offset, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        for _ in 0..count {
            encoder
                .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(WORDS, 1, 1))
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// A blit encoder of `command_buffer` that fills `output`'s first dispatch's words with
    /// [`FILL`], ended.
    fn fill(&self, command_buffer: &CommandBuffer, output: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(output, 0..WORDS * 4, FILL).unwrap();
        blit.end_encoding().unwrap();
    }

    /// Commit `command_buffer` and wait for it: it completes with no error.
    pub(super) fn run(&self, command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?}",
            command_buffer.error()
        );
    }

    /// Warm: the kernel's pipelines made, so each dispatch below records at once.
    pub(super) fn warm(&self) {
        let output = self.output(1);
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        self.dispatch(&command_buffer, (&output, 0));
        self.run(&command_buffer);
    }
}

/// What moved of the counts the parking names, from `before` to `after`.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// `parked_compute_taken`.
    parked: u64,
    /// `dispatches_settled_across_encoders`.
    across: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        Moved {
            parked: after.parked_compute_taken - before.parked_compute_taken,
            across: after.dispatches_settled_across_encoders
                - before.dispatches_settled_across_encoders,
        }
    }
}

/// The words `2 * i` of `count` dispatches, as floats' bits.
pub(super) fn written(count: u64) -> Vec<u32> {
    (0..count)
        .flat_map(|_| (0..WORDS).map(|i| (2.0 * i as f32).to_bits()))
        .collect()
}

/// Command buffers of [`ENCODERS`] compute encoders of the same bindings, each after a blit
/// encoder that fills their output, on a device with a live sampler state the kernel binds none
/// of (each dispatch lists it among the device's, and a parked entry lets it go): every compute
/// encoder but each command buffer's first opens on what the one before it parked and takes what
/// its dispatch settled, and the last one's words are what it wrote, after the fill before it.
#[test]
fn compute_encoders_of_the_same_bindings_settle_once_a_command_buffer() {
    let scratch = Scratch::new("parked-same");
    let kernel = Kernel::new(&scratch);
    let _sampler = (kernel.device.new_sampler_state(&SamplerDescriptor::new())).unwrap();
    kernel.warm();
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(1);
    for round in 0..2 {
        let before = kernel.device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..ENCODERS {
            kernel.fill(&command_buffer, &output);
            kernel.dispatch(&command_buffer, (&output, 0));
        }
        let moved = Moved::of(&before, &kernel.device.work_counts());
        kernel.run(&command_buffer);
        assert_eq!(words_of(&output), written(1), "round {round}: the words");
        let want = Moved {
            parked: ENCODERS - 1,
            across: ENCODERS - 1,
        };
        assert_eq!(moved, want, "round {round}: the parked and settled across");
    }
}

/// A command buffer of [`ENCODERS`] compute encoders that each write their own words of one
/// output: each but the first opens on what the one before it parked, and settles again, its
/// output's offset moved; every dispatch's words are what it wrote.
#[test]
fn a_compute_encoder_that_binds_another_offset_settles_again() {
    let scratch = Scratch::new("parked-moved");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(ENCODERS);
    let before = kernel.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for k in 0..ENCODERS {
        kernel.dispatch(&command_buffer, (&output, k * WORDS * 4));
    }
    let moved = Moved::of(&before, &kernel.device.work_counts());
    kernel.run(&command_buffer);
    assert_eq!(words_of(&output), written(ENCODERS), "the words");
    let want = Moved {
        parked: ENCODERS - 1,
        across: 0,
    };
    assert_eq!(moved, want, "the parked and settled across");
}

/// A compute encoder's core is made in the box the command buffer's last compute encoder parked
/// (`EncoderCore::open_placed`): the next encoder's core lies at the address where the last one's
/// lay, encoder after encoder, and each dispatch writes its words.
#[test]
fn a_compute_encoder_core_lies_in_the_box_its_command_buffer_parked() {
    let scratch = Scratch::new("parked-box");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(1);
    let command_buffer = queue.command_buffer().unwrap();
    let mut lay = None;
    for k in 0..3 {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let at = std::ptr::from_mut(encoder.core()).addr();
        if let Some(lay) = lay {
            assert_eq!(
                at, lay,
                "encoder {k}: its core lies in the box the last one parked"
            );
        }
        lay = Some(at);
        encoder.end_encoding().unwrap();
    }
    kernel.dispatch(&command_buffer, (&output, 0));
    kernel.run(&command_buffer);
    assert_eq!(words_of(&output), written(1), "the words");
}
