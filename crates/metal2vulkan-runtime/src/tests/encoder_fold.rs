//! Consecutive compute and blit encoders of one command buffer recorded into one
//! `VkCommandBuffer` (`encoder_fold.rs`): which encoders join a group and what ends one, the
//! barrier the tracker's rule records between two encoders of a group, a failure or work a
//! command-predication record leaves out that splits a group at the commit, a rebind that
//! encodes a group again whole, fences inside a group and across queues, and the refusals'
//! counts.
//!
//! [`dependent_encoders_of_a_group_read_what_the_encoders_before_them_wrote`] is the barrier's
//! check: each small encoder reads the tail of the large write the encoder before it made, which
//! a GPU that overlaps two dispatches with no barrier between them may read before it is written;
//! [`compute_and_blit_encoders_of_a_chain_record_into_one_command_buffer`] the same through a
//! blit encoder's copy between them.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), with a scale of 1.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::words_of;
use super::rebind_behind_a_wait::{Page, BYTES};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommitFailure,
    ComputePipelineState, Device, EncoderKind, Error, EventRef, FailedEncoders, Fence,
    HazardTrackingMode, Origin, ResourceOptions, Size, Submission, SubmissionCheck, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{DispatchType, PixelFormat, TextureUsage};
use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The words of a large write: threadgroups enough that a GPU runs them in many waves.
const LARGE: u64 = 1 << 20;

/// The words of a small read: one threadgroup.
const SMALL: u64 = 64;

/// The threads of a threadgroup.
const GROUP: u64 = 64;

/// The large writes of the chain: each word of the `k`th's is `(k + 1) * i`, below `2^24`, so
/// every float is exact.
const STEPS: u64 = 12;

/// The byte a blit encoder fills with.
const FILL: u8 = 0x5a;

/// A `Shared` buffer of `words` floats, word `i` `value(i)`.
fn floats(device: &Device, words: u64, value: impl Fn(u64) -> f32) -> Buffer {
    let bytes: Vec<u8> = (0..words).flat_map(|i| value(i).to_le_bytes()).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// An untracked `Shared` buffer of `words` floats, all 0: only a fence orders its commands
/// across encoders.
fn untracked(device: &Device, words: u64) -> Buffer {
    let options =
        ResourceOptions::shared().with_hazard_tracking_mode(HazardTrackingMode::Untracked);
    let bytes = vec![0u8; (words * 4) as usize];
    device.new_buffer_with_bytes(&bytes, options).unwrap()
}

/// The floats of `buffer`, a host-visible buffer no command writes now.
fn read(buffer: &Buffer) -> Vec<f32> {
    words_of(buffer).into_iter().map(f32::from_bits).collect()
}

/// `words` floats, word `i` `value(i)`.
fn expected(words: u64, value: impl Fn(u64) -> u64) -> Vec<f32> {
    (0..words).map(|i| value(i) as f32).collect()
}

/// The device, the kernel's state and its scale of 1.
struct Kernel {
    device: Device,
    state: ComputePipelineState,
    scale: Buffer,
}

impl Kernel {
    fn new(scratch: &Scratch) -> Kernel {
        let device = device(&scratch.0);
        let state = state(&device);
        let scale = device
            .new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())
            .unwrap();
        Kernel {
            device,
            state,
            scale,
        }
    }

    /// An encoder of `dispatch_type` of one dispatch over `words` threads, reading `input` from
    /// byte `offset` on and writing `output`, ended.
    fn dispatch(
        &self,
        command_buffer: &CommandBuffer,
        dispatch_type: DispatchType,
        (input, offset): (&Buffer, u64),
        (output, words): (&Buffer, u64),
    ) {
        let mut encoder = command_buffer
            .compute_command_encoder_with_dispatch_type(dispatch_type)
            .unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(input), offset, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(words, 1, 1), Size::new(GROUP, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// [`dispatch`](Self::dispatch) of a serial encoder that first waits for `wait` and last
    /// updates `update`, where given.
    fn fenced(
        &self,
        command_buffer: &CommandBuffer,
        (wait, update): (Option<&Fence>, Option<&Fence>),
        (input, offset): (&Buffer, u64),
        (output, words): (&Buffer, u64),
    ) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        if let Some(fence) = wait {
            encoder.wait_for_fence(fence).unwrap();
        }
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(input), offset, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(words, 1, 1), Size::new(GROUP, 1, 1))
            .unwrap();
        if let Some(fence) = update {
            encoder.update_fence(fence).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// A blit encoder of `command_buffer` that copies `words` floats of `source` from byte
    /// `offset` on into `destination`, ended.
    fn copy(
        &self,
        command_buffer: &CommandBuffer,
        (source, offset): (&Buffer, u64),
        (destination, words): (&Buffer, u64),
    ) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(source, offset, destination, 0, words * 4)
            .unwrap();
        blit.end_encoding().unwrap();
    }

    /// A blit encoder of `command_buffer` that fills `buffer`'s first `words` floats' bytes with
    /// [`FILL`], ended.
    fn fill(&self, command_buffer: &CommandBuffer, buffer: &Buffer, words: u64) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, 0..words * 4, FILL).unwrap();
        blit.end_encoding().unwrap();
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

/// What moved of the counts the groups name, from `before` to `after`.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// `encoders_folded`.
    folded: u64,
    /// `command_buffers_begun_encoders`.
    begun: u64,
    /// `command_barriers`.
    barriers: u64,
    /// `fold_fence_barriers`.
    fence_barriers: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        Moved {
            folded: after.encoders_folded - before.encoders_folded,
            begun: after.command_buffers_begun_encoders - before.command_buffers_begun_encoders,
            barriers: after.command_barriers - before.command_barriers,
            fence_barriers: after.fold_fence_barriers - before.fold_fence_barriers,
        }
    }
}

/// A chain of `STEPS` large writes, each from the buffer the one before wrote into the other,
/// each followed by a small read of the tail of what it wrote, each one encoder: one group of
/// `2 * STEPS` encoders in one `VkCommandBuffer`, each small read after a barrier from the large
/// write before it, and every word what the chain computes.
#[test]
fn dependent_encoders_of_a_group_read_what_the_encoders_before_them_wrote() {
    let scratch = Scratch::new("fold-chain");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let tail = (LARGE - SMALL) * 4;
    let large = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
    ];
    let small: Vec<Buffer> = (0..STEPS).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    // Warm, over buffers of its own: the shapes' pipelines are made, so the chain below records
    // each dispatch at once, leaving no segment that would end the group.
    let warm = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let command_buffer = queue.command_buffer().unwrap();
    let serial = DispatchType::Serial;
    kernel.dispatch(&command_buffer, serial, (&warm[0], 0), (&warm[1], LARGE));
    kernel.dispatch(&command_buffer, serial, (&warm[1], tail), (&warm[2], SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for k in 0..STEPS as usize {
        let (from, to) = (&large[k % 2], &large[(k + 1) % 2]);
        kernel.dispatch(&command_buffer, serial, (from, 0), (to, LARGE));
        kernel.dispatch(&command_buffer, serial, (to, tail), (&small[k], SMALL));
    }
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    // The words first: what a missing barrier changes.
    let wrong: Vec<(u64, usize)> = (small.iter().enumerate())
        .map(|(k, small)| {
            let k = k as u64;
            let value = |j| (k + 1) * (LARGE - SMALL + j) + j;
            let read = read(small);
            let wanted = expected(SMALL, value);
            (k, read.iter().zip(&wanted).filter(|(a, b)| a != b).count())
        })
        .filter(|&(_, wrong)| wrong > 0)
        .collect();
    let last = &large[(STEPS % 2) as usize];
    let wrong_last = (read(last).iter().zip(expected(LARGE, |i| STEPS * i)))
        .filter(|(a, b)| *a != b)
        .count();
    println!("FOLD-CHAIN {moved:?} small reads wrong (write, words) {wrong:?} last write's words wrong {wrong_last}");
    assert_eq!(
        (wrong, wrong_last),
        (Vec::new(), 0),
        "the small reads read what the large writes before them wrote"
    );
    let encoders = 2 * STEPS;
    assert_eq!(
        (moved.folded, moved.begun),
        (encoders - 1, 1),
        "every encoder after the first joined the first's group: {moved:?}"
    );
    assert!(
        moved.barriers >= STEPS,
        "each small read waits for the large write before it: {moved:?}"
    );
}

/// Encoders that read one buffer and each write a buffer of its own share no written resource:
/// one group with no barrier inside it.
#[test]
fn encoders_that_share_no_written_resource_record_no_barrier_between_them() {
    let scratch = Scratch::new("fold-apart");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs: Vec<Buffer> = (0..8).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    let warm = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for output in &outputs {
        kernel.dispatch(&command_buffer, serial, (&input, 0), (output, SMALL));
    }
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    let folded = outputs.len() as u64 - 1;
    assert_eq!(
        moved,
        Moved {
            folded,
            begun: 1,
            barriers: 0,
            fence_barriers: 0,
        }
    );
    for output in &outputs {
        assert_eq!(read(output), expected(SMALL, |i| 2 * i));
    }
}

/// A group holds encoders of one dispatch type: a serial encoder after a concurrent one, or a
/// concurrent one after a serial one, begins a group of its own.
#[test]
fn a_change_of_dispatch_type_begins_a_group() {
    let scratch = Scratch::new("fold-types");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let (serial, concurrent) = (DispatchType::Serial, DispatchType::Concurrent);
    let types = [serial, serial, concurrent, concurrent, serial];
    let outputs: Vec<Buffer> = (types.iter())
        .map(|_| floats(device, SMALL, |_| 0.0))
        .collect();
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.dispatch(&command_buffer, concurrent, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for (&dispatch_type, output) in types.iter().zip(&outputs) {
        kernel.dispatch(&command_buffer, dispatch_type, (&input, 0), (output, SMALL));
    }
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    assert_eq!((moved.folded, moved.begun), (2, 3), "{moved:?}");
    assert_eq!(
        after.folds_refused_kind - before.folds_refused_kind,
        2,
        "each change of dispatch type is a refusal of the group's kind"
    );
    for output in &outputs {
        assert_eq!(read(output), expected(SMALL, |i| 2 * i));
    }
}

/// An encoder that updates a fence stays in its group: the encoder that waits for the fence joins
/// it after a barrier from everything before (`encoder_fold.rs`'s **Fences**), and the encoder
/// after that one joins it too, with no such barrier. Each reads what the one before wrote.
#[test]
fn a_fence_update_stays_in_its_group() {
    let scratch = Scratch::new("fold-fence");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let buffers: Vec<Buffer> = (0..3).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    let fence = device.new_fence();
    let warm = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.fenced(
        &command_buffer,
        (None, Some(&fence)),
        (&input, 0),
        (&buffers[0], SMALL),
    );
    kernel.fenced(
        &command_buffer,
        (Some(&fence), None),
        (&buffers[0], 0),
        (&buffers[1], SMALL),
    );
    kernel.dispatch(
        &command_buffer,
        serial,
        (&buffers[1], 0),
        (&buffers[2], SMALL),
    );
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!(
        (moved.folded, moved.begun, moved.fence_barriers),
        (2, 1, 1),
        "{moved:?}"
    );
    for (at, buffer) in buffers.iter().enumerate() {
        let times = at as u64 + 2;
        assert_eq!(read(buffer), expected(SMALL, |i| times * i), "buffer {at}");
    }
}

/// A chain of `STEPS` large writes through untracked buffers, each encoder waiting for the fence
/// the one before updated and updating it, each followed by a small read of the tail of what it
/// wrote, also fenced: only the fence orders them. One group of `2 * STEPS` encoders in one
/// `VkCommandBuffer`, each after a barrier from everything before, and every word what the chain
/// computes.
#[test]
fn a_fenced_chain_over_untracked_buffers_records_into_one_command_buffer() {
    let scratch = Scratch::new("fold-fenced-chain");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let tail = (LARGE - SMALL) * 4;
    let large = [untracked(device, LARGE), untracked(device, LARGE)];
    let small: Vec<Buffer> = (0..STEPS).map(|_| untracked(device, SMALL)).collect();
    let fence = device.new_fence();
    let warm = [
        untracked(device, LARGE),
        untracked(device, LARGE),
        untracked(device, SMALL),
    ];
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&warm[0], 0), (&warm[1], LARGE));
    kernel.dispatch(&command_buffer, serial, (&warm[1], tail), (&warm[2], SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let fences = (Some(&fence), Some(&fence));
    for k in 0..STEPS as usize {
        let (from, to) = (&large[k % 2], &large[(k + 1) % 2]);
        kernel.fenced(&command_buffer, fences, (from, 0), (to, LARGE));
        kernel.fenced(&command_buffer, fences, (to, tail), (&small[k], SMALL));
    }
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    // The words first: what a missing barrier changes. Step `k` writes `(k + 1) * i`.
    let wrong: Vec<(u64, usize)> = (small.iter().enumerate())
        .map(|(k, small)| {
            let k = k as u64;
            let value = |j| (k + 1) * (LARGE - SMALL + j) + j;
            let read = read(small);
            let wanted = expected(SMALL, value);
            (k, read.iter().zip(&wanted).filter(|(a, b)| a != b).count())
        })
        .filter(|&(_, wrong)| wrong > 0)
        .collect();
    let last = &large[(STEPS % 2) as usize];
    let wrong_last = (read(last).iter().zip(expected(LARGE, |i| STEPS * i)))
        .filter(|(a, b)| *a != b)
        .count();
    println!(
        "FOLD-FENCED-CHAIN {moved:?} small reads wrong (step, words) {wrong:?} last write's \
         words wrong {wrong_last}"
    );
    assert_eq!(
        (wrong, wrong_last),
        (Vec::new(), 0),
        "each encoder read what the one before wrote"
    );
    let encoders = 2 * STEPS;
    assert_eq!(
        (moved.folded, moved.begun, moved.fence_barriers),
        (encoders - 1, 1, encoders - 1),
        "every encoder after the first joined the first's group after a barrier: {moved:?}"
    );
}

/// A group of two encoders on one queue, the first writing an untracked buffer and updating a
/// fence, the second joining it; a command buffer of another queue committed after it waits for
/// the fence and reads what the first wrote: the wait orders after the whole group.
#[test]
fn a_fence_updated_inside_a_group_orders_another_queue_s_wait() {
    let scratch = Scratch::new("fold-fence-queues");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let (writer, reader) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let tail = (LARGE - SMALL) * 4;
    let input = floats(device, LARGE, |i| i as f32);
    let written = untracked(device, LARGE);
    let other = untracked(device, SMALL);
    let read_back = untracked(device, SMALL);
    let fence = device.new_fence();
    let warm = [untracked(device, LARGE), untracked(device, SMALL)];
    let serial = DispatchType::Serial;
    let command_buffer = writer.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm[0], LARGE));
    kernel.dispatch(&command_buffer, serial, (&warm[0], tail), (&warm[1], SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let first = writer.command_buffer().unwrap();
    kernel.fenced(&first, (None, Some(&fence)), (&input, 0), (&written, LARGE));
    kernel.dispatch(&first, serial, (&input, 0), (&other, SMALL));
    let second = reader.command_buffer().unwrap();
    kernel.fenced(
        &second,
        (Some(&fence), None),
        (&written, tail),
        (&read_back, SMALL),
    );
    first.commit().unwrap();
    kernel.run(&second);
    first.wait_until_completed_for(WAIT).unwrap();
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!(
        read(&read_back),
        expected(SMALL, |j| 2 * (LARGE - SMALL + j) + j),
        "the other queue's read after the wait read what the group's first encoder wrote"
    );
    assert_eq!(read(&other), expected(SMALL, |i| 2 * i));
    assert_eq!(
        (moved.folded, moved.begun, moved.fence_barriers),
        (1, 2, 1),
        "the writer's two encoders in one group, the reader's alone: {moved:?}"
    );
}

/// Three serial compute encoders and a blit encoder's fill of one group, and a concurrent compute
/// encoder after it, which begins a group of its own, committed as failed with `encoders`: which
/// of the five ran, read from what each wrote, and how many encoders the commit encoded again
/// (`WorkCounts::encoders_reencoded`).
fn failed_with(encoders: FailedEncoders) -> ([bool; 5], u64) {
    let scratch = Scratch::new("fold-failed");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs: Vec<Buffer> = (0..5).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    let warm = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for output in &outputs[..3] {
        kernel.dispatch(&command_buffer, serial, (&input, 0), (output, SMALL));
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&outputs[3], 0..SMALL * 4, FILL).unwrap();
    blit.end_encoding().unwrap();
    let concurrent = DispatchType::Concurrent;
    kernel.dispatch(
        &command_buffer,
        concurrent,
        (&input, 0),
        (&outputs[4], SMALL),
    );
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (3, 2), "{moved:?}");
    let failure = CommitFailure {
        code: 2,
        encoders,
        signals: true,
    };
    command_buffer.commit_failed(failure).unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(2))
    );
    let after = device.work_counts();
    for (at, output) in outputs.iter().enumerate() {
        let words = words_of(output);
        let ran = words.iter().any(|&word| word != 0);
        let whole = match at {
            3 => words
                .iter()
                .all(|&word| word == u32::from_le_bytes([FILL; 4])),
            _ => read(output) == expected(SMALL, |i| 2 * i),
        };
        assert!(!ran || whole, "encoder {at} ran in part: {words:?}");
    }
    let ran = |at: usize| words_of(&outputs[at]).iter().any(|&word| word != 0);
    let reencoded = after.encoders_reencoded - before.encoders_reencoded;
    ([ran(0), ran(1), ran(2), ran(3), ran(4)], reencoded)
}

/// Three compute encoders and a blit encoder's fill of one group, with a submission check that
/// fails the blit encoder's submission (`command_buffer/checks.rs`): the group's
/// `VkCommandBuffer` holds commands of two of Metal's submissions, so the commit encodes its
/// encoders again apart (`State::split_failure`), and the check may run on the scheduler's
/// thread before those encodings take their places. Here it does: the test holds the command
/// buffer's turn of encodings again until the check's failure of the blit's submission is noted
/// (`State::fail_submission`, after the check returns). The blit encoder then takes no encoding
/// again and runs nothing; the compute encoders take theirs and run.
#[test]
fn a_check_that_fails_a_submission_before_its_group_is_split_keeps_it_failed() {
    let scratch = Scratch::new("fold-checked-early");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs: Vec<Buffer> = (0..4).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    let warm = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for output in &outputs[..3] {
        kernel.dispatch(&command_buffer, serial, (&input, 0), (output, SMALL));
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&outputs[3], 0..SMALL * 4, FILL).unwrap();
    blit.end_encoding().unwrap();
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (3, 1), "{moved:?}");
    let check: SubmissionCheck = Arc::new(move |submission: &Submission<'_>| {
        (submission.kind() == EncoderKind::Blit).then(|| NonZeroU64::new(8).unwrap())
    });
    command_buffer.set_submission_check(check).unwrap();
    let turn = command_buffer.hold_reencoding();
    let committer = {
        let command_buffer = command_buffer.clone();
        std::thread::spawn(move || command_buffer.commit())
    };
    let waiting = Instant::now();
    while !command_buffer.failure_noted() {
        assert!(
            waiting.elapsed() < WAIT,
            "the check failed the blit's submission while the encodings again waited for their turn"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(turn);
    committer.join().unwrap().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(8))
    );
    let after = device.work_counts();
    assert_eq!(after.encoders_reencoded - before.encoders_reencoded, 3);
    for output in &outputs[..3] {
        assert_eq!(read(output), expected(SMALL, |i| 2 * i));
    }
    assert!(
        words_of(&outputs[3]).iter().all(|&word| word == 0),
        "the blit encoder's fill"
    );
}

/// A failed set that names one encoder of a group fails that encoder alone, as on Metal: the
/// group's other three, the blit encoder among them, are encoded again apart at the commit, and
/// the encoder after it runs.
#[test]
fn a_failed_encoder_inside_a_group_fails_alone() {
    let set = FailedEncoders::Set(BTreeSet::from([1]));
    assert_eq!(failed_with(set), ([true, false, true, true, true], 3));
}

/// A failed set that names a group's blit encoder fails it alone: the group's compute encoders are
/// encoded again apart at the commit.
#[test]
fn a_failed_blit_encoder_inside_a_group_fails_alone() {
    let set = FailedEncoders::Set(BTreeSet::from([3]));
    assert_eq!(failed_with(set), ([true, true, true, false, true], 3));
}

/// A first failed encoder at a group's blit encoder fails from there on: the group's compute
/// encoders before it are encoded again apart at the commit, and run.
#[test]
fn a_first_failed_blit_encoder_inside_a_group_runs_the_encoders_before_it() {
    let from = FailedEncoders::From(3);
    assert_eq!(failed_with(from), ([true, true, true, false, false], 3));
}

/// A first failed encoder inside a group fails from there on, as on Metal: the group's first
/// encoder is encoded again apart at the commit, and runs.
#[test]
fn a_first_failed_encoder_inside_a_group_runs_the_group_s_encoders_before_it() {
    let from = FailedEncoders::From(1);
    assert_eq!(failed_with(from), ([true, false, false, false, false], 1));
}

/// A first failed encoder at a group's boundary (an encoder of the other dispatch type) splits no
/// group: the group before it runs as recorded.
#[test]
fn a_first_failed_encoder_at_a_group_s_boundary_fails_from_there() {
    let from = FailedEncoders::From(4);
    assert_eq!(failed_with(from), ([true, true, true, true, false], 0));
}

/// The shape of a consumer's protection probe: a compute encoder, then a second one, folded into
/// the first's group, the command buffer committed as failed from the second (code 16, no
/// signals), on a device with no buffer over host pages: the first encoder's words land and the
/// second's do not, as on Metal.
#[test]
fn a_failure_from_a_folded_encoder_runs_the_encoder_before_it() {
    let scratch = Scratch::new("fold-failed-second");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let warm = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for output in &outputs {
        kernel.dispatch(&command_buffer, serial, (&input, 0), (output, SMALL));
    }
    let folded = device.work_counts().encoders_folded - before.encoders_folded;
    assert_eq!(folded, 1, "the second encoder joined the first's group");
    let failure = CommitFailure {
        code: 16,
        encoders: FailedEncoders::From(1),
        signals: false,
    };
    command_buffer.commit_failed(failure).unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::Injected(16))
        )
    );
    assert_eq!(
        read(&outputs[0]),
        expected(SMALL, |i| 2 * i),
        "the first encoder ran"
    );
    assert_eq!(
        words_of(&outputs[1]),
        vec![0; SMALL as usize],
        "the failed encoder ran nothing"
    );
    assert_eq!(after.encoders_reencoded - before.encoders_reencoded, 1);
}

/// A chain of six dependent encoders in one group, each reading what the one before wrote,
/// committed as failed from the fifth: the four before it run, encoded again apart at the commit,
/// each after the one before; the last two run nothing.
#[test]
fn a_split_chain_runs_its_first_encoders_in_order() {
    let scratch = Scratch::new("fold-failed-chain");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let large = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
    ];
    let warm = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
    ];
    let serial = DispatchType::Serial;
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&warm[0], 0), (&warm[1], LARGE));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for k in 0..6 {
        let (from, to) = (&large[k % 2], &large[(k + 1) % 2]);
        kernel.dispatch(&command_buffer, serial, (from, 0), (to, LARGE));
    }
    let folded = device.work_counts().encoders_folded - before.encoders_folded;
    assert_eq!(folded, 5);
    let failure = CommitFailure {
        code: 2,
        encoders: FailedEncoders::From(4),
        signals: true,
    };
    command_buffer.commit_failed(failure).unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    // Step `k` writes `(k + 1) * i` into `large[(k + 1) % 2]`: steps 0 to 3 ran.
    assert_eq!(read(&large[0]), expected(LARGE, |i| 4 * i), "step 3's");
    assert_eq!(read(&large[1]), expected(LARGE, |i| 3 * i), "step 2's");
    assert_eq!(after.encoders_reencoded - before.encoders_reencoded, 4);
}

/// A compute encoder's log holds one `ComputeCall` per call, at most 64 bytes: the bound
/// `encoder_fold.rs`'s **Logs** gives what a command buffer's groups hold until the commit by.
#[test]
fn a_logged_compute_call_is_at_most_64_bytes() {
    let size = std::mem::size_of::<crate::replay::ComputeCall>();
    assert!(size <= 64, "{size} bytes");
}

/// A group of two encoders, the second writing a buffer over host pages, committed behind an
/// event wait only the host signals, the buffer then rebound before the signal: the rebind
/// encodes both encoders again, each into `VkCommandBuffer`s of its own, and the command buffer
/// writes the pages the buffer lies in at its submission.
#[test]
fn a_rebind_encodes_every_encoder_of_a_group_again() {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("fold-rebind");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let words = (BYTES / 4) as u64;
    let input = floats(device, words, |i| i as f32);
    let middle = floats(device, words, |_| 0.0);
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let output = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("FOLD-REBIND: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let serial = DispatchType::Serial;
    let warm = floats(device, words, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&middle, words));
    kernel.dispatch(&command_buffer, serial, (&middle, 0), (&warm, words));
    kernel.run(&command_buffer);

    let gate = device.new_shared_event().unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&middle, words));
    kernel.dispatch(&command_buffer, serial, (&middle, 0), (&output, words));
    command_buffer.commit().unwrap();
    assert_eq!(output.detach_backing(), Ok(true));
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    // Read before the signal: the rebind encodes again in its own call.
    let after = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    assert_eq!(after.encoders_folded - before.encoders_folded, 1);
    assert_eq!(
        (
            after.encoders_reencoded - before.encoders_reencoded,
            after.reencodes_refused - before.reencodes_refused
        ),
        (2, 0),
        "the rebind encoded both encoders of the group again"
    );
    assert!(
        !pages[0].written(),
        "the group wrote the pages the buffer left"
    );
    let written: Vec<f32> = pages[1]
        .words()
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    assert_eq!(written, expected(words, |i| 3 * i));
}

/// A chain of `STEPS` large writes, each from the buffer the one before wrote into the other,
/// each followed by a blit encoder's copy of the tail of what it wrote and a small read of that
/// copy, each one encoder: one group of `3 * STEPS` compute and blit encoders in one
/// `VkCommandBuffer`, each copy after a barrier from the large write before it and each small read
/// after one from the copy, and every word what the chain computes.
#[test]
fn compute_and_blit_encoders_of_a_chain_record_into_one_command_buffer() {
    let scratch = Scratch::new("fold-blit-chain");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let tail = (LARGE - SMALL) * 4;
    let large = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
    ];
    let copies: Vec<Buffer> = (0..STEPS).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    let small: Vec<Buffer> = (0..STEPS).map(|_| floats(device, SMALL, |_| 0.0)).collect();
    // Warm, over buffers of its own, as the compute chain warms.
    let warm = [
        floats(device, LARGE, |_| 0.0),
        floats(device, LARGE, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let command_buffer = queue.command_buffer().unwrap();
    let serial = DispatchType::Serial;
    kernel.dispatch(&command_buffer, serial, (&warm[0], 0), (&warm[1], LARGE));
    kernel.dispatch(&command_buffer, serial, (&warm[1], tail), (&warm[2], SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for k in 0..STEPS as usize {
        let (from, to) = (&large[k % 2], &large[(k + 1) % 2]);
        kernel.dispatch(&command_buffer, serial, (from, 0), (to, LARGE));
        kernel.copy(&command_buffer, (to, tail), (&copies[k], SMALL));
        kernel.dispatch(&command_buffer, serial, (&copies[k], 0), (&small[k], SMALL));
    }
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    let submitted = after.command_buffers_submitted - before.command_buffers_submitted;
    // Copy `k` holds the tail of write `k`, `(k + 1) * i`; read `k` adds `j` to word `j` of it.
    let wrong = |buffers: &[Buffer], plus: u64| -> Vec<(u64, usize)> {
        (buffers.iter().enumerate())
            .map(|(k, buffer)| {
                let k = k as u64;
                let wanted = expected(SMALL, |j| (k + 1) * (LARGE - SMALL + j) + plus * j);
                let read = read(buffer);
                (k, read.iter().zip(&wanted).filter(|(a, b)| a != b).count())
            })
            .filter(|&(_, wrong)| wrong > 0)
            .collect()
    };
    let (wrong_copies, wrong_reads) = (wrong(&copies, 0), wrong(&small, 1));
    println!(
        "FOLD-BLIT-CHAIN {moved:?} submitted {submitted} copies wrong (write, words) \
         {wrong_copies:?} reads wrong {wrong_reads:?}"
    );
    assert_eq!(
        (wrong_copies, wrong_reads),
        (Vec::new(), Vec::new()),
        "each copy read what the write before it wrote, each read what the copy before it wrote"
    );
    let encoders = 3 * STEPS;
    assert_eq!(
        (moved.folded, moved.begun),
        (encoders - 1, 1),
        "every encoder after the first joined the first's group: {moved:?}"
    );
    assert!(
        moved.barriers >= 2 * STEPS,
        "each copy waits for the write before it, each read for the copy: {moved:?}"
    );
}

/// An event signal between a compute encoder and a blit encoder ends the batch, and so the
/// group: the blit encoder begins a group of its own, which the compute encoder after it joins.
#[test]
fn an_event_signal_before_a_blit_encoder_begins_a_group() {
    let scratch = Scratch::new("fold-blit-event");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let copied = floats(device, SMALL, |_| 0.0);
    let event = device.new_shared_event().unwrap();
    let serial = DispatchType::Serial;
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&outputs[0], SMALL));
    command_buffer
        .encode_signal_event(EventRef::Shared(&event), 1)
        .unwrap();
    kernel.copy(&command_buffer, (&outputs[0], 0), (&copied, SMALL));
    kernel.dispatch(&command_buffer, serial, (&copied, 0), (&outputs[1], SMALL));
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    assert_eq!((moved.folded, moved.begun), (1, 2), "{moved:?}");
    assert_eq!(
        after.folds_refused_batch - before.folds_refused_batch,
        1,
        "the signal is a batch boundary the blit encoder's join is refused at"
    );
    assert_eq!(read(&copied), expected(SMALL, |i| 2 * i));
    assert_eq!(read(&outputs[1]), expected(SMALL, |i| 3 * i));
}

/// A group's compute encoders are of one dispatch type, a blit encoder between them or not: a
/// concurrent encoder after a serial encoder and a blit encoder begins a group of its own, which
/// the blit encoder after it joins.
#[test]
fn a_change_of_dispatch_type_across_a_blit_encoder_begins_a_group() {
    let scratch = Scratch::new("fold-blit-types");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let copies = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let (serial, concurrent) = (DispatchType::Serial, DispatchType::Concurrent);
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.dispatch(&command_buffer, concurrent, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for (dispatch_type, (output, copy)) in [serial, concurrent]
        .into_iter()
        .zip(outputs.iter().zip(&copies))
    {
        kernel.dispatch(&command_buffer, dispatch_type, (&input, 0), (output, SMALL));
        kernel.copy(&command_buffer, (output, 0), (copy, SMALL));
    }
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (2, 2), "{moved:?}");
    for copy in &copies {
        assert_eq!(read(copy), expected(SMALL, |i| 2 * i));
    }
}

/// No encoder joins a group once its command buffer used command predication: a blit encoder
/// after a compute encoder's If begins a `VkCommandBuffer` of its own, and so does the compute
/// encoder after it.
#[test]
fn no_blit_encoder_joins_a_group_after_command_predication() {
    let scratch = Scratch::new("fold-blit-predicated");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let copied = floats(device, SMALL, |_| 0.0);
    let word = device
        .new_buffer_with_bytes(&0u32.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    let serial = DispatchType::Serial;
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    // Comparison 0: `word == reference`, which holds.
    encoder.encode_start_if(Some(&word), 0, 0, 0).unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&outputs[0]), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(SMALL, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    assert!(encoder.encode_end_if().unwrap(), "the If ends");
    encoder.end_encoding().unwrap();
    kernel.copy(&command_buffer, (&outputs[0], 0), (&copied, SMALL));
    kernel.dispatch(&command_buffer, serial, (&copied, 0), (&outputs[1], SMALL));
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (0, 3), "{moved:?}");
    assert_eq!(read(&copied), expected(SMALL, |i| 2 * i));
    assert_eq!(read(&outputs[1]), expected(SMALL, |i| 3 * i));
}

/// A new `Private` texture of `SIDE`² `RGBA8Unorm` texels, shader-read.
fn new_texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The side of [`new_texture`]'s textures.
const SIDE: u64 = 16;

/// A blit encoder of `command_buffer` copying `source`, `SIDE`² texels of 4 bytes, into the whole
/// of `texture`, ended.
fn copy_whole(command_buffer: &CommandBuffer, source: &Buffer, texture: &Texture) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        source,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        Size::new(SIDE, SIDE, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

/// The bytes of `texture`, `SIDE`² texels of 4 bytes, copied out.
fn texels(device: &Device, texture: &Texture) -> Vec<u8> {
    let length = SIDE * SIDE * 4;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE * 4,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    // SAFETY: the shared buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length as usize) }
        .to_vec()
}

/// A blit encoder that writes a new `Private` texture whole before any of its commands reads it
/// (`Overwrite`, on a device that zero-fills new `Private` textures) ends its group: the compute
/// encoder after it begins a `VkCommandBuffer` of its own. Where every encoder of the group
/// before it is a blit encoder that named nothing of the texture, its write still leaves the
/// texture's zeros out (`WorkCounts::private_textures_overwritten_unfilled`), as it did in a
/// `VkCommandBuffer` of its own; after a compute encoder, whose commands may reach a texture
/// unnamed, it keeps them, as it did.
#[test]
fn a_blit_encoder_that_writes_a_new_texture_whole_ends_its_group() {
    let scratch = Scratch::new("fold-blit-overwrite");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let zero_fills = !device.properties().lossless_compression;
    let input = floats(device, SMALL, |i| i as f32);
    let output = floats(device, SMALL, |_| 0.0);
    let filled = floats(device, SMALL, |_| 0.0);
    let bytes: Vec<u8> = (0..SIDE * SIDE * 4).map(|i| (i * 37 + 11) as u8).collect();
    let source = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let serial = DispatchType::Serial;
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);
    let overwritten = |counts: &WorkCounts| counts.private_textures_overwritten_unfilled;

    // A fill, the copy into the whole texture, and a dispatch.
    let texture = new_texture(device);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.fill(&command_buffer, &filled, SMALL);
    copy_whole(&command_buffer, &source, &texture);
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&output, SMALL));
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    let left_out = overwritten(&after) - overwritten(&before);
    println!("FOLD-BLIT-OVERWRITE zero fills {zero_fills}: {moved:?} zeros left out {left_out}");
    let counted = u64::from(zero_fills);
    assert_eq!(
        (moved.folded, moved.begun, left_out),
        (2 - counted, 1 + counted, counted),
        "the copy joined the fill's group, and ended it where it wrote the texture whole"
    );
    assert_eq!(texels(device, &texture), bytes, "the copy");
    assert_eq!(read(&output), expected(SMALL, |i| 2 * i));

    // A dispatch, then the copy into the whole texture: it keeps the zeros.
    let texture = new_texture(device);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&output, SMALL));
    copy_whole(&command_buffer, &source, &texture);
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    assert_eq!(
        (
            moved.folded,
            moved.begun,
            overwritten(&after) - overwritten(&before)
        ),
        (1, 1, 0),
        "the copy joined the dispatch's group and took the zeros"
    );
    assert_eq!(
        texels(device, &texture),
        bytes,
        "the copy after the dispatch"
    );
}

/// A blit encoder that writes an ASTC texture's carrier, whose second twin's decode its batch's
/// submission records after its part (`texture/decoded_twin.rs`), ends its group: the compute
/// encoder after it begins a `VkCommandBuffer` of its own.
#[test]
fn a_blit_encoder_that_writes_an_astc_carrier_ends_its_group() {
    let scratch = Scratch::new("fold-blit-carrier");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let output = floats(device, SMALL, |_| 0.0);
    let filled = floats(device, SMALL, |_| 0.0);
    // Eight `ASTC_4x2` texels square, two 16-byte blocks a row of blocks, written in its top
    // half (a part of it, so no `Overwrite`).
    let side = 8;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::ASTC4x2LDR, side, side, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let carried = device.new_texture(&descriptor).unwrap();
    let blocks = device
        .new_buffer_with_bytes(&[0x5a; 64], ResourceOptions::shared())
        .unwrap();
    let serial = DispatchType::Serial;
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.fill(&command_buffer, &filled, SMALL);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &blocks,
        0,
        32,
        64,
        Size::new(side, side / 2, 1),
        &carried,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&output, SMALL));
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let moved = Moved::of(&before, &after);
    let checked = after.second_twin_checks - before.second_twin_checks;
    println!("FOLD-BLIT-CARRIER {moved:?} carrier writes checked {checked}");
    assert_eq!(checked, 1, "the copy wrote the carrier");
    assert_eq!(
        (moved.folded, moved.begun),
        (1, 2),
        "the copy joined the fill's group and ended it"
    );
    assert_eq!(read(&output), expected(SMALL, |i| 2 * i));
}

/// A group of a compute encoder, a blit encoder and a compute encoder that begins the next unit
/// of command predication (`predication.rs`: a blit encoder ends a unit), whose unbalanced End
/// leaves its unit's work out: the dispatch it leaves out lies in the group's `VkCommandBuffer`,
/// so the commit splits the group, encoding the first two encoders again apart, and the left out
/// dispatch runs nothing, as on Metal; the first two run.
#[test]
fn work_left_out_past_a_group_s_first_encoder_splits_the_group() {
    let scratch = Scratch::new("fold-blit-left-out");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let input = floats(device, SMALL, |i| i as f32);
    let outputs = [
        floats(device, SMALL, |_| 0.0),
        floats(device, SMALL, |_| 0.0),
    ];
    let copied = floats(device, SMALL, |_| 0.0);
    let serial = DispatchType::Serial;
    let warm = floats(device, SMALL, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&warm, SMALL));
    kernel.run(&command_buffer);

    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&outputs[0], SMALL));
    kernel.copy(&command_buffer, (&outputs[0], 0), (&copied, SMALL));
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&outputs[1]), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(SMALL, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    // No block is open: an unbalanced End, which answers NO and leaves the unit's work out.
    assert!(!encoder.encode_end_if().unwrap(), "the End is unbalanced");
    encoder.end_encoding().unwrap();
    let folded = device.work_counts().encoders_folded - before.encoders_folded;
    assert_eq!(
        folded, 2,
        "the blit and the second compute encoder joined the first's group"
    );
    kernel.run(&command_buffer);
    let after = device.work_counts();
    let reencoded = after.encoders_reencoded - before.encoders_reencoded;
    let left_out_ran = words_of(&outputs[1]).iter().any(|&word| word != 0);
    println!(
        "FOLD-BLIT-LEFT-OUT encoded again {reencoded}, the left out dispatch ran {left_out_ran}"
    );
    assert!(!left_out_ran, "the left out dispatch ran nothing");
    assert_eq!(
        read(&outputs[0]),
        expected(SMALL, |i| 2 * i),
        "the first encoder ran"
    );
    assert_eq!(
        read(&copied),
        expected(SMALL, |i| 2 * i),
        "the blit encoder ran"
    );
    assert_eq!(
        reencoded, 2,
        "the first two encoders were encoded again apart"
    );
}

/// A group of two serial compute encoders, the second writing a buffer over host pages, then a
/// concurrent compute encoder (which begins a group and a unit of command predication of its own)
/// of an If, so the command buffer used command predication: committed behind an event wait only
/// the host signals, the buffer rebound before the signal. The group's unit laid no step, so each
/// of its encoders encodes again alone (`Predication::alone`): the rebind encodes both again,
/// refusing none, and the command buffer writes the pages the buffer lies in.
#[test]
fn a_rebind_encodes_a_group_again_in_a_command_buffer_of_command_predication() {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("fold-rebind-predicated");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let words = (BYTES / 4) as u64;
    let input = floats(device, words, |i| i as f32);
    let middle = floats(device, words, |_| 0.0);
    let other = floats(device, words, |_| 0.0);
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let output = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("FOLD-REBIND-PREDICATED: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let word = device
        .new_buffer_with_bytes(&0u32.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    let serial = DispatchType::Serial;
    let warm = floats(device, words, |_| 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&middle, words));
    kernel.dispatch(&command_buffer, serial, (&middle, 0), (&warm, words));
    kernel.run(&command_buffer);

    let gate = device.new_shared_event().unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&middle, words));
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&middle), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(words, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer
        .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
        .unwrap();
    // Comparison 0: `word == reference`, which holds.
    encoder.encode_start_if(Some(&word), 0, 0, 0).unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&other), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(words, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    assert!(encoder.encode_end_if().unwrap(), "the If ends");
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(output.detach_backing(), Ok(true));
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    // Read before the signal: the rebind encodes again in its own call.
    let after = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    assert_eq!(after.encoders_folded - before.encoders_folded, 1);
    assert_eq!(
        (
            after.encoders_reencoded - before.encoders_reencoded,
            after.reencodes_refused - before.reencodes_refused
        ),
        (2, 0),
        "the rebind encoded both encoders of the group again"
    );
    assert!(
        !pages[0].written(),
        "the group wrote the pages the buffer left"
    );
    let written: Vec<f32> = pages[1]
        .words()
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    assert_eq!(written, expected(words, |i| 3 * i));
    assert_eq!(
        read(&other),
        expected(words, |i| 2 * i),
        "the If's dispatch ran"
    );
}
