//! Consecutive compute encoders of one command buffer recorded into one `VkCommandBuffer`
//! (`encoder_fold.rs`): which encoders join a group and what ends one, the barrier the tracker's
//! rule records between two encoders of a group, a failure that splits a group at the commit, and
//! a rebind that encodes a group again whole.
//!
//! [`dependent_encoders_of_a_group_read_what_the_encoders_before_them_wrote`] is the barrier's
//! check: each small encoder reads the tail of the large write the encoder before it made, which
//! a GPU that overlaps two dispatches with no barrier between them may read before it is written.
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
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommitFailure,
    ComputePipelineState, Device, Error, EventRef, FailedEncoders, ResourceOptions, Size,
    WorkCounts,
};
use metal2vulkan::metal::DispatchType;
use std::collections::BTreeSet;

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
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        Moved {
            folded: after.encoders_folded - before.encoders_folded,
            begun: after.command_buffers_begun_encoders - before.command_buffers_begun_encoders,
            barriers: after.command_barriers - before.command_barriers,
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
            barriers: 0
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
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (2, 3), "{moved:?}");
    for output in &outputs {
        assert_eq!(read(output), expected(SMALL, |i| 2 * i));
    }
}

/// An encoder that updates a fence ends its group, so a fence orders nothing inside one: the
/// encoder that waits for it begins a group of its own, which the encoder after it joins.
#[test]
fn a_fence_update_ends_a_group() {
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
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&buffers[0]), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(SMALL, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.update_fence(&fence).unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.wait_for_fence(&fence).unwrap();
    encoder.set_compute_pipeline_state(&kernel.state).unwrap();
    encoder.set_buffer(Some(&buffers[0]), 0, 0).unwrap();
    encoder.set_buffer(Some(&buffers[1]), 0, 1).unwrap();
    encoder.set_buffer(Some(&kernel.scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(SMALL, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    kernel.dispatch(
        &command_buffer,
        serial,
        (&buffers[1], 0),
        (&buffers[2], SMALL),
    );
    kernel.run(&command_buffer);
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (1, 2), "{moved:?}");
    for (at, buffer) in buffers.iter().enumerate() {
        let times = at as u64 + 2;
        assert_eq!(read(buffer), expected(SMALL, |i| times * i), "buffer {at}");
    }
}

/// Three encoders of one group, a blit encoder's fill, and a compute encoder after it, committed
/// as failed with `encoders`: which of the five ran, read from what each wrote, and how many
/// encoders the commit encoded again (`WorkCounts::encoders_reencoded`).
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
    kernel.dispatch(&command_buffer, serial, (&input, 0), (&outputs[4], SMALL));
    let moved = Moved::of(&before, &device.work_counts());
    assert_eq!((moved.folded, moved.begun), (2, 3), "{moved:?}");
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

/// A failed set that names one encoder of a group fails that encoder alone, as on Metal: the
/// group's other two are encoded again apart at the commit, and the encoders after it run.
#[test]
fn a_failed_encoder_inside_a_group_fails_alone() {
    let set = FailedEncoders::Set(BTreeSet::from([1]));
    assert_eq!(failed_with(set), ([true, false, true, true, true], 2));
}

/// A first failed encoder inside a group fails from there on, as on Metal: the group's first
/// encoder is encoded again apart at the commit, and runs.
#[test]
fn a_first_failed_encoder_inside_a_group_runs_the_group_s_encoders_before_it() {
    let from = FailedEncoders::From(1);
    assert_eq!(failed_with(from), ([true, false, false, false, false], 1));
}

/// A first failed encoder at a group's boundary (an encoder that is not a compute encoder) splits
/// no group: the group before it runs as recorded.
#[test]
fn a_first_failed_encoder_at_a_group_s_boundary_fails_from_there() {
    let from = FailedEncoders::From(3);
    assert_eq!(failed_with(from), ([true, true, true, false, false], 0));
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
