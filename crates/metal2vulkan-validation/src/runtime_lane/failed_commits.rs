//! Device tests of command buffers committed as failed (`CommandBuffer::commit_failed`, a
//! runtime extension, docs/API.md) on the device the cfg selects, under the validation layer.
//!
//! The answers are those a consumer measured of Metal's failed command buffers on an M3 Max
//! (macOS 26.7): none of a failed encoder's commands run, a render pass's clear included; the
//! encoders before the failing one run; the event waits hold the command buffer; its signals
//! fire for a GPU hang and none fires for a protection fault; its scheduled handlers run; and it
//! completes in its queue's order with status `Error` and the chosen code.
//!
//! The probe shapes: Metal splits a command buffer into submissions (each render pass its own,
//! consecutive blit encoders one, consecutive compute encoders one, an event wait or signal
//! splitting a run), and a submission that uses a resource whose memory was taken away runs nothing
//! while every other runs. A consumer models that from `CommandBuffer::ended_encoders` and fails
//! the set (`FailedEncoders::Set`); the tests do the same for each shape the consumer probed and
//! check every encoder's effect, the signals, the handlers and the status.

use super::degrade_notices::on_device;
use super::degrade_notices::on_device_with;
use super::render_state::{texels, texture};
use super::{
    at_plan, bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    CaseResources, PublicCase, WAIT,
};
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::metal::{PixelFormat, RenderStages, ResourceUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    CommitFailure, ComputePipelineState, Device, EncoderKind, EndedEncoder, Error, EventRef,
    FailedEncoders, Invalid, LoadAction, Problem, RenderPassDescriptor, SharedEvent, StoreAction,
    Texture,
};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a command buffer held by a wait is watched not to complete.
pub(super) const HELD: Duration = Duration::from_millis(500);

/// A command buffer of `queue` with a blit encoder for each of `fills`, `(offset, byte)`, each
/// filling 4 bytes of `buffer`, then, where `target` is given, a render pass that clears it to
/// `clear` and draws nothing.
fn encoded(
    queue: &CommandQueue,
    buffer: &Buffer,
    fills: &[(u64, u8)],
    target: Option<(&Texture, f64)>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    for &(offset, byte) in fills {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, offset..offset + 4, byte).unwrap();
        blit.end_encoding().unwrap();
    }
    if let Some((target, clear)) = target {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = ClearColor::new(clear, clear, clear, clear);
        attachment.store_action = StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer
}

/// A 16-byte buffer of zeros.
fn zeros(device: &Device) -> Buffer {
    buffer_with(device, &[0; 16]).unwrap()
}

/// A 1x1 `RGBA8Unorm` target, cleared to all ones by a pass that completed.
fn cleared_target(device: &Device, queue: &CommandQueue, scratch: &Buffer) -> Texture {
    let target = texture(device, PixelFormat::RGBA8Unorm, 1);
    let clear = encoded(queue, scratch, &[], Some((&target, 1.0)));
    clear.commit().unwrap();
    clear.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(clear.status(), CommandBufferStatus::Completed);
    assert_eq!(texels(&target, 4), [255; 4]);
    target
}

/// Wait for `command_buffer` and require it failed with `code`.
fn failed_with(command_buffer: &CommandBuffer, code: u64) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(code))
    );
    assert_eq!(
        command_buffer.error().map(CommandBufferError::metal_code),
        Some(code)
    );
}

/// Nothing of a command buffer failed from its first encoder runs: neither its blits' fills nor
/// its render pass's clear land. It fails with the chosen code, which its completed handler
/// sees, after its scheduled handler ran. A set naming an encoder past the ended ones is
/// refused, changing nothing: the command buffer then commits and runs. (A code of 0 was the
/// refused call; it is no error now,
/// [`a_command_buffer_failed_with_no_error_runs_nothing_and_completes`].)
#[test]
fn a_command_buffer_failed_from_its_first_encoder_runs_nothing() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let words = zeros(device);
        let target = cleared_target(device, &queue, &words);
        let command_buffer = encoded(
            &queue,
            &words,
            &[(0, 0x11), (4, 0x22)],
            Some((&target, 0.0)),
        );
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduled = Arc::clone(&seen);
        command_buffer
            .add_scheduled_handler(Box::new(move |_| {
                scheduled.lock().unwrap().push("scheduled".to_owned());
            }))
            .unwrap();
        let completed = Arc::clone(&seen);
        command_buffer
            .add_completed_handler(Box::new(move |buffer| {
                completed.lock().unwrap().push(format!(
                    "completed {:?} {:?}",
                    buffer.status(),
                    buffer.error()
                ));
            }))
            .unwrap();
        assert_eq!(
            command_buffer.commit_failed(past_the_ended(3)).err(),
            Some(Error::Invalid(Invalid::new(
                "commit_failed",
                Problem::IndexOutOfRange { index: 3, limit: 3 },
            )))
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
        command_buffer.commit_failed(CommitFailure::new(8)).unwrap();
        failed_with(&command_buffer, 8);
        assert_eq!(read_buffer(&words, 0, 16).unwrap(), [0; 16]);
        assert_eq!(texels(&target, 4), [255; 4]);
        assert_eq!(
            *seen.lock().unwrap(),
            [
                "scheduled".to_owned(),
                format!(
                    "completed {:?} {:?}",
                    CommandBufferStatus::Error,
                    Some(CommandBufferError::Injected(8))
                ),
            ]
        );

        let again = encoded(&queue, &words, &[(0, 0x11)], None);
        again.commit_failed(past_the_ended(1)).unwrap_err();
        again.commit().unwrap();
        again.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(again.status(), CommandBufferStatus::Completed);
        assert_eq!(read_buffer(&words, 0, 4).unwrap(), [0x11; 4]);
    });
}

/// A failure of code 8 naming encoder `ended` alone, past a command buffer's `ended` encoders.
fn past_the_ended(ended: usize) -> CommitFailure {
    let mut failure = CommitFailure::new(8);
    failure.encoders = FailedEncoders::Set(BTreeSet::from([ended]));
    failure
}

/// A code of 0 is no error, Metal's answer for a command buffer of a queue whose GPU hang is in
/// flight (the consumer's measurement on an M3 Max, 11 hangs at a draw of primitive type 9): of
/// two fills and a clear, none lands, the event signal fires, the scheduled handler runs, and the
/// command buffer completes with status `Completed` and no error, which its completed handler
/// sees. A later command buffer of the queue runs.
#[test]
fn a_command_buffer_failed_with_no_error_runs_nothing_and_completes() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let words = zeros(device);
        let target = cleared_target(device, &queue, &words);
        let command_buffer = encoded(
            &queue,
            &words,
            &[(0, 0x11), (4, 0x22)],
            Some((&target, 0.0)),
        );
        let done = device.new_shared_event().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&done), 1)
            .unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduled = Arc::clone(&seen);
        command_buffer
            .add_scheduled_handler(Box::new(move |_| {
                scheduled.lock().unwrap().push("scheduled".to_owned());
            }))
            .unwrap();
        let completed = Arc::clone(&seen);
        command_buffer
            .add_completed_handler(Box::new(move |buffer| {
                completed.lock().unwrap().push(format!(
                    "completed {:?} {:?}",
                    buffer.status(),
                    buffer.error()
                ));
            }))
            .unwrap();
        command_buffer.commit_failed(CommitFailure::new(0)).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(read_buffer(&words, 0, 16).unwrap(), [0; 16], "a fill ran");
        assert_eq!(texels(&target, 4), [255; 4], "the clear ran");
        assert_eq!(done.signaled_value(), 1, "the signal did not fire");
        assert_eq!(
            *seen.lock().unwrap(),
            [
                "scheduled".to_owned(),
                format!(
                    "completed {:?} {:?}",
                    CommandBufferStatus::Completed,
                    None::<CommandBufferError>
                ),
            ]
        );
        let later = encoded(&queue, &words, &[(8, 0x33)], None);
        later.commit().unwrap();
        later.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(later.error(), None);
        let mut want = [0u8; 16];
        want[8..12].fill(0x33);
        assert_eq!(
            read_buffer(&words, 0, 16).unwrap(),
            want,
            "the later command buffer"
        );
        eprintln!(
            "CBF-MARK {}: a code-0 failure ran nothing and completed with no error",
            device.name()
        );
    });
}

/// The encoders before the failing one run, and it and those after it run nothing: of three
/// fills and a clear, failed from the second encoder, the first fill alone lands.
#[test]
fn the_encoders_before_the_failing_one_run() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let words = zeros(device);
        let target = cleared_target(device, &queue, &words);
        let command_buffer = encoded(
            &queue,
            &words,
            &[(0, 0x11), (4, 0x22), (8, 0x33)],
            Some((&target, 0.0)),
        );
        let mut failure = CommitFailure::new(2);
        failure.encoders = FailedEncoders::From(1);
        command_buffer.commit_failed(failure).unwrap();
        failed_with(&command_buffer, 2);
        let mut want = [0u8; 16];
        want[..4].fill(0x11);
        assert_eq!(read_buffer(&words, 0, 16).unwrap(), want);
        assert_eq!(texels(&target, 4), [255; 4]);
    });
}

/// A failed command buffer's event wait holds it until the host signals the event. Its signal
/// fires where the failure keeps signals (a GPU hang's), so a command buffer of another queue
/// waiting for it runs; where it does not (a protection fault's), none fires, and the waiter
/// runs only once the host signals.
#[test]
fn a_failed_command_buffer_keeps_its_waits_and_its_signals_as_asked() {
    on_device_with(at_plan, |device, _| {
        for signals in [true, false] {
            let gate = device.new_shared_event().unwrap();
            let done = device.new_shared_event().unwrap();
            let words = zeros(device);
            let failing = device.new_command_queue().unwrap();
            let waiting = device.new_command_queue().unwrap();
            let failed = failing.command_buffer().unwrap();
            failed
                .encode_wait_for_event(EventRef::from(&gate), 1)
                .unwrap();
            let mut blit = failed.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 0..4, 0x11).unwrap();
            blit.end_encoding().unwrap();
            failed
                .encode_signal_event(EventRef::from(&done), 1)
                .unwrap();
            let code = if signals { 2 } else { 16 };
            let mut failure = CommitFailure::new(code);
            failure.signals = signals;
            failed.commit_failed(failure).unwrap();
            let waiter = waiting.command_buffer().unwrap();
            waiter
                .encode_wait_for_event(EventRef::from(&done), 1)
                .unwrap();
            let mut blit = waiter.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 4..8, 0x22).unwrap();
            blit.end_encoding().unwrap();
            waiter.commit().unwrap();

            assert!(failed.wait_until_completed_for(HELD).is_err(), "{signals}");
            assert!(waiter.wait_until_completed_for(HELD).is_err(), "{signals}");
            gate.set_signaled_value(1);
            failed_with(&failed, code);
            if signals {
                waiter.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(done.signaled_value(), 1);
            } else {
                assert!(waiter.wait_until_completed_for(HELD).is_err());
                assert_eq!(done.signaled_value(), 0);
                done.set_signaled_value(1);
                waiter.wait_until_completed_for(WAIT).unwrap();
            }
            assert_eq!(waiter.status(), CommandBufferStatus::Completed);
            let mut want = [0u8; 16];
            want[4..8].fill(0x22);
            assert_eq!(read_buffer(&words, 0, 16).unwrap(), want, "{signals}");
        }
    });
}

/// A failed command buffer completes in its queue's order: between the command buffers
/// committed before and after it, each one's completed handler running after the one before.
#[test]
fn a_failed_command_buffer_completes_in_its_queue_s_order() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let words = zeros(device);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let buffers: Vec<CommandBuffer> = (0..3u8)
            .map(|index| {
                let command_buffer = encoded(
                    &queue,
                    &words,
                    &[(4 * u64::from(index), 0x10 + index)],
                    None,
                );
                let scheduled = Arc::clone(&seen);
                command_buffer
                    .add_scheduled_handler(Box::new(move |_| {
                        scheduled.lock().unwrap().push(format!("scheduled {index}"));
                    }))
                    .unwrap();
                let completed = Arc::clone(&seen);
                command_buffer
                    .add_completed_handler(Box::new(move |buffer| {
                        completed
                            .lock()
                            .unwrap()
                            .push(format!("completed {index} {:?}", buffer.error()));
                    }))
                    .unwrap();
                command_buffer
            })
            .collect();
        buffers[0].commit().unwrap();
        let mut failure = CommitFailure::new(1);
        failure.signals = false;
        buffers[1].commit_failed(failure).unwrap();
        buffers[2].commit().unwrap();
        buffers[2].wait_until_completed_for(WAIT).unwrap();
        for command_buffer in &buffers {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        let completions: Vec<String> = seen
            .lock()
            .unwrap()
            .iter()
            .filter(|line| line.starts_with("completed"))
            .cloned()
            .collect();
        assert_eq!(
            completions,
            [
                "completed 0 None".to_owned(),
                format!("completed 1 {:?}", Some(CommandBufferError::Injected(1))),
                "completed 2 None".to_owned(),
            ]
        );
        assert!(seen.lock().unwrap().contains(&"scheduled 1".to_owned()));
        let mut want = [0u8; 16];
        want[..4].fill(0x10);
        want[8..12].fill(0x12);
        assert_eq!(read_buffer(&words, 0, 16).unwrap(), want);
    });
}

/// The kernel case a compute encoder of the probe shapes dispatches: one invocation writes a word
/// over its poisoned output buffer.
pub(super) const KERNEL: &str = "threadgroup-word";

/// The consumer's probe shapes (M3 Max, macOS 26.7), each with the encoders Metal ran nothing of.
/// `o` is a blit, `c` a compute encoder, `r` a render pass; the encoder that uses the resource
/// whose memory was taken away is a capital, or `d` for a blit; `s` is an event signal and `w` an
/// event wait.
pub(super) const SHAPES: [(&str, &[usize]); 7] = [
    ("oCo", &[1]),
    ("cdc", &[1]),
    ("rRr", &[1]),
    ("ocdco", &[2]),
    ("odo", &[0, 1, 2]),
    ("odso", &[0, 1]),
    ("odwo", &[0, 1]),
];

/// What one encoder of a shape does, which lands only where it runs.
enum Effect {
    /// A blit's fill of 4 bytes of the shape's words at `offset` with `byte`.
    Fill { offset: u64, byte: u8 },
    /// A dispatch of [`KERNEL`]: the `length` bytes at `offset` of `output`, which held `initial`
    /// before; its resources are held until the command buffer completed.
    Dispatch {
        output: Buffer,
        offset: u64,
        length: u64,
        initial: Vec<u8>,
        _resources: Box<CaseResources>,
    },
    /// A render pass's clear to zero of `target`, cleared to all ones before.
    Clear(Texture),
}

/// Metal's submissions of a command buffer's `encoders`, as the consumer measured them: each
/// render pass alone; consecutive blit encoders share one, as do consecutive compute encoders;
/// an event wait or signal between two encoders splits them. Each encoder's submission, from 0.
fn metal_submissions(encoders: &[EndedEncoder]) -> Vec<usize> {
    let mut submission = 0;
    let mut before: Option<&EndedEncoder> = None;
    let mut submissions = Vec::with_capacity(encoders.len());
    for encoder in encoders {
        if let Some(before) = before {
            let joins = encoder.kind == before.kind
                && encoder.kind != EncoderKind::Render
                && encoder.event_waits_before == before.event_waits_before
                && encoder.event_signals_before == before.event_signals_before;
            if !joins {
                submission += 1;
            }
        }
        submissions.push(submission);
        before = Some(encoder);
    }
    submissions
}

/// The encoders a consumer fails where `detached` lost its memory: every encoder of each Metal
/// submission one of whose encoders uses it ([`metal_submissions`]).
fn failed_for(encoders: &[EndedEncoder], detached: &Buffer) -> BTreeSet<usize> {
    let submissions = metal_submissions(encoders);
    let using: BTreeSet<usize> = encoders
        .iter()
        .zip(&submissions)
        .filter(|(encoder, _)| encoder.used_resources.buffers.contains(detached))
        .map(|(_, &submission)| submission)
        .collect();
    (0..encoders.len())
        .filter(|&at| using.contains(&submissions[at]))
        .collect()
}

/// Each probe shape, its failed set derived from `ended_encoders` as the consumer derives it,
/// runs every encoder but those of the submission that uses the detached buffer: their effects
/// alone are missing. Its event wait holds it, its signal fires, its scheduled handler runs, and
/// it completes with status `Error` and code 8, which its completed handler sees.
#[test]
fn the_probe_shapes_run_every_submission_but_the_detached_one() {
    let kernel = public_case(KERNEL).unwrap();
    on_device_with(at_plan, |device, _| {
        let pipeline = pipeline_state(device, &kernel).unwrap();
        for (shape, failed) in SHAPES {
            run_shape(device, &kernel, &pipeline, shape, failed);
        }
    });
}

/// Encode `shape` on a queue of `device`, check its ended encoders and the set Metal's split
/// gives, commit it failed with that set, and check what ran.
fn run_shape(
    device: &Device,
    kernel: &PublicCase,
    pipeline: &ComputePipelineState,
    shape: &str,
    expected: &[usize],
) {
    let encoded = Shape::encode(device, kernel, pipeline, shape);
    let failed = encoded.failed_set(expected);
    let seen = encoded.watch_handlers();
    let mut failure = CommitFailure::new(8);
    failure.encoders = FailedEncoders::Set(failed.clone());
    encoded.command_buffer.commit_failed(failure).unwrap();
    encoded.release_waits();
    encoded.finished(&seen, Some(8), &failed, kernel);
}

/// A probe shape encoded on a command buffer of its own (not committed), with what each encoder
/// does and the report the runtime should give of it ([`Shape::encode`]).
pub(super) struct Shape {
    pub(super) shape: String,
    pub(super) command_buffer: CommandBuffer,
    /// The buffer the using encoders use, as the resource whose memory was taken away.
    pub(super) detached: Buffer,
    /// The event each `w` waits for (value 1), and the event each `s` signals (value 1).
    pub(super) gate: SharedEvent,
    pub(super) done: SharedEvent,
    words: Buffer,
    effects: Vec<Effect>,
    kinds: Vec<EncoderKind>,
    places: Vec<(usize, usize)>,
    waits: usize,
    signals: usize,
}

impl Shape {
    /// Encode `shape` (the [`SHAPES`] spelling) on a new queue of `device`, its compute
    /// encoders dispatching `kernel` with `pipeline`.
    pub(super) fn encode(
        device: &Device,
        kernel: &PublicCase,
        pipeline: &ComputePipelineState,
        shape: &str,
    ) -> Shape {
        let queue = device.new_command_queue().unwrap();
        let words = buffer_with(device, &[0; 32]).unwrap();
        let detached = buffer_with(device, &[0; 16]).unwrap();
        let gate = device.new_shared_event().unwrap();
        let done = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut effects = Vec::new();
        let mut kinds = Vec::new();
        let mut places = Vec::new();
        let (mut waits, mut signals) = (0, 0);
        for step in shape.chars() {
            match step {
                's' => {
                    command_buffer
                        .encode_signal_event(EventRef::from(&done), 1)
                        .unwrap();
                    signals += 1;
                    continue;
                }
                'w' => {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&gate), 1)
                        .unwrap();
                    waits += 1;
                    continue;
                }
                _ => {}
            }
            let uses = step.is_ascii_uppercase() || step == 'd';
            let at = effects.len() as u64;
            places.push((waits, signals));
            match step.to_ascii_lowercase() {
                'o' | 'd' => {
                    let byte = 0x10 + at as u8;
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&words, 4 * at..4 * at + 4, byte).unwrap();
                    if uses {
                        blit.fill_buffer(&detached, 0..4, 0x5a).unwrap();
                    }
                    blit.end_encoding().unwrap();
                    kinds.push(EncoderKind::Blit);
                    effects.push(Effect::Fill {
                        offset: 4 * at,
                        byte,
                    });
                }
                'c' => {
                    let resources = CaseResources::new(device, kernel).unwrap();
                    let bound = bound_buffers(device, &kernel.compute, &resources).unwrap();
                    let ComputeOutput::Buffer {
                        binding,
                        offset,
                        length,
                    } = kernel.compute.output
                    else {
                        panic!("{KERNEL}: the case's output is no buffer");
                    };
                    let output = bound[&binding].clone();
                    let initial = read_buffer(&output, offset, length).unwrap();
                    assert_ne!(initial, kernel.metal_output, "{KERNEL}: no output to see");
                    let mut compute = command_buffer.compute_command_encoder().unwrap();
                    encode_case(&mut compute, pipeline, &kernel.compute, &bound, &resources)
                        .unwrap();
                    if uses {
                        compute
                            .use_resource(detached.as_resource(), ResourceUsage::READ)
                            .unwrap();
                    }
                    compute.end_encoding().unwrap();
                    kinds.push(EncoderKind::Compute);
                    effects.push(Effect::Dispatch {
                        output,
                        offset,
                        length,
                        initial,
                        _resources: Box::new(resources),
                    });
                }
                'r' => {
                    let target = cleared_target(device, &queue, &words);
                    let mut pass = RenderPassDescriptor::new();
                    let attachment = &mut pass.color_attachments[0];
                    attachment.texture = Some(target.clone());
                    attachment.load_action = LoadAction::Clear;
                    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                    attachment.store_action = StoreAction::Store;
                    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    if uses {
                        encoder
                            .use_resource(
                                detached.as_resource(),
                                ResourceUsage::READ,
                                RenderStages::FRAGMENT,
                            )
                            .unwrap();
                    }
                    encoder.end_encoding().unwrap();
                    kinds.push(EncoderKind::Render);
                    effects.push(Effect::Clear(target));
                }
                other => panic!("{shape}: no step {other}"),
            }
        }
        Shape {
            shape: shape.to_owned(),
            command_buffer,
            detached,
            gate,
            done,
            words,
            effects,
            kinds,
            places,
            waits,
            signals,
        }
    }

    /// Check the ended encoders' report (each one's kind and place among the events), and the
    /// set Metal's split fails for the detached buffer, which must be `expected`: the set.
    pub(super) fn failed_set(&self, expected: &[usize]) -> BTreeSet<usize> {
        let shape = &self.shape;
        let ended = self.command_buffer.ended_encoders();
        let reported: Vec<_> = ended.iter().map(|encoder| encoder.kind).collect();
        assert_eq!(reported, self.kinds, "{shape}");
        let reported: Vec<_> = ended
            .iter()
            .map(|encoder| (encoder.event_waits_before, encoder.event_signals_before))
            .collect();
        assert_eq!(reported, self.places, "{shape}");
        let failed = failed_for(&ended, &self.detached);
        assert_eq!(
            failed,
            expected.iter().copied().collect::<BTreeSet<_>>(),
            "{shape}"
        );
        failed
    }

    /// Each kind and range of Metal's submissions of the shape, by [`metal_submissions`].
    pub(super) fn submissions(&self) -> Vec<(EncoderKind, std::ops::Range<usize>)> {
        let ended = self.command_buffer.ended_encoders();
        let numbers = metal_submissions(&ended);
        let mut submissions: Vec<(EncoderKind, std::ops::Range<usize>)> = Vec::new();
        for (at, number) in numbers.into_iter().enumerate() {
            match submissions.get_mut(number) {
                Some((_, range)) => range.end = at + 1,
                None => submissions.push((ended[at].kind, at..at + 1)),
            }
        }
        submissions
    }

    /// A scheduled and a completed handler, which note what they see.
    pub(super) fn watch_handlers(&self) -> Arc<Mutex<Vec<String>>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let scheduled = Arc::clone(&seen);
        self.command_buffer
            .add_scheduled_handler(Box::new(move |_| {
                scheduled.lock().unwrap().push("scheduled".to_owned());
            }))
            .unwrap();
        let completed = Arc::clone(&seen);
        self.command_buffer
            .add_completed_handler(Box::new(move |buffer| {
                completed.lock().unwrap().push(format!(
                    "completed {:?} {:?}",
                    buffer.status(),
                    buffer.error()
                ));
            }))
            .unwrap();
        seen
    }

    /// Where the shape waits, check that its waits hold it, committed, then signal them.
    pub(super) fn release_waits(&self) {
        if self.waits > 0 {
            assert!(
                self.command_buffer.wait_until_completed_for(HELD).is_err(),
                "{}: its wait did not hold it",
                self.shape
            );
            self.gate.set_signaled_value(1);
        }
    }

    /// Wait for the command buffer, and check it finished failed with `code` (completed where
    /// `None`), its signal fired, its handlers ran, every encoder outside `failed` and none in
    /// it ran, and the detached buffer was never written; then that, completed, each encoder
    /// keeps its kind and lists nothing.
    pub(super) fn finished(
        &self,
        seen: &Arc<Mutex<Vec<String>>>,
        code: Option<u64>,
        failed: &BTreeSet<usize>,
        kernel: &PublicCase,
    ) {
        let shape = &self.shape;
        let (status, error) = match code {
            Some(code) => {
                failed_with(&self.command_buffer, code);
                (
                    CommandBufferStatus::Error,
                    Some(CommandBufferError::Injected(code)),
                )
            }
            None => {
                self.command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(self.command_buffer.error(), None, "{shape}");
                (CommandBufferStatus::Completed, None)
            }
        };
        assert_eq!(self.command_buffer.status(), status, "{shape}");
        assert_eq!(
            self.done.signaled_value(),
            u64::from(self.signals > 0),
            "{shape}"
        );
        assert_eq!(
            *seen.lock().unwrap(),
            [
                "scheduled".to_owned(),
                format!("completed {status:?} {error:?}"),
            ],
            "{shape}"
        );
        for (at, effect) in self.effects.iter().enumerate() {
            let ran = !failed.contains(&at);
            match effect {
                Effect::Fill { offset, byte } => {
                    let want = if ran { [*byte; 4] } else { [0; 4] };
                    assert_eq!(
                        read_buffer(&self.words, *offset, 4).unwrap(),
                        want,
                        "{shape}: encoder {at}"
                    );
                }
                Effect::Dispatch {
                    output,
                    offset,
                    length,
                    initial,
                    ..
                } => {
                    let want = if ran { &kernel.metal_output } else { initial };
                    assert_eq!(
                        &read_buffer(output, *offset, *length).unwrap(),
                        want,
                        "{shape}: encoder {at}"
                    );
                }
                Effect::Clear(target) => {
                    let want = if ran { [0; 4] } else { [255; 4] };
                    assert_eq!(texels(target, 4), want, "{shape}: encoder {at}");
                }
            }
        }
        // A using blit that ran wrote the detached buffer; none of the failed ones did.
        let using_blit_ran = self
            .shape
            .chars()
            .filter(|&step| !matches!(step, 's' | 'w'))
            .enumerate()
            .any(|(at, step)| step == 'd' && !failed.contains(&at));
        let want = if using_blit_ran { [0x5a; 4] } else { [0; 4] };
        assert_eq!(read_buffer(&self.detached, 0, 4).unwrap(), want, "{shape}");

        let ended = self.command_buffer.ended_encoders();
        assert_eq!(ended.len(), self.kinds.len(), "{shape}");
        assert!(ended.iter().all(|encoder| {
            let used = &encoder.used_resources;
            used.buffers.is_empty() && used.textures.is_empty() && used.heaps.is_empty()
        }));
    }
}

/// A set that names an encoder past the ended ones is refused, changing nothing: the command
/// buffer then commits failed with its own set, and the encoder outside it runs.
#[test]
fn a_set_past_the_ended_encoders_is_refused() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let words = zeros(device);
        let command_buffer = encoded(&queue, &words, &[(0, 0x11), (4, 0x22)], None);
        let mut failure = CommitFailure::new(8);
        failure.encoders = FailedEncoders::Set(BTreeSet::from([0, 2]));
        assert_eq!(
            command_buffer.commit_failed(failure).err(),
            Some(Error::Invalid(Invalid::new(
                "commit_failed",
                Problem::IndexOutOfRange { index: 2, limit: 2 },
            )))
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
        let mut failure = CommitFailure::new(8);
        failure.encoders = FailedEncoders::Set(BTreeSet::from([0]));
        command_buffer.commit_failed(failure).unwrap();
        failed_with(&command_buffer, 8);
        let mut want = [0u8; 16];
        want[4..8].fill(0x22);
        assert_eq!(read_buffer(&words, 0, 16).unwrap(), want);
    });
}
