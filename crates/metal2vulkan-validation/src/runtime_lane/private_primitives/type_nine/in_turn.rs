//! Device tests of a command buffer that draws primitive type 9 in more than one pass, on the
//! device the cfg selects, under the validation layer: it fails as Metal's GPU fails it, its
//! error and its timing, and its queue's command buffer committed right after answers as Metal's
//! did.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, 118 command buffers of a blit, then
//! per draw its pass, a blit and an event signal, then a blit and a signal; the numbers are in
//! `command_buffer/hang.rs` of the runtime):
//!
//! - **A fault, then later failures** (an indexed draw, then another or a direct one of 3
//!   vertices): the command buffer fails with the fault's error, `PageFault`, every event signal
//!   firing. It reads `Scheduled` once the GPU recovered from the fault, and completes once the
//!   GPU recovered from each later fault, then from a hang (in a hang's time), past which it
//!   reaches nothing. The queue's command buffer committed right after runs nothing where a hang
//!   follows the fault, as during a hang, and runs where faults alone do.
//! - **A hang, then a fault**: as the hang alone, `GpuHang`; the GPU never reaches the fault.
//!
//! The runtime runs nothing from the first failing pass on, as for one failure.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache,
};
use super::{targets, Answer, Lane, Tracked, FAULT, HANG, SCHEDULED};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::IndexType;
use metal2vulkan_runtime::{
    CommandBufferError, CommandBufferStatus, CommandQueue, DrawPrimitiveType, EventRef, Region,
    RenderPipelineState,
};
use std::time::Duration;

/// A pass's draw of type 9.
#[derive(Clone, Copy, Debug)]
enum Nine {
    /// An indexed draw of 3 16-bit indices, at which Metal's GPU faults.
    Faults,
    /// A direct draw of 3 vertices, at which Metal's GPU hangs.
    Hangs,
}

/// How a command buffer of [`Nine`]s ends, as Metal's does.
struct Expected {
    error: CommandBufferError,
    /// The earliest after its commit it reads `Scheduled`.
    scheduled: Duration,
    /// The earliest after it was scheduled (its `kernelEndTime`) it fails.
    fails: Duration,
    /// Whether the queue's command buffer committed right after it runs.
    next_runs: bool,
}

/// One command buffer's run ([`in_turn`]).
struct InTurn {
    drawn: Answer,
    /// Which of its marks ran: before the first pass, after each pass, at its end.
    marks: Vec<bool>,
    /// Its event signals' values: after each pass, at its end.
    signals: Vec<u64>,
    /// Whether a pass cleared its target.
    cleared: bool,
    /// The queue's command buffer committed right after it, and whether its mark ran.
    next: (Answer, bool),
}

/// A mark's byte, by slot.
fn mark_of(slot: usize) -> u8 {
    0x60 + slot as u8
}

/// On `queue`, one command buffer of a mark, then for each of `draws` its pass (over targets of
/// its own, under `pipeline`), a mark and a signal of an event of its own, then a mark and a
/// signal; and right after it one of a mark: both committed, waited for.
fn in_turn(
    lane: &Lane,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    draws: &[Nine],
) -> InTurn {
    let device = &lane.device;
    let slots = draws.len() + 3;
    let marks = buffer_with(device, &vec![0; slots * 4]).unwrap();
    let mark = |tracked: &Tracked, slot: usize| {
        let mut blit = tracked.command_buffer.blit_command_encoder().unwrap();
        let at = slot as u64 * 4;
        blit.fill_buffer(&marks, at..at + 4, mark_of(slot)).unwrap();
        blit.end_encoding().unwrap();
    };
    let indices = buffer_with(device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let events: Vec<_> = (0..=draws.len())
        .map(|_| device.new_shared_event().unwrap())
        .collect();
    let mut drawn = Tracked::new(queue);
    mark(&drawn, 0);
    let mut drawn_into = Vec::new();
    for (at, draw) in draws.iter().enumerate() {
        let (targets, pass) = targets(lane);
        let mut encoder = drawn.command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder
            .set_vertex_buffer(Some(&lane.positions), 0, 0)
            .unwrap();
        encoder
            .set_vertex_buffer(Some(&lane.offsets), 0, 1)
            .unwrap();
        let nine = DrawPrimitiveType::Nine;
        match draw {
            Nine::Faults => {
                encoder.draw_indexed_primitives(nine, 3, IndexType::UInt16, &indices, 0, 1, 0, 0)
            }
            Nine::Hangs => encoder.draw_primitives(nine, 0, 3, 1, 0),
        }
        .unwrap();
        encoder.end_encoding().unwrap();
        mark(&drawn, at + 1);
        drawn
            .command_buffer
            .encode_signal_event(EventRef::from(&events[at]), 1)
            .unwrap();
        drawn_into.push(targets);
    }
    mark(&drawn, draws.len() + 1);
    drawn
        .command_buffer
        .encode_signal_event(EventRef::from(&events[draws.len()]), 1)
        .unwrap();
    let mut next = Tracked::new(queue);
    mark(&next, draws.len() + 2);
    drawn.commit();
    next.commit();
    drawn.wait();
    next.wait();
    let read = read_buffer(&marks, 0, slots as u64 * 4).unwrap();
    let ran = |slot: usize| read[slot * 4] == mark_of(slot);
    let cleared = drawn_into.iter().any(|targets| {
        let mut texel = [0u8; 16];
        targets[0]
            .get_bytes(&mut texel, 16, 16, Region::new_2d(0, 0, 1, 1), 0, 0)
            .unwrap();
        texel == [0; 16]
    });
    InTurn {
        drawn: drawn.answer(),
        marks: (0..=draws.len() + 1).map(ran).collect(),
        signals: events.iter().map(|event| event.signaled_value()).collect(),
        cleared,
        next: (next.answer(), ran(draws.len() + 2)),
    }
}

/// Run each of `cases` on a queue of its own (a queue's second fault or hang revokes it) and
/// check it against Metal's answer: the command buffer fails with the error, reads `Scheduled`
/// no sooner than Metal's did after its commit and fails no sooner than Metal's did after it was
/// scheduled ([`Answer::since_kernel_end`]), runs only its mark before the first pass, every event
/// signal firing; the queue's command buffer committed right after completes cleanly after it,
/// running where Metal's did.
fn fails_in_turn(cases: &[(&[Nine], Expected)]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipeline = lane.pipeline("pt_plain");
        for (draws, expected) in cases {
            let label = format!("{draws:?} on {name}");
            let queue = lane.device.new_command_queue().unwrap();
            let run = in_turn(&lane, &queue, &pipeline, draws);
            let drawn = &run.drawn;
            assert_eq!(drawn.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(drawn.error, Some(expected.error), "{label}");
            let scheduled = drawn.to_scheduled();
            assert!(
                scheduled >= expected.scheduled,
                "{label}: read Scheduled {scheduled:?} after its commit"
            );
            let fails = drawn.since_kernel_end();
            assert!(
                fails >= expected.fails,
                "{label}: failed {fails:?} after it was scheduled (its scheduled handler {:?} \
                 after)",
                drawn.scheduled_handler_lag()
            );
            let mut only_before = vec![false; draws.len() + 2];
            only_before[0] = true;
            assert_eq!(run.marks, only_before, "{label}: the marks that ran");
            assert!(!run.cleared, "{label}: a pass cleared its target");
            assert!(
                run.signals.iter().all(|&value| value == 1),
                "{label}: its event signals: {:?}",
                run.signals
            );
            let (next, next_ran) = &run.next;
            assert!(
                next.completed_cleanly(),
                "{label}: the one right after: {next:?}"
            );
            assert_eq!(
                *next_ran, expected.next_runs,
                "{label}: whether the one right after ran"
            );
            assert!(
                next.completed >= drawn.completed,
                "{label}: the one right after completed first"
            );
            let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
            println!(
                "{label}: {:?} Scheduled at {:.1} ms, failed {:.1} ms after (its scheduled \
                 handler {:.2} ms after it was scheduled); the one right after ran: {next_ran}",
                drawn.error,
                ms(scheduled),
                ms(fails),
                ms(drawn.scheduled_handler_lag()),
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A fault, then later faults or a hang, in one command buffer: it fails with `PageFault`, reads
/// `Scheduled` once the GPU recovered from the fault and fails once it recovered from each later
/// fault and a hang, none past a hang; the queue's command buffer committed right after runs
/// nothing where a hang follows the fault, and runs where faults alone do.
#[test]
fn a_fault_then_later_failures_fail_as_metal_s_gpu_does() {
    let fault = |fails, next_runs| Expected {
        error: CommandBufferError::PageFault,
        scheduled: FAULT,
        fails,
        next_runs,
    };
    use Nine::{Faults, Hangs};
    fails_in_turn(&[
        (&[Faults, Hangs], fault(HANG, false)),
        (&[Faults, Faults], fault(FAULT, true)),
        (&[Faults, Faults, Hangs], fault(FAULT + HANG, false)),
        (&[Faults, Hangs, Faults], fault(HANG, false)),
    ]);
}

/// A hang, then a fault, in one command buffer fails as the hang alone does: `GpuHang`, and the
/// queue's command buffer committed right after runs nothing.
#[test]
fn a_hang_then_a_fault_fails_as_the_hang_does() {
    fails_in_turn(&[(
        &[Nine::Hangs, Nine::Faults],
        Expected {
            error: CommandBufferError::GpuHang,
            scheduled: SCHEDULED,
            fails: HANG,
            next_runs: false,
        },
    )]);
}
