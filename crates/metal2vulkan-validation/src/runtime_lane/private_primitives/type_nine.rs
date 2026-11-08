//! Device tests of a draw of Metal's private primitive type 9 ([`DrawPrimitiveType::Nine`]) on
//! the device the cfg selects, under the validation layer: its command buffer fails as Metal's
//! GPU fails it, and the queue's other command buffers answer as Metal's do.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, 47 runs, 33 of them hangs: the
//! scenario below, one that also commits after the hang read `Scheduled` and 20 ms on, and the
//! consumer's, which commits after 1 to 8 ms):
//!
//! - **A direct draw of 3 vertices or more** (1 or 2 instances) hangs Metal's GPU: the command
//!   buffer reads `Scheduled` 0.68 to 13.2 ms after its commit and fails with
//!   `MTLCommandBufferErrorTimeout` ([`CommandBufferError::GpuHang`]) 28.3 to 46.2 ms after it,
//!   mostly 27.5 to 28.9 ms after it read `Scheduled`.
//!   Its encoders before the pass run; the pass (its clear too) and every encoder after it run
//!   nothing; its event signal fires. Every command buffer of its queue scheduled meanwhile reads
//!   `Scheduled` from the hang's on, as committed or as its wait is met, and completes with no
//!   error together with it, its event signals firing: committed at once or 10 ms on, or with an
//!   event wait met during the hang by the host or by another queue's command buffer. Whether
//!   their work runs is a race, which the runtime answers as Metal mostly did where its hang read
//!   `Scheduled` 4.6 ms or more in: they run nothing (`command_buffer/hang.rs`). One whose event
//!   wait the host met after the hang completed ran, as did one of another queue (in 4 ms, during
//!   the hang) and one committed after the hang.
//! - **A direct draw of 1 or 2 vertices, or of no vertex or instance** draws nothing: the pass
//!   clears, and every command buffer runs.
//! - **An indexed draw** (any index count, none or of no instance too) faults Metal's GPU: the
//!   command buffer fails with `MTLCommandBufferErrorPageFault` ([`CommandBufferError::PageFault`])
//!   6.86 to 10.3 ms after it read `Scheduled`; the pass runs nothing (no clear), the encoder
//!   before it runs and its signal fires, and the queue's others run, the one committed at once
//!   reading `Scheduled` before it completed and completing after it. The encoders after the pass
//!   were lost to the GPU's recovery where the GPU ran just before, the runtime's answer (they ran
//!   where it was idle before, as after 3 s idle: `CommandBufferError::PageFault`).
//! - **An indirect draw** (of 3 vertices or none) **or an indexed indirect one** fails the command
//!   buffer as a victim of the GPU's recovery, `MTLCommandBufferErrorInternal`
//!   ([`CommandBufferError::GpuRecoveryVictim`]), with the indexed draw's other answers.
//! - **A queue's second fault or hang revokes it**; victims do not count. Once the queue's second
//!   command buffer that faulted or hung completed, every one committed on it fails at once with
//!   `MTLCommandBufferErrorAccessRevoked` ([`CommandBufferError::AccessRevoked`]): nothing of it
//!   runs and its event signals do not fire; a new queue runs on.
//!
//! Each scenario commits on a queue of its own the command buffer of the draw (a blit's mark, the
//! pass of the draw over a target the host filled, a mark, an event signal of 1, a mark), at once
//! one that marks and signals 2, and on another queue one that marks. 10 ms on, on the lane's
//! queue, it commits one that marks and three that wait for an event first, then mark: the host
//! meets the first's 15 ms after the draw's commit, another queue's command buffer committed then
//! the second's, and the host the third's only once the others completed. Last, it commits one
//! more that marks. Where that other queue's command buffer is slow to complete (beside other GPU
//! work), its signal can land only after the hang ended, and the second wait is then met after
//! the hang, as the third's is: its command buffer runs.

use super::super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use super::{Lane, FORMATS, SIDE};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{IndexType, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    DrawPrimitiveType, EventRef, LoadAction, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

mod in_turn;
mod loops;

/// The earliest Metal's command buffer its GPU hung at read `Scheduled` after its commit:
/// 5.06 ms.
const SCHEDULED: Duration = Duration::from_micros(5_060);

/// The earliest Metal failed a command buffer its GPU hung at after it read `Scheduled`:
/// 27.5 ms.
const HANG: Duration = Duration::from_micros(27_500);

/// The earliest Metal completed a command buffer its GPU faulted at, or failed as a victim of its
/// recovery, after it read `Scheduled`: 6.86 ms.
const FAULT: Duration = Duration::from_micros(6_860);

/// When the host meets the event wait of the queue's command buffer it meets during a hang,
/// after the draw's commit.
const DURING: Duration = Duration::from_millis(15);

/// The byte the host fills the pass's first target with, which its clear overwrites.
const FILLED: u8 = 0x5a;

/// The marks, by slot: each command buffer's, and the hanging one's three.
const MARKS: [(&str, u8); 10] = [
    ("before the pass", 0xa1),
    ("after the pass", 0xb1),
    ("after the signal", 0xb2),
    ("the queue's, at once", 0xc1),
    ("the queue's, 10 ms on", 0xc2),
    (
        "the queue's, its wait met by the host during the hang",
        0xc4,
    ),
    ("the queue's, its wait met by another queue's", 0xc5),
    ("the queue's, its wait met after the hang", 0xc3),
    ("another queue's", 0xd1),
    ("the one after", 0xe1),
];

/// The slots of [`MARKS`] that a scenario's command buffers other than the draw's mark, in the
/// order of [`Outcome::answers`] after the draw's.
const OTHERS: [usize; 7] = [3, 4, 5, 6, 7, 8, 9];

/// The draw of type 9 a scenario's pass makes.
#[derive(Clone, Copy, Debug)]
enum Draw {
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
    Direct { vertices: u32, instances: u32 },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`
    /// of 16-bit indices 0, 1, 2.
    Indexed { indices: u32, instances: u32 },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:` of this vertex count, one instance.
    Indirect(u32),
    /// `drawIndexedPrimitives:indexType:...:indirectBuffer:` of 3 16-bit indices.
    IndexedIndirect,
}

/// How a scenario's command buffer of the draw ends, as Metal's does.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expected {
    /// It draws nothing, and every command buffer runs.
    Nothing,
    /// Metal's GPU hangs at it.
    Hang,
    /// The pass runs nothing and the command buffer fails with the error.
    Fails(CommandBufferError),
}

/// One command buffer's answer.
#[derive(Debug)]
struct Answer {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// Its commit's instant.
    committed: Instant,
    /// When its scheduled handler ran.
    scheduled: Instant,
    /// When its completed handler ran.
    completed: Instant,
}

impl Answer {
    /// From its commit to its completed handler.
    fn took(&self) -> Duration {
        self.completed - self.committed
    }

    /// From its commit to its scheduled handler.
    fn to_scheduled(&self) -> Duration {
        self.scheduled - self.committed
    }

    /// From its scheduled handler to its completed handler.
    fn since_scheduled(&self) -> Duration {
        self.completed - self.scheduled
    }

    /// It completed with no error.
    fn completed_cleanly(&self) -> bool {
        self.status == CommandBufferStatus::Completed && self.error.is_none()
    }
}

/// A committed command buffer and the instants its scheduled and completed handlers ran.
struct Tracked {
    command_buffer: CommandBuffer,
    committed: Option<Instant>,
    scheduled: Arc<OnceLock<Instant>>,
    completed: Arc<OnceLock<Instant>>,
}

impl Tracked {
    fn new(queue: &CommandQueue) -> Tracked {
        let command_buffer = queue.command_buffer().unwrap();
        let (scheduled, completed) = (Arc::new(OnceLock::new()), Arc::new(OnceLock::new()));
        let set = Arc::clone(&scheduled);
        command_buffer
            .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                let _ = set.set(Instant::now());
            }))
            .unwrap();
        let set = Arc::clone(&completed);
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let _ = set.set(Instant::now());
            }))
            .unwrap();
        Tracked {
            command_buffer,
            committed: None,
            scheduled,
            completed,
        }
    }

    /// A blit that fills `marks`' slot `slot` with its mark.
    fn mark(&self, marks: &Buffer, slot: usize) {
        let mut blit = self.command_buffer.blit_command_encoder().unwrap();
        let at = slot as u64 * 4;
        blit.fill_buffer(marks, at..at + 4, MARKS[slot].1).unwrap();
        blit.end_encoding().unwrap();
    }

    fn commit(&mut self) {
        self.committed = Some(Instant::now());
        self.command_buffer.commit().unwrap();
    }

    fn wait(&self) {
        self.command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    fn answer(&self) -> Answer {
        Answer {
            status: self.command_buffer.status(),
            error: self.command_buffer.error(),
            committed: self.committed.unwrap(),
            scheduled: *self.scheduled.get().expect("its scheduled handler ran"),
            completed: *self.completed.get().expect("its completed handler ran"),
        }
    }
}

/// A scenario's answers: the command buffer of the draw's, then the others' in [`OTHERS`]'s order
/// (the queue's at once, 10 ms on, its wait met during the hang by the host and by another
/// queue's, its wait met after the hang; another queue's, the one after), the other queue's
/// command buffer that meets the second wait, the marks, the event's value and whether the pass
/// cleared its target.
struct Outcome {
    answers: [Answer; 8],
    signal: Answer,
    marks: [u8; 10],
    event: u64,
    cleared: bool,
}

/// The pass's targets, the first filled with [`FILLED`], and the pass that clears them.
fn targets(lane: &Lane) -> (Vec<Texture>, RenderPassDescriptor) {
    let targets: Vec<_> = FORMATS
        .iter()
        .map(|format| {
            let mut descriptor = TextureDescriptor::texture_2d(*format, SIDE, SIDE, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::shared();
            lane.device.new_texture(&descriptor).unwrap()
        })
        .collect();
    let filled = vec![FILLED; (SIDE * SIDE * 16) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    targets[0]
        .replace_region(region, 0, 0, &filled, SIDE * 16, SIDE * SIDE * 16)
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    for (i, target) in targets.iter().enumerate() {
        let color = &mut pass.color_attachments[i];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
    }
    (targets, pass)
}

/// Run the scenario (the module docs) of `draw` under `pipeline`.
fn scenario(
    lane: &Lane,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    draw: Draw,
) -> Outcome {
    let device = &lane.device;
    let marks = buffer_with(device, &[0; 40]).unwrap();
    let event = device.new_shared_event().unwrap();
    let during_gate = device.new_shared_event().unwrap();
    let gpu_gate = device.new_shared_event().unwrap();
    let gate = device.new_shared_event().unwrap();
    let other_queue = device.new_command_queue().unwrap();
    let indices = buffer_with(device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let words: &[u32] = match draw {
        Draw::Indirect(vertices) => &[vertices, 1, 0, 0],
        _ => &[3, 1, 0, 0, 0],
    };
    let arguments: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let arguments = buffer_with(device, &arguments).unwrap();
    let (targets, pass) = targets(lane);

    let mut drawn = Tracked::new(queue);
    drawn.mark(&marks, 0);
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
        Draw::Direct {
            vertices,
            instances,
        } => encoder.draw_primitives(nine, 0, vertices, instances, 0),
        Draw::Indexed {
            indices: count,
            instances,
        } => encoder.draw_indexed_primitives(
            nine,
            count,
            IndexType::UInt16,
            &indices,
            0,
            instances,
            0,
            0,
        ),
        Draw::Indirect(_) => encoder.draw_primitives_indirect(nine, &arguments, 0),
        Draw::IndexedIndirect => encoder.draw_indexed_primitives_indirect(
            nine,
            IndexType::UInt16,
            &indices,
            0,
            &arguments,
            0,
        ),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    drawn.mark(&marks, 1);
    drawn
        .command_buffer
        .encode_signal_event(EventRef::from(&event), 1)
        .unwrap();
    drawn.mark(&marks, 2);

    let mut at_once = Tracked::new(queue);
    at_once.mark(&marks, 3);
    at_once
        .command_buffer
        .encode_signal_event(EventRef::from(&event), 2)
        .unwrap();
    let mut later = Tracked::new(queue);
    later.mark(&marks, 4);
    let waiting = |slot: usize, gate| {
        let tracked = Tracked::new(queue);
        tracked
            .command_buffer
            .encode_wait_for_event(EventRef::from(gate), 1)
            .unwrap();
        tracked.mark(&marks, slot);
        tracked
    };
    let mut during = waiting(5, &during_gate);
    let mut on_gpu = waiting(6, &gpu_gate);
    let mut held = waiting(7, &gate);
    let mut other = Tracked::new(&other_queue);
    other.mark(&marks, 8);
    let mut signaller = Tracked::new(&other_queue);
    signaller
        .command_buffer
        .encode_signal_event(EventRef::from(&gpu_gate), 1)
        .unwrap();

    drawn.commit();
    at_once.commit();
    other.commit();
    std::thread::sleep(Duration::from_millis(10));
    later.commit();
    // The event waits last in the queue's order, so they hold back none of the others.
    during.commit();
    on_gpu.commit();
    held.commit();
    signaller.commit();
    let due = drawn.committed.unwrap() + DURING;
    if let Some(left) = due.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
    during_gate.set_signaled_value(1);
    for tracked in [&drawn, &at_once, &later, &during, &on_gpu] {
        tracked.wait();
    }
    gate.set_signaled_value(1);
    held.wait();
    other.wait();
    signaller.wait();
    let mut after = Tracked::new(queue);
    after.mark(&marks, 9);
    after.commit();
    after.wait();

    let read = read_buffer(&marks, 0, 40).unwrap();
    let mut texel = [0u8; 16];
    targets[0]
        .get_bytes(&mut texel, 16, 16, Region::new_2d(0, 0, 1, 1), 0, 0)
        .unwrap();
    Outcome {
        answers: [
            &drawn, &at_once, &later, &during, &on_gpu, &held, &other, &after,
        ]
        .map(Tracked::answer),
        signal: signaller.answer(),
        marks: std::array::from_fn(|slot| read[slot * 4]),
        event: event.signaled_value(),
        cleared: texel == [0; 16],
    }
}

/// Run each of `draws`'s scenarios on the test device and check it against `expected`.
fn answers_as_metal_does(draws: &[Draw], expected: Expected) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipeline = lane.pipeline("pt_plain");
        for &draw in draws {
            // A queue of its own: a queue's second fault or hang revokes it (`AccessRevoked`).
            let queue = lane.device.new_command_queue().unwrap();
            let before = lane.device.work_counts();
            let outcome = scenario(&lane, &queue, &pipeline, draw);
            let counts = lane.device.work_counts();
            // The answers first, so a failed check shows them too.
            println!(
                "{name}: {draw:?}: {}; {}",
                report(&outcome),
                submissions(&before, &counts)
            );
            check(&outcome, expected, &format!("{draw:?} on {name}"));
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// `outcome`'s answers in words: each command buffer's time from its commit to its completion,
/// to the draw's completion, and to its `Scheduled`; the other queue's signalling one's also from
/// the draw's commit.
fn report(outcome: &Outcome) -> String {
    let [drawn, rest @ ..] = &outcome.answers;
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
    let apart = |answer: &Answer| match answer.completed.checked_duration_since(drawn.completed) {
        Some(after) => ms(after),
        None => -ms(drawn.completed - answer.completed),
    };
    let rest: Vec<String> = rest
        .iter()
        .zip(OTHERS)
        .map(|(answer, slot)| {
            format!(
                "{} {:.1} ms ({:+.2} ms from the draw's, Scheduled at {:.1} ms)",
                MARKS[slot].0,
                ms(answer.took()),
                apart(answer),
                ms(answer.to_scheduled()),
            )
        })
        .collect();
    let signal = &outcome.signal;
    format!(
        "{:?} {:?} in {:.1} ms, Scheduled at {:.1} ms; {}; another queue's signal committed at \
         {:.1} ms from the draw's commit, {:.1} ms ({:+.2} ms from the draw's, Scheduled at \
         {:.1} ms)",
        drawn.status,
        drawn.error,
        ms(drawn.took()),
        ms(drawn.to_scheduled()),
        rest.join("; "),
        ms(signal.committed - drawn.committed),
        ms(signal.took()),
        apart(signal),
        ms(signal.to_scheduled()),
    )
}

/// How the scenario's batches went to their `VkQueue`s, from the device's counts `before` and
/// `after` it: the groups of batches kept while two or more earlier submissions of their
/// `VkQueue` ran and what sent the groups, the batches submitted behind two or more pending, and
/// the command queues that joined another's busy `VkQueue`.
fn submissions(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "groups kept pending {}, sent by a wait {}, drained {}, held {}; batches behind two or \
         more pending {}; queue joins {}",
        after.groups_kept_pending - before.groups_kept_pending,
        after.group_ends_wait - before.group_ends_wait,
        after.group_ends_drained - before.group_ends_drained,
        after.group_ends_held - before.group_ends_held,
        (after.batches_with_two_or_three_pending + after.batches_with_four_or_more_pending)
            - (before.batches_with_two_or_three_pending + before.batches_with_four_or_more_pending),
        after.queue_joins - before.queue_joins,
    )
}

/// Check `outcome` against Metal's answer `expected` (the module docs).
fn check(outcome: &Outcome, expected: Expected, label: &str) {
    let [drawn, at_once, later, during, on_gpu, held, other, after] = &outcome.answers;
    let marked = |slots: &[usize]| -> [u8; 10] {
        std::array::from_fn(|slot| match slots.contains(&slot) {
            true => MARKS[slot].1,
            false => 0,
        })
    };
    assert_eq!(outcome.event, 2, "{label}: the event's value");
    // The queue's command buffers committed during a hang.
    let queued = [
        (at_once, "at once"),
        (later, "10 ms on"),
        (during, "met by the host during the hang"),
        (on_gpu, "met by another queue's"),
    ];
    for (answer, which) in
        queued
            .into_iter()
            .chain([(held, "held"), (other, "another queue's"), (after, "after")])
    {
        assert!(answer.completed_cleanly(), "{label}: {which}: {answer:?}");
    }
    match expected {
        Expected::Nothing => {
            assert!(drawn.completed_cleanly(), "{label}: {drawn:?}");
            assert!(outcome.cleared, "{label}: the pass cleared its target");
            assert_eq!(
                outcome.marks,
                marked(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "{label}"
            );
        }
        Expected::Hang => {
            assert_eq!(drawn.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(drawn.error, Some(CommandBufferError::GpuHang), "{label}");
            assert!(
                drawn.since_scheduled() >= HANG,
                "{label}: failed {:?} after it read Scheduled",
                drawn.since_scheduled()
            );
            let scheduled = drawn.to_scheduled();
            assert!(
                scheduled >= SCHEDULED,
                "{label}: read Scheduled {scheduled:?} after its commit"
            );
            // The wait the other queue's command buffer meets was met during the hang where its
            // signal landed by then. Where the signal landed only after the hang's end (that
            // command buffer slow to complete beside other GPU work, committed 10 ms after the
            // draw's), the wait was met after the hang, and the command buffer runs, as the one
            // the host meets after the hang does and as Metal's met after the hang did. The hang
            // ends no sooner than [`SCHEDULED`] and [`HANG`] after the draw's commit, so it runs
            // only where the signal landed after that.
            let met_after = outcome.marks[6] == MARKS[6].1;
            if met_after {
                assert!(
                    outcome.signal.completed >= drawn.committed + SCHEDULED + HANG,
                    "{label}: met by another queue's ran, its signal landed during the hang"
                );
                assert!(
                    on_gpu.scheduled >= drawn.scheduled,
                    "{label}: met by another queue's read Scheduled before the hang"
                );
            }
            let discarded = match met_after {
                true => &queued[..3],
                false => &queued[..],
            };
            // The queue's command buffers committed during the hang read `Scheduled` from the
            // hang's on, before it completes, the host-met wait's once the host met it; the one
            // whose wait is met after the hang, after it.
            for &(answer, which) in discarded {
                assert!(
                    answer.scheduled >= drawn.scheduled && answer.scheduled < drawn.completed,
                    "{label}: {which} read Scheduled {:?} after its commit",
                    answer.to_scheduled()
                );
            }
            assert!(
                during.scheduled >= drawn.committed + DURING,
                "{label}: the host-met wait's read Scheduled before the host met it"
            );
            assert!(
                held.scheduled >= drawn.completed,
                "{label}: held read Scheduled before the hang completed"
            );
            assert!(!outcome.cleared, "{label}: the pass ran nothing");
            let ran: &[usize] = match met_after {
                true => &[0, 6, 7, 8, 9],
                false => &[0, 7, 8, 9],
            };
            assert_eq!(outcome.marks, marked(ran), "{label}");
            // Each completes after the hang, in the queue's order.
            for (answer, which) in queued {
                assert!(
                    answer.completed >= drawn.completed,
                    "{label}: {which} completed before the hang did"
                );
            }
            assert!(
                other.completed < drawn.completed,
                "{label}: another queue's command buffer waited for the hang"
            );
        }
        Expected::Fails(error) => {
            assert_eq!(drawn.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(drawn.error, Some(error), "{label}");
            assert!(
                drawn.since_scheduled() >= FAULT,
                "{label}: failed {:?} after it read Scheduled",
                drawn.since_scheduled()
            );
            // The queue's command buffer committed at once reads `Scheduled` during the fault and
            // completes after it.
            assert!(
                at_once.scheduled < drawn.completed && at_once.completed >= drawn.completed,
                "{label}: at once read Scheduled {:?} and completed {:?} after its commit",
                at_once.to_scheduled(),
                at_once.took()
            );
            assert!(!outcome.cleared, "{label}: the pass ran nothing");
            assert_eq!(outcome.marks, marked(&[0, 3, 4, 5, 6, 7, 8, 9]), "{label}");
        }
    }
}

/// A direct draw of type 9 of 3 vertices or more hangs Metal's GPU: its command buffer reads
/// `Scheduled` no sooner than Metal's did, long before it fails with `GpuHang` no sooner than
/// Metal's did, running nothing from the pass on but its event
/// signal, and the queue's command buffers committed meanwhile run nothing and complete cleanly
/// after it, those whose event waits were met during the hang too, while one whose wait is met
/// after it, another queue's and a later one run.
#[test]
fn a_direct_draw_of_type_nine_hangs_as_metal_s_gpu_does() {
    answers_as_metal_does(
        &[
            Draw::Direct {
                vertices: 3,
                instances: 1,
            },
            Draw::Direct {
                vertices: 4,
                instances: 1,
            },
            Draw::Direct {
                vertices: 3,
                instances: 2,
            },
        ],
        Expected::Hang,
    );
}

/// A direct draw of type 9 of fewer than 3 vertices, or of no instance, draws nothing: the pass
/// clears and every command buffer runs.
#[test]
fn a_direct_draw_of_type_nine_of_fewer_vertices_draws_nothing() {
    let direct = |vertices, instances| Draw::Direct {
        vertices,
        instances,
    };
    answers_as_metal_does(
        &[direct(0, 1), direct(1, 1), direct(2, 1), direct(3, 0)],
        Expected::Nothing,
    );
}

/// An indexed draw of type 9 fails its command buffer with `PageFault`, and an indirect one with
/// `GpuRecoveryVictim`, no sooner than Metal's did: the pass and the encoders after it run
/// nothing, the encoder before it and the queue's other command buffers run, its event signal
/// fires.
#[test]
fn indexed_draws_of_type_nine_fault_and_indirect_ones_are_victims() {
    let indexed = |indices, instances| Draw::Indexed { indices, instances };
    answers_as_metal_does(
        &[
            indexed(1, 1),
            indexed(2, 1),
            indexed(3, 1),
            indexed(0, 1),
            indexed(3, 0),
            indexed(0, 0),
        ],
        Expected::Fails(CommandBufferError::PageFault),
    );
    answers_as_metal_does(
        &[Draw::Indirect(3), Draw::Indirect(0), Draw::IndexedIndirect],
        Expected::Fails(CommandBufferError::GpuRecoveryVictim),
    );
}

/// One command buffer on `queue` of a mark into `marks` at `slot`, the draw of type 9 `draw` under
/// `pipeline` where there is one, and an event signal of 1; committed, waited for: its answer,
/// whether it marked, and the event's value.
fn one(
    lane: &Lane,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    draw: Option<Draw>,
) -> (Answer, bool, u64) {
    let device = &lane.device;
    let marks = buffer_with(device, &[0; 40]).unwrap();
    let event = device.new_shared_event().unwrap();
    let indices = buffer_with(device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let arguments: Vec<u8> = [3u32, 1, 0, 0, 0]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let arguments = buffer_with(device, &arguments).unwrap();
    let mut tracked = Tracked::new(queue);
    tracked.mark(&marks, 0);
    if let Some(draw) = draw {
        let (_targets, pass) = targets(lane);
        let mut encoder = tracked
            .command_buffer
            .render_command_encoder(&pass)
            .unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder
            .set_vertex_buffer(Some(&lane.positions), 0, 0)
            .unwrap();
        encoder
            .set_vertex_buffer(Some(&lane.offsets), 0, 1)
            .unwrap();
        let nine = DrawPrimitiveType::Nine;
        match draw {
            Draw::Direct {
                vertices,
                instances,
            } => encoder.draw_primitives(nine, 0, vertices, instances, 0),
            Draw::Indexed {
                indices: count,
                instances,
            } => encoder.draw_indexed_primitives(
                nine,
                count,
                IndexType::UInt16,
                &indices,
                0,
                instances,
                0,
                0,
            ),
            Draw::Indirect(_) => encoder.draw_primitives_indirect(nine, &arguments, 0),
            Draw::IndexedIndirect => encoder.draw_indexed_primitives_indirect(
                nine,
                IndexType::UInt16,
                &indices,
                0,
                &arguments,
                0,
            ),
        }
        .unwrap();
        encoder.end_encoding().unwrap();
    }
    tracked
        .command_buffer
        .encode_signal_event(EventRef::from(&event), 1)
        .unwrap();
    tracked.commit();
    tracked.wait();
    let marked = read_buffer(&marks, 0, 4).unwrap()[0] == MARKS[0].1;
    (tracked.answer(), marked, event.signaled_value())
}

/// On one queue, each of `draws` of type 9 in turn (its command buffer failing with its error),
/// each followed by a command buffer of no draw: those run until the queue's second fault or hang
/// completed, and from then on every command buffer of the queue fails at its commit with
/// `AccessRevoked`, running nothing and signalling nothing, while a new queue runs.
fn revokes_after(draws: &[(Draw, CommandBufferError)], revoked_from: Option<usize>) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipeline = lane.pipeline("pt_plain");
        let queue = lane.device.new_command_queue().unwrap();
        let revoked = |answer: &Answer, marked: bool, event: u64, label: &str| {
            assert_eq!(answer.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(
                answer.error,
                Some(CommandBufferError::AccessRevoked),
                "{label}"
            );
            assert!(!marked, "{label}: its mark ran");
            assert_eq!(event, 0, "{label}: its event signal fired");
            assert!(answer.scheduled <= answer.completed, "{label}");
        };
        for (at, &(draw, error)) in draws.iter().enumerate() {
            let label = format!("{draw:?} (draw {at}) on {name}");
            let (answer, marked, event) = one(&lane, &queue, &pipeline, Some(draw));
            match revoked_from {
                Some(from) if at > from => revoked(&answer, marked, event, &label),
                _ => {
                    assert_eq!(answer.status, CommandBufferStatus::Error, "{label}");
                    assert_eq!(answer.error, Some(error), "{label}");
                    assert!(marked, "{label}: its mark before the pass ran");
                    assert_eq!(event, 1, "{label}: its event signal");
                }
            }
            let (after, marked, event) = one(&lane, &queue, &pipeline, None);
            let label = format!("the command buffer after {label}");
            match revoked_from {
                Some(from) if at >= from => revoked(&after, marked, event, &label),
                _ => {
                    assert!(after.completed_cleanly(), "{label}: {after:?}");
                    assert!(marked && event == 1, "{label}: it ran");
                }
            }
            println!(
                "{name}: {draw:?}: {:?} in {:.1} ms; after it {:?} {:?} in {:.2} ms",
                answer.error,
                answer.took().as_secs_f64() * 1e3,
                after.status,
                after.error,
                after.took().as_secs_f64() * 1e3,
            );
        }
        let fresh = lane.device.new_command_queue().unwrap();
        let (answer, marked, event) = one(&lane, &fresh, &pipeline, None);
        assert!(
            answer.completed_cleanly(),
            "a new queue on {name}: {answer:?}"
        );
        assert!(marked && event == 1, "a new queue's command buffer ran");
    }
    layer_reported_nothing_at_all(&messages);
}

/// A queue's second command buffer that faulted revokes it, as Metal's does: every command buffer
/// committed on it after that one completed fails with `AccessRevoked` (a fault's draw too), and
/// a new queue runs.
#[test]
fn a_queue_s_second_fault_revokes_it() {
    let fault = CommandBufferError::PageFault;
    let indexed = Draw::Indexed {
        indices: 3,
        instances: 1,
    };
    revokes_after(
        &[(indexed, fault), (indexed, fault), (indexed, fault)],
        Some(1),
    );
}

/// Victims of the GPU's recovery do not count against their queue, a hang does: victims and a
/// fault leave the queue running, and a hang and a fault revoke it.
#[test]
fn victims_do_not_count_against_their_queue_and_a_hang_does() {
    let victim = CommandBufferError::GpuRecoveryVictim;
    let fault = CommandBufferError::PageFault;
    let indexed = Draw::Indexed {
        indices: 3,
        instances: 1,
    };
    revokes_after(
        &[
            (Draw::Indirect(3), victim),
            (Draw::IndexedIndirect, victim),
            (indexed, fault),
            (Draw::Indirect(0), victim),
        ],
        None,
    );
    let hang = Draw::Direct {
        vertices: 3,
        instances: 1,
    };
    revokes_after(
        &[
            (hang, CommandBufferError::GpuHang),
            (indexed, fault),
            (hang, CommandBufferError::GpuHang),
        ],
        Some(1),
    );
}
