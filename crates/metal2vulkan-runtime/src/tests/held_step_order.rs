//! A command buffer whose second batch waits at its host step's gate for the GPU's long work
//! before it (a rate-mapped pass executing an indirect command buffer whose records that work
//! resets, `tests/host_step_rules.rs`'s workload), and an unshared queue's command buffer
//! committed while that work runs (`scheduler/order.rs`, **Held batches give way**):
//!
//! - one small fill of a buffer of its own, which shares nothing with the pass: its submission
//!   waits for no host step, whether the pass's uses are bounded (it counts no visibility results,
//!   `DeferredPass::bound`) or not known until its step records it (it does: `tracker::ordered`'s
//!   `None`). On an RTX 3090 a step so held every queue's commits, a compositor's unshared ones
//!   included, until an 89 ms pass before it ended; on lavapipe, before the step gave way, the
//!   fill of the second kind went 111 ms after its commit, as the pass's batch did;
//! - a reset of the records the pass's step reads: it waits behind the step's batch, as the gate
//!   covers no write submitted after it.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::host_step_rules::{Heard, Workload};
use crate::{
    Buffer, CommandBuffer, CommandQueue, LoadAction, RenderPassDescriptor, ResourceOptions,
    StoreAction, TextureDescriptor, VisibilityResultMode,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// The bytes one fill of the long work writes.
const FILL_BYTES: u64 = 64 << 20;

/// The long work's length the fills are calibrated to, in seconds.
const LONG: f64 = 0.1;

/// The runs a leg makes, at most, for one whose fill was committed before the pass's batch went
/// (the test's thread late on a loaded host shows nothing).
const RUNS: usize = 3;

/// What the unshared queue's command buffer does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unshared {
    /// One small fill of a buffer of its own.
    Fill,
    /// A reset of the pass's empty command, in the records its step reads.
    Reset,
}

/// One run's host times, in seconds of `host_time`'s clock, and its holds.
struct Run {
    /// The unshared queue's commit returned.
    committed: f64,
    /// The scheduler called its submission (its kernel end).
    called: f64,
    /// The scheduler called the pass's batch's submission (the long command buffer's kernel end),
    /// once the long work completed and the step recorded the pass.
    pass_called: f64,
    /// The long command buffer's GPU end.
    long_end: f64,
    /// Looks that held a head behind an earlier commit
    /// (`WorkCounts::heads_held_for_earlier_commits`).
    held: u64,
}

impl Run {
    /// Whether the unshared one was committed at least half the long work before the pass's
    /// batch went: else the test's thread came late (a loaded host), and the run shows nothing.
    fn early(&self) -> bool {
        self.pass_called - self.committed > LONG / 2.0
    }
}

/// The fills that take about [`LONG`] of `queue`'s GPU.
fn long_fills(queue: &CommandQueue, target: &Buffer) -> usize {
    let mut fills = 64;
    for _ in 0..3 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for fill in 0..fills {
            blit.fill_buffer(target, 0..FILL_BYTES, fill as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        fills = ((fills as f64 * LONG / gpu.max(1e-6)) as usize).clamp(8, 4096);
    }
    fills
}

/// The long command buffer: `fills` fills of `big`, a reset of the buffer's empty command after
/// them, then the pass executing both commands, counting visibility results where `counted`.
fn long(work: &Workload, big: &Buffer, fills: usize, counted: bool) -> CommandBuffer {
    let size = work.map.physical_size_for_layer(0);
    let mut d =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    d.usage = TextureUsage::RENDER_TARGET;
    let target = work.device.new_texture(&d).unwrap();
    let command_buffer = work.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for fill in 0..fills {
        blit.fill_buffer(big, 0..FILL_BYTES, fill as u8).unwrap();
    }
    // The GPU writes the records last: the pass's step waits for the whole batch.
    blit.reset_commands_in_buffer(&work.icb, 1..2).unwrap();
    blit.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    pass.rasterization_rate_map = Some(work.map.clone());
    pass.visibility_result_buffer = Some(work.visibility.clone());
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&work.pipeline).unwrap();
    if counted {
        (encoder.set_visibility_result_mode(VisibilityResultMode::Boolean, 0)).unwrap();
    }
    encoder.execute_commands_in_buffer(&work.icb, 0..2).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Commit the long command buffer, wait until its pass's batch waits at its gate, then commit the
/// unshared queue's command buffer, `what`.
fn run(
    work: &Workload,
    unshared: &CommandQueue,
    (big, small): (&Buffer, &Buffer),
    fills: usize,
    (counted, what): (bool, Unshared),
) -> Run {
    let long = long(work, big, fills, counted);
    let before = work.device.work_counts();
    long.commit().unwrap();
    let began = Instant::now();
    while work.device.work_counts().heads_waiting_at_their_gate
        == before.heads_waiting_at_their_gate
    {
        assert!(
            began.elapsed() < WAIT,
            "the pass's batch waited at its gate"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let other = unshared.command_buffer().unwrap();
    let mut blit = other.blit_command_encoder().unwrap();
    match what {
        Unshared::Fill => blit.fill_buffer(small, 0..256, 1).unwrap(),
        Unshared::Reset => blit.reset_commands_in_buffer(&work.icb, 1..2).unwrap(),
    }
    blit.end_encoding().unwrap();
    other.commit().unwrap();
    let committed = crate::host_time();
    other.wait_until_completed_for(WAIT).unwrap();
    long.wait_until_completed_for(WAIT).unwrap();
    let after = work.device.work_counts();
    Run {
        committed,
        called: other.kernel_end_time(),
        pass_called: long.kernel_end_time(),
        long_end: long.gpu_end_time().unwrap(),
        held: after.heads_held_for_earlier_commits - before.heads_held_for_earlier_commits,
    }
}

/// The unshared queue's fill, the pass's uses bounded or not known, is submitted with no look
/// holding it behind an earlier commit, before half the long work ran; its reset of the records
/// the step reads waits behind the pass's batch, held so.
#[test]
fn an_unshared_queue_s_fill_does_not_wait_for_another_queue_s_host_step() {
    let scratch = Scratch::new("held-step-order");
    let work = Workload::new(&scratch, &Heard::default());
    let unshared = work.device.new_command_queue_unshared(64).unwrap();
    let private = ResourceOptions::private();
    let big = work.device.new_buffer(FILL_BYTES, private).unwrap();
    let small = work.device.new_buffer(4096, private).unwrap();
    let fills = long_fills(&work.queue, &big);
    let legs = [
        ("a fill, the pass bounded", false, Unshared::Fill),
        ("a fill, the pass's uses not known", true, Unshared::Fill),
        ("a reset of the records", false, Unshared::Reset),
    ];
    for (leg, counted, what) in legs {
        let run = (0..RUNS)
            .map(|_| run(&work, &unshared, (&big, &small), fills, (counted, what)))
            .inspect(|run| {
                println!(
                    "{leg}: committed at 0, called at {:.2} ms; the pass called at {:.2} ms, the \
                     long work's end {:.2} ms; {} looks held behind an earlier commit",
                    (run.called - run.committed) * 1e3,
                    (run.pass_called - run.committed) * 1e3,
                    (run.long_end - run.committed) * 1e3,
                    run.held
                )
            })
            .find(Run::early)
            .expect("a run committed the unshared one before half the long work ran");
        match what {
            Unshared::Fill => {
                assert_eq!(run.held, 0, "{leg}: no look held it behind the pass");
                assert!(
                    run.called - run.committed < LONG / 2.0,
                    "{leg}: it went before half the long work ran"
                );
            }
            Unshared::Reset => {
                assert!(run.held > 0, "{leg}: a look held it behind the pass");
                assert!(
                    run.called > run.pass_called,
                    "{leg}: it went after the pass's batch"
                );
            }
        }
    }
}
