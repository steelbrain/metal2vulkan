//! What a guest's window drag costs its committing thread where the device's calls of commands
//! are slow for the imports it lists, and some of the calls hold no command buffer
//! (`device/submit_calls.rs`). A guest's RAM of no-copy buffers over a sealed `memfd`'s pages,
//! each its own dma-buf (udmabuf) of [`LISTED_BYTES`], as a guest's RAM chunks are on a host of
//! unified memory, each bound once by a dispatch, so kept imported for good and listed in every
//! kernel submission of commands (`host_residency.rs`): [`LISTED`] of them, then as many more.
//! With each, after [`WARM`] frames, [`FRAMES`] frames of [`PARTS`] command buffers committed with
//! `commit_and_submit` one after another on one queue, as reims commits the parts it cuts from a
//! compositor's command buffer: each a dispatch, but the last, which encodes nothing but an
//! event's signal, so its batch holds no command buffer (the empty last part reims's decoder cut
//! from most of a window drag's buffers on the Radeon 780M; one that signals nothing makes no
//! call); the frame's parts are waited for before the next frame. Each with the driver's calls,
//! then with each call of commands made [`MIXED_PAUSE`] longer but one in each [`CALLS`]
//! (`Device::slow_submit_calls`, `Device::fast_submit_call_every`), a short call among slow ones,
//! as one guest queue's smaller batches come among another's, then but one in each three, as the
//! Iris Xe's compositor parts straddled the bound (`device/submit_calls.rs`, "Measured, not
//! named"). Each of those three times: as the queue's cost leaves the batches, then each at its
//! commit, then each left to the scheduler's thread (`Device::leave_calls_from`), so that what a
//! committing thread saves by leaving a batch stands beside what its GPU start loses.
//!
//! Printed (`DRAG-SUBMIT`), each: the parts submitted at their commit and those left for slow
//! calls (`WorkCounts::batches_submitted_at_commit`,
//! `WorkCounts::scheduler_at_commit_left_slow_calls`), the commit calls' time a part (the parts
//! with work's median, 90th percentile and mean; the empty ones' mean; all of them a frame), the
//! time from each part with work's commit call to its GPU start (`CommandBuffer::gpu_start_time`;
//! median and 90th percentile), the batch calls' mean (`WorkCounts::batch_submit_nanos` a
//! `queue_submissions`, the `VkCommandBuffer`s a call) and the queue's submission cost at the
//! end. And (`DRAG-PATH`) where the time from a commit call to its GPU start went, for each part
//! with work (medians and 90th percentiles): to the start of its submission
//! (`CommandBuffer::kernel_start_time`: the committing thread's, or the scheduler's thread's once
//! woken and at it), from there to the return of the call that carried it
//! (`CommandBuffer::submit_returned_time`, the device noting queue times: its planning, any
//! hold in a group, and the call), and from there to its GPU start; and why the commits' passes
//! left batches (the `WorkCounts::scheduler_at_commit_left_*` counts) and the groups held past
//! a pass for two submissions in flight (`WorkCounts::groups_kept_pending`); the device's
//! submission checks' report (`Device::submit_check_report`) heads them and ends them, as the
//! merge check decides whether its batches group. And (`DRAG-FRAME`)
//! each frame's whole (medians and 90th percentiles): from its first commit call to the GPU start
//! and the GPU end of its last part with work, and its commit calls' time, the committing
//! thread's for the frame. A part's time from its own commit call leaves out the commit calls
//! before it, which hold back its commit where the committing thread submits each part itself.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` where `/dev/udmabuf` opens,
//! `cargo test --release -p metal2vulkan-runtime --lib drag_submit_bench -- --ignored --nocapture`.

use super::blit_holds::{Scratch, WAIT};
use super::compute_parked::Kernel;
use super::host_residency::{udmabuf_openable, Ram};
use super::submit_tax_bench::bind_all;
use crate::device::submit_calls::CALLS;
use crate::gpu_time;
use crate::{Buffer, CommandBufferStatus, EventRef, HostMemory, ResourceOptions};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The imports of the first measurement, and of each more.
const LISTED: usize = 140;
/// Each import's bytes: a guest RAM chunk's.
const LISTED_BYTES: usize = 2 << 20;
/// The parts of a frame, the last empty.
const PARTS: usize = 13;
/// The frames before each measurement.
const WARM: usize = 20;
/// The frames measured.
const FRAMES: usize = 200;
/// How much longer each call of commands but one in each [`CALLS`], then one in each three, takes
/// in the second and third measurements of each.
const MIXED_PAUSE: Duration = Duration::from_micros(100);

/// What a measurement's frames noted, in microseconds, of its parts: each commit call's (those with
/// work, the empty ones'); and of those with work, from the commit call to the GPU start, to the
/// start of the part's submission, from there to the return of the call that carried it, and
/// from there to the GPU start. And of each frame: from its first commit call to its last part
/// with work's GPU start and GPU end, and its commit calls' time.
#[derive(Default)]
struct Times {
    work: Vec<f64>,
    empty: Vec<f64>,
    started: Vec<f64>,
    picked: Vec<f64>,
    called: Vec<f64>,
    ran: Vec<f64>,
    frame_started: Vec<f64>,
    frame_ended: Vec<f64>,
    frame_commits: Vec<f64>,
}

/// The `at` quantile of `values`, sorted.
fn quantile(values: &[f64], at: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
        .get(((sorted.len() as f64 * at) as usize).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// The mean of `values`.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn drag_submit_bench() {
    if !udmabuf_openable("drag_submit_bench") {
        return;
    }
    let scratch = Scratch::new("drag-submit");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let device = &kernel.device;
    // Each command buffer's submission's return (`CommandBuffer::submit_returned_time`).
    device.shared().timing.note_queue_times();
    println!(
        "DRAG-PATH on {}: completion {:?}, submissions merge {}; its submission checks: {}",
        device.name(),
        device.compile_device().policy().completion,
        device.submissions_merge(),
        device.submit_check_report(),
    );
    let len = 2 * LISTED * LISTED_BYTES;
    let ram = Ram::new(len, 0, len);
    let calls = Arc::new(AtomicUsize::new(0));
    let output = kernel.output(PARTS as u64);
    let queue = device.new_command_queue().unwrap();
    let event = device.new_event().unwrap();
    let signalled = AtomicU64::new(0);
    let mut listed: Vec<Buffer> = Vec::new();
    for _ in 0..2 {
        let made: Vec<Buffer> = (listed.len()..listed.len() + LISTED)
            .map(|at| {
                // SAFETY: the view outlives every buffer (dropped before `ram`), and the source
                // makes dma-bufs of its pages.
                unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::with_dma_buf(
                            ram.pointer(at * LISTED_BYTES),
                            ram.source(&calls),
                        ),
                        LISTED_BYTES as u64,
                        ResourceOptions::default(),
                        None,
                    )
                }
                .unwrap()
            })
            .collect();
        bind_all(&kernel, &made);
        listed.extend(made);
        let frame = |times: Option<&mut Times>| {
            let mut parts = Vec::with_capacity(PARTS);
            let mut took = Vec::with_capacity(PARTS);
            let mut committed = Vec::with_capacity(PARTS);
            for part in 0..PARTS {
                let command_buffer = queue.command_buffer().unwrap();
                if part + 1 < PARTS {
                    kernel.dispatch(&command_buffer, (&output, part as u64 * 256));
                } else {
                    let value = signalled.fetch_add(1, Ordering::Relaxed) + 1;
                    command_buffer
                        .encode_signal_event(EventRef::from(&event), value)
                        .unwrap();
                }
                committed.push(gpu_time::host_nanos());
                let called = Instant::now();
                command_buffer.commit_and_submit().unwrap();
                took.push(called.elapsed().as_nanos() as f64 / 1e3);
                parts.push(command_buffer);
            }
            for command_buffer in &parts {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            if let Some(times) = times {
                times.work.extend_from_slice(&took[..PARTS - 1]);
                times.empty.push(took[PARTS - 1]);
                times.frame_commits.push(took.iter().sum());
                let first = gpu_time::seconds(committed[0]);
                let last = &parts[PARTS - 2];
                if let (Ok(start), Ok(end)) = (last.gpu_start_time(), last.gpu_end_time()) {
                    times.frame_started.push((start - first) * 1e6);
                    times.frame_ended.push((end - first) * 1e6);
                }
                for (command_buffer, &at) in parts.iter().zip(&committed).take(PARTS - 1) {
                    let at = gpu_time::seconds(at);
                    let Ok(start) = command_buffer.gpu_start_time() else {
                        continue;
                    };
                    times.started.push((start - at) * 1e6);
                    let (kernel, returned) = (
                        command_buffer.kernel_start_time(),
                        command_buffer.submit_returned_time(),
                    );
                    if kernel > 0.0 && returned > 0.0 {
                        times.picked.push((kernel - at) * 1e6);
                        times.called.push((returned - kernel) * 1e6);
                        times.ran.push((start - returned) * 1e6);
                    }
                }
            }
        };
        for ((calls_were, pause, every), (placed, bound)) in [
            ("the driver's", None, None),
            (
                "100 us longer but one in each 15",
                Some(MIXED_PAUSE),
                Some(CALLS as u64),
            ),
            (
                "100 us longer but one in each 3",
                Some(MIXED_PAUSE),
                Some(3),
            ),
        ]
        .into_iter()
        .flat_map(|leg| {
            [
                ("as their cost leaves them", None),
                ("each at its commit", Some(u64::MAX)),
                ("each left to the scheduler's thread", Some(1)),
            ]
            .map(|placed| (leg, placed))
        }) {
            device.slow_submit_calls(pause);
            device.fast_submit_call_every(every);
            device.leave_calls_from(bound);
            for _ in 0..WARM {
                frame(None);
            }
            let mut times = Times::default();
            let before = device.work_counts();
            let started = Instant::now();
            for _ in 0..FRAMES {
                frame(Some(&mut times));
            }
            let Times {
                work,
                empty,
                started: gpu_started,
                picked,
                called,
                ran,
                frame_started,
                frame_ended,
                frame_commits,
            } = times;
            let elapsed = started.elapsed().as_secs_f64();
            let after = device.work_counts();
            let moved = |count: fn(&crate::WorkCounts) -> u64| count(&after) - count(&before);
            let parts = (FRAMES * PARTS) as f64;
            let at_commit = moved(|c| c.batches_submitted_at_commit);
            let slow = moved(|c| c.scheduler_at_commit_left_slow_calls);
            let submissions = moved(|c| c.queue_submissions).max(1) as f64;
            let cost = device.shared().queues[queue.vk_queue_sharing().0]
                .calls
                .cost();
            println!(
                "DRAG-SUBMIT listed {} (live dma-buf memories {}), calls of commands \
                 {calls_were}, batches {placed}, on {}: {} parts ({FRAMES} frames of {PARTS}, the \
                 last empty) in {elapsed:.2} s; at commit {at_commit} ({:.1}%), left for slow \
                 calls {slow} ({:.1}%); commit call a part, us: with work p50 {:.1} p90 {:.1} \
                 mean {:.1}, empty mean {:.1}, all a frame {:.0}; commit to GPU start, with \
                 work, us: p50 {:.1} p90 {:.1} ({} read); batch calls {} of mean {:.1} us, \
                 {:.1} VkCommandBuffers a call; the queue's submission cost {:.1} us",
                listed.len(),
                after.dma_buf_memories_live,
                device.name(),
                parts,
                100.0 * at_commit as f64 / parts,
                100.0 * slow as f64 / parts,
                quantile(&work, 0.5),
                quantile(&work, 0.9),
                mean(&work),
                mean(&empty),
                (work.iter().sum::<f64>() + empty.iter().sum::<f64>()) / FRAMES as f64,
                quantile(&gpu_started, 0.5),
                quantile(&gpu_started, 0.9),
                gpu_started.len(),
                submissions,
                moved(|c| c.batch_submit_nanos) as f64 / submissions / 1e3,
                moved(|c| c.command_buffers_submitted) as f64 / submissions,
                cost as f64 / 1e3,
            );
            let spread = |values: &[f64]| (quantile(values, 0.5), quantile(values, 0.9));
            println!(
                "DRAG-PATH listed {}, calls of commands {calls_were}, batches {placed}: commit to \
                 submission start p50/p90 {:.1?} us, to the call's return {:.1?}, to GPU start \
                 {:.1?} ({} read); passes left: under way {}, taken in {}, behind its queue {}, \
                 turns {}, slow calls {slow}, other {}; groups' calls {}, of {} batches, kept \
                 for two in flight {}",
                listed.len(),
                spread(&picked),
                spread(&called),
                spread(&ran),
                picked.len(),
                moved(|c| c.scheduler_at_commit_left_passing),
                moved(|c| c.scheduler_at_commit_left_taken_in),
                moved(|c| c.scheduler_at_commit_left_behind_its_queue),
                moved(|c| c.scheduler_at_commit_left_turns),
                moved(|c| c.scheduler_at_commit_left_other),
                moved(|c| c.grouped_submissions),
                moved(|c| c.batches_grouped),
                moved(|c| c.groups_kept_pending),
            );
            println!(
                "DRAG-FRAME listed {}, calls of commands {calls_were}, batches {placed}: a frame's \
                 first commit call to its last part's GPU start p50/p90 {:.1?} us, to its GPU \
                 end {:.1?} ({} read); its commit calls {:.1?} us",
                listed.len(),
                spread(&frame_started),
                spread(&frame_ended),
                frame_started.len(),
                spread(&frame_commits),
            );
        }
        device.slow_submit_calls(None);
        device.fast_submit_call_every(None);
        device.leave_calls_from(None);
    }
    println!(
        "DRAG-PATH after the measurements: submissions merge {}; its submission checks: {}",
        device.submissions_merge(),
        device.submit_check_report(),
    );
    drop(listed);
}
