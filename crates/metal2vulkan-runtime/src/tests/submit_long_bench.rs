//! Where a guest compute loop's `vkQueueSubmit` calls take a millisecond or more: the shape of a
//! simulation step's command buffers on a device answering GPU times from the host's clock, as a
//! consumer's is, beside the dma-bufs a guest's buffers and textures keep listed, with and without
//! an unshared queue (a compositor's) committing on a thread of its own.
//!
//! Each arm runs for [`RUN`] on a device made as the consumer's (`gpu_times_from_host`, its queues
//! planned, so an unshared queue shares the one `VkQueue` of RADV's graphics family): an
//! application thread commits command buffers of [`CHAINED`] compute encoders in a chain, each
//! updating a fence, so each goes in a `VkCommandBuffer` of its own with its barrier in one
//! between (47 a submission), then a fill of [`FILL_BYTES`] (about 3 ms of the Radeon 780M's
//! time), waiting for each before the next, or keeping [`AHEAD`] in flight. With the compositor, a
//! thread commits a small fill on an unshared queue every [`FRAME`]; arms with a fill of
//! [`SMALL_FILL`] leave the GPU's 3 ms out. Listed: [`LISTED`] no-copy buffers of 64 KiB for every
//! arm, then [`LARGE`] of [`LARGE_BYTES`] more for every arm again, each its own dma-buf (udmabuf)
//! over a sealed `memfd`'s pages, each bound once by a dispatch, so kept imported.
//!
//! Three arms leave the fence out, so the chain and the fill go in one `VkCommandBuffer` (the
//! runtime records the encoders of one batch into one where nothing between them needs a
//! boundary, `encoder_fold.rs`): the submission's cost of its 47 `VkCommandBuffer`s after the
//! wait, against one.
//!
//! Two more arms submit on the committing thread (`commit_and_submit`), waiting for each command
//! buffer: as it comes, and after spinning [`SPIN`] on the CPU first, which brings its core's clock
//! back up after the wait but leaves what the call reads as cold in the caches.
//!
//! Printed (`SUBMIT-LONG`), per arm: the submissions (those made at commit apart), the driver's
//! calls of a millisecond or more and the submitting thread's CPU time across them
//! (`WorkCounts::submit_long_calls` and kin), the mean call and its CPU time
//! (`WorkCounts::batch_submit_cpu_nanos`), the `VkCommandBuffer`s and timeline waits a
//! submission, and the heads held for the unshared queue, the batches' hold time and the
//! scheduler's slice rescues.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` where `/dev/udmabuf` opens,
//! `cargo test --release -p metal2vulkan-runtime --lib submit_long_bench -- --ignored --nocapture`.

use super::blit_holds::{options, Scratch, WAIT};
use super::host_residency::{udmabuf_openable, Ram};
use super::submit_tax_bench::Chain;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, Fence, HostMemory, ResourceOptions, Size,
    WorkCounts,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The small listed buffers, and each one's bytes.
const LISTED: usize = 320;
const LISTED_BYTES: usize = 64 << 10;
/// The large listed buffers, and each one's bytes (a 1920x1080 BGRA frame's).
const LARGE: usize = 40;
const LARGE_BYTES: usize = 8 << 20;
/// The chained encoders of a command buffer.
const CHAINED: u64 = 24;
/// The fill closing each command buffer.
const FILL_BYTES: u64 = 256 << 20;
/// The application's command buffers in flight, where it keeps several.
const AHEAD: usize = 3;
/// The compositor's period.
const FRAME: Duration = Duration::from_millis(16);
/// Each arm's length.
const RUN: Duration = Duration::from_secs(2);

/// The guest RAM of the bench: the small listed buffers' pages, then the large ones'.
struct Pages {
    ram: Ram,
    calls: Arc<AtomicUsize>,
}

impl Pages {
    fn new() -> Pages {
        let len = LISTED * LISTED_BYTES + LARGE * LARGE_BYTES;
        Pages {
            ram: Ram::new(len, 0, len),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A no-copy buffer of `length` bytes from `offset`, imported as a dma-buf of its pages.
    fn buffer(&self, device: &Device, offset: usize, length: usize) -> Buffer {
        // SAFETY: the view outlives every buffer (the test drops its buffers before the pages),
        // and the source makes dma-bufs of its pages.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::with_dma_buf(self.ram.pointer(offset), self.ram.source(&self.calls)),
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }
}

/// How an arm's application commits, how much its fill writes, and whether the compositor runs
/// beside it.
#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    ahead: usize,
    fill: u64,
    compositor: bool,
    submit: Submit,
    fenced: bool,
}

/// Who submits an arm's command buffers.
#[derive(Clone, Copy, PartialEq)]
enum Submit {
    /// The scheduler's thread, after `commit`.
    Scheduler,
    /// The committing thread itself (`commit_and_submit`).
    Committer,
    /// The committing thread itself, after spinning [`SPIN`] on the CPU: its core's clock up
    /// again after the wait for the command buffer before, its caches as cold.
    WarmCommitter,
}

/// How long a warm committer spins before each commit.
const SPIN: Duration = Duration::from_micros(500);

/// A small fill's bytes: the arms without the GPU's 3 ms.
const SMALL_FILL: u64 = 4096;

/// The arms, run once with the small listed buffers alone and once with the large ones too.
const ARMS: [Arm; 11] = [
    Arm {
        name: "waited for each",
        ahead: 1,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "3 in flight",
        ahead: AHEAD,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "waited for each, small fill",
        ahead: 1,
        fill: SMALL_FILL,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "3 in flight, small fill",
        ahead: AHEAD,
        fill: SMALL_FILL,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "waited for each, compositor",
        ahead: 1,
        fill: FILL_BYTES,
        compositor: true,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "3 in flight, compositor",
        ahead: AHEAD,
        fill: FILL_BYTES,
        compositor: true,
        submit: Submit::Scheduler,
        fenced: true,
    },
    Arm {
        name: "waited for each, submitted by the committer",
        ahead: 1,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::Committer,
        fenced: true,
    },
    Arm {
        name: "waited for each, submitted by the committer after a spin",
        ahead: 1,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::WarmCommitter,
        fenced: true,
    },
    Arm {
        name: "waited for each, unfenced",
        ahead: 1,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: false,
    },
    Arm {
        name: "3 in flight, unfenced",
        ahead: AHEAD,
        fill: FILL_BYTES,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: false,
    },
    Arm {
        name: "waited for each, small fill, unfenced",
        ahead: 1,
        fill: SMALL_FILL,
        compositor: false,
        submit: Submit::Scheduler,
        fenced: false,
    },
];

/// The application's command buffer: the chain, each encoder updating `fence` where there is
/// one, and the fill.
fn step(
    queue: &CommandQueue,
    chain: &Chain,
    fence: Option<&Fence>,
    (fill, bytes): (&Buffer, u64),
    submit: Submit,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    chain.encode(&command_buffer, CHAINED, fence);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(fill, 0..bytes, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    match submit {
        Submit::Scheduler => command_buffer.commit().unwrap(),
        Submit::Committer => command_buffer.commit_and_submit().unwrap(),
        Submit::WarmCommitter => {
            let start = Instant::now();
            while start.elapsed() < SPIN {
                std::hint::spin_loop();
            }
            command_buffer.commit_and_submit().unwrap();
        }
    }
    command_buffer
}

/// Run `arm` for [`RUN`]: the counts' change over it.
fn run(device: &Device, arm: Arm, chain: &Chain, fill: &Buffer) -> (WorkCounts, WorkCounts) {
    let queue = device.new_command_queue().unwrap();
    let fence = device.new_fence();
    let fence = arm.fenced.then_some(&fence);
    let stop = Arc::new(AtomicBool::new(false));
    let compositor = arm.compositor.then(|| {
        let device = device.clone();
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let queue = device.new_command_queue_unshared(4).unwrap();
            let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
            let mut next = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&target, 0..4096, 1).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                next += FRAME;
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
        })
    });
    // Warm the queue's runs before counting.
    step(&queue, chain, fence, (fill, arm.fill), arm.submit)
        .wait_until_completed_for(WAIT)
        .unwrap();
    let before = device.work_counts();
    let start = Instant::now();
    let mut flight: VecDeque<CommandBuffer> = VecDeque::new();
    while start.elapsed() < RUN {
        if flight.len() == arm.ahead {
            let oldest = flight.pop_front().unwrap();
            oldest.wait_until_completed_for(WAIT).unwrap();
        }
        flight.push_back(step(&queue, chain, fence, (fill, arm.fill), arm.submit));
    }
    for command_buffer in flight {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    super::scheduler_wakes::asleep(device, "the arm's last pass");
    let after = device.work_counts();
    stop.store(true, Ordering::Relaxed);
    if let Some(thread) = compositor {
        thread.join().unwrap();
    }
    (before, after)
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn submit_long_bench() {
    if !udmabuf_openable("submit_long_bench") {
        return;
    }
    let scratch = Scratch::new("submit-long");
    let mut asked = options(&scratch.0);
    asked.queue_limits = None;
    asked.gpu_times_from_host = true;
    let device = Device::system_default_with(&asked).unwrap();
    // The thread's CPU time across each submission is counted where queue times are noted.
    device.shared().timing.note_queue_times();
    println!("SUBMIT-LONG on {}", device.name());
    let chain = Chain::new(&device);
    let fill = device
        .new_buffer(FILL_BYTES, ResourceOptions::private())
        .unwrap();
    let pages = Pages::new();
    let small: Vec<Buffer> = (0..LISTED)
        .map(|i| pages.buffer(&device, i * LISTED_BYTES, LISTED_BYTES))
        .collect();
    bind_all(&device, &small);
    run_arms(&device, "small listed", &chain, &fill);
    let large: Vec<Buffer> = (0..LARGE)
        .map(|i| {
            pages.buffer(
                &device,
                LISTED * LISTED_BYTES + i * LARGE_BYTES,
                LARGE_BYTES,
            )
        })
        .collect();
    bind_all(&device, &large);
    run_arms(&device, "small and large listed", &chain, &fill);
    drop((small, large));
}

/// Each of `listed` bound by a dispatch of a command buffer completed, 48 a command buffer: kept
/// imported for good (`host_residency.rs`).
fn bind_all(device: &Device, listed: &[Buffer]) {
    let state = super::dispatch_pending::state(device);
    let scale =
        (device.new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())).unwrap();
    let binder = device.new_command_queue().unwrap();
    for group in listed.chunks(48) {
        let command_buffer = binder.command_buffer().unwrap();
        for buffer in group {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&scale), 0, 0).unwrap();
            encoder.set_buffer(Some(buffer), 0, 1).unwrap();
            encoder.set_buffer(Some(&scale), 0, 2).unwrap();
            encoder
                .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
}

/// Every arm of [`ARMS`] with what is listed now, `listed` naming it.
fn run_arms(device: &Device, listed: &str, chain: &Chain, fill: &Buffer) {
    println!(
        "SUBMIT-LONG {listed}: dma-buf memories live {}",
        device.work_counts().dma_buf_memories_live
    );
    for arm in ARMS {
        let (before, after) = run(device, arm, chain, fill);
        let calls = (after.queue_submissions - before.queue_submissions).max(1);
        println!(
            "SUBMIT-LONG {listed}, {}: {} submissions ({} at commit), {} calls of 1 ms or more \
             ({:.0} us of CPU in their {:.0} us); mean call {:.0} us, {:.0} us of it the thread's \
             CPU; {:.1} VkCommandBuffers and {:.2} timeline waits a submission; {} heads held for \
             the unshared queue, batches held {:.0} ms in all, {} slice rescues",
            arm.name,
            calls,
            after.batches_submitted_at_commit - before.batches_submitted_at_commit,
            after.submit_long_calls - before.submit_long_calls,
            (after.submit_long_call_cpu_nanos - before.submit_long_call_cpu_nanos) as f64 / 1e3,
            (after.submit_long_call_nanos - before.submit_long_call_nanos) as f64 / 1e3,
            (after.batch_submit_nanos - before.batch_submit_nanos) as f64 / calls as f64 / 1e3,
            (after.batch_submit_cpu_nanos - before.batch_submit_cpu_nanos) as f64
                / calls as f64
                / 1e3,
            (after.command_buffers_submitted - before.command_buffers_submitted) as f64
                / calls as f64,
            (after.timeline_waits - before.timeline_waits) as f64 / calls as f64,
            after.heads_held_for_unshared_isolation - before.heads_held_for_unshared_isolation,
            (after.batch_hold_nanos - before.batch_hold_nanos) as f64 / 1e6,
            after.scheduler_slice_rescues - before.scheduler_slice_rescues,
        );
    }
}
