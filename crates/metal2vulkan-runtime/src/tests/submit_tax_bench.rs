//! What a submission costs a guest's compute loop, term by term: each command buffer committed
//! and waited for, as a guest that times its kernels commits them, and per command buffer the
//! driver's time in its `vkQueueSubmit` calls (`WorkCounts::batch_submit_nanos`), the
//! `VkCommandBuffer`s they carried (`command_buffers_submitted`) and the time from its kernel end
//! (the call's start) to its GPU start (calibrated where the device calibrates).
//!
//! The arms, interleaved over [`ROUNDS`] rounds:
//! - **Listed imports**: none of the bench's own, or [`LISTED`] no-copy buffers of
//!   [`LISTED_BYTES`] over a sealed `memfd`'s pages, each its own dma-buf (udmabuf) as a guest's
//!   buffers are where the device refuses their pointer (RADV), each bound once by a dispatch, so
//!   kept imported for good (`host_residency.rs`) and listed in every kernel submission.
//! - **Encoders**: [`ENCODERS`] compute encoders of one dispatch each a command buffer, apart
//!   (each writes words of its own, so the runtime records them in one `VkCommandBuffer`), or
//!   [`CHAINED`] in a chain (each reads the words the one before it wrote), as a guest's
//!   simulation step's kernels are, each then after a barrier of its own; and that chain with
//!   each encoder updating a fence, which ends its group (`encoder_fold.rs`), so that each
//!   encoder goes in a `VkCommandBuffer` of its own and its barrier in one between: the
//!   submission's growth per `VkCommandBuffer`.
//! - **Churn**: with the listed imports, [`CHURN_ENCODERS`] encoders, and before each commit a
//!   no-copy buffer of [`CHURN_BYTES`] made over pages of its own and bound as the last dispatch's
//!   output, the one made [`CHURN_KEPT`] commits before released: an import and a free each
//!   command buffer, as a guest's buffers and textures over its pages come and go; beside the same
//!   arm with no churn. Then the two halves apart, each buffer of [`SPLIT_BYTES`] over pages of
//!   its own: an import each command buffer, the buffers kept to the arm's end ("imports"), and a
//!   free each command buffer of buffers made and bound before the arm ("frees"); beside both at
//!   that size. The live dma-buf memories grow by [`COMMITS`] over the imports' arm and fall by
//!   as many over the frees'.
//!
//! Printed (`SUBMIT-TAX`), per arm: the submission calls' median and 90th percentile, the
//! `VkCommandBuffer`s a call, the kernel end to GPU start's median and 90th percentile, and the
//! device's live dma-buf memories as the arm began.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` where `/dev/udmabuf` opens,
//! `cargo test --release -p metal2vulkan-runtime --lib submit_tax_bench -- --ignored --nocapture`.

use super::blit_holds::{Scratch, WAIT};
use super::compute_parked::Kernel;
use super::host_residency::{udmabuf_openable, Ram};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, Fence, HostMemory,
    ResourceOptions, Size,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The no-copy buffers of the listed arms.
const LISTED: usize = 384;
/// Each one's bytes.
const LISTED_BYTES: usize = 64 << 10;
/// The encoders a command buffer of the plain arms.
const ENCODERS: [u64; 2] = [1, 48];
/// The encoders a command buffer of the chained arms.
const CHAINED: u64 = 24;
/// The encoders a command buffer of the churn arms.
const CHURN_ENCODERS: u64 = 4;
/// The bytes of each buffer the churn makes.
const CHURN_BYTES: usize = 4 << 20;
/// The churn's buffers live at once, each over pages of its own.
const CHURN_KEPT: usize = 3;
/// The bytes of each buffer of the split churn arms, each over pages of its own.
const SPLIT_BYTES: usize = 1 << 20;
/// The command buffers of an arm a round.
const COMMITS: usize = 150;
/// The rounds.
const ROUNDS: usize = 3;
/// How long the bench waits for released imports to be freed.
const FREED: Duration = Duration::from_secs(30);

/// What an arm measured: per command buffer, its calls' nanoseconds and `VkCommandBuffer`s per
/// call, and its kernel end to GPU start in seconds.
#[derive(Default)]
struct Arm {
    calls: Vec<f64>,
    carried: Vec<f64>,
    gaps: Vec<f64>,
    live: Vec<u64>,
}

/// The `at` quantile of `values`, sorted, in microseconds.
fn quantile(values: &[f64], at: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
        .get(((sorted.len() as f64 * at) as usize).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// The guest RAM of the bench: the listed buffers' pages, then the churn's.
struct Pages {
    ram: Ram,
    calls: Arc<AtomicUsize>,
}

impl Pages {
    fn new() -> Pages {
        let len = Self::split_at(COMMITS);
        Pages {
            ram: Ram::new(len, 0, len),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Where the split churn's `slot`th buffer begins.
    fn split_at(slot: usize) -> usize {
        LISTED * LISTED_BYTES + (CHURN_KEPT + 1) * CHURN_BYTES + slot * SPLIT_BYTES
    }

    /// A no-copy buffer of `length` bytes from `offset`, imported as a dma-buf of its pages.
    fn buffer(&self, device: &Device, offset: usize, length: usize) -> Buffer {
        // SAFETY: the view outlives every buffer (each arm drops its buffers before the pages),
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

/// The chained arms' kernel: the same `scale_and_offset` as [`Kernel`]'s, its scale of 1, and a
/// buffer of `CHAINED + 1` dispatches' words, each dispatch reading the words the one before it
/// wrote.
pub(super) struct Chain {
    state: ComputePipelineState,
    scale: Buffer,
    words: Buffer,
}

impl Chain {
    pub(super) fn new(device: &Device) -> Chain {
        let bytes = vec![0; ((CHAINED + 1) * 256) as usize];
        Chain {
            state: super::dispatch_pending::state(device),
            scale: (device.new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared()))
                .unwrap(),
            words: (device.new_buffer_with_bytes(&bytes, ResourceOptions::shared())).unwrap(),
        }
    }

    /// `count` compute encoders of `command_buffer`, the `k`th reading the 64 words from byte
    /// `256 k` and writing those from `256 (k + 1)`, each updating `fence` where there is one.
    pub(super) fn encode(&self, command_buffer: &CommandBuffer, count: u64, fence: Option<&Fence>) {
        for k in 0..count {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&self.state).unwrap();
            encoder.set_buffer(Some(&self.words), k * 256, 0).unwrap();
            encoder
                .set_buffer(Some(&self.words), (k + 1) * 256, 1)
                .unwrap();
            encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
            encoder
                .dispatch_threads(Size::new(64, 1, 1), Size::new(64, 1, 1))
                .unwrap();
            if let Some(fence) = fence {
                encoder.update_fence(fence).unwrap();
            }
            encoder.end_encoding().unwrap();
        }
    }
}

/// How an arm's command buffers are encoded.
#[derive(Clone, Copy)]
enum Shape<'a> {
    /// Encoders apart, a dispatch each.
    Apart(u64),
    /// Encoders in a chain, each updating the fence where there is one.
    Chained(&'a Chain, u64, Option<&'a Fence>),
    /// Encoders apart, and the churn's buffer over `Pages` bound by one more.
    Churn(u64, &'a Pages),
    /// Encoders apart, and the split churn's half over `Pages`.
    Split(u64, &'a Pages, Half),
}

/// A half of the churn.
#[derive(Clone, Copy, PartialEq)]
enum Half {
    /// An import each command buffer, bound by one more encoder and kept to the arm's end.
    Imports,
    /// A free each command buffer, of a buffer made and bound before the arm.
    Frees,
    /// Both: an import each, bound by one more encoder, the one made before it released.
    Both,
}

/// Wait until the device's live dma-buf memories fall to `count` or [`FREED`] passed.
fn freed_to(device: &Device, count: u64) {
    let start = Instant::now();
    while device.work_counts().dma_buf_memories_live > count && start.elapsed() < FREED {
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run [`COMMITS`] command buffers of `shape` on `kernel`, into `arm`.
fn run_arm(kernel: &Kernel, shape: Shape<'_>, arm: &mut Arm) {
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let apart = match shape {
        Shape::Apart(encoders) | Shape::Churn(encoders, _) | Shape::Split(encoders, ..) => encoders,
        Shape::Chained(..) => 0,
    };
    let output = kernel.output(apart.max(1));
    let mut kept: Vec<Buffer> = Vec::new();
    if let Shape::Split(_, pages, Half::Frees) = shape {
        kept = (0..COMMITS)
            .map(|slot| pages.buffer(device, Pages::split_at(slot), SPLIT_BYTES))
            .collect();
        bind_all(kernel, &kept);
    }
    arm.live.push(device.work_counts().dma_buf_memories_live);
    for commit in 0..COMMITS {
        let command_buffer = queue.command_buffer().unwrap();
        for at in 0..apart {
            kernel.dispatch(&command_buffer, (&output, at * 256));
        }
        if let Shape::Chained(chain, count, fence) = shape {
            chain.encode(&command_buffer, count, fence);
        }
        if let Shape::Churn(_, pages) = shape {
            if kept.len() == CHURN_KEPT {
                drop(kept.remove(0));
            }
            let slot = commit % (CHURN_KEPT + 1);
            let made = pages.buffer(
                device,
                LISTED * LISTED_BYTES + slot * CHURN_BYTES,
                CHURN_BYTES,
            );
            kernel.dispatch(&command_buffer, (&made, 0));
            kept.push(made);
        }
        if let Shape::Split(_, pages, half) = shape {
            if half == Half::Frees {
                drop(kept.pop());
            } else {
                if half == Half::Both {
                    drop(kept.pop());
                }
                let made = pages.buffer(device, Pages::split_at(commit), SPLIT_BYTES);
                kernel.dispatch(&command_buffer, (&made, 0));
                kept.push(made);
            }
        }
        let before = device.work_counts();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?}",
            command_buffer.error()
        );
        let after = device.work_counts();
        let calls = (after.queue_submissions - before.queue_submissions).max(1) as f64;
        arm.calls
            .push((after.batch_submit_nanos - before.batch_submit_nanos) as f64 / calls);
        arm.carried.push(
            (after.command_buffers_submitted - before.command_buffers_submitted) as f64 / calls,
        );
        if let Ok(start) = command_buffer.gpu_start_time() {
            arm.gaps.push(start - command_buffer.kernel_end_time());
        }
    }
}

/// Each of `buffers` bound by a dispatch of a command buffer completed, 48 a command buffer: kept
/// imported for good (`host_residency.rs`).
fn bind_all(kernel: &Kernel, buffers: &[Buffer]) {
    let queue = kernel.device.new_command_queue().unwrap();
    for group in buffers.chunks(48) {
        let command_buffer = queue.command_buffer().unwrap();
        for buffer in group {
            kernel.dispatch(&command_buffer, (buffer, 0));
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
}

/// The place of the arm named `name` in `arms`, added where it is not there yet.
fn arm(arms: &mut Vec<(String, Arm)>, name: String) -> usize {
    match arms.iter().position(|(named, _)| *named == name) {
        Some(at) => at,
        None => {
            arms.push((name, Arm::default()));
            arms.len() - 1
        }
    }
}

fn print(name: &str, arm: &Arm) {
    println!(
        "SUBMIT-TAX {name}: {} command buffers; call median {:.1} us, p90 {:.1} us; {:.1} \
         VkCommandBuffers a call; kernel end -> GPU start median {:.1} us, p90 {:.1} us; live \
         dma-buf memories at the arm's start {:?}",
        arm.calls.len(),
        quantile(&arm.calls, 0.5) / 1e3,
        quantile(&arm.calls, 0.9) / 1e3,
        quantile(&arm.carried, 0.5),
        quantile(&arm.gaps, 0.5) * 1e6,
        quantile(&arm.gaps, 0.9) * 1e6,
        arm.live,
    );
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn submit_tax_bench() {
    if !udmabuf_openable("submit_tax_bench") {
        return;
    }
    let scratch = Scratch::new("submit-tax");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let device = &kernel.device;
    println!("SUBMIT-TAX on {}", device.name());
    let pages = Pages::new();
    let chain = Chain::new(device);
    let fence = device.new_fence();
    let mut arms: Vec<(String, Arm)> = Vec::new();
    let base = device.work_counts().dma_buf_memories_live;
    for _ in 0..ROUNDS {
        for encoders in ENCODERS {
            let at = arm(&mut arms, format!("listed 0, encoders {encoders}"));
            run_arm(&kernel, Shape::Apart(encoders), &mut arms[at].1);
        }
        let at = arm(&mut arms, format!("listed 0, encoders {CHAINED} chained"));
        run_arm(
            &kernel,
            Shape::Chained(&chain, CHAINED, None),
            &mut arms[at].1,
        );
        let at = arm(
            &mut arms,
            format!("listed 0, encoders {CHAINED} chained, fenced"),
        );
        run_arm(
            &kernel,
            Shape::Chained(&chain, CHAINED, Some(&fence)),
            &mut arms[at].1,
        );
        let listed: Vec<Buffer> = (0..LISTED)
            .map(|i| pages.buffer(device, i * LISTED_BYTES, LISTED_BYTES))
            .collect();
        bind_all(&kernel, &listed);
        for encoders in ENCODERS {
            let at = arm(&mut arms, format!("listed {LISTED}, encoders {encoders}"));
            run_arm(&kernel, Shape::Apart(encoders), &mut arms[at].1);
        }
        let at = arm(
            &mut arms,
            format!("listed {LISTED}, encoders {CHAINED} chained"),
        );
        run_arm(
            &kernel,
            Shape::Chained(&chain, CHAINED, None),
            &mut arms[at].1,
        );
        let at = arm(
            &mut arms,
            format!("listed {LISTED}, encoders {CHAINED} chained, fenced"),
        );
        run_arm(
            &kernel,
            Shape::Chained(&chain, CHAINED, Some(&fence)),
            &mut arms[at].1,
        );
        let plain = format!("listed {LISTED}, encoders {CHURN_ENCODERS}");
        let at = arm(&mut arms, plain.clone());
        run_arm(&kernel, Shape::Apart(CHURN_ENCODERS), &mut arms[at].1);
        let at = arm(&mut arms, format!("{plain}, churn"));
        run_arm(
            &kernel,
            Shape::Churn(CHURN_ENCODERS, &pages),
            &mut arms[at].1,
        );
        for (half, named) in [
            (Half::Imports, "imports"),
            (Half::Frees, "frees"),
            (Half::Both, "both"),
        ] {
            let at = arm(
                &mut arms,
                format!("{plain}, {} MiB {named}", SPLIT_BYTES >> 20),
            );
            run_arm(
                &kernel,
                Shape::Split(CHURN_ENCODERS, &pages, half),
                &mut arms[at].1,
            );
            freed_to(device, LISTED as u64 + base);
        }
        drop(listed);
        freed_to(device, base);
    }
    for (name, arm) in &arms {
        print(name, arm);
    }
    let counts = device.work_counts();
    println!(
        "SUBMIT-TAX counts: dma_buf_imports {}, host_imports_held_for_good {}, \
         host_imports_released {}",
        counts.dma_buf_imports, counts.host_imports_held_for_good, counts.host_imports_released
    );
}
